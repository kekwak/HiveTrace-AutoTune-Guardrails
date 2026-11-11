"""Offline static-batch reports. Timing samples are batches, not requests."""
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

from .core import digest, lines, read, sha, write
from .contracts import BUCKETS
from .precision import PrecisionTracker, physical_precision
from .quality import agreement
from .statistics import assess, baseline_diagnostics, paired, quantile


class InvalidArtifact(ValueError):
    """A fixed diagnostic code, never an arbitrary data-bearing exception."""


def require(condition, code):
    if not condition:
        raise InvalidArtifact(code)


def safe_path(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "unsafe_artifact_path")
    return path


def finite(value, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def trace_has_kernels(path):
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    require(any(e.get("cat") == "kernel" for e in events), "profile_without_kernels")


def response_rows(records, plans, timed=False, timestamps=False):
    from .contracts import validate_response
    require(isinstance(records, list) and len(records) == len(plans), "incomplete_batch_rows")
    labels = defaultdict(list)
    previous = None
    for record, plan in zip(records, plans, strict=True):
        require(record["batch"] == plan, "batch_plan_mismatch")
        scores = record["scores"]
        require(isinstance(scores, list) and len(scores) == len(plan["ids"]), "response_count_mismatch")
        if timed:
            require(finite(record["latency_ms"], True), "invalid_latency")
        if timestamps:
            start, end = record["started_unix"], record["finished_unix"]
            require(finite(start) and finite(end) and end >= start
                    and (previous is None or start >= previous), "invalid_timestamps")
            previous = end
        for key, score in zip(plan["ids"], scores, strict=True):
            labels[key].append(validate_response(score))
    return labels


class Evidence:
    def __init__(self, root, cfg):
        self.root, self.cfg = root, cfg
        self.inputs_hash = sha(safe_path(root, "inputs.json"))
        self.sources_hash = sha(safe_path(root, "source_files.json"))
        self.precision = PrecisionTracker(cfg["comparison"])
        self.identities, self.fixtures, self.contracts = set(), set(), []
        self.policies = defaultdict(set)
        self.padding_semantics = set()
        self.integrity_failed = False

    def checked(self, path, variant, phase, measurement=None):
        from .report_v2 import identity, validate_worker
        path = safe_path(self.root, path)
        for filename in ("result.json", "runtime.json", "telemetry_before.json", "packages.json"):
            safe_path(self.root, path / filename)
        saved = read(path / "result.json")
        for field, expected in (("config_sha256", digest(self.cfg)), ("inputs_sha256", self.inputs_hash),
                                ("variant", variant), ("phase", phase)):
            if saved.get(field) != expected:
                self.integrity_failed = True
                raise InvalidArtifact("worker_identity_mismatch")
        value = validate_worker(path, self.cfg, variant, self.inputs_hash)
        require(value["phase"] == phase, "worker_phase_mismatch")
        if value.get("adapter_spec") != self.cfg[variant]:
            self.integrity_failed = True
            raise InvalidArtifact("adapter_spec_mismatch")
        self.contracts.append(value["contract"])
        self.identities.add(identity(path))
        try:
            self.precision.add(variant, value)
        except ValueError:
            self.integrity_failed = True
            raise
        if self.cfg["model"] == "laya":
            require(value.get("precision_policy_version") == 1, "missing_precision_passport")
        if value.get("metadata", {}).get("test_fixture"):
            self.fixtures.add(variant)
        if measurement is not None:
            if value.get("batch_protocol_version") != 1 or value.get("sources_sha256") != self.sources_hash:
                self.integrity_failed = True
                raise InvalidArtifact("worker_source_mismatch")
            require(value["job"] == measurement and value["bucket"] == measurement["bucket"]
                    and value["batch_size"] == measurement["batch_size"], "worker_measurement_mismatch")
            shapes = sorted({(measurement["batch_size"], p["padded_length"])
                             for p in measurement["batches"]})
            require(value["prepared_shapes"] == [list(s) for s in shapes], "prepared_shapes_mismatch")
            count = value["prepared_compile_count"]
            require(type(count) is int and count >= 0, "invalid_compile_counter")
            require(type(value.get("final_compile_count")) is int and value["final_compile_count"] == count,
                    "compilation_during_measurement")
            for field in ("final_compile_count", "finished_compile_count"):
                if field in value:
                    require(value[field] == count, "compilation_during_measurement")
            if phase != "batch_preflight" and "prepared_compile_count" in value["metadata"]:
                require(value["metadata"]["prepared_compile_count"] == count, "compile_counter_drift")
            require(isinstance(value["metadata"].get("batch_policy"), dict), "missing_batch_policy")
            policy = value["metadata"]["batch_policy"]
            fields = ("padding", "padding_side", "truncation", "sort_by_length")
            require(all(k in policy for k in fields), "incomplete_batch_policy")
            self.padding_semantics.add(digest({**{k: policy[k] for k in fields},
                                              "pad_token_id": policy.get("pad_token_id")}))
            self.policies[variant].add(digest(policy))
            for field in ("load_seconds", "backend_preparation_seconds", "setup_seconds"):
                require(finite(value[field]), "invalid_preparation_duration")
            if phase != "batch_preflight":
                require(finite(value["warmup_seconds"]), "missing_warmup")
        return value

    def validate(self):
        require(not self.integrity_failed, "worker_integrity_failure")
        require(len(self.identities) == 1, "runtime_identity_mismatch")
        require(bool(self.contracts) and all(c == self.contracts[0] for c in self.contracts),
                "inference_contract_mismatch")
        from .contracts import model_contract
        require(self.contracts[0] == model_contract(self.cfg["model"], self.cfg["schema_version"]),
                "unexpected_inference_contract")
        require(all(len(p) == 1 for p in self.policies.values()), "batch_policy_drift")
        require(len(self.padding_semantics) == 1, "paired_padding_policy_mismatch")


def preflight(evidence, base, variant, measurement):
    path = base / "preflight" / variant
    value = evidence.checked(path, variant, "batch_preflight", measurement)
    require(value.get("aba_passed") is True and value["boundary"]["over_limit_rejected"] is True,
            "preflight_boundary_or_aba_failed")
    require(value["boundary"].get("tokens") == BUCKETS[measurement["bucket"]]["max"]
            and value["boundary"].get("model_limit") == value["contract"]["limit"],
            "wrong_batch_boundary")
    response_rows(value["verified_batches"], measurement["batches"])
    response_rows(value["timings"], measurement["batches"], timed=True)
    require(isinstance(value.get("memory"), dict), "missing_smoke_memory")
    if not value["metadata"].get("test_fixture"):
        proof = value["metadata"].get("batch_backend_proof", {})
        require(bool(proof.get("kernels")), "missing_batch_backend_proof")
    return value


def measured_cell(cfg, jobs, outcomes):
    groups = {"aa": defaultdict(dict), "ab": defaultdict(dict), "baseline": defaultdict(dict)}
    labels = {variant: defaultdict(list) for variant in ("baseline", "candidate")}
    rates = defaultdict(lambda: {"batches": 0, "texts": 0, "latency_ms": 0,
                                 "useful_tokens": 0, "padded_tokens": 0, "padding_tokens": 0})
    for job, out in zip(jobs, outcomes, strict=True):
        rows = out["rows"]
        phase, block, side = job["phase"], job["block"], job["side"]
        require(side not in groups[phase][block], "duplicate_pair_side")
        groups[phase][block][side] = rows
        values = response_rows(rows, job["batches"], timed=True, timestamps=True)
        for key, scores in values.items():
            labels[job["variant"]][key].extend(scores)
        if phase != "aa":
            rate = rates[job["variant"]]
            for row in rows:
                rate["batches"] += 1
                rate["texts"] += len(row["batch"]["ids"])
                rate["latency_ms"] += row["latency_ms"]
                for key in ("useful_tokens", "padded_tokens", "padding_tokens"):
                    rate[key] += row["batch"][key]
    series = {}
    for phase in ("aa", "ab"):
        series[phase] = []
        for block in sorted(groups[phase]):
            sides = groups[phase][block]
            require(set(sides) == {"a", "b"}, "missing_pair_side")
            require([r["batch"] for r in sides["a"]] == [r["batch"] for r in sides["b"]],
                    "paired_input_mismatch")
            series[phase].append(tuple([r["latency_ms"] for r in sides[s]] for s in ("a", "b")))
    if cfg.get("candidate"):
        if cfg["mode"] == "screening":
            item = assess(series["ab"], series["aa"], .10 if cfg["comparison"] == "parameter" else .20)
            item["faster"] = not item["reasons"] and item["p95_reduction"] > 0
        else:
            item = {"status": "not_evaluated_pilot", "faster": False,
                    "p50": paired(series["ab"], .5), "p95": paired(series["ab"])}
            for name in ("p50", "p95"):
                item[name]["ratio_interval_95"] = None
        verdicts = agreement(labels["baseline"], labels["candidate"])
    else:
        blocks = [[r["latency_ms"] for r in sides["a"]]
                  for _, sides in sorted(groups["baseline"].items())]
        item = {"status": "measured_baseline", "faster": False}
        if cfg["mode"] == "screening":
            item.update(baseline_diagnostics(blocks, series["aa"]))
        else:
            values = [v for block in blocks for v in block]
            item.update({name: {"baseline_ms": quantile(values, q)} for name, q in (("p50", .5), ("p95", .95))})
        verdicts = agreement(labels["baseline"], labels["baseline"])
    for rate in rates.values():
        rate["requests_per_second"] = rate["texts"] * 1000 / rate["latency_ms"]
        rate["padding_fraction"] = rate["padding_tokens"] / rate["padded_tokens"]
    item.update(verdicts=verdicts, throughput=dict(rates), timing_unit="batch",
                speed_status=item["status"], verdict_status="unstable" if verdicts["unstable_ids"] else
                "changed" if verdicts["changed_count"] else "unchanged")
    if cfg["mode"] == "pilot":
        item.update(status="pilot_complete", speed_status="not_evaluated_pilot")
    elif not cfg.get("candidate"):
        item["status"] = "baseline_unstable" if verdicts["unstable_ids"] else "baseline_complete"
    elif cfg["comparison"] == "quantization":
        item.update(status="quantization_measured", acceptance="not_assessed_without_quality_rule",
                    speed_status="inconclusive" if item["speed_status"] == "inconclusive" else "measured_no_acceptance_threshold")
    else:
        item["status"] = ("rejected" if item["verdict_status"] != "unchanged" or item["speed_status"] == "below_target"
                          else "screening_pass" if item["speed_status"] == "pass" else "inconclusive")
    return item, labels


def memory_profiles(evidence, base, variants, measurement, profiles, item):
    records = []
    item["memory"] = records
    for variant in variants:
        for repeat in range(3):
            value = evidence.checked(base / "memory" / f"{variant}_{repeat}", variant, "batch_memory", measurement)
            require(read(safe_path(evidence.root, base / "memory" / f"{variant}_{repeat}" / "memory.json")) == value["memory"],
                    "memory_summary_mismatch")
            records.append({k: value[k] for k in ("variant", "pid", "memory", "torch_setup", "torch_inference", "size", "setup_seconds")})
            item["memory_status"] = "incomplete"
            for window in ("setup", "inference"):
                require(value["memory"]["windows"][window]["status"] in ("measured", "unavailable"), "invalid_memory_window")
            for window in ("torch_setup", "torch_inference"):
                require(all(finite(value[window][k]) for k in ("allocated", "reserved")), "invalid_memory_counters")
    item["memory_status"] = ("measured" if all(m["memory"]["windows"][w]["status"] == "measured"
                                              for m in records for w in ("setup", "inference")) else "partially_unavailable")
    for variant in variants:
        path = base / "profiles" / variant
        evidence.checked(path, variant, "batch_profile", profiles)
        for label in ("short", "long"):
            trace_has_kernels(safe_path(evidence.root, path / (label + ".trace.json.gz")))
    return records


def build(directory):
    """Rebuild summary.json and REPORT.md using only authenticated saved artifacts."""
    from .batching import cells, memory_plans, schedule, smoke_plans
    root = Path(directory).resolve()
    cfg = read(safe_path(root, "config.json"))
    result = {"schema_version": cfg["schema_version"], "batch_protocol_version": 1,
              "model": cfg["model"], "mode": cfg["mode"], "comparison": cfg["comparison"],
              "status": "incomplete", "speed_status": "not_measured", "verdict_status": "not_measured",
              "memory_status": "not_measured", "quality_evaluated": False, "quality_preserved": None,
              "quality_status": "not_evaluated",
              "independently_confirmed": False, "all_cells_faster": False,
              "all_cells_meet_threshold": False, "cells": {}, "matrix": {}, "errors": []}
    evidence = None
    state = {}
    measured = {}
    try:
        state = read(safe_path(root, "status.json"))
        if state.get("quality_protocol_version") != 1:
            result["quality_batch_size"] = 1
        grid = cells(cfg)
        sizes = sorted({s for _, s in grid})
        result["matrix"] = {b: {str(s): "not_scheduled" for s in sizes} for b, _ in grid}
        data = defaultdict(list)
        for row in read(safe_path(root, "inputs.json")):
            require(row["bucket"] in BUCKETS and BUCKETS[row["bucket"]]["min"] <= row["index"]["total_tokens"]
                    <= BUCKETS[row["bucket"]]["max"], "input_outside_bucket")
            data[row["bucket"]].append(row)
        sources = read(safe_path(root, "source_files.json"))
        require(bool(sources), "missing_sources")
        for name, checksum in sources.items():
            require(sha(safe_path(root, name)) == checksum, "archived_source_hash_mismatch")
        evidence = Evidence(root, cfg)
        jobs = schedule(cfg, data)
        require(read(safe_path(root, "schedule.json")) == jobs, "schedule_mismatch")
        matrix_state = read(safe_path(root, "matrix_status.json"))
        require(set(matrix_state) <= {f"{b}/{s}" for b, s in grid}, "unexpected_matrix_cell")
        if isinstance(cfg.get("data"), dict) and cfg["data"].get("manifest_sha256"):
            require(read(safe_path(root, "dataset_identity.json"))["manifest_sha256"] == cfg["data"]["manifest_sha256"],
                    "dataset_identity_mismatch")
        variants = ["baseline"] + (["candidate"] if cfg.get("candidate") else [])
        for bucket, size in grid:
            key = f"{bucket}/{size}"
            base = root / "batches" / bucket / str(size)
            item = {"bucket": bucket, "batch_size": size, "status": "incomplete", "faster": False,
                    "completeness": "incomplete", "memory_status": "not_measured",
                    "speed_status": "not_measured", "verdict_status": "not_measured", "errors": []}
            result["cells"][key] = item
            try:
                plans = smoke_plans(data[bucket], size, cfg["seed"])
                measurement = {"bucket": bucket, "batch_size": size, "batches": plans}
                for variant in variants:
                    preflight(evidence, base, variant, measurement)
                status = matrix_state.get(key, {}).get("status")
                if cfg["mode"] == "smoke" or state.get("preflight_only"):
                    require(status == "preflight_passed", "cell_execution_incomplete")
                    item.update(status="smoke_passed" if cfg["mode"] == "smoke" else "preflight_passed", completeness="complete")
                else:
                    selected = [(i, job) for i, job in enumerate(jobs) if (job["bucket"], job["batch_size"]) == (bucket, size)]
                    outcomes = []
                    for i, job in selected:
                        path = root / "latency" / f"{i:03d}"
                        value = evidence.checked(path, job["variant"], "batch_latency", job)
                        require(value["rows"] == lines(safe_path(root, path / "latencies.jsonl")), "raw_rows_mismatch")
                        outcomes.append(value)
                    stats, labels = measured_cell(cfg, [j for _, j in selected], outcomes)
                    if cfg["mode"] == "screening":
                        expected_ids = {r["id"] for r in data[bucket]}
                        require(len(expected_ids) == 400 and all(set(labels[v]) == expected_ids for v in variants), "incomplete_unique_responses")
                    item.update(stats)
                    measured[key] = labels
                    item["preparation"] = [{"variant": o["variant"], "phase": o["job"]["phase"], "block": o["job"]["block"],
                                             **{k: o[k] for k in ("load_seconds", "backend_preparation_seconds", "setup_seconds", "warmup_seconds")}}
                                            for o in outcomes]
                    if cfg["mode"] == "screening":
                        memory = {"bucket": bucket, "batch_size": size, "batches": memory_plans(data[bucket], size, cfg["seed"])}
                        profiles = {**measurement, "batches": plans[:2]}
                        memory_profiles(evidence, base, variants, memory, profiles, item)
                    require(status == "completed", "cell_execution_incomplete")
                    item["completeness"] = "complete"
            except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, EOFError) as error:
                saved = matrix_state.get(key, {}).get("status")
                item.update(status=saved if saved in ("oom", "unsupported", "timeout", "failed", "not_ready") else "incomplete",
                            completeness="incomplete", faster=False)
                item["errors"].append(str(error) if isinstance(error, InvalidArtifact) else "invalid_or_missing_cell_artifact")
            result["matrix"][bucket][str(size)] = item["status"]
        if cfg.get("quality_data") and not state.get("preflight_only") and state.get("quality_protocol_version") != 1:
            from .quality import metrics
            result["quality"] = {}
            for variant in variants:
                path = root / "quality" / variant
                value = evidence.checked(path, variant, "quality")
                require(value.get("quality_batch_size") == 1, "invalid_quality_batch_size")
                calculated = metrics(read(safe_path(root, "quality_inputs.json")),
                                     lines(safe_path(root, path / "predictions.jsonl")), cfg["label_mapping"])
                require(calculated == value["quality"], "quality_summary_mismatch")
                result["quality"][variant] = calculated
            result.update(quality_evaluated=True, quality_status="evaluated_batch_1_only")
        if state.get("quality_protocol_version") == 1:
            from .quality_report import collect
            result.update(collect(root, cfg, lambda path, variant: evidence.checked(path, variant, "quality")))
        evidence.validate()
        result.update(runtime_identity=next(iter(evidence.identities)),
                      dataset_identity=read(safe_path(root, "dataset_identity.json")), **evidence.precision.summary())
        items = list(result["cells"].values())
        complete = bool(items) and all(i["completeness"] == "complete" for i in items)
        complete = complete and state.get("status") == "completed"
        result["completeness"] = "complete" if complete else "incomplete"
        verdict_statuses = {i["verdict_status"] for i in items}
        result["verdict_status"] = ("unstable" if "unstable" in verdict_statuses else "changed" if "changed" in verdict_statuses
                                    else "unchanged" if verdict_statuses == {"unchanged"} else "not_measured")
        result["memory_status"] = ("measured" if all(i["memory_status"] == "measured" for i in items)
                                   else "partially_unavailable" if any(i["memory_status"] != "not_measured" for i in items) else "not_measured")
        result["all_cells_faster"] = complete and cfg["mode"] == "screening" and all(i["faster"] for i in items)
        result["all_cells_meet_threshold"] = complete and cfg["mode"] == "screening" and cfg["comparison"] != "quantization" and bool(cfg.get("candidate")) and all(i["speed_status"] == "pass" for i in items)
        if complete:
            if cfg["mode"] == "smoke" or state.get("preflight_only"):
                result["status"] = "smoke_passed" if cfg["mode"] == "smoke" else "preflight_passed"
            elif cfg["mode"] == "pilot":
                result["status"] = "pilot_complete"
            elif not cfg.get("candidate"):
                result["status"] = "baseline_unstable" if any(i["verdict_status"] == "unstable" for i in items) else "baseline_complete"
            elif cfg["comparison"] == "quantization":
                result.update(status="quantization_measured", acceptance="not_assessed_without_quality_rule")
            else:
                statuses = {i["status"] for i in items}
                result["status"] = "rejected" if "rejected" in statuses else "screening_pass" if statuses == {"screening_pass"} else "inconclusive"
        result["speed_status"] = ("incomplete" if not complete else "not_evaluated_pilot" if cfg["mode"] == "pilot"
                                  else "not_measured" if cfg["mode"] == "smoke" or state.get("preflight_only")
                                  else "measured_baseline" if not cfg.get("candidate") else
                                  "measured_no_acceptance_threshold" if cfg["comparison"] == "quantization" else
                                  "screening_pass" if result["all_cells_meet_threshold"] else
                                  "inconclusive" if any(i["speed_status"] == "inconclusive" for i in items) else "below_target")
        result["macro_verdict_change_fraction_by_batch_size"] = {
            str(size): sum(result["cells"][f"{b}/{size}"]["verdicts"]["changed_fraction"] for b, s in grid if s == size) / sum(s == size for _, s in grid)
            for size in sizes if all("verdicts" in result["cells"][f"{b}/{size}"] for b, s in grid if s == size)}
        result["single_to_batch"] = {}
        for bucket, size in grid:
            if size == 1 or f"{bucket}/1" not in measured or f"{bucket}/{size}" not in measured:
                continue
            comparisons = {}
            for variant in variants:
                single, batched = measured[f"{bucket}/1"][variant], measured[f"{bucket}/{size}"][variant]
                common = single.keys() & batched.keys()
                if common:
                    comparisons[variant] = agreement({k: single[k] for k in sorted(common)}, {k: batched[k] for k in sorted(common)})
            result["single_to_batch"][f"{bucket}/{size}"] = comparisons
        if evidence.fixtures:
            result.update(status="fixture_complete" if complete else "incomplete", gpu_acceptance=False,
                          acceptance="interface_test_only", all_cells_faster=False, all_cells_meet_threshold=False)
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError, EOFError) as error:
        result.update(status="incomplete", completeness="incomplete", speed_status="incomplete",
                      all_cells_faster=False, all_cells_meet_threshold=False)
        if result["quality_evaluated"]:
            result.update(quality_evaluated=False, quality_status="invalid_evidence")
        result["errors"].append(str(error) if isinstance(error, InvalidArtifact) else "invalid_global_evidence")
    if state.get("quality_protocol_version") == 1 and "quality_by_batch_size" not in result:
        from .quality_report import collect
        try:
            result.update(collect(root, cfg))
        except (OSError, ValueError, KeyError, TypeError):
            result.update(quality_status="incomplete", quality_complete=False)
    if result.get("quality_fixture"):
        result.update(gpu_acceptance=False, acceptance="interface_test_only")
    write(root / "summary.json", result)
    report = ["# Батчевые измерения У1", "", f"Статус: {result['status']}. Время относится к целому батчу.",
              "Качество оценивается отдельно для запрошенных размеров; согласие вердиктов не доказывает safety.", ""]
    columns = sorted({size for row in result["matrix"].values() for size in row}, key=int)
    report += ["Бакет | " + " | ".join("Batch " + size for size in columns),
               "--- | " + " | ".join("---" for _ in columns)]
    for bucket, statuses in result["matrix"].items():
        report.append(bucket + " | " + " | ".join(statuses[s] for s in columns))
    report += ["", "Бакет | Batch | p50 baseline / candidate, мс | p95 baseline / candidate, мс | Изменённые ID | Память",
               "--- | --- | --- | --- | --- | ---"]
    for bucket, columns in result["matrix"].items():
        for size, status in columns.items():
            item = result["cells"].get(f"{bucket}/{size}", {})
            if not item:
                continue
            p, p50, v = item.get("p95", {}), item.get("p50", {}), item.get("verdicts", {})
            report.append(f"{bucket} | {size} | {p50.get('baseline_ms', '-')} / {p50.get('candidate_ms', '-')} | "
                          f"{p.get('baseline_ms', '-')} / {p.get('candidate_ms', '-')} | "
                          f"{v.get('changed_count', '-')} / {v.get('n', '-')} | {item['memory_status']}")
    report += ["", f"Ускорение во всей сетке: {result['all_cells_faster']}.",
               f"Порог достигнут во всей сетке: {result['all_cells_meet_threshold']}.", *result["errors"]]
    from .quality_report import markdown as quality_markdown
    report += quality_markdown(result)
    (root / "REPORT.md").write_text("\n".join(report) + "\n")
    return result


def table_rows(root, summary, source):
    """Rows for the parent's archive table; never substitute smoke for baseline."""
    root = Path(root).resolve()
    if summary["mode"] != "screening" or summary.get("acceptance") == "interface_test_only" or summary.get("errors"):
        return []
    cfg = read(safe_path(root, "config.json"))
    output = []
    for key, item in summary["cells"].items():
        if item["completeness"] != "complete" or "p95" not in item:
            continue
        path = root / "batches" / item["bucket"] / str(item["batch_size"]) / "preflight" / "baseline"
        pf = read(safe_path(root, path / "result.json"))
        runtime = read(safe_path(root, path / "runtime.json"))
        gpu = read(safe_path(root, path / "telemetry_before.json"))
        environment = {k: runtime.get(k) for k in ("python", "torch", "cuda", "cudnn", "gpu", "cpu_threads", "tf32", "hub_kernels")}
        environment.update(driver=gpu["driver"], packages=read(safe_path(root, path / "packages.json")))
        grouping = {"lock": read(safe_path(root, "lock.json")), "data": summary["dataset_identity"],
                    "environment": environment, "batch_protocol_version": 1,
                    "bucket": item["bucket"], "batch_size": item["batch_size"],
                    "batch_policy": pf["metadata"]["batch_policy"], "contract": pf["contract"]}
        precision = pf["metadata"].get("precision")
        if precision is not None:
            grouping["precision"] = physical_precision(precision)
        output.append({"source": str(source), "comparison_group": digest(grouping), "model": cfg["model"],
                       "bucket": item["bucket"], "batch_size": item["batch_size"], "timing_unit": "batch",
                       "p95_ms": item["p95"]["baseline_ms"], "p50_ms": item["p50"]["baseline_ms"],
                       "status": item["status"], "environment": environment,
                       "memory": [m for m in item.get("memory", []) if m["variant"] == "baseline"],
                       "size": pf["size"], "contract": pf["contract"], "precision": precision,
                       "throughput": item.get("throughput", {}).get("baseline"),
                       "baseline_drift_p95": item.get("baseline_drift_p95"),
                       "aa_envelope_p95": item.get("aa_envelope_p95")})
        if summary.get("quality_protocol_version") == 1:
            output[-1].update(quality_batch_sizes=summary["quality_batch_sizes"], quality_status=summary["quality_status"])
        else:
            output[-1]["quality_batch_size"] = 1
    return output
