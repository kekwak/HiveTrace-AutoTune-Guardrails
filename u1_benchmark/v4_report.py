"""Offline v4 evidence reports. No live sources, datasets, models or catalogs.

The saved schedule contains latency plans; numbered worker directories carry
the full jobs and their provenance fields. Quality labels live in
resolved ``quality_rows``: public id/target/slices/hash metadata only. Reference
files are authenticated by ``reference.json.files`` before their own frozen
snapshot and workers are checked. ``build(..., save=False)`` never writes.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import re
from zipfile import BadZipFile

from .core import digest, sha, write
from .precision import PrecisionTracker, validate_precision
from .statistics import assess, baseline_diagnostics, quantile
from .v4_config import PROTOCOL_VERSION
from .v4_evidence import assert_environment, environment_identity, reference_folder
from .v4_planning import cells, schedule, smoke_plans


class InvalidArtifact(ValueError):
    """Fixed, public diagnostic codes; never include private exception text."""


def require(condition, code):
    if not condition:
        raise InvalidArtifact(code)


def safe(root, name):
    path = root / name
    require(path.resolve().is_relative_to(root.resolve()), "unsafe_artifact_path")
    return path


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate_json_key")
        value[key] = item
    return value


def _constant(_):
    raise InvalidArtifact("nonfinite_json")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs,
                      parse_constant=_constant)


def finite(value, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def checksum(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def frozen_config(cfg, variant_identities=None):
    """Validate saved shape only. Never import native settings or fill defaults."""
    required = {"schema_version", "baseline", "mode", "pilot", "profile_trace", "comparison", "seed",
                "timeout_seconds", "quality_batch_sizes", "batch_sizes_by_bucket"}
    optional = {"candidate", "buckets", "environment", "data", "quality_data", "label_mapping",
                "baseline_ref", "monitor_validation"}
    require(isinstance(cfg, dict) and required <= cfg.keys() and not set(cfg) - required - optional,
            "incomplete_frozen_config")
    if variant_identities is not None:
        require(isinstance(variant_identities, dict)
                and set(variant_identities) == {k for k in ("baseline", "candidate") if k in cfg}
                and all(checksum(v) for v in variant_identities.values()), "invalid_variant_identities")
    require(cfg["schema_version"] == 4 and cfg["mode"] in ("baseline", "search", "confirm", "smoke", "quality")
            and cfg["comparison"] in ("engineering", "parameter", "quantization"), "invalid_frozen_config")
    require(type(cfg["pilot"]) is bool and type(cfg["profile_trace"]) is bool and type(cfg["seed"]) is int
            and type(cfg["timeout_seconds"]) is int and 0 < cfg["timeout_seconds"] <= 36000, "invalid_frozen_config")

    def sizes(values):
        return (isinstance(values, list) and bool(values) and all(type(v) is int and v > 0 for v in values)
                and len(set(values)) == len(values))

    def source(value, template=False):
        if template and value is None:
            return
        require(isinstance(value, dict), "invalid_frozen_source")
        if template and set(value) == {"text"}:
            require(isinstance(value["text"], str), "invalid_frozen_source")
        elif "repo" in value:
            require(isinstance(value["repo"], str) and bool(value["repo"]) and "path" not in value
                    and isinstance(value.get("revision"), str)
                    and re.fullmatch(r"[0-9a-f]{40}", value["revision"]) is not None, "invalid_frozen_source")
        else:
            require(isinstance(value.get("path"), str) and bool(value["path"])
                    and (checksum(value.get("sha256")) or (isinstance(value.get("inventory"), dict)
                        and bool(value["inventory"]) and all(checksum(v) for v in value["inventory"].values()))),
                    "invalid_frozen_source")

    require(sizes(cfg["quality_batch_sizes"]), "invalid_frozen_sizes")
    require(isinstance(cfg["batch_sizes_by_bucket"], dict), "invalid_frozen_grid")
    for name in ("baseline", "candidate"):
        if name not in cfg:
            continue
        spec = cfg[name]
        fields = {"adapter", "weights", "tokenizer", "template", "preparation", "scoring", "execution"}
        require(isinstance(spec, dict) and fields <= spec.keys(), "incomplete_frozen_variant")
        adapter = spec["adapter"]
        require((isinstance(adapter, str) and bool(adapter)) or (isinstance(adapter, dict)
                and isinstance(adapter.get("module"), str) and checksum(adapter.get("source_sha256"))
                and isinstance(adapter.get("dependencies"), dict)), "invalid_frozen_adapter")
        for key in ("weights", "tokenizer"):
            source(spec[key])
        source(spec["template"], template=True)
        require(isinstance(spec.get("sdk_assets", {}), dict), "invalid_frozen_assets")
        for asset in spec.get("sdk_assets", {}).values():
            source(asset)
        require(isinstance(spec["preparation"], dict) and type(spec["preparation"].get("limit")) is int
                and spec["preparation"]["limit"] > 0 and isinstance(spec["scoring"], dict), "invalid_frozen_preparation")
        execution = spec["execution"]
        require(isinstance(execution, dict) and execution.get("runtime") in ("eager", "compile", "ort")
                and execution.get("dtype") in ("float16", "float32", "bfloat16", "native_sdk"), "invalid_frozen_execution")
    mode = cfg["mode"]
    require(not (mode == "baseline" and "candidate" in cfg)
            and (mode not in ("search", "confirm") or "candidate" in cfg), "invalid_frozen_mode")
    require((mode != "search" or bool(cfg.get("baseline_ref")))
            and (not cfg.get("baseline_ref") or mode == "search"), "invalid_frozen_reference")
    require(bool(cfg.get("quality_data" if mode == "quality" else "data")), "missing_frozen_corpus")
    bounds = cfg.get("buckets", {})
    require(isinstance(bounds, dict) and set(cfg["batch_sizes_by_bucket"]) == set(bounds)
            and all(sizes(v) for v in cfg["batch_sizes_by_bucket"].values()), "invalid_frozen_grid")
    if mode != "quality":
        cells(cfg)
    for bound in bounds.values():
        require(isinstance(bound, dict) and set(bound) == {"min", "max"}
                and type(bound["min"]) is int and type(bound["max"]) is int
                and 0 < bound["min"] <= bound["max"] <= cfg["baseline"]["preparation"]["limit"],
                "invalid_frozen_bounds")
    if "candidate" in cfg:
        a, b = cfg["baseline"], cfg["candidate"]
        keys = ("tokenizer", "sdk_assets", "template", "preparation")
        require({k: a[k] for k in keys if k in a} == {k: b[k] for k in keys if k in b}
                and a["scoring"] == b["scoring"],
                "frozen_preparation_mismatch")
        if cfg["comparison"] == "engineering":
            require(a["weights"] == b["weights"] and a["execution"]["dtype"] == b["execution"]["dtype"],
                    "frozen_engineering_mismatch")
        if cfg["comparison"] == "quantization":
            baseline_identity = variant_identities["baseline"] if variant_identities is not None else digest(a)
            require(b.get("quantization", {}).get("baseline_identity") == baseline_identity, "frozen_quantization_mismatch")
    return cfg


def response(value):
    # Same native response contract as quality.py, without importing its catalog.
    require(isinstance(value, dict) and isinstance(value.get("label"), str)
            and bool(value["label"]), "invalid_native_verdict")
    require(not set(value) - {"label", "scores", "score_type", "category", "generated_tokens",
                             "p_safe", "p_unsafe", "logit_margin", "public_confidence"},
            "invalid_response_fields")
    for key in ("p_safe", "p_unsafe", "logit_margin", "public_confidence"):
        if key in value:
            require(type(value[key]) in (int, float) and math.isfinite(value[key]), "invalid_score")
    require(value.get("category") is None or isinstance(value["category"], str), "invalid_category")
    if "generated_tokens" in value:
        require(type(value["generated_tokens"]) is int and value["generated_tokens"] >= 0,
                "invalid_generated_tokens")
    scores = value.get("scores")
    if scores is not None:
        kind = value.get("score_type")
        require(isinstance(scores, dict) and bool(scores)
                and kind in ("probabilities", "logits", "independent_probabilities", "raw"),
                "invalid_named_scores")
        require(all(isinstance(k, str) and type(v) in (int, float) and math.isfinite(v)
                    for k, v in scores.items()), "nonfinite_score")
        if "probabilities" in kind:
            require(all(0 <= v <= 1 for v in scores.values()), "invalid_probability")
        if kind == "probabilities":
            require(abs(sum(scores.values()) - 1) <= 1e-5, "invalid_probability_sum")
    return value


def agreement(left, right):
    require(bool(left) and set(left) == set(right), "response_id_mismatch")
    changed, unstable, directions, deltas = [], [], Counter(), {}
    for key in sorted(left):
        a, b = left[key], right[key]
        require(bool(a) and bool(b), "missing_responses")
        for value in a + b:
            response(value)
        if len({s["label"] for s in a}) != 1 or len({s["label"] for s in b}) != 1:
            unstable.append(key)
        if a[0]["label"] != b[0]["label"]:
            changed.append(key)
            directions[a[0]["label"], b[0]["label"]] += 1
        if (a[0].get("scores") and all(s.get("score_type") == a[0].get("score_type")
                and set(s.get("scores") or {}) == set(a[0]["scores"]) for s in a + b)):
            for label in a[0]["scores"]:
                left_scores = [x["scores"][label] for x in a]
                right_scores = [x["scores"][label] for x in b]
                deltas[label] = max(deltas.get(label, 0),
                                    abs(min(left_scores) - max(right_scores)),
                                    abs(max(left_scores) - min(right_scores)))
    require(all(math.isfinite(v) for v in deltas.values()), "nonfinite_score_delta")
    return {"n": len(left), "changed_count": len(changed), "changed_fraction": len(changed) / len(left),
            "changed_ids": changed, "unstable_ids": unstable, "max_abs_score_delta": deltas or None,
            "directions": [{"baseline": a, "candidate": b, "count": n} for (a, b), n in sorted(directions.items())],
            "status": "not_ok" if changed or unstable else "ok"}


def _error(error):
    return str(error) if isinstance(error, InvalidArtifact) else "invalid_or_missing_artifact"


ARTIFACT_ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, OverflowError)


class Evidence:
    def __init__(self, root):
        self.root = root
        self.resolved = read(safe(root, "resolved_run.json"))
        self.state = read(safe(root, "status.json"))
        r = self.resolved
        require(r.get("schema_version") == r.get("result_version") == 4,
                "unsupported_old_format_explicit_migration_required")
        require(r.get("protocol_version") == PROTOCOL_VERSION, "unsupported_protocol")
        self.sha = sha(safe(root, "resolved_run.json"))
        require(self.state.get("resolved_sha256") == self.sha, "resolved_hash_mismatch")
        self.cfg = frozen_config(r["config"], r.get("variant_identities"))
        if "config_sha256" in r:
            require(checksum(r["config_sha256"]), "invalid_config_identity")
        inventory = r.get("source_inventory")
        require(isinstance(inventory, dict) and bool(inventory)
                and all(isinstance(k, str) and checksum(v) for k, v in inventory.items()),
                "invalid_source_inventory")
        require(isinstance(r.get("environment"), dict) and bool(r["environment"]), "missing_environment")
        require(checksum(r.get("data_manifest_sha256")), "missing_corpus_identity")
        self.precision = PrecisionTracker(self.cfg["comparison"])
        self.observed = {}
        self.measured = {}
        self.recorded_indices = set()
        self.data = {}
        self.records = []
        require(isinstance(r["schedule"], list), "missing_schedule")
        self.paths = sorted(safe(root, "jobs").glob("*/job.json"))
        directories = [p for p in safe(root, "jobs").iterdir() if p.is_dir()]
        require({p.parent for p in self.paths} == set(directories), "job_without_specification")
        self.jobs = [read(safe(root, path)) for path in self.paths]
        for path, job in zip(self.paths, self.jobs):
            require(re.fullmatch(r"[0-9]{5,}", path.parent.name) is not None, "invalid_job_index")
            require(isinstance(job, dict), "invalid_job")
            require(job.get("stage") in ("preflight", "latency", "quality", "profile", "memory"),
                    "unsupported_job_stage")
            require(job.get("variant") in self.variants, "invalid_job_variant")
        if self.cfg["mode"] != "quality":
            indices = r["input_indices"]
            require(set(indices) == set(self.cfg["buckets"]), "input_buckets_mismatch")
            for bucket, values in indices.items():
                self.data[bucket] = [{"id": row["id"], "index": row.get("index", row)} for row in values]
            rows = [row for values in self.data.values() for row in values]
            require(len({r["id"] for r in rows}) == len(rows)
                    and len({r["index"]["input_sha256"] for r in rows}) == len(rows), "duplicate_speed_inputs")
            for bucket, values in self.data.items():
                bounds = self.cfg["buckets"][bucket]
                require(all(bounds["min"] <= r["index"]["total_tokens"] <= bounds["max"] for r in values),
                        "input_outside_bucket")
            if not self.cfg["pilot"] and self.cfg["mode"] != "smoke":
                require(all(len(rows) == len({r["id"] for r in rows}) == 400
                            for rows in self.data.values()), "full_speed_requires_400_unique_ids")
        expected = schedule(self.cfg, self.data) if self.cfg["mode"] not in ("quality", "smoke") else []
        require(r["schedule"] == expected, "saved_schedule_mismatch")
        require(all((job.get("bucket"), job.get("batch_size")) in cells(self.cfg)
                    for job in self.jobs if job["stage"] in ("preflight", "latency", "profile")), "unexpected_worker_cell")
        self.invalid_cells = set()
        fields = ("phase", "block", "bucket", "batch_size", "variant", "side")
        positions = {tuple(j[k] for k in fields): i for i, j in enumerate(expected)}
        previous = -1
        for job in self.jobs:
            if job["stage"] != "latency":
                continue
            position = positions.get(tuple(job.get(k) for k in fields), -1)
            if position <= previous:
                self.invalid_cells.add((job["bucket"], job["batch_size"]))
            previous = max(position, previous)

    @property
    def variants(self):
        return ["baseline"] + (["candidate"] if "candidate" in self.cfg else [])

    def job_path(self, i, job):
        return safe(self.root, self.paths[i].parent)

    def checked(self, i, job, allow_partial=False, track=True):
        path = self.job_path(i, job)
        if not allow_partial:
            require(self.state["stages"][i].get("status") == "passed", "stage_not_passed")
        require(read(safe(self.root, path / "job.json")) == job, "saved_job_mismatch")
        value = read(safe(self.root, path / "result.json"))
        require(value.get("result_version") == 4, "unsupported_worker_format")
        require(value.get("job") == job and value.get("resolved_sha256") == self.sha
                and job.get("resolved_sha256") == self.sha and job.get("seed") == self.cfg["seed"],
                "worker_identity_mismatch")
        require(value.get("status") in (("passed", "failed", "oom", "unsupported", "error", "running", "timeout")
                if allow_partial else ("passed",)), "worker_not_passed")
        require(self.state["stages"][i].get("status") == value["status"], "worker_ledger_status_mismatch")
        for name in ("source_inventory", "protocol_version", "data_manifest_sha256"):
            if name in value:
                require(value[name] == self.resolved[name], "worker_provenance_mismatch")
        runtime = value.get("runtime")
        require(isinstance(runtime, dict) and bool(runtime), "missing_worker_runtime")
        require(environment_identity(runtime) == environment_identity(self.resolved["environment"]),
                "worker_environment_mismatch")
        precision = validate_precision(value.get("metadata", {}).get("precision"))
        try:
            tracker = self.precision if track else deepcopy(self.precision)
            tracker.add(job["variant"], value)
        except ValueError as error:
            raise InvalidArtifact("worker_precision_mismatch") from error
        dtype = self.cfg[job["variant"]]["execution"]["dtype"]
        require(dtype == "native_sdk" or dtype in precision["parameter_dtypes"]
                or (precision["autocast_enabled"] and precision["autocast_dtype"] == dtype),
                "configured_precision_mismatch")
        fields = ("artifacts", "preparation", "batch_policy", "attention", "linear_attention")
        metadata = value["metadata"]
        observed = {field: digest(metadata[field]) for field in fields if field in metadata}
        previous = self.observed.get(job["variant"])
        if previous is not None:
            for field in fields:
                # A variant may omit optional evidence entirely, but dropping it
                # in later workers must not hide a changed observed identity.
                require(previous.get(field) == observed.get(field), f"worker_{field}_mismatch")
        if track:
            self.observed[job["variant"]] = observed
            if i not in self.recorded_indices:
                self.records.append((job, value))
                self.recorded_indices.add(i)
        return value


def latency_rows(job, value):
    plans, rows = job["batches"], value["rows"]
    require(digest(plans) == job["plan_sha256"] and len(rows) == len(plans), "batch_count_or_hash_mismatch")
    require(all(finite(value.get(k)) for k in ("setup_seconds", "warmup_seconds")), "invalid_setup_duration")
    require(type(value.get("compile_before")) is int and value["compile_before"] >= 0
            and type(value.get("compile_after")) is int and value["compile_after"] == value["compile_before"],
            "compilation_during_measurement")
    labels, previous = defaultdict(list), None
    for row, plan in zip(rows, plans, strict=True):
        require(row.get("batch_id") == plan["id"] and all(row.get(k) == plan[k] for k in
                ("ids", "input_sha256", "total_tokens", "padded_tokens", "useful_tokens")), "batch_plan_mismatch")
        require(finite(row.get("latency_ms"), True), "invalid_latency")
        stamp = row.get("timestamp")
        require(finite(stamp) and (previous is None or stamp >= previous), "invalid_timestamp")
        previous = stamp
        require(isinstance(row.get("scores"), list) and len(row["scores"]) == len(plan["ids"]),
                "response_count_mismatch")
        for key, score in zip(plan["ids"], row["scores"], strict=True):
            labels[key].append(response(score))
    return labels


def _preflights(evidence, cell):
    cfg = evidence.cfg
    if cfg["mode"] == "quality":
        return
    variants = ["candidate"] if cfg["mode"] == "search" else evidence.variants
    required = {(cell[0], cell[1], v) for v in variants}
    seen = set()
    for i, job in enumerate(evidence.jobs):
        if job["stage"] != "preflight":
            continue
        key = (job["bucket"], job["batch_size"], job["variant"])
        if key[:2] != cell:
            continue
        require(key in required and key not in seen, "unexpected_preflight")
        value = evidence.checked(i, job)
        expected = smoke_plans(evidence.data[job["bucket"]], job["batch_size"], cfg["seed"])
        require(value.get("plans") == expected, "preflight_plan_mismatch")
        require(isinstance(value.get("backend_proof"), dict) and bool(value["backend_proof"])
                and value["backend_proof"].get("status", "passed") == "passed" and value.get("aba_passed") is True
                and value.get("over_limit_rejected") is True
                and value.get("boundary_checked") == cfg["buckets"][job["bucket"]]["max"]
                and job.get("bucket_max") == cfg["buckets"][job["bucket"]]["max"], "missing_preflight_proof")
        responses = value["responses"]
        require(len(responses) == len(expected), "preflight_response_count_mismatch")
        for values, plan in zip(responses, expected):
            require(len(values) == len(plan["ids"]), "preflight_response_count_mismatch")
            for score in values:
                response(score)
        seen.add(key)
        for latency in evidence.jobs:
            if latency["stage"] == "latency" and (latency["bucket"], latency["batch_size"], latency["variant"]) == key:
                require(latency.get("preflight_sha256") == sha(evidence.job_path(i, job) / "result.json"),
                        "preflight_hash_mismatch")
    require(seen == required, "missing_preflight")


def _profiles(evidence, cell):
    output = {}
    variants = ["candidate"] if evidence.cfg["mode"] == "search" else evidence.variants
    for i, job in enumerate(evidence.jobs):
        if job["stage"] != "profile" or (job["bucket"], job["batch_size"]) != cell:
            continue
        try:
            require(job["variant"] in variants and job["variant"] not in output, "unexpected_profile")
            value = evidence.checked(i, job, track=False)
            require(value.get("profiles") == ["short.trace.json.gz", "long.trace.json.gz"], "missing_profile_evidence")
            output[job["variant"]] = {"status": "passed", "profiles": value["profiles"], "errors": [],
                                      "worker_evidence": _worker_evidence(job, value)}
        except ARTIFACT_ERRORS as error:
            output[job["variant"]] = {"status": "failed", "errors": [_error(error)]}
    if evidence.cfg["profile_trace"]:
        for variant in variants:
            output.setdefault(variant, {"status": "unavailable", "errors": ["missing_profile"]})
    statuses = [v["status"] for v in output.values()]
    status = ("not_requested" if not statuses else "passed" if all(s == "passed" for s in statuses)
              else "partial" if "passed" in statuses else "failed" if "failed" in statuses else "unavailable")
    return {"status": status, "variants": output}


def _series(records):
    groups, labels, throughput = defaultdict(dict), defaultdict(lambda: defaultdict(list)), {}
    for job, value in records:
        key = (job["phase"], job["block"])
        require(job["side"] not in groups[key], "duplicate_pair_side")
        groups[key][job["side"]] = value["rows"]
        for key, scores in latency_rows(job, value).items():
            labels[job["variant"]][key].extend(scores)
        if job["phase"] != "aa":
            rate = throughput.setdefault(job["variant"], {"batches": 0, "texts": 0, "latency_ms": 0,
                                                          "useful_tokens": 0, "padded_tokens": 0})
            for row in value["rows"]:
                rate["batches"] += 1
                rate["texts"] += len(row["ids"])
                for k in ("latency_ms", "useful_tokens", "padded_tokens"):
                    rate[k] += row[k]
    series = defaultdict(list)
    for (phase, _), sides in sorted(groups.items()):
        if phase in ("aa", "ab"):
            require(set(sides) == {"a", "b"}, "missing_paired_side")
            require([r["batch_id"] for r in sides["a"]] == [r["batch_id"] for r in sides["b"]],
                    "paired_plan_mismatch")
            series[phase].append(tuple([r["latency_ms"] for r in sides[s]] for s in ("a", "b")))
        else:
            series[phase].append([r["latency_ms"] for r in next(iter(sides.values()))])
    for rate in throughput.values():
        rate["requests_per_second"] = rate["texts"] * 1000 / rate["latency_ms"]
        rate["padding_fraction"] = 1 - rate["useful_tokens"] / rate["padded_tokens"]
    return series, labels, throughput


def _reference(evidence):
    root = safe(evidence.root, "reference")
    manifest = read(safe(evidence.root, "reference.json"))
    require(checksum(manifest.get("sha256")) and isinstance(manifest.get("files"), dict),
            "invalid_reference_manifest")
    required = {"resolved_run.json", "status.json", "summary.json"}
    require(required <= manifest["files"].keys(), "missing_reference_files")
    for name, expected in manifest["files"].items():
        require(checksum(expected) and sha(safe(root, name)) == expected, "reference_file_hash_mismatch")
    ref = Evidence(root)
    required.update(str(ref.job_path(i, job).relative_to(root) / filename)
                    for i, job in enumerate(ref.jobs) for filename in ("job.json", "result.json"))
    require(required <= manifest["files"].keys(), "unhashed_reference_jobs")
    require(ref.cfg["mode"] == "baseline" and not ref.cfg["pilot"], "reference_not_full_baseline")
    verified = _build(root)[0]
    require(verified["status"] == "baseline_complete", "incomplete_reference")
    for k in ("baseline", "buckets", "batch_sizes_by_bucket", "seed"):
        require(ref.cfg[k] == evidence.cfg[k], "reference_config_mismatch")
    for k in ("source_inventory", "input_indices", "data_manifest_sha256", "protocol_version"):
        require(ref.resolved[k] == evidence.resolved[k], "reference_provenance_mismatch")
    assert_environment(ref.resolved["environment"], evidence.resolved["environment"])
    main = ref.resolved["schedule"]
    if evidence.cfg["pilot"]:
        original = {(j["bucket"], j["batch_size"], j["block"]): j for j in main if j["phase"] == "baseline"}
        for job in evidence.resolved["schedule"]:
            previous = original[job["bucket"], job["batch_size"], job["block"]]
            require(job["batches"] == previous["batches"][:len(job["batches"])], "reference_pilot_plan_mismatch")
    else:
        schedule(evidence.cfg, evidence.data, reference_schedule=main)
    for i, job in enumerate(ref.jobs):
        if job["stage"] in ("preflight", "latency"):
            value = ref.checked(i, job)
            evidence.precision.add("baseline", value)
    return ref, verified


def _cell(evidence, bucket, size, reference=None):
    cfg = evidence.cfg
    require((bucket, size) not in evidence.invalid_cells, "latency_schedule_mismatch")
    _preflights(evidence, (bucket, size))
    selected = [(i, j) for i, j in enumerate(evidence.jobs)
                if j["stage"] == "latency" and (j["bucket"], j["batch_size"]) == (bucket, size)]
    expected = [j for j in evidence.resolved["schedule"] if (j["bucket"], j["batch_size"]) == (bucket, size)]
    require(len(selected) == len(expected) and all({k: actual.get(k) for k in planned} == planned
            for (_, actual), planned in zip(selected, expected)), "latency_schedule_mismatch")
    records = [(job, evidence.checked(i, job)) for i, job in selected]
    series, labels, rates = _series(records)
    resources = [_worker_evidence(job, value) for job, value in records]
    mode, pilot = cfg["mode"], cfg["pilot"]
    item = {"bucket": bucket, "batch_size": size, "complete": True, "faster": False,
            "timing_unit": "batch", "throughput": rates, "errors": []}
    if mode == "search":
        ref, summary = reference
        records = [(j, ref.checked(i, j)) for i, j in enumerate(ref.jobs)
                   if j["stage"] == "latency" and (j["bucket"], j["batch_size"]) == (bucket, size)]
        base, base_labels, base_rates = _series(records)
        resources = [_worker_evidence(job, value, "reference") for job, value in records] + resources
        labels["baseline"] = base_labels["baseline"]
        rates.update(base_rates)
        for name, q in (("p50", .5), ("p95", .95)):
            a = quantile([v for block in base["baseline"] for v in block], q)
            b = quantile([v for block in series["search"] for v in block], q)
            item[name] = {"baseline_ms": a, "candidate_ms": b, "ratio": b / a,
                          "ratio_interval_95": None}
        item.update(status="exploratory", speed_status="exploratory", paired=False,
                    p95_reduction=1 - item["p95"]["ratio"],
                    reference_diagnostics=summary["cells"][f"{bucket}/{size}"],
                    uncertainty_note="Separate runs are not paired; no paired confidence interval or final acceptance.")
    elif mode == "baseline" and not pilot:
        item.update(baseline_diagnostics(series["baseline"], series["aa"]))
        item.update(status="baseline_complete", speed_status="measured_baseline")
    elif mode == "confirm" and not pilot:
        threshold = .10 if cfg["comparison"] == "parameter" else .20
        item.update(assess(series["ab"], series["aa"], threshold))
        item["speed_status"] = item["status"]
        item["faster"] = not item["reasons"] and item["p95_reduction"] > 0
        item["paired"] = True
    else:
        phases = ([block[0] for block in series["ab"]], [block[1] for block in series["ab"]]) if mode == "confirm" else (series["baseline"], [])
        for name, q in (("p50", .5), ("p95", .95)):
            item[name] = {"baseline_ms": quantile([v for block in phases[0] for v in block], q)}
            if phases[1]:
                item[name]["candidate_ms"] = quantile([v for block in phases[1] for v in block], q)
        item.update(status="pilot_complete", speed_status="not_evaluated_pilot")
    if not pilot:
        expected = {row["id"] for row in evidence.data[bucket]}
        require(all(set(labels[v]) == expected for v in evidence.variants), "incomplete_unique_responses")
    if pilot and mode == "search":
        labels["baseline"] = {key: labels["baseline"][key] for key in labels["candidate"]}
    verdict = agreement(labels["baseline"], labels["candidate"] if "candidate" in cfg else labels["baseline"])
    item.update(verdicts=verdict, verdict_status="unstable" if verdict["unstable_ids"] else
                "changed" if verdict["changed_count"] else "unchanged")
    if pilot:
        item.update(status="pilot_complete", speed_status="not_evaluated_pilot", faster=False)
    elif mode == "confirm":
        item["status"] = ("rejected" if verdict["status"] != "ok" or item["speed_status"] == "below_target"
                          else "confirm_pass" if item["speed_status"] == "pass" else "inconclusive")
        if cfg["comparison"] == "quantization":
            item.update(status="quantization_measured", speed_status="measured_no_acceptance_threshold")
    item.update(series_evidence=resources, **_resource_statuses(resources))
    digest(item)  # Reject overflow in derived rates/intervals as invalid evidence.
    evidence.measured[bucket, size] = labels
    return item


def metrics(rows, predictions, mapping):
    """The quality.py metric format, using only explicit saved label mappings."""
    labels = mapping["labels"]
    require(len(labels) >= 2 and len(set(labels)) == len(labels)
            and all(isinstance(label, str) and label for label in labels), "invalid_quality_mapping")
    require(all(isinstance(mapping[k], dict) and bool(mapping[k])
                and set(mapping[k].values()) <= set(labels) for k in ("targets", "verdicts")),
            "invalid_quality_mapping")
    require({r["target"] for r in rows} <= mapping["targets"].keys(), "unmapped_quality_target")
    if "positive_label" in mapping:
        require(len(labels) == 2 and mapping["positive_label"] in labels, "invalid_positive_label")
    by_id = {p["id"]: p for p in predictions}
    require(len(by_id) == len(predictions) and set(by_id) == {r["id"] for r in rows},
            "quality_prediction_ids_mismatch")

    def compute(subset):
        matrix = {a: dict.fromkeys(labels, 0) for a in labels}
        failures = Counter()
        for row in subset:
            p = by_id[row["id"]]
            if p.get("status") != "passed":
                failures[p.get("status", "missing")] += 1
                continue
            native = response(p["score"])["label"]
            require(native in mapping["verdicts"], "unmapped_native_verdict")
            matrix[mapping["targets"][row["target"]]][mapping["verdicts"][native]] += 1
        n = sum(sum(v.values()) for v in matrix.values())
        classes = {}
        for label in labels:
            tp = matrix[label][label]
            fp = sum(matrix[x][label] for x in labels if x != label)
            fn = sum(matrix[label][x] for x in labels if x != label)
            denom = 2 * tp + fp + fn
            classes[label] = {"support": sum(matrix[label].values()), "f1": 2 * tp / denom if denom else None}
        f1s = [v["f1"] for v in classes.values()]
        result = {"total": len(subset), "evaluated": n, "coverage": n / len(subset) if subset else None,
                  "failures": dict(failures), "confusion_matrix": matrix, "per_class": classes,
                  "accuracy": sum(matrix[l][l] for l in labels) / n if n else None,
                  "macro_f1": sum(f1s) / len(f1s) if all(v is not None for v in f1s) else None,
                  "undefined_f1_labels": [l for l in labels if classes[l]["f1"] is None]}
        if "positive_label" in mapping:
            pos = mapping["positive_label"]
            neg = next(l for l in labels if l != pos)
            negatives, positives = sum(matrix[neg].values()), sum(matrix[pos].values())
            result.update(fpr=matrix[neg][pos] / negatives if negatives else None,
                          fnr=matrix[pos][neg] / positives if positives else None,
                          negative_n=negatives, positive_n=positives)
        return result

    keys = set(rows[0].get("slices", {})) if rows else set()
    require(all(set(r.get("slices", {})) == keys for r in rows), "incomplete_quality_slices")
    slices = {name: {value: compute([r for r in rows if r["slices"][name] == value])
                     for value in sorted({r["slices"][name] for r in rows})} for name in sorted(keys)}
    return {"status": "evaluated", "quality_preserved": None, "overall": compute(rows), "slices": slices}


def _quality_plan(rows, plan, size, seed):
    # Recreate quality_batches.make_plan without importing adapters/catalogs.
    ids = [r["id"] for r in rows]
    random.Random(seed).shuffle(ids)
    indices, excluded = plan["indices"], plan["excluded"]
    require(set(excluded) <= set(ids) and set(indices) <= set(ids)
            and set(ids) - set(excluded) <= set(indices), "quality_index_ids_mismatch")
    eligible = [key for key in ids if key not in excluded]
    for index in indices.values():
        require(type(index.get("total_tokens")) is int and index["total_tokens"] > 0
                and checksum(index.get("input_sha256")), "invalid_quality_input_index")
    batches = []
    for start in range(0, len(eligible), size):
        members = eligible[start:start + size]
        lengths = [indices[key]["total_tokens"] for key in members]
        batches.append({"ids": members, "actual_batch_size": len(members), "lengths": lengths,
                        "padded_length": max(lengths), "inputs_sha256": digest([indices[key] for key in members])})
    require(plan == {"quality_protocol_version": 1, "batch_size": size, "seed": seed,
                     "ordered_ids": ids, "indices": indices, "excluded": excluded, "batches": batches},
            "quality_plan_mismatch")


def _quality_comparison(rows, left, right, mapping, same_size):
    if not left or not right:
        return {"status": "unavailable", "n": 0, "total": len(rows)}
    a, b = left[0]["plan"], right[0]["plan"]
    fields = ("ordered_ids", "indices", "excluded") + (("batches",) if same_size else ())
    if any(a[k] != b[k] for k in fields):
        return {"status": "blocked_input_mismatch", "n": 0, "total": len(rows)}
    common = set(r["id"] for r in rows)
    for item in left + right:
        common &= {p["id"] for p in item["predictions"] if p["status"] == "passed"}
    subset = [r for r in rows if r["id"] in common]
    if not subset:
        return {"status": "no_common_predictions", "n": 0, "total": len(rows), "coverage": 0}
    scores = []
    for side in (left, right):
        collected = defaultdict(list)
        for item in side:
            for p in item["predictions"]:
                if p["id"] in common:
                    collected[p["id"]].append(p["score"])
        scores.append(collected)
    changes = agreement(*scores)
    changes.pop("status")
    complete = len(common) == len(rows) and all(item["status"] == "passed" for item in left + right)
    return {**changes, "status": "complete" if complete else "partial", "total": len(rows),
            "coverage": len(common) / len(rows),
            "left_quality": metrics(subset, [p for p in left[0]["predictions"] if p["id"] in common], mapping),
            "right_quality": metrics(subset, [p for p in right[0]["predictions"] if p["id"] in common], mapping)}


def _quality(evidence):
    cfg = evidence.cfg
    rows = evidence.resolved.get("quality_rows")
    require(isinstance(rows, list) and bool(rows), "missing_public_quality_rows")
    public_ids = {}
    if all("id_sha256" in r and "id" not in r for r in rows):
        # Public rows retain corpus order but hash IDs. Resolve those hashes only
        # against the saved quality plan; never reopen the private corpus.
        plans = [safe(evidence.root, evidence.job_path(i, job) / "quality_plan.json")
                 for i, job in enumerate(evidence.jobs) if job["stage"] == "quality"]
        plan_path = next((p for p in plans if p.exists()), None)
        require(plan_path is not None, "missing_quality_plan")
        raw_ids = read(plan_path)["ordered_ids"]
        by_hash = {digest(key): key for key in raw_ids}
        require(len(by_hash) == len(raw_ids) and {r["id_sha256"] for r in rows} == set(by_hash),
                "quality_public_ids_mismatch")
        public_ids = {raw: hashed for hashed, raw in by_hash.items()}
        rows = [{**{k: v for k, v in row.items() if k != "id_sha256"}, "id": by_hash[row["id_sha256"]]} for row in rows]
    require(all(isinstance(r.get("id"), str) and r["id"] and "target" in r for r in rows)
            and len({r["id"] for r in rows}) == len(rows), "invalid_quality_rows")
    allowed = {"id", "target", "slices", "content_sha256", "text_sha256", "target_role", "target_index"}
    require(all(not set(r) - allowed for r in rows), "nonpublic_quality_rows")
    mapping = cfg["label_mapping"]
    expected = {(s, v) for s in cfg["quality_batch_sizes"] for v in evidence.variants}
    collected, errors, by_size = defaultdict(list), [], {}
    for i, job in enumerate(evidence.jobs):
        if job["stage"] != "quality":
            continue
        key = (job["quality_batch_size"], job["variant"])
        require(key in expected and job.get("label_mapping") == mapping, "unexpected_quality_job")
        try:
            value = evidence.checked(i, job, allow_partial=True)
            require(not any(value.get(k) == "PrecisionError" for k in
                            ("error_type", "cleanup_error_type", "close_error_type")),
                    "quality_precision_failure")
            path = evidence.job_path(i, job)
            plan = read(safe(evidence.root, path / "quality_plan.json"))
            require(value.get("quality_batch_size") == key[0]
                    and value.get("quality_plan_sha256") == digest(plan), "quality_plan_hash_mismatch")
            _quality_plan(rows, plan, key[0], cfg["seed"])
            predictions = [json.loads(line, object_pairs_hook=_pairs, parse_constant=_constant)
                           for line in safe(evidence.root, path / "predictions.jsonl").read_text().splitlines() if line.strip()]
            require([p["id"] for p in predictions] == plan["ordered_ids"], "quality_prediction_order_mismatch")
            for p in predictions:
                if p["id"] in plan["excluded"]:
                    require(p["status"] == plan["excluded"][p["id"]], "quality_exclusion_mismatch")
                elif value["status"] == "passed":
                    require(p["status"] == "passed", "unfinished_quality_prediction")
            if any(p["status"] == "passed" for p in predictions):
                require(isinstance(value.get("backend_proof"), dict) and bool(value["backend_proof"])
                        and value["backend_proof"].get("status", "passed") == "passed", "missing_quality_backend_proof")
            calculated = metrics(rows, predictions, mapping)
            require(calculated == value.get("quality"), "quality_summary_mismatch")
            if collected[key]:
                require(collected[key][0]["plan"] == plan, "quality_repeat_plan_mismatch")
            collected[key].append({"plan": plan, "predictions": predictions, "quality": calculated,
                                   "status": value["status"]})
        except ARTIFACT_ERRORS as error:
            errors.append(_error(error))
    complete, covered = not errors, True
    for size, variant in sorted(expected):
        items = collected[size, variant]
        cell = by_size.setdefault(str(size), {"variants": {}})
        if not items:
            cell["variants"][variant] = {"status": "incomplete", "total": len(rows), "quality_evaluated": False}
            complete = covered = False
            continue
        passed = all(i["status"] == "passed" for i in items)
        complete &= passed
        covered &= all(i["quality"]["overall"]["coverage"] == 1 for i in items)
        repeated = defaultdict(list)
        for item in items:
            for p in item["predictions"]:
                if p["status"] == "passed":
                    repeated[p["id"]].append(p["score"])
        unstable = [key for key, values in repeated.items() if len({s["label"] for s in values}) > 1]
        repeat_ids = sum(len(values) > 1 for values in repeated.values())
        cell["variants"][variant] = {"status": "passed" if passed else "incomplete", "total": len(rows),
            "quality": items[0]["quality"], "repeats": [i["quality"] for i in items],
            "quality_evaluated": bool(repeated), "batch_count": len(items[0]["plan"]["batches"]),
            "actual_batch_sizes": [b["actual_batch_size"] for b in items[0]["plan"]["batches"]],
            "repeat_stability": {"status": "unstable" if unstable else "stable" if repeat_ids else "not_assessed",
                                 "n": repeat_ids, "unstable_ids": sorted(unstable)}}
    single = {}
    for size in cfg["quality_batch_sizes"]:
        if "candidate" in cfg:
            comparison = _quality_comparison(rows, collected[size, "baseline"], collected[size, "candidate"], mapping, True)
            by_size[str(size)]["baseline_to_candidate"] = comparison
            if comparison["status"].startswith("blocked_"):
                complete = False
        if size != 1 and 1 in cfg["quality_batch_sizes"]:
            single[str(size)] = {v: _quality_comparison(rows, collected[1, v], collected[size, v], mapping, False)
                                 for v in evidence.variants}
            if any(v["status"].startswith("blocked_") for v in single[str(size)].values()):
                complete = False
    result = {"quality_by_batch_size": by_size, "quality_single_to_batch": single,
            "quality_batch_sizes": cfg["quality_batch_sizes"], "quality_complete": bool(complete),
            "quality_status": "evaluated" if complete and covered else "partial_coverage" if complete else "incomplete",
            "quality_evaluated": any(v.get("quality_evaluated") for c in by_size.values() for v in c["variants"].values()),
            "quality_preserved": None, "errors": errors}
    if public_ids:
        def public(value):
            if isinstance(value, dict):
                return {key: [public_ids[v] for v in item] if key in ("changed_ids", "unstable_ids")
                        else public(item) for key, item in value.items()}
            return [public(v) for v in value] if isinstance(value, list) else value
        result = public(result)
        result["id_encoding"] = "sha256"
    return result


def _monitoring(resolved):
    value = resolved.get("monitoring_validation")
    if not isinstance(value, dict) or value.get("status") != "passed":
        return "unavailable"
    if (value.get("environment") != environment_identity(resolved["environment"])
            or value.get("monitor_method") != resolved["environment"].get("monitor_method")
            or value.get("source_inventory_sha256") != digest(resolved["source_inventory"])):
        return "mismatch"
    checks = value.get("checks")
    if not isinstance(checks, dict) or set(checks) != {"gliner", "open"}:
        return "unavailable"
    for check in checks.values():
        if not isinstance(check, dict) or check.get("status") != "passed" or check.get("memory_available", True) is not True:
            return "unavailable"
        for metric in ("p50", "p95"):
            interval = check.get(metric, {}).get("ratio_interval_95") if isinstance(check.get(metric), dict) else None
            if (not isinstance(interval, list) or len(interval) != 2 or not all(finite(v, True) for v in interval)
                    or interval[0] > interval[1] or interval[1] > 1.05):
                return "unvalidated"
    return "passed"


def _availability(statuses):
    statuses = list(statuses)
    if not statuses or all(s == "unavailable" for s in statuses):
        return "unavailable"
    if all(s == "measured" for s in statuses):
        return "measured"
    return "invalid" if all(s == "invalid" for s in statuses) else "partial"


def _number(value, key, *, integer=False, positive=False):
    if key not in value or value[key] is None:
        return {"status": "unavailable", "value": None}
    valid = finite(value[key], positive) and (not integer or type(value[key]) is int)
    return {"status": "measured" if valid else "invalid", "value": value[key] if valid else None}


def _worker_evidence(job, value, source="run"):
    durations = {k: _number(value, k) for k in ("setup_seconds", "warmup_seconds", "backend_preparation_seconds")}
    preparation = {"status": _availability(v["status"] for v in durations.values()), **durations}
    raw_memory = value.get("memory")
    raw_memory = raw_memory if isinstance(raw_memory, dict) else {}
    windows = raw_memory.get("windows", {})
    windows = windows if isinstance(windows, dict) else {}
    sampler = {}
    for phase in ("setup", "inference"):
        window = windows.get(phase)
        window = window if isinstance(window, dict) else {}
        peak = _number(window, "sampled_gpu_peak_mib", positive=True)
        samples = _number(window, "valid_gpu_samples", integer=True, positive=True)
        measured = peak["status"] == samples["status"] == "measured"
        status = ("measured" if measured and window.get("status") == "measured" else "partial" if measured
                  else "invalid" if "invalid" in (peak["status"], samples["status"]) else "unavailable")
        sampler[phase] = {"status": status, "sampled_gpu_peak_mib": peak["value"],
                          "valid_gpu_samples": samples["value"],
                          "host_rss_peak_bytes": _number(window, "host_rss_peak_bytes", integer=True)}
    sampler["status"] = _availability(sampler[p]["status"] for p in ("setup", "inference"))
    sampler["sampling_may_miss_short_peaks"] = True
    counters = {}
    for phase in ("torch_setup", "torch_inference"):
        raw = value.get(phase)
        raw = raw if isinstance(raw, dict) else {}
        fields = {k: _number(raw, k, integer=True) for k in ("allocated_bytes", "reserved_bytes")}
        counters[phase] = {"status": _availability(v["status"] for v in fields.values()), **fields}
    memory = {"status": _availability([sampler["status"], *(v["status"] for v in counters.values())]),
              "sampler": sampler, **counters}
    raw_size = value.get("size")
    raw_size = raw_size if isinstance(raw_size, dict) else {}
    storage = _number(raw_size, "unique_parameter_buffer_storage_bytes", integer=True)
    size = {"unique_parameter_buffer_storage_bytes": storage}
    for field in ("checkpoint_files_bytes", "export_files_bytes"):
        files = raw_size.get(field)
        valid = (isinstance(files, dict) and all(isinstance(k, str) and type(v) is int and v >= 0 for k, v in files.items()))
        size[field] = {"status": "measured" if valid else "unavailable" if files is None else "invalid",
                       "files": files if valid else None, "total_bytes": sum(files.values()) if valid else None}
    size["status"] = _availability(v["status"] for v in size.values())
    if isinstance(raw_size.get("dtypes"), list) and all(isinstance(t, str) for t in raw_size["dtypes"]):
        size["dtypes"] = raw_size["dtypes"]
    if type(raw_size.get("original_encoder_retained")) is bool:
        size["original_encoder_retained"] = raw_size["original_encoder_retained"]
    return {"source": source, **{k: job[k] for k in ("stage", "variant", "bucket", "batch_size", "phase", "block", "side",
                                                    "quality_batch_size") if k in job},
            "preparation": preparation, "memory": memory, "size": size}


def _resource_statuses(records):
    return {name + "_status": _availability(r[name]["status"] for r in records)
            for name in ("preparation", "memory", "size")}


def _single_to_batch(evidence):
    output = {}
    for bucket, size in cells(evidence.cfg):
        if size == 1:
            continue
        variants = output[f"{bucket}/{size}"] = {}
        for variant in evidence.variants:
            left = evidence.measured.get((bucket, 1), {}).get(variant, {})
            right = evidence.measured.get((bucket, size), {}).get(variant, {})
            common = sorted(set(left) & set(right))
            total = len(evidence.data[bucket])
            item = {"status": "unavailable" if not left or not right else "no_common_predictions",
                    "n": len(common), "common_n": len(common), "total": total,
                    "single_unique_n": len(left), "batch_unique_n": len(right), "coverage": len(common) / total,
                    "changed_fraction": None, "quality_preserved": None}
            if common:
                changes = agreement({k: left[k] for k in common}, {k: right[k] for k in common})
                changes["verdict_status"] = changes.pop("status")
                item.update(changes, status="complete" if len(common) == total else "partial")
            variants[variant] = item
    return output


def _quantization(cfg, results):
    if cfg["comparison"] != "quantization" or "candidate" not in cfg:
        return {"status": "not_applicable", "quality_preserved": None, "by_batch_size": {}}
    by_size = {}
    for size in sorted({s for _, s in cells(cfg)}):
        required = [b for b, s in cells(cfg) if s == size]
        buckets = {}
        for bucket in required:
            item = results.get(f"{bucket}/{size}", {})
            verdict = item.get("verdicts")
            if not item.get("complete") or not verdict:
                continue
            buckets[bucket] = {"n": verdict["n"], "changed_count": verdict["changed_count"],
                               "changed_fraction": verdict["changed_fraction"], "unstable_count": len(verdict["unstable_ids"]),
                               "unstable_fraction": len(verdict["unstable_ids"]) / verdict["n"]}
        complete = len(buckets) == len(required)
        by_size[str(size)] = {"status": "complete" if complete else "partial" if buckets else "unavailable",
            "required_buckets": required, "measured_buckets": len(buckets), "buckets": buckets,
            "macro_changed_fraction": sum(v["changed_fraction"] for v in buckets.values()) / len(required) if complete else None,
            "macro_unstable_fraction": sum(v["unstable_fraction"] for v in buckets.values()) / len(required) if complete else None}
    return {"status": "measured" if all(v["status"] == "complete" for v in by_size.values()) else "partial",
            "quality_preserved": None, "acceptance": "not_assessed_without_quality_rule", "by_batch_size": by_size}


def _build(root):
    result = {"schema_version": 4, "result_version": 4, "protocol_version": PROTOCOL_VERSION,
              "resolved_sha256": None, "mode": None, "comparison": None, "status": "incomplete", "complete": False,
              "speed_status": "not_measured", "verdict_status": "not_measured", "cells": {},
              "all_cells_faster": False, "all_cells_meet_threshold": False, "independently_confirmed": False,
              "environment": None, "data_manifest_sha256": None, "monitoring_status": "unavailable",
              "precision": {}, "memory_status": "unavailable", "preparation_status": "unavailable", "size_status": "unavailable",
              "profile_status": "not_requested", "worker_evidence": [], "single_to_batch": {},
              "quantization": {"status": "not_applicable", "quality_preserved": None, "by_batch_size": {}},
              "macro_verdict_change_fraction_by_batch_size": {}, "quality": {"quality_status": "not_evaluated",
                  "quality_evaluated": False, "quality_preserved": None}, "errors": []}
    evidence = None
    try:
        if not safe(root, "resolved_run.json").exists() and safe(root, "config.json").exists():
            raise InvalidArtifact("unsupported_old_format_explicit_migration_required")
        evidence = Evidence(root)
        cfg = evidence.cfg
        result.update(resolved_sha256=evidence.sha, mode=cfg["mode"], comparison=cfg["comparison"],
                      environment=environment_identity(evidence.resolved["environment"]),
                      data_manifest_sha256=evidence.resolved["data_manifest_sha256"],
                      source_inventory_sha256=digest(evidence.resolved["source_inventory"]),
                      monitoring_status=_monitoring(evidence.resolved))
        run_finished = evidence.state.get("status") in ("completed", "baseline_complete", "baseline_unstable", "confirm_pass",
                "inconclusive", "rejected", "exploratory", "pilot_complete", "preflight_passed", "quality_complete",
                "quantization_measured", "incomplete", "partial")
        stages = evidence.state.get("stages")
        require(isinstance(stages, list) and len(stages) == len(evidence.jobs), "stage_ledger_incomplete")
        for stage, path, job in zip(stages, evidence.paths, evidence.jobs):
            require(stage.get("path") == str(path.parent.relative_to(root)) and stage.get("stage") == job["stage"]
                    and stage.get("variant") == job["variant"], "stage_ledger_mismatch")
        require(cfg["mode"] == "quality" or not any(j["stage"] == "quality" for j in evidence.jobs),
                "unexpected_quality_stage")
        reference = _reference(evidence) if cfg["mode"] == "search" else None
        if cfg["mode"] == "quality":
            require(all(j["stage"] == "quality" for j in evidence.jobs), "unexpected_quality_stage")
            result["quality"] = _quality(evidence)
            result["complete"] = result["quality"]["quality_complete"]
            result["status"] = "quality_complete" if result["complete"] else "incomplete"
        elif cfg["mode"] == "smoke":
            require(all(j["stage"] in ("preflight", "profile") for j in evidence.jobs), "unexpected_smoke_stage")
            for bucket, size in cells(cfg):
                item = {"bucket": bucket, "batch_size": size, "status": "preflight_passed", "complete": True,
                        "speed_status": "not_measured", "errors": []}
                result["cells"][f"{bucket}/{size}"] = item
                try:
                    _preflights(evidence, (bucket, size))
                except ARTIFACT_ERRORS as error:
                    item.update(status="incomplete", complete=False, errors=[_error(error)])
                item["profiles"] = _profiles(evidence, (bucket, size))
                item["profile_status"] = item["profiles"]["status"]
            result["complete"] = all(v["complete"] for v in result["cells"].values())
            result["status"] = "preflight_passed" if result["complete"] else "incomplete"
        else:
            for bucket, size in cells(cfg):
                key = f"{bucket}/{size}"
                try:
                    result["cells"][key] = _cell(evidence, bucket, size, reference)
                except ARTIFACT_ERRORS as error:
                    item = result["cells"].setdefault(key, {"bucket": bucket, "batch_size": size})
                    item.update(status="incomplete", complete=False, faster=False, errors=[_error(error)])
                result["cells"][key]["profiles"] = _profiles(evidence, (bucket, size))
                result["cells"][key]["profile_status"] = result["cells"][key]["profiles"]["status"]
            values = list(result["cells"].values())
            result["complete"] = all(v["complete"] for v in values)
            result["verdict_status"] = ("unstable" if any(v.get("verdict_status") == "unstable" for v in values)
                                        else "changed" if any(v.get("verdict_status") == "changed" for v in values)
                                        else "unchanged" if result["complete"] else "incomplete")
            if result["complete"]:
                if cfg["pilot"]:
                    result.update(status="pilot_complete", speed_status="not_evaluated_pilot")
                elif cfg["mode"] == "baseline":
                    result.update(status="baseline_complete", speed_status="measured_baseline")
                elif cfg["mode"] == "search":
                    result.update(status="exploratory", speed_status="exploratory")
                else:
                    result["all_cells_faster"] = result["monitoring_status"] == "passed" and all(v["faster"] for v in values)
                    result["all_cells_meet_threshold"] = all(v["speed_status"] == "pass" for v in values)
                    result["speed_status"] = "pass" if result["all_cells_meet_threshold"] else (
                        "below_target" if any(v["speed_status"] == "below_target" for v in values) else "inconclusive")
                    result["status"] = ("rejected" if any(v["status"] == "rejected" for v in values)
                                        else "confirm_pass" if all(v["status"] == "confirm_pass" for v in values)
                                        else "quantization_measured" if cfg["comparison"] == "quantization" else "inconclusive")
                    if result["status"] == "confirm_pass" and result["monitoring_status"] != "passed":
                        result["status"] = "inconclusive"
                        result["errors"].append("monitoring_validation_not_passed")
        for i, job in enumerate(evidence.jobs):
            if job["stage"] == "memory" and evidence.state["stages"][i].get("status") == "passed":
                # Optional missing memory is not a latency veto. A present worker
                # must still agree with the immutable run/precision identities.
                if (evidence.job_path(i, job) / "result.json").exists():
                    evidence.checked(i, job)
                else:
                    evidence.records.append((job, {}))
            elif job["stage"] == "memory":
                evidence.records.append((job, {}))
        for item in result["cells"].values():
            result["errors"].extend(code for code in item.get("errors", []) if code not in result["errors"])
        if not run_finished:
            result.update(complete=False, status="incomplete", all_cells_faster=False, all_cells_meet_threshold=False)
            result["errors"].append("run_not_completed")
        if not result["complete"] and any(v["complete"] for v in result["cells"].values()):
            result["status"] = "partial"
        result["precision"] = evidence.precision.summary()
        resources = [_worker_evidence(job, value) for job, value in evidence.records]
        if reference:
            resources += [_worker_evidence(job, value, "reference") for job, value in reference[0].records]
        result["worker_evidence"] = resources
        primary = [r for r in resources if r["stage"] == "latency"] or resources
        result.update(_resource_statuses(primary))
        for item in result["cells"].values():
            if "series_evidence" not in item:
                local = [r for r in resources if (r.get("bucket"), r.get("batch_size")) == (item["bucket"], item["batch_size"])]
                item.update(worker_evidence=local, **_resource_statuses(local))
        statuses = [v["profile_status"] for v in result["cells"].values() if v["profile_status"] != "not_requested"]
        result["profile_status"] = ("not_requested" if not statuses else "passed" if all(s == "passed" for s in statuses)
                                    else "partial" if any(s in ("passed", "partial") for s in statuses)
                                    else "failed" if "failed" in statuses else "unavailable")
        if cfg["mode"] not in ("smoke", "quality"):
            result["single_to_batch"] = _single_to_batch(evidence)
            result["quantization"] = _quantization(cfg, result["cells"])
            result["macro_verdict_change_fraction_by_batch_size"] = {
                size: item["macro_changed_fraction"] for size, item in result["quantization"]["by_batch_size"].items()}
    except ARTIFACT_ERRORS as error:
        result.update(status="incomplete", complete=False, all_cells_faster=False, all_cells_meet_threshold=False)
        result["errors"].append(_error(error))
    return result, evidence


def build(directory, save=True):
    """Replay a directory or ZIP; ZIP reports use <archive>.report.json/.md."""
    source = Path(directory).resolve()
    archived = source.is_file()
    with reference_folder(source) as root:
        result, _ = _build(root)
    if save:
        json_path = safe(source.parent, source.name + ".report.json") if archived else safe(source, "summary.json")
        md_path = safe(source.parent, source.name + ".report.md") if archived else safe(source, "summary.md")
        require(not any(p.is_symlink() for p in (json_path, md_path, json_path.with_suffix(".json.tmp"))),
                "unsafe_report_output")
        write(json_path, result)
        text = ["# Offline v4 report", "", f"Status: {result['status']}",
                f"Speed: {result['speed_status']}; memory: {result['memory_status']}",
                f"Preparation: {result['preparation_status']}; size: {result['size_status']}; profiles: {result['profile_status']}",
                f"Quality: {result['quality']['quality_status']}", "",
                "Cell | Status | Baseline p95 ms | Candidate p95 ms", "--- | --- | --- | ---"]
        for name, item in result["cells"].items():
            p95 = item.get("p95", {})
            text.append(f"{name.replace('|', '/')} | {item['status']} | {p95.get('baseline_ms', '-')} | {p95.get('candidate_ms', '-')}")
        text += ["", f"All cells faster above noise: {result['all_cells_faster']}",
                 f"All cells meet threshold: {result['all_cells_meet_threshold']}",
                 "No independent confirmation or automatic quality-preservation approval.", *result["errors"]]
        md_path.write_text("\n".join(text) + "\n", encoding="utf-8")
    return result


def table(inputs, out):
    """Keep separate environment/corpus/protocol groups; never pool run samples."""
    groups, rejected = {}, []
    for source in inputs:
        source = Path(source).resolve()
        try:
            with reference_folder(source) as root:
                summary, evidence = _build(root)
        except (*ARTIFACT_ERRORS, BadZipFile) as error:
            rejected.append({"source": str(source), "status": "incomplete", "errors": [_error(error)]})
            continue
        if evidence is None or not summary["complete"]:
            rejected.append({"source": str(source), "status": summary["status"], "errors": summary["errors"]})
            continue
        identity = {"environment": summary["environment"], "data_manifest_sha256": summary["data_manifest_sha256"],
                    "protocol_version": summary["protocol_version"], "source_inventory_sha256": summary["source_inventory_sha256"]}
        key = digest(identity)
        group = groups.setdefault(key, {"identity": identity, "runs": []})
        group["runs"].append({"source": str(source), "resolved_sha256": summary["resolved_sha256"],
                              "mode": summary["mode"], "status": summary["status"], "cells": summary["cells"],
                              "quality": summary["quality"], "precision": summary["precision"],
                              **{k: summary[k] for k in ("worker_evidence", "single_to_batch", "quantization",
                                  "macro_verdict_change_fraction_by_batch_size", "profile_status", "preparation_status",
                                  "memory_status", "size_status")}})
    result = {"schema_version": 4, "result_version": 4, "status": "partial" if rejected else "completed",
              "rows": sum(len(group["runs"]) for group in groups.values()), "omitted": len(rejected),
              "groups": groups, "rejected": rejected}
    target = Path(out)
    if target.suffix != ".json":
        target = target / "table.json"
    write(target, result)
    return result
