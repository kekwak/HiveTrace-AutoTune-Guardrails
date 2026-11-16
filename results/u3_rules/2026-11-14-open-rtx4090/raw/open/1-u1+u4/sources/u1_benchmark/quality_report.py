"""Offline per-size quality evidence; never grants safety/quality approval."""
from .core import digest, lines, read
from .quality import agreement, metrics
from .quality_batches import batch_sizes, make_plan


def execution_evidence(value, plan, size):
    """Require native execution evidence before admitting non-fixture predictions."""
    metadata = value.get("metadata", {})
    if metadata.get("test_fixture") is True:
        return None
    proof = metadata.get("batch_backend_proof" if size > 1 else "backend_proof")
    if not isinstance(proof, dict):
        raise ValueError("missing_quality_backend_proof")
    kernels = proof.get("kernels")
    if not isinstance(kernels, list) or not kernels or any(not isinstance(k, str) or not k.strip() for k in kernels):
        raise ValueError("invalid_quality_backend_proof")
    shapes = sorted({(b["actual_batch_size"], b["padded_length"]) for b in plan["batches"]})
    if value.get("prepared_shapes") != [list(s) for s in shapes]:
        raise ValueError("quality_prepared_shapes_mismatch")
    if size == 1:
        return None
    policy = metadata.get("batch_policy")
    if (not isinstance(policy, dict) or policy.get("padding") != "longest"
            or policy.get("padding_side") not in ("left", "right")
            or policy.get("truncation") is not False or policy.get("sort_by_length") is not False
            or type(policy.get("pad_token_id")) is not int or policy["pad_token_id"] < 0):
        raise ValueError("invalid_quality_batch_policy")
    return {k: policy[k] for k in ("padding", "padding_side", "truncation", "sort_by_length", "pad_token_id")}


def comparison(rows, left, right, mapping, same_size):
    if not left or not right:
        return {"status": "unavailable", "n": 0, "total": len(rows)}
    if left.get("runtime_identity") != right.get("runtime_identity"):
        return {"status": "blocked_runtime_mismatch", "n": 0, "total": len(rows)}
    if same_size and left.get("padding_policy") != right.get("padding_policy"):
        return {"status": "blocked_padding_mismatch", "n": 0, "total": len(rows)}
    a, b = left["plan"], right["plan"]
    fields = ("ordered_ids", "indices", "excluded") + (("batches",) if same_size else ())
    if any(a[k] != b[k] for k in fields):
        return {"status": "blocked_input_mismatch", "n": 0, "total": len(rows)}
    av = {p["id"]: p for p in left["predictions"] if p["status"] == "passed"}
    bv = {p["id"]: p for p in right["predictions"] if p["status"] == "passed"}
    common = [r for r in rows if r["id"] in av and r["id"] in bv]
    if not common:
        return {"status": "no_common_predictions", "n": 0, "total": len(rows)}
    ids = [r["id"] for r in common]
    changes = agreement({key: [av[key]["score"]] for key in ids},
                        {key: [bv[key]["score"]] for key in ids})
    # Changed labels are observations, not a lossless acceptance criterion.
    changes.pop("status")
    complete = len(common) == len(rows) and left["status"] == right["status"] == "passed"
    return {**changes, "status": "complete" if complete else "partial", "total": len(rows),
            "coverage": len(common) / len(rows),
            "left_quality": metrics(common, [av[key] for key in ids], mapping),
            "right_quality": metrics(common, [bv[key] for key in ids], mapping)}


def collect(root, cfg, checked=None):
    """Validate raw results, including completed predictions from failed workers."""
    from .core import sha
    for name, checksum in read(root / "source_files.json").items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != checksum:
            raise ValueError("quality_source_hash_mismatch")
    rows = read(root / "quality_inputs.json")
    inputs_hash = digest(rows)
    source_rows = {r["id"]: r for r in rows}
    variants = ["baseline"] + (["candidate"] if cfg.get("candidate") else [])
    sizes = batch_sizes(cfg)
    output, evidence = {}, {}
    from .contracts import model_contract
    from .precision import PrecisionTracker
    from .report_v2 import identity
    precision = PrecisionTracker(cfg["comparison"])
    identities, fixtures = set(), False
    execution = read(root / "quality_status.json") if (root / "quality_status.json").exists() else {}
    if not isinstance(execution, dict) or set(execution) - {f"{size}/{v}" for size in sizes for v in variants}:
        raise ValueError("invalid_quality_execution_ledger")
    for size in sizes:
        cell = {"variants": {}}
        output[str(size)] = cell
        for variant in variants:
            directory = root / "quality" / str(size) / variant
            item = {"status": "incomplete", "quality_evaluated": False, "total": len(rows)}
            cell["variants"][variant] = item
            try:
                value = read(directory / "result.json")
                expected = {"quality_protocol_version": 1, "quality_batch_size": size,
                            "quality_inputs_sha256": inputs_hash, "config_sha256": digest(cfg),
                            "variant": variant, "phase": "quality", "inputs_sha256": sha(root / "inputs.json")}
                if any(value.get(k) != v for k, v in expected.items()):
                    raise ValueError("quality_identity_mismatch")
                if value["status"] not in ("passed", "failed", "oom", "unsupported", "running"):
                    raise ValueError("quality_worker_status_invalid")
                fixtures = fixtures or bool(value.get("metadata", {}).get("test_fixture"))
                item["status"] = "incomplete" if value["status"] == "running" else value["status"]
                execution_status = execution.get(f"{size}/{variant}", {}).get("status")
                if execution_status is None:
                    item["status"] = "incomplete"
                elif execution_status != "passed":
                    item["status"] = execution_status
                if value.get("error_type"):
                    item["error_type"] = value["error_type"]
                if not (directory / "quality_plan.json").exists():
                    if value["status"] == "passed":
                        raise ValueError("missing_quality_plan")
                    continue
                if (value.get("error_type") == "PrecisionError"
                        or value.get("cleanup_error_type") == "PrecisionError"
                        or execution.get(f"{size}/{variant}", {}).get("error_type") == "PrecisionError"):
                    raise ValueError("quality_precision_drift_invalidates_partial_predictions")
                # Even failed workers may have partial predictions; their provenance
                # must be valid before those predictions enter metrics/comparisons.
                if (value.get("contract") != model_contract(cfg["model"], cfg["schema_version"])
                        or value.get("adapter_spec") != cfg[variant]):
                    raise ValueError("quality_contract_mismatch")
                if cfg["model"] == "laya" and value.get("precision_policy_version") != 1:
                    raise ValueError("missing_quality_precision_passport")
                precision.add(variant, value)
                runtime_id = identity(directory)
                identities.add(runtime_id)
                if value["status"] == "passed" and checked is not None:
                    checked(directory, variant)
                plan = read(directory / "quality_plan.json")
                if (plan != make_plan(rows, plan["indices"], plan["excluded"], size, cfg["seed"])
                        or digest(plan) != value.get("quality_plan_sha256")):
                    raise ValueError("quality_plan_mismatch")
                predictions = lines(directory / "predictions.jsonl")
                if [p["id"] for p in predictions] != plan["ordered_ids"]:
                    raise ValueError("quality_prediction_order_mismatch")
                for p in predictions:
                    if any(p.get(k) != source_rows[p["id"]][k]
                           for k in ("content_sha256", "text_sha256") if k in source_rows[p["id"]]):
                        raise ValueError("quality_prediction_identity_mismatch")
                    if p["id"] in plan["excluded"] and p["status"] != plan["excluded"][p["id"]]:
                        raise ValueError("quality_exclusion_mismatch")
                    if value["status"] == "passed" and p["id"] not in plan["excluded"] and p["status"] != "passed":
                        raise ValueError("unfinished_quality_prediction")
                padding_policy = None
                if any(p["status"] == "passed" for p in predictions) or (value["status"] == "passed" and plan["batches"]):
                    padding_policy = execution_evidence(value, plan, size)
                calculated = metrics(rows, predictions, cfg["label_mapping"])
                if value.get("quality") != calculated:
                    raise ValueError("quality_summary_mismatch")
                item.update(quality=calculated, quality_evaluated=calculated["overall"]["evaluated"] > 0,
                            batch_count=len(plan["batches"]),
                            actual_batch_sizes=[b["actual_batch_size"] for b in plan["batches"]],
                            plan_sha256=digest(plan))
                evidence[size, variant] = {"plan": plan, "predictions": predictions, "status": item["status"],
                                           "runtime_identity": runtime_id, "padding_policy": padding_policy}
            except (OSError, ValueError, KeyError, TypeError):
                item.update(status="incomplete", error="invalid_or_missing_quality_artifact")
        if cfg.get("candidate"):
            cell["baseline_to_candidate"] = comparison(rows, evidence.get((size, "baseline")),
                                                       evidence.get((size, "candidate")), cfg["label_mapping"], True)
    single = {}
    if 1 in sizes:
        for size in sizes:
            if size != 1:
                single[str(size)] = {variant: comparison(rows, evidence.get((1, variant)),
                    evidence.get((size, variant)), cfg["label_mapping"], False) for variant in variants}
    items = [v for cell in output.values() for v in cell["variants"].values()]
    complete = all(v["status"] == "passed" for v in items) and len(identities) == 1
    paired = [c["baseline_to_candidate"] for c in output.values() if "baseline_to_candidate" in c]
    if any(c["status"].startswith("blocked_") for c in paired):
        complete = False
    covered = complete and all(v.get("quality", {}).get("overall", {}).get("coverage") == 1 for v in items)
    return {"quality_protocol_version": 1, "quality_batch_sizes": sizes,
            "quality_by_batch_size": output, "quality_single_to_batch": single,
            "quality_evaluated": any(v["quality_evaluated"] for v in items),
            "quality_status": "evaluated" if covered else "partial_coverage" if complete else "incomplete",
            "quality_complete": complete, "quality_preserved": None,
            "quality_fixture": fixtures,
            "quality_runtime_identity": next(iter(identities)) if len(identities) == 1 else None}


def markdown(result):
    cells = result.get("quality_by_batch_size", {})
    if not cells:
        return []
    text = ["", "## Качество по размерам батча", "",
            "Batch | Вариант | Статус | N / всего | Accuracy | Macro-F1", "--- | --- | --- | --- | --- | ---"]
    for size, cell in cells.items():
        for variant, value in cell["variants"].items():
            m = value.get("quality", {}).get("overall", {})
            text.append(f"{size} | {variant} | {value['status']} | {m.get('evaluated', 0)} / {m.get('total', value['total'])} | {m.get('accuracy')} | {m.get('macro_f1')}")
    text.append("\nСравнения на общих ID и изменения относительно batch=1 сохранены в JSON; автоматического заключения о сохранении качества нет.")
    return text
