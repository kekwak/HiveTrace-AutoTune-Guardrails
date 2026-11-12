"""Explicit v4 contracts; no checkpoint-name inference or legacy dispatch."""
from copy import deepcopy
from pathlib import Path
import re

from .core import digest, read, sha

SCHEMA_VERSION = 4
RESULT_VERSION = 4
PROTOCOL_VERSION = "u1-static-batch-v4"
MONITOR_VERSION = "nvidia-smi-stream-100ms-v2"


class ConfigurationError(ValueError):
    pass


def positive(value):
    return type(value) is int and value > 0


def sizes(value):
    if not isinstance(value, list) or not value or not all(positive(v) for v in value) or len(set(value)) != len(value):
        raise ConfigurationError("Batch sizes must be unique positive integers")
    return list(value)


def source(value, *, template=False):
    if template and value is None:
        return
    if not isinstance(value, dict):
        raise ConfigurationError("Explicit source required")
    if template and set(value) == {"text"} and isinstance(value["text"], str):
        return
    if "repo" in value:
        allowed = {"repo", "revision", "subfolder", "file", "files"}
        if set(value) - allowed or "path" in value or not isinstance(value["repo"], str):
            raise ConfigurationError("Invalid HF source")
        if not re.fullmatch(r"[0-9a-f]{40}", value.get("revision", "")):
            raise ConfigurationError("HF sources require a full pinned commit")
    elif "path" in value:
        if set(value) - {"path", "inventory", "sha256", "file"}:
            raise ConfigurationError("Unknown local source setting")
        if not value.get("inventory") and not value.get("sha256"):
            raise ConfigurationError("Local sources require an inventory or SHA256")
    else:
        raise ConfigurationError("Missing source location")
    if template and not value.get("file"):
        raise ConfigurationError("Template source requires a file")


def variant(spec):
    required = {"adapter", "weights", "tokenizer", "template", "preparation", "scoring", "execution"}
    if not isinstance(spec, dict) or not required <= spec.keys() or set(spec) - required - {"sdk_assets", "quantization"}:
        raise ConfigurationError("Every variant must be complete; candidate inheritance is not supported")
    adapter = spec["adapter"]
    if isinstance(adapter, dict):
        if set(adapter) != {"module", "source_sha256", "dependencies"}:
            raise ConfigurationError("External adapter requires module, hash and dependencies")
        if not re.fullmatch(r"[0-9a-f]{64}", adapter["source_sha256"]):
            raise ConfigurationError("Invalid adapter hash")
    elif adapter not in {"halo", "gliner", "koala", "llama", "laya", "hive", "open", "shield"}:
        raise ConfigurationError("Unknown explicit adapter")
    source(spec["weights"])
    source(spec["tokenizer"])
    source(spec["template"], template=True)
    for asset in spec.get("sdk_assets", {}).values():
        source(asset)
    if not positive(spec["preparation"].get("limit")):
        raise ConfigurationError("Explicit positive input limit required")
    execution = spec["execution"]
    if execution.get("runtime") not in {"eager", "compile", "ort"}:
        raise ConfigurationError("Explicit runtime required")
    if execution.get("dtype") not in {"float16", "float32", "bfloat16", "native_sdk"}:
        raise ConfigurationError("Explicit precision required")
    return spec


def preparation_identity(spec):
    keys = ("adapter", "tokenizer", "sdk_assets", "template", "preparation")
    return digest({k: spec[k] for k in keys if k in spec})


def declared_inputs(spec):
    # Different adapters can implement the same input contract. Actual IDs/masks
    # are checked against the baseline index before any comparison is timed.
    return {k: spec[k] for k in ("tokenizer", "sdk_assets", "template", "preparation") if k in spec}


def variant_identity(spec):
    return digest(spec)


def validate_config(raw):
    cfg = deepcopy(read(raw) if isinstance(raw, (str, Path)) else raw)
    allowed = {"schema_version", "baseline", "candidate", "buckets", "batch_sizes_by_bucket", "quality_batch_sizes",
               "mode", "pilot", "profile_trace", "comparison", "seed", "timeout_seconds", "environment", "data",
               "quality_data", "label_mapping", "baseline_ref", "monitor_validation"}
    if not isinstance(cfg, dict) or cfg.get("schema_version") != 4:
        raise ConfigurationError("Unsupported schema; v4 requires explicit migration")
    if set(cfg) - allowed:
        raise ConfigurationError("Unknown configuration fields")
    variant(cfg.get("baseline"))
    if cfg.get("candidate") is not None:
        variant(cfg["candidate"])
    else:
        cfg.pop("candidate", None)
    from .v4_preparation import validate_settings
    for name in ("baseline", "candidate"):
        if name in cfg:
            cfg[name] = validate_settings(cfg[name])
    defaults = {"mode": "search" if "candidate" in cfg else "baseline", "pilot": False,
                "profile_trace": False, "comparison": "engineering", "seed": 17,
                "timeout_seconds": 36000, "quality_batch_sizes": [1]}
    cfg = {**defaults, **cfg}
    if cfg["mode"] not in {"baseline", "search", "confirm", "smoke", "quality"}:
        raise ConfigurationError("Unknown mode")
    if cfg["pilot"] and cfg["mode"] in {"smoke", "quality"}:
        raise ConfigurationError("Pilot applies only to baseline/search/confirm")
    if type(cfg["seed"]) is not int or not positive(cfg["timeout_seconds"]) or cfg["timeout_seconds"] > 36000:
        raise ConfigurationError("Invalid seed or timeout")
    if any(type(cfg[k]) is not bool for k in ("pilot", "profile_trace")):
        raise ConfigurationError("Boolean flags required")
    if cfg["comparison"] not in {"engineering", "parameter", "quantization"}:
        raise ConfigurationError("Unknown comparison class")
    if cfg["mode"] == "baseline" and "candidate" in cfg:
        raise ConfigurationError("Baseline mode cannot contain a candidate")
    if cfg["mode"] in {"search", "confirm"} and "candidate" not in cfg:
        raise ConfigurationError("Comparison requires a complete candidate")
    if cfg.get("baseline_ref") and cfg["mode"] != "search":
        raise ConfigurationError("baseline-ref is only supported in search")
    if cfg["mode"] == "search" and not cfg.get("baseline_ref"):
        raise ConfigurationError("Search requires baseline-ref; baseline will not be rerun")
    if cfg["mode"] == "quality" and not cfg.get("quality_data"):
        raise ConfigurationError("Quality requires labeled data")
    if cfg["mode"] != "quality" and not cfg.get("data"):
        raise ConfigurationError("Speed mode requires data")
    cfg["quality_batch_sizes"] = sizes(cfg["quality_batch_sizes"])
    bounds = cfg.get("buckets", {})
    if cfg["mode"] != "quality" and not bounds:
        raise ConfigurationError("Explicit required buckets missing")
    for name, bound in bounds.items():
        if not isinstance(name, str) or set(bound) != {"min", "max"} or not positive(bound["min"]) or not positive(bound["max"]) or bound["min"] > bound["max"]:
            raise ConfigurationError("Invalid bucket bounds")
        if bound["max"] > cfg["baseline"]["preparation"]["limit"]:
            raise ConfigurationError("Bucket exceeds input contract")
    grid = cfg.get("batch_sizes_by_bucket", {b: [1] for b in bounds})
    if set(grid) != set(bounds):
        raise ConfigurationError("Batch grid must cover exactly the configured buckets")
    cfg["batch_sizes_by_bucket"] = {b: sizes(v) for b, v in grid.items()}
    if "candidate" in cfg:
        a, b = cfg["baseline"], cfg["candidate"]
        if declared_inputs(a) != declared_inputs(b) or a["scoring"] != b["scoring"]:
            raise ConfigurationError("Comparison requires identical preparation and task/scoring")
        if cfg["comparison"] == "engineering" and (a["weights"] != b["weights"] or a["execution"]["dtype"] != b["execution"]["dtype"]):
            raise ConfigurationError("Engineering must preserve weights and precision")
        if cfg["comparison"] == "quantization" and b.get("quantization", {}).get("baseline_identity") != variant_identity(a):
            raise ConfigurationError("Quantized artifact must identify its exact baseline")
    return cfg


def resolve_options(raw, *, mode=None, environment=None, baseline_ref=None, pilot=False, profile_trace=False):
    cfg = deepcopy(read(raw) if isinstance(raw, (str, Path)) else raw)
    for k, v in (("mode", mode), ("environment", environment), ("baseline_ref", baseline_ref)):
        if v is not None:
            cfg[k] = v
    if pilot:
        cfg["pilot"] = True
    if profile_trace:
        cfg["profile_trace"] = True
    return validate_config(cfg)


def verify_local_sources(spec):
    values = [spec.get(k) for k in ("weights", "tokenizer", "template")]
    values.extend(spec.get("sdk_assets", {}).values())
    for value in values:
        if not isinstance(value, dict) or "path" not in value:
            continue
        root = Path(value["path"]).resolve()
        if not root.exists():
            raise ConfigurationError("Local artifact missing")
        if "sha256" in value:
            actual = sha(root) if root.is_file() else digest({str(p.relative_to(root)): sha(p)
                for p in sorted(root.rglob("*")) if p.is_file()})
            if actual != value["sha256"]:
                raise ConfigurationError("Local artifact checksum mismatch")
        if "inventory" in value and set(value["inventory"]) != {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}:
            raise ConfigurationError("Local artifact inventory is incomplete")
        for name, expected in value.get("inventory", {}).items():
            target = (root / name).resolve()
            if not target.is_relative_to(root) or not target.is_file() or sha(target) != expected:
                raise ConfigurationError("Local artifact inventory mismatch")
