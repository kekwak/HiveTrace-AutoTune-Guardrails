"""Versioned adapter contracts; importing this module never loads a model."""
import math
from pathlib import Path

from .core import ROOT, digest, read, sha

MODEL_SPECS = read(ROOT / "speed_dataset/contracts.json")["models"]
BUCKETS = read(ROOT / "speed_dataset/contracts.json")["buckets"]
DTYPES = {m: "float16" if m in ("halo", "hive", "open") else "float32" for m in MODEL_SPECS}
RULES = {
    "halo": "author_prompt; thinking_off; safe/unsafe softmax; unsafe>=0.65; reference linear/conv/norm; eager full attention",
    "gliner": "extract safety [safe,unsafe]; argmax; tie=safe",
    "koala": "native nine-class argmax; tie=lowest class index",
    "llama": "native benign/malicious argmax; tie=lowest class index; selected hipocap mirror",
    "laya": "SDK predict; danger choice A/B; FP32 weights; per-request FP16 autocast; efficient SDPA",
    "hive": "native template; greedy generate max_new_tokens=1; native EOS; SDPA math; parse safe/unsafe",
    "open": "pinned 2510 template; greedy generate max_new_tokens=10; native EOS; SDPA math; safe/unsafe and category",
    "shield": "Dangerous Content; eager softcap; Yes/No softmax; tie=No",
}


def model_contract(model, version=2):
    spec = MODEL_SPECS[model]
    result = {"model": spec["repo"], "revision": spec["revision"],
            "tokenizer": {"repo": spec["repo"], "revision": spec["revision"],
                          "subfolder": spec.get("subfolder", "")},
            "preparation_sha256": sha(ROOT / "speed_dataset/preparation.py"),
            "preparation_spec": spec, "rules": RULES[model], "limit": spec["limit"]}
    if version == 3:
        result["input_version"] = 3
        result["preparation_sha256"] = digest({name: sha(ROOT / "u1_benchmark" / name)
            for name in ("messages.py", "native_preparation.py")})
        result["preparation_spec"] = {k: v for k, v in spec.items()
                                      if k not in ("template_repo", "template_revision")}
        result["role_policy"] = {"formats": ["user-only", "assistant-only", "user+assistant"],
                                 "unsupported": "explicit; never relabel or synthesize context"}
        if model == "open":
            result["rules"] = RULES[model].replace("pinned 2510 template", "0124 weights; pinned 2510 task template; user-only")
            result["preparation_sha256"] = digest({"native": result["preparation_sha256"],
                "task_override": sha(ROOT / "u1_benchmark/role_preparation.py")})
            result["preparation_spec"] = spec
            result["preparation_profile"] = "open-0124-task-2510-v1"
        if model == "shield":
            result["rules"] += "; native prompt/response Dangerous Content wording"
    elif version != 2:
        raise ValueError("Unknown input contract version")
    return result


def required_buckets(model):
    return [key for key, spec in BUCKETS.items() if model in spec["models"]]


def file_inventory(path):
    path = Path(path).resolve()
    paths = [path] if path.is_file() else sorted(path.rglob("*"))
    if not paths:
        raise ValueError("Artifact is empty or missing")
    result = {}
    for item in paths:
        if item.is_symlink():
            raise ValueError("Artifact symlinks are not accepted")
        if item.is_file():
            result[item.name if path.is_file() else str(item.relative_to(path))] = sha(item)
    if not result:
        raise ValueError("Artifact contains no files")
    return result


def adapter_spec(value, model, kind, candidate=False):
    spec = dict(value)
    if set(spec) - {"backend", "module", "source_sha256", "dependencies", "artifact", "options"}:
        raise ValueError("Unknown adapter fields")
    if "module" in spec:
        path = Path(spec["module"]).resolve()
        if path.suffix != ".py" or sha(path) != spec.get("source_sha256"):
            raise ValueError("External adapter source hash mismatch")
        spec["module"] = str(path)
        if not isinstance(spec.get("dependencies"), dict):
            raise ValueError("External adapter requires exact dependency versions (possibly {})")
    else:
        allowed = {"eager"} | ({"fla"} if model == "halo" else {"compile", "ort"} if model == "gliner" else set())
        if spec.get("backend", "eager") not in allowed or (not candidate and spec.get("backend", "eager") != "eager"):
            raise ValueError("Unsupported backend/baseline")
    if kind == "quantization" and candidate:
        if "module" not in spec:
            raise ValueError("Quantized artifact needs an explicit external adapter")
        a = spec.get("artifact", {})
        if set(a) != {"path", "sha256", "base_checkpoint", "base_revision", "method", "bits", "runtime"}:
            raise ValueError("Missing quantized artifact provenance")
        if (a["base_checkpoint"], a["base_revision"]) != (MODEL_SPECS[model]["repo"], MODEL_SPECS[model]["revision"]):
            raise ValueError("Quantized base checkpoint mismatch")
        if not a["method"] or not a["runtime"] or type(a["bits"]) is not int or a["bits"] <= 0:
            raise ValueError("Invalid quantization metadata")
        if digest(file_inventory(a["path"])) != a["sha256"]:
            raise ValueError("Quantized artifact hash mismatch")
        spec["artifact"] = {**a, "path": str(Path(a["path"]).resolve())}
    return spec


def parse_config(value):
    value = read(value) if isinstance(value, (str, Path)) else dict(value)
    allowed = {"schema_version", "model", "baseline", "candidate", "data", "quality_data", "label_mapping",
               "comparison", "environment", "seed", "mode", "timeout_seconds"}
    if set(value) - allowed or value.get("schema_version") not in (2, 3) or value.get("model") not in MODEL_SPECS:
        raise ValueError("Invalid v2 configuration")
    cfg = {"comparison": "engineering", "mode": "screening", "seed": 17, "timeout_seconds": 36000, **value}
    if cfg["comparison"] not in ("engineering", "parameter", "quantization") or cfg["mode"] not in ("smoke", "pilot", "screening", "quality"):
        raise ValueError("Unknown mode/comparison")
    if cfg["seed"] not in (17, 42, 123) or not 0 < cfg["timeout_seconds"] <= 36000:
        raise ValueError("Invalid seed/deadline")
    if cfg["mode"] == "smoke":
        cfg["timeout_seconds"] = min(cfg["timeout_seconds"], 3600)
    cfg["environment"] = str(Path(cfg["environment"]).resolve())
    cfg["baseline"] = adapter_spec(cfg.get("baseline", {"backend": "eager"}), cfg["model"], cfg["comparison"])
    if cfg.get("candidate") is not None:
        cfg["candidate"] = adapter_spec(cfg["candidate"], cfg["model"], cfg["comparison"], True)
    for key in ("data", "quality_data"):
        if key not in cfg or (key == "data" and cfg["mode"] == "quality"):
            continue
        d = cfg[key]
        if d == {"profile": "published"} and key == "data":
            continue
        if set(d) != {"path", "manifest_sha256"}:
            raise ValueError("Dataset requires path and pinned manifest_sha256")
        cfg[key] = {**d, "path": str(Path(d["path"]).resolve())}
        if sha(Path(d["path"]) / "manifest.json") != d["manifest_sha256"]:
            raise ValueError("Dataset manifest hash mismatch")
    if cfg["mode"] == "quality" and "quality_data" not in cfg:
        raise ValueError("quality mode requires a separate labeled dataset")
    if cfg["mode"] != "quality" and "data" not in cfg:
        raise ValueError("Speed data required")
    return cfg


def validate_response(value):
    """Keep native labels and optional scores; never invent probabilities."""
    if not isinstance(value, dict) or not isinstance(value.get("label"), str) or not value["label"]:
        raise ValueError("A nonempty native verdict is required")
    if set(value) - {"label", "scores", "score_type", "category", "generated_tokens", "p_safe", "p_unsafe", "logit_margin", "public_confidence"}:
        raise ValueError("Unexpected response fields; raw generated text/input is not an output field")
    if value.get("category") is not None and not isinstance(value["category"], str):
        raise ValueError("Category must be a native string or null")
    for field in ("p_safe", "p_unsafe", "logit_margin", "public_confidence"):
        if field in value and (isinstance(value[field], bool) or not isinstance(value[field], (int, float)) or not math.isfinite(value[field])):
            raise ValueError("Invalid numeric diagnostic score")
    scores = value.get("scores")
    if scores is not None:
        if not isinstance(scores, dict) or not scores or value.get("score_type") not in (
                "probabilities", "logits", "independent_probabilities", "raw"):
            raise ValueError("Named scores require an explicit score_type")
        for name, number in scores.items():
            if not isinstance(name, str) or isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                raise ValueError("Invalid/nonfinite named score")
        if "probabilities" in value["score_type"] and any(not 0 <= x <= 1 for x in scores.values()):
            raise ValueError("Invalid probability")
        if value["score_type"] == "probabilities" and abs(sum(scores.values()) - 1) > 1e-5:
            raise ValueError("Probabilities must sum to one")
    if "generated_tokens" in value and (type(value["generated_tokens"]) is not int or value["generated_tokens"] < 0):
        raise ValueError("Invalid generated token count")
    return value
