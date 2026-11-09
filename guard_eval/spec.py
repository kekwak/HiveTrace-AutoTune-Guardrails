"""Конфигурация модели (spec) и её ключ кэша."""

import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "specs" / "models"

# Не влияют на скоры: порог применяется при подсчёте метрик, имя — только подпись.
NOT_IN_KEY = {"name", "threshold"}
# Меняется при изменении логики адаптеров или квантования — старые прогоны перестают совпадать.
CODE_VERSION = "guard_eval-2"


def load_spec(model, overrides=()):
    """`model` — псевдоним из specs/models (koala) или путь к YAML; overrides — строки key=value."""
    path = Path(model)
    if not path.suffix:
        path = MODELS_DIR / f"{model}.yaml"
    if not path.is_file():
        known = ", ".join(sorted(p.stem for p in MODELS_DIR.glob("*.yaml")))
        raise FileNotFoundError(f"Нет spec {path}. Известные модели: {known}")
    spec = yaml.safe_load(path.read_text())
    for item in overrides:
        key, _, value = item.partition("=")
        target = spec
        *parents, leaf = key.split(".")
        for p in parents:
            target = target.setdefault(p, {})
        target[leaf] = yaml.safe_load(value)
    return spec


def with_quant(spec, quant):
    spec = copy.deepcopy(spec)
    if quant is not None:
        spec["quant"] = quant
    return spec


def resolve_dtype(spec, device):
    """dtype: auto -> bfloat16 на GPU с аппаратным bf16 (sm_80+), float16 на старых GPU (T4), float32 на CPU.
    Явный bfloat16 на GPU без аппаратной поддержки — ошибка, а не медленная эмуляция."""
    import torch
    spec = copy.deepcopy(spec)
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    capability = torch.cuda.get_device_capability(torch.device(device)) if cuda else None
    if spec.get("dtype", "auto") == "auto":
        spec["dtype"] = ("bfloat16" if capability >= (8, 0) else "float16") if cuda else "float32"
    elif spec["dtype"] == "bfloat16" and cuda and capability < (8, 0):
        raise ValueError(f"GPU sm_{capability[0]}{capability[1]} не поддерживает bf16 аппаратно: "
                         "используйте dtype=auto или dtype=float16")
    return spec


def spec_key(spec, environment):
    """Всё, что меняет скоры: spec без порога, версия кода, класс GPU и версии библиотек."""
    payload = {
        "spec": {k: v for k, v in spec.items() if k not in NOT_IN_KEY},
        "code": CODE_VERSION,
        "device": environment["device_class"],
        "packages": {k: environment["packages"].get(k) for k in ("torch", "transformers", "torchao", "bitsandbytes", "gliner2")},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def run_label(spec):
    return f"{spec['name']}__{spec.get('quant', 'none')}__{spec['dtype']}"
