"""Динамическое квантование nn.Linear любой модели.

none              — без квантования (эталон)
torchao_int8_dyn  — torchao W8A8: веса int8, активации квантуются в int8 на лету (INT8 GEMM)
torchao_fp8_dyn   — torchao: веса и активации в FP8 e4m3 с масштабом по строкам, активации на лету; нужен sm_89+
bnb_int8          — bitsandbytes LLM.int8(): активации на лету по векторам, выбросы считаются в fp16
"""

import torch
from torch import nn

METHODS = ("none", "torchao_int8_dyn", "torchao_fp8_dyn", "bnb_int8")
# Классификационные головы и lm_head оставляем в исходной точности: от них напрямую зависят логиты вердикта.
DEFAULT_SKIP = ("lm_head", "classifier", "score", "pooler")
MIN_DIM, ALIGN = 64, 16


class UnsupportedQuant(RuntimeError):
    pass


def _eligible(name, module, skip):
    if not isinstance(module, nn.Linear):
        return False
    if any(part in name.split(".") or name.endswith(part) for part in skip):
        return False
    i, o = module.in_features, module.out_features
    return min(i, o) >= MIN_DIM and i % ALIGN == 0 and o % ALIGN == 0


def _linears(model, skip):
    eligible, skipped = [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            (eligible if _eligible(name, module, skip) else skipped).append(name)
    return eligible, skipped


def check_support(method, device):
    """Причина неподдержки или None. Проверяется только наличие библиотеки и архитектура GPU."""
    if method == "none":
        return None
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return "динамическое квантование в guard_eval поддерживается только на CUDA"
    major, minor = torch.cuda.get_device_capability(torch.device(device))
    if method.startswith("torchao"):
        try:
            import torchao  # noqa: F401
        except ImportError:
            return "не установлен torchao"
        if method == "torchao_fp8_dyn" and (major, minor) < (8, 9):
            return f"FP8 требует sm_89+, у GPU sm_{major}{minor}"
    if method == "bnb_int8":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            return "не установлен bitsandbytes"
    if method not in METHODS:
        return f"неизвестный метод {method}; доступны {METHODS}"
    return None


def quantize(model, method, device, skip=DEFAULT_SKIP):
    """Квантует модель на месте. Модель для bnb_int8 должна быть ещё на CPU — перенос на GPU здесь."""
    reason = check_support(method, device)
    if reason:
        raise UnsupportedQuant(reason)
    eligible, skipped = _linears(model, skip)
    info = {"method": method, "quantized_linear": 0 if method == "none" else len(eligible),
            "eligible_linear": len(eligible), "kept_linear": len(skipped), "kept_examples": skipped[:10]}
    if method == "none":
        model.to(device)
        info["weight_bytes"] = weight_bytes(model)
        return info
    if not eligible:
        raise UnsupportedQuant("нет подходящих nn.Linear для квантования")
    names = set(eligible)

    if method.startswith("torchao"):
        from torchao.quantization import quantize_
        if method == "torchao_int8_dyn":
            from torchao.quantization import Int8DynamicActivationInt8WeightConfig
            config = Int8DynamicActivationInt8WeightConfig()
        else:
            from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow
            config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
        model.to(device)
        quantize_(model, config, filter_fn=lambda module, fqn: fqn in names)
    else:
        import bitsandbytes as bnb
        for name in eligible:
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            old = getattr(parent, child)
            new = bnb.nn.Linear8bitLt(old.in_features, old.out_features, bias=old.bias is not None,
                                      has_fp16_weights=False, threshold=6.0)
            new.weight = bnb.nn.Int8Params(old.weight.data.cpu(), requires_grad=False, has_fp16_weights=False)
            if old.bias is not None:
                new.bias = nn.Parameter(old.bias.data.cpu(), requires_grad=False)
            setattr(parent, child, new)
        model.to(device)   # Int8Params квантуются при переносе на GPU
    info["weight_bytes"] = weight_bytes(model)
    return info


def weight_bytes(model):
    seen, total = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) in seen:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    return total


def smoke(device, dtype=torch.bfloat16):
    """Проверка всех методов на маленькой сети: работает ли и насколько отличается от эталона."""
    torch.manual_seed(0)
    x = torch.randn(32, 512, dtype=dtype)
    reference = nn.Sequential(nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, 512)).to(dtype)
    with torch.inference_mode():
        expected = reference.to(device)(x.to(device)).float()
    report = {}
    for method in METHODS[1:]:
        model = nn.Sequential(nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, 512)).to(dtype)
        model.load_state_dict(reference.state_dict())
        model.cpu()
        try:
            quantize(model, method, device, skip=())
            with torch.inference_mode():
                got = model(x.to(device)).float()
            rel = ((got - expected).norm() / expected.norm()).item()
            report[method] = {"ok": True, "relative_error": round(rel, 5)}
        except Exception as e:  # фиксируем как результат, а не роняем проверку
            report[method] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
    return report
