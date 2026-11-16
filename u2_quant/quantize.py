"""Which nn.Linear layers are quantized and how; heads outside the given module stay untouched."""
from torch import nn

METHODS = ("none", "fp8")
# Нижняя граница amax активации при расчёте per-row масштаба: строки паддинга у gliner целиком нулевые,
# без неё масштаб = 0 и FP8 даёт 0/0 = NaN (замечено на RTX 5090 при батче > 1). На ненулевые строки не влияет.
ACTIVATION_VALUE_LB = 1e-12


def linear_targets(module, skip):
    names = [name for name, child in module.named_modules() if isinstance(child, nn.Linear)]
    unknown = set(skip) - set(names)
    if unknown:
        raise ValueError(f"skip_modules are not Linear layers of this module: {sorted(unknown)}")
    return [name for name in names if name not in skip]


def apply(module, method, skip):
    if method not in METHODS:
        raise ValueError(f"Unknown quantization method: {method}")
    targets = linear_targets(module, skip)
    if method == "none":
        return {"method": "none", "granularity": None, "quantized": [], "skipped": list(skip), "torchao": None}
    import torchao
    from torchao import quantization
    chosen = set(targets)
    config = quantization.Float8DynamicActivationFloat8WeightConfig(granularity=quantization.PerRow(),
                                                                    activation_value_lb=ACTIVATION_VALUE_LB)
    quantization.quantize_(module, config, filter_fn=lambda child, name: name in chosen and isinstance(child, nn.Linear))
    return {"method": method, "granularity": "per_row", "quantized": targets, "skipped": list(skip),
            "torchao": torchao.__version__, "activation_value_lb": ACTIVATION_VALUE_LB}
