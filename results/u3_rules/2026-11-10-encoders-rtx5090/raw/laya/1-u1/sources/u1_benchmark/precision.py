"""Observed execution precision; no device-based dtype selection or GPU imports."""
from copy import deepcopy


PARAMETER_DTYPES = {
    "bool", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64",
    "float16", "bfloat16", "float32", "float64", "complex32", "complex64", "complex128",
    "qint8", "quint8", "qint32", "quint4x2", "quint2x4", "float4_e2m1fn_x2",
    "float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz", "float8_e8m0fnu",
}


class PrecisionError(ValueError):
    """A missing or changed execution mode invalidates the whole worker."""


def validate_precision(value):
    fields = {"parameter_dtypes", "autocast_enabled", "autocast_dtype", "policy"}
    if not isinstance(value, dict) or set(value) != fields:
        raise PrecisionError("Execution precision passport is required")
    dtypes = value["parameter_dtypes"]
    if (not isinstance(dtypes, list) or not dtypes
            or any(not isinstance(t, str) or t not in PARAMETER_DTYPES for t in dtypes)
            or dtypes != sorted(set(dtypes))):
        raise PrecisionError("Invalid parameter dtypes in precision passport")
    enabled, dtype = value["autocast_enabled"], value["autocast_dtype"]
    if type(enabled) is not bool or (enabled and dtype not in ("float16", "bfloat16")) or (not enabled and dtype is not None):
        raise PrecisionError("Invalid autocast in precision passport")
    if value["policy"] not in ("native_sdk", "explicit"):
        raise PrecisionError("Invalid precision policy")
    return deepcopy(value)


def physical_precision(value):
    return {k: v for k, v in value.items() if k != "policy"}


class PrecisionGuard:
    def __init__(self, adapter, required=False):
        self.adapter = adapter
        value = adapter.metadata.get("precision")
        self.expected = validate_precision(value) if required or value is not None else None

    def check(self):
        if self.adapter.metadata.get("precision") != self.expected:
            raise PrecisionError("Execution precision changed during worker")

    def call(self, function, *args, **kwargs):
        self.check()
        try:
            return function(*args, **kwargs)
        finally:
            self.check()


class PrecisionTracker:
    """Check per-variant workers and engineering pairs, including archived runs."""
    def __init__(self, comparison):
        self.comparison = comparison
        self.variants = {}

    def add(self, variant, result):
        value = result.get("metadata", {}).get("precision")
        version = result.get("precision_policy_version")
        if version not in (None, 1):
            raise PrecisionError("Unknown execution precision policy version")
        value = validate_precision(value) if version == 1 or value is not None else None
        if variant in self.variants and self.variants[variant] != value:
            raise PrecisionError("Execution precision differs between workers of one variant")
        if any((other is None) != (value is None) for other in self.variants.values()):
            raise PrecisionError("Cannot mix recorded and unrecorded execution precision")
        if self.comparison == "engineering" and value is not None:
            if any(physical_precision(other) != physical_precision(value) for other in self.variants.values()):
                raise PrecisionError("Engineering baseline/candidate precision differs")
        self.variants[variant] = value

    def summary(self):
        if not self.variants or all(v is None for v in self.variants.values()):
            return {"precision_status": "historical_unrecorded"}
        baseline, candidate = (self.variants.get(k) for k in ("baseline", "candidate"))
        return {"precision_status": "recorded_consistent", "precision": deepcopy(self.variants),
                "precision_changed": (physical_precision(baseline) != physical_precision(candidate)
                                      if baseline is not None and candidate is not None else None)}
