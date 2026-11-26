"""Minimal explicit extension: delegate to the native GLiNER implementation.

This is an interface example, not quantization. A custom runtime replaces the
constructor and implements the same methods; it must return native batch results
and prove its actual backend instead of looping over single requests.
"""
from copy import deepcopy

from u1_benchmark.core import digest
from u1_benchmark.v4_preparation import create_preparer as native_preparer
from u1_benchmark.v4_adapters import NativeAdapter


def native_spec(spec):
    result = deepcopy(spec)
    result["adapter"] = "gliner"
    return result


def create_preparer(spec):
    return native_preparer(native_spec(spec))


def create(spec, directory):
    adapter = NativeAdapter(native_spec(spec), directory)
    adapter.metadata["variant_sha256"] = digest(spec)
    return adapter
