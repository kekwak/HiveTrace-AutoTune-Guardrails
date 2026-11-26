"""Optional local extension; standard classifiers should use the built-in family.

Pin this module's SHA256 and its dependencies in adapter. Keep both complete
variant specs and native scoring settings. This example is not quantization.
Subclass FamilyAdapter for a genuinely different runtime; expose actual input
tensors and all parameter/readout tensors, not just a list of verdicts.
"""
from copy import deepcopy

from u1_benchmark.core import digest
from u1_benchmark.family_preparation import ClassificationPreparer
from u1_benchmark.classifier_family import ClassificationAdapter


def native_spec(spec):
    value = deepcopy(spec)
    value['adapter'] = 'hf_sequence_classification'
    return value


def create_preparer(spec):
    return ClassificationPreparer(native_spec(spec))


def create(spec, directory):
    adapter = ClassificationAdapter(native_spec(spec), directory)
    adapter.metadata['variant_sha256'] = digest(spec)
    return adapter
