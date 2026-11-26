"""Declared implementation paths, never a claim of measured CUDA support."""


def describe(spec):
    adapter, execution = spec['adapter'], spec['execution']
    if isinstance(adapter, dict):
        return {'status': 'runtime_check_required', 'implementation': 'external',
                'requested': {k: execution[k] for k in ('runtime', 'dtype')},
                'reason': 'External code is not executed by plan'}
    if adapter == 'hf_sequence_classification':
        paths = {'eager': 'AutoModelForSequenceClassification.forward -> logits',
                 'compile': 'classifier tensor-forward -> logits',
                 'ort': 'classifier tensor-forward -> logits; CUDA IOBinding; metadata-cpu-v1'}
        constraints = ['Native input_ids/attention_mask signature', 'Configured labels and positional capacity',
                       'Requested attention must be implemented by the loaded architecture']
    elif adapter == 'd3_decision':
        paths = {'eager': 'native d3 probabilities/readout', 'compile': 'd3 text tensor-forward'}
        constraints = ['Pinned text-only d3-code-readout-v1', 'BF16 backbone and FP32 readout',
                       'Explicit linear dispatch requires Qwen3.5 GatedDeltaNet', 'ORT is unsupported']
    else:
        paths = {'eager': 'existing native adapter'}
        if adapter == 'gliner':
            paths.update(compile='encoder tensor-forward', ort='encoder CUDA IOBinding; metadata-cpu-v1')
        constraints = ['Existing adapter contract; backend proof required']
    runtime = execution['runtime']
    return {'status': 'runtime_check_required' if runtime in paths else 'unsupported',
            'implementation': adapter, 'runtime_paths': paths, 'requested': dict(execution),
            'target': paths.get(runtime), 'constraints': constraints, 'gpu_verified': False}


def variants(config):
    return {name: describe(config[name]) for name in ('baseline', 'candidate') if name in config}
