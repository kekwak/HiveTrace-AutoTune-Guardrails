"""Compare actual pinned weights and native inputs on CPU; no CUDA acceptance."""
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import traceback

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from u1_benchmark.classifier_family import ClassificationAdapter
from u1_benchmark.core import ROOT
from u1_benchmark.family_adapter import FamilyAdapter
from u1_benchmark.family_preparation import ClassificationPreparer, validate_classifier
from u1_benchmark.v4_models import hf_load_options


def weights(model):
    return {name: {'shape': list(t.shape), 'dtype': str(t.dtype),
                   'sha256': hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()}
            for name, t in model.state_dict().items()}


result = {'diagnostic_only': True, 'device': 'cpu', 'gpu_checked': False,
          'versions': {name: importlib.metadata.version(name)
                       for name in ('torch', 'transformers', 'tokenizers')}}
try:
    torch.set_num_threads(2)
    spec = validate_classifier(json.loads((ROOT / 'configs/v4/protectai.json').read_text())['baseline'])
    texts = ['Library hours?', 'The public library is open.', '']
    prep = ClassificationPreparer(spec)
    path = prep.snapshot
    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(path), dtype=torch.float32, local_files_only=True).eval()
    expected_weights = weights(model)
    expected_inputs = tok(texts, padding=True, truncation=False, return_tensors='pt')
    with torch.inference_mode():
        expected_logits = model(**expected_inputs).logits.clone()
    del model
    gc.collect()
    model, loading = AutoModelForSequenceClassification.from_pretrained(
        str(path), local_files_only=True, **hf_load_options(spec['execution']))
    result['loading_info'] = {key: sorted(loading.get(key, [])) for key in
        ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')}
    result['weight_tensors_identical'] = weights(model) == expected_weights
    result['weight_tensor_count'] = len(expected_weights)
    a = ClassificationAdapter.__new__(ClassificationAdapter)
    FamilyAdapter.__init__(a, spec, Path(__file__).parent, prep, model.eval(), device='cpu')
    a.labels = ['SAFE', 'INJECTION']
    a.check_device()
    actual_inputs = a._hf_inputs(texts)
    result['input_ids_identical'] = torch.equal(actual_inputs['input_ids'], expected_inputs['input_ids'])
    result['masks_identical'] = torch.equal(actual_inputs['attention_mask'], expected_inputs['attention_mask'])
    with torch.inference_mode():
        actual_logits = model(**actual_inputs).logits
    result['max_logit_abs_difference'] = (actual_logits - expected_logits).abs().max().item()
    result['native_logits'] = expected_logits.tolist()
    result['adapter_logits'] = actual_logits.tolist()
    result['responses'] = a.request_batch(texts)
    result['status'] = 'passed' if (result['weight_tensors_identical'] and result['input_ids_identical']
        and result['masks_identical'] and torch.equal(actual_logits, expected_logits)
        and not any(result['loading_info'].values())) else 'failed'
    a.close()
except BaseException as error:
    result.update(status='failed', error_type=type(error).__name__, traceback=traceback.format_exc())
with Path(__file__).with_suffix('.r2.json').open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result, indent=2))
