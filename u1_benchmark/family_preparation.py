"""CPU-only configuration and preparation for standard HF classifiers."""
from copy import deepcopy
import importlib.metadata
import math

from .core import digest, sha
from .messages import messages, UnsupportedInput
from .v4_preparation import Preparer, TOKENIZER_FILES, _keys, source_path, validate_source


def validate_classifier(spec):
    result = deepcopy(spec)
    _keys(result, ('adapter', 'weights', 'tokenizer', 'template', 'preparation', 'scoring',
                   'execution', 'sdk_assets'), 'classifier variant')
    for key in ('weights', 'tokenizer'):
        validate_source(result.get(key))
        if 'file' in result[key]:
            raise ValueError('Classifier sources must be directories')
    if result.get('template') is not None or result.get('sdk_assets'):
        raise ValueError('Sequence classification uses plain text, not a chat template/SDK')
    p, s, e = (result.setdefault(key, {}) for key in ('preparation', 'scoring', 'execution'))
    _keys(p, ('limit', 'text_selection', 'add_special_tokens'), 'classifier preparation')
    if type(p.get('limit')) is not int or p['limit'] < 1:
        raise ValueError('Positive classifier input limit required')
    p.setdefault('text_selection', 'assistant_else_user')
    p.setdefault('add_special_tokens', True)
    if p['text_selection'] not in ('assistant_else_user', 'user', 'assistant'):
        raise ValueError('Unsupported text selection')
    if type(p['add_special_tokens']) is not bool:
        raise ValueError('add_special_tokens must be boolean')
    _keys(s, ('method', 'labels', 'threshold', 'positive_label'), 'classifier scoring')
    s.setdefault('method', 'softmax_argmax')
    if s['method'] not in ('softmax_argmax', 'softmax_threshold'):
        raise ValueError('Unsupported classifier scoring')
    labels = s.get('labels')
    if labels is not None and (not isinstance(labels, list) or len(labels) < 2 or
            any(not isinstance(v, str) or not v for v in labels) or len(set(labels)) != len(labels)):
        raise ValueError('Expected labels must be distinct nonempty strings')
    if s['method'] == 'softmax_threshold':
        t = s.get('threshold')
        if (labels is None or len(labels) != 2 or s.get('positive_label') not in labels or
                type(t) not in (float, int) or not math.isfinite(t) or not 0 <= t <= 1):
            raise ValueError('Threshold requires two labels, positive_label and a finite threshold')
    elif {'threshold', 'positive_label'} & s.keys():
        raise ValueError('Threshold options require softmax_threshold')
    _keys(e, ('runtime', 'dtype', 'attention', 'sdpa_backend', 'linear_attention',
              'compile_backend', 'compile_mode', 'dynamic', 'fullgraph', 'opset', 'cpu_policy'), 'classifier execution')
    for key, value in {'runtime': 'eager', 'dtype': 'float32', 'attention': 'native',
                       'sdpa_backend': None, 'linear_attention': 'reference'}.items():
        e.setdefault(key, value)
    if e['runtime'] not in ('eager', 'compile', 'ort') or e['dtype'] not in ('float16', 'float32', 'bfloat16'):
        raise ValueError('Unsupported classifier runtime/precision')
    if e['attention'] not in ('native', 'eager', 'sdpa') or e['linear_attention'] != 'reference':
        raise ValueError('Unsupported classifier attention/linear implementation')
    if e['sdpa_backend'] not in (None, 'math', 'efficient') or (e['sdpa_backend'] and e['attention'] != 'sdpa'):
        raise ValueError('Explicit SDPA backend requires attention=sdpa')
    compile_options = {'compile_backend': 'inductor', 'compile_mode': 'default', 'dynamic': True, 'fullgraph': True}
    ort_options = {'opset': 17, 'cpu_policy': 'metadata-cpu-v1'}
    for runtime, options in (('compile', compile_options), ('ort', ort_options)):
        if e['runtime'] != runtime and options.keys() & e.keys():
            raise ValueError('Backend options do not match classifier runtime')
        if e['runtime'] == runtime:
            for key, value in options.items():
                e.setdefault(key, value)
                if e[key] != value or type(e[key]) is not type(value):
                    raise ValueError('Unsupported classifier backend option: ' + key)
    if e['runtime'] == 'ort' and e['dtype'] != 'float32':
        raise ValueError('Classifier ORT requires FP32')
    return result


class ClassificationPreparer(Preparer):
    def __init__(self, spec):
        from transformers import AutoTokenizer
        self.spec = validate_classifier(spec)
        self.name = self.spec['adapter']
        self.settings = self.spec['preparation']
        self.limit = self.settings['limit']
        self.snapshot = source_path(self.spec['tokenizer'], TOKENIZER_FILES)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.snapshot), trust_remote_code=False)
        self.tokenizer.model_max_length = 10**9
        self.input_changed = False
        self.assets = {'preparation_source': sha(__file__)}
        backend = getattr(self.tokenizer, 'backend_tokenizer', None)
        if backend is not None:
            self.assets['effective_tokenizer'] = digest(backend.to_str())
        self.metadata = {'adapter': self.name, 'limit': self.limit, 'assets': self.assets,
                         'text_selection': self.settings['text_selection'],
                         'dependencies': {name: importlib.metadata.version(name)
                                          for name in ('transformers', 'tokenizers')}}

    def choice(self, value):
        source = messages(value)
        selection = self.settings['text_selection']
        if selection == 'assistant_else_user':
            target = len(source) - 1
        else:
            candidates = [i for i, item in enumerate(source) if item['role'] == selection]
            if not candidates:
                raise UnsupportedInput('Requested evaluation role is absent')
            target = candidates[-1]
        return {'source_mode': '+'.join(m['role'] for m in source) if len(source) == 2
                else source[0]['role'] + '-only', 'target_index': target,
                'target_role': source[target]['role'], 'context_indices': [],
                'effective_messages': [source[target]], 'source_sha256': digest(source)}

    def render(self, value):
        return self.choice(value)['effective_messages'][0]['content']

    def prepare(self, value):
        from speed_dataset.preparation import normalized
        # Single-text classification has no token-type segments; HF defaults them to zero.
        return normalized(self.tokenizer(self.render(value), truncation=False, padding=False,
            add_special_tokens=self.settings['add_special_tokens'], return_attention_mask=True,
            return_token_type_ids=False))

    def parity(self):
        from speed_dataset.preparation import normalized, CHECK_TEXTS
        checks = []
        for text in CHECK_TEXTS:
            role = 'assistant' if self.settings['text_selection'] == 'assistant' else 'user'
            value = [{'role': role, 'content': text}]
            expected = normalized(self.tokenizer([self.render(value)], truncation=False, padding=False,
                add_special_tokens=self.settings['add_special_tokens'], return_attention_mask=True,
                return_token_type_ids=False, return_tensors='pt'))
            if self.prepare(value) != expected:
                raise ValueError('HF single/batch preparation differs')
            checks.append({'source_sha256': digest(value), 'input_sha256': digest(expected), 'status': 'passed'})
        return {'status': 'passed', 'scope': 'CPU native preparation only; no weights or GPU', 'checks': checks}
