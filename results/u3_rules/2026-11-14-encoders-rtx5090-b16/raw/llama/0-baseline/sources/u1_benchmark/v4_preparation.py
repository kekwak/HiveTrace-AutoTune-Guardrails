"""Config-only native preparation. Importing/constructing this never loads weights."""
from copy import deepcopy
from glob import has_magic
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
import sys

from speed_dataset.preparation import GUIDELINE, QUESTIONS, author_prompt, normalized
from .core import digest, sha
from .messages import UnsupportedInput, messages, select


NATIVE_KEYS = frozenset(('halo', 'gliner', 'koala', 'llama', 'laya', 'hive', 'open', 'shield'))
RESPONSE_GUIDELINE = GUIDELINE.replace('The prompt shall not contain or seek generation of content',
                                    'The chatbot shall not generate content')
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                   'tokenizer.model', 'spm.model', 'vocab.json', 'vocab.txt', 'merges.txt',
                   'added_tokens.json', 'chat_template.jinja', 'chat_templates/*.jinja', 'config.json')


def _keys(value, allowed, where):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError('Unsupported ' + where + ' settings')


def validate_source(source):
    if not isinstance(source, dict):
        raise ValueError('Source must be a dictionary')
    if 'path' in source:
        _keys(source, ('path', 'sha256', 'inventory', 'file'), 'local source')
        if not isinstance(source['path'], str) or not source['path']:
            raise ValueError('Local source requires path')
    else:
        _keys(source, ('repo', 'revision', 'subfolder', 'files', 'file'), 'Hub source')
        if any(not isinstance(source.get(k), str) or not source[k] for k in ('repo', 'revision')):
            raise ValueError('Hub source requires explicit repo and revision')
        part = source.get('subfolder', '')
        if not isinstance(part, str) or Path(part).is_absolute() or '..' in Path(part).parts:
            raise ValueError('Invalid source subfolder')
    if 'sha256' in source:
        value = source['sha256']
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Invalid source sha256')
    inventory, files = source.get('inventory', {}), source.get('files', [])
    if not isinstance(inventory, dict) or not isinstance(files, list):
        raise ValueError('Source inventory/files must be a dictionary/list')
    if any(not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
           for v in inventory.values()):
        raise ValueError('Invalid source inventory hash')
    for name in list(inventory) + files + ([source['file']] if 'file' in source else []):
        if not isinstance(name, str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Invalid source file name')


def source_path(source, patterns=None):
    """Resolve exactly the configured source; directory digests hash file inventories."""
    validate_source(source)
    if 'path' in source:
        path = Path(source['path']).expanduser().resolve()
        if not path.exists():
            raise ValueError('Local source does not exist')
        if 'sha256' in source:
            actual = sha(path) if path.is_file() else digest({str(p.relative_to(path)): sha(p)
                for p in sorted(path.rglob('*')) if p.is_file()})
            if actual != source['sha256']:
                raise ValueError('Local source hash mismatch')
        for name, expected in source.get('inventory', {}).items():
            target = (path / name).resolve()
            if not target.is_relative_to(path) or not target.is_file() or sha(target) != expected:
                raise ValueError('Local source inventory mismatch')
        return path
    from huggingface_hub import hf_hub_download, snapshot_download
    prefix = source.get('subfolder', '')
    if patterns is None:
        patterns = [source['file']] if 'file' in source else source.get('files')
    allow = None if patterns is None else [(prefix + '/' if prefix else '') + p for p in patterns]
    if allow is not None and len(allow) == 1 and not has_magic(allow[0]):
        # Exact pinned files can be cached even when the repository tree is unavailable.
        path = Path(hf_hub_download(source['repo'], filename=allow[0], revision=source['revision']))
        return path.parents[len(Path(patterns[0]).parts) - 1]
    root = Path(snapshot_download(source['repo'], revision=source['revision'], allow_patterns=allow))
    return root / prefix


def external_module(spec):
    entry = spec['adapter']
    _keys(entry, ('module', 'source_sha256', 'dependencies'), 'external adapter')
    path = Path(entry['module']).resolve()
    if path.suffix != '.py' or sha(path) != entry.get('source_sha256'):
        raise ValueError('External adapter source hash mismatch')
    if not isinstance(entry.get('dependencies'), dict):
        raise ValueError('External adapter requires explicit dependencies')
    for package, wanted in entry['dependencies'].items():
        if importlib.metadata.version(package) != wanted:
            raise ValueError('External adapter dependency mismatch: ' + package)
    name = 'u1_v4_external_' + entry['source_sha256'] + '_' + digest(str(path))[:12]
    if name not in sys.modules:
        definition = importlib.util.spec_from_file_location(name, path)
        if definition is None or definition.loader is None:
            raise ValueError('Cannot load external adapter')
        module = importlib.util.module_from_spec(definition)
        # dataclasses resolves annotations through sys.modules during execution.
        sys.modules[name] = module
        try:
            definition.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    module = sys.modules[name]
    for method in ('create_preparer', 'create'):
        if not callable(getattr(module, method, None)):
            raise ValueError('External module missing ' + method)
    return module


def validate_settings(spec):
    """Return a normalized independent spec, rejecting ignored native overrides."""
    result = deepcopy(spec)
    if isinstance(result.get('adapter'), dict):
        return result  # External modules own their native option namespace.
    _keys(result, ('adapter', 'weights', 'tokenizer', 'template', 'preparation', 'scoring',
                   'execution', 'sdk_assets'), 'native variant')
    name = result.get('adapter')
    if name not in NATIVE_KEYS:
        raise ValueError('Unknown native adapter')
    for key in ('weights', 'tokenizer'):
        validate_source(result.get(key))
        if 'file' in result[key]:
            raise ValueError(key + ' requires a directory source, not a selected file')
    p, s, e = (result.setdefault(key, {}) for key in ('preparation', 'scoring', 'execution'))
    prep_keys = {'limit'} | ({'system_prompt', 'cue', 'enable_thinking'} if name == 'halo' else
        {'token_pooling'} if name == 'gliner' else {'questions'} if name == 'laya' else
        {'guideline', 'response_guideline'} if name == 'shield' else set())
    score_keys = {'threshold'} if name == 'halo' else {'max_new_tokens'} if name in ('hive', 'open') else set()
    _keys(p, prep_keys, name + ' preparation')
    _keys(s, score_keys, name + ' scoring')
    _keys(e, ('runtime', 'dtype', 'attention', 'sdpa_backend', 'linear_attention',
              'compile_backend', 'compile_mode', 'dynamic', 'fullgraph', 'opset', 'cpu_policy'), 'execution')
    if type(p.get('limit')) is not int or p['limit'] < 1:
        raise ValueError('preparation.limit must be a positive integer')
    defaults = {'runtime': 'eager', 'dtype': 'native_sdk' if name == 'laya' else
                'bfloat16' if name in ('halo', 'hive', 'open', 'shield') else 'float32',
                'attention': 'sdpa' if name == 'laya' else 'eager' if name in ('gliner', 'shield') else 'native',
                'linear_attention': 'native' if name == 'halo' else 'reference'}
    for key, value in defaults.items():
        e.setdefault(key, value)
    e.setdefault('sdpa_backend', None)
    if e['runtime'] not in ('eager', 'compile', 'ort') or e['dtype'] not in ('float16', 'float32', 'bfloat16', 'native_sdk'):
        raise ValueError('Unsupported runtime or dtype')
    if e['attention'] not in ('native', 'eager', 'sdpa') or e['sdpa_backend'] not in (None, 'math', 'efficient'):
        raise ValueError('Unsupported attention settings')
    if e['attention'] != 'sdpa' and e['sdpa_backend'] is not None:
        raise ValueError('sdpa_backend requires SDPA')
    if e['linear_attention'] not in ('native', 'reference', 'fla') or (name != 'halo' and e['linear_attention'] != 'reference'):
        raise ValueError('FLA is supported only for Halo')
    if name != 'gliner' and e['runtime'] != 'eager':
        raise ValueError('compile/ORT are supported only for GLiNER')
    if (name == 'laya') != (e['dtype'] == 'native_sdk'):
        raise ValueError('Laya requires native_sdk; other adapters require a physical dtype')
    if name == 'laya' and e['attention'] != 'sdpa':
        raise ValueError('Laya SDK requires SDPA; kernel selection is independent')
    if name in ('gliner', 'shield') and e['attention'] != 'eager':
        raise ValueError(name + ' currently requires eager full attention')
    compile_keys = {'compile_backend', 'compile_mode', 'dynamic', 'fullgraph'}
    ort_keys = {'opset', 'cpu_policy'}
    if (e['runtime'] != 'compile' and compile_keys & e.keys()) or (e['runtime'] != 'ort' and ort_keys & e.keys()):
        raise ValueError('Backend options do not match runtime')
    if e['runtime'] == 'compile':
        for key, value in {'compile_backend': 'inductor', 'compile_mode': 'default', 'dynamic': True, 'fullgraph': True}.items():
            e.setdefault(key, value)
            if e[key] != value or type(e[key]) is not type(value):
                raise ValueError('Unsupported compile option: ' + key)
    if e['runtime'] == 'ort':
        e.setdefault('opset', 17)
        e.setdefault('cpu_policy', 'metadata-cpu-v1')
        if e['dtype'] != 'float32' or e['opset'] != 17 or e['cpu_policy'] != 'metadata-cpu-v1' or p['limit'] > 512:
            raise ValueError('ORT requires FP32, opset 17, metadata-cpu-v1, limit <=512')
    if name == 'halo':
        p.setdefault('cue', 'Safety: ')
        p.setdefault('enable_thinking', False)
        s.setdefault('threshold', .65)
        if type(p['enable_thinking']) is not bool or not isinstance(p['cue'], str):
            raise ValueError('Invalid Halo preparation')
        if 'system_prompt' in p and not isinstance(p['system_prompt'], str):
            raise ValueError('Halo system_prompt must be text')
        if type(s['threshold']) not in (int, float) or not 0 <= s['threshold'] <= 1:
            raise ValueError('Invalid Halo threshold')
    sdk = result.get('sdk_assets', {})
    needed = {'prompt'} if name == 'halo' and 'system_prompt' not in p else {'config'} if name in ('gliner', 'laya') else set()
    if not isinstance(sdk, dict) or set(sdk) != needed:
        raise ValueError('sdk_assets must contain exactly: ' + ', '.join(sorted(needed)))
    for asset in sdk.values():
        validate_source(asset)
    if name in ('hive', 'open'):
        s.setdefault('max_new_tokens', 1 if name == 'hive' else 10)
        if type(s['max_new_tokens']) is not int or s['max_new_tokens'] < 1:
            raise ValueError('Invalid generation budget')
    if name == 'gliner' and p.get('token_pooling', 'first') not in ('first', 'mean', 'max'):
        raise ValueError('Unsupported GLiNER token_pooling')
    if name == 'laya':
        p.setdefault('questions', deepcopy(QUESTIONS))
        q = p['questions']
        if (not isinstance(q, dict) or set(q) != {'danger'} or not isinstance(q['danger'], dict)
                or set(q['danger']) != {'type', 'instructions', 'criteria'} or q['danger']['type'] != 'choice'
                or not isinstance(q['danger']['instructions'], str) or not isinstance(q['danger']['criteria'], dict)
                or set(q['danger']['criteria']) != {'A', 'B'}
                or any(not isinstance(v, str) for v in q['danger']['criteria'].values())):
            raise ValueError('Laya requires one danger choice question with A/B criteria')
    if name == 'shield':
        p.setdefault('guideline', GUIDELINE)
        p.setdefault('response_guideline', RESPONSE_GUIDELINE)
        if any(not isinstance(p[k], str) for k in ('guideline', 'response_guideline')):
            raise ValueError('Shield guidelines must be text')
    template = result.setdefault('template', None)
    if template is not None:
        if name in ('gliner', 'koala', 'llama', 'laya'):
            raise ValueError(name + ' has no chat template; override would be ignored')
        if isinstance(template, dict) and set(template) == {'text'}:
            if not isinstance(template['text'], str) or not template['text']:
                raise ValueError('Inline template must be nonempty text')
        else:
            if not isinstance(template, dict) or 'file' not in template:
                raise ValueError('Template requires text or source plus file')
            validate_source({k: v for k, v in template.items() if k != 'file'})
            f = template['file']
            if not isinstance(f, str) or not f or Path(f).is_absolute() or '..' in Path(f).parts:
                raise ValueError('Invalid template file')
    return result


class Preparer:
    def __init__(self, spec):
        from transformers import AutoTokenizer
        self.spec = validate_settings(spec)
        self.name = self.spec['adapter']
        self.settings = self.spec['preparation']
        self.limit = self.settings['limit']
        self.snapshot = source_path(self.spec['tokenizer'], TOKENIZER_FILES)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.snapshot), trust_remote_code=False)
        self.tokenizer.model_max_length = 10**9
        self.assets = {}
        self.versions = {}
        self.input_changed = False
        template = self.spec['template']
        if template is not None:
            if 'text' in template:
                self.tokenizer.chat_template = template['text']
            else:
                root = source_path({k: v for k, v in template.items() if k != 'file'}, [template['file']])
                path = root / template['file'] if root.is_dir() else root
                self.tokenizer.chat_template = path.read_text()
        if self.name == 'halo':
            self.prompt = self.settings.get('system_prompt')
            if self.prompt is None:
                self.prompt = author_prompt(self._asset('prompt', 'README.md').read_text())
            self.assets['system_prompt'] = digest(self.prompt)
        if self.name == 'gliner':
            from gliner2.processor import SchemaTransformer
            from gliner2.inference.schema import Schema
            config = json.loads(self._asset('config', 'config.json').read_text())
            pooling = self.settings.get('token_pooling', config.get('token_pooling', 'first'))
            if pooling not in ('first', 'mean', 'max'):
                raise ValueError('GLiNER checkpoint requests unsupported token_pooling')
            self.processor = SchemaTransformer(tokenizer=self.tokenizer, token_pooling=pooling)
            self.schema = Schema().classification(task='safety', labels=['safe', 'unsafe'], multi_label=False)
            self.assets['processor'] = digest(inspect.getsource(SchemaTransformer))
            self.assets['token_pooling'] = digest(pooling)
        if self.name == 'laya':
            from laya.agent import Agent
            self.agent = Agent.__new__(Agent)
            self.agent.tok = self.tokenizer
            self.agent.cfg = json.loads(self._asset('config', 'rl_agent_config.json').read_text())
            if self.agent.cfg.get('max_len') != self.limit:
                raise ValueError('Laya limit must equal checkpoint SDK budget')
            self.questions = self.settings['questions']
            self.internal = {k: Agent._to_internal(v) for k, v in self.questions.items()}
            self.assets['sdk_encoder'] = digest(inspect.getsource(Agent._encode_state))
        backend = getattr(self.tokenizer, 'backend_tokenizer', None)
        if backend is not None:
            self.assets['effective_tokenizer'] = digest(backend.to_str())
        self.assets['effective_chat_template'] = digest(self.tokenizer.chat_template)
        self.assets['preparation_source'] = sha(__file__)
        self.metadata = {'adapter': self.name, 'limit': self.limit, 'assets': dict(self.assets)}

    def _asset(self, key, default_file):
        source = self.spec['sdk_assets'][key]
        filename = source.get('file', default_file)
        root = source_path(source, [filename])
        path = root / filename if root.is_dir() else root
        self.assets[key] = sha(path)
        return path

    def render(self, value):
        choice = select(self.name, value)
        effective = choice['effective_messages']
        if self.name == 'halo':
            return self.tokenizer.apply_chat_template(
                [{'role': 'system', 'content': self.prompt}, {'role': 'user', 'content': effective[0]['content']}],
                tokenize=False, add_generation_prompt=True, enable_thinking=self.settings['enable_thinking']) + self.settings['cue']
        if self.name == 'hive':
            return self.tokenizer.apply_chat_template(effective, tokenize=False)
        if self.name == 'open':
            return self.tokenizer.apply_chat_template(effective, tokenize=False, add_generation_prompt=True)
        if self.name == 'shield':
            policy = self.settings['response_guideline' if choice['target_role'] == 'assistant' else 'guideline']
            return self.tokenizer.apply_chat_template(effective, tokenize=False, guideline=policy)
        return effective[0]['content']

    def prepare(self, value):
        text = self.render(value)
        self.input_changed = False
        if self.name == 'gliner':
            batch = self.processor.collate_fn_inference([(text, self.schema.build())], max_len=None,
                                                       error_policy='raise', architecture='span')
            self.input_changed = batch.original_texts != [text]
            return normalized({'input_ids': batch.input_ids, 'attention_mask': batch.attention_mask})
        if self.name == 'laya':
            items = self.agent._encode_state(text, list(self.questions), self.internal,
                                             max_len=max(self.limit, len(text) * 8 + 4096))
            if len(items) != 1:
                raise ValueError('Laya must encode exactly one question per payload')
            ids = items[0]['ids']
            return {'input_ids': ids, 'attention_mask': [1] * len(ids)}
        kwargs = {'truncation': False, 'padding': False, 'return_attention_mask': True}
        if self.name in ('halo', 'shield'):
            kwargs['add_special_tokens'] = False
        return normalized(self.tokenizer(text, **kwargs))

    def index(self, value):
        row = value if isinstance(value, dict) else None
        if row is not None:
            value = row.get('messages', row.get('text'))
        source = messages(value)
        common = {'source_sha256': digest(source)}
        if row is not None and 'id' in row:
            common['id'] = row['id']
        try:
            choice = select(self.name, source)
        except UnsupportedInput as error:
            return {**common, 'supported': False, 'reason': str(error), 'total_tokens': None,
                    'input_sha256': None, 'ids_sha256': None, 'mask_sha256': None}
        data = self.prepare(source)
        n = len(data['input_ids'])
        supported = 0 < n <= self.limit and not self.input_changed
        return {**common, **{k: v for k, v in choice.items() if k != 'effective_messages'},
                'total_tokens': n, 'user_tokens': sum(len(self.tokenizer.encode(m['content'], add_special_tokens=False))
                    for m in source if m['role'] == 'user'),
                'content_tokens': sum(len(self.tokenizer.encode(m['content'], add_special_tokens=False)) for m in source),
                'supported': supported, 'input_changed': self.input_changed,
                'reason': 'processor_changed_text' if self.input_changed else None if supported else 'outside_input_limit',
                'input_sha256': digest(data), 'ids_sha256': digest(data['input_ids']),
                'mask_sha256': digest(data['attention_mask'])}

    def close(self):
        pass

    def parity(self):
        """Compare single-input indexing to native batch/template/SDK preparation."""
        from speed_dataset.preparation import CHECK_TEXTS
        checks = []
        cases = [[{'role': 'user', 'content': text}] for text in CHECK_TEXTS]
        cases += [[{'role': 'assistant', 'content': 'Library hours.'}],
                  [{'role': 'user', 'content': 'Library hours?'},
                   {'role': 'assistant', 'content': 'Nine oclock.'}]]
        for source in cases:
            record = {'source_sha256': digest(source)}
            try:
                choice = select(self.name, source)
            except UnsupportedInput:
                checks.append({**record, 'status': 'not_applicable'})
                continue
            actual = self.prepare(source)
            text = choice['effective_messages'][0]['content']
            if self.name == 'gliner':
                from experiments.adapters import GlinerAdapter
                from types import SimpleNamespace
                native = GlinerAdapter.__new__(GlinerAdapter)
                native.model = SimpleNamespace(processor=self.processor)
                native.schema, native.device = self.schema, 'cpu'
                native.limit = max(self.limit, len(actual['input_ids']))
                try:
                    expected = normalized(native.prepare([text]))
                except ValueError:
                    if not self.input_changed:
                        raise
                    checks.append({**record, 'status': 'expected_rejection', 'reason': 'processor_changed_text'})
                    continue
            elif self.name == 'laya':
                from laya.common import collate_items
                items = self.agent._encode_state(text, list(self.questions), self.internal)
                expected = normalized(collate_items([items], self.tokenizer.pad_token_id))
                if len(actual['input_ids']) > self.limit:
                    checks.append({**record, 'status': 'expected_rejection', 'reason': 'outside_input_limit'})
                    continue
            elif self.name in ('hive', 'open', 'shield'):
                kwargs = {'tokenize': True, 'return_dict': True, 'return_attention_mask': True}
                if self.name == 'open':
                    kwargs['add_generation_prompt'] = True
                if self.name == 'shield':
                    kwargs['guideline'] = self.settings['response_guideline'
                        if choice['target_role'] == 'assistant' else 'guideline']
                expected = normalized(self.tokenizer.apply_chat_template(choice['effective_messages'], **kwargs))
            else:
                kwargs = {'padding': False, 'truncation': False, 'return_attention_mask': True,
                          'return_tensors': 'pt'}
                if self.name == 'halo':
                    kwargs['add_special_tokens'] = False
                expected = normalized(self.tokenizer([self.render(source)], **kwargs))
            if actual != expected:
                raise ValueError('Native preparation parity failed: ' + self.name)
            checks.append({**record, 'status': 'passed', 'input_sha256': digest(actual),
                           'total_tokens': len(actual['input_ids'])})
        if not any(item['status'] == 'passed' for item in checks):
            raise ValueError('No successful native parity cases')
        return {'status': 'passed', 'scope': 'CPU native preparation only; no weights or GPU', 'checks': checks}


def create_preparer(spec):
    if isinstance(spec.get('adapter'), dict):
        result = external_module(spec).create_preparer(deepcopy(spec))
        if any(not callable(getattr(result, key, None)) for key in ('index', 'prepare', 'parity')):
            raise ValueError('External preparer requires index, prepare and parity')
        return result
    return Preparer(spec)
