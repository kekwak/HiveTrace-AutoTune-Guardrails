"""Native decoder preparation; no weights, truncation, or fabricated context.

Callers load tokenizer/template/model card from adapter_contracts' pinned
revisions. Output-only ShieldGemma is explicitly missing-context diagnostic.
Other decoder output-only inputs are rejected pending a native role contract.
"""
import ast
import re


def official_policies(readme):
    policies = []
    for name in ['Dangerous Content', 'Harassment', 'Hate Speech', 'Sexually Explicit Information']:
        found = re.findall(r'^\| '+re.escape(name)+r'\s*\| `([^`]+)`', readme, re.MULTILINE)
        if len(found) != 2:
            raise ValueError('Pinned card must contain input and output wording for every official policy')
        policies.append(dict(name=name, input=found[0], output=found[1]))
    return policies


def author_prompt(readme):
    found = []
    for block in re.findall(r'```python\s*\n(.*?)```', readme, re.S):
        try:
            nodes = ast.parse(block).body
        except SyntaxError:
            continue
        for node in nodes:
            if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == 'GUARD_SYSTEM_PROMPT'
                    for target in node.targets):
                try:
                    found.append(ast.literal_eval(node.value))
                except (ValueError, TypeError, SyntaxError):
                    raise ValueError('Author prompt is not a literal') from None
    if len(found) != 1 or not isinstance(found[0], str) or not found[0]:
        raise ValueError('Expected one literal author system prompt')
    return found[0]


def prepare_decoder(model, tokenizer, text, *, system_prompt=None, template=None,
                    guideline=None, output=False):
    if not isinstance(text, str):
        raise ValueError('Text must be a string')
    if output and model != 'shield':
        raise ValueError('Output-only role not supported by this preparation contract')
    messages = [{'role': 'user', 'content': text}]
    options = dict(tokenize=False)
    if model == 'halo':
        if not system_prompt:
            raise ValueError('Pinned author system prompt required')
        messages.insert(0, {'role': 'system', 'content': system_prompt})
        options.update(add_generation_prompt=True, enable_thinking=False)
    elif model == 'open':
        if not template:
            raise ValueError('Pinned 2510 task template required for 0124 checkpoint')
        options.update(chat_template=template, add_generation_prompt=True)
    elif model == 'shield':
        if not guideline:
            raise ValueError('Explicit pinned policy wording required')
        options['guideline'] = guideline
        if output:
            messages = [{'role': 'user', 'content': ''}, {'role': 'assistant', 'content': text}]
    elif model != 'hive':
        raise ValueError('Not a decoder preparation contract')
    rendered = tokenizer.apply_chat_template(messages, **options)
    if model == 'halo':
        rendered += 'Safety: '
    return tokenizer(rendered, add_special_tokens=model in ('hive', 'open'),
                     truncation=False, padding=False)['input_ids']


def pair_token_ids(model, tokenizer):
    labels = ('No', 'Yes') if model == 'shield' else ('safe', 'unsafe') if model == 'halo' else None
    if labels is None:
        raise ValueError('Not a pair-scoring model')
    ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(len(value) != 1 for value in ids) or ids[0] == ids[1]:
        raise ValueError('Native verdicts must be distinct single tokens')
    return tuple(value[0] for value in ids)
