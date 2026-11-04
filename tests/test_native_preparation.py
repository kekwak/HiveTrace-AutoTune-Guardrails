import pytest
from src.vllm_probe.native_preparation import author_prompt, prepare_decoder, pair_token_ids, official_policies


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.options = messages, kwargs
        return 'rendered'

    def __call__(self, text, **kwargs):
        self.text, self.tokenize_options = text, kwargs
        return {'input_ids': [1, 2]}

    def encode(self, text, **kwargs):
        return {'safe': [3], 'unsafe': [4], 'Yes': [5], 'No': [6]}[text]


def test_author_prompt_is_literal_and_unambiguous():
    assert author_prompt('```python\nGUARD_SYSTEM_PROMPT = "policy"\n```') == 'policy'
    for card in ['nothing', '```python\nGUARD_SYSTEM_PROMPT = compute()\n```']:
        with pytest.raises(ValueError):
            author_prompt(card)


def test_halo_preserves_author_system_thinking_off_and_safety_suffix():
    tok = Tokenizer()
    assert prepare_decoder('halo', tok, 'hello', system_prompt='policy') == [1, 2]
    assert tok.messages[0] == {'role': 'system', 'content': 'policy'}
    assert tok.options['enable_thinking'] is False
    assert tok.text == 'renderedSafety: '
    assert tok.tokenize_options['add_special_tokens'] is False


def test_open_requires_explicit_pinned_template():
    tok = Tokenizer()
    with pytest.raises(ValueError):
        prepare_decoder('open', tok, 'hello')
    prepare_decoder('open', tok, 'hello', template='pinned')
    assert tok.options['chat_template'] == 'pinned'
    assert tok.tokenize_options['add_special_tokens'] is True


def test_shield_pair_ids_are_ordered_safe_then_unsafe():
    assert pair_token_ids('shield', Tokenizer()) == (6, 5)
    assert pair_token_ids('halo', Tokenizer()) == (3, 4)


def test_shield_requires_real_policy_and_preserves_empty_output_context():
    tok = Tokenizer()
    with pytest.raises(ValueError):
        prepare_decoder('shield', tok, 'hello')
    prepare_decoder('shield', tok, 'response', guideline='output policy', output=True)
    assert tok.messages == [{'role': 'user', 'content': ''}, {'role': 'assistant', 'content': 'response'}]
    assert tok.options['guideline'] == 'output policy'
    with pytest.raises(ValueError):
        prepare_decoder('hive', tok, 'response', output=True)


def test_four_official_policy_tables_required_in_pinned_card():
    names = ['Dangerous Content', 'Harassment', 'Hate Speech', 'Sexually Explicit Information']
    card = '\n'.join(f'| {name} | `{mode} {name}` |' for mode in ['input', 'output'] for name in names)
    policies = official_policies(card)
    assert len(policies) == 4
    assert policies[0] == {'name': names[0], 'input': 'input Dangerous Content', 'output': 'output Dangerous Content'}
    with pytest.raises(ValueError): official_policies(card.splitlines()[0])
