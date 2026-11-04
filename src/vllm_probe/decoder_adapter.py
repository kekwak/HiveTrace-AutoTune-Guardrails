"""vLLM decoder scoring on prevalidated native token IDs, no text rewriting.

Imports no GPU libraries until create(). Engine injection is for CPU contract
tests only; it is not GPU validation. Pair IDs are (safe, unsafe), explicitly
checked against the pinned tokenizer by the preparer.
"""
from .adapter_contracts import MODELS, classify_scores, parse_verdict
from .shield_scoring import probability_yes


class DecoderAdapter:
    def __init__(self, model, engine, sampling_params, pair_ids=None):
        self.model = model
        self.spec = MODELS[model]
        if self.spec['kind'] not in ('generate', 'pair', 'policy_pair'):
            raise ValueError('Not a decoder contract')
        self.engine = engine
        self.sampling_params = sampling_params
        self.pair_ids = pair_ids
        if self.spec['kind'] != 'generate':
            if (pair_ids is None or len(pair_ids) != 2 or pair_ids[0] == pair_ids[1]
                    or any(type(i) is not int or i < 0 for i in pair_ids)):
                raise ValueError('Two distinct native single-token verdict IDs required')

    @classmethod
    def create(cls, model, pair_ids=None):
        from vllm import LLM, SamplingParams
        import importlib.metadata
        if importlib.metadata.version('vllm') != '0.32.0':
            raise ValueError('This experiment is pinned to vLLM 0.32.0')
        spec = MODELS[model]
        # ShieldGemma runtime patch must be applied in a separate process
        # before importing/creating the engine; never silently change caps.
        engine = LLM(model=spec['repo'], revision=spec['revision'],
                     dtype=spec['dtype'], max_model_len=spec['limit'],
                     max_num_seqs=4, max_num_batched_tokens=1024,
                     gpu_memory_utilization=.92, enforce_eager=True,
                     enable_prefix_caching=False, logprobs_mode='raw_logprobs')
        return cls(model, engine, SamplingParams, pair_ids)

    def score(self, input_ids):
        if not input_ids:
            return []
        generated = self.spec.get('max_new_tokens', 1)
        if any(not ids or len(ids) + generated > self.spec['limit']
               or any(type(i) is not int or i < 0 for i in ids) for ids in input_ids):
            raise ValueError('Invalid or over-limit input; no truncation')
        options = dict(temperature=0, max_tokens=generated)
        if self.pair_ids is not None:
            options['logprob_token_ids'] = list(self.pair_ids)
        outputs = self.engine.generate(
            [{'prompt_token_ids': list(ids)} for ids in input_ids],
            self.sampling_params(**options), use_tqdm=False)
        if len(outputs) != len(input_ids):
            raise ValueError('Engine result count mismatch')
        results = []
        for output in outputs:
            if len(output.outputs) != 1:
                raise ValueError('Expected one completion per prompt')
            completion = output.outputs[0]
            if self.spec['kind'] == 'generate':
                if self.model == 'open' and getattr(completion, 'finish_reason', None) == 'length':
                    raise ValueError('OpenGuardrails category may be truncated at token limit')
                results.append(parse_verdict(completion.text, self.model))
                continue
            if not completion.logprobs or any(i not in completion.logprobs[0] for i in self.pair_ids):
                raise ValueError('Missing native verdict logprobs')
            safe, unsafe = (completion.logprobs[0][i].logprob for i in self.pair_ids)
            p = probability_yes(unsafe, safe)
            results.append(classify_scores([1-p, p], ['safe', 'unsafe'],
                                           threshold=self.spec['threshold']))
        return results
