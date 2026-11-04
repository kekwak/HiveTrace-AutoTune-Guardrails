"""vLLM pooling callers for native classifiers, Laya logits and GLiNER hidden.

GLiNER hidden output is an encoder integration boundary, never a safety verdict.
Laya returns decision logits: calibration belongs to the pinned SDK contract.
"""
from .adapter_contracts import MODELS
from .encoder_contracts import parse_pooling, validate_metadata


class PoolingAdapter:
    def __init__(self, model, engine, pooling_params, labels=None):
        if model not in ('koala', 'llama', 'gliner', 'laya'):
            raise ValueError('Not an encoder/custom pooling contract')
        self.model, self.engine, self.pooling_params = model, engine, pooling_params
        self.labels = labels
        if model in ('koala', 'llama') and (labels is None or len(labels) != (9 if model == 'koala' else 2)):
            raise ValueError('Native classifier label order required')

    @classmethod
    def create(cls, model, artifact, labels=None):
        import importlib.metadata
        from vllm import LLM, ModelRegistry, PoolingParams
        if importlib.metadata.version('vllm') != '0.32.0':
            raise ValueError('Pinned vLLM 0.32.0 required')
        plugins = importlib.metadata.entry_points(group='vllm.general_plugins')
        if not any(plugin.name == 'guardrail_adapters' for plugin in plugins):
            raise ValueError('Install this repository with pip install --no-deps -e . before spawning vLLM')
        from .registration import register
        register(ModelRegistry)
        engine = LLM(model=str(artifact), runner='pooling',
                     pooler_config={'task': 'token_classify'}, dtype='float32',
                     max_model_len=512, max_num_seqs=4, max_num_batched_tokens=2048,
                     enable_chunked_prefill=False, enforce_eager=True,
                     enable_prefix_caching=False, gpu_memory_utilization=.5,
                     disable_hybrid_kv_cache_manager=True)
        return cls(model, engine, PoolingParams, labels)

    def encode(self, input_ids, metadata=None):
        if not input_ids:
            return []
        if any(not ids or len(ids) > MODELS[self.model]['limit']
               or any(type(i) is not int or i < 0 for i in ids) for ids in input_ids):
            raise ValueError('Invalid or over-limit input; no truncation')
        options = []
        if self.model == 'laya':
            if metadata is None or len(metadata) != len(input_ids):
                raise ValueError('Laya metadata count mismatch')
            for ids, item in zip(input_ids, metadata):
                validate_metadata('laya', len(ids), item)
                options.append(self.pooling_params(use_activation=False, extra_kwargs={'laya': item}))
        else:
            options = [self.pooling_params(use_activation=False) for _ in input_ids]
        outputs = self.engine.encode([{'prompt_token_ids': ids} for ids in input_ids],
                                     pooling_task='token_classify', pooling_params=options,
                                     use_tqdm=False)
        if len(outputs) != len(input_ids):
            raise ValueError('Pooling result count mismatch')
        return [row.outputs.data.tolist() for row in outputs]

    def score(self, input_ids):
        if self.model not in ('koala', 'llama'):
            raise ValueError('Custom hidden/logit outputs require native head/calibration')
        return [parse_pooling(values, self.labels) for values in self.encode(input_ids)]
