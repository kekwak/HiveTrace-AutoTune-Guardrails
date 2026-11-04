"""vLLM general plugin: register custom models in *each* engine process.

Registering only inside a parent factory is lost when the engine is spawned.
Keep imports lazy: neither Torch nor custom model code is imported here.
"""


def register(registry=None):
    if registry is None:
        from vllm import ModelRegistry
        registry = ModelRegistry
    registry.register_model('NativeHFEncoder',
                            'src.vllm_probe.hf_encoder_vllm_model:NativeHFEncoder')
    registry.register_model('LayaForDecisions',
                            'src.vllm_probe.laya_vllm_model:LayaForDecisions')
