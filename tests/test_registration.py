import importlib.util
import sys
from pathlib import Path
import tomllib
import ast

from src.vllm_probe.registration import register


def test_plugin_entry_point_registers_lazy_models_without_gpu_imports():
    class Registry:
        def __init__(self): self.models = {}
        def register_model(self, architecture, target): self.models[architecture] = target
    registry = Registry()
    register(registry)
    assert registry.models == {
        'NativeHFEncoder': 'src.vllm_probe.hf_encoder_vllm_model:NativeHFEncoder',
        'LayaForDecisions': 'src.vllm_probe.laya_vllm_model:LayaForDecisions',
    }
    assert all(importlib.util.find_spec(target.split(':')[0]) is not None
               for target in registry.models.values())
    assert 'src.vllm_probe.hf_encoder_vllm_model' not in sys.modules


def test_plugin_is_discoverable_by_spawned_vllm_processes():
    config = tomllib.loads(Path('pyproject.toml').read_text())
    entries = config['project']['entry-points']['vllm.general_plugins']
    assert entries['guardrail_adapters'] == 'src.vllm_probe.registration:register'


def test_dense_encoder_declares_bidirectional_not_default_decoder_attention():
    # Static contract check, intentionally import-free on CPU-only dev hosts.
    tree = ast.parse(Path('src/vllm_probe/hf_encoder_vllm_model.py').read_text())
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'NativeHFEncoder')
    attributes = {target.id: ast.literal_eval(node.value)
                  for node in model.body if isinstance(node, ast.Assign)
                  for target in node.targets if isinstance(target, ast.Name)}
    assert attributes.get('attn_type') == 'encoder_only'
