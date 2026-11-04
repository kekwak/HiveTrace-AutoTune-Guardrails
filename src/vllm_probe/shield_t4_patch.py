"""Narrow research-only SM75/FP32/head256 Triton tile workaround.

Apply only inside a disposable Colab runtime. Does not change soft caps,
attention masks, weights, precision, or model configuration.
"""
import hashlib
import json
from pathlib import Path

MARKER = '# HIVETRACE_SM75_FP32_TILE16_RESEARCH'
ANCHOR = '''    """Select tile size with Gemma3-specific optimization."""
    if _is_gemma3_attention(head_size, sliding_window):'''
REPLACEMENT = '''    """Select tile size with Gemma3-specific optimization."""
    # HIVETRACE_SM75_FP32_TILE16_RESEARCH
    cap = current_platform.get_device_capability() if current_platform.is_cuda() else None
    if (cap is not None and cap.major == 7 and cap.minor == 5
            and head_size == 256 and element_size == 4):
        return 16
    if _is_gemma3_attention(head_size, sliding_window):'''


def select_t4_tile(default_tile, major, minor, head_size, element_size):
    return 16 if (major,minor,head_size,element_size)==(7,5,256,4) else default_tile


def patch_source(source):
    if MARKER in source:
        if source.count(REPLACEMENT)!=1:
            raise ValueError('unexpected partially patched source')
        return source
    if source.count(ANCHOR)!=1 or source.count('def _get_tile_size(')!=1:
        raise ValueError('unexpected upstream tile-selection source; refusing patch')
    return source.replace(ANCHOR,REPLACEMENT,1)


def apply_runtime_patch():
    import importlib.metadata
    import vllm
    if importlib.metadata.version('vllm')!='0.32.0':
        raise ValueError('patch validated only against vLLM 0.32.0')
    path=Path(vllm.__file__).parent/'v1/attention/ops/triton_unified_attention.py'
    before=path.read_text()
    after=patch_source(before)
    # Syntax-check before changing the installed runtime source.
    compile(after,str(path),'exec')
    path.write_text(after)
    return dict(vllm_version='0.32.0',before_sha256=hashlib.sha256(before.encode()).hexdigest(),
                after_sha256=hashlib.sha256(after.encode()).hexdigest(),
                changed=before!=after,scope='sm75 FP32 head256 tile16; caps unchanged')


if __name__=='__main__':
    report=apply_runtime_patch()
    Path('/content/shield-patch-manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
