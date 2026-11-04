import pytest
from src.vllm_probe.shield_t4_patch import select_t4_tile, patch_source


@pytest.mark.parametrize('major,minor,head,size,expected', [
    (7,5,256,4,16), (7,5,256,2,32), (8,0,256,4,32),
    (7,0,256,4,32), (7,5,128,4,32),
])
def test_tile_change_is_narrow(major,minor,head,size,expected):
    assert select_t4_tile(32,major,minor,head,size)==expected


def test_patch_rejects_unexpected_source():
    with pytest.raises(ValueError):
        patch_source('def something_else():\n    return 32\n')


def test_patch_is_idempotent_and_retains_original_policy():
    source='''def _get_tile_size(
    head_size: int,
    sliding_window: int,
    element_size: int,
    is_prefill: bool,
) -> int:
    """Select tile size with Gemma3-specific optimization."""
    if _is_gemma3_attention(head_size, sliding_window):
        return 32
    return 32 if is_prefill else 16
'''
    patched=patch_source(source)
    class Capability:
        major=7
        minor=5
    class Platform:
        def is_cuda(self):return True
        def get_device_capability(self):return Capability()
    scope={'current_platform':Platform(),'_is_gemma3_attention':lambda *args:False}
    exec(patched,scope)
    assert scope['_get_tile_size'](256,4096,4,True)==16
    assert scope['_get_tile_size'](128,4096,4,True)==32
    assert patch_source(patched)==patched
