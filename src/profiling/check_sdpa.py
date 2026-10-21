import time

import torch as th
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

T = 1441
q = th.randn(1, 16, T, 128, device="cuda", dtype=th.float16)
k = th.randn(1, 8, T, 128, device="cuda", dtype=th.float16)
v = th.randn(1, 8, T, 128, device="cuda", dtype=th.float16)
k_rep = k.repeat_interleave(2, dim=1)
v_rep = v.repeat_interleave(2, dim=1)

CASES = {
    "math + gqa": (SDPBackend.MATH, k, v, True),
    "efficient + gqa": (SDPBackend.EFFICIENT_ATTENTION, k, v, True),
    "efficient + repeat_kv": (SDPBackend.EFFICIENT_ATTENTION, k_rep, v_rep, False),
}

with th.inference_mode():
    for name, (backend, kk, vv, gqa) in CASES.items():
        try:
            with sdpa_kernel(backend):
                for _ in range(10):
                    F.scaled_dot_product_attention(q, kk, vv, is_causal=True, enable_gqa=gqa)
                th.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(50):
                    F.scaled_dot_product_attention(q, kk, vv, is_causal=True, enable_gqa=gqa)
                th.cuda.synchronize()
            print(f"{name}: OK, {(time.perf_counter() - t0) / 50 * 1e3:.2f} ms per layer")
        except RuntimeError as e:
            print(f"{name}: FAIL, {str(e).splitlines()[0]}")
