"""GPU sanity before any measurement: RTX 5090, torchao 0.17.0, FP8 per-row Linear + compile close to bf16.

python -m u2_quant.preflight   # на поде, в /workspace/u2-env; код возврата 0 — можно запускать замеры
"""
import json
import sys


def check():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; FP8 preflight needs the pod GPU")
    name, capability = torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
    if capability < (8, 9):
        raise RuntimeError(f"FP8 needs compute capability >= 8.9, got {capability} on {name}")
    import torchao
    from u2_quant import quantize
    torch.manual_seed(17)
    layer = torch.nn.Sequential(torch.nn.Linear(1024, 1024)).to("cuda", torch.bfloat16)
    x = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        reference = layer(x)
    info = quantize.apply(layer, "fp8", [])
    compiled = torch.compile(layer, dynamic=True, fullgraph=True)
    with torch.inference_mode():
        actual = compiled(x)
    error = ((actual.float() - reference.float()).norm() / reference.float().norm()).item()
    if not error < 0.05:
        raise RuntimeError(f"FP8 linear relative error {error:.4f} >= 0.05")
    return {"gpu": name, "capability": list(capability), "torch": torch.__version__,
            "torchao": torchao.__version__, "quantized": info["quantized"], "relative_error": error}


if __name__ == "__main__":
    try:
        print(json.dumps(check()))
    except Exception as error:
        print(f"PREFLIGHT FAILED: {error}", file=sys.stderr)
        sys.exit(1)
