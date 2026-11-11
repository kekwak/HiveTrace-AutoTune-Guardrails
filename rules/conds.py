import importlib.util
import json
import os
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path

os.environ["USE_HUB_KERNELS"] = "NO"
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

import torch as th
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import PretrainedConfig

from adapter import GROUPS
from u1_benchmark.adapters import load
from u1_benchmark.datasets import load_speed
from u1_benchmark.messages import payload
from u1_benchmark.worker_v2 import current_rows

name = sys.argv[1]
spec = json.loads(Path(f"configs/benchmark-v3/{name}_baseline.json").read_text())
data, _ = load_speed(spec["data"]["path"], name, "smoke")
texts = [payload(r) for r in current_rows("smoke", data)]
base = load(name, spec["baseline"], Path("runs/rules/conds"), input_version=3)
model, target = base.model, base.target
cfg = next(c for c in (getattr(m, "config", None) for m in model.modules())
           if isinstance(c, PretrainedConfig) and hasattr(c.get_text_config(), "num_attention_heads")).get_text_config()
cap = th.cuda.get_device_capability()
fp32 = all(p.dtype == th.float32 for p in model.parameters())
can_generate = hasattr(model, "can_generate") and model.can_generate()
seen = {"calls": 0, "args": None, "autocast": None, "max_new_tokens": None}


def observe(module, args, kwargs):
    seen["calls"] += 1
    if seen["args"] is None:
        seen["args"] = args, kwargs
        seen["autocast"] = str(th.get_autocast_dtype("cuda")) if th.is_autocast_enabled("cuda") else None


generate = getattr(model, "generate", None)
if generate is not None:
    def recorded(*a, **k):
        seen["max_new_tokens"] = k.get("max_new_tokens")
        return generate(*a, **k)
    model.generate = recorded

with th.inference_mode():
    handle = target.register_forward_pre_hook(observe, with_kwargs=True)
    labels = [base.request(t)["label"] for t in texts]
    handle.remove()
    calls = seen["calls"] // len(texts)

    def strict(module, args, kwargs):
        th.cuda.set_sync_debug_mode("warn")

    def relaxed(module, args, out):
        th.cuda.set_sync_debug_mode(0)

    hooks = [target.register_forward_pre_hook(strict, with_kwargs=True), target.register_forward_hook(relaxed)]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            base.request(texts[0])
        sync = sum("called a synchronizing" in str(w.message) for w in caught)
    finally:
        th.cuda.set_sync_debug_mode(0)
        for h in hooks:
            h.remove()

    try:
        a = base.prepare("Hello there")["input_ids"][0].tolist()
        b = base.prepare("Bye now")["input_ids"][0].tolist()
        prefix = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y) - 1
        total = len(base.prepare(texts[0])["input_ids"][0])
    except Exception as error:
        prefix, total = None, type(error).__name__

    first = None
    if seen["max_new_tokens"] and seen["max_new_tokens"] > 1:
        first = []
        for t, label in zip(texts, labels):
            x = base.prepare(t)
            with sdpa_kernel(SDPBackend.MATH):
                ids = generate(**x, do_sample=False, max_new_tokens=1)[0, x["input_ids"].shape[1]:]
            first.append(base.tokenizer.decode(ids, skip_special_tokens=True).strip() == str(label))

args, kwargs = seen["args"]
kwargs = {k: v for k, v in kwargs.items() if k not in ("past_key_values", "cache_position", "use_cache")}
th._dynamo.reset()
amp = th.autocast("cuda", dtype=getattr(th, seen["autocast"].removeprefix("torch."))) if seen["autocast"] else nullcontext()
try:
    with th.inference_mode(), amp:
        breaks = th._dynamo.explain(target)(*args, **kwargs).graph_break_count
except Exception as error:
    breaks = type(error).__name__

heads, kv = cfg.num_attention_heads, getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
softcap = getattr(cfg, "attn_logit_softcapping", None)
linear = "linear_attention" in (getattr(cfg, "layer_types", None) or []) or any(
    type(m).__name__.endswith("GatedDeltaNet") for m in model.modules())
groups = sorted({"/".join(g) for m in model.modules() for g in GROUPS
                 if all(isinstance(getattr(m, k, None), nn.Linear) for k in g)
                 and len({getattr(m, k).bias is None for k in g}) == 1})
one_step = can_generate and calls == 1 and seen["max_new_tokens"] in (None, 1)
old_torch = tuple(int(v) for v in th.__version__.split("+")[0].split(".")[:2]) < (2, 14)

conds = {
    "U1": th.cuda.is_available(),
    "U3": bool(sync) or seen["max_new_tokens"] is not None,
    "U4": True,
    "C1": kv < heads and not softcap and (cap < (8, 0) or old_torch),
    "C2": can_generate and bool(prefix and prefix > 0),
    "C3": one_step and hasattr(model, "lm_head"),
    "C4": can_generate and bool(seen["max_new_tokens"] and seen["max_new_tokens"] > 1) and bool(first) and all(first),
    "C5": fp32 and not th.cuda.is_bf16_supported() and seen["autocast"] is None,
    "C6": bool(softcap) and cap >= (8, 0),
    "C7": linear and importlib.util.find_spec("fla") is not None,
    "C8": any(hasattr(m, "get_rel_embedding") for m in model.modules()),
    "C9": fp32 and seen["autocast"] is not None,
    "C10": not can_generate,
    "C12": bool(groups),
    "C13": can_generate and seen["max_new_tokens"] == 1,
}
facts = {"cap": cap, "torch": th.__version__, "fp32": fp32, "can_generate": can_generate, "forward_calls": calls,
         "max_new_tokens": seen["max_new_tokens"], "autocast": seen["autocast"], "sync_in_forward": sync,
         "graph_breaks": breaks, "heads": [heads, kv], "softcap": softcap, "linear_attention": linear,
         "prefix_tokens": [prefix, total], "first_token_verdict": first, "groups": groups}
print(json.dumps({"model": name, "conds": conds, "facts": facts}, default=str), flush=True)
