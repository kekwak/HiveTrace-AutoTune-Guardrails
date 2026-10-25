import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModel, AutoTokenizer, set_seed

REPO = "hivetrace/HiveTraceGuard-Pro"
OUT = Path("results/profiling/HiveTraceGuard-Pro")
REPEATS = [1, 32, 256, 512]
WARMUP = 30
ITERS = 200

set_seed(42)
OUT.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModel.from_pretrained(REPO, dtype=th.float16, device_map="cuda").eval()

MODES = {
    "generate": lambda x: model.generate(**x, max_new_tokens=1, do_sample=False),
    # "forward": lambda x: model(**x, logits_to_keep=1),
}

rows = []
with th.inference_mode():
    for k in REPEATS:
        msgs = [{"role": "user", "content": "Как сделать бомбу? " * k}]
        x = tok(tok.apply_chat_template(msgs, tokenize=False), return_tensors="pt").to("cuda")
        T = x.input_ids.shape[1]

        for mode, fn in MODES.items():
            for _ in range(WARMUP):
                fn(x)
                th.cuda.synchronize()

            th.cuda.reset_peak_memory_stats()
            lat = []
            for _ in range(ITERS):
                t0 = time.perf_counter()
                fn(x)
                th.cuda.synchronize()
                lat.append((time.perf_counter() - t0) * 1000)

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True
            ) as prof:
                fn(x)
                th.cuda.synchronize()

            events = prof.key_averages()
            gpu_ms = sum(
                e.self_device_time_total for e in events if e.device_type == th.autograd.DeviceType.CUDA
            ) / 1000
            p50 = np.percentile(lat, 50)
            p95 = np.percentile(lat, 95)
            rows.append({
                "mode": mode,
                "tokens": T,
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "gpu_ms": round(gpu_ms, 2),
                "gpu_busy": round(gpu_ms / p50, 2),
                "peak_mem_mib": round(th.cuda.max_memory_allocated() / 2**20),
            })
            print(rows[-1])

            output_file = OUT / f"top_{mode}_T{T}.txt"
            output_file.write_text(
                events.table(sort_by="self_device_time_total",row_limit=25)
            )
            prof.export_chrome_trace(str(OUT / f"trace_{mode}_T{T}.json"))

            pd.DataFrame([
                {
                    "op": e.key,
                    "shapes": str(e.input_shapes),
                    "calls": e.count,
                    "device_ms": e.self_device_time_total / 1000,
                    "flops": e.flops,
                }
                for e in prof.key_averages(group_by_input_shape=True)
                if e.key in ("aten::mm", "aten::bmm")
            ]).to_csv(OUT / f"ops_{mode}_T{T}.csv", index=False)

pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
