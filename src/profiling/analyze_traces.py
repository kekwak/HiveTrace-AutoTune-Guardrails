import json
from pathlib import Path

import pandas as pd

DIR = Path("results/profiling")


def category(name):
    if "gemm" in name and "fp16" in name:
        return "linear (fp16 gemm)"
    if "sgemm" in name:
        return "attention (fp32 sgemm)"
    if "softmax" in name:
        return "attention (softmax)"
    if "gemv" in name:
        return "lm_head (gemv)"
    return "other (elementwise, norms, casts)"


rows = []
for path in sorted(DIR.glob("trace_*.json")):
    events = json.loads(path.read_text())["traceEvents"]
    ops = {e["name"] for e in events if e.get("cat") == "cpu_op"}
    for e in events:
        if e.get("cat") == "kernel":
            rows.append({
                "trace": path.stem.removeprefix("trace_"),
                "category": category(e["name"]),
                "ms": e["dur"] / 1e3,
                "sdpa_math": "aten::_scaled_dot_product_attention_math" in ops,
            })

df = pd.DataFrame(rows)
table = df.pivot_table(index="trace", columns="category", values="ms", aggfunc="sum").fillna(0)
share = table.div(table.sum(axis=1), axis=0).round(2)
share["total_ms"] = table.sum(axis=1).round(1)
share["sdpa_math"] = df.groupby("trace")["sdpa_math"].first()

print(share.to_string())
share.to_csv(DIR / "kernel_breakdown.csv")
