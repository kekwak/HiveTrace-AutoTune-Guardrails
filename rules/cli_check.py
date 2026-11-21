import json
import random
import statistics
import sys
import time

import torch

from guardrail_tuner.cli import inspect_model, plan_checks
from guardrail_tuner.execution import Inference
from guardrail_tuner.rules import RuledModel, compare_outputs

random.seed(0)
WORDS = (
    "please tell me how the weather is today and whether you can help with my homework "
    "this message contains no harmful content but it is long enough to vary the length"
).split()
TEXTS = [" ".join(random.choice(WORDS) for _ in range(random.randint(3, 90))) for _ in range(300)]


def timed(inference, texts):
    out, lat = [], []
    for text in texts:
        torch.cuda.synchronize()
        t = time.perf_counter()
        out += inference.predict([text])
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1000)
    lat.sort()
    return out, {"p50": statistics.median(lat), "p95": lat[int(len(lat) * 0.95)]}


for model in sys.argv[1:]:
    plan = plan_checks(inspect_model(model, None), "cuda")
    config = {**plan["execution"], "batch_size": 1}
    inference = Inference({**config, "rules": []})
    reference, eager = timed(inference, TEXTS)
    row = {"model": model, "rules": config["rules"], "eager": eager}
    for name, rules in (("compile_only", ["u4"]), ("planned", config["rules"])):
        ruled = RuledModel(inference.model, rules, inference.tokenizer.pad_token_id)
        inference.runner.model = ruled
        start = time.perf_counter()
        timed(inference, TEXTS)  # warm-up: compilation and graph capture per bucket
        warm = time.perf_counter() - start
        outputs, latency = timed(inference, TEXTS)
        row[name] = {
            **latency,
            "applied": ruled.applied,
            "skipped": ruled.skipped,
            "warmup_s": round(warm, 1),
            "parity": compare_outputs(reference, outputs),
        }
        torch._dynamo.reset()
        inference = Inference({**config, "rules": []})
    print(json.dumps(row), flush=True)
