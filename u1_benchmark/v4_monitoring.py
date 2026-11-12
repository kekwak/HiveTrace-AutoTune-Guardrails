"""One-time paired on/off check of the concurrent sampler, not a benchmark."""
import time

from .core import write
from .messages import payload
from .statistics import paired
from .v4_config import MONITOR_VERSION
from .v4_planning import materialize


class DisabledSampler:
    def __init__(self, **kwargs):
        pass

    def start(self):
        pass

    def inference(self):
        pass

    def stop(self, path):
        return {"status": "disabled", "method": MONITOR_VERSION}


def measure(adapter, rows, plans, guard, torch, gpu_uuid, directory):
    from .v4_memory import Sampler
    from .v4_worker import responses
    groups = [[payload(row) for row in materialize(plan, rows)] for plan in plans]
    guard.call(adapter.prepare_batch_backend, groups)
    proof = guard.call(adapter.prove_batch_backend, groups[0])
    result = {"status": "running", "monitor_method": MONITOR_VERSION, "backend_proof": proof, "blocks": []}
    for block in range(3):
        sides = {}
        for side in (("off", "on") if block % 2 == 0 else ("on", "off")):
            sampler = Sampler(gpu_uuid) if side == "on" else DisabledSampler()
            sampler.start()
            try:
                for i in range(20):
                    responses(guard.call(adapter.request_batch, groups[i % len(groups)]), 1)
                torch.cuda.synchronize()
                before = adapter.compile_count
                sampler.inference()
                values = []
                for plan, group in zip(plans, groups, strict=True):
                    torch.cuda.synchronize()
                    start = time.perf_counter_ns()
                    scores = guard.call(adapter.request_batch, group)
                    torch.cuda.synchronize()
                    ms = (time.perf_counter_ns() - start) / 1e6
                    responses(scores, 1)
                    values.append({"batch_id": plan["id"], "latency_ms": ms})
                if adapter.compile_count != before:
                    raise ValueError("Compilation during monitoring check")
                sides[side] = {"rows": values}
            finally:
                memory = sampler.stop(directory / f"monitor-{block}-{side}.json")
            sides[side]["memory"] = memory
        result["blocks"].append(sides)
        write(directory / "monitoring.json", result)
    blocks = [tuple([r["latency_ms"] for r in block[side]["rows"]] for side in ("off", "on"))
              for block in result["blocks"]]
    result.update(p50=paired(blocks, .5), p95=paired(blocks, .95))
    memory_ok = all(block["on"]["memory"].get("windows", {}).get("inference", {}).get("status") == "measured"
                    for block in result["blocks"])
    result["status"] = ("passed" if memory_ok and all(result[k]["ratio_interval_95"][1] <= 1.05
                                                    for k in ("p50", "p95")) else "unvalidated")
    result["memory_available"] = memory_ok
    write(directory / "monitoring.json", result)
    return result
