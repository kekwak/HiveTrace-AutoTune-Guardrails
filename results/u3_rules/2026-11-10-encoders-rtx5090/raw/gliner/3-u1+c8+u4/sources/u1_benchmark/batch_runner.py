"""Sequential batch cells with isolated failures and one diagnostic stage per cell."""
from pathlib import Path

from .core import read, write


def cell_path(root, bucket, size):
    return Path(root) / "batches" / bucket / str(size)


def execute_matrix(cfg, data, output, launch, preflight_only=False):
    from .batching import cells, memory_plans, schedule, smoke_plans

    variants = ["baseline"] + (["candidate"] if cfg.get("candidate") else [])
    jobs = schedule(cfg, data)
    states = {}

    def save():
        write(Path(output) / "matrix_status.json", states)

    def invoke(phase, variant, directory, measurement, state):
        try:
            launch(phase, variant, directory, measurement)
            if read(directory / "result.json")["status"] != "passed":
                raise RuntimeError("Incomplete batch worker")
        except (TimeoutError, KeyboardInterrupt) as error:
            state.update(status="timeout" if isinstance(error, TimeoutError) else "interrupted", phase=phase, variant=variant)
            save()
            raise
        except Exception as error:
            result = read(directory / "result.json") if (directory / "result.json").exists() else {}
            state.update(status=result.get("status", "failed"), phase=phase, variant=variant,
                         error_type=result.get("error_type", type(error).__name__))
            if state["status"] == "passed":
                state["status"] = "failed"
            save()
            return False
        return True

    # Diagnostics are outside all A/A and A/B workers; no per-side diagnostic pass.
    for bucket, size in cells(cfg):
        key = f"{bucket}/{size}"
        state = states[key] = {"status": "pending", "bucket": bucket, "batch_size": size}
        root = cell_path(output, bucket, size)
        try:
            plans = smoke_plans(data[bucket], size, cfg["seed"])
        except ValueError as error:
            state.update(status="not_ready", error_type=type(error).__name__)
            save()
            continue
        measurement = {"bucket": bucket, "batch_size": size, "batches": plans}
        for variant in variants:
            if not invoke("batch_preflight", variant, root / "preflight" / variant, measurement, state):
                break
        else:
            state["status"] = "preflight_passed"
        save()

    if preflight_only or cfg["mode"] == "smoke":
        return states
    for bucket, size in cells(cfg):
        state = states[f"{bucket}/{size}"]
        if state["status"] != "preflight_passed":
            continue
        root = cell_path(output, bucket, size)
        selected = [(i, job) for i, job in enumerate(jobs)
                    if job["bucket"] == bucket and job["batch_size"] == size]
        for i, job in selected:
            if not invoke("batch_latency", job["variant"], Path(output) / "latency" / f"{i:03d}", job, state):
                break
        else:
            state["status"] = "latency_complete"
        if state["status"] != "latency_complete":
            continue
        if cfg["mode"] == "screening":
            plans = memory_plans(data[bucket], size, cfg["seed"])
            memory = {"bucket": bucket, "batch_size": size, "batches": plans}
            profiles = {"bucket": bucket, "batch_size": size,
                        "batches": smoke_plans(data[bucket], size, cfg["seed"])[:2]}
            for variant in variants:
                for repeat in range(3):
                    if not invoke("batch_memory", variant, root / "memory" / f"{variant}_{repeat}", memory, state):
                        break
                else:
                    if invoke("batch_profile", variant, root / "profiles" / variant, profiles, state):
                        continue
                break
            else:
                state["status"] = "completed"
        else:
            state["status"] = "completed"
        save()
    return states
