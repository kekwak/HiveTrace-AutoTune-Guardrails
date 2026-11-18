"""A bounded sequence of stages sharing one loaded adapter and sampler."""
from copy import deepcopy
import os
from pathlib import Path
import random

from .core import digest, read, sha, write
from .precision import PrecisionGuard
from .v4_config import CACHE_POLICY, PROTOCOL_VERSION, RESULT_VERSION


FIXED_METADATA = ("artifacts", "preparation", "batch_policy", "attention", "linear_attention")


def check_group(manifest):
    jobs = [read(p) for p in manifest["jobs"]]
    first = jobs[0]
    resolved_path = Path(first["resolved_path"])
    if sha(resolved_path) != first["resolved_sha256"]:
        raise ValueError("Lifecycle resolved snapshot changed")
    resolved = read(resolved_path)
    if (resolved.get("result_version"), resolved.get("protocol_version")) != (RESULT_VERSION, PROTOCOL_VERSION):
        raise ValueError("Unsupported lifecycle protocol")
    index = manifest["group_index"]
    if type(index) is not int or not 0 <= index < len(resolved["execution_groups"]):
        raise ValueError("Invalid lifecycle index")
    expected = resolved["execution_groups"][index]
    if (manifest["lifecycle_id"] != f"worker-{index:05d}" or manifest["variant"] != expected["variant"]
            or digest(manifest["main_batches"]) != expected["main_batches_sha256"]
            or len(jobs) != len(expected["stages"])):
        raise ValueError("Lifecycle plan differs from frozen snapshot")
    for order, (path, job, stage) in enumerate(zip(manifest["jobs"], jobs, expected["stages"], strict=True)):
        if (not Path(path).resolve().is_relative_to(resolved_path.parent / "jobs")
                or job["resolved_path"] != str(resolved_path)
                or job["resolved_sha256"] != first["resolved_sha256"]
                or job.get("lifecycle_id") != manifest["lifecycle_id"] or job.get("stage_order") != order
                or any(job.get(k) != v for k, v in stage.items())):
            raise ValueError("Lifecycle stage differs from frozen snapshot")
        if job["stage"] == "latency" and (digest(job["batches"]) != job["plan_sha256"]
                or job["ids"] != [key for batch in job["batches"] for key in batch["ids"]]):
            raise ValueError("Lifecycle batch plan changed")


class StageSampler:
    def __init__(self, owner, stage_id, stage):
        self.owner, self.stage_id, self.stage = owner, stage_id, stage
        self._result = None

    def start(self):
        owner = self.owner
        if owner.sampler is None:
            owner.sampler = owner.sampler_factory(gpu_uuid=owner.runtime["gpu_uuid"])
            owner.sampler.start()
        self.window("setup" if owner.adapter is None else "validation")

    def window(self, name):
        sampler = self.owner.sampler
        if callable(getattr(sampler, "begin_window", None)):
            sampler.begin_window(self.stage_id, name,
                                 "inference" if name in ("latency", "quality", "preflight") else "setup")

    def inference(self):
        self.window(self.stage)

    def checkpoint(self, path):
        if self._result is not None:
            return deepcopy(self._result)
        sampler = self.owner.sampler
        if callable(getattr(sampler, "stage_snapshot", None)):
            return sampler.stage_snapshot(path, self.stage_id)
        return {"status": "unavailable"}

    def stop(self, path):
        if self._result is not None:
            return deepcopy(self._result)
        sampler = self.owner.sampler
        if callable(getattr(sampler, "end_window", None)):
            sampler.end_window()
        if callable(getattr(sampler, "stage_snapshot", None)):
            self._result = sampler.stage_snapshot(path, self.stage_id, complete=True)
        else:
            self._result = {"status": "unavailable"}
        return deepcopy(self._result)


class Lifecycle:
    def __init__(self, manifest, *, adapter_factory, runtime_factory, torch_module, sampler_factory):
        self.manifest = manifest
        self.adapter_factory, self.runtime_factory = adapter_factory, runtime_factory
        self.torch, self.sampler_factory = torch_module, sampler_factory
        self.adapter = self.sampler = self.runtime = None
        self.guard = self.identity = self.rng = None
        self.main_prepared = False

    def get_runtime(self, environment):
        if self.runtime is None:
            self.runtime = self.runtime_factory(environment)
        return self.runtime

    def load(self, spec, directory):
        if self.adapter is None:
            self.adapter = self.adapter_factory(spec, directory)
            self.spec_sha = digest(spec)
            self.guard = PrecisionGuard(self.adapter, required=True)
            self.identity = {k: deepcopy(self.adapter.metadata.get(k)) for k in FIXED_METADATA}
            self.rng = {"python": random.getstate()}
            import numpy as np
            self.rng["numpy"] = np.random.get_state()
            if callable(getattr(self.torch, "get_rng_state", None)):
                self.rng["torch"] = self.torch.get_rng_state().clone()
                self.rng["cuda"] = [s.clone() for s in self.torch.cuda.get_rng_state_all()]
        elif digest(spec) != self.spec_sha:
            raise ValueError("A lifecycle cannot switch model variants")
        self.check()
        return self.adapter

    def check(self):
        if callable(getattr(self.adapter, "check_device", None)):
            self.adapter.check_device()
        self.guard.check()
        if any(self.adapter.metadata.get(k) != v for k, v in self.identity.items()):
            raise ValueError("Loaded artifact or execution contract changed between stages")

    def reset_rng(self):
        import numpy as np
        random.setstate(self.rng["python"])
        np.random.set_state(self.rng["numpy"])
        if "torch" in self.rng:
            self.torch.set_rng_state(self.rng["torch"].clone())
            self.torch.cuda.set_rng_state_all([s.clone() for s in self.rng["cuda"]])


def execute_group(path, *, adapter_factory=None, runtime_factory=None, torch_module=None, sampler_factory=None):
    from .v4_worker import execute, error_status
    from .v4_evidence import configure_gpu
    path = Path(path)
    manifest = read(path)
    check_group(manifest)
    # Configure caches before importing Torch/Triton or initializing CUDA.
    cache = path.parent / (".cache-" + manifest["lifecycle_id"])
    old_env = {k: os.environ.get(k) for k in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH")}
    for key in old_env:
        target = cache / key.lower()
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    context, last, failure = None, None, None
    try:
        if torch_module is None:
            import torch as torch_module
        if adapter_factory is None:
            from .v4_adapters import load_adapter
            adapter_factory = load_adapter
        if sampler_factory is None:
            from .v4_memory import Sampler
            sampler_factory = Sampler
        context = Lifecycle(manifest, adapter_factory=adapter_factory,
                            runtime_factory=runtime_factory or configure_gpu,
                            torch_module=torch_module, sampler_factory=sampler_factory)
        for index, name in enumerate(manifest["jobs"]):
            job_path = Path(name)
            job = read(job_path)
            if job.get("lifecycle_id") != manifest["lifecycle_id"] or job.get("stage_order") != index:
                raise ValueError("Lifecycle stage order changed")
            if failure:
                write(job_path.parent / "result.json", {
                    "result_version": 5, "status": "not_run", "job": job,
                    "resolved_sha256": job["resolved_sha256"], "reason": "previous_stage_failed"})
                continue
            if job["stage"] == "latency":
                proof = Path(job["preflight_path"])
                if not proof.is_file():
                    failure = "PreflightUnavailable"
                    write(job_path.parent / "result.json", {
                        "result_version": 5, "status": "not_run", "job": job,
                        "resolved_sha256": job["resolved_sha256"], "reason": "preflight_unavailable"})
                    continue
                if "preflight_sha256" not in job:
                    job["preflight_sha256"] = sha(proof)
                    write(job_path, job)
            last = job_path.parent
            result = execute(job_path, adapter_factory=context.load,
                             runtime_factory=context.get_runtime, torch_module=torch_module,
                             sampler_factory=lambda **kw: StageSampler(context, str(index), job["stage"]),
                             lifecycle=context)
            if result["status"] != "passed":
                failure = result.get("error_type", result["status"])
    finally:
        close_error = None
        if context is not None:
            if context.sampler is not None:
                try:
                    memory = context.sampler.stop(path.parent / (manifest["lifecycle_id"] + ".memory.json"))
                    if memory.get("errors") and last is not None and (last / "result.json").exists():
                        result = read(last / "result.json")
                        result["sampler_shutdown"] = {"status": "partial", "error_count": len(memory["errors"])}
                        write(last / "result.json", result)
                except Exception as error:
                    if last is not None and (last / "result.json").exists():
                        result = read(last / "result.json")
                        result["memory"] = {"status": "unavailable", "error_type": type(error).__name__}
                        write(last / "result.json", result)
            if context.adapter is not None:
                try:
                    context.adapter.close()
                except Exception as error:
                    close_error = error
        if close_error is not None and last is not None and (last / "result.json").exists():
            result = read(last / "result.json")
            result["close_error_type"] = type(close_error).__name__
            if result["status"] == "passed":
                result["status"] = error_status(close_error)
            write(last / "result.json", result)
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return {"status": "failed" if failure or close_error else "passed", "lifecycle_id": manifest["lifecycle_id"]}
