"""H3 feasibility only, in a dedicated runtime with user-installed vLLM.

Never rewrites the CUDA stack. Do not interpret setup timing as a speedup.
"""

import importlib.metadata
import math
import time
import traceback
import uuid
from pathlib import Path

from .adapters import HALO_ID, HALO_REVISION, HaloAdapter, UnsupportedExperiment
from .common import archive, regression, write_json
from .data import materialize
from .runner import configure_seed, evaluate, hardware


def two_logprobs(values, safe_id, unsafe_id, threshold=.65):
    if safe_id not in values or unsafe_id not in values:
        raise UnsupportedExperiment("Runtime did not return both requested token probabilities")
    a, b = float(values[safe_id]), float(values[unsafe_id])
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Nonfinite label logprobs")
    maximum = max(a, b)
    x, y = math.exp(a - maximum), math.exp(b - maximum)
    p = y / (x + y)
    return {"p_safe": x / (x + y), "p_unsafe": p, "logit_margin": b - a,
            "label": "unsafe" if p >= threshold else "safe"}


def run_probe(output_root):
    directory = Path(output_root) / ("vllm_probe_" + uuid.uuid4().hex[:10])
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "in_progress", "scope": "feasibility_not_performance", "quality_evaluated": False}
    adapter = None
    try:
        import torch
        from vllm import LLM, SamplingParams
        write_json(directory / "environment.json", hardware())
        configure_seed()
        params = SamplingParams(max_tokens=1, temperature=1.0, top_p=1.0, top_k=-1,
                                logprob_token_ids=[0, 1], detokenize=False, seed=42)
        adapter = HaloAdapter({"dtype": "float32"}, directory)
        rows = materialize(adapter, "development")
        write_json(directory / "inputs.json", rows)
        write_json(directory / "contract.json", adapter.contract)
        reference = evaluate(adapter, rows, directory / "reference_scores.jsonl")
        safe, unsafe = adapter.label_ids
        params.logprob_token_ids = [safe, unsafe]
        prompts = [{"prompt_token_ids": adapter.tokenizer.encode(adapter.render(r["text"]), add_special_tokens=False)}
                   for r in rows]
        adapter.close()
        adapter = None
        # Restore optional bindings changed for the historical FP32 control.
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
        for key, value in getattr(qwen, "_experiment_originals", {}).items():
            setattr(qwen, key, value)
        started = time.perf_counter()
        engine = LLM(model=HALO_ID, revision=HALO_REVISION, tokenizer_revision=HALO_REVISION,
                     dtype="half", tensor_parallel_size=1, max_model_len=1200,
                     enable_prefix_caching=False, gpu_memory_utilization=.75,
                     trust_remote_code=False, seed=42, max_num_seqs=1)
        manifest["engine_init_seconds"] = time.perf_counter() - started
        candidates = []
        from .common import append_jsonl
        for row, prompt in zip(rows, prompts):
            request = engine.generate([prompt], params, use_tqdm=False)[0]
            logs = request.outputs[0].logprobs
            if not logs or len(logs) != 1:
                raise UnsupportedExperiment("Expected one next-token logprob record")
            score = two_logprobs({k: v.logprob for k, v in logs[0].items()}, safe, unsafe)
            record = {"id": row["id"], "input_hash": row["input_hash"], **score}
            candidates.append(record)
            append_jsonl(directory / "candidate_scores.jsonl", record)
        result = regression(reference, candidates, "float16", .65)
        write_json(directory / "regression.json", result)
        manifest.update(status="completed" if result["passed"] else "regression_failed",
                        vllm_version=importlib.metadata.version("vllm"),
                        next_step="Dedicated paired performance adapter after feasibility review; not implemented by this probe")
        del engine
        torch.cuda.synchronize()
    except Exception as error:
        manifest.update(status="unsupported" if isinstance(error, (UnsupportedExperiment, ImportError, TypeError)) else "failed",
                        error=str(error))
        (directory / "error.txt").write_text(traceback.format_exc())
    finally:
        if adapter is not None:
            adapter.close()
        write_json(directory / "manifest.json", manifest)
        print("H3 feasibility:", manifest["status"], archive(directory))
    return directory
