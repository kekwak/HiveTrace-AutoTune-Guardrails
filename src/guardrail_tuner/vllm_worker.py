"""Private subprocess for the vLLM probe. Never print prompts or generated text."""

import contextlib
import gc
import importlib
import json
import os
import sys
from pathlib import Path


def run(payload):
    import torch
    import transformers
    from huggingface_hub import snapshot_download

    from .manifest import set_seed

    vllm = importlib.import_module("vllm")
    seed = int(os.environ.get("GUARDRAIL_TUNER_SEED", "42"))
    set_seed(seed)

    model_id = payload["model"]
    local = Path(model_id).expanduser()
    if local.is_dir():
        if payload["revision"]:
            raise ValueError("revision_with_local_model")
        path, revision = str(local.resolve()), None
    else:
        path = snapshot_download(repo_id=model_id, revision=payload["revision"])
        revision = Path(path).name
    config = transformers.AutoConfig.from_pretrained(path, trust_remote_code=False)
    architectures = getattr(config, "architectures", None) or []
    if len(architectures) != 1:
        raise ValueError("ambiguous_architecture")
    cls = getattr(transformers, architectures[0], None)
    if (
        cls is None
        or not isinstance(cls, type)
        or not issubclass(cls, transformers.PreTrainedModel)
    ):
        raise ValueError("unsupported_architecture")
    tokenizer = transformers.AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    model = cls.from_pretrained(path, trust_remote_code=False).float().to("cuda").eval()
    prompts, expected = [], []
    for text in payload["inputs"]:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=payload["max_length"]
        )
        ids = inputs["input_ids"][0].tolist()
        prompts.append({"prompt_token_ids": ids})
        inputs = {key: val.to("cuda") for key, val in inputs.items()}
        with torch.inference_mode():
            if payload["task"] == "classification":
                logits = model(**inputs).logits[0].float()
                result = (
                    torch.softmax(logits, dim=-1)
                    if payload["output_mode"] == "multiclass"
                    else torch.sigmoid(logits)
                )
                expected.append(result.cpu().tolist())
            else:
                result = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=payload["max_new_tokens"],
                    eos_token_id=[],
                )
                expected.append(result[0, len(ids) :].tolist())
    del model, inputs, result
    gc.collect()
    torch.cuda.empty_cache()
    engine = vllm.LLM(
        model=path,
        tokenizer=path,
        dtype="float32",
        trust_remote_code=False,
        seed=seed,
        max_model_len=payload["max_length"] + payload["max_new_tokens"],
        **({"runner": "pooling"} if payload["task"] == "classification" else {}),
    )
    if payload["task"] == "classification":
        outputs = engine.classify(prompts, use_tqdm=False)
    else:
        outputs = engine.generate(
            prompts,
            vllm.SamplingParams(
                temperature=0, max_tokens=payload["max_new_tokens"], ignore_eos=True
            ),
            use_tqdm=False,
        )
    if len(outputs) != len(expected):
        raise ValueError("output_count_mismatch")
    max_score_error = 0.0
    if payload["task"] == "classification":
        differences = 0
        for output, ref in zip(outputs, expected):
            candidate = torch.as_tensor(output.outputs.probs, dtype=torch.float32).cpu()
            baseline = torch.tensor(ref, dtype=torch.float32)
            if candidate.shape != baseline.shape or not torch.isfinite(candidate).all():
                raise ValueError("invalid_classification_scores")
            max_score_error = max(max_score_error, float((candidate - baseline).abs().max()))
            if payload["output_mode"] == "multiclass":
                differences += int(candidate.argmax() != baseline.argmax())
            else:
                differences += int(
                    not torch.equal(
                        candidate >= payload["threshold"], baseline >= payload["threshold"]
                    )
                )
        status = (
            "scores_match"
            if differences == 0 and max_score_error <= payload["tolerance"]
            else "scores_differ"
        )
    else:
        differences = sum(
            list(output.outputs[0].token_ids) != ref for output, ref in zip(outputs, expected)
        )
        status = "tokens_match" if differences == 0 else "tokens_differ"
    return {
        "status": status,
        "model_revision": revision,
        "architecture": architectures[0],
        "gpu": torch.cuda.get_device_name(),
        "samples": len(expected),
        "mismatches": differences,
        "max_abs_score_error": max_score_error if payload["task"] == "classification" else None,
        "recommended": False,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "vllm": getattr(vllm, "__version__", "unknown"),
            "cuda": torch.version.cuda,
        },
        "settings": {
            "dtype": "float32",
            "seed": seed,
            "max_length": payload["max_length"],
            "max_new_tokens": payload["max_new_tokens"],
            "greedy": True,
            "ignore_eos": True,
            "task": payload["task"],
            "output_mode": payload["output_mode"] if payload["task"] == "classification" else None,
            "threshold": payload["threshold"] if payload["task"] == "classification" else None,
            "tolerance": payload["tolerance"] if payload["task"] == "classification" else None,
        },
        "limitations": [
            "Parsed application outputs and performance are not checked",
            "vLLM recommendation requires application quality and benchmark checks",
        ],
    }


def main():
    payload = json.load(sys.stdin)
    try:
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            result = run(payload)
    except Exception as exc:
        result = {"status": "failed", "reason": type(exc).__name__}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
