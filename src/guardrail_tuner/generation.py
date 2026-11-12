"""Verify greedy generation with and without model cache, including optional compile."""

import math
import platform
from pathlib import Path
from typing import Any

from .benchmark import texts
from .datasets import DatasetSource, load_source


def check_generation(
    model_id: str,
    revision: str | None,
    device: str,
    quality: DatasetSource,
    text_column: str,
    max_length: int,
    max_new_tokens: int,
    tolerance: float,
    compile_candidate: bool,
) -> dict[str, Any]:
    try:
        import torch
        import transformers
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ValueError("Install the inference extra to check generation") from exc

    if max_length < 1 or max_new_tokens < 1 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid generation length or tolerance")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable to PyTorch")
    data, metadata = load_source(quality)
    inputs = texts(data, text_column, "quality")
    local = Path(model_id).expanduser()
    if local.is_dir():
        if revision:
            raise ValueError("--revision applies only to Hugging Face models")
        path, resolved = str(local.resolve()), None
    else:
        path = snapshot_download(repo_id=model_id, revision=revision)
        resolved = Path(path).name
    config = transformers.AutoConfig.from_pretrained(path, trust_remote_code=False)
    architectures = getattr(config, "architectures", None) or []
    if len(architectures) != 1:
        raise ValueError("Model must declare exactly one Transformers architecture")
    cls = getattr(transformers, architectures[0], None)
    if (
        cls is None
        or not isinstance(cls, type)
        or not issubclass(cls, transformers.PreTrainedModel)
    ):
        raise ValueError("Declared architecture is not available in installed Transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    model = cls.from_pretrained(path, trust_remote_code=False).float().to(device).eval()
    if not callable(getattr(model, "generate", None)):
        raise ValueError("Model does not expose a generation interface")

    def generate(text: str, cache: bool) -> Any:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            return model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=cache,
                return_dict_in_generate=True,
                output_scores=True,
            )

    def score_error(left: Any, right: Any) -> float:
        if (
            left.shape != right.shape
            or torch.isnan(left).any()
            or torch.isnan(right).any()
            or not torch.equal(torch.isposinf(left), torch.isposinf(right))
            or not torch.equal(torch.isneginf(left), torch.isneginf(right))
        ):
            raise ValueError("Generation scores have incompatible shapes or non-finite masks")
        finite = torch.isfinite(left)
        return (
            float((left[finite].float() - right[finite].float()).abs().max())
            if finite.any()
            else 0.0
        )

    cache_mismatches = 0
    cache_score_error = 0.0
    candidate_mismatches = 0
    candidate_score_error = 0.0
    candidate_error = None
    for text in inputs:
        uncached, cached = generate(text, False), generate(text, True)
        cache_mismatches += int(not torch.equal(uncached.sequences, cached.sequences))
        if len(uncached.scores) != len(cached.scores):
            cache_mismatches += 1
        else:
            for left, right in zip(uncached.scores, cached.scores):
                cache_score_error = max(cache_score_error, score_error(left, right))
    cache_passed = cache_mismatches == 0 and cache_score_error <= tolerance
    if compile_candidate and cache_passed:
        try:
            original_forward = model.forward
            optimized_forward = torch.compile(original_forward)
            for text in inputs:
                model.forward = original_forward
                reference = generate(text, True)
                model.forward = optimized_forward
                optimized = generate(text, True)
                candidate_mismatches += int(
                    not torch.equal(reference.sequences, optimized.sequences)
                )
                if len(reference.scores) != len(optimized.scores):
                    candidate_mismatches += 1
                else:
                    for left, right in zip(reference.scores, optimized.scores):
                        candidate_score_error = max(candidate_score_error, score_error(left, right))
        except (RuntimeError, OSError) as exc:
            candidate_error = type(exc).__name__
    candidate_passed = (
        compile_candidate
        and cache_passed
        and candidate_error is None
        and candidate_mismatches == 0
        and candidate_score_error <= tolerance
    )
    return {
        "model": model_id,
        "model_revision": resolved,
        "architecture": architectures[0],
        "device": device,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
        },
        "dataset": metadata,
        "settings": {
            "text_column": text_column,
            "max_length": max_length,
            "max_new_tokens": max_new_tokens,
            "max_score_error": tolerance,
            "decoding": "greedy",
            "reference_dtype": "float32",
        },
        "cache": {
            "passed": cache_passed,
            "sequence_mismatches": cache_mismatches,
            "max_abs_score_error": cache_score_error,
        },
        "candidate": {
            "runtime": "torch_compile" if compile_candidate else None,
            "passed": candidate_passed if compile_candidate else None,
            "sequence_mismatches": candidate_mismatches if compile_candidate else None,
            "max_abs_score_error": candidate_score_error if compile_candidate else None,
            "error": candidate_error,
        },
        "limitations": [
            "Only greedy text generation is checked",
            "Prompt construction and parsed application outputs are not checked",
            "No performance measurement or deployment compatibility check",
        ],
    }
