"""Text-only serving contracts for classification and deterministic generation."""

import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from .execution import Inference

ROUTES = {
    "/classify": "sequence_classification",
    "/v1/completions": "generation",
    "/v1/chat/completions": "generation",
}


class APIRequestError(ValueError):
    pass


def text_inputs(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if (
        not isinstance(value, list)
        or not 0 < len(value) <= 128
        or any(not isinstance(text, str) or not text.strip() for text in value)
    ):
        raise APIRequestError("Expected non-empty text or a list of up to 128 text inputs")
    return value


def serve_request(inference: "Inference", path: str, body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise APIRequestError("Expected a JSON object")
    if inference.config["task"] != ROUTES[path]:
        raise APIRequestError("The loaded model does not support this route")
    model = inference.config["model"]
    if body.get("model", model) != model:
        raise APIRequestError("Requested model does not match the loaded model")
    created = int(time.time())

    if path == "/classify":
        reject_options(body, {"model", "input", "user"})
        data, prompt_tokens = classify(inference, text_inputs(body.get("input")))
        return {
            "id": f"classify-{uuid4().hex}",
            "object": "list",
            "created": created,
            "model": model,
            "data": data,
            "usage": token_usage(prompt_tokens, 0),
        }

    chat = path == "/v1/chat/completions"
    input_key = "messages" if chat else "prompt"
    reject_options(
        body,
        {
            "model",
            input_key,
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "stream",
            "user",
        },
    )
    if body.get("stream", False) is not False or body.get("temperature", 0) != 0:
        raise APIRequestError("Supported generation uses stream=false and temperature=0")
    limits = [body[key] for key in ("max_tokens", "max_completion_tokens") if key in body]
    maximum = inference.config["max_new_tokens"]
    limit = limits[0] if limits else maximum
    if len(limits) > 1 or type(limit) is not int or not 1 <= limit <= maximum:
        raise APIRequestError(f"Specify one token limit between 1 and {maximum}")
    prompts = (
        [chat_prompt(inference, body.get(input_key))] if chat else text_inputs(body.get(input_key))
    )
    choices, prompt_tokens, completion_tokens = generate(inference, prompts, limit, chat)
    return {
        "id": f"{'chatcmpl' if chat else 'cmpl'}-{uuid4().hex}",
        "object": "chat.completion" if chat else "text_completion",
        "created": created,
        "model": model,
        "choices": choices,
        "usage": token_usage(prompt_tokens, completion_tokens),
    }


def reject_options(body: dict[str, Any], supported: set[str]) -> None:
    unknown = sorted(body.keys() - supported)
    if unknown:
        raise APIRequestError(f"Unsupported request fields: {', '.join(unknown)}")


def token_usage(prompt: int, completion: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def classify(inference: "Inference", inputs: list[str]) -> tuple[list[dict[str, Any]], int]:
    data: list[dict[str, Any]] = []
    tokens = 0
    batch_size = inference.config["batch_size"]
    sigmoid = inference.model.config.problem_type == "multi_label_classification"
    for start in range(0, len(inputs), batch_size):
        encoded = inference.runner.encode(inputs[start : start + batch_size])
        tokens += int(encoded.attention_mask.sum().item())
        logits = inference.runner.forward(encoded)
        scores = logits.sigmoid() if sigmoid or logits.shape[1] == 1 else logits.softmax(dim=-1)
        for row in scores.tolist():
            best = max(range(len(row)), key=row.__getitem__)
            data.append(
                {
                    "index": len(data),
                    "label": inference.labels.get(best, str(best)),
                    "probs": row,
                    "num_classes": len(row),
                }
            )
    return data, tokens


def chat_prompt(inference: "Inference", messages: Any) -> str:
    from jinja2 import TemplateError

    if (
        not isinstance(messages, list)
        or not 0 < len(messages) <= 128
        or any(
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in ("system", "user", "assistant")
            or not isinstance(message["content"], str)
            for message in messages
        )
    ):
        raise APIRequestError("Expected text messages with system, user or assistant roles")
    try:
        return inference.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except (TemplateError, ValueError, RuntimeError, TypeError) as exc:
        raise APIRequestError("The model chat template could not render these messages") from exc


def generate(
    inference: "Inference", prompts: list[str], limit: int, chat: bool
) -> tuple[list[dict[str, Any]], int, int]:
    choices = []
    prompt_tokens = completion_tokens = 0
    eos = inference.model.generation_config.eos_token_id
    stop_tokens = eos if isinstance(eos, list) else [eos]
    for index, prompt in enumerate(prompts):
        ids, count = inference.generate(prompt, limit, add_special_tokens=not chat)
        prompt_tokens += count
        completion_tokens += len(ids)
        content = inference.tokenizer.decode(ids, skip_special_tokens=True)
        stopped = len(ids) < limit or bool(ids and ids[-1] in stop_tokens)
        payload = (
            {"message": {"role": "assistant", "content": content}} if chat else {"text": content}
        )
        choices.append(
            {
                "index": index,
                "finish_reason": "stop" if stopped else "length",
                "logprobs": None,
                **payload,
            }
        )
    return choices, prompt_tokens, completion_tokens
