import functools
import importlib
from typing import Any

RULES = ("c7", "c12", "c8", "u4", "u1")
BUCKETS = (64, 128, 256, 512, 1024, 2048, 4096)
GROUPS = (
    ("q_proj", "k_proj", "v_proj"),
    ("gate_proj", "up_proj"),
    ("query_proj", "key_proj", "value_proj"),
    ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
)
RELATIVE = ("query_proj", "key_proj", "pos_key_proj", "pos_query_proj", "pos_proj", "pos_q_proj")


class Skip(Exception):
    """The rule does not apply to the loaded model."""


def rule_features(config: dict[str, Any]) -> dict[str, bool]:
    text = config.get("text_config")
    text = text if isinstance(text, dict) else config
    layers = text.get("layer_types")
    return {
        "relative_positions": text.get("relative_attention") is True
        or text.get("model_type") == "deberta",
        "linear_attention": isinstance(layers, list) and "linear_attention" in layers,
    }


def select_rules(features: dict[str, bool], execution: dict[str, Any] | None) -> dict[str, Any]:
    if execution is None:
        task, cuda = None, False
    elif execution.get("runtime") != "transformers":
        return {"rules": [], "skipped": {}}
    else:
        task, cuda = execution["task"], execution["device"] == "cuda"
    forward = task == "sequence_classification" and cuda
    enabled = {
        "c7": features.get("linear_attention", False),
        "c12": task != "generation",
        "c8": task != "generation" and features.get("relative_positions", False),
        "u4": forward,
        "u1": forward,
    }
    reason = None
    if task is None:
        reason = "no inference contract: the forward to wrap is unknown"
    elif task == "generation":
        reason = "torch.compile inside generate was 13-26% slower"
    elif not forward:
        reason = "measured only on CUDA"
    skipped = dict.fromkeys(("u4", "u1"), reason) if reason else {}
    result: dict[str, Any] = {"rules": [r for r in RULES if enabled[r]], "skipped": skipped}
    if enabled["c7"]:
        result["requires"] = ["flash-linear-attention"]
    return result


class RuledModel:
    """Sequence classification forward with rules; calls return an object with `.logits`."""

    def __init__(self, model: Any, rules: list[str], pad_token_id: int | None) -> None:
        self.model = model
        self.rules = rules
        self.pad = pad_token_id or 0
        self.inner = lambda x: model(**x).logits
        self.graphs: dict[int, Any] | None = None
        self.applied: list[str] = []
        self.skipped: dict[str, str] = {}
        for name in sorted(rules, key=RULES.index):
            try:
                getattr(self, name)()
                self.applied.append(name)
            except Skip as reason:
                self.skipped[name] = str(reason)

    def __call__(self, **inputs: Any) -> Any:
        from transformers.modeling_outputs import SequenceClassifierOutput

        rows, length = inputs["input_ids"].shape
        if self.graphs is not None and rows == 1 and length <= BUCKETS[-1]:
            try:
                return SequenceClassifierOutput(logits=self.replay(inputs))
            except RuntimeError as exc:
                self.graphs = None
                self.applied.remove("u1")
                self.skipped["u1"] = f"graph capture failed: {type(exc).__name__}"
        return SequenceClassifierOutput(logits=self.inner(inputs))

    def c7(self) -> None:
        try:
            importlib.import_module("fla")
        except Exception as exc:
            raise Skip(f"flash-linear-attention does not import: {type(exc).__name__}") from exc

    def c12(self) -> None:
        import torch

        _, fused_cls, part_cls = modules()
        groups = 0
        for module in list(self.model.modules()):
            for names in GROUPS:
                linears: list[Any] = [getattr(module, name, None) for name in names]
                if not all(isinstance(lin, torch.nn.Linear) for lin in linears):
                    continue
                if len({lin.bias is None for lin in linears}) != 1:
                    continue
                fused = fused_cls(linears)
                setattr(module, "fused_" + names[0], fused)
                for i, name in enumerate(names):
                    setattr(module, name, part_cls(fused, i))
                groups += 1
        if not groups:
            raise Skip("projections are already fused")

    def c8(self) -> None:
        import torch

        memo_cls, _, _ = modules()
        encoders = [m for m in self.model.modules() if hasattr(m, "get_rel_embedding")]
        if not encoders:
            raise Skip("no relative position embeddings")
        cache_positions()
        with torch.no_grad():
            for encoder in encoders:
                rel = encoder.get_rel_embedding()
                if rel is None:
                    raise Skip("relative attention is disabled")
                encoder.get_rel_embedding = lambda rel=rel: rel
                for module in list(encoder.modules()):
                    for name in RELATIVE:
                        if isinstance(getattr(module, name, None), torch.nn.Module):
                            setattr(module, name, memo_cls(getattr(module, name), rel))

    def u4(self) -> None:
        import torch

        inner = self.inner
        self.inner = torch.compile(
            lambda x: inner(x), dynamic=False if "u1" in self.rules else None
        )

    def u1(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise Skip("CUDA is unavailable")
        cache_positions()
        self.graphs = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.stream = torch.cuda.Stream()

    def replay(self, inputs: dict[str, Any]) -> Any:
        import torch

        assert self.graphs is not None
        length = inputs["input_ids"].shape[1]
        bucket = next(b for b in BUCKETS if b >= length)
        if bucket not in self.graphs:
            static = {
                key: torch.zeros(1, bucket, dtype=value.dtype, device=value.device)
                for key, value in inputs.items()
            }
            static["input_ids"].fill_(self.pad)
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                for _ in range(3):
                    self.inner(static)
            torch.cuda.current_stream().wait_stream(self.stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                out = self.inner(static)
            self.graphs[bucket] = static, graph, out
        static, graph, out = self.graphs[bucket]
        for key, value in inputs.items():
            static[key].fill_(self.pad if key == "input_ids" else 0)
            static[key][:, :length] = value
        graph.replay()
        return out.clone()


def compare_outputs(
    reference: list[dict[str, Any]], candidate: list[dict[str, Any]]
) -> dict[str, Any]:
    changes = sum(a["label"] != b["label"] for a, b in zip(reference, candidate))
    error = max(
        (
            abs(x - y)
            for a, b in zip(reference, candidate)
            for x, y in zip(a["logits"], b["logits"])
        ),
        default=0.0,
    )
    return {"passed": changes == 0, "label_changes": changes, "max_abs_logit_error": error}


def cache_positions() -> None:
    from transformers.models.deberta import modeling_deberta as v1
    from transformers.models.deberta_v2 import modeling_deberta_v2 as v2

    for module in (v1, v2):
        build = getattr(module, "build_relative_position", None)
        if build is None or getattr(build, "memo", False):
            continue

        def cached(*args: Any, build: Any = build, memo: dict = {}, **kwargs: Any) -> Any:
            key = tuple(getattr(a, "shape", a) for a in args) + tuple(kwargs.items())
            key += tuple(a.device for a in args if hasattr(a, "device"))
            if key not in memo:
                memo[key] = build(*args, **kwargs)
            return memo[key]

        cached.memo = True  # type: ignore[attr-defined]
        module.build_relative_position = cached


@functools.cache
def modules() -> tuple[Any, Any, Any]:
    import torch
    import torch.nn.functional as F

    class Memo(torch.nn.Module):
        def __init__(self, linear: Any, rel: Any) -> None:
            super().__init__()
            self.linear = linear
            self.ptr = rel.untyped_storage().data_ptr()
            self.cache: dict[Any, Any] = {}

        def forward(self, x: Any) -> Any:
            if x.untyped_storage().data_ptr() != self.ptr:
                return self.linear(x)
            key = (x.data_ptr(), x.shape)
            if key not in self.cache:
                self.cache[key] = self.linear(x)
            return self.cache[key].clone()

    class Fused(torch.nn.Module):
        def __init__(self, linears: list[Any]) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.cat([lin.weight for lin in linears]).detach())
            bias = None if linears[0].bias is None else torch.cat([lin.bias for lin in linears])
            self.bias = None if bias is None else torch.nn.Parameter(bias.detach())
            self.sizes = [lin.out_features for lin in linears]
            self.x: Any = None
            self.out: Any = None

    class Part(torch.nn.Module):
        def __init__(self, fused: Any, i: int) -> None:
            super().__init__()
            self.f = [fused]
            self.i = i
            start = sum(fused.sizes[:i])
            self.span = slice(start, start + fused.sizes[i])

        def forward(self, x: Any) -> Any:
            f = self.f[0]
            if x is not f.x:
                if self.i:
                    bias = None if f.bias is None else f.bias[self.span]
                    return F.linear(x, f.weight[self.span], bias)
                f.x, f.out = x, F.linear(x, f.weight, f.bias).split(f.sizes, -1)
            out = f.out[self.i]
            if self.i == len(f.sizes) - 1:
                f.x, f.out = None, None
            return out

    return Memo, Fused, Part
