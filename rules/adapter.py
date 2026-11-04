import gc
import sys
import types
from contextlib import nullcontext

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo.backends.registry import lookup_backend
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AttentionInterface, StaticCache
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
from transformers.modeling_outputs import BaseModelOutput

from experiments.adapters import score_from_logits
from u1_benchmark.adapters import Builtin, parse_generated
from u1_benchmark.models import quiet_dispatch

BUCKETS = [64, 128, 256, 512, 1024, 2048, 4096]
ORDER = ["c13", "c4", "c1", "c12", "c8", "c3", "c2", "u3", "u4", "u1"]
DEC = ("hive", "halo", "shield")
GROUPS = [("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj"), ("query_proj", "key_proj", "value_proj"),
          ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")]
REL = ("query_proj", "key_proj", "pos_key_proj", "pos_query_proj", "pos_proj", "pos_q_proj")


class Skip(Exception):
    pass


class Entry(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, **x):
        return self.fn(**x)


class Rules(Builtin):
    def __init__(self, model, spec, directory, input_version):
        super().__init__(model, {"backend": "eager"}, directory, input_version)
        self.rules = spec["options"]["rules"]
        self.compiles, self.prefix = 0, None
        self.math, self.fast, self.short_head, self.drop_mask = True, False, False, False
        self.setup()
        applied, skipped = [], {}
        for name in sorted(self.rules, key=ORDER.index):
            try:
                getattr(self, name)()
                applied.append(name)
            except Skip as reason:
                skipped[name] = str(reason)
        if self.name not in ("hive", "open") or self.fast:
            self.target = self.entry
        self.metadata.update(rules_applied=applied, rules_skipped=skipped)

    def setup(self):
        m = self.model
        if self.name in DEC:
            if self.name == "halo":
                self.labels = self.legacy.label_ids
            elif self.name == "shield":
                self.labels = self.label_ids
            else:
                self.labels = [self.tokenizer.encode(s, add_special_tokens=False)[0] for s in ("safe", "unsafe")]
            self.body = lambda x: m.model(input_ids=x["input_ids"], attention_mask=x.get("attention_mask"),
                                          use_cache=False).last_hidden_state[0]
            self.softcap = getattr(m.config.get_text_config(), "final_logit_softcapping", None)
            self.head = lambda h: self.cap(m.lm_head(h))
            self.inner = lambda x: self.head(last(self.body(x), x))
        elif self.name == "open":
            self.labels = [self.tokenizer.encode("safe", add_special_tokens=False)[0]]
            self.inner = lambda x: m(**x, use_cache=True, logits_to_keep=1)
        elif self.name == "gliner":
            forward = m.encoder.forward
            self.inner = lambda x: forward(**x).last_hidden_state
            m.encoder.forward = lambda input_ids=None, attention_mask=None, **kw: BaseModelOutput(
                last_hidden_state=self.entry(input_ids=input_ids, attention_mask=attention_mask))
        else:
            self.inner = lambda x: m(**x).logits[0]
        self.entry = Entry(self.route)

    def cap(self, logits):
        if self.softcap is None:
            return logits
        return th.tanh(logits / self.softcap) * self.softcap

    def ctx(self):
        if self.name == "halo":
            return quiet_dispatch(self.legacy.qwen, self.legacy.quiet_implementations)
        if self.name in ("hive", "open") and self.math:
            return sdpa_kernel(SDPBackend.MATH)
        return nullcontext()

    def core(self, x):
        with self.ctx():
            return self.inner(x)

    def route(self, **x):
        if self.drop_mask:
            x.pop("attention_mask", None)
        if self.prefix is not None:
            n, ids = len(self.prefix), x["input_ids"]
            if ids.shape[1] <= n or not th.equal(ids[0, :n], self.prefix):
                return self.full(x)
            x = {"input_ids": ids[:, n:]}
        return self.core(x)

    def need_core(self, graph=True):
        if self.name in ("hive", "open") and not self.fast:
            raise Skip("эталон через generate, нужен " + ("c13" if self.name == "hive" else "c4"))
        if graph and self.name == "open":
            raise Skip("категорию open догенерирует generate")

    @property
    def compile_count(self):
        return getattr(self.legacy, "compile_count", 0) + self.compiles

    def request(self, text):
        if self.name == "gliner" or self.name in ("hive", "open") and not self.fast:
            return super().request(text)
        if self.name == "open":
            return self.request_open(text)
        logits = self.entry(**self.prepare(text)).float()
        if not th.isfinite(logits).all():
            raise ValueError("Nonfinite logits")
        if self.name in ("koala", "llama"):
            probs = logits.softmax(-1).cpu().tolist()
            label = self.labels[probs.index(max(probs))]
            return {"label": label, "scores": dict(zip(self.labels, probs)), "score_type": "probabilities"}
        pair = logits if self.short_head else logits[self.labels]
        if self.name == "halo":
            v = score_from_logits(pair, self.legacy.threshold)
            return {**v, "scores": {"safe": v["p_safe"], "unsafe": v["p_unsafe"]}, "score_type": "probabilities"}
        if self.name == "shield":
            probs = pair.softmax(-1).cpu().tolist()
            label = "Yes" if probs[0] > probs[1] else "No"
            return {"label": label, "scores": dict(zip(("Yes", "No"), probs)), "score_type": "probabilities"}
        token = self.labels[int(pair.argmax())] if self.short_head else int(logits.argmax())
        return parse_generated(self.tokenizer.decode([token], skip_special_tokens=True), "hive", 1)

    def request_open(self, text):
        x = self.prepare(text)
        out = self.entry(**x)
        token = out.logits[0, -1].argmax().view(1, 1)
        if token.item() == self.labels[0]:
            return parse_generated("safe", "open", 1)
        ids = th.cat([x["input_ids"], token], 1)
        with self.ctx():
            ids = self.model.generate(input_ids=ids, attention_mask=th.ones_like(ids),
                                      past_key_values=out.past_key_values, do_sample=False, max_new_tokens=9)
        new = ids[0, x["input_ids"].shape[1]:].tolist()
        return parse_generated(self.tokenizer.decode(new, skip_special_tokens=True), "open", len(new))

    def prepare_backend(self, texts):
        super().prepare_backend(texts)
        for text in texts:
            self.request(text)

    def prove_backend(self, text):
        with th.profiler.profile() as p:
            self.request(text)
            th.cuda.synchronize()
        self.metadata["backend_proof"] = sorted({e.key for e in p.key_averages()})

    def u1(self):
        self.need_core()
        cache_positions()
        AttentionMaskInterface.register("eager", eager_mask)
        core, graphs, pool, stream = self.core, {}, th.cuda.graph_pool_handle(), th.cuda.Stream()

        def capture(b, x):
            keys = set(x) if self.name in DEC else {"input_ids", "attention_mask"}
            static = {k: th.zeros(1, b, dtype=th.long, device="cuda") for k in keys}
            if self.name in DEC:
                static["last"] = th.zeros(1, dtype=th.long, device="cuda")
            th.cuda.empty_cache()
            stream.wait_stream(th.cuda.current_stream())
            with th.cuda.stream(stream):
                for _ in range(3):
                    core(static)
            th.cuda.current_stream().wait_stream(stream)
            graph = th.cuda.CUDAGraph()
            with th.cuda.graph(graph, pool=pool):
                out = core(static)
            return static, graph, out

        def run(x):
            n = x["input_ids"].shape[1]
            b = next(b for b in BUCKETS if b >= n)
            if b not in graphs:
                graphs[b] = capture(b, x)
            static, graph, out = graphs[b]
            static["input_ids"][:, :n] = x["input_ids"]
            if "attention_mask" in static:
                static["attention_mask"].zero_()
                static["attention_mask"][:, :n] = 1
            if "last" in static:
                static["last"].fill_(n - 1)
            graph.replay()
            return out[:, :n].clone() if out.dim() == 3 else out.clone()

        self.core = run

    def u3(self):
        if self.name == "gliner":
            raise Skip("вход строит GLiNER")
        self.need_core(graph=False)
        self.drop_mask = True

    def u4(self):
        self.need_core()
        inductor = lookup_backend("inductor")

        def counted(graph, inputs):
            self.compiles += 1
            return inductor(graph, inputs)

        th._dynamo.config.cache_size_limit = 64
        self.inner = th.compile(self.inner, backend=counted, dynamic=False if "u1" in self.rules else None)

    def c1(self):
        if self.name not in DEC + ("open",):
            raise Skip("не decoder")
        cfg = self.model.config.get_text_config()
        if cfg.num_key_value_heads == cfg.num_attention_heads:
            raise Skip("нет GQA")
        if getattr(cfg, "attn_logit_softcapping", None):
            raise Skip("softcap внимания есть только в eager")
        self.need_core(graph=False)
        AttentionInterface.register("sdpa_rep", sdpa_rep)
        AttentionMaskInterface.register("sdpa_rep", sdpa_mask)
        self.model.set_attn_implementation("sdpa_rep")
        self.math = False

    def c2(self):
        self.need_core()
        if self.name not in DEC:
            raise Skip("не decoder")
        a, b = (self.prepare(t)["input_ids"][0] for t in ("Hello there", "Bye now"))
        n = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y) - 1
        model, body = self.model.model, self.body
        cache = StaticCache(config=self.model.config, max_cache_len=n + BUCKETS[-1])
        with self.ctx():
            model(input_ids=a[None, :n], past_key_values=cache, use_cache=True)
        counters = [l.cumulative_length for l in cache.layers if hasattr(l, "cumulative_length")]
        sliding = [l for l in cache.layers if hasattr(l, "cumulative_length_int")]
        states = [t for l in cache.layers for d in (getattr(l, "conv_states", {}), getattr(l, "recurrent_states", {}))
                  for t in d.values() if t is not None]
        snapshot = [t.clone() for t in states]

        def cached(x):
            for c in counters:
                c.fill_(n)
            for t, s in zip(states, snapshot):
                t.copy_(s)
            h = model(input_ids=x["input_ids"], past_key_values=cache, use_cache=True).last_hidden_state[0]
            for l in sliding:
                l.cumulative_length_int = n
            return h

        def full(x):
            with self.ctx():
                return self.head(last(body(x), x))

        self.prefix = a[:n]
        self.body = cached
        self.full = full

    def c3(self):
        self.need_core()
        if self.name not in DEC:
            raise Skip("не decoder")
        w = self.model.lm_head.weight[self.labels].clone()
        self.head = lambda h: self.cap(h @ w.T)
        self.short_head = True

    def c4(self):
        if self.name != "open":
            raise Skip("только open генерирует категорию")
        self.fast = True

    def c8(self):
        encoders = [m for m in self.model.modules() if hasattr(m, "get_rel_embedding")]
        if not encoders:
            raise Skip("не DeBERTa")
        cache_positions()
        for e in encoders:
            rel = e.get_rel_embedding()
            e.get_rel_embedding = lambda rel=rel: rel
            for m in list(e.modules()):
                for name in REL:
                    if isinstance(getattr(m, name, None), nn.Module):
                        setattr(m, name, Memo(getattr(m, name), rel))

    def c12(self):
        groups = 0
        for m in list(self.model.modules()):
            for names in GROUPS:
                lins = [getattr(m, k, None) for k in names]
                if all(isinstance(l, nn.Linear) for l in lins) and len({l.bias is None for l in lins}) == 1:
                    fused = Fused(lins)
                    setattr(m, "fused_" + names[0], fused)
                    for i, k in enumerate(names):
                        setattr(m, k, Part(fused, i))
                    groups += 1
        if not groups:
            raise Skip("проекции уже слиты")

    def c13(self):
        if self.name != "hive":
            raise Skip("для open это c4" if self.name == "open" else "эталон и так один forward")
        self.fast = True


def sdpa_rep(module, q, k, v, mask, scaling=None, dropout=0.0, **kw):
    n = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(n, 1), v.repeat_interleave(n, 1)
    causal = mask is None and q.shape[2] > 1
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scaling, is_causal=causal)
    return out.transpose(1, 2).contiguous(), None


def eager_mask(*args, dtype=th.float32, **kw):
    kw.pop("allow_is_causal_skip", None)
    mask = sdpa_mask(*args, allow_is_causal_skip=False, **kw)
    if mask is None:
        return None
    return th.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill_(~mask, th.finfo(dtype).min)


def cache_positions():
    from transformers.models.deberta import modeling_deberta as v1
    from transformers.models.deberta_v2 import modeling_deberta_v2 as v2

    for mod in (v1, v2):
        fn = mod.build_relative_position
        if hasattr(fn, "memo"):
            continue

        def cached(q, k, *args, fn=fn, memo={}, **kw):
            key = (q.size(-2), k.size(-2), q.device, args, tuple(kw.items()))
            if key not in memo:
                memo[key] = fn(q, k, *args, **kw)
            return memo[key]

        cached.memo = True
        mod.build_relative_position = cached


def last(h, x):
    return h[x["last"]][0] if "last" in x else h[-1]


class Memo(nn.Module):
    def __init__(self, lin, rel):
        super().__init__()
        self.lin = lin
        self.ptr = rel.untyped_storage().data_ptr()
        self.cache = {}

    def forward(self, x):
        if x.untyped_storage().data_ptr() != self.ptr:
            return self.lin(x)
        key = (x.data_ptr(), x.shape)
        if key not in self.cache:
            self.cache[key] = self.lin(x)
        return self.cache[key].clone()


class Fused(nn.Module):
    def __init__(self, lins):
        super().__init__()
        self.weight = nn.Parameter(th.cat([l.weight for l in lins]).detach())
        self.bias = None if lins[0].bias is None else nn.Parameter(th.cat([l.bias for l in lins]).detach())
        self.sizes = [l.out_features for l in lins]
        self.x, self.out = None, None


class Part(nn.Module):
    def __init__(self, fused, i):
        super().__init__()
        self.f = [fused]
        self.i = i
        start = sum(fused.sizes[:i])
        self.span = slice(start, start + fused.sizes[i])

    def forward(self, x):
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


def create(*, model, spec, directory, contract):
    sys.modules.setdefault(__name__, next(o for o in gc.get_referrers(globals()) if isinstance(o, types.ModuleType)))
    return Rules(model, spec, directory, contract.get("input_version", 2))
