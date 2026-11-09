from .causal_verdict import CausalVerdictAdapter
from .gliner import GlinerAdapter
from .seq_cls import SeqClsAdapter

ADAPTERS = {"seq_cls": SeqClsAdapter, "causal_verdict": CausalVerdictAdapter, "gliner": GlinerAdapter}


def make_adapter(spec, device):
    if spec["adapter"] not in ADAPTERS:
        raise ValueError(f"Неизвестный adapter {spec['adapter']}; доступны {sorted(ADAPTERS)}")
    return ADAPTERS[spec["adapter"]](spec, device)
