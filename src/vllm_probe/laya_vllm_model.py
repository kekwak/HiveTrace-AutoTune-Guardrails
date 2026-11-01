"""Experimental FP32 decision-logit-only Laya integration for vLLM 0.32.0.

No action head, calibration, or public ReadStrategy. Input metadata must be
validated by the caller. Reference architecture: pinned Laya rl_common.py.
"""
import torch
from torch import nn
from vllm.model_executor.models.modernbert import ModernBertModel
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool


class LayaDecisionPooler(Pooler):
    def __init__(self, hidden_size):
        super().__init__()
        self.pooling = AllPool()
        layer = nn.TransformerEncoderLayer(hidden_size, hidden_size // 64,
            4 * hidden_size, 0.1, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.type_emb = nn.Embedding(3, hidden_size)
        self.scorer = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size),
            nn.GELU(), nn.Linear(hidden_size, 1))

    def get_supported_tasks(self):
        return {'token_classify'}

    def get_pooling_updates(self, task):
        return PoolingParamsUpdate()

    def forward(self, hidden_states, pooling_metadata):
        outputs = []
        for hidden, params in zip(self.pooling(hidden_states, pooling_metadata),
                                  pooling_metadata.pooling_params):
            if hidden is None:
                outputs.append(None)
                continue
            metadata = (params.extra_kwargs or {}).get('laya')
            if metadata is None:
                outputs.append(hidden.new_empty(0))
                continue
            qtype = torch.tensor([metadata['qtype']], device=hidden.device)
            h = hidden.unsqueeze(0) + self.type_emb(qtype)[:, None, :]
            h = self.head(h)
            markers = torch.tensor(metadata['marker_positions'], device=hidden.device)
            outputs.append(self.scorer(h[0, markers]).squeeze(-1).float())
        return outputs


@default_pooling_type(tok_pooling_type='ALL')
class LayaForDecisions(nn.Module):
    is_pooling_model = True

    def __init__(self, *, vllm_config, prefix=''):
        super().__init__()
        if vllm_config.model_config.dtype != torch.float32:
            raise ValueError('experimental Laya integration supports only FP32')
        self.encoder = ModernBertModel(vllm_config=vllm_config, prefix=f'{prefix}.encoder')
        self.pooler = LayaDecisionPooler(vllm_config.model_config.hf_config.hidden_size)

    def embed_input_ids(self, input_ids):
        return self.encoder.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        return self.encoder(input_ids, positions, intermediate_tensors, inputs_embeds)

    def load_weights(self, weights):
        weights = dict(weights)
        encoder_weights = [(k.removeprefix('encoder.'), v) for k, v in weights.items()
                           if k.startswith('encoder.')]
        loaded = {'encoder.' + k for k in self.encoder.load_weights(encoder_weights)}
        head_weights = {k: v for k, v in weights.items()
                        if k.startswith(('head.', 'type_emb.', 'scorer.'))}
        self.pooler.load_state_dict(head_weights, strict=True)
        loaded.update('pooler.' + k for k in head_weights)
        unexpected = [k for k in weights if not k.startswith(
            ('encoder.', 'head.', 'type_emb.', 'scorer.', 'act_head.', 'temperature'))]
        if unexpected:
            raise ValueError(f'unexpected checkpoint tensor names: {unexpected}')
        return loaded
