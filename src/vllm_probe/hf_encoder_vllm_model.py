"""Dense native HF encoder inside vLLM 0.32.0, not a fused encoder backend.

Preserves DeBERTa relative positions and classifier/pooler semantics. Uses eager
full-sequence batches and CPU position inspection, hence no CUDA-graph claim.
The hidden mode supports GLiNER's unchanged external schema heads.
"""
import torch
from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification
from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.model_executor.models.interfaces_base import default_pooling_type

from .encoder_contracts import sequence_lengths


class NativeEncoderPooler(Pooler):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.pooling = AllPool()

    def get_supported_tasks(self):
        return {'token_classify'}

    def get_pooling_updates(self, task):
        return PoolingParamsUpdate()

    def forward(self, hidden_states, pooling_metadata):
        results = []
        for hidden in self.pooling(hidden_states, pooling_metadata):
            if hidden is None:
                results.append(None)
            elif self.mode == 'classify':
                results.append(hidden[0].float().softmax(-1))
            else:
                results.append(hidden)
        return results


@default_pooling_type(tok_pooling_type='ALL')
class NativeHFEncoder(nn.Module):
    is_pooling_model = True
    attn_type = 'encoder_only'

    def __init__(self, *, vllm_config, prefix=''):
        super().__init__()
        config = vllm_config.model_config.hf_config
        if vllm_config.model_config.dtype != torch.float32:
            raise ValueError('Native encoder experiment requires FP32')
        if config.model_type not in ('deberta', 'deberta-v2'):
            raise ValueError('Only audited DeBERTa architectures supported')
        self.mode = getattr(config, 'vllm_encoder_mode', 'classify')
        if self.mode not in ('classify', 'hidden'):
            raise ValueError('Unknown native encoder mode')
        config._attn_implementation = 'eager'
        loader = AutoModelForSequenceClassification if self.mode == 'classify' else AutoModel
        self.hf_model = loader.from_config(config, trust_remote_code=False).eval()
        self.pooler = NativeEncoderPooler(self.mode)
        self.pad_token_id = config.pad_token_id or 0

    def embed_input_ids(self, input_ids):
        return self.hf_model.get_input_embeddings()(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if intermediate_tensors is not None or inputs_embeds is not None:
            raise ValueError('Pipeline parallelism and external embeddings not supported')
        lengths = sequence_lengths(positions.detach().cpu().tolist())
        pieces = input_ids.split(lengths)
        padded = nn.utils.rnn.pad_sequence(pieces, batch_first=True, padding_value=self.pad_token_id)
        sizes = torch.tensor(lengths, device=input_ids.device)
        mask = (torch.arange(padded.shape[1], device=input_ids.device)[None, :] < sizes[:, None]).long()
        output = self.hf_model(input_ids=padded, attention_mask=mask, return_dict=True)
        if self.mode == 'classify':
            return output.logits.repeat_interleave(sizes, dim=0)
        return torch.cat([output.last_hidden_state[i, :length] for i, length in enumerate(lengths)])

    def load_weights(self, weights):
        weights = dict(weights)
        self.hf_model.load_state_dict(weights, strict=True)
        return {'hf_model.' + name for name in weights}
