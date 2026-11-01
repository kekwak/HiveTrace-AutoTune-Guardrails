"""Small offline CPU fixtures; no historical tests, notebooks or checkpoints."""


def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "safe", "unsafe",
                  "safety", "The", "public", "library", "opens", "every", "morning", "."]
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(vocabulary)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
                                  cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")


def tiny_encoder_config(tokenizer):
    from transformers import DebertaV2Config

    return DebertaV2Config(
        vocab_size=len(tokenizer), hidden_size=24, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=48, max_position_embeddings=512, relative_attention=True,
        position_buckets=16, pos_att_type=["p2c", "c2p"], hidden_dropout_prob=0,
        attention_probs_dropout_prob=0)


def tiny_gliner_adapter():
    import torch
    from gliner2 import GLiNER2
    from gliner2.configuration import ExtractorConfig
    from experiments.adapters import GlinerAdapter

    torch.set_num_threads(1)
    torch.manual_seed(42)
    tokenizer = tiny_tokenizer()
    model = GLiNER2(ExtractorConfig(model_name="unused-local", max_width=2),
                    encoder_config=tiny_encoder_config(tokenizer), tokenizer=tokenizer,
                    use_flashdeberta=False).eval()
    adapter = GlinerAdapter.__new__(GlinerAdapter)
    adapter.model = model
    adapter.schema = model.create_schema().classification(task="safety", labels=["safe", "unsafe"], multi_label=False)
    adapter.tokenizer = model.processor.tokenizer
    adapter.device = torch.device("cpu")
    adapter.spec = {"dtype": "float32"}
    adapter.contract = {"fixture": "random CPU only"}
    adapter.metadata = {"fixture": "random CPU only"}
    adapter.tensor_call = lambda inputs: model.encoder(**inputs).last_hidden_state
    return adapter


def tiny_laya_agent():
    import threading
    import torch
    from laya.agent import Agent
    from laya.common import DecisionModel
    from transformers import DebertaV2Model

    torch.set_num_threads(1)
    torch.manual_seed(42)
    tokenizer = tiny_tokenizer()
    agent = Agent.__new__(Agent)
    agent.tok = tokenizer
    agent.cfg = {"max_len": 512, "head_max_len": 192}
    agent.model = DecisionModel(DebertaV2Model(tiny_encoder_config(tokenizer)),
                                head_layers=0, dropout=0).eval()
    agent.model_id = "random-offline-cpu-fixture"
    agent.device = torch.device("cpu")
    agent.dtype = torch.float32
    agent.amp_enabled = False
    agent._fast = None
    agent.hooks = []
    agent.hooks_raise = True
    agent._hooks_lock = threading.RLock()
    agent.temperature = [1.0, 1.0, 1.0]
    agent.temperature_by_options = {}
    agent.lang_temperatures = {}
    return agent
