"""Run on Colab before benchmarking the decision-only reference."""
import torch
from laya_dataset_probe import get_reference
from laya_fair_benchmark import decision_forward
from safetensors.torch import load_file
from transformers import AutoTokenizer

root, ref, cfg = get_reference()
model = ref.build_model(cfg, encoder_dir=str(root/'encoder'))
model.load_state_dict(load_file(str(root/'model.safetensors')), strict=True)
model = model.float().cuda().eval()
model.encoder.config.reference_compile = False
tok = AutoTokenizer.from_pretrained(root/'tokenizer')
question = dict(t='choice', ins='Is this harmful?', crit={'unsafe': 'harmful', 'safe': 'safe'})
ids, markers = ref.build_sequence(tok, 'Hello!', question, 512, 192)
tokens = torch.tensor([ids], device='cuda')
positions = torch.tensor([markers], device='cuda')
with torch.no_grad():
    expected, _ = model(tokens, torch.ones_like(tokens), positions,
        torch.ones_like(positions, dtype=torch.bool), torch.zeros(1, device='cuda', dtype=torch.long))
    actual = decision_forward(model, tokens, torch.ones_like(tokens), positions)
torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
print('DECISION_ONLY_PARITY_PASS', flush=True)
