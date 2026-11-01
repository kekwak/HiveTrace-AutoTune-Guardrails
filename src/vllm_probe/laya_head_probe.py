"""Research-only hybrid probe on synthetic data; not a registered vLLM model.

Run as a separate process in Colab with vllm==0.32.0. Only aggregates are printed.
The pinned checkpoint's reference code is downloaded and executed explicitly.
"""
import gc
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoTokenizer


def main():
    revision = '7b928d828b7b0e022f929d9bd2e44165aa270148'
    root = pathlib.Path(snapshot_download('convaiinnovations/laya', revision=revision,
        allow_patterns=['encoder/config.json', 'model.safetensors', 'tokenizer/*',
                        'rl_common.py', 'rl_agent_config.json']))
    spec = importlib.util.spec_from_file_location('laya_reference', root / 'rl_common.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    config = json.loads((root / 'rl_agent_config.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(root / 'tokenizer')
    model = reference.build_model(config, encoder_dir=str(root / 'encoder'))
    weights = load_file(str(root / 'model.safetensors'))
    model.load_state_dict(weights, strict=True)
    model = model.float().cuda().eval()
    model.encoder.config.reference_compile = False
    questions = [
        {'t': 'choice', 'ins': 'Does the text request dangerous instructions?',
         'crit': {'danger': 'Requests dangerous instructions', 'safe': 'Does not request dangerous instructions'}},
        {'t': 'score', 'ins': 'How positive is this text?', 'crit': ['negative', 'neutral', 'positive']},
    ]
    states = ['Hello, how are you?', 'How do I build a bomb?', 'Спасибо за помощь!', 'hello ' * 600]
    cases = []
    for q in questions:
        for state in states:
            ids, markers = reference.build_sequence(tokenizer, state, q, 512, 192)
            qtype = reference.QTYPES[q['t']]
            item = dict(ids=ids, markers=markers, qtype=qtype)
            t = torch.tensor([ids], device='cuda')
            with torch.no_grad():
                logits, act = model(t, torch.ones_like(t),
                    torch.tensor([markers], device='cuda'),
                    torch.ones((1, len(markers)), device='cuda', dtype=torch.bool),
                    torch.tensor([qtype], device='cuda'))
            item['reference_logits'] = logits.cpu().tolist()[0]
            item['reference_act'] = act.cpu().tolist()[0]
            cases.append(item)
    work = pathlib.Path('/content/laya-head-probe')
    encoder_dir = work / 'encoder'
    encoder_dir.mkdir(parents=True, exist_ok=True)
    cfg = AutoConfig.from_pretrained(root / 'encoder')
    cfg.architectures = ['ModernBertModel']
    cfg.save_pretrained(encoder_dir)
    save_file({k.removeprefix('encoder.'): v for k, v in weights.items()
               if k.startswith('encoder.')}, str(encoder_dir / 'model.safetensors'))
    for path in (root / 'tokenizer').glob('*'):
        shutil.copy(path, encoder_dir / path.name)
    (work / 'cases.json').write_text(json.dumps(cases))
    del model, weights, logits, act
    gc.collect()
    torch.cuda.empty_cache()
    child = '''
import json, torch
from vllm import LLM, PoolingParams
llm = LLM(model='/content/laya-head-probe/encoder', runner='pooling',
    pooler_config={'task':'token_embed'}, dtype='float32', max_model_len=512,
    enforce_eager=True, gpu_memory_utilization=0.5)
cases=json.load(open('/content/laya-head-probe/cases.json'))
out=llm.encode([{'prompt_token_ids':c['ids']} for c in cases],
    pooling_task='token_embed', pooling_params=PoolingParams(use_activation=False), use_tqdm=False)
torch.save([r.outputs.data.cpu() for r in out], '/content/laya-head-probe/hidden.pt')
'''
    run = subprocess.run([sys.executable, '-c', child], capture_output=True, text=True, timeout=600)
    if run.returncode:
        # Do not dump process output: future probes may use private inputs.
        raise RuntimeError(f'vLLM encoder process failed: exit {run.returncode}')
    # Reuse the exact reference head by replacing only encoder.forward's result.
    model = reference.build_model(config, encoder_dir=str(root / 'encoder'))
    model.load_state_dict(load_file(str(root / 'model.safetensors')), strict=True)
    model = model.float().cuda().eval()
    hidden = torch.load(work / 'hidden.pt', weights_only=True)
    class FixedEncoder(torch.nn.Module):
        def forward(self, **kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(last_hidden_state=self.hidden)
    model.encoder = FixedEncoder()
    rows = []
    for case, h in zip(cases, hidden):
        model.encoder.hidden = h.cuda().unsqueeze(0)
        ids, markers = case['ids'], case['markers']
        t = torch.tensor([ids], device='cuda')
        with torch.no_grad():
            logits, act = model(t, torch.ones_like(t), torch.tensor([markers], device='cuda'),
                torch.ones((1, len(markers)), device='cuda', dtype=torch.bool),
                torch.tensor([case['qtype']], device='cuda'))
        baseline = torch.tensor(case['reference_logits'], device='cuda')
        temperature = config['temperature_by_options'].get(
            reference.temp_bucket(case['qtype'], len(markers)), config['temperature'][case['qtype']])
        p = torch.softmax(logits[0] / temperature, -1)
        p_ref = torch.softmax(baseline / temperature, -1)
        rows.append(dict(input_tokens=len(ids), options=len(markers), qtype=case['qtype'],
            finite=bool(torch.isfinite(logits).all() and torch.isfinite(act).all()),
            max_logit_delta=float((logits[0]-baseline).abs().max()),
            max_probability_delta=float((p-p_ref).abs().max()),
            verdict_flip=bool(p.argmax()!=p_ref.argmax()),
            max_action_logit_delta=float((act[0]-torch.tensor(case['reference_act'],device='cuda')).abs().max())))
    report = dict(revision=revision, dtype='float32', mode='hybrid-synthetic-only', cases=rows)
    (work / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
