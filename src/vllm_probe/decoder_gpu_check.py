"""Small private-sample native-vs-vLLM checks for all four decoder models.

Download only aggregate reports. Generation retains global argmax/categories;
pair scoring retains native threshold. Shield scores all four official policies.
"""
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .adapter_contracts import MODELS, classify_scores, parse_verdict
from .custom_heads import ShieldPoliciesAdapter, shield_policies
from .decoder_adapter import DecoderAdapter
from .native_preparation import author_prompt, official_policies, pair_token_ids, prepare_decoder
from .shield_scoring import probability_yes
from .small_check_reporting import parity_report


def run(name, input_path, output_path):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if name not in ('hive', 'open', 'halo', 'shield'):
        raise ValueError('Decoder worker only')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise ValueError('Requires CUDA T4')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = json.loads(Path(input_path).read_text())
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    spec = MODELS[name]
    reference, candidate = [None]*len(rows), [None]*len(rows)
    statuses = ['pending']*len(rows)
    started = time.monotonic()
    phase_times = {}
    info = dict(model=name, checkpoint=spec, gpu=torch.cuda.get_device_name(0),
                input_sha256=hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
                versions={p: importlib.metadata.version(p) for p in ('vllm', 'torch', 'transformers')},
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in Path(__file__).parent.glob('*.py')},
                limitations=['smoke parity only; not ground-truth accuracy or controlled throughput',
                             'output lacks original request: Shield diagnostic, other decoders explicitly unsupported',
                             'no quantization or precision substitution'])

    def checkpoint(complete=False):
        elapsed = round(time.monotonic()-started, 3)
        phase_times.setdefault(info.get('stage', 'initial'), elapsed)
        info.update(elapsed_seconds=elapsed, phase_started_seconds=dict(phase_times))
        report = dict(**info, **parity_report(rows, reference, candidate, statuses, complete))
        temporary = output/(name+'-report.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(output/(name+'-report.json'))

    checkpoint()
    try:
        info['stage'] = 'prepare'
        checkpoint()
        tok = AutoTokenizer.from_pretrained(spec['repo'], revision=spec['revision'])
        tok.model_max_length = 10**9
        options = {}
        policies = None
        if name in ('halo', 'shield'):
            card = Path(hf_hub_download(spec['repo'], 'README.md', revision=spec['revision'])).read_text()
            if name == 'halo':
                options['system_prompt'] = author_prompt(card)
            else:
                policies = official_policies(card)
                info['policies'] = policies
        if name == 'open':
            path = hf_hub_download(spec['template_repo'], 'chat_template.jinja', revision=spec['template_revision'])
            options['template'] = Path(path).read_text()
        pair = pair_token_ids(name, tok) if name in ('halo', 'shield') else None
        prepared, accepted = {}, []
        for i, row in enumerate(rows):
            output_only = row['dataset'].endswith('output')
            if output_only and name != 'shield':
                statuses[i] = 'unsupported_output_role'
                continue
            if name == 'shield':
                ids = [prepare_decoder(name, tok, row['text'], guideline=p['output' if output_only else 'input'],
                                       output=output_only) for p in policies]
            else:
                ids = [prepare_decoder(name, tok, row['text'], **options)]
            if max(map(len, ids)) + spec.get('max_new_tokens', 1) > spec['limit']:
                statuses[i] = 'over_context'
                continue
            prepared[i] = ids
            accepted.append(i)
        checkpoint()
        info['stage'] = 'reference'
        checkpoint()
        model = AutoModelForCausalLM.from_pretrained(spec['repo'], revision=spec['revision'],
            dtype=getattr(torch, spec['dtype']), attn_implementation='eager').cuda().eval()
        info['stage'] = 'reference_inference'
        checkpoint()
        eos = model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos] if eos is not None else []
        good = []
        for i in accepted:
            try:
                values = []
                for ids in prepared[i]:
                    tensor = torch.tensor([ids], device='cuda')
                    with torch.inference_mode():
                        if pair is not None:
                            logits = model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                                           use_cache=False, logits_to_keep=1).logits[0, -1].float()
                            safe, unsafe = logits[list(pair)].cpu().tolist()
                            p = probability_yes(unsafe, safe)
                            values.append(classify_scores([1-p, p], ['safe', 'unsafe'], threshold=spec['threshold']))
                        else:
                            tokens = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                                do_sample=False, max_new_tokens=spec['max_new_tokens'])[0, len(ids):].cpu().tolist()
                            if name == 'open' and len(tokens) >= spec['max_new_tokens'] and tokens[-1] not in eos_ids:
                                statuses[i] = 'reference_truncated'
                                break
                            values.append(parse_verdict(tok.decode(tokens, skip_special_tokens=True), name))
                if statuses[i] == 'reference_truncated':
                    continue
                reference[i] = (shield_policies([v['scores']['unsafe'] for v in values])
                                if name == 'shield' else values[0])
                statuses[i] = 'reference_ok'
                good.append(i)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                statuses[i] = 'reference_oom'
            except ValueError:
                statuses[i] = 'reference_invalid_output'
            checkpoint()
        (output/(name+'-reference-summary.json')).write_text(json.dumps(dict(
            model=name, total=len(rows), evaluated=len(good), statuses={s: statuses.count(s) for s in set(statuses)}), indent=2))
        del model
        # These tensors may retain the final forward output; drop before engine load.
        tensor = logits = None
        gc.collect()
        torch.cuda.empty_cache()
        info['stage'] = 'candidate'
        checkpoint()
        if name == 'shield':
            result = subprocess.run([sys.executable, '-m', 'src.vllm_probe.shield_t4_patch'],
                                    capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise RuntimeError('Shield runtime patch failed; no caps disabled')
            info['patch'] = json.loads(Path('/content/shield-patch-manifest.json').read_text())
        info['stage'] = 'candidate_engine_startup'
        checkpoint()
        decoder = DecoderAdapter.create(name, pair_ids=pair)
        adapter = ShieldPoliciesAdapter(decoder) if name == 'shield' else decoder
        info['stage'] = 'candidate_inference'
        checkpoint()
        for i in good:
            try:
                candidate[i] = (adapter.score([prepared[i]])[0] if name == 'shield'
                                else adapter.score(prepared[i])[0])
                statuses[i] = 'ok'
            except ValueError:
                statuses[i] = 'candidate_invalid_output'
            checkpoint()
        info['stage'] = 'complete'
        checkpoint(complete=True)
        print('DECODER_CHECK_FINISHED', name, flush=True)
    except Exception as error:
        info['error_type'] = type(error).__name__
        checkpoint()
        raise


if __name__ == '__main__':
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    run(*sys.argv[1:])
