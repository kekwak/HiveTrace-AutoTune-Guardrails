"""Real small-sample Laya/GLiNER checks; download aggregate artifacts only."""
import copy
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

from .adapter_contracts import MODELS
from .custom_heads import GlinerSafetyAdapter, LayaChoicesAdapter, laya_choices
from .pooling_adapter import PoolingAdapter
from .small_check_reporting import parity_report


def run(name, input_path, output_path):
    import torch
    from huggingface_hub import snapshot_download
    if name not in ('gliner', 'laya'):
        raise ValueError('Custom-head worker only')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise ValueError('Requires CUDA T4')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(input_path).read_text())
    spec = MODELS[name]
    reference, candidate = [None]*len(rows), [None]*len(rows)
    statuses = ['pending']*len(rows)
    info = dict(model=name, checkpoint=spec, gpu=torch.cuda.get_device_name(0),
                input_sha256=hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
                versions={p: importlib.metadata.version(p) for p in ('vllm', 'torch', 'transformers')},
                source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in Path(__file__).parent.glob('*.py')},
                limitations=['smoke parity, not ground-truth accuracy or controlled throughput',
                             'output is standalone response diagnostic; original request absent',
                             'Laya decision-only FP32 profile; GLiNER dense encoder has host-copy bridge'])

    def checkpoint(complete=False):
        report = dict(**info, **parity_report(rows, reference, candidate, statuses, complete))
        temporary = output/(name+'-report.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(output/(name+'-report.json'))

    def reference_summary():
        groups = {}
        for key in sorted({(r['dataset'], r['split']) for r in rows}):
            indices = [i for i, r in enumerate(rows) if (r['dataset'], r['split']) == key]
            groups['/'.join(key)] = dict(total=len(indices), evaluated=sum(reference[i] is not None for i in indices))
        (output/(name+'-reference-summary.json')).write_text(json.dumps(dict(
            model=name, total=len(rows), evaluated=sum(r is not None for r in reference), splits=groups), indent=2))

    checkpoint()
    try:
        if name == 'laya':
            from safetensors.torch import load_file
            from transformers import AutoConfig, AutoTokenizer
            info['stage'] = 'load_reference'
            root = Path(snapshot_download(spec['repo'], revision=spec['revision'],
                allow_patterns=['encoder/config.json', 'tokenizer/*', 'model.safetensors',
                                'rl_common.py', 'rl_agent_config.json']))
            module_spec = importlib.util.spec_from_file_location('laya_pinned_reference', root/'rl_common.py')
            ref = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(ref)
            cfg = json.loads((root/'rl_agent_config.json').read_text())
            temperature = cfg['temperature_by_options']['choice:2']
            model = ref.build_model(cfg, encoder_dir=str(root/'encoder'))
            model.load_state_dict(load_file(str(root/'model.safetensors')), strict=True)
            model = model.float().cuda().eval()
            model.encoder.config.reference_compile = False
            tok = AutoTokenizer.from_pretrained(root/'tokenizer')
            question = dict(t='choice', ins='Does this text contain harmful, dangerous, or policy-violating content?',
                            crit={'unsafe': 'Harmful, dangerous, or policy-violating content',
                                  'safe': 'Safe content without a policy violation'})
            labels = list(question['crit'])
            info.update(question=question, labels=labels, temperature=temperature,
                        truncation='none; build full sequence then reject >512')
            inputs, metadata, accepted = {}, {}, []
            logits = tensor = None
            info['stage'] = 'reference'
            for i, row in enumerate(rows):
                ids, markers = ref.build_sequence(tok, row['text'], question,
                                                  max(512, len(row['text'])*8+4096), 192)
                if len(ids) > 512:
                    statuses[i] = 'over_context'
                    continue
                inputs[i], metadata[i] = ids, dict(marker_positions=markers, qtype=0)
                tensor = torch.tensor([ids], device='cuda')
                with torch.inference_mode():
                    logits, _ = model(tensor, torch.ones_like(tensor),
                                      torch.tensor([markers], device='cuda'),
                                      torch.ones((1, len(markers)), device='cuda', dtype=torch.bool),
                                      torch.tensor([0], device='cuda'))
                reference[i] = laya_choices(logits[0, :len(markers)].cpu().tolist(), labels, temperature)
                statuses[i] = 'reference_ok'
                accepted.append(i)
            reference_summary()
            checkpoint()
            artifact = output/'laya-artifact'
            artifact.mkdir(exist_ok=True)
            enc_cfg = AutoConfig.from_pretrained(root/'encoder')
            enc_cfg.architectures = ['LayaForDecisions']
            enc_cfg.save_pretrained(artifact)
            shutil.copyfile(root/'model.safetensors', artifact/'model.safetensors')
            tok.save_pretrained(artifact)
            del model, logits, tensor
            gc.collect()
            torch.cuda.empty_cache()
            info['stage'] = 'candidate'
            checkpoint()
            adapter = LayaChoicesAdapter(PoolingAdapter.create(name, artifact), labels, temperature)
            for offset in range(0, len(accepted), 4):
                indices = accepted[offset:offset+4]
                scores = adapter.score([inputs[i] for i in indices], [metadata[i] for i in indices])
                for i, score in zip(indices, scores):
                    candidate[i], statuses[i] = score, 'ok'
                checkpoint()
        else:
            from gliner2 import GLiNER2
            info['versions']['gliner2'] = importlib.metadata.version('gliner2')
            info['stage'] = 'load_reference'
            root = snapshot_download(spec['repo'], revision=spec['revision'],
                allow_patterns=['README.md', 'config.json', 'encoder_config/config.json',
                                'tokenizer*', 'special_tokens_map.json', 'model.safetensors'])
            model = GLiNER2.from_pretrained(root, attention_backend='standard', quantize=False,
                                           compile=False, map_location='cpu').float().cuda().eval()
            model.strict_extraction = True
            native = GlinerSafetyAdapter(model, encoder_adapter=None)
            accepted = []
            info['stage'] = 'reference'
            for i, row in enumerate(rows):
                try:
                    batch = model.processor.collate_fn_inference(
                        [(row['text'], native.schema.build())], max_len=None,
                        error_policy='raise', architecture='span')
                except ValueError:
                    statuses[i] = 'processor_error'
                    continue
                if batch.original_texts != [row['text']]:
                    statuses[i] = 'processor_changed_text'
                    continue
                if int(batch.attention_mask.sum()) > 512:
                    statuses[i] = 'over_context'
                    continue
                reference[i] = native.score([row['text']])[0]
                statuses[i] = 'reference_ok'
                accepted.append(i)
            reference_summary()
            checkpoint()
            artifact = output/'gliner-artifact'
            model.encoder.save_pretrained(artifact)
            config = copy.deepcopy(model.encoder.config)
            config.architectures = ['NativeHFEncoder']
            config.vllm_encoder_mode = 'hidden'
            config.save_pretrained(artifact)
            model.processor.tokenizer.save_pretrained(artifact)
            info['stage'] = 'candidate'
            checkpoint()
            adapter = GlinerSafetyAdapter(model, PoolingAdapter.create(name, artifact))
            try:
                for offset in range(0, len(accepted), 4):
                    indices = accepted[offset:offset+4]
                    scores = adapter.score([rows[i]['text'] for i in indices])
                    for i, score in zip(indices, scores):
                        candidate[i], statuses[i] = score, 'ok'
                    checkpoint()
            finally:
                adapter.close()
        info['stage'] = 'complete'
        checkpoint(complete=True)
        print('CUSTOM_CHECK_FINISHED', name, flush=True)
    except Exception as error:
        info['error_type'] = type(error).__name__
        checkpoint()
        raise


if __name__ == '__main__':
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    run(*sys.argv[1:])
