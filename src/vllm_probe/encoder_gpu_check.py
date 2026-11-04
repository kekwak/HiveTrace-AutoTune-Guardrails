"""Bounded-run worker: actual FP32 classifier parity on a private small sample.

Run as python -m src.vllm_probe.encoder_gpu_check MODEL INPUT_JSON OUTPUT_DIR.
Only the aggregate report should be downloaded. Input JSON is a list with
text, dataset, split fields. Raw text/predictions are never printed.
"""
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from .adapter_contracts import MODELS, classify_scores, compare_results
from .pooling_adapter import PoolingAdapter


def run(name, input_path, output):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if name not in ('koala', 'llama'):
        raise ValueError('Classifier worker only')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise ValueError('Requires CUDA T4')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    spec = MODELS[name]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(input_path).read_text())
    tok = AutoTokenizer.from_pretrained(spec['repo'], revision=spec['revision'])
    ids = [tok(row['text'], truncation=False)['input_ids'] for row in rows]
    accepted = [i for i, values in enumerate(ids) if 0 < len(values) <= spec['limit']]
    model = AutoModelForSequenceClassification.from_pretrained(
        spec['repo'], revision=spec['revision'], dtype=torch.float32,
        attn_implementation='eager').eval().cuda()
    labels = [str(model.config.id2label[i]) for i in range(model.config.num_labels)]
    artifact = output / (name + '-artifact')
    # Save unchanged native weights with a vLLM architecture selector.
    model.save_pretrained(artifact)
    config = copy.deepcopy(model.config)
    config.architectures = ['NativeHFEncoder']
    config.vllm_encoder_mode = 'classify'
    config.save_pretrained(artifact)
    tok.save_pretrained(artifact)
    references = [None] * len(rows)
    start = time.perf_counter()
    with torch.inference_mode():
        for index in accepted:
            tensor = torch.tensor([ids[index]], device='cuda')
            probs = model(input_ids=tensor, attention_mask=torch.ones_like(tensor)).logits[0].float().softmax(-1)
            references[index] = classify_scores(probs.cpu().tolist(), labels)
    reference_seconds = time.perf_counter() - start
    reference_summary = dict(model=name, checkpoint=spec, total=len(rows),
                             evaluated=len(accepted), rejected=len(rows)-len(accepted),
                             labels=labels, finite=True,
                             splits={})
    for key in sorted({(r['dataset'], r['split']) for r in rows}):
        indices = [i for i, row in enumerate(rows) if (row['dataset'], row['split']) == key]
        reference_summary['splits']['/'.join(key)] = dict(
            total=len(indices), evaluated=sum(references[i] is not None for i in indices))
    (output / (name+'-reference-summary.json')).write_text(json.dumps(reference_summary, indent=2))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    adapter = PoolingAdapter.create(name, artifact, labels)
    candidates = [None] * len(rows)
    start = time.perf_counter()
    for offset in range(0, len(accepted), 4):
        indices = accepted[offset:offset+4]
        scores = adapter.score([ids[i] for i in indices])
        for i, score in zip(indices, scores):
            candidates[i] = score
    candidate_seconds = time.perf_counter() - start
    groups = {}
    for key in sorted({(r['dataset'], r['split']) for r in rows}):
        indices = [i for i, row in enumerate(rows) if (row['dataset'], row['split']) == key]
        groups['/'.join(key)] = compare_results([references[i] for i in indices], [candidates[i] for i in indices])
    report = dict(model=name, checkpoint=spec, labels=labels, records=len(rows),
                  input_sha256=hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
                  gpu=torch.cuda.get_device_name(0), splits=groups,
                  versions={p: importlib.metadata.version(p) for p in ('vllm', 'torch', 'transformers')},
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path(__file__).parent.glob('*.py')},
                  pooled=compare_results(references, candidates),
                  rejected_over_limit=len(rows)-len(accepted),
                  one_pass_seconds={'reference_singleton': reference_seconds, 'candidate_batch4': candidate_seconds},
                  limitations=['one-pass timings are not a controlled speed comparison',
                               'custom dense HF encoder under vLLM scheduling, not fused vLLM attention',
                               'output records are standalone response diagnostics; original request unavailable'])
    (output / (name+'-report.json')).write_text(json.dumps(report, indent=2, allow_nan=False))
    print('ENCODER_CHECK_FINISHED', name, len(accepted), flush=True)


if __name__ == '__main__':
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    run(*sys.argv[1:])
