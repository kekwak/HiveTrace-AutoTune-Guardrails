"""Decision-only FP32 research benchmark; aggregates only, no public API.

Requires prepared private-inputs.json from laya_dataset_probe prepare.
Separate processes ensure the reference GPU model is not resident with vLLM.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

import torch
from batching import batch_indices, restore_order
from laya_dataset_probe import WORK, DATASETS, get_reference


def decision_forward(model, ids, mask, markers):
    h = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
    types = torch.zeros(ids.shape[0], device=ids.device, dtype=torch.long)
    h = h + model.type_emb(types)[:, None, :]
    if model.head is not None:
        for layer in model.head.layers:
            h = layer(h, src_key_padding_mask=~mask.bool())
    selected = torch.gather(h, 1, markers[:, :, None].expand(-1, -1, h.shape[-1]))
    return model.scorer(selected).squeeze(-1).float()


class DeviceMonitor:
    """Sample total device usage, not exact tensor peaks; same path for both engines."""
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop.is_set():
            started = time.perf_counter()
            try:
                text = subprocess.check_output(['nvidia-smi',
                    '--query-gpu=memory.used,temperature.gpu,clocks.sm,power.draw',
                    '--format=csv,noheader,nounits'], text=True, timeout=5).strip()
                values = [float(x.strip()) for x in text.split(',')]
                self.samples.append(dict(time=started, memory_used_mib=values[0],
                    temperature_c=values[1], clock_mhz=values[2], power_w=values[3]))
            except Exception:
                pass
            self.stop.wait(0.25)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=6)


def main(backend, allocation):
    torch.set_num_threads(2)
    all_rows = json.loads((WORK/'private-inputs.json').read_text())
    rows = []
    for dataset in DATASETS:
        for split in ('safe', 'harm'):
            group = [r for r in all_rows if r['dataset']==dataset and r['split']==split]
            rows.extend(group[(i*len(group))//48] for i in range(48))
    lengths = [len(r['ids']) for r in rows]
    input_hash = hashlib.sha256(json.dumps([r['ids'] for r in rows]).encode()).hexdigest()
    if backend == 'reference':
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        root,ref,cfg = get_reference()
        model = ref.build_model(cfg, encoder_dir=str(root/'encoder'))
        model.load_state_dict(load_file(str(root/'model.safetensors')),strict=True)
        model = model.float().cuda().eval()
        model.encoder.config.reference_compile = False
        pad_id = AutoTokenizer.from_pretrained(root/'tokenizer').pad_token_id
        def run(batch):
            n, length = len(batch), max(len(r['ids']) for r in batch)
            ids = torch.full((n,length),pad_id,dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i,row in enumerate(batch):
                ids[i,:len(row['ids'])] = torch.tensor(row['ids'])
                mask[i,:len(row['ids'])] = 1
            markers = torch.tensor([r['markers'] for r in batch])
            with torch.no_grad():
                z = decision_forward(model, ids.cuda(), mask.cuda(), markers.cuda())
            return z.cpu().tolist()
        configurations = [(b, bucket) for b in (1,4,16,32) for bucket in (False,True)]
    else:
        from vllm import LLM, PoolingParams, ModelRegistry
        ModelRegistry.register_model('LayaForDecisions','laya_vllm_model:LayaForDecisions')
        llm = LLM(model=str(WORK), runner='pooling', pooler_config={'task':'token_classify'},
            dtype='float32',max_model_len=512,enforce_eager=True,gpu_memory_utilization=0.7,
            enable_prefix_caching=False,disable_hybrid_kv_cache_manager=True,
            max_num_batched_tokens=2048,max_num_seqs=16)
        def run(batch):
            out = llm.encode([{'prompt_token_ids':r['ids']} for r in batch],
                pooling_task='token_classify', pooling_params=[PoolingParams(use_activation=False,
                extra_kwargs={'laya':dict(marker_positions=r['markers'],qtype=0)}) for r in batch],
                use_tqdm=False)
            return [r.outputs.data.cpu().flatten().tolist() for r in out]
        configurations = [(b,bucket) for b in (1,4,16,32) for bucket in (False,True)]
    random.Random(1792+int(allocation)).shuffle(configurations)
    trials = []
    def checkpoint():
        (WORK/f'fair-progress-{backend}-{allocation}.json').write_text(json.dumps(
            dict(backend=backend,allocation=allocation,complete=False,trials=trials),indent=2))
    # Sustained common warmup; no assertion that this proves thermal steady state.
    warm_start = time.perf_counter()
    while time.perf_counter()-warm_start < 15:
        run(rows[:4])
    with DeviceMonitor() as monitor:
        for batch_size,bucketed in configurations:
            schedule = batch_indices(lengths,batch_size,bucketed)
            try:
                for indices in schedule[:2]: run([rows[i] for i in indices])
                for repeat in range(3):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    indexed=[]
                    # Include sorting/grouping and restoring order in timing.
                    for indices in batch_indices(lengths,batch_size,bucketed):
                        values=run([rows[i] for i in indices])
                        if len(values)!=len(indices): raise ValueError('output count mismatch')
                        indexed.extend(zip(indices,values))
                    outputs=restore_order(indexed,len(rows))
                    torch.cuda.synchronize()
                    end=time.perf_counter()
                    if not all(len(v)==2 and all(torch.isfinite(torch.tensor(v))) for v in outputs):
                        raise ValueError('invalid logits')
                    # Only this disposable runtime keeps per-record outputs.
                    path=WORK/f'fair-private-{backend}-{batch_size}-{bucketed}.json'
                    path.write_text(json.dumps(outputs))
                    samples=[s for s in monitor.samples if start<=s['time']<=end]
                    trials.append(dict(batch_size=batch_size,bucketed=bucketed,repeat=repeat,
                        records=len(outputs),seconds=end-start,records_per_second=len(outputs)/(end-start),
                        sampled_peak_device_mib=max((s['memory_used_mib'] for s in samples),default=None),
                        telemetry_samples=len(samples)))
                    checkpoint()
            except Exception as exc:
                trials.append(dict(batch_size=batch_size,bucketed=bucketed,error=type(exc).__name__))
                checkpoint()
                torch.cuda.empty_cache()
    gpu_info = subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version',
                                       '--format=csv,noheader'],text=True).strip()
    names=['laya_fair_benchmark.py','laya_dataset_probe.py','laya_vllm_model.py','batching.py']
    report=dict(backend=backend,allocation=allocation,complete=True,dtype='float32',decision_only=True,
        records=len(rows),input_ids_sha256=input_hash,gpu=gpu_info,
        versions={name:importlib.metadata.version(name) for name in ('torch','vllm','transformers')},
        source_sha256={n:hashlib.sha256(Path('/content',n).read_bytes()).hexdigest() for n in names},
        trials=trials,telemetry=monitor.samples)
    (WORK/f'fair-{backend}-{allocation}.json').write_text(json.dumps(report,indent=2))
    print('FAIR_BENCHMARK_FINISHED',backend,allocation,flush=True)


if __name__=='__main__':
    main(sys.argv[1],sys.argv[2])
