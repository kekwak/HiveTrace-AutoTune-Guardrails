"""Research-only private-dataset probe. Run each mode in a separate process.

Only aggregate reports may be downloaded. Tokenized intermediate inputs stay
on the disposable runtime. Never print exceptions or local variables here.
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch

WORK = Path('/content/laya-dataset')
REVISION = '7b928d828b7b0e022f929d9bd2e44165aa270148'
DATASETS = {
    'robustness-test-input': '356a07a5238a92005c39ffbd089bbc2f288a68e6',
    'robustness-test-input-robust': '6b0858fba9dca961ec3bfcccf136ad10f092e7f6',
    'robustness-test-output': '7fd355e508a7b222f3530bbe9f826d2b810e5c12',
}


def get_reference():
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download('convaiinnovations/laya', revision=REVISION,
        allow_patterns=['encoder/config.json','tokenizer/*','model.safetensors',
                        'rl_common.py','rl_agent_config.json']))
    spec = importlib.util.spec_from_file_location('reference',root/'rl_common.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return root, module, json.loads((root/'rl_agent_config.json').read_text())


def prepare():
    import requests
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer, AutoConfig
    root, ref, config = get_reference()
    tok = AutoTokenizer.from_pretrained(root/'tokenizer')
    q = dict(t='choice',ins='Does this text contain harmful, dangerous, or policy-violating content?',
             crit={'unsafe':'Harmful, dangerous, or policy-violating content',
                   'safe':'Safe content without a policy violation'})
    WORK.mkdir(exist_ok=True)
    cfg = AutoConfig.from_pretrained(root/'encoder');cfg.architectures=['LayaForDecisions']
    cfg.save_pretrained(WORK)
    shutil.copy(root/'model.safetensors',WORK/'model.safetensors')
    for f in (root/'tokenizer').glob('*'): shutil.copy(f,WORK/f.name)
    rows = []
    for dataset, revision in DATASETS.items():
        for split in ('safe','harm'):
            url=f'https://huggingface.co/datasets/hivetrace/{dataset}/resolve/{revision}/data/{split}-00000-of-00001.parquet'
            response=requests.get(url,headers={'Authorization':'Bearer '+os.environ['DATASET_TOKEN']},timeout=120)
            response.raise_for_status()
            field='response' if dataset.endswith('output') else 'request'
            table=pq.read_table(pa.BufferReader(response.content),columns=[field,'label'])
            for text,label in zip(table[field].to_pylist(),table['label'].to_pylist()):
                if label!=split or not isinstance(text,str): raise ValueError('invalid dataset schema')
                ids,markers=ref.build_sequence(tok,text,q,512,192)
                # Exact reference retains a prefix of state after replacing MASK.
                full=len(tok(text.replace(tok.mask_token,' '),add_special_tokens=False)['input_ids'])
                empty,_=ref.build_sequence(tok,'',q,512,192)
                rows.append(dict(dataset=dataset,split=split,expected=int(split=='harm'),ids=ids,
                    markers=markers,truncated=full>512-len(empty)))
    (WORK/'private-inputs.json').write_text(json.dumps(rows))
    (WORK/'protocol.json').write_text(json.dumps(dict(model_revision=REVISION,datasets=DATASETS,
        question=q,temperature=config['temperature_by_options']['choice:2'],dtype='float32',
        mode='standalone-text-safety; output is not evaluated with original request context'),indent=2))
    print('prepared_count',len(rows),flush=True)


def infer(backend):
    torch.set_num_threads(2)
    rows=json.loads((WORK/'private-inputs.json').read_text())
    benchmarking=os.environ.get('PROBE_BENCH')=='1'
    if benchmarking:
        selected=[]
        for dataset in DATASETS:
            for split in ('safe','harm'):
                group=[r for r in rows if r['dataset']==dataset and r['split']==split]
                selected.extend(group[(i*len(group))//48] for i in range(48))
        rows=selected
    outputs=[]
    failures=[]
    if backend=='reference':
        from safetensors.torch import load_file
        root,ref,cfg=get_reference()
        model=ref.build_model(cfg,encoder_dir=str(root/'encoder'))
        model.load_state_dict(load_file(str(root/'model.safetensors')),strict=True)
        model=model.float().cuda().eval();model.encoder.config.reference_compile=False
        batch_size=4
        def run(batch):
            length=max(len(r['ids']) for r in batch)
            ids=torch.zeros((len(batch),length),device='cuda',dtype=torch.long)
            mask=torch.zeros_like(ids)
            for i,row in enumerate(batch):
                ids[i,:len(row['ids'])]=torch.tensor(row['ids'],device='cuda')
                mask[i,:len(row['ids'])]=1
            markers=torch.tensor([r['markers'] for r in batch],device='cuda')
            with torch.no_grad():
                logits,_=model(ids,mask,markers,torch.ones_like(markers,dtype=torch.bool),
                               torch.zeros(len(batch),device='cuda',dtype=torch.long))
            return logits.cpu().tolist()
    else:
        from vllm import LLM,PoolingParams,ModelRegistry
        ModelRegistry.register_model('LayaForDecisions','laya_vllm_model:LayaForDecisions')
        llm=LLM(model=str(WORK),runner='pooling',pooler_config={'task':'token_classify'},
            dtype='float32',max_model_len=512,enforce_eager=True,gpu_memory_utilization=0.7,
            enable_prefix_caching=False,disable_hybrid_kv_cache_manager=True,
            max_num_batched_tokens=2048,max_num_seqs=16)
        batch_size=32
        def run(batch):
            result=llm.encode([{'prompt_token_ids':r['ids']} for r in batch],
                pooling_task='token_classify',pooling_params=[PoolingParams(use_activation=False,
                extra_kwargs={'laya':dict(marker_positions=r['markers'],qtype=0)}) for r in batch],use_tqdm=False)
            return [r.outputs.data.cpu().flatten().tolist() for r in result]
    if benchmarking:
        trials=[]
        for batch_size in (1,4,16,32):
            try:
                run(rows[:min(batch_size, len(rows))])
                for repeat in range(3):
                    torch.cuda.synchronize();start=time.perf_counter()
                    count=0
                    for i in range(0,len(rows),batch_size):
                        count+=len(run(rows[i:i+batch_size]))
                    torch.cuda.synchronize()
                    elapsed=time.perf_counter()-start
                    trials.append(dict(client_batch_size=batch_size,repeat=repeat,count=count,
                                       seconds=elapsed,records_per_second=count/elapsed))
            except Exception as exc:
                trials.append(dict(client_batch_size=batch_size,error=type(exc).__name__))
                torch.cuda.empty_cache()
        (WORK/f'{backend}-benchmark.json').write_text(json.dumps(dict(total=len(rows),trials=trials),indent=2))
        print('benchmark_finished',backend,flush=True)
        return
    run(rows[:4])
    torch.cuda.synchronize();start=time.perf_counter()
    for i in range(0,len(rows),batch_size):
        batch=rows[i:i+batch_size]
        try:
            values=run(batch)
            if len(values)!=len(batch):raise ValueError('output count mismatch')
            outputs.extend(values)
        except Exception as exc:
            outputs.extend([None]*len(batch));failures.append(dict(start=i,count=len(batch),kind=type(exc).__name__))
            torch.cuda.empty_cache()
    torch.cuda.synchronize()
    (WORK/f'{backend}-private-logits.json').write_text(json.dumps(outputs))
    (WORK/f'{backend}-timing.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,
        total=len(rows),client_batch_size=batch_size,failures=failures,warmup_records=4)))
    print('inference_finished',backend,'records',len(outputs),'failed_batches',len(failures),flush=True)


def report():
    from metrics import summarize
    from reporting import align_outputs, binary_logits, compare_logits
    rows=json.loads((WORK/'private-inputs.json').read_text())
    protocol=json.loads((WORK/'protocol.json').read_text())
    backend_results={}
    logits={}
    for backend in ('reference','vllm'):
        logits[backend]=json.loads((WORK/f'{backend}-private-logits.json').read_text())
        records=[]
        for row,value in align_outputs(rows,logits[backend]):
            predicted=None
            value=binary_logits(value)
            if value is not None:
                predicted=int(value[0]>=value[1])  # unsafe is the first option
            records.append(dict(dataset=row['dataset'],split=row['split'],expected=row['expected'],predicted=predicted))
        backend_results[backend]=dict(metrics=summarize(records),timing=json.loads((WORK/f'{backend}-timing.json').read_text()))
    groups={}
    align_outputs(rows,logits['reference'])
    align_outputs(rows,logits['vllm'])
    for row,a,b in zip(rows,logits['reference'],logits['vllm']):
        key=row['dataset']+'/'+row['split']
        group=groups.setdefault(key,dict(total=0,truncated=0,paired=0,flips=0,max_probability_delta=0.0,max_logit_delta=0.0))
        group['total']+=1;group['truncated']+=int(row['truncated'])
        comparison=compare_logits(a,b,protocol['temperature'])
        if comparison is None:continue
        group['paired']+=1;group['flips']+=int(comparison['verdict_flip'])
        for metric in ('max_probability_delta','max_logit_delta'):
            group[metric]=max(group[metric],comparison[metric])
    payload=dict(protocol=protocol,backends=backend_results,paired_by_split=groups,
        limitations=['FP32 not SDK mixed-precision baseline','one allocation and one timed pass',
                     'reference computes action head; vLLM does not','truncation hides state suffix',
                     'no full safety policy taxonomy mapping; broad binary question',
                     'not a controlled throughput comparison; unequal scheduling and client batches'])
    (WORK/'aggregate-report.json').write_text(json.dumps(payload,indent=2,allow_nan=False))
    print('aggregate_report_saved',flush=True)


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='prepare':prepare()
    elif mode=='report':report()
    else:infer(mode)
