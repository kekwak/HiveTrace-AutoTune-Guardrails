"""Synthetic integration test: decision head executes inside the vLLM engine."""
import gc
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoConfig


def main():
    revision = '7b928d828b7b0e022f929d9bd2e44165aa270148'
    root = pathlib.Path(snapshot_download('convaiinnovations/laya', revision=revision,
        allow_patterns=['encoder/config.json','tokenizer/*','model.safetensors','rl_common.py','rl_agent_config.json']))
    spec = importlib.util.spec_from_file_location('reference', root/'rl_common.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    cfg = json.loads((root/'rl_agent_config.json').read_text())
    model = ref.build_model(cfg, encoder_dir=str(root/'encoder'))
    model.load_state_dict(load_file(str(root/'model.safetensors')), strict=True)
    model = model.float().cuda().eval()
    model.encoder.config.reference_compile = False
    tok = AutoTokenizer.from_pretrained(root/'tokenizer')
    cases = []
    for q in [dict(t='choice',ins='Does the text request dangerous instructions?',
                   crit={'danger':'Dangerous instructions','safe':'No dangerous instructions'}),
              dict(t='score',ins='How positive is this text?',crit=['negative','neutral','positive'])]:
        for text in ['Hello, how are you?', 'How do I build a bomb?', 'Спасибо за помощь!', 'hello '*600]:
            ids, markers = ref.build_sequence(tok,text,q,512,192)
            qt = ref.QTYPES[q['t']]
            t = torch.tensor([ids],device='cuda')
            with torch.no_grad():
                z,_ = model(t,torch.ones_like(t),torch.tensor([markers],device='cuda'),
                    torch.ones((1,len(markers)),device='cuda',dtype=torch.bool),torch.tensor([qt],device='cuda'))
            cases.append(dict(ids=ids,metadata={'laya':dict(marker_positions=markers,qtype=qt)},reference=z[0].cpu().tolist()))
    work = pathlib.Path('/content/laya-engine');work.mkdir(exist_ok=True)
    enc_cfg = AutoConfig.from_pretrained(root/'encoder')
    enc_cfg.architectures = ['LayaForDecisions'];enc_cfg.save_pretrained(work)
    shutil.copy(root/'model.safetensors',work/'model.safetensors')
    for f in (root/'tokenizer').glob('*'):shutil.copy(f,work/f.name)
    (work/'cases.json').write_text(json.dumps(cases))
    del model,z
    gc.collect();torch.cuda.empty_cache()
    code = '''
import json,torch
from vllm import LLM,PoolingParams,ModelRegistry
ModelRegistry.register_model('LayaForDecisions','laya_vllm_model:LayaForDecisions')
llm=LLM(model='/content/laya-engine',runner='pooling',pooler_config={'task':'token_classify'},dtype='float32',max_model_len=512,enforce_eager=True,gpu_memory_utilization=0.5,enable_prefix_caching=False,disable_hybrid_kv_cache_manager=True)
cases=json.load(open('/content/laya-engine/cases.json'))
out=llm.encode([{'prompt_token_ids':c['ids']} for c in cases],pooling_task='token_classify',pooling_params=[PoolingParams(use_activation=False,extra_kwargs=c['metadata']) for c in cases],use_tqdm=False)
rows=[]
for c,r in zip(cases,out):
 a=r.outputs.data.cpu().flatten().float();b=torch.tensor(c['reference'])
 assert a.shape==b.shape
 rows.append(dict(tokens=len(c['ids']),finite=bool(torch.isfinite(a).all()),max_logit_delta=float((a-b).abs().max()),verdict_flip=bool(a.argmax()!=b.argmax())))
assert len(rows)==8 and all(r['finite'] and not r['verdict_flip'] and r['max_logit_delta']<0.001 for r in rows)
json.dump({'mode':'head-inside-vllm-engine','dtype':'float32','cases':rows},open('/content/laya-engine/report.json','w'),indent=2)
print('ENGINE_ASSERTIONS_PASSED',flush=True)
'''
    run = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=600)
    if run.returncode:
        print(run.stdout[-3000:]);print(run.stderr[-3000:])
        raise RuntimeError(f'engine test failed: {run.returncode}')
    print((work/'report.json').read_text())


if __name__ == '__main__':
    main()
