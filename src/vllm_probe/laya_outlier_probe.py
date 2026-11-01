"""Private-input numerical diagnosis; writes only aggregate summaries publicly."""
import json
from pathlib import Path
import sys
import torch
from laya_dataset_probe import WORK,get_reference
from laya_fair_benchmark import decision_forward


def main(backend):
    torch.set_num_threads(2)
    rows=json.loads((WORK/'private-inputs.json').read_text())
    index=json.loads((WORK/'private-outlier-indices.json').read_text())[0]
    target=rows[index]
    outputs={}
    if backend=='reference':
        from safetensors.torch import load_file
        root,ref,cfg=get_reference()
        model=ref.build_model(cfg,encoder_dir=str(root/'encoder'))
        model.load_state_dict(load_file(str(root/'model.safetensors')),strict=True)
        model=model.float().cuda().eval();model.encoder.config.reference_compile=False
        def run(batch):
            length=max(len(r['ids']) for r in batch)
            ids=torch.zeros((len(batch),length),device='cuda',dtype=torch.long)
            mask=torch.zeros_like(ids)
            for i,r in enumerate(batch):
                ids[i,:len(r['ids'])]=torch.tensor(r['ids'],device='cuda');mask[i,:len(r['ids'])]=1
            with torch.no_grad():
                return decision_forward(model,ids,mask,torch.tensor([r['markers'] for r in batch],device='cuda')).cpu().tolist()
        original_start=index//4*4
        outputs['original_group']=run(rows[original_start:original_start+4])[index-original_start]
    else:
        from vllm import LLM,PoolingParams,ModelRegistry
        ModelRegistry.register_model('LayaForDecisions','laya_vllm_model:LayaForDecisions')
        llm=LLM(model=str(WORK),runner='pooling',pooler_config={'task':'token_classify'},dtype='float32',
            max_model_len=512,enforce_eager=True,gpu_memory_utilization=0.7,enable_prefix_caching=False,
            disable_hybrid_kv_cache_manager=True,max_num_batched_tokens=2048,max_num_seqs=16)
        def run(batch):
            out=llm.encode([{'prompt_token_ids':r['ids']} for r in batch],pooling_task='token_classify',
                pooling_params=[PoolingParams(use_activation=False,extra_kwargs={'laya':dict(marker_positions=r['markers'],qtype=0)}) for r in batch],use_tqdm=False)
            return [r.outputs.data.cpu().flatten().tolist() for r in out]
        original_start=index//32*32
        outputs['original_group']=run(rows[original_start:original_start+32])[index-original_start]
    outputs['singleton']=run([target])[0]
    outputs['singleton_repeat']=run([target])[0]
    outputs['duplicate_group']=run([target]*4)[0]
    # Stay private on disposable VM; summary is produced separately.
    (WORK/f'private-outlier-{backend}.json').write_text(json.dumps(outputs))
    print('OUTLIER_PROBE_FINISHED',backend)


if __name__=='__main__':main(sys.argv[1])
