"""One controlled CUDA loading probe per fresh process; never grants approval."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--environment', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--route', choices=('native', 'local', 'legacy-dtype', 'loading-info',
                                           'prepared-inputs', 'adapter'), required=True)
    parser.add_argument('--loading', choices=('default', 'serial'), default='default')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.repo.resolve()))
    # Explicit experimental factors, set before importing either CUDA or HF.
    os.environ.update(CUDA_VISIBLE_DEVICES='0', CUDA_LAUNCH_BLOCKING='1',
                      OMP_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8', USE_HUB_KERNELS='NO',
                      HF_DEACTIVATE_ASYNC_LOAD='1' if args.loading == 'serial' else '0')
    result = {'diagnostic_only': True, 'approval': False, 'status': 'running',
              'route': args.route, 'loading': args.loading, 'cuda_launch_blocking': True,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def save(phase):
        result['phase'] = phase
        temporary = args.output / 'result.tmp'
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(args.output / 'result.json')

    try:
        save('environment')
        import torch
        from u1_benchmark.core import digest
        from u1_benchmark.v4_evidence import configure_gpu
        from u1_benchmark.v4_preparation import source_path, validate_settings
        from u1_benchmark.v4_adapters import load_adapter
        from u1_benchmark.v4_models import hf_load_options
        result['runtime'] = configure_gpu(args.environment)
        spec = validate_settings(json.loads(args.config.read_text())['baseline'])
        result['variant_sha256'] = digest(spec)
        texts = ['Library hours?', 'The public library is open.']
        if spec['adapter'] == 'd3_decision' and args.route not in ('native', 'adapter'):
            raise ValueError('d3 probe supports only native or adapter route')
        save('load')
        if args.route == 'adapter':
            adapter = load_adapter(spec, args.output / 'adapter')
            torch.cuda.synchronize()
            result['metadata'] = adapter.metadata
            save('request')
            result['responses'] = adapter.request_batch(texts)
        elif spec['adapter'] == 'hf_sequence_classification':
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from u1_benchmark.family_preparation import ClassificationPreparer
            weights = spec['weights']
            if args.route == 'native':
                location = weights['repo']
                source_options = {'revision': weights['revision']}
            else:
                location = str(source_path(weights))
                source_options = {}
            loader_options = {'dtype': torch.float32}
            if args.route == 'legacy-dtype':
                loader_options = {'torch_dtype': torch.float32}
            elif args.route in ('loading-info', 'prepared-inputs'):
                loader_options = hf_load_options(spec['execution'])
            loaded = AutoModelForSequenceClassification.from_pretrained(
                location, **source_options, **loader_options)
            model = loaded[0] if isinstance(loaded, tuple) else loaded
            save('to_cuda')
            model = model.to('cuda:0').eval()
            torch.cuda.synchronize()
            save('prepare_inputs')
            if args.route == 'prepared-inputs':
                p = ClassificationPreparer(spec)
                inputs = p.tokenizer.pad([p.prepare(text) for text in texts], padding=True, return_tensors='pt')
            else:
                t = spec['tokenizer']
                tokenizer = AutoTokenizer.from_pretrained(t['repo'], revision=t['revision'])
                inputs = tokenizer(texts, padding=True, truncation=False, return_tensors='pt')
            result['inputs'] = {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                                      'sha256': digest(value.tolist())} for key, value in inputs.items()}
            inputs = {key: value.to('cuda:0') for key, value in inputs.items()}
            torch.cuda.synchronize()
            save('request')
            with torch.inference_mode():
                logits = model(**inputs).logits
                torch.cuda.synchronize()
                result['logits'] = logits.cpu().tolist()
                if not bool(logits.isfinite().all()):
                    raise ValueError('Nonfinite logits')
        elif spec['adapter'] == 'd3_decision':
            from u1_benchmark.d3_family import D3Preparer, _check_asset_directory, _token_files, _weight_inventory
            p = D3Preparer(spec)
            files = _token_files(p.manifest) + ['decision_config.json', 'config.json']
            root = source_path(spec['weights'], files + ['*.safetensors', '*.safetensors.index.json'])
            _check_asset_directory(root, files, p.manifest)
            result['weights'] = _weight_inventory(root, spec['weights'])
            native = p.runtime.D3(root, device='cuda:0', batch_size=1,
                max_length=p.limit, readout_dtype='float32', permutation_average=False)
            torch.cuda.synchronize()
            save('request')
            prepared = [p.native_prepare(text)[0] for text in texts]
            result['probabilities'] = native.probabilities(
                [v.sequences['decision'] for v in prepared], [len(v.questions['decision'].keys) for v in prepared])
        else:
            raise ValueError('Unsupported diagnostic family')
        torch.cuda.synchronize()
        result['status'] = 'passed'
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, traceback=traceback.format_exc())
    finally:
        save(result.get('phase', 'startup'))
    print(json.dumps({k: result[k] for k in ('status', 'route', 'loading', 'phase')}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
