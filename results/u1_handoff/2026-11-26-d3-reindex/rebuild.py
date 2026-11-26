"""Rebuild D3 inputs in its pinned environment, preserving corpus and bucket IDs."""
import argparse
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, required=True)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.repo.resolve()))

from u1_benchmark.core import read, sha, write
from u1_benchmark.v4_data import (_bucket_record, _factory, _jsonl, _membership,
                                  _native, _references, verified)

assert importlib.metadata.version('transformers') == '5.17.0'
assert importlib.metadata.version('tokenizers') == '0.23.3'
expected_source = '2dce7a829dab27437d607847318fb96e78175f1cf9870d7b013ec3835749aafb'
rows, old = verified(args.source, expected_source)
assert not args.output.exists()
assert not args.output.resolve().is_relative_to(args.source.resolve())
entries = _references([{'name': 'accuknox', 'config': args.repo / 'configs/v4/accuknox.json'}])
key, entry = next(iter(entries.items()))
values, index, parity = _native(entry, rows, _factory)
old_index = {r['id']: r for r in (json.loads(line) for line in
             (args.source / old['indices'][key]['index_file']).read_text().splitlines())}
changed = [{'id': i, 'fields': sorted(k for k in index[i] if index[i].get(k) != old_index[i].get(k)),
            'old_total_tokens': old_index[i]['total_tokens'], 'total_tokens': index[i]['total_tokens']}
           for i in index if index[i] != old_index[i]]
seen_ids, seen_inputs = set(), set()
for bucket, bounds in old['model_buckets'][key].items():
    ids = read(args.source / bounds['ids_file'])
    assert len(ids) == 400
    _membership(ids, index, bounds, seen_ids, seen_inputs)

args.output.mkdir(parents=True, exist_ok=False)
for name in old['files']:
    target = args.output / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.source / name, target)
manifest = deepcopy(old)
manifest['version'] = 'u1-role-speed-new-families-d3-20261126'
manifest['provenance'] = {'source_manifest_sha256': expected_source,
    'note': 'Same corpus and bucket IDs; D3 inputs rebuilt with Transformers 5.17.0; ProtectAI index unchanged'}
manifest['indices'][key]['parity'] = parity
_jsonl(args.output / manifest['indices'][key]['index_file'], values)
for bucket, bounds in old['model_buckets'][key].items():
    ids = read(args.source / bounds['ids_file'])
    eligible = len({v['input_sha256'] for v in index.values() if v['supported'] and
                    bounds['min'] <= v['total_tokens'] <= bounds['max']})
    manifest['model_buckets'][key][bucket] = _bucket_record(
        args.output, key, bucket, ids, eligible, index, rows,
        {k: bounds[k] for k in ('min', 'max')})
manifest['files'] = {name: sha(args.output / name) for name in sorted(old['files'])}
write(args.output / 'manifest.json', manifest)
new_rows, checked = verified(args.output)
assert new_rows == rows and sha(args.output / 'corpus.jsonl') == old['files']['corpus.jsonl']
assert checked['status'] == 'ready'
assert sha(args.source / 'manifest.json') == expected_source
for name, value in old['files'].items():
    assert sha(args.source / name) == value
    if name != manifest['indices'][key]['index_file']:
        assert sha(args.output / name) == value
report = {'status': 'ready', 'weights_loaded': False, 'gpu_used': False,
          'source_manifest_sha256': expected_source,
          'manifest_sha256': sha(args.output / 'manifest.json'),
          'corpus_sha256': sha(args.output / 'corpus.jsonl'), 'records': len(rows),
          'same_messages_and_ids': True, 'same_bucket_ids': True,
          'protectai_index_unchanged': True, 'parity': parity,
          'changed_records': changed, 'changed_record_count': len(changed)}
write(args.output.parent / (args.output.name + '.validation.json'), report)
print(json.dumps(report))
