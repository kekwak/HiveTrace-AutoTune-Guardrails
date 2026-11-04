"""Rebuild a committed sample selection from private local Parquet snapshots.

No network or GPU allocation. CLI output must be outside this repository;
private texts are never printed. The JSON format matches the measured workers.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

DATASETS = {'robustness-test-input', 'robustness-test-input-robust', 'robustness-test-output'}


def validate_output_path(path):
    path = Path(path).resolve()
    repository = subprocess.run(['git', '-C', str(path.parent), 'rev-parse', '--show-toplevel'],
                                capture_output=True, text=True)
    if path.is_relative_to(Path(__file__).resolve().parents[2]) or repository.returncode == 0:
        raise ValueError('Write the private sample outside any Git working tree')


def prepare_sample(dataset_root, manifest):
    import pyarrow.parquet as pq
    dataset_root = Path(dataset_root)
    snapshots = json.loads((dataset_root/'download-manifest.json').read_text())
    revisions = {item['repo'].removeprefix('hivetrace/'): item['revision']
                 for item in snapshots['datasets']}
    groups = manifest['selection']
    expected = {(name, split) for name in DATASETS for split in ('safe', 'harm')}
    if len(groups) != 6 or {(g['dataset'], g['split']) for g in groups} != expected:
        raise ValueError('Expected exactly the six pinned dataset/split groups')
    rows = []
    for group in groups:
        name, split = group['dataset'], group['split']
        if revisions.get(name) != group['revision']:
            raise ValueError('Local dataset revision differs from the sample manifest')
        indices = group['indices']
        if (not indices or len(set(indices)) != len(indices)
                or any(type(i) is not int or not 0 <= i < group['total'] for i in indices)):
            raise ValueError('Invalid or duplicate sample indices')
        field = 'response' if name.endswith('output') else 'request'
        table = pq.read_table(dataset_root/name/'data'/f'{split}-00000-of-00001.parquet', columns=[field])
        if len(table) != group['total']:
            raise ValueError('Local split count differs from the sample manifest')
        for index in indices:
            text = table[field][index].as_py()
            if not isinstance(text, str):
                raise ValueError('Dataset text must be a string')
            rows.append(dict(text=text, dataset=name, split=split))
    if len(rows) != manifest['records']:
        raise ValueError('Sample record count differs from the manifest')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    validate_output_path(args.output)
    rows = prepare_sample(args.dataset_root, json.loads(args.manifest.read_text()))
    payload = json.dumps(rows)
    with args.output.open('x', encoding='utf-8') as output:
        output.write(payload)
    print(json.dumps(dict(records=len(rows), sha256=hashlib.sha256(payload.encode()).hexdigest())))


if __name__ == '__main__':
    main()
