"""Attribute targeted Laya A/B copy kernels by CPU op and recorded tensor shape.

Usage: python3 src/profiling/analyze_ab.py VALIDATION_RUN_0.zip OUTPUT_DIRECTORY
The targeted traces contain three warmed forward calls each, not latency samples.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import zipfile


def save(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    shapes, summary = [], []
    with zipfile.ZipFile(args.archive) as z:
        for name in sorted(z.namelist()):
            if not (name.startswith('ab_') and '_L' in name and name.endswith('.json')):
                continue
            mode, length = name[3:-5].rsplit('_L', 1)
            events = json.loads(z.read(name))['traceEvents']
            cpu = {e.get('args', {}).get('External id'): e for e in events if e.get('cat') == 'cpu_op'}
            kernels = [e for e in events if e.get('cat') == 'kernel']
            groups = defaultdict(lambda: [0, 0.0])
            for event in kernels:
                op = cpu.get(event.get('args', {}).get('External id'), {})
                if op.get('name') != 'aten::copy_':
                    continue
                dims = op.get('args', {}).get('Input Dims', [[]])[0]
                key = json.dumps(dims)
                groups[key][0] += 1
                groups[key][1] += event['dur']
            for shape, (count, dur) in sorted(groups.items(), key=lambda x: -x[1][1]):
                shapes.append({'mode': mode, 'L': int(length), 'shape': shape,
                               'copy_kernels_per_forward': count/3, 'copy_kernel_ms_per_forward': dur/3000})
            summary.append({'mode': mode, 'L': int(length), 'kernels_per_forward': len(kernels)/3,
                            'kernel_ms_per_forward': sum(e['dur'] for e in kernels)/3000,
                            'to_copy_calls_per_forward': sum(e['name'] == 'aten::_to_copy' for e in cpu.values())/3,
                            'copy_kernel_ms_per_forward': sum(v[1] for v in groups.values())/3000})
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / 'ab_copy_shapes.csv', shapes)
    save(args.output / 'ab_profile_summary.csv', summary)
    print('Attributed', len(summary), 'targeted traces; these profiler times are NOT ordinary latency')


if __name__ == '__main__':
    main()
