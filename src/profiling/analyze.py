"""Reproduce derived kernel tables and annotated timelines from a baseline ZIP.

Standard library only. Does not run models or rewrite measured latency samples.
Usage: python3 src/profiling/analyze.py ARCHIVE.zip RESULTS_DIRECTORY
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import statistics
import zipfile

COLORS = {'attention': '#8b5cf6', 'matmul': '#2563eb', 'copy': '#ea580c',
          'elementwise': '#16a34a', 'softmax': '#db2777', 'norm': '#0891b2',
          'reduce': '#854d0e', 'other': '#64748b'}


def kernel_kind(name):
    name = name.lower()
    for kind, words in [('attention', ('flash', 'fmha', 'attention')),
                        ('matmul', ('gemm', 'gemv', 'matmul', 'cutlass')),
                        ('softmax', ('softmax',)), ('norm', ('norm',)),
                        ('copy', ('copy', 'memcpy')),
                        ('elementwise', ('elementwise', 'vectorized', 'unrolled')),
                        ('reduce', ('reduce',))]:
        if any(word in name for word in words):
            return kind
    return 'other'


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def timeline(events, path, title):
    request = next(e for e in events if e.get('name') == 'request_0' and 'dur' in e)
    start, end = request['ts'], request['ts'] + request['dur']
    cpu = [e for e in events if e.get('cat') == 'cpu_op' and start <= e['ts'] <= end]
    ids = {e.get('args', {}).get('External id') for e in cpu}
    kernels = [e for e in events if e.get('cat') == 'kernel'
               and e.get('args', {}).get('External id') in ids]
    end = max([end] + [e['ts'] + e['dur'] for e in kernels])
    span = end - start
    phases = [e for e in events if 'dur' in e and e.get('name', '')[:2] in ('1_', '2_', '3_')
              and start <= e['ts'] < request['ts'] + request['dur']]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="390" viewBox="0 0 1200 390">',
           '<rect width="1200" height="390" fill="white"/>',
           '<style>text{font:14px sans-serif;fill:#0f172a}.small{font-size:12px}</style>',
           f'<text x="25" y="30" font-weight="bold">{html.escape(title)}</text>',
           '<text x="25" y="55" class="small">First request; derived from original Chrome trace, not a Perfetto UI screenshot.</text>']
    x = lambda t: 180 + 970 * (t-start) / span
    for i in range(6):
        xx = 180 + 970*i/5
        svg.append(f'<path d="M{xx} 80V255" stroke="#e2e8f0"/>')
        svg.append(f'<text x="{xx}" y="275" class="small">{span*i/5000:.1f} ms</text>')
    for row, phase in enumerate(sorted(phases, key=lambda e: e['ts'])[:3]):
        yy = 85 + row*40
        svg.append(f'<text x="25" y="{yy+17}" class="small">{html.escape(phase["name"][:22])}</text>')
        svg.append(f'<rect x="{x(phase["ts"]):.2f}" y="{yy}" width="{max(.8,970*phase["dur"]/span):.2f}" height="22" fill="#cbd5e1"/>')
    svg.append('<text x="25" y="235">CUDA kernels</text>')
    for event in kernels:
        kind = kernel_kind(event['name'])
        svg.append(f'<rect x="{x(event["ts"]):.2f}" y="215" width="{max(.2,970*event["dur"]/span):.2f}" height="28" fill="{COLORS[kind]}"><title>{html.escape(event["name"])}</title></rect>')
    for i, (kind, color) in enumerate(COLORS.items()):
        xx = 25 + i*145
        svg.append(f'<rect x="{xx}" y="300" width="12" height="12" fill="{color}"/><text x="{xx+17}" y="311" class="small">{kind}</text>')
    svg.append('<text x="25" y="345" class="small">CPU spans enqueue asynchronous CUDA work. D2H span may wait for preceding compute; it is NOT isolated copy cost.</text>')
    svg.append('<text x="25" y="370" class="small">Profiler/hooks add overhead. Absolute latency must be read from the separate latency samples.</text></svg>')
    path.write_text('\n'.join(svg))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    by_kind, operator_rows, audits, gpu_rows = [], [], [], []
    with zipfile.ZipFile(args.archive) as z:
        assert z.testzip() is None, 'Archive CRC failure'
        manifest = json.loads(z.read('manifest.json'))
        assert manifest['status'] == 'completed'
        steps = manifest['profile_steps']
        checksums = {name: hashlib.sha256(z.read(name)).hexdigest() for name in sorted(z.namelist())}
        trace_names = sorted((n for n in z.namelist() if n.startswith('trace_L') and n.endswith('.json')),
                             key=lambda n: int(n[7:-5]))
        for name in trace_names:
            L = int(name[7:-5])
            events = json.loads(z.read(name))['traceEvents']
            kernels = [e for e in events if e.get('cat') == 'kernel']
            assert kernels, name
            cpu_ops = {e.get('args', {}).get('External id'): e['name'] for e in events if e.get('cat') == 'cpu_op'}
            kernel_groups, kinds, ops = defaultdict(float), defaultdict(float), defaultdict(float)
            missing = 0
            for e in kernels:
                kind = kernel_kind(e['name'])
                kernel_groups[(e['name'], kind)] += e['dur']
                kinds[kind] += e['dur']
                op = cpu_ops.get(e.get('args', {}).get('External id'), 'unattributed')
                missing += op == 'unattributed'
                ops[op] += e['dur']
            total = sum(kinds.values())
            intervals = sorted((e['ts'], e['ts'] + e['dur']) for e in kernels)
            begin, finish = intervals[0]
            union = 0
            for left, right in intervals[1:]:
                if left > finish:
                    union += finish - begin
                    begin, finish = left, right
                else:
                    finish = max(finish, right)
            union += finish - begin
            markers = [e for e in events if e.get('name', '').startswith('forward_') and 'dur' in e]
            window_start = min([e['ts'] for e in markers] + [e['ts'] for e in kernels])
            window_end = max([e['ts'] + e['dur'] for e in markers + kernels])
            gpu_rows.append({'requested_text_tokens': L, 'kernel_sum_ms': total/1000/steps,
                             'kernel_union_ms': union/1000/steps, 'kernels_per_forward': len(kernels)/steps,
                             'profile_window_ms_per_forward': (window_end-window_start)/1000/steps,
                             'profile_gpu_busy_pct': 100*union/(window_end-window_start)})
            by_kind.extend({'requested_text_tokens': L, 'kind': k, 'kernel_time_pct': 100*v/total}
                           for k, v in sorted(kinds.items()))
            top = [{'kernel': n, 'kind': k, 'dur_us': us, 'us_per_forward': us/steps}
                   for (n, k), us in sorted(kernel_groups.items(), key=lambda v: -v[1])[:30]]
            write_csv(args.output / f'top_kernels_L{L}.csv', top)
            operator_rows.extend({'requested_text_tokens': L, 'cpu_operator': op,
                                  'kernel_ms_per_forward': dur/1000/steps}
                                 for op, dur in sorted(ops.items(), key=lambda x: -x[1]))
            audits.append({'L': L, 'kernels': len(kernels), 'unattributed_kernels': missing})
        write_csv(args.output / 'kernel_breakdown.csv', by_kind)
        write_csv(args.output / 'operator_kernel_breakdown.csv', operator_rows)
        write_csv(args.output / 'gpu_profile_recomputed.csv', gpu_rows)
        original_gpu = {r['requested_text_tokens']: r for r in read_csv_archive(z, 'gpu_profile.csv')}
        for row in gpu_rows:
            original = original_gpu[str(row['requested_text_tokens'])]
            for key in row:
                if key != 'requested_text_tokens':
                    assert abs(row[key]-float(original[key])) < .001, (key, row)
        figures = args.output / 'figures'
        figures.mkdir(exist_ok=True)
        for L in (16, 512):
            events = json.loads(z.read(f'viz_safe_L{L}.json'))['traceEvents']
            timeline(events, figures / f'timeline_L{L}.svg', f'{manifest["model"]}: L{L} baseline timeline')
        samples = list(csv.DictReader(io.StringIO(z.read('latency_samples.csv').decode())))
        summary = list(csv.DictReader(io.StringIO(z.read('latency.csv').decode())))
        for row in summary:
            for mode in ('forward', 'e2e'):
                values = [float(s['ms']) for s in samples if s['requested_text_tokens'] == row['requested_text_tokens'] and s['mode'] == mode]
                assert len(values) == manifest['iterations']
                assert abs(statistics.median(values)-float(row[f'{mode}_p50_ms'])) < 1e-6
                assert abs(statistics.quantiles(values, n=100, method='inclusive')[94]-float(row[f'{mode}_p95_ms'])) < 1e-6
        record = {'archive_sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                  'archive_members_sha256': checksums, 'trace_attribution': audits,
                  'verified_samples': len(samples), 'kernel_category_rule': 'copy/memcpy; nocast is arithmetic'}
        (args.output / 'trace_audit.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f'{manifest["model"]}: {len(samples)} samples verified; {len(trace_names)} traces attributed; timelines regenerated')


def read_csv_archive(z, name):
    return list(csv.DictReader(io.StringIO(z.read(name).decode())))


if __name__ == '__main__':
    main()
