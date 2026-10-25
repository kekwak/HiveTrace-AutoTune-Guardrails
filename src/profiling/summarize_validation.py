"""Import downloaded follow-up ZIPs and summarize independent kernel runs.

Usage: python3 src/profiling/summarize_validation.py DOWNLOAD_DIRECTORY RESULTS_ROOT
Source ZIPs must be named laya-run-N.zip / shieldgemma-2b-run-N.zip.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import statistics
import zipfile


def read_csv(z, name):
    return list(csv.DictReader(io.StringIO(z.read(name).decode())))


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('downloads', type=Path)
    p.add_argument('results', type=Path)
    args = p.parse_args()
    for model in ('laya', 'shieldgemma-2b'):
        root = args.results / model / 'validation'
        root.mkdir(parents=True, exist_ok=True)
        groups = defaultdict(list)
        telemetry = []
        checksums = []
        for run in range(3):
            source = args.downloads / f'{model}-run-{run}.zip'
            with zipfile.ZipFile(source) as z:
                assert z.testzip() is None
                manifest = json.loads(z.read('manifest.json'))
                assert manifest['status'] == 'completed' and manifest['run'] == run
                target = root / f'run-{run}'
                target.mkdir(exist_ok=True)
                # Keep full raw evidence including targeted A/B/actual-e2e traces after clone.
                shutil.copyfile(source, target / 'artifacts.zip')
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                checksums.append(f'{digest}  run-{run}/artifacts.zip')
                for name in z.namelist():
                    if Path(name).name == name and not name.startswith('ab_') and name.endswith(('.csv', '.json')) and name != 'actual_e2e.json':
                        (target / name).write_bytes(z.read(name))
                    elif name in ('ab_summary.csv', 'ab_samples.csv', 'ab_failures.json'):
                        (target / name).write_bytes(z.read(name))
                samples = read_csv(z, 'repeat_samples.csv')
                rows = read_csv(z, 'repeat_summary.csv')
                assert len(samples) == 400 and len(rows) == 8
                for row in rows:
                    values = [float(s['ms']) for s in samples if s['case'] == row['case'] and s['mode'] == row['mode']]
                    assert len(values) == 50
                    assert abs(statistics.median(values)-float(row['p50_ms'])) < 1e-6
                    assert abs(statistics.quantiles(values, n=100, method='inclusive')[94]-float(row['p95_ms'])) < 1e-6
                    groups[(row['case'], row['mode'])].append(row)
                for row in read_csv(z, 'telemetry.csv'):
                    values = next(csv.reader([row['gpu_csv']])) if row['returncode'] == '0' else []
                    if len(values) == 7:
                        telemetry.append({'run': run, 'timestamp': values[0].strip(),
                                          'temperature_C': values[1].strip(), 'sm_clock_MHz': values[2].strip(),
                                          'mem_clock_MHz': values[3].strip(), 'power_W': values[4].strip(),
                                          'gpu_util_pct': values[5].strip(), 'used_memory_MiB': values[6].strip()})
        rows = []
        for (case, mode), members in sorted(groups.items()):
            p50 = [float(r['p50_ms']) for r in members]
            p95 = [float(r['p95_ms']) for r in members]
            assert len(members) == 3
            rows.append({'case': case, 'mode': mode, 'independent_kernels': 3,
                         'input_tokens': members[0]['input_tokens'], 'truncated': members[0]['truncated'],
                         'median_run_p50_ms': statistics.median(p50),
                         'min_run_p50_ms': min(p50), 'max_run_p50_ms': max(p50),
                         'min_run_p95_ms': min(p95), 'max_run_p95_ms': max(p95)})
        save_csv(root / 'repeat_aggregate.csv', rows)
        save_csv(root / 'telemetry.csv', telemetry)
        (root / 'SHA256SUMS').write_text('\n'.join(checksums) + '\n')
        print(model, ': 3 kernels, 1200 raw samples verified; source ZIPs and summaries imported')


if __name__ == '__main__':
    main()
