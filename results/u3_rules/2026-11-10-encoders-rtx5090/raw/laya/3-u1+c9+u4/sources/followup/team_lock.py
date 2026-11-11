"""Build an audited wheel lock from a successful isolated smoke; no installer/GPU code."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
WORKERS = ('cuda', 'ort_cuda', 'gliner', 'halo')
HOSTS = {'files.pythonhosted.org', 'pypi.nvidia.com', 'download.pytorch.org', 'download-r2.pytorch.org'}


def read(path):
    return json.loads(Path(path).read_text())


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def normalized(packages):
    result = {}
    for name, version in packages.items():
        key = canonical(name)
        if key in result:
            raise ValueError('Duplicate normalized package: ' + key)
        result[key] = version
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_manifest(run):
    run = Path(run)
    status = read(run / 'status.json')
    if status.get('status') != 'passed' or not status.get('environment_validated'):
        raise ValueError('A completed environment smoke is required')
    report = read(run / 'installation/install.json')
    environment = report['environment']
    if (environment['sys_platform'], environment['platform_machine'], environment['python_version']) != (
            'linux', 'x86_64', '3.13'):
        raise ValueError('Expected Linux x86_64 CPython 3.13 wheels')
    packages = []
    for row in report['install']:
        name, version = canonical(row['metadata']['name']), row['metadata']['version']
        download = row['download_info']
        url = download['url']
        parsed = urlsplit(url)
        checksum = download.get('archive_info', {}).get('hashes', {}).get('sha256', '')
        wheel = unquote(parsed.path).rsplit('/', 1)[-1]
        if (parsed.scheme != 'https' or parsed.hostname not in HOSTS or parsed.username or parsed.password
                or parsed.query or parsed.fragment or not wheel.endswith('.whl')
                or not re.fullmatch(r'[a-f0-9]{64}', checksum) or row.get('is_yanked')):
            raise ValueError('Unpinned/unexpected artifact: ' + name)
        parts = wheel.split('-')
        if canonical(parts[0]) != name or parts[1] != version:
            raise ValueError('Wheel metadata disagrees: ' + name)
        packages.append({'name': name, 'version': version, 'url': url, 'sha256': checksum})
    if not packages or len({p['name'] for p in packages}) != len(packages):
        raise ValueError('Empty or duplicate package inventory')
    expected = {p['name']: p['version'] for p in packages}
    worker_hashes = {}
    for stage in WORKERS:
        if status['stages'][stage]['status'] != 'passed':
            raise ValueError('Incomplete worker: ' + stage)
        path = run / 'stages' / stage / 'packages.json'
        if normalized(read(path)) != expected:
            raise ValueError('Install report and worker inventory differ: ' + stage)
        worker_hashes[stage] = sha(path)
    return {'format_version': 1, 'run': run.name, 'package_count': len(packages),
            'source_install_report_sha256': sha(run / 'installation/install.json'),
            'worker_inventory_sha256': worker_hashes, 'source_environment': environment,
            'source_pip_version': report['pip_version'],
            'source_smoke_passed': True, 'lock_redeployment_verified': False,
            'gpu_performance_verified': False, 'quality_evaluated': False,
            'packages': sorted(packages, key=lambda p: p['name'])}


def requirements(manifest):
    lines = ['# Generated from ' + manifest['run'] + '; Linux x86_64 / CPython 3.13 only.',
             '# Exact wheels; install into a fresh pip-less venv with --no-deps --require-hashes.',
             '# Original smoke passed; redeployment from this lock has NOT been verified.']
    for p in manifest['packages']:
        lines.extend(['# ' + p['name'] + '==' + p['version'],
                      f"{p['name']} @ {p['url']} --hash=sha256:{p['sha256']}"])
    return '\n'.join(lines) + '\n'


def verify_inventory(manifest, actual):
    expected = {p['name']: p['version'] for p in manifest['packages']}
    actual = normalized(actual)
    if actual != expected:
        raise ValueError(json.dumps({'missing': sorted(expected.keys() - actual.keys()),
            'extra': sorted(actual.keys() - expected.keys()),
            'different': {k: {'expected': v, 'actual': actual[k]} for k, v in expected.items()
                          if k in actual and actual[k] != v}}, sort_keys=True))


def verify_python(manifest, python):
    code = '''import importlib.metadata as m, json, sys, platform
assert sys.prefix != sys.base_prefix, "Use an isolated venv"
assert sys.platform == "linux" and platform.machine() == "x86_64"
assert sys.version_info[:2] == (3, 13)
print(json.dumps({p.metadata["Name"]: p.version for p in m.distributions() if p.metadata.get("Name")}))
'''
    result = subprocess.run([str(python), '-c', code], check=True, capture_output=True, text=True, timeout=60)
    if result.stderr:
        print(result.stderr, end='')
    verify_inventory(manifest, json.loads(result.stdout))
    print(f"Exact inventory matches: {manifest['package_count']} packages. GPU compatibility not retested.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    build.add_argument('run', type=Path)
    verify = sub.add_parser('verify')
    verify.add_argument('--python', required=True, type=Path)
    args = parser.parse_args()
    destination = ROOT / 'env/team_environment.lock.json'
    if args.command == 'build':
        manifest = make_manifest(args.run)
        destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
        (ROOT / 'env/team_environment.lock.txt').write_text(requirements(manifest))
        print(f"Locked {manifest['package_count']} packages; all worker inventories match.")
    else:
        verify_python(read(destination), args.python)
