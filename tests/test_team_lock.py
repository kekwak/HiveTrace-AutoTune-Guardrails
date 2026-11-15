import copy
import json
from pathlib import Path
import tempfile
import unittest

from followup import team_lock as lock


class TeamLockTests(unittest.TestCase):
    def fixture(self, root):
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        write(root / 'status.json', {'status': 'passed', 'environment_validated': True,
                                    'stages': {s: {'status': 'passed'} for s in lock.WORKERS}})
        report = {'pip_version': '24.1.2', 'environment': {'sys_platform': 'linux',
                  'platform_machine': 'x86_64', 'python_version': '3.13'},
                  'install': [{'metadata': {'name': 'fixture_pkg', 'version': '1.0'},
                               'download_info': {'url': 'https://files.pythonhosted.org/fixture_pkg-1.0-py3-none-any.whl',
                                                 'archive_info': {'hashes': {'sha256': 'a' * 64}}}}]}
        write(root / 'installation/install.json', report)
        for s in lock.WORKERS:
            write(root / 'stages' / s / 'packages.json', {'fixture-pkg': '1.0'})
        return report

    def test_checked_in_lock_exact_and_scoped(self):
        manifest = lock.read(lock.ROOT / 'env/team_environment.lock.json')
        self.assertEqual(manifest['package_count'], 96)
        self.assertEqual(len(manifest['packages']), 96)
        self.assertTrue(manifest['source_smoke_passed'])
        self.assertFalse(manifest['lock_redeployment_verified'])
        self.assertEqual(set(manifest['worker_inventory_sha256']), set(lock.WORKERS))
        text = (lock.ROOT / 'env/team_environment.lock.txt').read_text()
        self.assertEqual(text, lock.requirements(manifest))
        self.assertEqual(text.count('--hash=sha256:'), 96)
        self.assertNotIn('sitecustomize', text)
        packages = {p['name']: p['version'] for p in manifest['packages']}
        spec = lock.read(lock.ROOT / 'env/team_base.json')
        from followup.team_smoke import pins
        for name, version in pins(spec).items():
            self.assertEqual(packages[lock.canonical(name)], version)
        for p in manifest['packages']:
            self.assertRegex(p['sha256'], r'^[a-f0-9]{64}$')

    def test_generation_verifies_every_worker_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            manifest = lock.make_manifest(root)
            self.assertEqual(lock.make_manifest(root), manifest)
            self.assertEqual(manifest['source_install_report_sha256'], lock.sha(root / 'installation/install.json'))
            for s in lock.WORKERS:
                path = root / 'stages' / s / 'packages.json'
                path.write_text('{"fixture-pkg": "2.0"}')
                with self.assertRaisesRegex(ValueError, 'inventory differ'):
                    lock.make_manifest(root)
                path.write_text('{"fixture-pkg": "1.0"}')

    def test_missing_hash_bad_platform_or_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = self.fixture(root)
            changes = [lambda r: r['install'][0]['download_info']['archive_info']['hashes'].clear(),
                       lambda r: r['environment'].update(sys_platform='darwin'),
                       lambda r: r['install'][0]['download_info'].update(url='https://example.com/package.whl'),
                       lambda r: r['install'][0]['metadata'].update(version='2.0'),
                       lambda r: r['install'].append(copy.deepcopy(r['install'][0]))]
            for change in changes:
                changed = copy.deepcopy(report)
                change(changed)
                (root / 'installation/install.json').write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    lock.make_manifest(root)

    def test_failed_smoke_cannot_be_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / 'status.json').write_text('{"status":"partial","environment_validated":false}')
            with self.assertRaises(ValueError):
                lock.make_manifest(root)

    def test_inventory_rejects_missing_extra_version_and_duplicates(self):
        manifest = {'packages': [{'name': 'fixture-pkg', 'version': '1.0'}]}
        lock.verify_inventory(manifest, {'Fixture_Pkg': '1.0'})
        for actual in ({}, {'fixture-pkg': '2.0'}, {'fixture-pkg': '1.0', 'pip': '24.1.2'},
                       {'fixture-pkg': '1.0', 'fixture_pkg': '1.0'}):
            with self.assertRaises(ValueError):
                lock.verify_inventory(manifest, actual)


if __name__ == '__main__':
    unittest.main()
