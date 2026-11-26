import ast
import gzip
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from followup import team_smoke as smoke
from followup.prepare import ROOT


class TeamEnvironmentSmokeTests(unittest.TestCase):
    def test_smoke_texts_survive_real_gliner_normalization(self):
        from gliner2.processor import SchemaTransformer
        for text in smoke.TEXTS:
            self.assertEqual(SchemaTransformer._normalize_text(text), text)
        old = 'The library keeps a catalogue of books. ' * 18
        self.assertNotEqual(SchemaTransformer._normalize_text(old), old)

    def test_child_python_environment_is_clean_and_restored_on_failure(self):
        with patch.dict(os.environ, {'PYTHONPATH': '/host/injection', 'PYTHONHOME': '/host/python'}):
            with self.assertRaisesRegex(ValueError, 'failed'):
                with smoke.isolated_python_env():
                    self.assertNotIn('PYTHONPATH', os.environ)
                    self.assertNotIn('PYTHONHOME', os.environ)
                    raise ValueError('failed')
            self.assertEqual(os.environ['PYTHONPATH'], '/host/injection')
            self.assertEqual(os.environ['PYTHONHOME'], '/host/python')

    def test_partial_scores_survive_next_input_failure(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        score = {'p_safe': .9, 'p_unsafe': .1, 'logit_margin': -2., 'label': 'safe'}
        mask = SimpleNamespace(sum=lambda: 3)
        adapter = SimpleNamespace(prepare=Mock(side_effect=[{'attention_mask': mask}, ValueError('input')]),
                                  scores=Mock(return_value=[score]), input_hash=Mock(return_value='hash'))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'input'):
                smoke.evaluate(adapter, tmp)
            self.assertEqual(len(smoke.read(Path(tmp) / 'scores_partial.json')), 1)
            self.assertEqual(smoke.read(Path(tmp) / 'progress.json')['input_index'], 1)

    def test_notebook_is_fresh_valid_and_self_contained(self):
        import nbformat
        notebook = smoke.read(ROOT / 'notebooks/team_environment_smoke.ipynb')
        self.assertEqual(notebook, smoke.build_notebook(ROOT))
        nbformat.validate(nbformat.from_dict(notebook))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                self.assertEqual(cell['outputs'], [])

    def test_pins_retain_measured_extensions_and_reject_drift(self):
        spec = smoke.read(ROOT / 'env/team_base.json')
        versions = smoke.pins(spec)
        self.assertEqual(versions['onnxruntime-gpu'], '1.23.2')
        self.assertEqual(versions['gliner2'], '2.1.0')
        self.assertEqual(versions['flash-linear-attention'], '0.4.2')
        actual = {k.replace('_', '-'): v for k, v in versions.items()}
        smoke.check_versions(spec, actual)
        actual['torch'] = '2.11.0+cu128'
        with self.assertRaisesRegex(RuntimeError, 'Installed pins differ'):
            smoke.check_versions(spec, actual)

    def test_ort_needs_real_cuda_nodes_and_no_cpu_fallback(self):
        node = lambda provider: {'cat': 'Node', 'args': {'provider': provider}}
        self.assertEqual(smoke.check_ort_events([node('CUDAExecutionProvider')])['executed_cuda_nodes'], 1)
        for events in ([], [node('CPUExecutionProvider')],
                       [node('CUDAExecutionProvider'), node('CPUExecutionProvider')]):
            with self.assertRaises(RuntimeError):
                smoke.check_ort_events(events)

    def test_scores_reject_drift_flips_and_nonfinite(self):
        score = lambda p, label='safe': {'p_safe': 1-p, 'p_unsafe': p, 'logit_margin': 0., 'label': label}
        reference = [score(.1)]
        self.assertTrue(smoke.check_scores(reference, [score(.10001)], 'float32')['passed'])
        self.assertFalse(smoke.check_scores(reference, [score(.11)], 'float32')['passed'])
        self.assertFalse(smoke.check_scores(reference, [score(.1, 'unsafe')], 'float32')['passed'])
        with self.assertRaises(ValueError):
            smoke.check_scores(reference, [score(float('nan'))], 'float32')

    def test_archive_keeps_failures_but_excludes_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / 'run'
            work.mkdir()
            for name in ('encoder.onnx', 'encoder.onnx.data', 'model.safetensors', 'reference.pt'):
                (work / name).write_bytes(b'not-for-publication')
            (work / 'error.txt').write_text('failed')
            (work / 'ort-test.json').write_text('[1,2,3]')
            with zipfile.ZipFile(smoke.archive(work)) as archive:
                self.assertEqual(set(archive.namelist()), {'error.txt', 'ort-test.json.gz'})
                self.assertEqual(gzip.decompress(archive.read('ort-test.json.gz')), b'[1,2,3]')

    def prepare(self, temp):
        work = Path(temp) / 'run'
        work.mkdir()
        smoke.write(work / 'team_base.json', smoke.read(ROOT / 'env/team_base.json'))
        return work

    def test_worker_pin_failure_prevents_models_and_is_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = self.prepare(tmp)
            with patch.object(smoke, 'package_snapshot', return_value={}), \
                    patch.object(smoke, 'setup_torch') as setup, patch.object(smoke, 'model_probe') as model:
                result = smoke.worker(work, 'halo')
            self.assertEqual(result['status'], 'failed')
            setup.assert_not_called()
            model.assert_not_called()
            self.assertTrue((work / 'stages/halo/error.txt').exists())

    def simulate(self, temp, failure=None):
        from types import SimpleNamespace
        work = self.prepare(temp)
        calls = []
        def process(args, directory, name, timeout, cwd=None):
            calls.append(args)
            if name == failure:
                return {'status': 'timeout'}
            if name == 'worker':
                stage = args[-1]
                smoke.write(Path(directory) / 'status.json', {
                    'status': 'failed' if stage == failure else 'passed'})
            return {'status': 'completed'}
        with patch.object(smoke.platform, 'system', return_value='Linux'), \
                patch.object(smoke.sys, 'version_info', (3, 13, 15)), \
                patch.object(smoke.subprocess, 'run', return_value=SimpleNamespace(stdout='Tesla T4, 580.178.04\n')), \
                patch.object(smoke, 'package_snapshot', return_value={}), \
                patch('experiments.round2.run_process', side_effect=process):
            result = smoke.supervise(work)
        self.assertTrue(work.with_suffix('.zip').exists())
        return result, calls

    def test_success_uses_isolated_venv_and_does_not_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.simulate(tmp)
        self.assertEqual(result['status'], 'passed')
        self.assertFalse(result['benchmark_performed'])
        self.assertFalse(result['quality_evaluated'])
        self.assertEqual([c[-1] for c in calls if 'worker' in c], list(smoke.STAGES))
        for c in calls:
            if 'worker' in c:
                self.assertIn('_run_venv/bin/python', c[0])
            if 'install' in c or 'check' in c or 'freeze' in c:
                self.assertEqual(c[:4], [sys.executable, '-m', 'pip', '--python'])
                self.assertIn('_run_venv/bin/python', c[4])
        creation, = [c for c in calls if 'venv' in c]
        self.assertIn('--without-pip', creation)

    def test_real_pip_targets_pipless_venv_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            venv = root / 'venv'
            subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(venv)], check=True,
                           capture_output=True, text=True, timeout=60)
            python = venv / 'bin/python'
            wheel = root / 'u1_smoke_fixture-0.0.1-py3-none-any.whl'
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('u1_smoke_fixture.py', 'VALUE = 42\n')
                archive.writestr('u1_smoke_fixture-0.0.1.dist-info/METADATA',
                                 'Metadata-Version: 2.1\nName: u1-smoke-fixture\nVersion: 0.0.1\n')
                archive.writestr('u1_smoke_fixture-0.0.1.dist-info/WHEEL',
                                 'Wheel-Version: 1.0\nGenerator: unittest\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
                archive.writestr('u1_smoke_fixture-0.0.1.dist-info/RECORD', '')
            env = {**os.environ, 'PIP_CONFIG_FILE': os.devnull, 'PIP_DISABLE_PIP_VERSION_CHECK': '1'}
            subprocess.run([*smoke.target_pip(python), 'install', '--no-index', '--no-deps', str(wheel)],
                           check=True, capture_output=True, text=True, timeout=60, env=env)
            subprocess.run([python, '-c', 'import u1_smoke_fixture; assert u1_smoke_fixture.VALUE == 42'],
                           check=True, capture_output=True, text=True, timeout=20)
            subprocess.run([*smoke.target_pip(python), 'check'], check=True,
                           capture_output=True, text=True, timeout=30, env=env)
            result = subprocess.run([*smoke.target_pip(python), 'freeze', '--all'], check=True,
                                    capture_output=True, text=True, timeout=30, env=env)
            self.assertIn('u1-smoke-fixture', result.stdout.lower())
            subprocess.run([sys.executable, '-c',
                            'import importlib.util; assert importlib.util.find_spec("u1_smoke_fixture") is None'],
                           check=True, capture_output=True, text=True, timeout=20)

    def test_ort_failure_does_not_cancel_gliner_or_halo(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.simulate(tmp, failure='ort_cuda')
        self.assertEqual(result['status'], 'partial')
        self.assertFalse(result['environment_validated'])
        self.assertEqual([c[-1] for c in calls if 'worker' in c], list(smoke.STAGES))

    def test_cuda_failure_skips_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.simulate(tmp, failure='cuda')
        self.assertEqual([c[-1] for c in calls if 'worker' in c], ['cuda'])
        self.assertEqual(result['stages']['halo']['status'], 'not_run')

    def test_install_timeout_preserves_partial_archive_without_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.simulate(tmp, failure='install')
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(any('worker' in c for c in calls))


if __name__ == '__main__':
    unittest.main()
