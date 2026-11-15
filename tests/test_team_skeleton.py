import ast
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from followup import team_skeleton as skeleton
from followup.team_lock import read, requirements

ROOT = Path(__file__).resolve().parents[1]


class SkeletonTests(unittest.TestCase):
    def fixture(self, tmp):
        work = Path(tmp) / 'work'
        (work / 'setup').mkdir(parents=True)
        manifest = read(ROOT / 'env/team_environment.lock.json')
        skeleton.write(work / 'setup/lock.json', manifest)
        (work / 'setup/requirements.txt').write_text(requirements(manifest))
        (work / 'experiment.py').write_text('print("hello")\n')
        return skeleton.Session(work)

    def fake_command(self, session, failure=None):
        def run(args, cwd, log, timeout):
            log = Path(log)
            log.write_text('fixture log')
            if log.name == failure:
                raise RuntimeError('fixture failure')
            if log.name == 'venv.log':
                session.venv.mkdir()
            if log.name == 'probe.log':
                skeleton.write(session.work / 'setup/runtime.json', {'fixture': True})
            return {'status': 'passed'}
        return run

    def test_notebook_self_contained_fresh_and_syntax(self):
        import nbformat
        book = read(ROOT / 'notebooks/team_environment_skeleton.ipynb')
        self.assertEqual(book, skeleton.build_notebook(ROOT))
        nbformat.validate(nbformat.from_dict(book))
        for cell in book['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if source.startswith('%%writefile'):
                    source = source.split('\n', 1)[1]
                ast.parse(source)
                self.assertFalse(cell['outputs'])
        source = ''.join(book['cells'][1]['source'])
        tree = ast.parse(source)
        values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                  if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id in ('PAYLOAD', 'CHECKSUM')}
        from followup.runtime import digest
        self.assertEqual(digest(values['PAYLOAD']), values['CHECKSUM'])
        self.assertEqual(values['PAYLOAD']['requirements'], requirements(values['PAYLOAD']['lock']))
        self.assertEqual(set(values['PAYLOAD']['sources']), {'team_lock.py', 'team_skeleton.py'})

    def test_setup_targets_only_venv_and_reuses_after_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = self.fixture(tmp)
            with patch.object(skeleton, 'precheck', return_value={'fixture': True}), \
                    patch.object(skeleton, 'command', side_effect=self.fake_command(s)) as commands:
                self.assertEqual(s.setup()['status'], 'passed')
                install = next(c.args[0] for c in commands.call_args_list if c.args[2].name == 'install.log')
                self.assertEqual(install[:5], [sys.executable, '-m', 'pip', '--python', str(s.python)])
                for option in ('--no-deps', '--require-hashes', '--only-binary=:all:'):
                    self.assertIn(option, install)
                commands.reset_mock()
                self.assertEqual(s.setup()['status'], 'passed')
                self.assertEqual([c.args[2].name for c in commands.call_args_list], ['pip-check.log', 'probe.log'])

    def test_precheck_failure_does_not_install_and_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = self.fixture(tmp)
            with patch.object(skeleton, 'precheck', side_effect=RuntimeError('Wrong Python')), \
                    patch.object(skeleton, 'command') as command:
                self.assertEqual(s.setup()['status'], 'failed')
                command.assert_not_called()
                self.assertEqual(s.run()['status'], 'not_run')
            with zipfile.ZipFile(s.work.with_suffix('.zip')) as z:
                self.assertIn('setup/error.txt', z.namelist())

    def test_install_failure_and_tampered_lock_stop_setup(self):
        for failure in ('install.log', 'pip-check.log', 'probe.log', 'tamper'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                s = self.fixture(tmp)
                if failure == 'tamper':
                    (s.work / 'setup/requirements.txt').write_text('other==1')
                with patch.object(skeleton, 'precheck', return_value={}), \
                        patch.object(skeleton, 'command', side_effect=self.fake_command(s, failure)):
                    self.assertEqual(s.setup()['status'], 'failed')
                    self.assertFalse((s.venv / 'u1-lock.sha256').exists())
                    self.assertEqual(s.run()['status'], 'not_run')

    def test_each_experiment_snapshots_source_and_preserves_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = self.fixture(tmp)
            (s.work / 'helper.py').write_text('VALUE = 42\n')
            with patch.object(skeleton, 'precheck', return_value={}), \
                    patch.object(skeleton, 'command', side_effect=self.fake_command(s)):
                s.setup()
            with patch.object(skeleton, 'command', side_effect=self.fake_command(s)) as command:
                first = s.run()
                self.assertEqual(first['status'], 'passed')
                self.assertEqual(command.call_args.args[0][0], str(s.python))
            (s.work / 'experiment.py').write_text('raise ValueError("test")\n')
            with patch.object(skeleton, 'command', side_effect=self.fake_command(s, 'stdout.log')):
                second = s.run()
            self.assertNotEqual(first['run'], second['run'])
            self.assertEqual(second['status'], 'failed')
            self.assertEqual((s.work / 'runs' / first['run'] / 'experiment.py').read_text(), 'print("hello")\n')
            self.assertEqual((s.work / 'runs' / first['run'] / 'source_helpers/helper.py').read_text(), 'VALUE = 42\n')
            self.assertTrue((s.work / 'runs' / second['run'] / 'error.txt').exists())
            # A later failed setup must revoke the ready state, even with an old marker.
            with patch.object(skeleton, 'precheck', side_effect=RuntimeError('GPU gone')):
                s.setup()
            self.assertTrue((s.venv / 'u1-lock.sha256').exists())
            self.assertEqual(s.run()['status'], 'not_run')

    def test_real_process_logs_failure_and_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            skeleton.command([sys.executable, '-c', 'print("ok")'], path, path / 'ok.log', 10)
            self.assertEqual((path / 'ok.log').read_text().strip(), 'ok')
            for name, code, timeout in [('failure', 'raise ValueError("fixture")', 10),
                                         ('timeout', 'import time; time.sleep(10)', .05)]:
                with self.assertRaises(RuntimeError):
                    skeleton.command([sys.executable, '-c', code], path, path / (name + '.log'), timeout)
                expected = 'timeout' if name == 'timeout' else 'failed'
                self.assertEqual(read(path / (name + '.process.json'))['status'], expected)

    def test_archive_excludes_inputs_weights_secrets_and_large_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = self.fixture(tmp)
            run = s.work / 'runs/one/results'
            run.mkdir(parents=True)
            for name in ('small.json', '.secret', 'model.safetensors', 'large.trace'):
                (run / name).write_text('x' * (500 if name == 'large.trace' else 1))
            (run / 'link').symlink_to(s.work / 'experiment.py')
            (s.work / 'private_input.txt').write_text('private')
            with zipfile.ZipFile(skeleton.archive(s.work, max_file=400)) as z:
                self.assertIn('runs/one/results/small.json', z.namelist())
                for name in ('.secret', 'model.safetensors', 'large.trace', 'link'):
                    self.assertNotIn('runs/one/results/' + name, z.namelist())
                self.assertNotIn('private_input.txt', z.namelist())
                self.assertIn('archive_contents.json', z.namelist())

    def test_child_env_does_not_leak_host_python_configuration(self):
        with patch.dict(os.environ, {'PYTHONPATH': '/host', 'PYTHONHOME': '/host', 'HF_TOKEN': 'fixture'}):
            env = skeleton.child_env()
            self.assertNotIn('PYTHONPATH', env)
            self.assertNotIn('PYTHONHOME', env)
            self.assertEqual(env['HF_TOKEN'], 'fixture')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0')

    def test_incomplete_existing_venv_is_not_reinstalled(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = self.fixture(tmp)
            s.venv.mkdir()
            with patch.object(skeleton, 'precheck', return_value={}), patch.object(skeleton, 'command') as command:
                self.assertEqual(s.setup()['status'], 'failed')
                command.assert_not_called()

    def test_precheck_rejects_wrong_python_before_inspecting_gpu(self):
        with patch.object(skeleton.platform, 'system', return_value='Linux'), \
                patch.object(skeleton.platform, 'machine', return_value='x86_64'), \
                patch.object(skeleton.sys, 'version_info', (3, 12, 0)), \
                patch.object(skeleton.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'Python 3.13'):
                skeleton.precheck()
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
