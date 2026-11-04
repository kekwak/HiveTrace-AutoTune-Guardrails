import json
import pytest
import subprocess

from src.vllm_probe.prepare_sample import prepare_sample, validate_output_path


def make_snapshots(tmp_path):
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')
    datasets = []
    selection = []
    for name in ('robustness-test-input', 'robustness-test-input-robust', 'robustness-test-output'):
        datasets.append({'repo': 'hivetrace/'+name, 'revision': 'a'*40})
        for split in ('safe', 'harm'):
            field = 'response' if name.endswith('output') else 'request'
            path = tmp_path/name/'data'/f'{split}-00000-of-00001.parquet'
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table({field: ['first', 'Пример']}), path)
            selection.append({'dataset': name, 'split': split, 'revision': 'a'*40,
                              'total': 2, 'indices': [1]})
    (tmp_path/'download-manifest.json').write_text(json.dumps({'datasets': datasets}))
    return {'records': 6, 'selection': selection}


def test_rebuilds_exact_private_sample_without_normalization(tmp_path):
    manifest = make_snapshots(tmp_path)
    rows = prepare_sample(tmp_path, manifest)
    assert len(rows) == 6
    assert rows[0] == {'text': 'Пример', 'dataset': 'robustness-test-input', 'split': 'safe'}
    assert rows[-1]['dataset'] == 'robustness-test-output'


def test_rejects_stale_revision_duplicate_indices_and_missing_splits(tmp_path):
    manifest = make_snapshots(tmp_path)
    manifest['selection'][0]['revision'] = 'b'*40
    with pytest.raises(ValueError, match='revision'): prepare_sample(tmp_path, manifest)
    manifest['selection'][0]['revision'] = 'a'*40
    manifest['selection'][0]['indices'] = [1, 1]
    with pytest.raises(ValueError, match='indices'): prepare_sample(tmp_path, manifest)
    manifest['selection'][0]['indices'] = [1]
    manifest['selection'].pop()
    with pytest.raises(ValueError, match='six'): prepare_sample(tmp_path, manifest)


def test_private_sample_cannot_be_written_in_another_git_worktree(tmp_path):
    repository = tmp_path/'another-project'
    repository.mkdir()
    subprocess.run(['git', 'init', '--quiet', str(repository)], check=True)
    with pytest.raises(ValueError, match='Git working tree'):
        validate_output_path(repository/'sample.json')
    validate_output_path(tmp_path/'sample.json')
