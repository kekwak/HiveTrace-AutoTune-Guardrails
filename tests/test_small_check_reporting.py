from src.vllm_probe.small_check_reporting import parity_report


def test_partial_custom_check_never_hides_rejections_or_claims_completion():
    rows = [{'dataset': 'input', 'split': 'safe'}, {'dataset': 'output', 'split': 'harm'}]
    reference = [{'label': 'safe', 'scores': {'safe': .9, 'unsafe': .1}}, None]
    candidate = [{'label': 'safe', 'scores': {'safe': .8, 'unsafe': .2}}, None]
    report = parity_report(rows, reference, candidate, ['ok', 'over_context'], complete=False)
    assert report['complete'] is False
    assert report['pooled']['total'] == 2
    assert report['pooled']['paired'] == 1
    assert report['statuses'] == {'ok': 1, 'over_context': 1}
    assert report['splits']['output/harm']['missing'] == 1
