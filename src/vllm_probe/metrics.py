"""Binary quality metrics with explicit missing-prediction coverage."""


def summarize(records: list[dict]) -> list[dict]:
    groups = {('overall', None, None): []}
    for record in records:
        expected, predicted = record['expected'], record['predicted']
        if type(expected) is not int or expected not in (0, 1):
            raise ValueError('expected must be 0 or 1')
        if predicted is not None and (type(predicted) is not int or predicted not in (0, 1)):
            raise ValueError('predicted must be 0, 1, or None')
        dataset, split = record['dataset'], record['split']
        for key in [('overall', None, None), ('dataset', dataset, None), ('split', dataset, split)]:
            groups.setdefault(key, []).append(record)

    def ratio(a, b):
        return a / b if b else None

    result = []
    for (scope, dataset, split), group in groups.items():
        evaluated = [r for r in group if r['predicted'] is not None]
        tp = sum(r['expected'] == 1 and r['predicted'] == 1 for r in evaluated)
        tn = sum(r['expected'] == 0 and r['predicted'] == 0 for r in evaluated)
        fp = sum(r['expected'] == 0 and r['predicted'] == 1 for r in evaluated)
        fn = sum(r['expected'] == 1 and r['predicted'] == 0 for r in evaluated)
        result.append(dict(scope=scope, dataset=dataset, split=split, total=len(group),
                           evaluated=len(evaluated), coverage=ratio(len(evaluated), len(group)),
                           tp=tp, tn=tn, fp=fp, fn=fn,
                           accuracy=ratio(tp + tn, len(evaluated)), precision=ratio(tp, tp + fp),
                           recall=ratio(tp, tp + fn), specificity=ratio(tn, tn + fp),
                           f1=ratio(2 * tp, 2 * tp + fp + fn)))
    return result
