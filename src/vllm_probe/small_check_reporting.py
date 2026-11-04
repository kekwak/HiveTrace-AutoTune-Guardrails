"""Aggregate-only partial checkpoints for the bounded GPU workers."""
from .adapter_contracts import compare_results


def parity_report(rows, reference, candidate, statuses, complete):
    if not len(rows) == len(reference) == len(candidate) == len(statuses):
        raise ValueError('Sample/result/status count mismatch')
    groups = {}
    for key in sorted({(row['dataset'], row['split']) for row in rows}):
        indices = [i for i, row in enumerate(rows) if (row['dataset'], row['split']) == key]
        groups['/'.join(key)] = compare_results([reference[i] for i in indices],
                                                [candidate[i] for i in indices])
    counts = {status: statuses.count(status) for status in sorted(set(statuses))}
    return dict(complete=complete, records=len(rows), statuses=counts,
                pooled=compare_results(reference, candidate), splits=groups)
