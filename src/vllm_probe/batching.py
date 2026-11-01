"""Stable length grouping with explicit restoration of request ordering."""


def batch_indices(lengths, batch_size, bucketed):
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('batch size must be a positive integer')
    indices = list(range(len(lengths)))
    if bucketed:
        indices.sort(key=lambda i: lengths[i])
    return [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]


def restore_order(indexed_outputs, count):
    if len(indexed_outputs) != count:
        raise ValueError('output count mismatch')
    result = [None] * count
    seen = set()
    for index, value in indexed_outputs:
        if type(index) is not int or not 0 <= index < count or index in seen:
            raise ValueError('invalid or duplicate output index')
        seen.add(index)
        result[index] = value
    return result
