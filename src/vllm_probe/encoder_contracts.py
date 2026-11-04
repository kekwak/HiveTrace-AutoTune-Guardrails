"""Pure guards shared by custom vLLM encoder models and their callers."""
from .adapter_contracts import classify_scores


def sequence_lengths(positions):
    if not positions or positions[0] != 0:
        raise ValueError('Expected full unchunked sequences starting at zero')
    lengths = []
    length = 0
    for position in positions:
        if type(position) is not int or position < 0:
            raise ValueError('Invalid encoder position')
        if position == 0:
            if length:
                lengths.append(length)
            length = 1
        elif position == length:
            length += 1
        else:
            raise ValueError('Chunked, padded or noncontiguous encoder positions')
    return lengths + [length]


def validate_metadata(model, length, metadata):
    if model != 'laya' or not isinstance(metadata, dict):
        raise ValueError('Laya marker metadata required')
    markers = metadata.get('marker_positions', [])
    qtype = metadata.get('qtype')
    if (type(qtype) is not int or qtype not in (0, 1, 2) or not markers
            or any(type(i) is not int or not 0 <= i < length for i in markers)
            or len(set(markers)) != len(markers)):
        raise ValueError('Invalid Laya marker metadata')
    return metadata


def parse_pooling(values, labels):
    return classify_scores(values, labels)
