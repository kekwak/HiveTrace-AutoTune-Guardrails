"""Experimental request metadata contract; not a vLLM model integration."""


def parse_laya_metadata(extra_kwargs: dict | None, prompt_len: int) -> tuple[tuple[int, ...], int] | None:
    if extra_kwargs is None or 'laya' not in extra_kwargs:
        return None
    schema = extra_kwargs['laya']
    if not isinstance(schema, dict):
        raise ValueError('laya metadata must be an object')
    markers = schema.get('marker_positions')
    qtype = schema.get('qtype')
    if type(qtype) is not int or qtype not in (0, 1, 2):
        raise ValueError('qtype must be choice=0, score=1, or noul=2')
    if not isinstance(markers, list) or len(markers) < 2:
        raise ValueError('at least two marker positions required')
    if any(type(pos) is not int or not 0 <= pos < prompt_len for pos in markers):
        raise ValueError('marker positions must be within the prompt')
    if len(set(markers)) != len(markers):
        raise ValueError('marker positions must be unique')
    return tuple(markers), qtype
