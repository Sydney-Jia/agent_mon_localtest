"""Small metadata-only capability check; never probes a live manager or tensor."""
from __future__ import annotations

import hashlib
import inspect
import marshal
from types import SimpleNamespace


class GetterCompatibilityError(ValueError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def validate_block_getter(module):
    """Validate the helper on fresh metadata, not engine-owned KV objects.

    File identity and manager method identity are checked by the caller first.
    This bounded check is not a proof of arbitrary callable equivalence.
    """
    cls = getattr(module, 'KVCacheBlocks', None)
    method = getattr(cls, 'get_block_ids', None)
    if not inspect.isclass(cls) or not callable(method):
        raise GetterCompatibilityError('GETTER_MISSING_OR_NOT_CALLABLE')
    signature = inspect.signature(method, follow_wrapped=False)
    parameters = tuple(signature.parameters.values())
    if (tuple(p.name for p in parameters) != ('self', 'allow_none')
            or any(p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD for p in parameters)
            or parameters[0].default is not inspect.Parameter.empty
            or parameters[1].default is not False):
        raise GetterCompatibilityError('GETTER_SIGNATURE_MISMATCH')
    cases = (
        (([], []), ([], [])),
        (([SimpleNamespace(block_id=7), SimpleNamespace(block_id=11)],
          [SimpleNamespace(block_id=7)]), ([7, 11], [7])),
        (([], [SimpleNamespace(block_id=13)]), ([], [13])),
    )
    for blocks, expected in cases:
        before = tuple(tuple(id(block) for block in group) for group in blocks)
        obj = cls(blocks=blocks)
        for allow_none in (False, True):
            result = obj.get_block_ids(allow_none=True) if allow_none else obj.get_block_ids()
            if allow_none and not any(expected):
                valid = result is None
            else:
                valid = (type(result) is tuple and len(result) == len(expected)
                         and all(type(group) is list and all(type(b) is int for b in group) for group in result)
                         and result == expected)
            if not valid:
                raise GetterCompatibilityError('GETTER_BEHAVIOR_MISMATCH')
            if (obj.blocks is not blocks
                    or tuple(tuple(id(block) for block in group) for group in blocks) != before
                    or tuple([block.block_id for block in group] for group in blocks) != expected):
                raise GetterCompatibilityError('GETTER_MUTATED_METADATA')
    code = getattr(method, '__code__', None)
    return {
        'module_file': str(getattr(module, '__file__', '')),
        'class': 'KVCacheBlocks', 'method': 'get_block_ids',
        'line': getattr(code, 'co_firstlineno', None),
        'code_filename': getattr(code, 'co_filename', None),
        'signature': str(signature.replace(
            parameters=[p.replace(annotation=inspect.Parameter.empty) for p in parameters],
            return_annotation=inspect.Signature.empty)),
        'loaded_code_sha256': hashlib.sha256(marshal.dumps(code)).hexdigest() if code else None,
        'wrapped': hasattr(method, '__wrapped__'),
        'validation': 'behavior_pass', 'validation_mode': 'METADATA_HELPER_CAPABILITY',
        'test_cases': ['two_empty_groups', 'two_nonempty_groups', 'mixed_empty_group'],
        'engine_owned_state_accessed': False,
    }
