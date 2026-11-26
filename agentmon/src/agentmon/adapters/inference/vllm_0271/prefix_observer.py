"""Pinned, opt-in observation of prefix hash-map operations; no Tensor reads.

Unlike request selection, this observes the WHOLE map lifetime. Sampling only
selected requests would make cache hits and removals impossible to reconcile.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import functools
import hashlib
import inspect
import os
from pathlib import Path
import sys
import uuid
import weakref

MODULE = 'vllm.v1.core.block_pool'
FLAG = 'AGENTMON_VLLM_0271_PREFIX_ENABLE'
SOURCE_SHA256 = '51cad2fd425128a0ff433ca4685acfc022e40659153ea5d7180fc586e1eebb6c'
REVISION = '6e448d0ea9bf3d88d898b65449ca6dc2aec170ac'
METHODS = {'__init__': ('self',), 'get_one_block': ('self', 'key'),
           'insert': ('self', 'key', 'block'), 'pop': ('self', 'key', 'block_id')}
_REQUEST = ContextVar('agentmon_prefix_request', default=None)
# Synchronous BlockPool operation, set only by the opt-in content observer.
# This distinguishes map relocation/reset from eviction without time guessing.
_POOL_OPERATION = ContextVar('agentmon_pool_operation', default=None)


@contextmanager
def request_context(trace, mode, args, kwargs):
    """Use the actual synchronous manager call, never a nearest-time join."""
    if mode not in {'lookup', 'allocate', 'free'}:
        yield
        return
    token = None
    try:
        request = args[1] if len(args) > 1 else kwargs.get('request')
        request_id = getattr(request, 'request_id', None)
        external = getattr(request, 'external_req_id', None)
        trace._process()
        token = _REQUEST.set({'request_id': request_id if isinstance(request_id, str) else None,
            'external_id': external if isinstance(external, str) else None,
            'logical_pool_scope': trace.scope(args[0]), 'manager_operation': mode})
    except Exception:
        trace.failure()
    try:
        yield
    finally:
        if token is not None:
            _REQUEST.reset(token)


class PrefixObserver:
    def __init__(self, trace):
        self.trace = trace
        self.pid = os.getpid()
        self.scopes = weakref.WeakKeyDictionary()
        self.patches = []

    def scope(self, owner):
        if self.pid != os.getpid():
            self.pid = os.getpid()
            self.scopes = weakref.WeakKeyDictionary()
        if owner not in self.scopes:
            self.scopes[owner] = uuid.uuid4().hex
        return self.scopes[owner]

    def record(self, owner, operation, values, result):
        context = _REQUEST.get() or {}
        fields = {'prefix_schema': 'AGENTMON_PREFIX_MAP_V1',
            'cache_map_scope': self.scope(owner),
            'logical_pool_scope': context.get('logical_pool_scope'),
            'manager_operation': context.get('manager_operation'),
            'observation_scope': 'ALL_MAP_OPERATIONS_IN_PROCESS_NOT_REQUEST_BUDGETED',
            'source_revision': REVISION}
        fields.update(_POOL_OPERATION.get() or {})
        if operation == '__init__':
            event = 'PREFIX_MAP_CREATED'
        else:
            # The pinned upstream key is hash bytes followed by a uint32 Group.
            # Hash again for a compact opaque identity; never log tokens or keys.
            key = values['key']
            if not isinstance(key, bytes) or len(key) <= 4:
                raise ValueError('unsupported prefix key; no string/repr fallback')
            fields.update(content_key=hashlib.sha256(key).hexdigest(),
                          group_id=int.from_bytes(key[-4:], 'big'),
                          content_key_semantics='OPAQUE_ENGINE_CACHE_KEY_WITH_GROUP; MAP_LOCAL_DOMAIN')
            block = values.get('block') if operation == 'insert' else result
            fields['block_id'] = getattr(block, 'block_id', None)
            fields['ref_count_snapshot'] = getattr(block, 'ref_cnt', None)
            if operation == 'get_one_block':
                event = 'PREFIX_LOOKUP'
                fields['hit'] = result is not None
            elif operation == 'insert':
                event = 'PREFIX_INSERT'
            else:
                event = 'PREFIX_REMOVE'
                fields['removed'] = result is not None
                fields['requested_block_id'] = values['block_id']
                fields['removal_reason'] = 'NOT_OBSERVED_MAP_POP_IS_NOT_ALWAYS_EVICTION'
                if fields.get('operation_path'):
                    fields['removal_reason'] = 'OBSERVED_CALL_PATH:' + '/'.join(fields['operation_path'])
        self.trace.emit(event, 'EnginePrefixMap', MODULE+'.BlockHashToBlockMap.'+operation,
            fields.get('pool_request_id') or context.get('request_id'),
            fields.get('pool_external_request_id') or context.get('external_id'), **fields)

    def wrapper(self, original, operation):
        signature = inspect.signature(original)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            # Call once, preserve identity/exception. Never inspect iterator input.
            result = original(*args, **kwargs)
            try:
                values = signature.bind(*args, **kwargs).arguments
                self.record(args[0], operation, values, result)
            except Exception:
                self.trace.failure()
            return result
        wrapped.__agentmon_prefix__ = True
        return wrapped

    def patch_module(self, module):
        """Validate every target before mutation; no alternate SHA whitelist."""
        cls = module.BlockHashToBlockMap
        path = Path(module.__file__).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
            raise ValueError('prefix BlockPool source SHA mismatch')
        compiled = compile(path.read_bytes(), str(path), 'exec', dont_inherit=True,
                           optimize=sys.flags.optimize)
        class_code = next(c for c in compiled.co_consts if inspect.iscode(c)
                          and c.co_name == 'BlockHashToBlockMap')
        originals = {}
        for name, parameters in METHODS.items():
            method = inspect.getattr_static(cls, name)
            if not inspect.isfunction(method) or hasattr(method, '__wrapped__'):
                raise ValueError('prefix method already wrapped or unsupported')
            if tuple(inspect.signature(method).parameters) != parameters:
                raise ValueError('prefix signature mismatch')
            expected = next(c for c in class_code.co_consts if inspect.iscode(c) and c.co_name == name)
            if method.__code__ != expected:
                raise ValueError('prefix loaded code differs from pinned source')
            originals[name] = method
        for name, original in originals.items():
            replacement = self.wrapper(original, name)
            setattr(cls, name, replacement)
            self.patches.append((cls, name, original, replacement))
        self.trace.emit('PREFIX_TARGET_INSTALLED', 'bootstrap', MODULE,
            file_sha256=SOURCE_SHA256, loaded_code_matches_source=True,
            source_revision=REVISION, actual_execution='NOT_OBSERVED_UNTIL_PREFIX_MAP_CREATED')

    def uninstall(self):
        for cls, name, original, replacement in reversed(self.patches):
            if inspect.getattr_static(cls, name) is replacement:
                setattr(cls, name, original)
        self.patches.clear()
