"""Opt-in logical cache-content observation for the pinned BlockPool.

No Tensor reads, policy changes or iterator wrapping. Slot leases are NOT
content generations. Prefix keys remain opaque, pool-scoped engine identities.
"""
import functools
import hashlib
import inspect
import sys
import time
import uuid
import weakref
from pathlib import Path

from .prefix_observer import (MODULE, SOURCE_SHA256, REVISION, _REQUEST,
                              _POOL_OPERATION)

FLAG = 'AGENTMON_VLLM_030_CONTENT_ENABLE'
METHODS = ('__init__', 'unpin_blocks', 'get_cached_block', 'get_new_blocks', 'touch', 'free_blocks',
           '_maybe_evict_cached_block', 'evict_blocks', 'move_block_hashes',
           'reset_prefix_cache', '_insert_block_hash', 'cache_full_blocks', 'cache_partial_block')


class ContentObserver:
    def __init__(self, trace, prefix):
        self.trace, self.prefix = trace, prefix
        self.patches = []
        self.states = weakref.WeakKeyDictionary()

    def state(self, pool):
        # trace.scope resets on fork. Never carry a parent's observation epoch.
        self.trace._process()
        scope = self.trace.scope(pool)
        state = self.states.get(pool)
        if state is None or state['scope'] != scope:
            state = dict(scope=scope, active={}, leases={}, initialized=False)
            self.states[pool] = state
        return state

    def snapshot(self, pool, block, state):
        keys = set(pool.cached_block_hashes_by_block.get(block.block_id, ()))
        if block.block_hash is not None:
            keys.add(block.block_hash)
        if any(not isinstance(k, bytes) or len(k) <= 4 for k in keys):
            raise ValueError('unsupported content key')
        return dict(block_id=block.block_id, ref_count=block.ref_cnt,
                    is_null=block.is_null,
                    slot_lease=state['leases'].get(block.block_id),
                    primary_prefix_tokens=block.block_hash_num_tokens,
                    content_keys=sorted(hashlib.sha256(k).hexdigest() for k in keys))

    def emit(self, event, context, **fields):
        request = _REQUEST.get() or {}
        self.trace.emit(event, 'EngineBlockPool', MODULE+'.BlockPool.'+context['operation_path'][-1],
                        context.get('pool_request_id') or request.get('request_id'),
                        context.get('pool_external_request_id') or request.get('external_id'),
                        content_schema='AGENTMON_CONTENT_LIFECYCLE_V1',
                        logical_pool_scope=request.get('logical_pool_scope'),
                        manager_operation=request.get('manager_operation'),
                        **context, **fields)

    def begin(self, pool, name, values):
        state = self.state(pool)
        parent = _POOL_OPERATION.get() or {}
        context = dict(pool_scope=state['scope'], operation_id=uuid.uuid4().hex,
                       parent_operation_id=parent.get('operation_id'),
                       operation_path=parent.get('operation_path', [])+[name])
        context['observed_call_started_monotonic_ns']=time.monotonic_ns()
        # 保留原始 cache_full/partial 调用中的实际 Group block_size。
        # _insert_block_hash 自身只有 prefix 长度，不能用全局 128/32 猜测区间。
        context['registration_kind'] = (
            name if name in {'cache_full_blocks', 'cache_partial_block'}
            else parent.get('registration_kind'))
        context['registration_block_size'] = (
            values.get('block_size') if name in {'cache_full_blocks', 'cache_partial_block'}
            else parent.get('registration_block_size'))
        # cache_full/partial may run outside the three manager hooks. Their
        # actual Request argument is stronger evidence than timestamp proximity.
        request = values.get('request')
        context['pool_request_id'] = getattr(request, 'request_id', None) if request is not None else parent.get('pool_request_id')
        context['pool_external_request_id'] = getattr(request, 'external_req_id', None) if request is not None else parent.get('pool_external_request_id')
        if name != '__init__':
            context['pool_map_scope'] = self.prefix.scope(pool.cached_block_hash_to_block)
        before, coverage = {}, 'COMPLETE_TRACKED_ACTIVE_SET'
        if name in {'free_blocks', 'unpin_blocks'}:
            # free_blocks accepts a one-shot iterable. Do NOT enumerate it or
            # replace its arguments. Diff only the observer's active slot set.
            blocks = list(state['active'].values())
            if not state['initialized']:
                coverage = 'INITIAL_POOL_NOT_OBSERVED'
        elif name == 'touch':
            blocks = values['blocks']
            if type(blocks) not in (list, tuple):
                blocks, coverage = [], 'UNSUPPORTED_SEQUENCE_NOT_INSPECTED'
        elif name in {'_maybe_evict_cached_block', '_insert_block_hash'}:
            blocks = [values['block']]
        elif name == 'move_block_hashes':
            blocks = [values['src_block'], values['dst_block']]
        else:
            blocks = []
        for block in blocks:
            before[block.block_id] = (block, self.snapshot(pool, block, state))
        self.emit('CONTENT_OPERATION_BEGIN', context, coverage=coverage)
        return state, context, before, coverage

    def finish(self, pool, name, values, result, capture, error=None):
        state, context, before, coverage = capture
        if name == '__init__' and error is None:
            state['initialized'] = True
            context['pool_map_scope'] = self.prefix.scope(pool.cached_block_hash_to_block)
            self.emit('CONTENT_POOL_CREATED', context, num_blocks=pool.num_gpu_blocks,
                      null_block_id=pool.null_block.block_id, hash_block_size=pool.hash_block_size,
                      enable_caching=pool.enable_caching)
        if name == 'get_new_blocks' and error is None:
            for block in result:
                state['leases'][block.block_id] = state['leases'].get(block.block_id, 0)+1
                state['active'][block.block_id] = block
                self.emit('CONTENT_SLOT_ALLOCATED', context,
                          after=self.snapshot(pool, block, state),
                          identity_semantics='SLOT_LEASE_NOT_CONTENT_GENERATION')
        for block, old in before.values():
            new = self.snapshot(pool, block, state)
            if not block.is_null:
                if block.ref_cnt > 0:
                    state['active'][block.block_id] = block
                else:
                    state['active'].pop(block.block_id, None)
            if old != new:
                self.emit('CONTENT_BLOCK_CHANGED', context, before=old, after=new,
                          coverage=coverage, call_succeeded=error is None)
        if name == '_insert_block_hash' and error is None:
            key = values['block_hash_with_group_id']
            if not isinstance(key, bytes) or len(key) <= 4:
                raise ValueError('unsupported content key')
            end, size = values.get('num_tokens'), context.get('registration_block_size')
            kind = context.get('registration_kind')
            token_interval = None
            if type(end) is int and type(size) is int and end > 0 and size > 0:
                if kind == 'cache_full_blocks' and end % size == 0:
                    token_interval = [end-size, end]
                elif kind == 'cache_partial_block' and end % size:
                    token_interval = [end//size*size, end]
            self.emit('CONTENT_PREFIX_REGISTERED', context,
                      content_key=hashlib.sha256(key).hexdigest(), group_id=int.from_bytes(key[-4:], 'big'),
                      block_id=values['block'].block_id, prefix_tokens=values.get('num_tokens'),
                      slot_lease=state['leases'].get(values['block'].block_id),
                      token_interval=token_interval,
                      prefix_dependency_interval=[0, end] if token_interval else None,
                      position_basis='PINNED_CACHE_REGISTRATION_ARGUMENTS' if token_interval else 'NOT_OBSERVED',
                      position_semantics='ENGINE_CACHE_KEY_POSITION_NOT_TENSOR_ROW_RANGE',
                      meaning='ENGINE_PREFIX_BOUNDARY_NOT_MESSAGE_SPAN_OR_COMPUTE_COMPLETION')
        if name == 'reset_prefix_cache' and error is None:
            self.emit('CONTENT_CACHE_RESET', context, succeeded=result is True,
                      new_map_scope=self.prefix.scope(pool.cached_block_hash_to_block))
        if coverage != 'COMPLETE_TRACKED_ACTIVE_SET':
            self.emit('CONTENT_OBSERVATION_GAP', context, reason=coverage)
        ended=time.monotonic_ns()
        # Free queue includes unreferenced cached blocks eligible for reclaim;
        # its size is NOT "empty KV content slots" or HBM allocation bytes.
        capacity=None
        if name in {'__init__','get_new_blocks','free_blocks','cache_full_blocks',
                    'cache_partial_block','reset_prefix_cache','evict_blocks'} and error is None:
            free=getattr(getattr(pool,'free_block_queue',None),'num_free_blocks',None)
            count=getattr(pool,'num_gpu_blocks',None)
            cache=getattr(getattr(pool,'cached_block_hash_to_block',None),'_cache',None)
            capacity=dict(total_slots=count if type(count) is int else None,
                          free_or_reclaimable_slots=free if type(free) is int else None,
                          cached_key_count=len(cache) if type(cache) is dict else None,
                          source='PINNED_BLOCK_POOL_CPU_ATTRIBUTES',
                          meaning='FREE_QUEUE_INCLUDES_CACHED_RECLAIMABLE_SLOTS; NOT_PHYSICAL_FREE_BYTES')
        self.emit('CONTENT_OPERATION_END', context, succeeded=error is None,
                  exception_type=type(error).__name__ if error is not None else None,
                  boolean_result=result if type(result) is bool else None,
                  observed_call_ended_monotonic_ns=ended,
                  host_inclusive_observation_ns=ended-context['observed_call_started_monotonic_ns'],
                  timing_scope='HOST_CALL_AND_NESTED_OBSERVER_NOT_DEVICE_COMPLETION; DO_NOT_SUM_NESTED_CALLS',
                  pool_capacity=capacity)

    def wrapper(self, original, name):
        signature = inspect.signature(original)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            capture = token = None
            try:
                values = signature.bind(*args, **kwargs).arguments
                capture = self.begin(args[0], name, values)
                token = _POOL_OPERATION.set(capture[1])
            except Exception:
                self.trace.failure()
            try:
                result = original(*args, **kwargs)
            except BaseException as error:
                if capture is not None:
                    try:
                        self.finish(args[0], name, values, None, capture, error)
                    except Exception:
                        self.trace.failure()
                raise
            else:
                if capture is not None:
                    try:
                        self.finish(args[0], name, values, result, capture)
                    except Exception:
                        self.trace.failure()
                return result
            finally:
                if token is not None:
                    _POOL_OPERATION.reset(token)
        return wrapped

    def patch_module(self, module):
        # Validate all functions before installing any wrapper. No relaxed hash
        # list and no acceptance based only on inspect.signature.
        path = Path(module.__file__).resolve()
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
            raise ValueError('content BlockPool source SHA mismatch')
        code = compile(source, str(path), 'exec', dont_inherit=True, optimize=sys.flags.optimize)
        cls_code = next(c for c in code.co_consts if inspect.iscode(c) and c.co_name == 'BlockPool')
        cls = module.BlockPool
        originals = {}
        for name in METHODS:
            method = inspect.getattr_static(cls, name)
            expected = next(c for c in cls_code.co_consts if inspect.iscode(c) and c.co_name == name)
            if not inspect.isfunction(method) or hasattr(method, '__wrapped__') or method.__code__ != expected:
                raise ValueError('content loaded code mismatch: '+name)
            originals[name] = method
        for name, original in originals.items():
            replacement = self.wrapper(original, name)
            setattr(cls, name, replacement)
            self.patches.append((cls, name, original, replacement))
        self.trace.emit('CONTENT_TARGET_INSTALLED', 'bootstrap', MODULE,
                        file_sha256=SOURCE_SHA256, source_revision=REVISION,
                        loaded_code_matches_source=True,
                        actual_execution='NOT_OBSERVED_UNTIL_CONTENT_POOL_CREATED')

    def uninstall(self):
        for cls, name, original, replacement in reversed(self.patches):
            if inspect.getattr_static(cls, name) is replacement:
                setattr(cls, name, original)
        self.patches.clear()
