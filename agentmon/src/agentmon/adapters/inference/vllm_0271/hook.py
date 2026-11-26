"""Read-only wrappers, installed only at startup, never in an existing engine.

The KV contract is pinned to the user's custom source file. Other endpoints are
capability checked on import and their actual loaded signatures/files reported.
"""
from __future__ import annotations

import atexit
import contextvars
from dataclasses import dataclass
import functools
import hashlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import inspect
import marshal
import multiprocessing
import os
from pathlib import Path
import re
import sys
import uuid
import weakref

from .contracts import EXPECTED_VERSIONS, KV_FILE_SHA256, KV_MODULE, KV_SIGNATURES, TARGETS
from .compat import GetterCompatibilityError, validate_block_getter
from .events import EventSink

FLAG = 'AGENTMON_VLLM_0271_TRACE_ENABLE'
_API_CONTEXT = contextvars.ContextVar('agentmon_vllm0271_api', default=None)
_ACTIVE = None


def truthy(value):
    return str(value or '').strip().lower() in {'1', 'true', 'yes', 'on'}


def identity(value):
    return value if isinstance(value, str) and 0 < len(value) <= 256 else None


def integer(value):
    return value if type(value) is int else None


def request_fields(request):
    return identity(getattr(request, 'request_id', None)), identity(getattr(request, 'external_req_id', None))


@dataclass(frozen=True)
class TraceConfig:
    # 这些值在进程启动时读取并冻结；修改宿主机 shell 环境不会更新运行中的配置。
    # 请求预算、日志队列、Block 快照大小分别控制不同资源，不能混称“任务上限”。
    directory: Path
    run_id: str
    evidence_status: str = 'VERIFIED'
    max_blocks_per_group: int = 256  # 每个 Group 最多记录多少 ID；超出会标记截断。
    queue_capacity: int = 512  # 暂存事件条数；不是请求数，也不是字节容量。
    arm_file: Path | None = None
    max_requests_per_process: int = 1  # 历史字段名；V1.3 按进程内 ARM 会话计数，0 为不限。
    holdings_checkpoint_interval: int = 32
    max_file_bytes: int = 16*1024*1024

    def __post_init__(self):
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,96}', self.run_id):
            raise ValueError('run_id must be 1..96 safe identifier characters')
        if not 1 <= self.max_blocks_per_group <= 4096:
            raise ValueError('max_blocks_per_group out of bounds')
        if not 1 <= self.queue_capacity <= 4096:
            raise ValueError('queue_capacity out of bounds')
        if not 0 <= self.max_requests_per_process <= 1000000:
            raise ValueError('max_requests_per_process out of bounds')
        if not 1 <= self.holdings_checkpoint_interval <= 1024:
            raise ValueError('holdings_checkpoint_interval out of bounds')
        if not 1024 <= self.max_file_bytes <= 1024*1024*1024:
            raise ValueError('max_file_bytes out of bounds')


class Trace:
    def __init__(self, config):
        self.config = config
        self.pid = None
        self.sink = None
        self.selected = None
        self.selected_ids = set()
        self.external_ids = {}
        self.started = set()
        self.scopes = weakref.WeakKeyDictionary()
        self.session = None
        self.session_counts = {}
        self.request_sessions = {}
        self.holdings = {}
        self.suppressed_holdings = 0
        self.input_observer = None

    def armed(self):
        try:
            return self.config.arm_file is None or Path(self.config.arm_file).is_file()
        except OSError:
            return False

    def _process(self):
        # Forked children must never reuse a parent's queue, thread or lock.
        # API、Engine、Worker 可能是不同进程；每个进程有自己的 epoch、队列与预算。
        # 不把多个进程的剩余额度相加成一个服务的“可观测请求数”。
        if self.pid != os.getpid():
            self.pid = os.getpid()
            self.sink = EventSink(self.config.directory, self.config.run_id,
                                  evidence_status=self.config.evidence_status,
                                  capacity=self.config.queue_capacity, max_file_bytes=self.config.max_file_bytes)
            self.selected = None
            self.selected_ids = set()
            self.external_ids = {}
            self.started = set()
            self.scopes = weakref.WeakKeyDictionary()
            self.session = None
            self.session_counts = {}
            self.request_sessions = {}
            self.holdings = {}
            self.suppressed_holdings = 0
            # 进程启动时冻结研究开关/密钥；不会热改已经运行的服务。
            from .input_observer import InputObserver
            self.input_observer = InputObserver()
        return self.sink

    def failure(self):
        try:
            self._process().observer_failure()
        except Exception:
            try:
                print('AGENTMON_0271_OBSERVER_FAILURE; metadata omitted', file=sys.stderr)
            except Exception:
                pass

    def emit(self, event, role, source, request_id=None, external_id=None, **fields):
        fields.setdefault('trace_session_id', self.request_sessions.get(request_id, self.session))
        return self._process().emit(event, process_role=role, source_function=source,
                             request_id=request_id, external_request_id=external_id, **fields)

    def observing(self):
        # 撤掉 ARMED 后不再选择新请求，但已选请求继续观察到终态，避免人为截断引用。
        return self.armed() or (self.pid == os.getpid() and bool(self.selected_ids))

    def session_token(self):
        # 文件内容是采集会话身份，而不仅是开关；新 token 开始新预算，相同 token 沿用。
        # 只控制采集，不负责启动模型，也不会把已导入的 Hook 代码热更新。
        if self.config.arm_file is None:
            return 'PROCESS_DEFAULT'
        try:
            path = Path(self.config.arm_file)
            if not path.is_file(): return None
            return path.read_text(encoding='utf-8').strip()[:256] or 'LEGACY_ARMED'
        except Exception:
            return None

    def selected_request(self, request_id):
        self._process()
        if request_id is None: return False
        # 同一活跃请求的多次 lookup/allocate 不重复消耗请求预算。
        if request_id in self.selected_ids: return True
        token = self.session_token()
        if token is None: return False
        if token != self.session:
            self.session = token
            # Only active old sessions need in-memory counters.
            active = set(self.request_sessions.values())
            self.session_counts = {k:v for k,v in self.session_counts.items() if k in active}
            self.session_counts.setdefault(token, 0)
            self.emit('TRACE_SESSION_STARTED', 'observer', 'Trace.selected_request',
                      trace_session_id=token, request_budget_scope='ARM_SESSION',
                      max_requests_per_process=self.config.max_requests_per_process)
        used = self.session_counts[token]
        # limit=0 时跳过数量判断；正数耗尽后仅拒绝新增观测，不拒绝原模型推理。
        # benchmark 预检会主动阻止覆盖不足的实验，以免跑完却拿不到完整证据。
        if self.config.max_requests_per_process and used >= self.config.max_requests_per_process:
            self.emit('TRACE_REQUEST_NOT_SELECTED', 'observer', 'Trace.selected_request',
                      request_id=request_id, trace_session_id=token, reason={'selection':'REQUEST_BUDGET_EXHAUSTED'})
            return False
        self.session_counts[token] += 1
        self.selected_ids.add(request_id)
        self.request_sessions[request_id] = token
        if self.selected is None: self.selected = request_id
        return True

    def retire(self, request_id):
        # 只清理本观察器的活跃身份与持有快照；不是调用 vLLM free 或释放 Tensor。
        # 缺少真实终态时不能用超时替代：那会伪造生命周期结束。
        self.selected_ids.discard(request_id)
        self.external_ids.pop(request_id, None)
        self.request_sessions.pop(request_id, None)
        for key in [k for k in self.holdings if k[0] == request_id]:
            self.holdings.pop(key, None)

    def start(self, role, source, location):
        self._process()
        key = (role, source)
        if key not in self.started:
            self.started.add(key)
            try:
                comm = Path('/proc/self/comm').read_text().strip()[:128]
            except OSError:
                comm = None
            process_identity = {'multiprocessing_name': multiprocessing.current_process().name[:128],
                                'linux_comm': comm,
                                'linux_comm_status': 'VERIFIED' if comm else 'NOT_OBSERVED'}
            self.emit('HOOK_STARTED', role, source, loaded_target=location,
                      runtime_process_identity=process_identity,
                      selection_policy='EXPLICIT_ARM_SESSION_WITH_OPTIONAL_REQUEST_BUDGET',
                      request_budget_scope='ARM_SESSION',
                      max_requests_per_process=self.config.max_requests_per_process,
                      reason={'runtime_pid': 'PID executing this original target function, not a docker-exec probe'})

    def groups(self, value):
        if value is None:
            return None
        groups = getattr(value, 'blocks', None)
        if not isinstance(groups, (tuple, list)) or len(groups) > 32:
            raise TypeError('unsupported KVCacheBlocks layout')
        output = []
        for group_id, group in enumerate(groups):
            total = len(group)
            ids = []
            # 控制热路径事件大小。保存 total 与 truncated，使离线分析知道列表不完整；
            # 截断后“列表没有某 ID”不能作为它已释放的证据，也不改变实际 Group。
            for block in group[:self.config.max_blocks_per_group]:
                block_id = integer(getattr(block, 'block_id', None))
                if block_id is None:
                    raise TypeError('noninteger logical block_id')
                ids.append(block_id)
            output.append({'group_id': group_id, 'block_ids': ids,
                           'total_block_count': total, 'blocks_truncated': total > len(ids)})
        return output

    def held(self, manager, request_id):
        # Proven getter: KVCacheManager.get_blocks:703-705 in provided source.
        return self.groups(manager.get_blocks(request_id))

    def scope(self, manager):
        if manager not in self.scopes:
            self.scopes[manager] = uuid.uuid4().hex
        return self.scopes[manager]

    def block_event(self, event, source, request_id, manager, groups, **fields):
        common = {'logical_pool_scope': self.scope(manager),
                  'logical_pool_scope_semantics': 'Observer UUID for KVCacheManager instance, not physical pool identity',
                  'group_id_semantics': 'KVCacheBlocks.blocks tuple index; not layer or TP/EP rank',
                  'token_counts_scope': 'Whole request counters repeated across groups; do not sum group rows',
                  'null_block_semantics': 'NOT_OBSERVED; block IDs may include placeholder/null blocks',
                  'reason': {'ref_count_if_provable': 'BlockPool ref-count implementation not captured',
                             'generation': 'logical ID reuse possible; allocator generation NOT_OBSERVED',
                             'physical_allocation': 'logical metadata only; no backing allocation observed'}}
        if groups is None:
            self.emit(event, 'EngineCore/Scheduler', source, request_id,
                      self.external_ids.get(request_id), observation_status='NOT_OBSERVED',
                      **common, **fields)
            return
        for group in groups:
            key = (request_id, common['logical_pool_scope'], group['group_id'])
            # Only repeated complete holdings with NO returned blocks can be
            # coalesced. Prefix, NONE, partial and release facts are retained.
            fingerprint = (tuple(group['block_ids']), fields.get('reuse_kind'))
            previous = self.holdings.get(key)
            coalescible = (event == 'KV_BLOCKS_ASSIGNED' and not group['blocks_truncated']
                and not fields.get('returned_new_block_ids') and not fields.get('attached_prefix_candidate_ids'))
            count = previous[1]+1 if previous and previous[0] == fingerprint else 0
            if coalescible and previous and previous[0] == fingerprint and count < self.config.holdings_checkpoint_interval:
                self.holdings[key] = (fingerprint, count)
                self.suppressed_holdings += 1
                continue
            accepted = self.emit(event, 'EngineCore/Scheduler', source, request_id,
                      self.external_ids.get(request_id), **common, **group,
                      observation_detail='COMPLETE_HOLDINGS_CHECKPOINT' if coalescible else 'FULL_EVENT',
                      suppressed_unchanged_holdings_total=self.suppressed_holdings, **fields)
            if accepted and event == 'KV_BLOCKS_ASSIGNED': self.holdings[key] = (fingerprint, 0)

    def close(self):
        if self.sink and self.pid == os.getpid():
            self.sink.close()


def _location(original, mode):
    filename = inspect.getsourcefile(original) or inspect.getfile(original)
    path = Path(filename)
    return {'file': str(path.resolve()), 'line': getattr(original, '__code__', None).co_firstlineno,
            'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
            'loaded_code_sha256': hashlib.sha256(marshal.dumps(original.__code__)).hexdigest(),
            'observer_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'signature': str(inspect.signature(original)),
            'contract': 'PINNED_SERVER_KV_SOURCE' if mode in {'lookup', 'allocate', 'free'} else 'LOADED_SIGNATURE_AND_EXPLICIT_FIELDS'}


def _request(bound):
    # Exact named field first; conversion argument names vary, so examine only
    # objects passed directly to this function, never derive IDs from strings.
    if 'request' in bound.arguments:
        return bound.arguments['request']
    for name, value in bound.arguments.items():
        if name not in {'self', 'cls'} and identity(getattr(value, 'request_id', None)):
            return value
    return None


def _status(request):
    value = getattr(request, 'status', None)
    return identity(getattr(value, 'name', None))


def _wrapper(original, mode, trace, source, location):
    from .prefix_observer import request_context, FLAG as PREFIX_FLAG
    from .content_observer import FLAG as CONTENT_FLAG
    prefix_enabled = truthy(os.environ.get(PREFIX_FLAG)) or truthy(os.environ.get(CONTENT_FLAG))

    def invoke(args, kwargs):
        if prefix_enabled:
            with request_context(trace, mode, args, kwargs):
                return original(*args, **kwargs)
        return original(*args, **kwargs)
    signature = inspect.signature(original)
    role = 'API' if mode in {'api_chat', 'bind'} else 'EngineCore' if mode == 'engine_add' else 'EngineCore/Scheduler'

    def before(args, kwargs):
        trace.start(role, source, location)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        request = _request(bound)
        req_id, external_id = request_fields(request)
        selected = trace.selected_request(req_id)
        state = {'bound': bound, 'request': request, 'id': req_id, 'external': external_id,
                 'selected': selected, 'manager': bound.arguments.get('self')}
        if selected and external_id is not None:
            trace.external_ids[req_id] = external_id
        if selected and mode in {'allocate', 'free'}:
            try:
                state['before'] = trace.held(state['manager'], req_id)
            except Exception:
                state['before'] = None
                trace.failure()
        return state

    def after(state, result):
        if state is None:
            return
        request, req_id, external_id = state['request'], state['id'], state['external']
        if mode in {'bind', 'convert'}:
            # assign_request_id mutates IDs; conversion returns a new Request.
            target = request if mode == 'bind' else result
            new_id, new_external = request_fields(target)
            external_id = new_external or external_id
            # Source and destination conversion IDs must explicitly agree.
            if mode == 'convert' and (new_id is None or new_id != req_id):
                return
            if mode == 'bind' and trace.selected == req_id:
                trace.selected = new_id
            if mode == 'bind' and req_id in trace.selected_ids and new_id is not None:
                trace.selected_ids.discard(req_id)
                trace.selected_ids.add(new_id)
                token = trace.request_sessions.pop(req_id, None)
                if token is not None: trace.request_sessions[new_id] = token
                trace.external_ids.pop(req_id, None)
            if not trace.selected_request(new_id):
                return
            if external_id:
                trace.external_ids[new_id] = external_id
            trace.emit('REQUEST_ID_BOUND', role, source, new_id, external_id,
                       previous_request_id=req_id,
                       binding_kind='EXPLICIT_FIELDS_AFTER_ASSIGN' if mode == 'bind' else 'EXPLICIT_FIELDS_ACROSS_CONVERSION',
                       reason={} if external_id else {'external_request_id': 'No explicit external_req_id field available'})
            context = _API_CONTEXT.get()
            if context is not None and mode == 'bind':
                context['bindings'].append((new_id, external_id))
            if mode == 'convert':
                trace.input_observer.observe(target, lambda event, **fields:
                    trace.emit(event, role, source, new_id, external_id, **fields))
            return
        if not state['selected']:
            return
        manager = state['manager']
        if mode in {'engine_add', 'scheduler_add'}:
            trace.emit('ENGINE_REQUEST_ADDED' if mode == 'engine_add' else 'SCHEDULER_REQUEST_ADDED',
                       role, source, req_id, trace.external_ids.get(req_id),
                       request_status=_status(request))
        elif mode == 'lookup':
            # 原函数返回的是前缀查询候选；lookup 本身不证明本请求已附着引用。
            if not isinstance(result, tuple) or len(result) != 3:
                raise TypeError('expected evidenced three-item lookup return')
            blocks, computed, boundary = result
            trace.block_event('KV_CACHE_LOOKUP', source, req_id, manager, trace.groups(blocks),
                              block_event_kind='PREFIX_LOOKUP_RESULT',
                              reuse_kind='PREFIX_HIT_CANDIDATE' if integer(computed) and computed > 0 else 'NO_COMPUTED_TOKEN_HIT',
                              num_computed_tokens_if_available=integer(computed),
                              shared_prefix_boundary_if_available=integer(boundary),
                              reference_attached=False)
        elif mode == 'allocate':
            # 观察原函数返回与调用后持有视图；返回 None 也记录，不擅自归因为 OOM。
            # returned_new_block_ids 是逻辑分配结果，不能当作新 Tensor/物理 malloc。
            values = state['bound'].arguments
            held_after = trace.held(manager, req_id)
            before_groups = state.get('before')
            new_groups = trace.groups(result) if result is not None else None
            supplied_groups = trace.groups(values.get('new_computed_blocks'))
            for group in held_after or [{'group_id': None, 'block_ids': [], 'blocks_truncated': False, 'total_block_count': None}]:
                index = group['group_id']
                def ids(groups):
                    return groups[index]['block_ids'] if groups is not None and index is not None and index < len(groups) else None
                supplied = ids(supplied_groups)
                attached = [b for b in supplied or [] if b in group['block_ids']] if result is not None else []
                trace.block_event('KV_ALLOCATION_UNAVAILABLE' if result is None else 'KV_BLOCKS_ASSIGNED',
                                  source, req_id, manager, [group],
                                  block_event_kind='ALLOCATION_RETURNED_NONE' if result is None else 'REQUEST_HOLDINGS_AFTER_ALLOCATION',
                                  returned_new_block_ids=ids(new_groups), before_block_ids=ids(before_groups),
                                  supplied_prefix_block_ids=supplied, attached_prefix_candidate_ids=attached,
                                  reuse_kind='PREFIX_CANDIDATES_PRESENT_IN_HOLDINGS' if attached else 'NO_PREFIX_ATTACHMENT_OBSERVED',
                                  num_new_tokens_if_available=integer(values.get('num_new_tokens')),
                                  num_computed_tokens_if_available=integer(values.get('num_new_computed_tokens')),
                                  allocation_result='NONE' if result is None else 'KVCacheBlocks',
                                  new_blocks_semantics='Coordinator.allocate_new_blocks return; not backing allocation bytes',
                                  attachment_status='INFERRED_BY_SAME_GROUP_ID_INTERSECTION' if attached else 'NOT_OBSERVED',
                                  before_snapshot_observed=before_groups is not None)
        elif mode == 'free':
            # 比较 free 前后请求持有集合；引用解除不等于底层 HBM Buffer 被销毁。
            try:
                after_groups = trace.held(manager, req_id)
                after_error = None
            except Exception as exc:
                after_groups = None
                after_error = type(exc).__name__
                trace.failure()
            groups = state.get('before')
            for group in groups or [{'group_id': None, 'block_ids': [], 'blocks_truncated': False, 'total_block_count': None}]:
                index = group['group_id']
                after_ids = after_groups[index]['block_ids'] if after_groups is not None and index is not None and index < len(after_groups) else None
                trace.block_event('KV_REQUEST_DETACHED_OR_FREED', source, req_id, manager, [group],
                                  block_event_kind='REQUEST_REFERENCE_FREE_CALL_COMPLETED',
                                  after_block_ids=after_ids, after_snapshot_error_type=after_error,
                                  after_blocks_truncated=after_groups[index]['blocks_truncated'] if after_groups is not None and index is not None and index < len(after_groups) else None,
                                  after_total_block_count=after_groups[index]['total_block_count'] if after_groups is not None and index is not None and index < len(after_groups) else None,
                                  detached_ids_observed=[b for b in group['block_ids'] if b not in after_ids] if after_ids is not None else None,
                                  physical_release_proven=False, eviction_proven=False,
                                  before_snapshot_observed=groups is not None,
                                  partial_tail_pins='free() also releases pins; pin IDs/refcounts NOT_OBSERVED')
        elif mode == 'scheduler_free':
            status = _status(request)
            if status and status.startswith('FINISHED'):
                trace.emit('REQUEST_FINISHED', role, source, req_id, trace.external_ids.get(req_id),
                           request_status=status, terminal_evidence='Explicit Request.status enum name after _free_request',
                           suppressed_unchanged_holdings_total=trace.suppressed_holdings)
                trace.retire(req_id)

    def failed(state, exc):
        if state and state['selected']:
            trace.emit('ORIGINAL_CALL_EXCEPTION', role, source, state['id'], trace.external_ids.get(state['id']),
                       exception_type=type(exc).__name__, exception_message_omitted=True)

    def safe(callback, *args, **kwargs):
        # 观察失败只记 health；原函数异常仍由外层原样抛出，不能被采集逻辑吞掉。
        try:
            return callback(*args, **kwargs)
        except Exception:
            trace.failure()
            return None

    def link_api(context, result):
        api_id = identity(getattr(result, 'id', None))
        if api_id is None and isinstance(result, dict):
            api_id = identity(result.get('id'))
        for internal_id, external_id in context['bindings']:
            trace.emit('API_REQUEST_LINKED', 'API', source, internal_id, external_id,
                       api_request_id=api_id, binding_kind='SAME_ASYNC_REQUEST_CONTEXT_AND_ACTUAL_RESPONSE_ID',
                       reason={} if api_id else {'api_request_id': 'Streaming/error response has no observed id; use controlled non-streaming smoke'})
            trace.retire(internal_id)

    if inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def wrapped(*args, **kwargs):
            if not trace.observing():
                return await original(*args, **kwargs)
            if mode != 'api_chat':
                state = safe(before, args, kwargs)
                try:
                    result = await original(*args, **kwargs)
                except BaseException as exc:
                    safe(failed, state, exc)
                    raise
                safe(after, state, result)
                return result
            safe(trace.start, 'API', source, location)
            context = {'bindings': []}
            token = _API_CONTEXT.set(context)
            try:
                result = await original(*args, **kwargs)
                safe(link_api, context, result)
                return result
            finally:
                for internal_id, _ in context['bindings']: safe(trace.retire, internal_id)
                _API_CONTEXT.reset(token)
    else:
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if not trace.observing():
                return invoke(args, kwargs)
            state = safe(before, args, kwargs)
            try:
                result = invoke(args, kwargs)
            except BaseException as exc:
                safe(failed, state, exc)
                raise
            safe(after, state, result)
            return result
    wrapped.__signature__ = signature
    wrapped.__agentmon0271__ = True
    return wrapped


class Installation:
    def __init__(self, config, *, test_only=False):
        self.trace = Trace(config)
        self.test_only = test_only
        self.patches = []
        self.finder = None
        self.kv_runtime_validation = None
        from .prefix_observer import PrefixObserver, FLAG as PREFIX_FLAG, MODULE as PREFIX_MODULE
        from .content_observer import ContentObserver, FLAG as CONTENT_FLAG
        content_enabled = truthy(os.environ.get(CONTENT_FLAG))
        self.prefix = PrefixObserver(self.trace) if truthy(os.environ.get(PREFIX_FLAG)) or content_enabled else None
        self.content = ContentObserver(self.trace, self.prefix) if content_enabled else None
        self.modules = set(TARGETS) | ({PREFIX_MODULE} if self.prefix else set())

    def patch_class(self, cls, method_name, mode, *, strict_kv=False):
        descriptor = inspect.getattr_static(cls, method_name)
        original = descriptor.__func__ if isinstance(descriptor, (staticmethod, classmethod)) else descriptor
        if getattr(original, '__agentmon0271__', False):
            return
        if not callable(original) or inspect.isasyncgenfunction(original):
            raise TypeError('unsupported hook target')
        signature = inspect.signature(original)
        if strict_kv and tuple(signature.parameters) != KV_SIGNATURES[method_name]:
            raise TypeError('KV signature mismatch')
        if mode in {'bind', 'engine_add', 'scheduler_add', 'scheduler_free', 'lookup', 'allocate', 'free'} and 'request' not in signature.parameters:
            raise TypeError('request parameter missing')
        location = _location(original, mode)
        location['loaded_code_matches_pinned_source'] = strict_kv and not self.test_only
        if strict_kv and self.kv_runtime_validation is not None:
            location['kv_runtime_validation'] = self.kv_runtime_validation
        wrapped = _wrapper(original, mode, self.trace, cls.__module__+'.'+cls.__name__+'.'+method_name, location)
        replacement = staticmethod(wrapped) if isinstance(descriptor, staticmethod) else classmethod(wrapped) if isinstance(descriptor, classmethod) else wrapped
        setattr(cls, method_name, replacement)
        self.patches.append((cls, method_name, descriptor, replacement))

    def patch_module(self, module):
        from .prefix_observer import MODULE as PREFIX_MODULE
        if module.__name__ == PREFIX_MODULE and self.prefix:
            stage = 'PREFIX_TARGET_REJECTED'
            try:
                self.prefix.patch_module(module)
                if self.content:
                    stage = 'CONTENT_TARGET_REJECTED'
                    self.content.patch_module(module)
            except Exception as exc:
                self.trace.emit(stage, 'bootstrap', PREFIX_MODULE,
                                error_type=type(exc).__name__, observation_status='UNRESOLVED')
            return
        stage, target = 'TARGET_INSTALL', None
        try:
            if module.__name__ == KV_MODULE and not self.test_only:
                stage = 'KV_FILE_SHA256'
                path = Path(module.__file__)
                if hashlib.sha256(path.read_bytes()).hexdigest() != KV_FILE_SHA256:
                    raise ValueError('KV file SHA256 differs from evidenced server build')
                compiled = compile(path.read_bytes(), str(path), 'exec', dont_inherit=True, optimize=sys.flags.optimize)
                def code_for(parent, name):
                    return next(item for item in parent.co_consts if inspect.iscode(item) and item.co_name == name)
                manager_code = code_for(compiled, 'KVCacheManager')
                for name, parameters in KV_SIGNATURES.items():
                    stage, target = 'MANAGER_SIGNATURE', 'KVCacheManager.' + name
                    method = getattr(module.KVCacheManager, name)
                    if tuple(inspect.signature(method).parameters) != parameters:
                        raise TypeError('KV source signature mismatch')
                    stage = 'MANAGER_LOADED_CODE'
                    if method.__code__ != code_for(manager_code, name):
                        raise ValueError('loaded KV code differs from compiled pinned disk source')
                stage, target = 'GETTER_CAPABILITY', 'KVCacheBlocks.get_block_ids'
                # The getter may have overload stubs with the same co_name.
                # Verify its actual metadata contract, not the first code constant.
                self.kv_runtime_validation = validate_block_getter(module)
            stage = 'TARGET_INSTALL'
            for class_name, method_name, mode in TARGETS[module.__name__]:
                target = class_name + '.' + method_name
                cls = getattr(module, class_name, None)
                if cls is None or not hasattr(cls, method_name):
                    self.trace.emit('HOOK_TARGET_UNAVAILABLE', 'bootstrap', module.__name__+'.'+class_name+'.'+method_name,
                                    observation_status='NOT_OBSERVED')
                    continue
                if mode == 'scheduler_finish':
                    # Signature/target evidence only. _free_request observes the
                    # actual terminal Request; finish_requests may be an abort batch.
                    self.trace.emit('HOOK_TARGET_LOCATED', 'bootstrap', module.__name__+'.'+class_name+'.'+method_name,
                                    loaded_target=_location(getattr(cls, method_name), mode), wrapped=False)
                    continue
                self.patch_class(cls, method_name, mode, strict_kv=module.__name__ == KV_MODULE)
            if module.__name__ == KV_MODULE and self.kv_runtime_validation is not None:
                self.trace.emit('HOOK_TARGET_LOCATED', 'bootstrap', module.__name__ + '.KVCacheBlocks.get_block_ids',
                                wrapped=False, kv_runtime_validation=self.kv_runtime_validation,
                                manager_wrappers_installed=True, request_budget_scope='ARM_SESSION',
                                max_requests_per_process=self.trace.config.max_requests_per_process)
        except Exception as exc:
            self.trace.emit('HOOK_MODULE_REJECTED', 'bootstrap', module.__name__,
                            error_type=type(exc).__name__, observation_status='UNRESOLVED',
                            validation_stage=stage, validation_target=target,
                            rejection_reason=exc.reason if isinstance(exc, GetterCompatibilityError) else stage + '_REJECTED')

    def uninstall(self):
        if self.content:
            self.content.uninstall()
        if self.prefix:
            self.prefix.uninstall()
        if self.finder in sys.meta_path:
            sys.meta_path.remove(self.finder)
        for cls, name, original, replacement in reversed(self.patches):
            if inspect.getattr_static(cls, name) is replacement:
                setattr(cls, name, original)
        self.patches.clear()
        self.trace.close()


class _Loader(importlib.abc.Loader):
    def __init__(self, wrapped, installation):
        self.wrapped = wrapped
        self.installation = installation

    def create_module(self, spec):
        method = getattr(self.wrapped, 'create_module', None)
        return method(spec) if method else None

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        # Never hide an exception from the real module's import itself.
        try:
            self.installation.patch_module(module)
        except Exception:
            self.installation.trace.failure()


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, installation):
        self.installation = installation

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.installation.modules:
            return None
        # Delegate only FORWARD from our position, retaining downstream plugins.
        # sitecustomize installs logical first, then placement at index 0:
        # [placement, logical, ...]. Scanning every finder except ourselves
        # would call placement again and recurse placement -> logical -> placement.
        # Both interceptors must use the same forward-only traversal; do not
        # replace sys.meta_path, monkey-patch another finder, or bypass plugins.
        spec = None
        found = False
        for finder in tuple(sys.meta_path):
            if finder is self:
                found = True
                continue
            if not found:
                continue
            method = getattr(finder, 'find_spec', None)
            if method:
                spec = method(fullname, path, target)
                if spec is not None:
                    break
        if spec and spec.loader and hasattr(spec.loader, 'exec_module'):
            spec.loader = _Loader(spec.loader, self.installation)
        return spec


def install_from_env(*, startup=False):
    """Arm future imports from sitecustomize. Loaded vLLM modules are rejected."""
    global _ACTIVE
    if not truthy(os.environ.get(FLAG)):
        return None
    if _ACTIVE is not None:
        return _ACTIVE
    if not startup:
        raise RuntimeError('startup only: no hot installation into a running engine')
    if truthy(os.environ.get('AGENTMON_VLLM_P2_ENABLE')):
        raise RuntimeError('cannot combine vllm_018 P2 and vllm_0271 tracing')
    from .prefix_observer import FLAG as PREFIX_FLAG, MODULE as PREFIX_MODULE
    from .content_observer import FLAG as CONTENT_FLAG
    modules = set(TARGETS) | ({PREFIX_MODULE} if truthy(os.environ.get(PREFIX_FLAG)) or truthy(os.environ.get(CONTENT_FLAG)) else set())
    if any(name in sys.modules for name in modules):
        raise RuntimeError('target module already loaded; startup import coverage cannot be proven')
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    if versions != EXPECTED_VERSIONS:
        raise RuntimeError('runtime versions differ from evidenced custom build')
    # 常态化采集可显式 MAX_REQUESTS=0；队列、快照截断和磁盘增长仍需独立监控。
    # 这些启动参数没有现成的运行时重配置接口，ARM 会话切换只重置计数。
    config = TraceConfig(Path(os.environ['AGENTMON_VLLM_0271_TRACE_DIR']),
                         os.environ['AGENTMON_VLLM_0271_TRACE_RUN_ID'],
                         max_blocks_per_group=int(os.environ.get('AGENTMON_VLLM_0271_MAX_BLOCKS', '256')),
                         queue_capacity=int(os.environ.get('AGENTMON_VLLM_0271_QUEUE_CAPACITY', '512')),
                         holdings_checkpoint_interval=int(os.environ.get('AGENTMON_VLLM_0271_HOLDINGS_CHECKPOINT', '32')),
                         max_file_bytes=int(os.environ.get('AGENTMON_VLLM_0271_MAX_FILE_BYTES', str(16*1024*1024))),
                         max_requests_per_process=int(os.environ.get('AGENTMON_VLLM_0271_MAX_REQUESTS', '1')),
                         arm_file=Path(os.environ.get('AGENTMON_VLLM_0271_TRACE_ARM_FILE',
                                       str(Path(os.environ['AGENTMON_VLLM_0271_TRACE_DIR'])/'ARMED'))))
    installation = Installation(config)
    installation.finder = _Finder(installation)
    sys.meta_path.insert(0, installation.finder)
    atexit.register(installation.trace.close)
    _ACTIVE = installation
    return installation


def uninstall():
    """Only reverses this process's own wrappers; not a remote detach operation."""
    global _ACTIVE
    if _ACTIVE:
        _ACTIVE.uninstall()
        _ACTIVE = None


def install_for_isolated_test(config, targets):
    """Explicit Python test API, unavailable through production environment flags."""
    if config.evidence_status != 'SYNTHETIC_ISOLATED':
        raise ValueError('test installation must be labelled SYNTHETIC_ISOLATED')
    installation = Installation(config, test_only=True)
    for cls, name, mode in targets:
        installation.patch_class(cls, name, mode)
    return installation
