"""Observe request Block tables and live V4.1 backing views at Worker steps.

No Tensor values, physical addresses, device copies or extra synchronization.
Only weak references are retained: the observer must not keep KV storage alive.
"""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import uuid
import weakref

CORE = 'vllm_ascend.core.deepseek_v41'
CORE_SHA = 'd0d89cb4a0ef22672e1de92a80a2846ac86f8fd4ec8e7d14ab050bf1539d7acf'
RUNNER_SHA = '67035d97f1cea4ae2df31adcc33f1de952f4cab6d8421e76df512296e0e3185e'
GPU_SHA = '993f2c926012241bcb8cc7f568c8e82968e4d81ca125641c831ecf501aac02ac'
SPEC_TYPES = {'DeepseekV41FullSpec', 'DeepseekV41IndexerSpec', 'DeepseekV41SWASpec',
              'DeepseekV41DraftSWASpec', 'DeepseekV41CompressorStateSpec'}
CONTRACT = 'ASCEND_V41_BLOCK_AXIS0_STRIDED_VIEW_V1'


def verify_function(function, expected_sha, class_name=None):
    function = inspect.unwrap(function)
    path = Path(inspect.getsourcefile(function)).resolve()
    raw = path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(f'request media source SHA mismatch: {path}; expected={expected_sha}; actual={actual_sha}')
    code = compile(raw, str(path), 'exec', dont_inherit=True, optimize=sys.flags.optimize)
    if class_name:
        code = next(c for c in code.co_consts if inspect.iscode(c) and c.co_name == class_name)
    expected = next(c for c in code.co_consts if inspect.iscode(c) and c.co_name == function.__name__)
    if function.__code__ != expected:
        raise ValueError(f'request media loaded code differs from source: {path}; function={function.__qualname__}')
    return {'file': str(path), 'sha256': expected_sha, 'function': function.__qualname__}


def verify_runtime_contract(owner):
    core = sys.modules[CORE]
    gpu = sys.modules['vllm.v1.worker.gpu_model_runner']
    runner = sys.modules['vllm_ascend.worker.model_runner_v1']
    if runner.GPUModelRunner is not gpu.GPUModelRunner or type(owner) is not runner.NPUModelRunner:
        raise ValueError('unreviewed Worker inheritance or parent binding')
    actual = inspect.unwrap(owner._reshape_kv_cache_tensors)
    if getattr(actual, '__func__', actual) is not inspect.unwrap(runner.NPUModelRunner._reshape_kv_cache_tensors):
        raise ValueError('unreviewed Worker reshape override')
    # Check the actual function resolved by the runner, not merely a disk file.
    if runner.reshape_cache is not core.reshape_cache:
        raise ValueError('runner reshape_cache binding changed')
    return [verify_function(core.reshape_cache, CORE_SHA),
        verify_function(runner.NPUModelRunner._reshape_kv_cache_tensors, RUNNER_SHA, 'NPUModelRunner'),
        verify_function(gpu.GPUModelRunner._update_states, GPU_SHA, 'GPUModelRunner')]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def describe_plane(tensor, spec, num_blocks, plane):
    """Validate the actual view against pinned reshape_cache's axis-0 rule."""
    # 第0维就是实际Block索引；偏移相对于storage，不是HBM物理地址。
    # stride可能大于一个Layer的有效字节，不能用block_id * payload代替。
    cls = type(spec)
    if cls.__module__ != CORE or cls.__name__ not in SPEC_TYPES:
        raise ValueError('unsupported cache spec')
    indexer = cls.__name__ == 'DeepseekV41IndexerSpec'
    if plane not in (0, 1) or (plane == 1 and not indexer):
        raise ValueError('unsupported plane')
    width = spec.head_size if plane == 0 else spec.scale_dim
    dtype = spec.dtype if plane == 0 else spec.scale_dtype
    expected = [num_blocks, spec.storage_block_size, spec.num_kv_heads, width]
    shape = [int(n) for n in tensor.shape]
    stride = [int(n) for n in tensor.stride()]
    element = int(tensor.element_size())
    offset = int(tensor.storage_offset()) * element
    extent = int(tensor.untyped_storage().nbytes())
    if (shape != expected or str(tensor.dtype) != str(dtype) or len(stride) != 4 or
        stride[1:] != [spec.num_kv_heads * width, width, 1] or element <= 0 or offset < 0):
        raise ValueError('observed tensor disagrees with V4.1 plane layout')
    payload = spec.storage_block_size * spec.num_kv_heads * width * element
    step = stride[0] * element
    if step < payload or offset + (num_blocks-1)*step + payload > extent:
        raise ValueError('block view outside storage extent')
    device = str(tensor.device)
    kind = device.split(':', 1)[0]
    medium = {'npu': 'HBM', 'cpu': 'DDR', 'cuda': 'DEVICE_RAM'}.get(kind)
    return {'contract': CONTRACT, 'block_axis': 0, 'num_blocks': num_blocks,
        'shape': shape, 'stride_elements': stride, 'element_bytes': element,
        'storage_offset_bytes': offset, 'block_stride_bytes': step,
        'block_payload_bytes': payload, 'storage_extent_bytes': extent,
        'dtype': str(tensor.dtype), 'device': device, 'device_status': 'VERIFIED',
        'medium': medium, 'medium_status': 'INFERRED' if medium else 'NOT_OBSERVED',
        'medium_basis': 'OBSERVED_TENSOR_BACKEND_ADDRESS_SPACE; NOT_OS_PAGE_RESIDENCY',
        'resource_kind': cls.__name__, 'plane_index': plane,
        'layout_status': 'VERIFIED', 'physical_address': None}


class RequestMediaObserver:
    def __init__(self, observer, max_blocks_per_group=16384):
        self.observer = observer
        if type(max_blocks_per_group) is not int or max_blocks_per_group <= 0:
            raise ValueError('positive observation bound required')
        self.max_blocks = max_blocks_per_group
        self.owners = weakref.WeakKeyDictionary()
        self.storage_registry = observer.storage_registry

    def storage_id(self, tensor):
        return self.storage_registry.describe(tensor)['storage_object_id']

    def register(self, owner, config):
        evidence = verify_runtime_contract(owner)
        groups = []
        for group_id, group in enumerate(config.kv_cache_groups):
            members = getattr(group.kv_cache_spec, 'kv_cache_specs', None)
            for layer in group.layer_names:
                spec = members[layer] if isinstance(members, dict) else group.kv_cache_spec
                groups.append((group_id, layer, spec))
        blocks = config.num_blocks
        if type(blocks) is not int or blocks <= 0 or not groups:
            raise ValueError('Worker cache configuration incomplete')
        state = {'owner_scope': uuid.uuid4().hex, 'groups': groups, 'num_blocks': blocks,
                 'group_count': len(config.kv_cache_groups), 'layout_id': None,
                 'layout_digest': None, 'views': {}, 'step': 0}
        self.owners[owner] = state
        self.observer.emit('KV_MEDIA_CONTRACT_VERIFIED', 'RequestMediaObserver.register',
            owner_scope=state['owner_scope'], contract=CONTRACT, source_bindings=evidence,
            storage_registry_id=self.storage_registry.registry_id,
            storage_identity_scope='PROCESS_LIVE_STORAGE_REGISTRY_V1',
            observation_point='AFTER_WORKER_UPDATE_STATES_BEFORE_FORWARD',
            request_table='owner.requests[request_id].block_ids',
            group_count=state['group_count'], num_blocks=blocks,
            rank=getattr(getattr(owner.vllm_config,'parallel_config',None),'rank',None))

    def observe_step(self, owner, scheduler_output):
        # 在原_update_states已经完成后采集，不修改请求表、不接触Tensor内容。
        # 每步记录紧凑引用；只有布局/Block表变化才重写详细视图。
        state = self.owners.get(owner)
        if state is None:
            self.observer.emit('KV_MEDIA_STEP_UNAVAILABLE', 'RequestMediaObserver.observe_step',
                               reason='INITIAL_MEDIA_CONTRACT_NOT_REGISTERED')
            return
        state['step'] += 1
        rank = getattr(getattr(owner.vllm_config, 'parallel_config', None), 'rank', None)
        planes, errors = [], []
        context = owner.compilation_config.static_forward_context
        for group, layer, spec in state['groups']:
            try:
                entries = context[layer].kv_cache
                if not isinstance(entries, (list, tuple)) or len(entries) != 1:
                    raise ValueError('unsupported virtual engine binding')
                value = entries[0]
                tensors = value if isinstance(value, tuple) else (value,)
                expected_planes = 2 if type(spec).__name__ == 'DeepseekV41IndexerSpec' else 1
                if len(tensors) != expected_planes:
                    raise ValueError('plane count mismatch')
                for index, tensor in enumerate(tensors):
                    plane = describe_plane(tensor, spec, state['num_blocks'], index)
                    planes.append({**plane, 'worker_group_id': group, 'layer_name': layer,
                                   **self.storage_registry.describe(tensor)})
            except Exception as exc:
                errors.append({'worker_group_id': group, 'layer_name': layer, 'reason': type(exc).__name__})
        geometry = {'planes': planes, 'errors': errors}
        current = digest(geometry)
        if current != state['layout_digest']:
            previous = state['layout_id']
            state['layout_digest'], state['layout_id'] = current, uuid.uuid4().hex
            self.observer.emit('KV_MEDIA_LAYOUT', 'RequestMediaObserver.observe_step',
                owner_scope=state['owner_scope'], layout_id=state['layout_id'], previous_layout_id=previous,
                rank=rank, group_count=state['group_count'], **geometry,
                change_semantics='BACKING_VIEW_CHANGE_NOT_PROOF_OF_CONTENT_TRANSFER')
        scheduled = scheduler_output.num_scheduled_tokens
        if not isinstance(scheduled, dict):
            raise TypeError('scheduled request identities not observed')
        references = []
        work=[]
        for req_id, tokens in scheduled.items():
            if type(tokens) is not int or tokens <= 0:
                continue
            request = owner.requests.get(req_id)
            if not isinstance(req_id, str) or request is None or request.req_id != req_id:
                raise ValueError('exact Worker request identity unavailable')
            blocks = request.block_ids
            if not isinstance(blocks, (tuple, list)) or len(blocks) != state['group_count']:
                raise ValueError('Worker Block group count mismatch')
            groups = []
            for index, ids in enumerate(blocks):
                if not isinstance(ids, (list, tuple)) or any(type(b) is not int or b < 0 for b in ids):
                    raise TypeError('Worker Block IDs must be CPU integer lists')
                groups.append({'worker_group_id': index, 'block_ids': list(ids[:self.max_blocks]),
                    'total_blocks': len(ids), 'truncated': len(ids) > self.max_blocks})
            signature = digest([state['layout_id'], groups])
            prior = state['views'].get(req_id)
            if prior is None or prior[0] != signature:
                view_id = uuid.uuid4().hex
                state['views'][req_id] = signature, view_id
                self.observer.emit('KV_REQUEST_MEDIA_VIEW', 'RequestMediaObserver.observe_step',
                    owner_scope=state['owner_scope'], request_id=req_id, view_id=view_id,
                    layout_id=state['layout_id'], groups=groups, rank=rank,
                    semantics='ACTUAL_WORKER_REQUEST_BLOCK_TABLE; NOT_CONTENT_GENERATION')
            references.append({'request_id': req_id, 'view_id': state['views'][req_id][1]})
            computed=getattr(request,'num_computed_tokens',None)
            prompt=getattr(request,'num_prompt_tokens',None)
            if type(prompt) is not int:
                prompt_ids=getattr(request,'prompt_token_ids',None)
                prompt=len(prompt_ids) if isinstance(prompt_ids,(list,tuple)) else None
            work.append({'request_id':req_id,'scheduled_tokens':tokens,
                         'computed_tokens_before_forward':computed if type(computed) is int and computed>=0 else None,
                         'prompt_tokens':prompt if type(prompt) is int and prompt>=0 else None})
        self.observer.emit('KV_MEDIA_STEP', 'RequestMediaObserver.observe_step',
            owner_scope=state['owner_scope'], step_index=state['step'], rank=rank,
            layout_id=state['layout_id'], request_views=references,
            scheduled_work=work,work_observation_schema='WORKER_SCHEDULED_CPU_METADATA_V1',
            work_semantics='SCHEDULED_NOT_PROOF_OF_DEVICE_COMPLETION; DO_NOT_SUM_TP_RANKS',
            observation_point='AFTER_WORKER_UPDATE_STATES_BEFORE_FORWARD',
            actual_attention_read='NOT_OBSERVED')
        # Request churn does not create an unbounded request cache in the Hook.
        state['views'] = {k: v for k, v in state['views'].items() if k in owner.requests}
