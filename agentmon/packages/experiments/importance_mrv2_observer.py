"""Opt-in Ascend MRV2 adapter, colocated with DeepSeek V4.1 model sources."""
import os
import time
import uuid
from vllm.v1.kv_offload.agentmon_boundary import observe, research_gap


def bind_request(state, slot, new_request):
    try:
        if not hasattr(state, '_am_v2_ids'):
            state._am_v2_ids = {}
        state._am_v2_ids[int(slot)] = new_request.req_id
    except Exception as exc:
        getattr(state, '_am_v2_ids', {}).pop(int(slot), None)
        research_gap('MRV2_BIND_REQUEST', exc)


def begin_batch(state, batch, capture, ubatch, direction):
    try:
        state._am_v2_ctx = None
        if state.decoder_replay_state is not None:
            state.decoder_replay_state._am_v2_ctx = None
            state.decoder_replay_state.layers._am_v2_active = None
        if capture or batch.is_dummy:
            return
        poll(state)
        context = dict(id=uuid.uuid4().hex, ubatch=int(ubatch), rank=os.environ.get('RANK'),
                       device_ordinal=getattr(state.device, 'index', None),
                       slots=getattr(state, '_am_v2_ids', {}), owner=state, mode=str(direction))
        state._am_v2_ctx = context
        if state.decoder_replay_state is not None:
            state.decoder_replay_state._am_v2_ctx = context
    except Exception as exc:
        research_gap('MRV2_BEGIN_BATCH', exc)


def common(context, branch):
    return dict(worker_batch_id=context['id'], ubatch=context['ubatch'], rank_env=context['rank'],
                device_ordinal=context['device_ordinal'],
                graph_mode=context['mode'], branch=branch)


def poll(state):
    pending = getattr(state, '_am_v2_timers', [])
    live = []
    for identity, start, end in pending:
        try:
            if end.query():
                observe('KV_IMPORTANCE_V2_BRANCH_STREAM_TIME', **identity, device_ms=start.elapsed_time(end),
                        scope='CURRENT_STREAM_ONLY', synchronized=False)
            else:
                live.append((identity, start, end))
        except Exception as exc:
            research_gap('MRV2_TIMER_POLL', exc, **identity)
    state._am_v2_timers = live


def start_call(context, branch):
    if context is None:
        return None
    identity = common(context, branch)
    timer = None
    try:
        owner = context['owner']
        count = getattr(owner, '_am_v2_call_count', {}).get(branch, 0)
        if not hasattr(owner, '_am_v2_call_count'):
            owner._am_v2_call_count = {}
        owner._am_v2_call_count[branch] = count + 1
        if count % 128 == 0:
            if len(owner._am_v2_timers) < 64:
                import torch
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                timer = start, end
            else:
                observe('KV_IMPORTANCE_V2_TIMER_UNAVAILABLE', **identity, reason='PENDING_LIMIT')
    except Exception as exc:
        research_gap('MRV2_TIMER_START', exc, **identity)
    return (identity, time.monotonic_ns(), timer)


def end_call(context, call):
    if call is None:
        return
    now = time.monotonic_ns()
    identity, start_ns, timer = call
    if timer is not None:
        try:
            timer[1].record()
            context['owner']._am_v2_timers.append((identity, *timer))
        except Exception as exc:
            research_gap('MRV2_TIMER_END', exc, **identity)
    observe('KV_IMPORTANCE_V2_BRANCH_RETURN', **identity, host_ns=now-start_ns,
            device_sample_requested=timer is not None,
            timing_scope='WORKER_HOST_CALL_NO_SYNCHRONIZATION')


def encoder_enter(state):
    try:
        state._am_v2_encoder_call = start_call(state._am_v2_ctx, 'PREFIX_CACHEABLE_GROUP_SLOT_MASK_KERNEL')
    except Exception as exc:
        research_gap('MRV2_ENCODER_ENTER', exc)


def encoder_return(state, batch, replay_starts, window, groups, pcp):
    try:
        ctx = state._am_v2_ctx
        if ctx is None:
            return
        end_call(ctx, state._am_v2_encoder_call)
        state._am_v2_encoder_call = None
        rows = []
        for i in range(batch.num_reqs):
            slot = int(batch.idx_mapping_np[i])
            start = int(replay_starts[i])
            end = int(batch.seq_lens_np[i])
            length = int(batch.query_start_loc_np[i+1]) - int(batch.query_start_loc_np[i])
            left, right = max(end-length, start), min(end, start+window)
            rows.append(dict(request_id=ctx['slots'].get(slot), request_slot=slot,
                host_scheduled_span=[end-length,end], mask_span=[left,right] if start > 0 and right>left else None,
                post_replay_span=[max(end-length,start+window),end] if start>0 and end>max(end-length,start+window) else None))
        observe('KV_IMPORTANCE_V2_ENCODER_MASK', **common(ctx,'PREFIX_CACHEABLE_GROUP_SLOT_MASK_KERNEL'),
                rows=rows, window=int(window), cacheable_group_count=int(groups.numel()), pcp_global_batch=pcp,
                scope='HOST_RANGES_INTERSECT_ACTUAL_SUBMITTED_MASK; GLOBAL_CONTENT_EQUALITY_UNMEASURED')
    except Exception as exc:
        research_gap('MRV2_ENCODER_RETURN', exc)


def decoder_prepared(state, batch, kept, pcp):
    try:
        ctx = getattr(state, '_am_v2_ctx', None)
        if ctx is None:
            return
        rows = []
        for i in range(batch.num_reqs):
            slot = int(batch.idx_mapping_np[i])
            end = int(batch.seq_lens_np[i])
            n = int(kept[i])
            rows.append(dict(request_id=None if pcp else ctx['slots'].get(slot), request_slot=slot, kept_rows=n,
                             kept_span=None if pcp else [end-n,end]))
        state.layers._am_v2_active = (ctx,id(state.layers.rows),rows,pcp)
    except Exception as exc:
        research_gap('MRV2_DECODER_PREPARED', exc)


def decoder_enter(layers):
    try:
        active = getattr(layers, '_am_v2_active', None)
        layers._am_v2_decoder_call = None
        if active is None:
            return
        ctx, rows_id, rows, pcp = active
        if rows_id != id(layers.rows):
            raise ValueError('MRV2_DECODER_ROWS_BINDING_LOST')
        observe('KV_IMPORTANCE_V2_DECODER_ROWS', **common(ctx,'DECODER_TRIMMED_RUN_LAYERS'),
                rows=rows, actual_rows=int(layers.num_actual_rows), pcp_local_rows=pcp,
                timeline_scope='ABOUT_TO_RUN_TRIMMED_DECODER_LAYERS')
        layers._am_v2_decoder_call = start_call(ctx, 'DECODER_TRIMMED_RUN_LAYERS')
    except Exception as exc:
        research_gap('MRV2_DECODER_ENTER', exc)


def decoder_return(layers):
    try:
        if layers._am_v2_decoder_call is not None:
            end_call(layers._am_v2_active[0], layers._am_v2_decoder_call)
        layers._am_v2_decoder_call = None
    except Exception as exc:
        research_gap('MRV2_DECODER_RETURN', exc)
