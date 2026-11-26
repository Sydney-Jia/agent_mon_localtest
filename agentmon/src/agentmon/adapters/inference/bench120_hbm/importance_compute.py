"""Observation hooks appended to the reviewed HBM-only runtime boundary."""


def importance_request_begin(scheduler, request, origin):
    try:
        when = time.monotonic_ns()
        states = getattr(scheduler, '_am_importance_compute', None)
        if states is None:
            states = scheduler._am_importance_compute = {}
        previous = states.get(request.request_id)
        segment = previous['segment'] + 1 if previous is not None else 0
        states[request.request_id] = dict(segment=segment, enqueue_ns=when,
            first_dispatch_ns=None, emitted_first=False, pending=None, committed=None,
            query_call=None, groups={})
        observe('KV_IMPORTANCE_REQUEST_QUEUED', request_id=request.request_id,
                segment=segment, origin=origin, scheduler_monotonic_ns=when,
                clock_scope='SCHEDULER_PROCESS_MONOTONIC_NS')
    except Exception as exc:
        research_gap('IMPORTANCE_REQUEST_BEGIN', exc)


def importance_request_free(scheduler, request):
    try:
        states = getattr(scheduler, '_am_importance_compute', {})
        state = states.pop(request.request_id, None)
        if state is not None:
            observe('KV_IMPORTANCE_REQUEST_RELEASED', request_id=request.request_id,
                    segment=state['segment'], saw_first_accepted=state['emitted_first'],
                    scheduler_monotonic_ns=time.monotonic_ns())
    except Exception as exc:
        research_gap('IMPORTANCE_REQUEST_RELEASE', exc)


def importance_compute_query(scheduler, query, hit_length, window_need):
    """Only record results delivered by the manager inside this schedule call."""
    try:
        if scheduler is None:
            return
        state = scheduler._am_importance_compute[query['request_id']]
        call = scheduler._agentmon_research_call_id
        if state['query_call'] != call:
            state.update(query_call=call, groups={})
        gid = tuple(query['group_ids'])
        previous = state['groups'].get(gid)
        if previous is None or hit_length >= previous['hit_tokens']:
            state['groups'][gid] = dict(group_ids=list(gid), hit_tokens=hit_length,
                window_blocks_required=window_need, query_id=query['query_id'],
                manager_class=query['manager_class'])
    except Exception as exc:
        research_gap('IMPORTANCE_QUERY_RESULT', exc, request_id=query.get('request_id'))


def importance_prefix_selected(scheduler, request, local, external, async_load):
    try:
        state = scheduler._am_importance_compute[request.request_id]
        call = scheduler._agentmon_research_call_id
        groups = list(state['groups'].values()) if state['query_call'] == call else []
        full_hits = [g['hit_tokens'] for g in groups if g['window_blocks_required'] is None]
        window_hits = [g['hit_tokens'] for g in groups if g['window_blocks_required'] is not None]
        full = max(full_hits) if full_hits else None
        # Evidence of fallback applies only where a queried window supports
        # the actual final boundary. This never attributes a missing Key.
        observed_fallback = None
        if full is not None and window_hits and not external:
            observed_fallback = full > local and local in window_hits
        state['pending'] = dict(call=call, local=local, external=external,
            generation=request.num_preemptions, load_async=async_load,
            prefix_decision_id=scheduler._am_boundary_decision, group_candidates=groups,
            full_candidate_tokens=full, observed_full_to_window_fallback=observed_fallback)
        state['groups'] = {}
    except Exception as exc:
        research_gap('IMPORTANCE_PREFIX_SELECTED', exc, request_id=request.request_id)


def importance_compute_schedule(scheduler, output):
    try:
        batch = output._agentmon_research_batch
        now = time.monotonic_ns()
        rows = []
        for request_row in batch['requests']:
            rid = request_row['request_id']
            state = getattr(scheduler, '_am_importance_compute', {}).get(rid)
            if state is None:
                research_gap('IMPORTANCE_COMPUTE_SCHEDULE', ValueError('REQUEST_ENQUEUE_UNOBSERVED'),
                             request_id=rid, batch_id=batch['batch_id'])
                continue
            start = request_row['start_token']
            end = start + request_row['scheduled_tokens']
            prompt = request_row['prompt_tokens']
            first = state['first_dispatch_ns'] is None
            if first:
                state['first_dispatch_ns'] = now
            generation = request_row['recompute_generation']
            selected = state['pending']
            if selected is not None and selected['call'] == batch['schedule_call_id']:
                state['committed'] = selected
            state['pending'] = None
            selected = state['committed']
            if selected is not None and selected['generation'] != generation:
                selected = None
            # All prefills are retained. Pure generation shares the existing
            # complete dispatch stream; only the first or sampled ones repeat here.
            if start >= prompt and not first and not batch['full_capture']:
                continue
            prefix_path = 'DECISION_UNAVAILABLE'
            if selected is not None:
                if selected['external'] > 0:
                    prefix_path = 'EXTERNAL_TOKENS_SELECTED'
                elif selected['local'] > 0:
                    prefix_path = 'LOCAL_PREFIX_SELECTED'
                else:
                    prefix_path = 'ZERO_CACHED_TOKENS_SELECTED'
            fallback_range = None
            if selected is not None and selected['observed_full_to_window_fallback']:
                left = max(start, selected['local'])
                right = min(end, prompt, selected['full_candidate_tokens'])
                if right > left:
                    fallback_range = [left, right]
            rows.append(dict(request_id=rid, segment=state['segment'],
                request_decision_id=request_row['request_decision_id'],
                recompute_generation=generation, prefix_path=prefix_path,
                selection=selected, scheduled_span=[start, end],
                prompt_span=[start, min(end, prompt)] if start < prompt else None,
                fallback_prefill_span=fallback_range,
                post_prompt_span=[max(start, prompt), end] if end > prompt else None,
                scheduled_media_encoder_inputs=output.scheduled_encoder_inputs.get(rid, []),
                first_dispatch=first, enqueue_to_first_dispatch_ns=
                    (now - state['enqueue_ns']) if first else None))
        if rows:
            observe('KV_IMPORTANCE_COMPUTE_BATCH', batch_id=batch['batch_id'],
                    batch_sha256=batch['sha256'], schedule_call_id=batch['schedule_call_id'],
                    scheduler_monotonic_ns=now, batch_requests=len(batch['requests']),
                    full_device_sample_requested=batch['full_capture'], requests=rows,
                    bounded_swa_replay='NOT_OBSERVED_NO_EXECUTION_BRANCH_IN_PINNED_PROFILE',
                    replay_span=None, new_suffix_span=None,
                    semantics='SCHEDULED_WORK; PREFIX_CANDIDATES_PER_GROUP_MAX_IN_SELECTION_CALL')
    except Exception as exc:
        research_gap('IMPORTANCE_COMPUTE_SCHEDULE', exc)


def importance_first_output(scheduler, request, output, before, is_stale):
    try:
        if (before != 0 or is_stale or not request._output_token_ids
                or request.status.name == 'FINISHED_ERROR'):
            return
        state = scheduler._am_importance_compute[request.request_id]
        if state['emitted_first']:
            return
        now = time.monotonic_ns()
        start = state['first_dispatch_ns']
        batch = output._agentmon_research_batch
        observe('KV_IMPORTANCE_FIRST_ACCEPTED', request_id=request.request_id,
                segment=state['segment'], batch_id=batch['batch_id'],
                batch_sha256=batch['sha256'], scheduler_monotonic_ns=now,
                enqueue_to_first_accepted_ns=now - state['enqueue_ns'],
                first_dispatch_to_first_accepted_ns=None if start is None else now - start,
                timing_scope='SCHEDULER_ENQUEUE_TO_ACCEPTED_OUTPUT; API_DELIVERY_UNOBSERVED')
        state['emitted_first'] = True
    except Exception as exc:
        research_gap('IMPORTANCE_FIRST_ACCEPTED', exc, request_id=request.request_id)


def importance_forward_stamp(runner, ident, begin):
    """Two cheap host stamps inside the existing sampled device timer."""
    if ident is None:
        return
    try:
        now = time.monotonic_ns()
        if begin:
            runner._am_importance_host_forward = [ident['forward_id'], now, None]
        else:
            record = runner._am_importance_host_forward
            if record[0] != ident['forward_id']:
                raise ValueError('FORWARD_ID_CHANGED')
            record[2] = now
    except Exception as exc:
        research_gap('IMPORTANCE_FORWARD_STAMP', exc, **ident)


def importance_forward_result(runner, ident, actual, padded, requests, capture_mode):
    if ident is None:
        return
    try:
        record = runner._am_importance_host_forward
        if record[0] != ident['forward_id'] or record[2] is None:
            raise ValueError('INCOMPLETE_HOST_FORWARD')
        observe('KV_IMPORTANCE_FORWARD_COST', **ident, actual_tokens=int(actual),
                padded_tokens=int(padded), batch_requests=int(requests),
                graph_mode=str(capture_mode), host_enter_ns=record[1], host_return_ns=record[2],
                host_call_ns=record[2] - record[1], clock_scope='WORKER_PROCESS_MONOTONIC_NS',
                device_time_join='RESEARCH_FORWARD_STREAM_TIME:BATCH_SHA_RANK_LAYOUT_EPOCH_FORWARD',
                compute_scope='SAMPLED_WHOLE_MODEL_FORWARD_HOST_CALL',
                consumer_ready_ns=None, component_incremental_ns=None)
        runner._am_importance_host_forward = None
    except Exception as exc:
        research_gap('IMPORTANCE_FORWARD_RESULT', exc, **ident)
