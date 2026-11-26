"""BlockPool queue observations appended to the reviewed boundary module."""


def importance_queue_edges(pool):
    queue = pool.free_block_queue
    head = queue.fake_free_list_head.next_free_block
    tail = queue.fake_free_list_tail.prev_free_block
    return dict(free=queue.num_free_blocks,
                head=None if head is queue.fake_free_list_tail else head.block_id,
                tail=None if tail is queue.fake_free_list_head else tail.block_id)


def importance_queue_checkpoint(pool):
    """Capture once at construction, after reserving the null slot."""
    try:
        pool._am_queue_seq = 0
        queue = pool.free_block_queue
        edges = importance_queue_edges(pool)
        # Traverse once, bounded by the physical slot count even on corruption.
        ids = []
        cursor = queue.fake_free_list_head.next_free_block
        while cursor is not queue.fake_free_list_tail and len(ids) < pool.num_gpu_blocks:
            if cursor.ref_cnt or cursor.is_null or cursor.block_hash is not None:
                raise ValueError('INITIAL_FREE_QUEUE_NOT_COLD')
            ids.append(cursor.block_id)
            cursor = cursor.next_free_block
        if (cursor is not queue.fake_free_list_tail or len(ids) != edges['free']
                or len(ids) != pool.num_gpu_blocks - 1 or len(set(ids)) != len(ids)
                or pool.null_block.block_id in ids or pool._reuse_watchers
                or pool._agentmon_research_leases or pool.cached_block_hashes_by_block
                or len(pool.cached_block_hash_to_block)):
            raise ValueError('CONSTRUCTION_QUEUE_CHECKPOINT_MISMATCH')
        parts = max(1, (len(ids) + 255) // 256)
        for part in range(parts):
            observe('IMPORTANCE_QUEUE_CHECKPOINT', pool_scope=pool._agentmon_research_scope,
                    queue_seq=0, part=part, parts=parts, capacity=pool.num_gpu_blocks,
                    null_slot=pool.null_block.block_id, initial_indexed=0,
                    initial_watched=0, delta_columns=['block_id', 'prior_or_current_lease',
                        'refcount_at_queue_edit', 'reuse_watcher_registered'],
                    ordered_slots=ids[256*part:256*(part+1)], **edges)
    except Exception as exc:
        research_gap('IMPORTANCE_QUEUE_CHECKPOINT', exc,
                     pool_scope=getattr(pool, '_agentmon_research_scope', None))


def importance_queue_request(pool, count):
    """Decision-time facts before the original allocation feasibility check."""
    try:
        pool._am_queue_seq += 1
        seq = pool._am_queue_seq
        try:
            request_id = research_request().request_id
        except ValueError:
            request_id = None
        observe('IMPORTANCE_QUEUE_REQUEST', pool_scope=pool._agentmon_research_scope,
                queue_seq=seq, request_id=request_id, requested_blocks=count,
                candidate_order='FREE_QUEUE_HEAD_FIRST', **importance_queue_edges(pool))
        return seq
    except Exception as exc:
        research_gap('IMPORTANCE_QUEUE_REQUEST', exc,
                     pool_scope=getattr(pool, '_agentmon_research_scope', None))
        return None


def importance_queue_edit(pool, action, blocks, decision=None):
    """Record actual mutations, before allocation increments the old lease/ref."""
    try:
        if not blocks and action not in ('ALLOCATE_POP', 'INSUFFICIENT_FREE', 'RESET_INDEX'):
            return
        pool._am_queue_seq += 1
        seq = pool._am_queue_seq
        parts = max(1, (len(blocks) + 127) // 128)
        edges = importance_queue_edges(pool)
        for part in range(parts):
            rows = [[b.block_id, pool._agentmon_research_leases.get(b.block_id),
                     b.ref_cnt, b.block_id in pool._reuse_watchers]
                    for b in blocks[128 * part:128 * (part+1)]]
            observe('IMPORTANCE_QUEUE_EDIT', pool_scope=pool._agentmon_research_scope,
                    queue_seq=seq, request_seq=decision, action=action, part=part,
                    parts=parts, slot_count=len(blocks), slots=rows, **edges)
    except Exception as exc:
        research_gap('IMPORTANCE_QUEUE_EDIT', exc, action=action,
                     pool_scope=getattr(pool, '_agentmon_research_scope', None))
