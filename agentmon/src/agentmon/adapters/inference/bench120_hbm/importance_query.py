"""Observation hooks embedded in the reviewed Engine boundary module."""
import hashlib
import sys
import uuid
import weakref


# Request.update_block_hashes() in this release only extends its native list.
# Keep one cursor per live Request; never retain a Request or reuse a string ID.
_importance_hash_cursors = weakref.WeakKeyDictionary()


def importance_group_rules(coordinator):
    joint = getattr(coordinator, "attention_groups", None)
    return dict(
        scheduler_block_tokens=coordinator.scheduler_block_size,
        joint_query_groups=(None if joint is None else
                            [list(group.group_ids) for group in joint]),
        groups=[
            dict(group_id=m.kv_cache_group_id, pool_scope=m.block_pool._agentmon_research_scope,
                 manager=type(m).__name__, spec=type(m.kv_cache_spec).__name__,
                 block_tokens=m.block_size, alignment_tokens=m.cache_hit_alignment_tokens,
                 window_tokens=getattr(m.kv_cache_spec, "sliding_window", None),
                 prefix_cacheable=getattr(m.kv_cache_spec, "prefix_cacheable", None),
                 fine_grained_prefix_cache=getattr(m, "fine_grained_prefix_cache", None),
                 retains_longer_hit=getattr(m, "retains_longer_hit", None),
                 dcp=m.dcp_world_size, pcp=m.pcp_world_size, use_eagle=m.use_eagle)
            for m in coordinator.single_type_managers
        ],
    )


def importance_demand(scheduler, request, phase):
    """Record Engine-produced hashes even past an early miss or skipped query."""
    try:
        coordinator = scheduler.kv_cache_manager.coordinator
        pool = coordinator.block_pool
        size = pool.hash_block_size
        hashes = request.block_hashes
        if type(size) is not int or size <= 0 or type(hashes) is not list:
            raise ValueError("DEMAND_HASH_VECTOR_FORMAT")
        cursor = _importance_hash_cursors.get(request)
        if phase != "QUERY" or cursor is None:
            cursor = dict(series_id=uuid.uuid4().hex, count=0, chain=bytes(32), last=None)
        elif (len(hashes) < cursor["count"] or
              (cursor["count"] and hashes[cursor["count"] - 1] != cursor["last"])):
            raise ValueError("DEMAND_APPEND_CONTRACT_BROKEN")
        start = cursor["count"]
        if phase == "QUERY" and request in _importance_hash_cursors and start == len(hashes):
            return cursor
        identity = dict(request_id=request.request_id, pool_scope=pool._agentmon_research_scope,
                        series_id=cursor["series_id"], hash_block_tokens=size)
        chain = cursor["chain"]
        # Bound one record and working memory. An absent chunk is detectable
        # using its index and the chain in the corresponding version record.
        for first in range(start, len(hashes), 128):
            part = hashes[first:first + 128]
            before = chain.hex()
            bases = []
            for raw in part:
                if not isinstance(raw, bytes) or not raw:
                    raise ValueError("DEMAND_HASH_ENCODING")
                bases.append(raw.hex())
                chain = hashlib.sha256(chain + len(raw).to_bytes(4, "big") + raw).digest()
            observe("KV_DEMAND_HASH_CHUNK", **identity, index_start=first,
                    base_hashes_hex=bases, previous_chain_sha256=before, chain_sha256=chain.hex())
        hash_utils = sys.modules.get("vllm.v1.core.kv_cache_utils")
        root_hash = getattr(hash_utils, "NONE_HASH", None)
        tokens = request.prompt_token_ids
        observe("KV_DEMAND_VERSION", **identity, phase=phase, parent_count=start,
                hash_count=len(hashes), chain_sha256=chain.hex(),
                hash_chain_schema="SHA256(PREV32 || U32BE(LENGTH) || RAW_HASH); INITIAL=ZERO32",
                key_schema="SHA256(RAW_ENGINE_BASE_HASH || GROUP_U32BE)",
                configured_hash_algorithm=getattr(scheduler.cache_config, "prefix_caching_hash_algo", None),
                engine_hash_root_sha256=(hashlib.sha256(root_hash).hexdigest()
                                         if isinstance(root_hash, bytes) else None),
                cache_salt_sha256=digest(request.cache_salt),
                prompt_token_sha256=digest(tokens) if type(tokens) in (list, tuple) else None,
                prompt_token_count=request.num_prompt_tokens, all_token_count=request.num_tokens,
                computed_tokens=request.num_computed_tokens, hash_covered_tokens=len(hashes)*size,
                hasher_present=request._block_hasher is not None,
                skip_reading_prefix_cache=request.skip_reading_prefix_cache,
                prefix_cache_enabled=scheduler.kv_cache_manager.enable_caching,
                rules=importance_group_rules(coordinator))
        cursor = dict(series_id=cursor["series_id"], count=len(hashes),
                      last=hashes[-1] if hashes else None, chain=chain)
        _importance_hash_cursors[request] = cursor
        return cursor
    except Exception as exc:
        research_gap("IMPORTANCE_DEMAND", exc, request_id=getattr(request, "request_id", None))
        return None


def importance_query_binding(request, pool, query):
    try:
        scheduler = _research_schedule_context.get()
        if scheduler is None or scheduler.requests.get(request.request_id) is not request:
            raise ValueError("DEMAND_QUERY_SCHEDULER_SCOPE")
        if scheduler.kv_cache_manager.coordinator.block_pool is not pool:
            raise ValueError("DEMAND_QUERY_POOL_SCOPE")
        cursor = importance_demand(scheduler, request, "QUERY")
        if cursor is None:
            return
        # Native Writer attaches schedule/decision IDs only from a verified live
        # schedule frame. Do not mint an identity from a stale context here.
        observe("KV_DEMAND_QUERY", request_id=request.request_id, query_id=query["query_id"],
                pool_scope=pool._agentmon_research_scope, series_id=cursor["series_id"],
                hash_count=cursor["count"], chain_sha256=cursor["chain"].hex(),
                group_ids=query["group_ids"], max_length=query["max_length"])
    except Exception as exc:
        research_gap("IMPORTANCE_QUERY_BINDING", exc, query_id=query.get("query_id"))


def importance_group_result(pool, base_hash, groups, hit_blocks, first_missing):
    """Read the actual function's local hit list before its original return."""
    try:
        # Only match the exact caller that owns this query. An unrelated pool
        # call in the same process must never inherit an older query identity.
        caller = sys._getframe(2).f_locals
        query = caller.get("_am_research_query")
        if query is None:
            return
        active = _research_queries.get()
        if (not active or query is not active[-1] or caller.get("block_pool") is not pool
                or list(groups) != query["group_ids"]):
            raise ValueError("GROUP_QUERY_SCOPE_MISMATCH")
        visited = []
        for group, block in zip(groups, hit_blocks):
            key = key_fields(base_hash + group.to_bytes(4, "big"))
            lease = pool._agentmon_research_leases.get(block.block_id)
            visited.append(dict(**key, result="HIT", block_id=block.block_id,
                                allocation_lease=lease, lease_status="OBSERVED" if lease is not None else "UNKNOWN"))
        if first_missing is not None:
            if len(visited) >= len(groups) or groups[len(visited)] != first_missing:
                raise ValueError("GROUP_SHORT_CIRCUIT_POSITION")
            key = key_fields(base_hash + first_missing.to_bytes(4, "big"))
            visited.append(dict(**key, result="MISS", block_id=None, allocation_lease=None,
                                lease_status="NOT_APPLICABLE"))
        elif len(visited) != len(groups):
            raise ValueError("GROUP_HIT_COUNT")
        observe("KV_QUERY_GROUP_RESULTS", query_id=query["query_id"], request_id=query["request_id"],
                ordinal=query["probes"] + 1, pool_scope=pool._agentmon_research_scope,
                visited=visited, first_missing_group=first_missing,
                unvisited_group_ids=list(groups[len(visited):]),
                termination="FIRST_MISS" if first_missing is not None else "ALL_VISITED")
    except Exception as exc:
        research_gap("IMPORTANCE_GROUP_RESULT", exc)
