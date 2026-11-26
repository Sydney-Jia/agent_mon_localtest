"""Compare Worker host-side views at the same sampled pre-forward cut."""
import hashlib
import os
import sys
import uuid


_importance_worker_sources = {}


def importance_worker_source(table_object, ident):
    cls = type(table_object)
    key = (os.getpid(), cls, ident["layout_epoch"])
    if key in _importance_worker_sources:
        return _importance_worker_sources[key]
    binding = uuid.uuid4().hex
    methods = {}
    for method in ("get_numpy_array", "append_row", "add_row"):
        owner = next((base for base in cls.__mro__ if method in vars(base)), None)
        if owner is None:
            continue
        qualified = owner.__qualname__ + "." + method
        try:
            module = sys.modules[owner.__module__]
            fn = getattr(table_object, method)
            if getattr(fn, "__func__", None) is not getattr(owner, method):
                raise ValueError("TABLE_METHOD_OVERRIDDEN")
            proof = module_proof(module, [qualified])
            methods[method] = dict(owner=owner.__module__+"."+qualified, file=proof["path"],
                                   source_sha256=proof["sha256"], code=proof["methods"][qualified])
        except Exception as exc:
            methods[method] = dict(owner=owner.__module__+"."+qualified, proof="UNAVAILABLE",
                                   reason=type(exc).__name__+":"+str(exc)[:160])
    observe("KV_WORKER_TABLE_SOURCE", rank=ident["rank"], layout_epoch=ident["layout_epoch"],
            source_id=binding, table_class=cls.__module__+"."+cls.__qualname__, methods=methods)
    _importance_worker_sources[key] = binding
    return binding


def importance_worker_integer(value, np):
    # In particular, never int() a scalar residing on the NPU.
    return int(value) if type(value) is int or isinstance(value, np.integer) else None


def importance_worker_table(runner, ident, rid, row, gid, table, expected):
    """Describe raw table units and logical Worker state without applying a mapping."""
    try:
        np = sys.modules["numpy"]
        if type(table) is not np.ndarray or table.ndim != 2 or table.dtype.kind not in ("i","u"):
            raise ValueError("WORKER_TABLE_NOT_HOST_INTEGER_NDARRAY")
        obj = runner.input_batch.block_table[gid]
        source = importance_worker_source(obj, ident)
        reported_row = runner.input_batch.req_id_to_index.get(rid)
        if reported_row != row or row < 0 or row >= len(table):
            raise ValueError("WORKER_REQUEST_ROW_IDENTITY")
        values = table[row]
        counts = getattr(obj, "num_blocks_per_row", None)
        host_sequence = (list, tuple, np.ndarray)
        active = (importance_worker_integer(counts[row],np)
                  if type(counts) in host_sequence and 0 <= row < len(counts) else None)
        if active is not None and not 0 <= active <= len(values):
            raise ValueError("WORKER_TABLE_ACTIVE_COUNT")
        req = runner.requests.get(rid)
        groups = getattr(req,"block_ids",None)
        logical = groups[gid] if type(groups) in (tuple,list) and gid < len(groups) else None
        if type(logical) not in host_sequence:
            logical = None
        elif type(logical) is np.ndarray:
            if logical.ndim != 1 or logical.dtype.kind not in ("i","u"):
                logical = None
        elif any(importance_worker_integer(x,np) is None for x in logical):
            logical = None
        kernels = getattr(runner,"kernel_block_sizes",None)
        kernel = kernels[gid] if type(kernels) in (list,tuple) and gid < len(kernels) else None
        if type(kernel) in (list,tuple):
            kernel = [importance_worker_integer(k,np) for k in kernel]
        else:
            kernel = importance_worker_integer(kernel,np)
        # SHA is a local table cross-check, never a checksum of KV tensor data.
        def vector_sha(vector):
            return hashlib.sha256(np.asarray(vector,dtype="<i8").tobytes()).hexdigest()
        stat = {}
        witnesses = {}
        for block in expected:
            pos = block["position"]
            if type(pos) is not int or pos < 0:
                raise ValueError("WORKER_INVALID_SCHEDULER_POSITION")
            sched = block["block_id"]
            logical_id = int(logical[pos]) if logical is not None and pos < len(logical) else None
            raw_id = int(values[pos]) if pos < len(values) else None
            l = ("LOGICAL_UNKNOWN" if logical_id is None else
                 "LOGICAL_SAME" if sched == logical_id else "LOGICAL_DIFF")
            p = ("ACTIVE_COUNT_UNKNOWN" if active is None else "BEYOND_ACTIVE" if pos >= active else
                 "RAW_SAME" if raw_id == sched else "RAW_DIFF")
            category = ("NULL" if block["is_null"] else "VALID") + "/" + l + "/" + p
            stat[category] = stat.get(category,0) + 1
            if category not in witnesses:
                witnesses[category] = dict(position=pos,scheduler_block_id=sched,
                    logical_worker_block_id=logical_id,raw_cpu_table_slot=raw_id,
                    scheduler_block_tokens=block["block_size_tokens"],pool_scope=block["pool_scope"],
                    lease=block.get("allocation_lease"),key_digest=block.get("key_digest"),
                    logical_token_span=block["logical_token_span"])
        sample_size = next((b["block_size_tokens"] for b in expected), None)
        observe("KV_WORKER_TABLE_EVIDENCE", **ident, request_id=rid, group_id=gid,
                table_source_id=source, input_batch_row=row, scheduler_rows=len(expected),
                scheduler_block_tokens=sample_size, configured_kernel_block_tokens=kernel,
                table_block_size_attribute=importance_worker_integer(getattr(obj,"block_size",None),np),
                raw_cpu_row_width=len(values), raw_cpu_active_count=active,
                raw_cpu_count_source="num_blocks_per_row" if active is not None else "NOT_OBSERVED",
                raw_cpu_active_sha256=vector_sha(values[:active]) if active is not None else None,
                worker_logical_count=len(logical) if logical is not None else None,
                worker_logical_sha256=vector_sha(logical) if logical is not None else None,
                vector_digest_encoding="SHA256_LITTLE_ENDIAN_I64_VECTOR", counts=stat, first=witnesses,
                mapping_status="RAW_POSITIONS_COMPARED_NO_REMAP_ASSERTED",
                device_consumption="NOT_OBSERVED")
    except Exception as exc:
        research_gap("IMPORTANCE_WORKER_TABLE",exc,request_id=rid,group_id=gid,**ident)
