"""Extend only the existing full-capture Worker branch."""
TARGETS = ("agentmon_boundary.py",)
BOUNDARY_METHODS = ("importance_worker_source", "importance_worker_table", "importance_worker_integer")


def transform(name, raw):
    if name not in TARGETS:
        return raw
    anchor = b"                expected=[b for b in batch['blocks'] if b['group_id']==gid and b['request_id']==rid]\n"
    call = b"                importance_worker_table(runner, ident, rid, row, gid, table, expected)\n"
    if raw.count(anchor) != 1 or call in raw:
        raise ValueError("IMPORTANCE_WORKER_SOURCE_DRIFT")
    updated = raw.replace(anchor, anchor + call)
    if updated.replace(call, b"") != raw:
        raise ValueError("IMPORTANCE_WORKER_NON_OBSERVATION_DELTA")
    compile(updated, name, "exec", dont_inherit=True)
    return updated
