"""Add observations at audited branches; the original source remains reversible."""
import ast


TARGETS = ("core_scheduler.py", "agentmon_boundary.py", "block_pool.py")
BOUNDARY_METHODS = (
    "importance_group_rules", "importance_demand", "importance_query_binding",
    "importance_group_result",
)


def transform(name, raw):
    if name not in TARGETS:
        return raw
    if name == "core_scheduler.py":
        edits = [
            ("            self.requests[request.request_id] = request\n",
             "            _am_research.importance_demand(self, request, 'ADMISSION')\n"),
            ("        session.status = RequestStatus.WAITING\n",
             "        _am_research.importance_demand(self, session, 'STREAMING_UPDATE')\n"),
        ]
    elif name == "agentmon_boundary.py":
        edits = [
            ("        observe('HBM_ELIGIBILITY_BEGIN', **q)\n",
             "        importance_query_binding(request, values['block_pool'], q)\n"),
        ]
    else:
        edits = [
            ("            if not block:\n                return None\n            cached_blocks.append(block)\n        return cached_blocks\n",
             None),
        ]
    source = raw.decode("utf-8")
    updated = source
    reverse = []
    for anchor, insertion in edits:
        if updated.count(anchor) != 1:
            raise ValueError("IMPORTANCE_QUERY_SOURCE_DRIFT:" + name)
        if insertion is not None:
            replacement = anchor + insertion
        else:
            replacement = (
                "            if not block:\n"
                "                _am_research.importance_group_result(self, block_hash, kv_cache_group_ids, cached_blocks, group_id)\n"
                "                return None\n"
                "            cached_blocks.append(block)\n"
                "        _am_research.importance_group_result(self, block_hash, kv_cache_group_ids, cached_blocks, None)\n"
                "        return cached_blocks\n"
            )
        updated = updated.replace(anchor, replacement)
        reverse.append((replacement, anchor))
    restored = updated
    for replacement, anchor in reversed(reverse):
        if restored.count(replacement) != 1:
            raise ValueError("IMPORTANCE_QUERY_INSERTION_AMBIGUOUS:" + name)
        restored = restored.replace(replacement, anchor)
    if restored != source or ast.dump(ast.parse(restored)) != ast.dump(ast.parse(source)):
        raise ValueError("IMPORTANCE_QUERY_NON_OBSERVATION_DELTA:" + name)
    compile(updated, name, "exec", dont_inherit=True)
    return updated.encode("utf-8")
