"""Build AgentMon observations on top of the audited Bench120 HBM-only output."""
import ast
import hashlib
import re

if __package__:
    from . import importance_query_build as query, importance_pool_build as pool
    from . import importance_compute_build as compute, importance_worker_build as worker
else:
    import importance_query_build as query
    import importance_pool_build as pool
    import importance_compute_build as compute
    import importance_worker_build as worker

CONTRACT = "AGENTMON_KV_IMPORTANCE_HBM_V1"
FRAGMENTS = (("query", query), ("pool", pool), ("compute", compute), ("worker", worker))
INHERITED = ("add_request", "_update_request_as_session", "_free_blocks")
BOUNDARY = 'vllm.v1.kv_offload.agentmon_boundary'
CORE = 'vllm.v1.core.sched.scheduler'
ASCEND = 'vllm_ascend.patch.platform.patch_balance_schedule'
WORKER = 'vllm_ascend.worker.model_runner_v1'


def admission_methods(root):
    """Names from reviewed build inputs, independently of an Engine's receipt."""
    base = ast.parse((root / 'research_boundary' / 'agentmon_boundary.py').read_bytes())
    fn = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == 'modules')
    assign = next(n for n in fn.body if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'targets')
    targets = ast.literal_eval(assign.value)
    del targets['vllm.v1.kv_offload.cpu.manager']
    del targets['vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler']
    helpers = [n.name for label, _ in FRAGMENTS for n in
               ast.parse((root / ('importance_' + label + '.py')).read_bytes()).body
               if isinstance(n, ast.FunctionDef)]
    targets[BOUNDARY] += ['hbm_startup', 'hbm_token_identity', 'hbm_dispatched',
                          'hbm_accepted_output', 'hbm_worker_startup'] + helpers
    targets[CORE] += ['Scheduler.' + name for name in ('_preempt_request', 'update_from_output',
                      '_free_request_blocks', '_free_cow_retained_blocks', '_drain_deferred_frees', *INHERITED)]
    targets['vllm.v1.core.block_pool'] += ['BlockPool.get_cached_block']
    targets[ASCEND] = ['BalanceScheduler.__init__', 'BalanceScheduler.schedule',
                        'BalanceScheduler.balance_gather', 'BalanceDPEngineCoreProc._has_global_unfinished_reqs',
                        '_balance_scheduling_enabled']
    worker_proofs = {BOUNDARY: ['hbm_worker_startup', 'research_tensor', 'research_initialize_layout',
                                'research_forward_begin', *helpers],
                     WORKER: ['NPUModelRunner.initialize_kv_cache_tensors', 'NPUModelRunner.execute_model']}
    return targets, worker_proofs


def replace(text, anchor, edited, count=1):
    if text.count(anchor) != count:
        raise ValueError("IMPORTANCE_PROOF_ANCHOR_CHANGED:" + anchor)
    return text.replace(anchor, edited)


def overlay(root, baseline):
    outputs = dict(baseline)
    helpers = []
    boundary_methods = []
    for label, adapter in FRAGMENTS:
        for name in adapter.TARGETS:
            if name not in outputs:
                outputs[name] = (root / "research_candidate" / name).read_bytes()
            outputs[name] = adapter.transform(name, outputs[name])
        source = (root / ("importance_" + label + ".py")).read_text(encoding="utf-8")
        functions = [n.name for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
        if not functions or set(boundary_methods).intersection(functions):
            raise ValueError("IMPORTANCE_HELPER_NAMES:" + label)
        boundary_methods.extend(functions)
        helpers.append(source)

    text = outputs["agentmon_boundary.py"].decode()
    anchor = "mode='HBM_ONLY_C50_V1', native_connector_present=False, pools=rows,"
    text = replace(text, anchor, anchor + "\n                      kv_importance_contract=" + repr(CONTRACT)
                   + ", scheduler_class_module=type(scheduler).__module__,")
    anchor = "observe('HBM_WORKER_READY', rank=int(runner.parallel_config.rank),"
    text = replace(text, anchor, anchor + "\n            kv_importance_contract=" + repr(CONTRACT) + ",")
    # Inputs are read-only reviewed sources. Pin the real resulting files and
    # include the newly executing call sites in the existing in-process proofs.
    for name, constant in (("core_scheduler.py", "CORE_SHA"), ("ascend_scheduler.py", "ASCEND_SHA"),
                           ("model_runner.py", "RESEARCH_WORKER_SHA")):
        text, count = re.subn("^" + constant + " = '[a-f0-9]{64}'",
                              constant + " = '" + hashlib.sha256(outputs[name]).hexdigest() + "'",
                              text, flags=re.M)
        if count != 1:
            raise ValueError("IMPORTANCE_MODULE_PIN:" + name)
    anchor = "'Scheduler._drain_deferred_frees']"
    text = replace(text, anchor, anchor[:-1] + ", " + ", ".join(repr("Scheduler." + x) for x in INHERITED) + "]", 2)
    anchor = "    if cls.reset_prefix_cache is not base.reset_prefix_cache:"
    text = replace(text, anchor, "    if any(getattr(cls, n) is not getattr(base, n) for n in "
                   + repr(INHERITED) + "):\n        raise AttestationError('IMPORTANCE_UNREVIEWED_SCHEDULER_OVERRIDE')\n" + anchor)
    anchor = "    for name, methods in targets.items():"
    text = replace(text, anchor, "    if hbm_only:\n"
                   "        targets['vllm.v1.core.block_pool'].append('BlockPool.get_cached_block')\n"
                   "        targets['vllm.v1.kv_offload.agentmon_boundary'].extend("
                   + repr(boundary_methods) + ")\n" + anchor)
    anchor = "'research_initialize_layout', 'research_forward_begin'])"
    # Worker startup proves its execution helpers in that Worker interpreter.
    text = replace(text, anchor, "'research_initialize_layout', 'research_forward_begin'] + "
                   + repr(boundary_methods) + ")")
    outputs["agentmon_boundary.py"] = ("\n".join([text, *helpers]) + "\n").encode()
    return outputs
