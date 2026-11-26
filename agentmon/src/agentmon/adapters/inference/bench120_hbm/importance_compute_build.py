"""Insert side-effect-only calls at the reviewed R2.2 compute boundaries."""
import ast


TARGETS = ('core_scheduler.py', 'ascend_scheduler.py', 'model_runner.py', 'agentmon_boundary.py')
HOOKS = ('importance_request_begin', 'importance_request_free', 'importance_compute_query',
         'importance_prefix_selected', 'importance_compute_schedule', 'importance_first_output',
         'importance_forward_stamp', 'importance_forward_result')


def transform(name, raw):
    if name not in TARGETS:
        return raw
    text = raw.decode('utf-8')
    original = ast.dump(ast.parse(text))

    def insert(anchor, before=(), after=(), unique_prefix=False):
        nonlocal text
        lines = text.splitlines(keepends=True)
        indexes = [i for i, line in enumerate(lines)
                   if (line.strip().startswith(anchor) if unique_prefix else line.strip() == anchor)]
        if len(indexes) != 1:
            raise ValueError('IMPORTANCE_COMPUTE_ANCHOR:' + name + ':' + anchor)
        i = indexes[0]
        pad = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
        lines[i:i+1] = [pad + v + '\n' for v in before] + [lines[i]] + [pad + v + '\n' for v in after]
        text = ''.join(lines)

    if name == 'agentmon_boundary.py':
        insert('_research_queries.set(stack[:-1])', after=(
            'importance_compute_query(_research_schedule_context.get(), q, hit_length, window_need)',))
    elif name in ('core_scheduler.py', 'ascend_scheduler.py'):
        insert("_am_boundary.decision(self, request, 'FINAL_SELECTED',", unique_prefix=True, after=(
            '_am_research.importance_prefix_selected(self, request, num_new_local_computed_tokens, num_external_computed_tokens, load_kv_async)',))
        insert('_am_research.hbm_dispatched(self, scheduler_output)', before=(
            '_am_research.importance_compute_schedule(self, scheduler_output)',))
        if name == 'core_scheduler.py':
            insert('self.requests[request.request_id] = request', after=(
                "_am_research.importance_request_begin(self, request, 'NEW_REQUEST')",))
            insert('session.status = RequestStatus.WAITING', after=(
                "_am_research.importance_request_begin(self, session, 'STREAM_UPDATE_APPLIED')",))
            insert('del self.requests[request.request_id]', after=(
                '_am_research.importance_request_free(self, request)',))
            insert('_am_research.hbm_accepted_output(self, request, locals())', before=(
                '_am_research.importance_first_output(self, request, scheduler_output, num_output_tokens_before, output_is_stale)',))
    else:
        insert('hidden_states = self._model_forward(num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs)',
               before=('_am_research.importance_forward_stamp(self, _am_research_ident, True)',),
               after=('_am_research.importance_forward_stamp(self, _am_research_ident, False)',))
        insert('_am_research.research_device_end(self, _am_research_timer)', after=(
            '_am_research.importance_forward_result(self, _am_research_ident, num_tokens_unpadded, num_tokens_padded, num_reqs, cudagraph_mode)',))

    class Strip(ast.NodeTransformer):
        def visit_Expr(self, node):
            f = getattr(node.value, 'func', None)
            if (isinstance(f, ast.Name) and f.id in HOOKS or isinstance(f, ast.Attribute)
                    and isinstance(f.value, ast.Name) and f.value.id == '_am_research' and f.attr in HOOKS):
                return None
            return self.generic_visit(node)

    if ast.dump(Strip().visit(ast.parse(text))) != original:
        raise ValueError('IMPORTANCE_COMPUTE_BASELINE_AST_CHANGED:' + name)
    return text.encode('utf-8')
