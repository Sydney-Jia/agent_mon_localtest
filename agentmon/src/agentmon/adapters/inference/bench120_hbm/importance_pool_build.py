"""Checked observation insertions in the reviewed BlockPool implementation."""
import ast
from collections import Counter


TARGETS = ('block_pool.py',)
HOOKS = ('importance_queue_checkpoint', 'importance_queue_edges',
         'importance_queue_request', 'importance_queue_edit')
POOL_METHODS = ('BlockPool.__init__', 'BlockPool.get_new_blocks',
                'BlockPool.touch', 'BlockPool.unpin_blocks',
                'BlockPool.free_blocks', 'BlockPool.reset_prefix_cache')


def _call(name, args):
    return ast.parse('_am_research.' + name + '(' + args + ')').body[0]


def _edit(action, blocks, decision=False):
    return _call('importance_queue_edit', 'self, ' + repr(action) + ', ' + blocks
                 + (', _am_queue_request_seq' if decision else ''))


class _Add(ast.NodeTransformer):
    def __init__(self):
        self.fn = None
        self.hits = Counter()
        self.queue_calls = Counter()

    def visit_ClassDef(self, node):
        return self.generic_visit(node) if node.name == 'BlockPool' else node

    def visit_FunctionDef(self, node):
        old = self.fn
        self.fn = node.name
        self.generic_visit(node)
        self.fn = old
        return node

    def generic_visit(self, node):
        super().generic_visit(node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if ast.unparse(node.func.value) == 'self.free_block_queue':
                self.queue_calls[self.fn, node.func.attr] += 1
        if not isinstance(node, ast.stmt):
            return node
        if self.fn == 'get_new_blocks' and isinstance(node, ast.If):
            if ast.unparse(node.test) == 'num_blocks > self.get_num_free_blocks()':
                if len(node.body) != 1 or not isinstance(node.body[0], ast.Raise):
                    raise ValueError('IMPORTANCE_POOL_FAILURE_BRANCH_CHANGED')
                node.body.insert(0, _edit('INSUFFICIENT_FREE', '()', True))
                self.hits['decision'] += 1
                return [ast.parse('_am_queue_request_seq = '
                                  '_am_research.importance_queue_request(self, num_blocks)').body[0],
                        node]
        if not isinstance(node, (ast.Expr, ast.AnnAssign, ast.Return)):
            return node
        anchors = {
            ('__init__', '_am_research.research_pool(self)'):
                ('init', _call('importance_queue_checkpoint', 'self')),
            ('get_new_blocks', 'ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)'):
                ('alloc', _edit('ALLOCATE_POP', 'ret', True)),
            ('touch', 'self.free_block_queue.remove(block)'):
                ('touch', _edit('TOUCH_REMOVE', '(block,)')),
            ('unpin_blocks', 'self.free_block_queue.append_n(released)'):
                ('unpin', _edit('UNPIN_APPEND', 'released')),
            ('free_blocks', 'self.free_block_queue.prepend_n(blocks_to_evict_first)'):
                ('front', _edit('FREE_PREPEND', 'blocks_to_evict_first')),
            ('free_blocks', 'self.free_block_queue.append_n(blocks_to_evict_last)'):
                ('back', _edit('FREE_APPEND', 'blocks_to_evict_last')),
            ('reset_prefix_cache', 'return True'):
                ('reset', _edit('RESET_INDEX', '()')),
        }
        hit = anchors.get((self.fn, ast.unparse(node)))
        if hit:
            name, new = hit
            self.hits[name] += 1
            return [new, node] if name == 'reset' else [node, new]
        return node


class _Strip(ast.NodeTransformer):
    def visit_Expr(self, node):
        call = node.value
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and ast.unparse(call.func.value) == '_am_research' and call.func.attr in HOOKS):
            return None
        return node

    def visit_Assign(self, node):
        if len(node.targets) == 1 and ast.unparse(node.targets[0]) == '_am_queue_request_seq':
            return None
        return node


def transform(name, raw):
    if name not in TARGETS:
        return raw
    tree = ast.parse(raw)
    before = ast.dump(tree)
    if '_am_queue_request_seq' in before or 'importance_queue_' in before:
        raise ValueError('IMPORTANCE_POOL_ALREADY_INSTRUMENTED')
    add = _Add()
    text = ast.unparse(ast.fix_missing_locations(add.visit(tree))) + '\n'
    expected = Counter(dict.fromkeys(('init', 'alloc', 'decision', 'touch', 'unpin',
                                      'front', 'back', 'reset'), 1))
    queues = Counter(dict.fromkeys((
        ('__init__', 'popleft'), ('get_new_blocks', 'popleft_n'),
        ('touch', 'remove'), ('unpin_blocks', 'append_n'), ('free_blocks', 'prepend_n'),
        ('free_blocks', 'append_n')), 1))
    if add.hits != expected or add.queue_calls != queues:
        raise ValueError('IMPORTANCE_POOL_QUEUE_ANCHORS_CHANGED')
    if ast.dump(_Strip().visit(ast.parse(text))) != before:
        raise ValueError('IMPORTANCE_POOL_NON_OBSERVATION_DELTA')
    return text.encode()
