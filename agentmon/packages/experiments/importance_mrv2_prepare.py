"""Prepare reviewed MRV2 observability sources in a fresh directory (no deployment)."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile


EXPECTED = {
    'engram/model_state.py': 'c93e5adbffdc780eac95be35d6a065843278b46a206abc3b22d1ec4c57dddd22',
    'decoder_replay_state.py': '7cbb6a05ae320855dde076bfb439d13864afb563406830bd05b69fbfc3c5401c',
    'decoder_replay_layers.py': '283b9c380d48d0a357c8a2a2859c5d2f8ebc62c084d54560b2fcdb2c50e66333',
}


def transform_mrv2(name, raw):
    if hashlib.sha256(raw).hexdigest() != EXPECTED[name]:
        raise ValueError('MRV2_SOURCE_SHA_MISMATCH:' + name)
    lines = raw.decode('utf-8').splitlines(keepends=True)
    tree = ast.parse(raw)
    inserts = []

    def insert_before(line, statement, after=None):
        pad = lines[line-1][:len(lines[line-1]) - len(lines[line-1].lstrip())]
        inserts.append((after if after is not None else line-1, pad + statement + '\n'))

    def at(anchor, statement, after=False):
        [idx] = [i+1 for i,s in enumerate(lines) if s.strip() == anchor]
        insert_before(idx,statement,idx if after else None)

    import_stmt = 'from vllm_ascend.models.deepseek_v41 import _agentmon_replay_observer as _am_mrv2'
    insert_before(tree.body[0].lineno,import_stmt)
    if name == 'engram/model_state.py':
        at('super().add_request(req_index, new_req_data)', '_am_mrv2.bind_request(self, req_index, new_req_data)', True)
        at('self._cg_mode = cudagraph_mode',
           '_am_mrv2.begin_batch(self, input_batch, for_capture, ubatch_idx, cudagraph_mode)', True)
        launches = [n for n in ast.walk(tree) if isinstance(n,ast.Expr)
                    and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Subscript)
                    and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id == '_pad_v2_replayed_slots_kernel']
        if len(launches) != 2:
            raise ValueError('MRV2_KERNEL_ANCHOR_COUNT')
        for n in launches:
            global_batch = 'global_reqs' in ast.unparse(n.value.func.slice)
            args = 'global_batch, global_replay_np' if global_batch else 'input_batch, replay_np'
            insert_before(n.lineno,'_am_mrv2.encoder_enter(self)')
            insert_before(n.lineno, f'_am_mrv2.encoder_return(self, {args}, replay_window, cacheable_groups, {global_batch})', n.end_lineno)
    elif name == 'decoder_replay_state.py':
        at('self.layers.context_factory = self.forward_context',
           '_am_mrv2.decoder_prepared(self, batch, kept_lens, inputs.pcp_context is not None)',True)
    else:
        anchor = 'row_outputs = self.run_layers(*replay_states)'
        at(anchor,'_am_mrv2.decoder_enter(self)')
        at(anchor,'_am_mrv2.decoder_return(self)',True)
    for idx, statement in sorted(inserts,reverse=True):
        lines.insert(idx,statement)
    observed = ''.join(lines)

    class Strip(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            return None if any(n.asname == '_am_mrv2' for n in node.names) else node

        def visit_Expr(self,node):
            f = getattr(node.value,'func',None)
            return None if isinstance(f,ast.Attribute) and isinstance(f.value,ast.Name) and f.value.id == '_am_mrv2' else node

    if ast.dump(Strip().visit(ast.parse(observed))) != ast.dump(tree):
        raise ValueError('MRV2_OBSERVATION_DELTA:' + name)
    compile(observed,name,'exec')
    return observed.encode('utf-8')


def prepare(model_sources, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    files = {name:transform_mrv2(name,(Path(model_sources)/name).read_bytes()) for name in EXPECTED}
    files['_agentmon_replay_observer.py'] = Path(__file__).with_name('importance_mrv2_observer.py').read_bytes()
    sha = lambda b: hashlib.sha256(b).hexdigest()
    receipt = dict(status='OPTIONAL_MRV2_SOURCE_CANDIDATE', inputs=EXPECTED,
                   outputs={name:sha(b) for name,b in files.items()}, observation_only_ast_checked=True,
                   runtime_installed=False, runtime_loaded=False, external_dependency=
                   'reviewed vllm.v1.kv_offload.agentmon_boundary observe/research_gap + active Writer',
                   acceptance='proc/rank/epoch/source and loaded-method proof plus real branch and request joins required')
    # Publish only a fully compiled set. mkdir(exist_ok=False) reserves the
    # target against a second preparer; never replace another user's output.
    with tempfile.TemporaryDirectory(dir=output.parent) as td:
        temp = Path(td)
        for name,b in files.items():
            f=temp/name
            f.parent.mkdir(parents=True,exist_ok=True)
            f.write_bytes(b)
        (temp/'MRV2_PROOF.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        output.mkdir()
        for item in sorted(temp.iterdir(), key=lambda x: x.name == 'MRV2_PROOF.json'):
            os.replace(item,output/item.name)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-sources', required=True, help='Reviewed vllm_ascend/models/deepseek_v41 directory')
    parser.add_argument('--output', required=True, help='New offline source candidate directory')
    args = parser.parse_args()
    print(json.dumps(prepare(args.model_sources,args.output)))
