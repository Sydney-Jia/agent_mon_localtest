"""Exact-source, in-memory call-site instrumentation; never edits vLLM files.

Used where an outer Python API discards the native return codes. Replacing that
outer API with a success assumption would be incorrect. Every original call is
still evaluated once with the same operands, return value and exception.
"""
import ast
import hashlib
import inspect
import sys
import types
import uuid
from pathlib import Path


def checked_methods(module, expected_sha, targets):
    path=Path(module.__file__).resolve()
    source=path.read_bytes()
    if hashlib.sha256(source).hexdigest()!=expected_sha:
        raise ValueError('SOURCE_SHA_MISMATCH')
    compiled=compile(source,str(path),'exec',dont_inherit=True,optimize=sys.flags.optimize)
    originals={}
    for cls_name,names in targets.items():
        cls=module if cls_name is None else getattr(module,cls_name)
        class_code=compiled if cls_name is None else next(c for c in compiled.co_consts if inspect.iscode(c) and c.co_name==cls_name)
        for name in names:
            original=inspect.getattr_static(cls,name)
            expected=next(c for c in class_code.co_consts if inspect.iscode(c) and c.co_name==name)
            if (not inspect.isfunction(original) or hasattr(original,'__wrapped__')
                    or original.__code__!=expected):
                raise ValueError('LOADED_CODE_MISMATCH:'+str(cls_name)+'.'+name)
            if original.__closure__ and not (
                name == '__init__' and expected.co_freevars == ('__class__',)
                and len(original.__closure__) == 1 and original.__closure__[0].cell_contents is cls
            ):
                raise ValueError('LOADED_CODE_MISMATCH:'+str(cls_name)+'.'+name)
            originals[(cls_name,name)]=original
    return source,originals


class Calls(ast.NodeTransformer):
    def __init__(self,helper,selectors):
        self.helper=helper;self.selectors=selectors;self.counts={k:0 for k in selectors}

    def visit_Yield(self,node):
        node=self.generic_visit(node)
        if '$yield' not in self.selectors:return node
        self.counts['$yield']+=1
        node.value=ast.copy_location(ast.Call(func=ast.Name(id=self.helper,ctx=ast.Load()),
            args=[ast.Constant(self.selectors['$yield']),ast.Constant(None),
                  ast.Tuple(elts=[node.value],ctx=ast.Load()),ast.Dict(keys=[],values=[]),
                  ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[])],keywords=[]),node.value)
        return node

    def visit_Call(self,node):
        text=ast.unparse(node.func)
        node=self.generic_visit(node)
        if text not in self.selectors:return node
        if any(k.arg is None for k in node.keywords) or any(isinstance(a,ast.Starred) for a in node.args):
            raise ValueError('UNSUPPORTED_CALL_SHAPE')
        self.counts[text]+=1
        # locals() is used only for scalar descriptors; no Tensor content is read.
        return ast.copy_location(ast.Call(func=ast.Name(id=self.helper,ctx=ast.Load()),
            args=[ast.Constant(self.selectors[text]),node.func,ast.Tuple(elts=node.args,ctx=ast.Load()),
                  ast.Dict(keys=[ast.Constant(k.arg) for k in node.keywords],values=[k.value for k in node.keywords]),
                  ast.Call(func=ast.Name(id='locals',ctx=ast.Load()),args=[],keywords=[])],keywords=[]),node)


def instrument_method(source,module,cls_name,name,original,selectors,callback):
    tree=ast.parse(source)
    cls=tree if cls_name is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls_name)
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    helper='_agentmon_call_'+uuid.uuid4().hex
    transform=Calls(helper,selectors)
    transform.visit(method)
    if any(n!=1 for n in transform.counts.values()):
        raise ValueError('EXPECTED_ONE_EXACT_CALLSITE:'+str(transform.counts))
    ast.fix_missing_locations(tree)
    compiled=compile(tree,module.__file__,'exec',dont_inherit=True,optimize=sys.flags.optimize)
    class_code=compiled if cls_name is None else next(c for c in compiled.co_consts if inspect.iscode(c) and c.co_name==cls_name)
    code=next(c for c in class_code.co_consts if inspect.iscode(c) and c.co_name==name)
    if helper in original.__globals__:raise ValueError('HELPER_COLLISION')
    original.__globals__[helper]=callback
    replacement=types.FunctionType(code,original.__globals__,original.__name__,original.__defaults__)
    replacement.__kwdefaults__=original.__kwdefaults__
    replacement.__annotations__=original.__annotations__
    replacement.__dict__.update(original.__dict__)
    replacement.__wrapped__=original
    return replacement,(original.__globals__,helper)
