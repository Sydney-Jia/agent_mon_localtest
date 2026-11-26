"""Startup-only .030 observation. Default OFF; no original disk source edits."""
import atexit
import functools
import importlib.abc
import importlib.machinery
import inspect
import json
import os
from pathlib import Path
import re
import sys
import uuid
import weakref

from .events import EventSink
from .prefix_observer import PrefixObserver, _REQUEST
from .content_observer import ContentObserver
from .instrument import checked_methods, instrument_method
from .transfer import TransferObserver
from .filesystem import FileTierObserver, METHODS as FILESYSTEM_TARGETS

POOL='vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.'
TARGETS={
    'vllm.v1.core.block_pool':None,
    'vllm.v1.core.kv_cache_manager':{'KVCacheManager':['get_computed_blocks','allocate_slots','free','get_blocks']},
    'vllm.v1.core.sched.scheduler':{'Scheduler':['add_request','_free_request']},
    POOL+'pool_worker':{'KVPoolWorker':['start_load_kv','__init__','register_kv_caches']},
    POOL+'kv_transfer':{'KVCacheStoreSendingThread':['_handle_stored_request'],
                       'KVCacheStoreRecvingThread':['_handle_request'],
                       'KVTransferThread':['set_finished_request']},
    POOL+'backend.mooncake_backend':{'MooncakeBackend':['put','get','__init__']},
    POOL+'metadata':{'ChunkedTokenDatabase':['process_token_key_strings_with_block_ids']},
}
CALLS={
    ('KVPoolWorker','start_load_kv'):{'self.token_database.prepare_value':'prepare','self.m_store.get':'backend_get'},
    ('KVCacheStoreSendingThread','_handle_stored_request'):{'self._prepare_value':'prepare','self.m_store.put':'backend_put'},
    ('KVCacheStoreRecvingThread','_handle_request'):{'self._prepare_value':'prepare','self.m_store.get':'backend_get'},
    ('MooncakeBackend','put'):{'self.store.batch_put_from_multi_buffers':'native_put'},
    ('MooncakeBackend','get'):{'self.store.batch_get_into_multi_buffers':'native_get'},
    ('ChunkedTokenDatabase','process_token_key_strings_with_block_ids'):{'$yield':'key_yield'},
}


class Trace:
    def __init__(self,directory,run_id,capacity=4096,evidence_status='VERIFIED'):
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,96}',run_id):raise ValueError('invalid run ID')
        self.directory=Path(directory);self.run_id=run_id;self.capacity=capacity
        self.evidence_status=evidence_status
        self.pid=None;self.sink=None;self.scopes=weakref.WeakKeyDictionary()

    def _process(self):
        if self.pid!=os.getpid():
            self.pid=os.getpid();self.scopes=weakref.WeakKeyDictionary()
            self.sink=EventSink(self.directory,self.run_id,capacity=self.capacity,evidence_status=self.evidence_status)
        return self.sink

    def scope(self,owner):
        self._process()
        if owner not in self.scopes:self.scopes[owner]=uuid.uuid4().hex
        return self.scopes[owner]

    def emit(self,event,role,source,request_id=None,external_id=None,**fields):
        return self._process().emit(event,process_role=role,source_function=source,request_id=request_id,
                                   external_request_id=external_id,**fields)

    def failure(self):
        try:self._process().observer_failure()
        except Exception:pass

    def close(self):
        if self.pid==os.getpid() and self.sink:self.sink.close()


class Installation:
    def __init__(self,trace,*,filesystem=False,legacy=True):
        self.trace=trace;self.prefix=PrefixObserver(trace);self.content=ContentObserver(trace,self.prefix)
        self.transfer=TransferObserver(trace);self.patches=[];self.helpers=[];self.installed=set()
        self.hashes=json.loads(Path(__file__).with_name('bindings.json').read_text())
        self.targets=dict(TARGETS) if legacy else {}
        if filesystem:
            self.targets.update(FILESYSTEM_TARGETS)
        self.filesystem=FileTierObserver(trace)

    def safe(self,fn,*args):
        try:return fn(*args)
        except Exception:self.trace.failure()

    def holdings(self,manager,request,phase):
        req=request.request_id
        groups=manager.get_blocks(req).blocks
        rows=[]
        for group_id,blocks in enumerate(groups):
            # A per-group row avoids arbitrary request budgets or list truncation.
            # Only existing CPU metadata are read. Placeholder remains explicit.
            for block in blocks:
                pool=getattr(block,'pool',None)
                state=self.content.state(pool) if pool is not None else None
                rows.append(dict(group_id=group_id,block_id=block.block_id,is_null=block.is_null,
                    pool_scope=state['scope'] if state else None,
                    slot_lease=state['leases'].get(block.block_id) if state else None,
                    ref_count=block.ref_cnt))
        self.trace.emit('REQUEST_KV_HOLDINGS','EngineKVManager','KVCacheManager.get_blocks',req,
            getattr(request,'external_req_id',None),phase=phase,holdings=rows,
            logical_pool_scope=self.trace.scope(manager),complete=True)
        if phase=='AFTER_free':
            for group_id,blocks in enumerate(groups):
                self.trace.emit('KV_REQUEST_DETACHED_OR_FREED','EngineKVManager','KVCacheManager.free',req,
                    getattr(request,'external_req_id',None),group_id=group_id,
                    after_block_ids=[b.block_id for b in blocks],after_total_block_count=len(blocks),
                    after_blocks_truncated=False,after_snapshot_error_type=None,
                    logical_pool_scope=self.trace.scope(manager),physical_release='NOT_PROVEN')

    def manager(self,original,name):
        signature=inspect.signature(original)
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            token=None;request=None
            try:
                values=signature.bind(*args,**kwargs).arguments
                request=values['request'];manager=args[0]
                token=_REQUEST.set(dict(request_id=request.request_id,
                    external_id=getattr(request,'external_req_id',None),
                    logical_pool_scope=self.trace.scope(manager),
                    manager_operation={'get_computed_blocks':'lookup','allocate_slots':'allocate','free':'free'}[name]))
                self.safe(self.holdings,manager,request,'BEFORE_'+name)
            except Exception:self.trace.failure()
            try:
                result=original(*args,**kwargs)
                if request is not None:
                    self.safe(self.holdings,args[0],request,'AFTER_'+name)
                    self.safe(self.trace.emit,'MANAGER_CALL_RETURNED','EngineKVManager',name,
                        request.request_id,getattr(request,'external_req_id',None),
                        )
                return result
            finally:
                if token is not None:_REQUEST.reset(token)
        return wrapped

    def scheduler(self,original,name):
        signature=inspect.signature(original)
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            result=original(*args,**kwargs)
            def record():
                req=signature.bind(*args,**kwargs).arguments['request']
                # _free_request can defer block release; record method return,
                # never claim that this event frees every associated buffer.
                self.trace.emit('ENGINE_REQUEST_ADDED' if name=='add_request' else 'REQUEST_FINISHED',
                    'EngineScheduler',name,req.request_id,getattr(req,'external_req_id',None))
            self.safe(record)
            return result
        return wrapped

    def backend_init(self,original):
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            result=original(*args,**kwargs)
            def record():
                b=args[0];c=b.config
                self.trace.emit('POOL_BACKEND_CREATED','AscendStore','MooncakeBackend.__init__',
                    backend_scope=self.trace.scope(b),protocol=c.protocol,
                    enable_ssd_offload=c.enable_ssd_offload,global_segment_size=c.global_segment_size,
                    local_buffer_size=c.local_buffer_size,device_id=b.device_id,
                    store_initialized=b._store_initialized,replica_inventory='NOT_OBSERVED',
                    master_events='NOT_OBSERVED',native_abi='NOT_VERIFIED_BY_CONSTRUCTOR_RETURN')
            self.safe(record)
            return result
        return wrapped

    def worker_config(self,original,name):
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            result=original(*args,**kwargs)
            def record():
                w=args[0]
                layerwise=getattr(w,'use_layerwise',None);mismatch=getattr(w,'tp_mismatch',None)
                self.trace.emit('TRANSFER_PATH_CONFIG','AscendStore','KVPoolWorker.'+name,
                    phase=name,worker_rank=getattr(w,'tp_rank',None),use_layerwise=layerwise,
                    load_async=getattr(w,'load_async',None),tp_mismatch=mismatch,
                    observer_path_supported=layerwise is False and mismatch is False,
                    meaning='OBSERVER_CALLSITE_COVERAGE_NOT_BACKEND_CORRECTNESS')
            self.safe(record)
            return result
        return wrapped

    def patch(self,module):
        if module.__name__ in self.installed:return
        names=self.targets[module.__name__]
        if module.__name__ in FILESYSTEM_TARGETS:
            self.filesystem.patch(self,module)
        elif names is None:
            # Both observers validate their entire target method sets first.
            try:self.prefix.patch_module(module);self.content.patch_module(module)
            except Exception:self.prefix.uninstall();self.content.uninstall();raise
        else:
            source,originals=checked_methods(module,self.hashes[module.__name__],names)
            pending=[];helpers=[]
            try:
                for (cls_name,name),original in originals.items():
                    replacement=original
                    if (cls_name,name) in CALLS:
                        replacement,helper=instrument_method(source,module,cls_name,name,original,
                                                CALLS[(cls_name,name)],self.transfer.call)
                        helpers.append(helper)
                        if cls_name not in ('MooncakeBackend','ChunkedTokenDatabase'):
                            replacement=self.transfer.context_wrapper(replacement)
                    elif cls_name=='KVCacheManager' and name!='get_blocks':replacement=self.manager(original,name)
                    elif cls_name=='Scheduler':replacement=self.scheduler(original,name)
                    elif cls_name=='MooncakeBackend':replacement=self.backend_init(original)
                    elif cls_name=='KVTransferThread':replacement=self.transfer.finish_wrapper(original)
                    elif cls_name=='KVPoolWorker':replacement=self.worker_config(original,name)
                    if replacement is not original:pending.append((getattr(module,cls_name),name,original,replacement))
            except Exception:
                for namespace,key in helpers:namespace.pop(key,None)
                raise
            for cls,name,original,replacement in pending:setattr(cls,name,replacement)
            self.patches.extend(pending);self.helpers.extend(helpers)
        self.installed.add(module.__name__)
        self.trace.emit('V030_TARGET_INSTALLED','bootstrap',module.__name__,
                        file_sha256=self.hashes[module.__name__],loaded_code_verified=True,
                        execution='NOT_OBSERVED_UNTIL_ORIGINAL_CALL')

    def on_import(self,module):
        try:self.patch(module)
        except Exception as exc:
            self.trace.emit('V030_TARGET_REJECTED','bootstrap',module.__name__,reason=str(exc)[:160])

    def uninstall(self):
        self.content.uninstall();self.prefix.uninstall()
        for cls,name,old,new in reversed(self.patches):
            if inspect.getattr_static(cls,name) is new:setattr(cls,name,old)
        for namespace,key in self.helpers:namespace.pop(key,None)
        self.patches.clear();self.helpers.clear();self.installed.clear()


class Loader(importlib.abc.Loader):
    def __init__(self,loader,installation):self.loader=loader;self.installation=installation
    def create_module(self,spec):return self.loader.create_module(spec) if hasattr(self.loader,'create_module') else None
    def exec_module(self,module):
        self.loader.exec_module(module)
        self.installation.on_import(module)


class Finder(importlib.abc.MetaPathFinder):
    def __init__(self,installation):self.installation=installation
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in self.installation.targets:return None
        # Do not iterate sys.meta_path: that produced recursive finder cycles
        # in older deployments. Earlier custom loaders retain their precedence.
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec is not None and spec.loader is not None:spec.loader=Loader(spec.loader,self.installation)
        return spec


_ACTIVE=None
def install():
    global _ACTIVE
    filesystem=os.getenv('AGENTMON_VLLM_030_FS_ENABLE')=='1'
    legacy=os.getenv('AGENTMON_VLLM_030_ENABLE')=='1'
    if not (filesystem or legacy):return None
    if _ACTIVE:return _ACTIVE[0]
    trace=Trace(os.environ['AGENTMON_VLLM_030_EVENTS'],os.environ['AGENTMON_VLLM_030_RUN_ID'])
    installation=Installation(trace,filesystem=filesystem,legacy=legacy);finder=Finder(installation)
    index=next((i for i,f in enumerate(sys.meta_path) if f is importlib.machinery.PathFinder),len(sys.meta_path))
    sys.meta_path.insert(index,finder)
    for name in installation.targets:
        if name in sys.modules:installation.on_import(sys.modules[name])
    _ACTIVE=(installation,finder);atexit.register(trace.close)
    return installation


def uninstall():
    global _ACTIVE
    if _ACTIVE:
        installation,finder=_ACTIVE
        if finder in sys.meta_path:sys.meta_path.remove(finder)
        installation.uninstall();installation.trace.close();_ACTIVE=None
