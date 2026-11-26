"""Observe actual AscendStore/Mooncake operations; submits NO copies itself.

Only the pinned non-layerwise, TP-matched paths are supported. Native outcomes
are API evidence, NOT device completion or proof of the last physical replica.
"""
from contextvars import ContextVar
import functools
import hashlib
import inspect
import time
import uuid

_OP=ContextVar('agentmon030_transfer',default=None)


def opaque(value):
    if type(value) is str:return hashlib.sha256(value.encode()).hexdigest()
    raise ValueError('non-string pool key')


def ints(value):
    if type(value) not in (list,tuple) or any(type(x) is not int or x<0 for x in value):
        raise ValueError('unsupported descriptor')
    return list(value)


class TransferObserver:
    def __init__(self,trace):self.trace=trace

    def emit(self,event,**fields):
        self.trace.emit(event,'AscendStore','v030.transfer',**fields)

    def guard(self,fn,*args):
        try:return fn(*args)
        except Exception:self.trace.failure()

    def context_wrapper(self,original):
        @functools.wraps(original)
        def wrapper(*args,**kwargs):
            token=_OP.set(dict(id=uuid.uuid4().hex,mappings={}))
            try:return original(*args,**kwargs)
            finally:_OP.reset(token)
        return wrapper

    def finish_wrapper(self,original):
        signature=inspect.signature(original)
        @functools.wraps(original)
        def wrapper(*args,**kwargs):
            result=original(*args,**kwargs)
            def record():
                request=signature.bind(*args,**kwargs).arguments['req_id']
                op=_OP.get()
                self.emit('CONNECTOR_REQUEST_FINISH_SIGNAL',request_id=request,
                    operation_id=op['id'] if op else None,
                    native_outcomes=op.get('native_outcomes',[]) if op else None,
                    meaning='ORIGINAL_FINISHED_NOTIFICATION_NOT_PROOF_OF_LOAD_SUCCESS_OR_DEVICE_CONSUMPTION')
            self.guard(record)
            return result
        return wrapper

    def descriptor(self,local,result):
        # This executes AFTER the real prepare_value. Match key and addresses
        # from the actual construction site, not from time or matching block IDs.
        op=_OP.get()
        if op is None:return
        request=local.get('request',local.get('req_meta'))
        req=getattr(request,'req_id',None)
        if type(req) is not str:raise ValueError('request identity missing')
        index=local.get('index')
        key=local.get('key') if 'key' in local and 'key_list' in local else None
        if key is None and type(index) is int:key=local['keys'][index]
        group=local['group_id']
        if type(group) is not int:raise ValueError('group unknown')
        addresses,sizes,block=result
        addresses,sizes=ints(addresses),ints(sizes)
        if len(addresses)!=len(sizes) or type(block) is not int:raise ValueError('bad spans')
        start=local['start']
        end=local.get('end') if 'key_list' in local else local['ends'][index]
        hash_value=local.get('_block_hash')
        # Save path does not preserve each raw hash in a dedicated local unless
        # upstream KV events are enabled. Do NOT guess from textual key suffixes.
        content_key=(hashlib.sha256(hash_value+group.to_bytes(4,'big')).hexdigest()
                     if type(hash_value) is bytes else None)
        observed=op.get('content_keys',{}).get((opaque(key),group,block,start,end),set())
        if content_key is None and len(observed)==1:content_key=next(iter(observed))
        owner=local['self']
        row=dict(request_id=req,group_id=group,block_id=block,token_interval=[start,end],
                 backend_key=opaque(key),content_key=content_key,
                 worker_rank=getattr(owner,'tp_rank',None),
                 slot_lease=None,content_generation='UNRESOLVED',
                 operation_id=op['id'],payload_bytes=sum(sizes),
                 view_descriptor_hash=hashlib.sha256(repr((addresses,sizes)).encode()).hexdigest(),
                 medium='NPU_BUFFER_CANDIDATE_HBM',medium_status='INFERRED_FROM_PROVIDER',
                 evidence_basis='EXACT_PREPARE_VALUE_CALL_AND_RETURN')
        signature=(opaque(key),tuple(addresses),tuple(sizes))
        op['mappings'].setdefault(signature,[]).append(row)
        self.emit('POOL_BLOCK_DESCRIPTOR',**row)

    def key_yield(self,value,local):
        op=_OP.get()
        if op is None:return
        start,end,key,raw_hash,block=value
        group=local['kv_cache_group_id']
        if type(raw_hash) is not bytes or type(group) is not int:return
        content=hashlib.sha256(raw_hash+group.to_bytes(4,'big')).hexdigest()
        op.setdefault('content_keys',{}).setdefault((opaque(key),group,block,start,end),set()).add(content)

    def backend_scope(self,local,args):
        op=_OP.get()
        keys,addresses,sizes=args[:3]
        if type(keys) not in (list,tuple) or len(keys)!=len(addresses) or len(keys)!=len(sizes):
            raise ValueError('batch shape')
        rows=[]
        for key,addr,size in zip(keys,addresses,sizes):
            a,s=ints(addr),ints(size)
            signature=(opaque(key),tuple(a),tuple(s))
            matches=op['mappings'].get(signature,[]) if op else []
            # Deduplicate repeated construction of an identical descriptor only.
            unique={repr(sorted(m.items())):m for m in matches}
            rows.append(dict(backend_key=opaque(key),payload_bytes=sum(s),
                       binding=next(iter(unique.values())) if len(unique)==1 else None,
                       binding_status='EXACT_DESCRIPTOR' if len(unique)==1 else 'NOT_OBSERVED_OR_AMBIGUOUS'))
        return dict(id=uuid.uuid4().hex,parent_operation_id=op['id'] if op else None,rows=rows,
                    backend_owner=local.get('self'))

    def native_result(self,scope,label,result,started,ended,error,backend):
        if not scope:
            self.emit('TRANSFER_COVERAGE_GAP',reason='NATIVE_CALL_WITHOUT_EXACT_BATCH_SCOPE');return
        rows=scope['rows']
        # Never consume an unknown iterator returned by a C extension.
        codes=list(result) if type(result) in (list,tuple) else None
        valid=codes is not None and len(codes)==len(rows) and all(type(x) is int for x in codes)
        config=getattr(backend,'config',None)
        namespace=(getattr(config,'master_server_address',None),getattr(config,'tenant_id',None))
        namespace_id=(hashlib.sha256(repr(namespace).encode()).hexdigest()
                      if all(type(x) is str and x for x in namespace) else None)
        for i,row in enumerate(rows):
            code=codes[i] if valid else None
            # put follows the pinned backend's negative-code failure contract;
            # get also treats zero as non-error. The byte count is a requested
            # descriptor, not measured wire traffic or physical DDR allocation.
            status='EXCEPTION' if error else 'NOT_OBSERVED' if code is None else 'BACKEND_ERROR' if code<0 else 'BACKEND_API_SUCCESS'
            op=_OP.get()
            if op is not None:op.setdefault('native_outcomes',[]).append(dict(operation_id=scope['id'],key_index=i,status=status))
            binding=row['binding'] or {}
            self.emit('TRANSFER_KEY_RESULT',operation_id=scope['id'],parent_operation_id=scope['parent_operation_id'],
                direction='SAVE' if label=='native_put' else 'LOAD',backend_key=row['backend_key'],
                request_id=binding.get('request_id'),group_id=binding.get('group_id'),block_id=binding.get('block_id'),
                content_key=binding.get('content_key'),token_interval=binding.get('token_interval'),
                view_descriptor_hash=binding.get('view_descriptor_hash'),
                worker_rank=binding.get('worker_rank'),descriptor_binding=row['binding_status'],
                payload_bytes=row['payload_bytes'],successful_descriptor_bytes=row['payload_bytes'] if status=='BACKEND_API_SUCCESS' else None,
                return_code=code,status=status,exception_type=type(error).__name__ if error else None,
                started_monotonic_ns=started,ended_monotonic_ns=ended,
                call_duration_ns=ended-started,timing_scope='NATIVE_API_HOST_CALL_NOT_DEVICE_EVENT',
                timing_shared_by_batch=True,
                pool_scope=namespace_id,pool_scope_semantics='CONFIGURED_ENDPOINT_TENANT_NOT_MASTER_EPOCH',
                backend_scope=self.trace.scope(backend) if backend is not None else None,
                replica_id=None,content_generation='UNRESOLVED',
                configured_ssd_offload=getattr(config,'enable_ssd_offload',None),
                target_or_source_medium='EXTERNAL_POOL_RESIDENCY_NOT_CONFIRMED',
                actual_wire_bytes=None,physical_completion='NOT_OBSERVED',key_index=i)

    def call(self,label,original,args,kwargs,local):
        if label=='key_yield':
            self.guard(self.key_yield,args[0],local)
            return args[0]
        if label=='prepare':
            result=original(*args,**kwargs)
            self.guard(self.descriptor,local,result)
            return result
        if label in ('backend_put','backend_get'):
            scope=self.guard(self.backend_scope,local,args)
            token=_BATCH.set(scope)
            try:return original(*args,**kwargs)
            finally:_BATCH.reset(token)
        if label in ('native_put','native_get'):
            scope=_BATCH.get();started=time.monotonic_ns()
            try:result=original(*args,**kwargs)
            except BaseException as error:
                self.guard(self.native_result,scope,label,None,started,time.monotonic_ns(),error,local.get('self'))
                raise
            self.guard(self.native_result,scope,label,result,started,time.monotonic_ns(),None,local.get('self'))
            return result
        self.trace.failure()
        return original(*args,**kwargs)


_BATCH=ContextVar('agentmon030_native_batch',default=None)
