"""Opt-in runtime Tensor metadata inventory. Never reads Tensor values or exports memory addresses.

Targets are supplied in a reviewed server-source profile, not guessed from a
version number. Inventory files are separate from logical KV event archives.
"""
from __future__ import annotations
import atexit
import functools
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import os
from pathlib import Path
import sys
import time
import uuid
from .events import EventSink
from .kv_spec_metadata import describe_group_spec
from .storage_identity import StorageIdentityRegistry

FLAG='AGENTMON_STATE_PLACEMENT_ENABLE'
BUILD='agentmon-placement-v0.4-20261026-storage-identity'
_ACTIVE=None


def observed(value, status='VERIFIED', reason=None):
    return {'value':value,'status':status,'reason':reason}


def medium_for_device(device):
    # 用户当前需要介质层归属，不需要真实地址。device 是直接元数据，HBM/DDR 是
    # 基于后端的候选分类；CPU mmap 页是否真正驻留、SSD backing 仍不能靠它证明。
    kind=str(device).split(':',1)[0].lower()
    if kind in {'npu','cuda'}:
        return observed('HBM' if kind=='npu' else 'DEVICE_RAM','INFERRED',
            'Tensor device address space observed; physical residency/backend must be verified separately')
    if kind=='cpu':
        return observed('DDR','INFERRED','CPU Tensor address space; mmap backing and resident pages not established')
    return observed(None,'NOT_OBSERVED','No established medium for device '+str(device))


def resolve(owner, path):
    value=owner
    for part in path.split('.'):
        if not part or part.startswith('__'):raise ValueError('invalid attribute path')
        value=getattr(value,part)
    return value


def scalar_metadata_sequence(value,depth=0):
    if depth>3:raise ValueError('metadata sequence too deep')
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,(list,tuple)) and len(value)<=128:return [scalar_metadata_sequence(v,depth+1) for v in value]
    raise TypeError('not a bounded scalar metadata sequence')


def is_tensor(value):
    return all(hasattr(value,key) for key in ('shape','dtype','device','numel','element_size'))


def tensor_metadata(tensor):
    """No data_ptr, cpu, to, numpy, item, clone, synchronize or content hash."""
    # 只读形状/类型/storage 元数据，不搬运 Tensor 内容，不新增设备同步。
    # logical view、storage extent、allocator reserved 分账，不能重复相加估算 HBM。
    device=str(tensor.device);shape=[int(n) for n in tensor.shape]
    row={'device':observed(device),'medium':medium_for_device(device),
        'shape':shape,'dtype':str(tensor.dtype),
        'logical_view_bytes':observed(int(tensor.numel())*int(tensor.element_size())),
        'storage_extent_bytes':observed(None,'NOT_OBSERVED'),
        'allocator_reserved_bytes':observed(None,'NOT_OBSERVED'),
        'physical_address':observed(None,'NOT_OBSERVED','Addresses are deliberately not collected'),
        'content_generation':observed(None,'UNRESOLVED'),
        'resident_page_bytes':observed(None,'NOT_OBSERVED')}
    storage=None
    try:
        storage=tensor.untyped_storage()
        row['storage_extent_bytes']=observed(int(storage.nbytes()),reason='Storage extent, not allocator/HBM reservation')
    except Exception as exc:
        row['storage_extent_bytes']['reason']=type(exc).__name__
    return row,storage


def tensors(value, path, state_kind, depth=0):
    if depth>8:raise ValueError('inventory tree depth exceeded')
    if is_tensor(value):
        yield path,value,state_kind,'EXPLICIT_TENSOR_ROOT';return
    if hasattr(value,'named_parameters') and hasattr(value,'named_buffers'):
        for name,tensor in value.named_parameters(recurse=True):
            yield path+'.'+name,tensor,state_kind,'REGISTERED_PARAMETER'
        for name,tensor in value.named_buffers(recurse=True):
            yield path+'.'+name,tensor,'MODEL_BUFFER' if state_kind=='WEIGHT' else state_kind,'REGISTERED_BUFFER'
    elif isinstance(value,dict):
        for name,item in value.items():
            if isinstance(name,(str,int)):
                yield from tensors(item,path+'['+str(name)[:200]+']',state_kind,depth+1)
    elif isinstance(value,(list,tuple)):
        for i,item in enumerate(value):yield from tensors(item,path+'['+str(i)+']',state_kind,depth+1)
    else:raise TypeError('unsupported inventory root; provide reviewed explicit Tensor/model/list/dict root')


class InventoryObserver:
    def __init__(self,directory,service_run_id,*,provenance='VERIFIED',max_objects=4096):
        if provenance not in {'VERIFIED','SYNTHETIC_ISOLATED'}:raise ValueError('invalid provenance')
        self.directory=Path(directory);self.service_run_id=service_run_id
        self.provenance=provenance;self.max_objects=max_objects;self.pid=None;self.sink=None
        self.storage_registry=StorageIdentityRegistry()

    def _sink(self):
        if self.pid!=os.getpid():
            self.pid=os.getpid()
            self.sink=EventSink(self.directory,self.service_run_id,evidence_status=self.provenance,capacity=4096)
        return self.sink

    def emit(self,kind,source,**fields):
        try:
            active=getattr(sys.modules.get(__package__+'.hook'),'_ACTIVE',None)
            trace=getattr(active,'trace',None);logical_sink=getattr(trace,'sink',None)
            link=None
            if logical_sink is not None and getattr(logical_sink,'pid',None)==os.getpid() and getattr(getattr(trace,'config',None),'run_id',None)==self.service_run_id:
                link={'run_id':self.service_run_id,'pid':os.getpid(),'engine_epoch':logical_sink.epoch}
            return self._sink().emit(kind,process_role='STATE_METADATA_OBSERVER',source_function=source,
                inventory_build_id=BUILD,logical_trace_identity=observed(link,'VERIFIED' if link else 'NOT_OBSERVED'),**fields)
        except Exception:
            try:
                if self.sink is not None:self.sink.observer_failure()
            except Exception:pass
            return False

    def capture(self,owner,profile,source,configuration=None,returned=None,arguments=None):
        snapshot=uuid.uuid4().hex;count=errors=0;truncated=False
        stores={} # Keep storage wrappers only during this metadata snapshot, not after it.
        try:rank=resolve(owner,profile.get('rank_path','rank'))
        except Exception:rank=None
        rank=rank if type(rank)is int else None
        self.emit('INVENTORY_BEGIN',source,snapshot_id=snapshot,stage=profile.get('stage'),
            owner_class=type(owner).__module__+'.'+type(owner).__qualname__,
            owner_mro=[c.__module__+'.'+c.__qualname__ for c in type(owner).__mro__],
            rank=observed(rank,'VERIFIED' if rank is not None else 'NOT_OBSERVED'),
            rank_field=profile.get('rank_path','rank'),
            rank_semantics='REVIEWED_OWNER_FIELD_NOT_AUTOMATIC_TP_EP_OR_LAYER_MAPPING',
            coverage='EXPLICIT_REVIEWED_ROOTS_ONLY',root_specs=profile.get('roots',[]))
        try:
            controls={}
            for name,path in profile.get('control_fields',{}).items():
                try:
                    value=resolve(owner,path)
                    if value is None or isinstance(value,(str,int,float,bool,list,tuple)):controls[name]=observed(scalar_metadata_sequence(value))
                    else:controls[name]={'value':None,'status':'VERIFIED','object_present':True,'object_type':type(value).__name__}
                except Exception as exc:controls[name]=observed(None,'NOT_OBSERVED',type(exc).__name__)
            if controls:self.emit('STATE_CONTROL_CONFIG_OBSERVED',source,snapshot_id=snapshot,fields=controls,
                semantics='EFFECTIVE_RUNTIME_OWNER_CONFIG; KV_TRANSFER_AND_ENGRAM_OFFLOAD_ARE_SEPARATE')
            config_path=profile.get('config_root');config=configuration
            if config_path:
                try:config=resolve(owner,config_path)
                except Exception:pass
            layer_groups={};ambiguous=set()
            if config is not None:
                try:
                    for group_id,group in enumerate(config.kv_cache_groups):
                        for layer in group.layer_names:
                            if not isinstance(layer,str):continue
                            if layer in layer_groups and layer_groups[layer]!=group_id:ambiguous.add(layer)
                            layer_groups[layer]=group_id
                    for layer in ambiguous:layer_groups.pop(layer,None)
                except Exception:layer_groups={}

            def emit_tensor(state_path,tensor,kind,role,layer_name=None):
                nonlocal count
                metadata,storage=tensor_metadata(tensor)
                storage_id=None;alias=False
                identity={}
                if storage is not None:
                    handle=getattr(storage,'_cdata',None)
                    key=('impl',handle) if type(handle)is int else ('wrapper',id(storage))
                    if key in stores:storage_id=stores[key][1];alias=True
                    else:
                        storage_id=uuid.uuid4().hex;stores[key]=(storage,storage_id)
                    try:
                        identity=self.storage_registry.describe(tensor)
                        storage_id=identity.pop('storage_object_id')
                    except Exception as exc:
                        # Preserve inventory metadata even for a backend without
                        # a live identity witness. Such IDs stay snapshot-local.
                        identity={'storage_identity_scope':'SNAPSHOT_LOCAL_ONLY',
                                  'storage_identity_status':'NOT_OBSERVED',
                                  'storage_identity_reason':type(exc).__name__}
                group=layer_groups.get(layer_name) if layer_name is not None else None
                self.emit('STATE_STORAGE_OBSERVED',source,snapshot_id=snapshot,state_path=state_path,
                    state_kind=kind,tensor_role=role,rank=rank,metadata=metadata,
                    storage_object_id=storage_id,storage_object_alias=alias,
                    **identity,
                    kv_layer_name=layer_name,group_id=observed(group,'VERIFIED' if group is not None else 'NOT_OBSERVED'),
                    group_binding_scope='EXACT_LAYER_KEY_TO_WORKER_CONFIG_GROUP_NOT_ENGINE_BLOCK_LAYOUT',
                    request_binding='NOT_OBSERVED',transfer_observed=False)
                count+=1

            for root in profile.get('roots',[]):
                path=root['path'];state_kind=root['state_kind']
                try:
                    value=returned if root.get('source')=='RETURN_VALUE' else (arguments or {})[root['argument']] if root.get('argument') else resolve(owner,path)
                    if root.get('fields'):
                        # Custom Engram/CPU managers may own storage outside
                        # registered Torch parameters. Only reviewed scalar
                        # fields are read, never arbitrary object serialization.
                        scalar_fields={}
                        for name,field_path in root['fields'].items():
                            field_value=resolve(value,field_path)
                            if field_value is not None and not isinstance(field_value,(str,int,float,bool)):
                                raise TypeError('backend metadata field is not scalar')
                            scalar_fields[name]=observed(field_value)
                        self.emit('STATE_BACKEND_METADATA_OBSERVED',source,snapshot_id=snapshot,
                            state_path=path,state_kind=state_kind,backend_fields=scalar_fields,
                            semantics='OBSERVED_SCALARS; units and medium require actual backend source contract',
                            request_binding='NOT_OBSERVED')
                        count+=1
                        continue
                    entries=[]
                    if root.get('layer_keyed') and isinstance(value,dict):
                        entries=((entry,name if isinstance(name,str) else None) for name,item in value.items() if isinstance(name,(str,int))
                            for entry in tensors(item,path+'['+name+']',state_kind))
                    else:entries=((entry,None) for entry in tensors(value,path,state_kind))
                    for (state_path,tensor,kind,role),layer_name in entries:
                        if count>=self.max_objects:truncated=True;break
                        emit_tensor(state_path,tensor,kind,role,layer_name)
                    if truncated:break
                except Exception as exc:
                    errors+=1;self.emit('INVENTORY_ROOT_UNAVAILABLE',source,snapshot_id=snapshot,
                        root_path=path,error_type=type(exc).__name__)
            for spec in profile.get('module_storage_roots',[]):
                matched=found=unavailable=0
                try:
                    modules=resolve(owner,'model').named_modules()
                    for module_path,instance in modules:
                        cls=type(instance)
                        if cls.__module__!=spec['module'] or cls.__name__!=spec['class']:continue
                        matched+=1
                        file=inspect.getsourcefile(cls)
                        if file is None or Path(file).resolve()!=Path(spec['source_file']).resolve() or hashlib.sha256(Path(file).read_bytes()).hexdigest()!=spec['file_sha256']:
                            unavailable+=1;continue
                        backend_fields={}
                        for name,field_path in spec.get('metadata_fields',{}).items():
                            try:backend_fields[name]=observed(scalar_metadata_sequence(resolve(instance,field_path)))
                            except Exception as exc:backend_fields[name]=observed(None,'NOT_OBSERVED',type(exc).__name__)
                        if backend_fields:self.emit('STATE_BACKEND_METADATA_OBSERVED',source,snapshot_id=snapshot,
                            state_path='model.'+module_path,state_kind='ENGRAM',backend_fields=backend_fields,
                            request_binding='NOT_OBSERVED',semantics='SOURCE_PINNED_NODE_SHARDED_ENGRAM_FIELDS')
                        for field in spec['attributes']:
                            formats=spec.get('attribute_formats',{}).get(field)
                            if formats and getattr(instance,'storage_format',None) not in formats:continue
                            try:
                                value=resolve(instance,field)
                                for state_path,tensor,kind,role in tensors(value,'model.'+module_path+'.'+field,'ENGRAM'):
                                    if count>=self.max_objects:truncated=True;break
                                    emit_tensor(state_path,tensor,kind,'SOURCE_PINNED_ENGRAM_ATTRIBUTE');found+=1
                            except Exception:unavailable+=1
                except Exception:unavailable+=1
                self.emit('ENGRAM_SOURCE_ROOT_COVERAGE',source,snapshot_id=snapshot,
                    declared_class=spec['module']+'.'+spec['class'],matched_instances=matched,
                    tensor_objects=found,unavailable_attributes=unavailable,
                    coverage_status='OBSERVED' if found and not unavailable else 'NOT_OBSERVED_OR_PARTIAL',
                    no_match_semantics='NOT_PROOF_OFFLOAD_DISABLED')
            if config_path or configuration is not None:
                try:
                    role=profile.get('config_role','WORKER')
                    pool=None
                    if role=='ENGINE':
                        # Reuse the logical observer's identity for this exact
                        # manager instance. This is an observer UUID, not HBM.
                        active=getattr(sys.modules.get(__package__+'.hook'),'_ACTIVE',None)
                        trace=getattr(active,'trace',None)
                        if (trace is not None and trace.pid==os.getpid()
                                and trace.config.run_id==self.service_run_id):
                            pool=trace.scope(owner)
                    self.capture_config(config,source,snapshot,role=role,logical_pool_scope=pool)
                except Exception as exc:
                    errors+=1;self.emit('INVENTORY_CONFIG_UNAVAILABLE',source,snapshot_id=snapshot,error_type=type(exc).__name__)
        finally:
            stores.clear()
            self.emit('INVENTORY_END',source,snapshot_id=snapshot,object_count=count,root_errors=errors,
                complete_for_declared_roots=not truncated and not errors,truncated=truncated)

    def capture_config(self,config,source,snapshot,role='WORKER',logical_pool_scope=None):
        groups=getattr(config,'kv_cache_groups',None)
        if not isinstance(groups,(list,tuple)):raise TypeError('kv_cache_groups not observed')
        blocks=getattr(config,'num_blocks',None)
        self.emit('KV_CACHE_PLAN_OBSERVED',source,snapshot_id=snapshot,
            config_role=role,
            num_blocks=observed(blocks,'VERIFIED' if type(blocks)is int else 'NOT_OBSERVED'),
            semantics='WORKER_CACHE_CONFIGURATION_NOT_ALLOCATOR_RESERVATION')
        for i,group in enumerate(groups):
            spec=getattr(group,'kv_cache_spec',None)
            layers=getattr(group,'layer_names',None)
            layer_names=list(layers) if isinstance(layers,(list,tuple)) and all(isinstance(x,str) for x in layers) else None
            details=describe_group_spec(spec,layer_names)
            self.emit('KV_GROUP_CONFIG_OBSERVED',source,snapshot_id=snapshot,group_id=i,
                config_role=role,
                logical_pool_scope=logical_pool_scope,
                logical_pool_scope_semantics='OBSERVED_LOGICAL_HOOK_MANAGER_INSTANCE_UUID' if logical_pool_scope else 'NOT_OBSERVED',
                layers=layer_names,spec_type=type(spec).__name__,**details,
                mapping_scope='GROUP_CONFIGURATION_NOT_ENGINE_POOL_OR_BLOCK_PHYSICAL_BINDING')

    def close(self):
        if self.sink and self.pid==os.getpid():self.sink.close()


class TransferRecorder:
    """Record actual provider submit/completion facts. Does not perform transfers."""
    def __init__(self,observer):self.observer=observer;self.pending={}
    def submit(self,source,*,state_id,src_medium,dst_medium,bytes_count,request_id=None,submitted_ns=None):
        if type(bytes_count)is not int or bytes_count<0:raise ValueError('invalid actual provider byte count')
        if src_medium not in {'HBM','DDR','SSD'} or dst_medium not in {'HBM','DDR','SSD'}:raise ValueError('unknown medium')
        transfer=uuid.uuid4().hex;start=time.perf_counter_ns() if submitted_ns is None else submitted_ns
        if type(start)is not int or start<=0:raise ValueError('invalid actual host submit timestamp')
        self.pending[transfer]=(start,bytes_count,source)
        self.observer.emit('STATE_TRANSFER_SUBMITTED',source,transfer_id=transfer,state_id=state_id,
            request_id=request_id,src_medium=src_medium,dst_medium=dst_medium,bytes_count=bytes_count,
            submitted_ns=start,transfer_clock_domain='HOST_PERF_COUNTER_PROCESS',
            byte_count_source='ACTUAL_PROVIDER_FIELD')
        return transfer
    def complete(self,transfer,*,completion_kind,completed_bytes=None,device_elapsed_ns=None):
        if completion_kind not in {'BACKEND_CALLBACK','DEVICE_EVENT','SYNCHRONOUS_BACKEND_RETURN'}:
            raise ValueError('enqueue return does not prove completion')
        start,planned,source=self.pending[transfer]
        actual=planned if completed_bytes is None else completed_bytes
        if type(actual)is not int or not 0<=actual<=planned:raise ValueError('invalid completed bytes')
        if device_elapsed_ns is not None and (type(device_elapsed_ns)is not int or device_elapsed_ns<=0):
            raise ValueError('invalid device timer')
        self.pending.pop(transfer)
        end=time.perf_counter_ns()
        self.observer.emit('STATE_TRANSFER_COMPLETED',source,transfer_id=transfer,
            completed_bytes=actual,completed_ns=end,submit_to_complete_ns=end-start,
            completion_kind=completion_kind,device_elapsed_ns=device_elapsed_ns,
            device_timer_scope='TRANSFER_EXECUTION' if device_elapsed_ns else 'NOT_OBSERVED')


class ProfileInstallation:
    def __init__(self,profile,observer):
        self.profile=profile;self.observer=observer;self.patches=[];self.finder=None
        from .transfer_observer import SimpleNPUTransferObserver,NativeNPUTransferObserver
        self.transfer_observer=SimpleNPUTransferObserver(observer)
        self.native_transfer_observer=NativeNPUTransferObserver(observer)
        from .request_media import RequestMediaObserver
        self.request_media_observer=RequestMediaObserver(observer)
    def patch_module(self,module):
        for target in self.profile['targets']:
            if target['module']!=module.__name__:continue
            source=target['module']+'.'+target['class']+'.'+target['method']
            try:
                cls=getattr(module,target['class']);descriptor=inspect.getattr_static(cls,target['method'])
                if not inspect.isfunction(descriptor):raise TypeError('only instance-method targets supported')
                original=descriptor
                if hasattr(original,'__wrapped__') and not target.get('allow_wrapped',False):
                    raise ValueError('decorated method needs explicit reviewed allow_wrapped contract')
                pinned=inspect.unwrap(original)
                path=Path(inspect.getsourcefile(pinned)).resolve()
                if path!=Path(target['source_file']).resolve():raise ValueError('loaded source path mismatch')
                if hashlib.sha256(path.read_bytes()).hexdigest()!=target['file_sha256']:raise ValueError('source SHA mismatch')
                if list(inspect.signature(original).parameters)!=target['parameters']:raise ValueError('signature mismatch')
                if inspect.iscoroutinefunction(original):raise TypeError('async lifecycle requires reviewed completion adapter')
                if hasattr(original,'__agentmon_placement__'):raise ValueError('duplicate installation')
                # Source/code equality is checked against the uniquely defined class method.
                compiled=compile(path.read_bytes(),str(path),'exec',dont_inherit=True,optimize=sys.flags.optimize)
                class_code=next(x for x in compiled.co_consts if inspect.iscode(x) and x.co_name==target['class'])
                method_code=next(x for x in class_code.co_consts if inspect.iscode(x) and x.co_name==target['method'])
                if pinned.__code__!=method_code:raise ValueError('loaded code differs from pinned disk method')
                wrapped=self.make_wrapper(original,target,source)
                setattr(cls,target['method'],wrapped);self.patches.append((cls,target['method'],original,wrapped))
                self.observer.emit('INVENTORY_TARGET_INSTALLED',source,
                    loaded_file=str(path),file_sha256=target['file_sha256'],loaded_code_matches_source=True,
                    target_profile=target,actual_method_execution='NOT_OBSERVED_UNTIL_INVENTORY_BEGIN')
            except Exception as exc:
                self.observer.emit('INVENTORY_TARGET_REJECTED',source,error_type=type(exc).__name__)
    def make_wrapper(self,original,target,source):
        observer=self.observer
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            # Metadata can be popped by handle_preemptions, and loads are
            # submitted inside start_kv_transfers: observe BEFORE original.
            # Failures here must never prevent the original call/exception.
            handler=target.get('handler')
            if handler in {'NATIVE_METADATA_LOAD','NATIVE_METADATA_STORE','NATIVE_METADATA_FLUSH'}:
                try:
                    allowed=target.get('expected_owner_classes')
                    if not allowed or any(c.__module__+'.'+c.__qualname__ in allowed for c in type(args[0]).__mro__):
                        bound=inspect.signature(original).bind(*args,**kwargs)
                        metadata=bound.arguments.get('metadata',bound.arguments.get('kv_connector_metadata'))
                        self.native_transfer_observer.before_metadata(args[0],metadata,handler.rsplit('_',1)[1],source)
                except Exception:
                    try:observer._sink().observer_failure()
                    except Exception:pass
                return original(*args,**kwargs)
            submitted=time.perf_counter_ns() if target.get('handler') in {'NPU_DMA_SUBMIT','NATIVE_NPU_SUBMIT','ENGRAM_CHECKPOINT_LOAD'} else None
            result=original(*args,**kwargs)
            try:
                allowed=target.get('expected_owner_classes')
                if allowed and not any(c.__module__+'.'+c.__qualname__ in allowed for c in type(args[0]).__mro__):return result
                bound=inspect.signature(original).bind_partial(*args,**kwargs)
                bound.apply_defaults()
                handler=target.get('handler')
                if handler=='REQUEST_MEDIA_STEP':
                    self.request_media_observer.observe_step(args[0],bound.arguments['scheduler_output'])
                    return result
                if handler=='NPU_DMA_SUBMIT':
                    self.transfer_observer.after_submit(args[0],bound.arguments,source,submitted);return result
                if handler=='SIMPLE_OFFLOAD_POLL':
                    self.transfer_observer.after_poll(args[0],bound.arguments['is_store'],result,source);return result
                if handler=='SIMPLE_OFFLOAD_FLUSH':
                    self.transfer_observer.after_flush(args[0],source);return result
                if handler=='NATIVE_NPU_SUBMIT':
                    self.native_transfer_observer.after_submit(args[0],bound.arguments,source,submitted,result);return result
                if handler=='NATIVE_NPU_FINISHED':
                    self.native_transfer_observer.after_finished(args[0],result,source);return result
                if handler=='NATIVE_CONNECTOR_FINISHED':
                    self.native_transfer_observer.after_connector_finished(args[0],result,source);return result
                if handler=='ENGRAM_CHECKPOINT_LOAD':
                    ended=time.perf_counter_ns()
                    payload=0;destinations=[]
                    for field in ('weight','weight_scale'):
                        tensor=getattr(args[0],field,None)
                        if is_tensor(tensor):
                            payload+=int(tensor.numel())*int(tensor.element_size())
                            destinations.append(str(tensor.device))
                    observer.emit('STATE_WEIGHT_CHECKPOINT_LOAD_COMPLETED',source,
                        call_elapsed_ns=ended-submitted,destination_view_bytes=payload,
                        destination_devices=sorted(set(destinations)),
                        metric_scope='INITIAL_CHECKPOINT_MATERIALIZATION_NOT_REQUEST_OFFLOAD_OR_PHYSICAL_DISK_BANDWIDTH',
                        actual_ssd_read_bytes=observed(None,'NOT_OBSERVED'),request_binding='NOT_APPLICABLE_STARTUP_LOAD')
                config=bound.arguments.get(target.get('config_argument'))
                observer.capture(args[0],target,source,configuration=config,returned=result,arguments=bound.arguments)
                if target.get('request_media'):
                    try:
                        self.request_media_observer.register(args[0],config)
                    except Exception as exc:
                        observer.emit('KV_MEDIA_CONTRACT_REJECTED',source,error_type=type(exc).__name__,
                            reason=str(exc)[:1200])
                        observer._sink().observer_failure()
            except Exception:
                try:observer._sink().observer_failure()
                except Exception:pass
            return result
        wrapped.__agentmon_placement__=True
        return wrapped
    def close(self):
        if self.finder in sys.meta_path:sys.meta_path.remove(self.finder)
        for cls,name,old,new in reversed(self.patches):
            if inspect.getattr_static(cls,name) is new:setattr(cls,name,old)
        self.observer.close()


class _Loader:
    def __init__(self,base,installation):self.base=base;self.installation=installation
    def create_module(self,spec):
        return self.base.create_module(spec) if hasattr(self.base,'create_module') else None
    def exec_module(self,module):
        self.base.exec_module(module)
        try:self.installation.patch_module(module)
        except Exception:
            try:self.installation.observer._sink().observer_failure()
            except Exception:pass
    def __getattr__(self,name):return getattr(self.base,name)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self,installation):self.installation=installation
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in {t['module'] for t in self.installation.profile['targets']}:return None
        found=False
        for finder in tuple(sys.meta_path):
            if finder is self:found=True;continue
            if not found:continue
            method=getattr(finder,'find_spec',None)
            spec=method(fullname,path,target) if method else None
            if spec is not None:
                if spec.loader and hasattr(spec.loader,'exec_module'):spec.loader=_Loader(spec.loader,self.installation)
                return spec
        return None


def install_from_env(*,startup=False):
    global _ACTIVE
    if str(os.environ.get(FLAG,'')).lower() not in {'1','true','yes','on'}:return None
    if _ACTIVE:return _ACTIVE
    if not startup:raise RuntimeError('startup only; no injection into live workers')
    path=Path(os.environ['AGENTMON_STATE_PLACEMENT_PROFILE'])
    profile=json.loads(path.read_text(encoding='utf-8'))
    if not profile.get('targets'):raise ValueError('reviewed actual server targets required')
    if any(t['module'] in sys.modules for t in profile['targets']):raise RuntimeError('target already loaded')
    observer=InventoryObserver(os.environ['AGENTMON_STATE_PLACEMENT_DIR'],os.environ['AGENTMON_VLLM_0271_TRACE_RUN_ID'])
    installation=ProfileInstallation(profile,observer);installation.finder=_Finder(installation)
    sys.meta_path.insert(0,installation.finder);atexit.register(observer.close);_ACTIVE=installation
    return installation
