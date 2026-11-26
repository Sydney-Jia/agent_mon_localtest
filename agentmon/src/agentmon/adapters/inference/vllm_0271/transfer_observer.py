"""Passive adapter for the pinned SimpleCPUOffloadNPUWorker DMA backend.

Reads byte descriptors only; never reads src/dst base addresses, polls an Event,
records an Event, synchronizes a stream, or submits a copy itself.
"""
import uuid
import math
import weakref


class SimpleNPUTransferObserver:
    def __init__(self,observer,max_pending=4096):
        self.observer=observer;self.max_pending=max_pending;self.pending={};self.recorder=None
    def after_submit(self,backend,args,source,submitted_ns):
        from .placement import TransferRecorder
        if self.recorder is None:self.recorder=TransferRecorder(self.observer)
        direction=args['is_store'];index=args['event_idx'];events=args['events_list']
        if type(direction)is not bool or type(index)is not int:raise ValueError('unrecognized provider event')
        key=(id(events),direction,index)
        if key in self.pending:
            self.pending[key]['ambiguous']=True
            self.observer.emit('TRANSFER_PROVIDER_ID_COLLISION',source,event_idx=index,is_store=direction)
            return
        if len(self.pending)>=self.max_pending:
            self.observer.emit('TRANSFER_OBSERVATION_CAPACITY_EXCEEDED',source,capacity=self.max_pending);return
        params=backend._store_params if direction else backend._load_params
        bpb=[int(x) for x in params.bpb] # CPU NumPy descriptor, not model/Tensor content.
        src=args['src_blocks'];dst=args['dst_blocks']
        if len(src)!=len(dst) or any(x<0 for x in bpb):raise ValueError('invalid actual DMA descriptors')
        size=len(src)*sum(bpb)
        transfer=self.recorder.submit(source,state_id='kv-dma-bundle-'+uuid.uuid4().hex,
            src_medium='HBM' if direction else 'DDR',dst_medium='DDR' if direction else 'HBM',
            bytes_count=size,submitted_ns=submitted_ns)
        self.pending[key]={'transfer_id':transfer,'events':events,'ambiguous':False}
        self.observer.emit('STATE_TRANSFER_BLOCK_PLAN',source,transfer_id=transfer,event_idx=index,is_store=direction,
            src_blocks=list(src[:256]),dst_blocks=list(dst[:256]),block_pairs=len(src),
            block_list_truncated=len(src)>256,bytes_per_block_by_tensor=bpb,
            byte_formula='len(src_blocks) * sum(actual BatchMemcpyParams.bpb)',
            block_namespace='SIMPLE_OFFLOAD_PROVIDER_SLOTS_NOT_PROVEN_ENGINE_BLOCK_IDS',
            request_binding='NOT_OBSERVED_AT_DMA_SUBMIT',medium_basis='PINNED_NPU_DMA_DIRECTION_CONTRACT')
    def after_poll(self,worker,is_store,watermark,source,kind='DEVICE_EVENT'):
        if type(watermark)is not int or type(is_store)is not bool:raise ValueError('invalid observed high-water mark')
        events=worker._store_events if is_store else worker._load_events
        matched=[key for key in self.pending if key[0]==id(events) and key[1]==is_store and key[2]<=watermark]
        for key in matched:
            entry=self.pending[key]
            if entry['ambiguous']:continue
            self.recorder.complete(entry['transfer_id'],completion_kind=kind)
            metadata=getattr(worker,'_connector_metadata',None)
            mapping=getattr(metadata,'load_event_to_reqs',None) if not is_store else None
            reqs=mapping.get(key[2]) if isinstance(mapping,dict) else None
            self.observer.emit('STATE_TRANSFER_PROVIDER_COMPLETION',source,transfer_id=entry['transfer_id'],
                event_idx=key[2],observed_watermark=watermark,is_store=is_store,
                request_ids=sorted(reqs) if isinstance(reqs,(list,tuple,set)) and all(isinstance(r,str) for r in reqs) else None,
                request_id_scope='EXPLICIT_CONNECTOR_REQUEST_IDS_NOT_API_RESPONSE_IDS',
                completion_basis='ORIGINAL_PROVIDER_QUERY_OR_SYNC_ALREADY_COMPLETED',
                timing_scope='SUBMIT_TO_PROVIDER_OBSERVED_COMPLETION_INCLUDES_QUEUE_AND_POLL_WAIT',
                extra_device_poll_or_sync=False)
            del self.pending[key]
    def after_flush(self,worker,source):
        self.after_poll(worker,False,worker._load_hwm,source,kind='SYNCHRONOUS_BACKEND_RETURN')
        self.after_poll(worker,True,worker._store_hwm,source,kind='SYNCHRONOUS_BACKEND_RETURN')


class NativeNPUTransferObserver:
    """Use actual queued Transfer.num_bytes and returned TransferResult timers."""
    def __init__(self,observer,max_pending=4096):
        self.observer=observer;self.max_pending=max_pending;self.pending={};self.recorder=None
        self.owners=weakref.WeakKeyDictionary()
        self.request_jobs={}
        self.completed_loads={}

    def before_metadata(self,worker,metadata,mode,source):
        """Record the connector's explicit identity before it pops/defer jobs.

        Exact descriptor object identity, direction and job ID are all required.
        Holding the CPU descriptors prevents Python id reuse. No tensor read,
        stream wait or connector mutation is performed here.
        """
        stores=mode in {'STORE','FLUSH'}
        if stores and worker._is_store_writer is not True:return
        jobs=metadata.store_jobs if stores else metadata.load_jobs
        flush=(metadata.jobs_to_flush or set()) if mode=='FLUSH' else None
        for job,entry in jobs.items():
            if flush is not None and job not in flush:continue
            if type(job)is not int or not isinstance(entry.req_id,str) or not entry.req_id:
                raise ValueError('invalid explicit connector job identity')
            key=(job,stores,id(entry.src_spec),id(entry.dst_spec))
            identity=(self.owner_id(worker),entry.req_id)
            old=self.request_jobs.get(key)
            if old:
                if old['identity']!=identity:
                    old['ambiguous']=True
                    self.observer.emit('TRANSFER_REQUEST_BINDING_COLLISION',source,job_id=job)
                continue
            if len(self.request_jobs)>=self.max_pending:
                self.observer.emit('TRANSFER_REQUEST_BINDING_CAPACITY_EXCEEDED',source,capacity=self.max_pending)
                continue
            self.request_jobs[key]=dict(identity=identity,src=entry.src_spec,dst=entry.dst_spec,
                ambiguous=False,source=source)
    def owner_id(self,owner):
        if owner not in self.owners:self.owners[owner]=uuid.uuid4().hex
        return self.owners[owner]
    def after_submit(self,owner,args,source,submitted_ns,result):
        if result is not True:return
        from .placement import TransferRecorder
        if self.recorder is None:self.recorder=TransferRecorder(self.observer)
        job=args['job_id'];key=(self.owner_id(owner),job)
        if key in self.pending:
            self.pending[key]['ambiguous']=True
            self.observer.emit('TRANSFER_PROVIDER_ID_COLLISION',source,job_id=job);return
        if len(self.pending)>=self.max_pending:
            self.observer.emit('TRANSFER_OBSERVATION_CAPACITY_EXCEEDED',source,capacity=self.max_pending);return
        actual=next((t for t in reversed(owner._transfers) if t.job_id==job),None)
        if actual is None:raise ValueError('actual queued provider Transfer not observed')
        direction=owner.npu_to_cpu
        if type(direction)is not bool or type(actual.num_bytes)is not int:raise ValueError('unrecognized actual provider metadata')
        token=self.recorder.submit(source,state_id='native-kv-dma-bundle-'+uuid.uuid4().hex,
            src_medium='HBM' if direction else 'DDR',dst_medium='DDR' if direction else 'HBM',
            bytes_count=actual.num_bytes,submitted_ns=submitted_ns)
        bridge_key=(job,direction,id(args.get('src_spec')),id(args.get('dst_spec')))
        bridge=self.request_jobs.pop(bridge_key,None)
        request_ids=[bridge['identity'][1]] if bridge and not bridge['ambiguous'] else None
        self.pending[key]={'token':token,'ambiguous':False,'request_ids':request_ids,
            'connector_owner_id':bridge['identity'][0] if request_ids else None,'is_store':direction}
        if request_ids:
            self.observer.emit('STATE_TRANSFER_REQUEST_BOUND',source,transfer_id=token,job_id=job,
                request_ids=request_ids,connector_owner_id=bridge['identity'][0],
                binding_basis='EXACT_JOB_DIRECTION_AND_SAME_SOURCE_DESTINATION_DESCRIPTOR_OBJECTS',
                connector_evidence_source=bridge['source'],
                request_id_scope='EXPLICIT_CONNECTOR_REQUEST_IDS_NOT_API_RESPONSE_IDS',
                engine_identity_bridge='REQUIRES_SEPARATE_VERIFIED_WORKER_ENGINE_BINDING')
        self.observer.emit('STATE_TRANSFER_NATIVE_JOB',source,transfer_id=token,job_id=job,
            direction='NPU_TO_CPU' if direction else 'CPU_TO_NPU',actual_provider_num_bytes=actual.num_bytes,
            request_binding='EXPLICIT_CONNECTOR_DESCRIPTOR_BINDING' if request_ids else 'NOT_OBSERVED_FROM_JOB_ID_ALONE',
            medium_basis='PINNED_NATIVE_NPU_BACKEND_CONTRACT')
        try:
            self.observe_endpoints(owner,args,source,token,job)
        except Exception as exc:
            # Metadata failure must not discard the already observed transfer
            # or stop the original provider. Completion remains independently
            # auditable; missing endpoints do not become imaginary replicas.
            self.observer.emit('STATE_TRANSFER_ENDPOINTS_UNAVAILABLE',source,
                transfer_id=token,job_id=job,reason=type(exc).__name__)

    def observe_endpoints(self,owner,args,source,token,job):
        """Pinned native handler metadata, not raw DMA pointers or Tensor data.

        Source contract: cpu_npu.py acb05a70... transfer_async uses src/dst
        tensors and group_data_refs. Provider slots are NOT Engine Block IDs.
        This records containers and descriptors; no content-validity inference.
        """
        import hashlib
        import json
        from .placement import tensor_metadata
        src,dst=owner.src_tensors,owner.dst_tensors
        if not isinstance(src,list) or not isinstance(dst,list) or len(src)!=len(dst):
            raise ValueError('native tensor pair metadata invalid')
        npu=args['src_spec'] if owner.npu_to_cpu else args['dst_spec']
        groups=owner.kv_cache_groups_data_refs
        sizes,indices=npu.group_sizes,npu.block_indices
        if len(sizes)!=len(groups) or len(indices)!=len(groups):
            raise ValueError('native group metadata mismatch')
        # Only CPU NumPy integer descriptors are accepted. Never invoke
        # Tensor.tolist()/cpu()/numpy() to retrieve device contents.
        def block_ids(spec):
            value=spec.block_ids
            if (type(value).__module__!='numpy' or type(value).__name__!='ndarray'
                    or value.ndim!=1 or value.dtype.kind not in 'iu'):
                raise TypeError('expected CPU NumPy block descriptor')
            if len(value)>65536:raise ValueError('endpoint descriptor bound exceeded')
            result=value.tolist()
            if any(type(b)is not int or b<0 for b in result):raise ValueError('invalid provider slot')
            return result
        source_blocks,destination_blocks=block_ids(args['src_spec']),block_ids(args['dst_spec'])
        records=[]
        for i,(a,b) in enumerate(zip(src,dst)):
            records.append(dict(kind='TENSOR_PAIR',tensor_index=i,
                source={**self.observer.storage_registry.describe(a),'metadata':tensor_metadata(a)[0]},
                destination={**self.observer.storage_registry.describe(b),'metadata':tensor_metadata(b)[0]}))
        for i,refs in enumerate(groups):
            size,index=sizes[i],indices[i]
            if type(size)is not int or size<0 or type(index)is not int or index<0:
                raise ValueError('invalid native group size/index')
            for ref_index,ref in enumerate(refs):
                if (type(ref.tensor_idx)is not int or not 0<=ref.tensor_idx<len(src)
                        or type(ref.page_size_bytes)is not int or ref.page_size_bytes<=0):
                    raise ValueError('invalid canonical tensor ref')
                records.append(dict(kind='GROUP_TENSOR_REF',provider_group_id=i,ref_index=ref_index,
                    group_size=size,block_index=index,tensor_index=ref.tensor_idx,page_size_bytes=ref.page_size_bytes))
        for side,values in [('SOURCE',source_blocks),('DESTINATION',destination_blocks)]:
            for offset in range(0,len(values),256):
                records.append(dict(kind='BLOCK_DESCRIPTOR',side=side,offset=offset,
                                    total=len(values),provider_slots=values[offset:offset+256]))
        # Exact count + digest exposes a missing tail/chunk. Empty descriptors
        # are explicit. Nothing here labels a cache content replica as Ready.
        checksum=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        common=dict(transfer_id=token,job_id=job,provider_owner_id=self.owner_id(owner),
                    endpoint_contract='NATIVE_NPU_TENSOR_ENDPOINTS_V1')
        for index,record in enumerate(records):
            self.observer.emit('STATE_TRANSFER_ENDPOINT_RECORD',source,**common,
                               record_index=index,record=record)
        self.observer.emit('STATE_TRANSFER_ENDPOINTS_END',source,**common,
            record_count=len(records),records_sha256=checksum,
            source_block_count=len(source_blocks),destination_block_count=len(destination_blocks),
            src_blocks_per_chunk=owner.src_blocks_per_chunk,dst_blocks_per_chunk=owner.dst_blocks_per_chunk,
            semantics='PROVIDER_STORAGE_AND_SLOT_DESCRIPTORS; NOT_ENGINE_BLOCK_OR_CONTENT_REPLICA',
            request_binding='NOT_OBSERVED_FROM_JOB_ID_ALONE',replica_validity='NOT_OBSERVED')
    def after_finished(self,owner,results,source):
        if not isinstance(results,list):raise ValueError('native completion must be original list result')
        identity=self.owner_id(owner)
        for result in results:
            key=(identity,result.job_id);entry=self.pending.get(key)
            if entry is None:
                self.observer.emit('TRANSFER_COMPLETE_WITHOUT_OBSERVED_SUBMIT',source,job_id=result.job_id);continue
            if entry['ambiguous']:continue
            if result.success is not True:
                self.observer.emit('STATE_TRANSFER_FAILED',source,transfer_id=entry['token'],job_id=result.job_id)
                self.recorder.pending.pop(entry['token'],None);del self.pending[key];continue
            elapsed=result.transfer_time
            timer=int(round(elapsed*1e9)) if isinstance(elapsed,(int,float)) and math.isfinite(elapsed) and elapsed>0 else None
            if timer==0:timer=None
            self.recorder.complete(entry['token'],completion_kind='DEVICE_EVENT',
                completed_bytes=result.transfer_size,device_elapsed_ns=timer)
            self.observer.emit('STATE_TRANSFER_PROVIDER_COMPLETION',source,transfer_id=entry['token'],job_id=result.job_id,
                request_ids=entry.get('request_ids'),
                request_id_scope='EXPLICIT_CONNECTOR_REQUEST_IDS_NOT_API_RESPONSE_IDS',
                completion_basis='ORIGINAL_GET_FINISHED_END_EVENT_QUERY_TRUE',
                timer_source='ORIGINAL_TRANSFER_RESULT_TRANSFER_TIME_SECONDS',
                timer_status='VERIFIED' if timer else 'NOT_OBSERVED_OR_BELOW_RESOLUTION',
                extra_device_poll_or_sync=False)
            if not entry['is_store'] and entry['request_ids']:
                request=entry['request_ids'][0]
                ready_key=(entry['connector_owner_id'],request)
                # Bound by transfer count, not merely the number of requests.
                if sum(len(x) for x in self.completed_loads.values())<self.max_pending:
                    self.completed_loads.setdefault(ready_key,[]).append(entry['token'])
                else:self.observer.emit('TRANSFER_REQUEST_BINDING_CAPACITY_EXCEEDED',source,capacity=self.max_pending)
            del self.pending[key]

    def after_connector_finished(self,worker,result,source):
        """Observe original connector readiness; never resume a request ourselves."""
        if not isinstance(result,tuple) or len(result)!=2 or not isinstance(result[1],set):
            raise ValueError('unexpected original connector completion result')
        owner=self.owner_id(worker)
        for request in sorted(result[1]):
            transfers=self.completed_loads.pop((owner,request),[])
            self.observer.emit('STATE_TRANSFER_CONNECTOR_READY',source,
                connector_owner_id=owner,request_id=request,transfer_ids=transfers,
                evidence_status='VERIFIED' if transfers else 'NOT_OBSERVED',
                readiness_scope='ORIGINAL_CONNECTOR_FINISHED_RECVING_NOT_ENGINE_RESUME_OR_DEVICE_CONSUMPTION')
