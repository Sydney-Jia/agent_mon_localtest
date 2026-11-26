"""Opt-in observations of the vLLM 0.30 Python secondary file tier."""
import contextvars
import functools
import hashlib
import os
import sys
import time
import weakref

from .instrument import checked_methods, instrument_method


IO_MODULE = 'vllm.v1.kv_offload.tiering.fs.io'
THREAD_MODULE = 'vllm.v1.kv_offload.tiering.fs.thread_pool'
MANAGER_MODULE = 'vllm.v1.kv_offload.tiering.fs.manager'
METHODS = {
    IO_MODULE: {None: ['batch_store_block', 'batch_load_block', '_store_block', '_load_block']},
    THREAD_MODULE: {'DualQueueThreadPool': ['enqueue_load','enqueue_store','get_finished','_worker'],
                    'JobState':['task_done']},
    MANAGER_MODULE: {'FileSystemTierManager': ['__init__','lookup','submit_store', 'submit_load', 'get_finished_jobs']},
}
CALLS = {
    '_store_block': {'os.path.exists':'exists', 'os.open':'open_store', 'os.write':'write', 'os.replace':'publish'},
    '_load_block': {'os.open':'open_load', 'os.readv':'readv'},
    'submit_store': {'self._pool.enqueue_store':'STORE'},
    'submit_load': {'self._pool.enqueue_load':'LOAD'},
}
_JOB = contextvars.ContextVar('agentmon_fs_job', default=None)


def path_id(path):
    return hashlib.sha256(os.fsencode(path)).hexdigest()


def key_id(key):
    if not isinstance(key, bytes):
        raise TypeError('UNSUPPORTED_FS_KEY_REPRESENTATION')
    return hashlib.sha256(key).hexdigest()


class FileTierObserver:
    def __init__(self, trace):
        self.trace = trace
        self.roots = weakref.WeakKeyDictionary()

    def emit(self, name, identity, **fields):
        try:
            self.trace.emit(name, 'FileSystemTierManager', name, fs_schema='agentmon.fs-tier/1',
                            **identity, **fields)
        except Exception:
            self.trace.failure()

    def trace_work(self, work, identity):
        @functools.wraps(work)
        def run():
            token = _JOB.set(identity)
            start = time.monotonic_ns()
            status = 'ERROR'
            try:
                result = work()
                status = 'RETURNED'
                return result
            finally:
                end = time.monotonic_ns()
                _JOB.reset(token)
                self.emit('FS_TASK_END', identity, start_ns=start, end_ns=end, status=status,
                          timing_scope='WORKER_HOST_TASK_INCLUDING_OBSERVER')
        return run

    def enqueue(self, direction, original, args, kwargs, local):
        # Read the argument lists already built by the reviewed manager. Do not
        # consume job.keys a second time or call FileMapper / os.path.exists.
        try:
            job_id, count, tasks = args
            if kwargs or count != 1 or len(tasks) != 1:
                raise ValueError('FS_TASK_SHAPE')
            owner, job = local['self'], local['job_metadata']
            if type(owner._pool) is not sys.modules[THREAD_MODULE].DualQueueThreadPool:
                raise TypeError('FS_POOL_TYPE_MISMATCH')
            if direction == 'STORE':
                paths, view, offsets, stride, direct = tasks[0].args
            else:
                paths, offsets = local['paths'], local['offsets']
                view, stride, direct = owner._primary_kv_view, owner._block_size, owner._use_o_direct
            keys = local['keys']
            if not (len(keys) == len(paths) == len(offsets) == len(job.chunk_ids)):
                raise ValueError('FS_DESCRIPTOR_LENGTHS')
            io = sys.modules[IO_MODULE]
            identity = dict(tier_scope=self.trace.scope(owner), job_id=int(job_id),
                            request_id=job.req_context.req_id, direction=direction)
            self.emit('FS_JOB_PLAN', identity, blocks=len(keys), host_slot_stride_bytes=stride,
                      host_pool_bytes=view.nbytes, use_o_direct=direct,
                      data_path='FS_IO_C' if io._HAS_FSIO_C else 'PYTHON',
                      source_medium='DDR' if direction=='STORE' else 'FILE',
                      target_medium='FILE' if direction=='STORE' else 'DDR',
                      enclosing_hbm_job_id=None, host_slot_lease=None)
            for offset in range(0, len(keys), 128):
                rows = [dict(ordinal=i, key_digest=key_id(keys[i]), offload_key_hex=keys[i].hex(), file_id=path_id(paths[i]),
                             file_relative=os.path.relpath(paths[i], self.roots[owner]) if owner in self.roots else None,
                             chunk_id=int(job.chunk_ids[i]), host_offset=offsets[i],
                             file_offset=0) for i in range(offset, min(offset+128, len(keys)))]
                self.emit('FS_JOB_SLOTS', identity, first=offset, slots=rows)
            args = (job_id, count, [self.trace_work(tasks[0],identity)])
        except Exception:
            self.trace.failure()
            return original(*args, **kwargs)
        start = time.monotonic_ns()
        status = 'ERROR'
        try:
            result = original(*args, **kwargs)
            status = 'ENQUEUED'
            return result
        finally:
            self.emit('FS_ENQUEUE_RETURN', identity, status=status, start_ns=start, end_ns=time.monotonic_ns())

    def data_call(self, operation, original, args, kwargs, local):
        job = _JOB.get()
        if job is None:
            return original(*args, **kwargs)
        start = time.monotonic_ns()
        try:
            result = original(*args, **kwargs)
        except Exception as exc:
            self._data_result(job,operation,local,start,time.monotonic_ns(),None,exc)
            raise
        self._data_result(job,operation,local,start,time.monotonic_ns(),result,None)
        return result

    def _data_result(self, job, operation, local, start, end, result, error):
        try:
            # Original paths resolve the same file_id as FS_JOB_SLOTS. No payload or address logged.
            path = local['source_path'] if operation in ('open_load','readv') else local['dest_path']
            self.emit('FS_FILE_OPERATION', job, file_id=path_id(path), host_offset=local['offset'],
                      slot_bytes=local['block_size'], operation=operation, start_ns=start,
                      end_ns=end, error=type(error).__name__ if error else None,
                      errno=getattr(error,'errno',None), syscall_bytes=result if operation in ('write','readv') else None,
                      skipped_existing=(result if operation=='exists' and error is None else None))
        except Exception:
            self.trace.failure()

    def lookup(self, original):
        @functools.wraps(original)
        def wrapped(owner, key, req_context):
            result=original(owner,key,req_context)
            try:
                self.emit('FS_LOOKUP',dict(tier_scope=self.trace.scope(owner),request_id=req_context.req_id),
                          key_digest=key_id(key),lookup_result=result.name)
            except Exception:
                self.trace.failure()
            return result
        return wrapped

    def initialize(self, original):
        @functools.wraps(original)
        def wrapped(owner, *args, **kwargs):
            original(owner, *args, **kwargs)
            try:
                import inspect
                bound = inspect.signature(original).bind(owner, *args, **kwargs)
                bound.apply_defaults()
                cfg = bound.arguments
                self.roots[owner] = cfg['root_dir']
                view = owner._primary_kv_view
                spec = cfg['offloading_spec']
                self.emit('FS_GEOMETRY', dict(tier_scope=self.trace.scope(owner)),
                          shape=list(view.shape), strides_bytes=list(view.strides),
                          itemsize=view.itemsize, format=view.format,
                          host_pool_bytes=view.nbytes, host_slot_stride_bytes=owner._block_size,
                          num_chunks=spec.num_chunks, blocks_per_chunk=spec.blocks_per_chunk,
                          world_size=spec.config.parallel.world_size,
                          replicated_layout=spec.replicated_layout,
                          cpu_page_size_per_worker=spec.cpu_page_size_per_worker,
                          kv_bytes_per_chunk=spec.kv_bytes_per_chunk,
                          root_dir=str(cfg['root_dir']), n_read_threads=cfg['n_read_threads'],
                          n_write_threads=cfg['n_write_threads'], use_o_direct=owner._use_o_direct,
                          data_path='FS_IO_C' if sys.modules[IO_MODULE]._HAS_FSIO_C else 'PYTHON')
            except Exception:
                self.trace.failure()
        return wrapped

    def finished(self, original):
        @functools.wraps(original)
        def poll(owner):
            results=original(owner)
            try:
                # Reviewed version returns a list, never iterate an unverified generator.
                if not isinstance(results,list):
                    raise TypeError('FS_FINISHED_RESULT_SHAPE')
                for row in results:
                    identity=dict(tier_scope=self.trace.scope(owner),job_id=int(row.job_id))
                    successful=row.successful_keys
                    self.emit('FS_FINISHED_JOB',identity,success=row.success,
                              successful_keys=None if successful is None else [key_id(k) for k in successful],
                              native_transfer_time_s=row.transfer_time,host_polled_ns=time.monotonic_ns())
            except Exception:
                self.trace.failure()
            return results
        return poll

    def patch(self, installation, module):
        name=module.__name__
        source, methods = checked_methods(module, installation.hashes[name], METHODS[name])
        if name == THREAD_MODULE:
            return
        if name == MANAGER_MODULE:
            io = sys.modules.get(IO_MODULE)
            if (IO_MODULE not in installation.installed or THREAD_MODULE not in installation.installed or io is None or
                    any(getattr(module, n) is not getattr(io, n) for n in ('batch_load_block','batch_store_block'))):
                raise ValueError('FS_IO_MODULE_BINDING_UNVERIFIED')
        pending, helpers = [], []
        try:
            for (cls, method), original in methods.items():
                if method.startswith('batch_'):
                    continue
                owner = module if cls is None else getattr(module, cls)
                if method=='__init__':
                    replacement=self.initialize(original)
                elif method=='lookup':
                    replacement=self.lookup(original)
                elif method=='get_finished_jobs':
                    replacement=self.finished(original)
                else:
                    callback=self.data_call if cls is None else self.enqueue
                    replacement, helper=instrument_method(source,module,cls,method,original,CALLS[method],callback)
                    helpers.append(helper)
                pending.append((owner,method,original,replacement))
        except Exception:
            for namespace,key in helpers:
                namespace.pop(key,None)
            raise
        for owner,method,_,replacement in pending:
            setattr(owner,method,replacement)
        installation.patches.extend(pending)
        installation.helpers.extend(helpers)
