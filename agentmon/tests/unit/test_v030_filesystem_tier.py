"""Runs upstream v0.30.0 file methods/worker threads on a temporary filesystem."""
import importlib.util
import logging
from enum import Enum
import os
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest

from agentmon.adapters.inference.vllm_030 import filesystem as fs
from agentmon.adapters.inference.vllm_030.hook import Installation, install, uninstall


FIX = Path(__file__).resolve().parents[1]/'fixtures/v030_fs'


def load_fixture(name, filename, monkeypatch):
    spec=importlib.util.spec_from_file_location(name,FIX/filename)
    mod=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,name,mod)
    spec.loader.exec_module(mod)
    return mod


class Capture:
    def __init__(self):
        self.events=[]
        self.failures=0

    def emit(self, event, role, source, **fields):
        self.events.append(dict(event=event,**fields))

    def scope(self, _):
        return 'test-tier'

    def failure(self):
        self.failures+=1


@pytest.fixture
def observed(tmp_path,monkeypatch):
    trace=Capture()
    monitor=Installation(trace,filesystem=True,legacy=False)
    # Only the vLLM data types and non-FS dependencies are stubbed. Compile
    # all file I/O, manager submit/finish and actual worker methods unmodified.
    def stub(name, **symbols):
        mod=types.ModuleType(name)
        mod.__dict__.update(symbols)
        monkeypatch.setitem(sys.modules,name,mod)
        return mod
    def job_result(**kw):
        return types.SimpleNamespace(**({'successful_keys':None}|kw))
    stub('vllm.logger',init_logger=logging.getLogger)
    stub('vllm.v1.kv_offload.base',Locality=str,LookupResult=Enum('LookupResult','HIT MISS RETRY'),Medium=type('Medium',(),{'STORAGE':'FS'}),
         OffloadingEvent=dict,OffloadKey=bytes,ReqContext=object)
    stub('vllm.v1.kv_offload.file_mapper',FileMapper=object)
    stub('vllm.v1.kv_offload.tiering.async_lookup',AsyncLookupManager=object)
    class Secondary:
        def __init__(self, spec, view, tier_type):
            self._primary_kv_view=view
            self.tier_type=tier_type
    stub('vllm.v1.kv_offload.tiering.base',JobId=int,JobResult=job_result,
         RequestOffloadingContext=object,ScheduleEndContext=object,SecondaryTierManager=Secondary,TransferJob=object)
    stub('typing_extensions',override=lambda f:f)
    io=load_fixture(fs.IO_MODULE,'io.py',monkeypatch)
    if not hasattr(os,'readv'):
        # Windows compatibility for the POSIX data syscall only.
        def readv(fd, buffers):
            with open(fd,'rb',closefd=False) as f:
                return f.readinto(buffers[0])
        monkeypatch.setattr(os,'readv',readv,raising=False)
    threads=load_fixture(fs.THREAD_MODULE,'thread_pool.py',monkeypatch)
    manager=load_fixture(fs.MANAGER_MODULE,'manager.py',monkeypatch)
    for mod in (io,threads,manager):
        monitor.patch(mod)
    owner=object.__new__(manager.FileSystemTierManager)
    owner._primary_kv_view=memoryview(bytearray(b'aaaabbbbcccc'))
    owner._block_size=4
    owner.file_mapper=types.SimpleNamespace(get_file_name=lambda k:str(tmp_path/k.hex()))
    owner._use_o_direct=False
    owner._pool=threads.DualQueueThreadPool(1,1,'agentmon-test')
    owner._load_progress={}
    owner._load_job_keys={}
    owner._lookup_manager=types.SimpleNamespace(mark_miss=lambda _:None)
    owner.events=None
    try:
        yield owner,trace,manager,io
    finally:
        owner._pool.wait_idle()
        owner._pool.shutdown()
        monitor.uninstall()


def job(j, keys, chunks):
    return types.SimpleNamespace(job_id=j,req_context=types.SimpleNamespace(req_id='request-A'),
                                 keys=keys,chunk_ids=chunks)


def finish(m):
    m._pool.wait_idle()
    return m.get_finished_jobs()[0]


def test_python_fs_job_real_io_skip_and_partial_load(observed):
    m,t,_,_=observed
    keys=[b'k1',b'k2']
    m.submit_store(job(8,keys,[0,2]))
    assert finish(m).success
    m.submit_store(job(9,keys,[0,2]))
    assert finish(m).success
    m.submit_load(job(10,keys,[1,0]))
    assert finish(m).success
    assert m._primary_kv_view.tobytes()==b'ccccaaaacccc'
    m.submit_load(job(11,[b'k1',b'new'],[0,1]))
    result=finish(m)
    assert not result.success and result.successful_keys==(b'k1',)
    by=lambda i,kind:[r for r in t.events if r.get('job_id')==i and r['event']==kind]
    assert [r['operation'] for r in by(8,'FS_FILE_OPERATION')]==['exists','open_store','write','publish']*2
    skipped=by(9,'FS_FILE_OPERATION')
    assert len(skipped)==2 and all(r['skipped_existing'] is True for r in skipped)
    reads=[r for r in by(10,'FS_FILE_OPERATION') if r['operation']=='readv']
    assert [r['syscall_bytes'] for r in reads]==[4,4]
    assert [r['host_offset'] for r in reads]==[4,0]
    planned=by(10,'FS_JOB_SLOTS')[0]['slots']
    assert [r['file_id'] for r in reads]==[r['file_id'] for r in planned]
    assert all(r['request_id']=='request-A' for r in reads)
    assert by(11,'FS_FINISHED_JOB')[0]['successful_keys']==[fs.key_id(keys[0])]
    assert by(11,'FS_TASK_END')[0]['status']=='ERROR'
    assert any(r['operation']=='open_load' and r['errno'] for r in by(11,'FS_FILE_OPERATION'))
    assert by(10,'FS_TASK_END')[0]['start_ns'] >= by(10,'FS_ENQUEUE_RETURN')[0]['start_ns']
    assert not t.failures


def test_real_fs_constructor_before_first_job(observed,tmp_path,monkeypatch):
    _,trace,manager,_=observed
    mapper=types.SimpleNamespace(get_config_file_path=lambda:str(tmp_path/'config.json'),get_run_config=lambda:{})
    monkeypatch.setattr(manager,'FileMapper',types.SimpleNamespace(from_offloading_spec=lambda **kw:mapper))
    monkeypatch.setattr(manager,'probe_o_direct',lambda _:False)
    monkeypatch.setattr(manager,'FsAsyncLookupManager',lambda **kw:None)
    spec=types.SimpleNamespace(num_chunks=3,blocks_per_chunk=8,config=types.SimpleNamespace(parallel=types.SimpleNamespace(world_size=8)),
                               replicated_layout=False,cpu_page_size_per_worker=16,kv_bytes_per_chunk=128)
    m=manager.FileSystemTierManager(spec,memoryview(bytearray(384)).cast('B',(3,128)),'fs',str(tmp_path))
    try:
        geometry,=[r for r in trace.events if r['event']=='FS_GEOMETRY']
        assert geometry['host_slot_stride_bytes']==128 and geometry['shape']==[3,128]
        assert not any(r['event']=='FS_JOB_PLAN' for r in trace.events)
        assert not trace.failures
    finally:
        m._pool.shutdown()


def test_async_lookup_keeps_retry_distinct(observed):
    m,t,_,_=observed
    answers=iter([None,False,True])
    m._lookup_manager.lookup=lambda *_:next(answers)
    req=types.SimpleNamespace(req_id='request-A')
    assert [m.lookup(b'k',req).name for _ in range(3)]==['RETRY','MISS','HIT']
    assert [r['lookup_result'] for r in t.events if r['event']=='FS_LOOKUP']==['RETRY','MISS','HIT']


def test_c_branch_retains_batch_only(observed,monkeypatch):
    m,t,_,io=observed
    monkeypatch.setattr(io,'_HAS_FSIO_C',True)
    monkeypatch.setattr(io,'batch_store_block_C',lambda *args:None,raising=False)
    m.submit_store(job(4,[b'k'],[0]))
    assert finish(m).success
    assert any(r.get('data_path')=='FS_IO_C' for r in t.events)
    assert not any(r['event']=='FS_FILE_OPERATION' for r in t.events)


def test_fail_closed_source_and_loaded_identity(observed,monkeypatch,tmp_path):
    _,t,_,io=observed
    other=Installation(t,filesystem=True,legacy=False)
    path=tmp_path/'io.py'
    path.write_bytes((FIX/'io.py').read_bytes()+b'\n')
    with patch.object(io,'__file__',str(path)):
        with pytest.raises(ValueError,match='SOURCE_SHA_MISMATCH'):
            other.patch(io)
    # First installation's rewritten method must not pass as the pinned original.
    with pytest.raises(ValueError,match='LOADED_CODE_MISMATCH'):
        other.patch(io)


def test_fs_startup_is_opt_in(tmp_path,monkeypatch):
    monkeypatch.delenv('AGENTMON_VLLM_030_ENABLE',raising=False)
    monkeypatch.delenv('AGENTMON_VLLM_030_FS_ENABLE',raising=False)
    assert install() is None
    monkeypatch.setenv('AGENTMON_VLLM_030_FS_ENABLE','1')
    monkeypatch.setenv('AGENTMON_VLLM_030_EVENTS',str(tmp_path))
    monkeypatch.setenv('AGENTMON_VLLM_030_RUN_ID','fs-test')
    try:
        assert set(install().targets)==set(fs.METHODS)
    finally:
        uninstall()
