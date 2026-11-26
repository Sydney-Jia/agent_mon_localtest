"""One-shot startup inspection of configured-off Engine/Worker processes.

No vLLM imports, wrappers, requests or tensor reads. The daemon stops after one
module inspection (or timeout), before formal inference measurements. This
proves the observed startup point, not absence of arbitrary future injections.
"""
import hashlib
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import time
import uuid
from .contracts import KV_FILE_SHA256

FLAG='AGENTMON_OBSERVER_OFF_ATTEST_ENABLE'
DIRECTORY='AGENTMON_OBSERVER_OFF_ATTEST_DIR'
DISABLED_FLAGS=('AGENTMON_VLLM_0271_TRACE_ENABLE','AGENTMON_STATE_PLACEMENT_ENABLE',
    'AGENTMON_VLLM_0271_CONTENT_ENABLE','AGENTMON_VLLM_0271_PREFIX_ENABLE','AGENTMON_VLLM_P2_ENABLE')
_PID=None
_REGISTERED=False


def inspect_loaded(module,role):
    """Inspect only a fully initialized module already imported by vLLM."""
    if module is None or getattr(getattr(module,'__spec__',None),'_initializing',False):return None
    names=('get_computed_blocks','allocate_slots','free') if role=='Engine' else ('load_model','initialize_kv_cache_tensors')
    cls=getattr(module,'KVCacheManager' if role=='Engine' else 'NPUModelRunner',None)
    if cls is None:return None
    methods={}
    for name in names:
        fn=inspect.getattr_static(cls,name,None)
        if not inspect.isfunction(fn):return None
        marked=any(k.startswith('__agentmon') and v for k,v in vars(fn).items())
        methods[name]=dict(agentmon_marker=bool(marked),code_file=fn.__code__.co_filename)
    path=Path(module.__file__);source=path.read_bytes();sha=hashlib.sha256(source).hexdigest()
    code_matches=None
    if role=='Engine':
        compiled=compile(source,str(path),'exec',dont_inherit=True,optimize=sys.flags.optimize)
        code=next(c for c in compiled.co_consts if inspect.iscode(c) and c.co_name=='KVCacheManager')
        code_matches=all(inspect.getattr_static(cls,n).__code__==next(c for c in code.co_consts if inspect.iscode(c) and c.co_name==n) for n in names)
    active=[]
    for name in ('hook','placement'):
        loaded=sys.modules.get('agentmon.adapters.inference.vllm_0271.'+name)
        if loaded is not None and getattr(loaded,'_ACTIVE',None) is not None:active.append(name)
    flags={k:str(os.environ.get(k,'')).lower() in ('1','true','yes','on') for k in DISABLED_FLAGS}
    okay=not active and not any(flags.values()) and not any(r['agentmon_marker'] for r in methods.values())
    if role=='Engine':okay=okay and sha==KV_FILE_SHA256 and code_matches is True
    return dict(status='OFF_RUNTIME_ATTESTED' if okay else 'OFF_RUNTIME_NOT_VERIFIED',process_role=role,
        module_file=str(path),module_sha256=sha,loaded_code_matches_disk=code_matches,
        methods=methods,enabled_flags=flags,active_agentmon_observers=active,
        scope='ONE_STARTUP_OBSERVATION_POINT; NO_FUTURE_INJECTION_GUARANTEE')


def _watch(directory):
    deadline=time.monotonic()+1800
    while time.monotonic()<deadline:
        try:
            try:comm=Path('/proc/self/comm').read_text().strip()
            except OSError:comm=''
            name=multiprocessing.current_process().name;role=None
            if 'EngineCore' in comm+name:role='Engine'
            elif 'Worker' in comm+name:role='Worker'
            module=sys.modules.get('vllm.v1.core.kv_cache_manager' if role=='Engine' else 'vllm_ascend.worker.model_runner_v1') if role else None
            row=inspect_loaded(module,role) if role else None
            if row:
                row.update(pid=os.getpid(),activation_id=uuid.uuid4().hex,utc_ns=time.time_ns(),linux_comm=comm,
                           multiprocessing_name=name,evidence_status='VERIFIED',inference_calls_observed=0)
                directory.mkdir(parents=True,exist_ok=True)
                path=directory/f'off-{os.getpid()}-{row["activation_id"]}.json'
                temporary=path.with_suffix('.tmp')
                temporary.write_text(json.dumps(row,indent=2)+'\n',encoding='utf-8');temporary.replace(path)
                return
        except Exception:
            # Failure is missing attestation, never model startup failure.
            return
        time.sleep(.1)


def start():
    global _PID,_REGISTERED
    if _PID==os.getpid():return
    _PID=os.getpid();directory=Path(os.environ[DIRECTORY])
    if hasattr(os,'register_at_fork') and not _REGISTERED:
        os.register_at_fork(after_in_child=start);_REGISTERED=True
    threading.Thread(target=_watch,args=(directory,),name='agentmon-off-startup-attestation',daemon=True).start()
