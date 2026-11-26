"""Optional host-side process; only reads /proc and npu-smi during the real run."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time


ENABLE = 'AGENTMON_IMPORTANCE_HARDWARE'
INTERVAL = 'AGENTMON_IMPORTANCE_HW_SECONDS'
HERE = Path(__file__).resolve().parent
STATUS = 'HARDWARE_STATUS.json'


def options(env):
    flag = env.get(ENABLE, '0')
    if flag not in ('0','1'):
        raise ValueError('INVALID_HARDWARE_ENABLE')
    interval = int(env.get(INTERVAL, '10'))
    if not 5 <= interval <= 300:
        raise ValueError('INVALID_HARDWARE_INTERVAL_5_TO_300')
    return flag == '1', interval


def systemd_env():
    # systemd --user does not forward arbitrary shell environment variables.
    # Two known settings only. Malformed config is recorded by the optional
    # sampler, independent of the workload's approval and runtime budget.
    return ['--setenv=' + k + '=' + (v if re.fullmatch(r'[0-9]{1,3}', v) else 'INVALID')
            for k in (ENABLE, INTERVAL) if (v := os.environ.get(k)) is not None]


def begin(out, identity):
    try:
        enabled, interval = options(os.environ)
        if not enabled:
            return None, dict(status='DISABLED')
        if (not identity.get('run_id') or not identity.get('service_run_id')
                or not re.fullmatch(r'[0-9a-f]{64}', identity.get('container_id') or '')
                or not identity.get('engine_incarnation')):
            raise ValueError('SAMPLER_IDENTITY_INCOMPLETE')
        cmd=[sys.executable,'-I','-B',str(HERE/'importance_hardware.py'),'--out',str(out),
             '--interval',str(interval),'--identity',json.dumps(identity,separators=(',',':'))]
        with (out/'hardware_sampler.log').open('xb') as log:
            proc=subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
        return proc, dict(status='STARTED',pid=proc.pid)
    except Exception as exc:
        return None, dict(status='UNAVAILABLE', reason=type(exc).__name__+':'+str(exc))


def end(handle,out):
    proc, result = handle
    if proc is None:
        return result
    try:
        proc.stdin.close()
        try:
            rc=proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
            return dict(status='SAMPLER_STOP_TIMEOUT', pid=proc.pid)
        if rc != 0:
            return dict(status='SAMPLER_FAILED',exit_code=rc)
        report=json.loads((out/STATUS).read_bytes())
        report['status_path']=str(out/STATUS)
        return report
    except Exception as exc:
        return dict(status='SAMPLER_FAILED',reason=type(exc).__name__+':'+str(exc))


def hardware_targets(stdout):
    """Bind card and chip only to actual inventory rows with a PCI address."""
    active = None
    targets = []
    for line in stdout.splitlines():
        cells = [s.strip() for s in line.strip().strip('|').split('|')]
        if len(cells)<2:
            continue
        top = re.fullmatch(r'(\d+)\s+(\S+)', cells[0])
        if top and cells[1].lower() in ('ok','warning','alarm','critical','unknown','fault'):
            active=top[1]
            continue
        chip = re.fullmatch(r'(\d+)(?:\s+(?:\d+|-))?',cells[0])
        if chip and active is not None and re.fullmatch(r'(?:[0-9a-fA-F]{4}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d+',cells[1]):
            targets.append((active,chip[1],cells[1]))
    return list(dict.fromkeys(targets))


def sample_host(procroot):
    from agentmon.sampling.host import parse_pressure,read_diskstats,read_key_value_kb,read_text
    start=time.monotonic_ns()
    all_mem=read_key_value_kb(str(procroot/'meminfo'))
    mem={k: all_mem[k] for k in ('MemTotal','MemFree','MemAvailable','Cached','SwapFree','SwapTotal','SReclaimable','Mlocked','Unevictable') if k in all_mem}
    pressure={k:parse_pressure(read_text(str(procroot/'pressure'/k))) for k in ('cpu','io','memory')}
    disks=read_diskstats(str(procroot/'diskstats'))
    return dict(start_monotonic_ns=start,end_monotonic_ns=time.monotonic_ns(),
                meminfo_bytes=mem, pressure=pressure,
                diskstats={k:v for k,v in disks.items() if not k.startswith(('loop','ram'))},
                missing_sources=[name for name, data in [('meminfo',mem),('diskstats',disks),
                    *[('pressure/'+k,v) for k,v in pressure.items()]] if not data])


def verified_device_metrics(raw):
    from agentmon.sampling.ascend_smi import normalize_key,parse_hbm_bandwidth
    values={}
    for line in raw.splitlines():
        key,sep,value=line.partition(':')
        if not sep:
            continue
        field=normalize_key(key)
        if field=='hbm_bandwidth_bytes_per_s':
            measured=parse_hbm_bandwidth(key,value.strip())
            if measured:
                values[measured[0]]=measured[1]
            continue
        unit_pattern={'hbm_total_bytes':r'(?:Ki|Mi|Gi|K|M|G)?B',
            'hbm_used_bytes':r'(?:Ki|Mi|Gi|K|M|G)?B','aicore_util_pct':'%',
            'aicpu_util_pct':'%','temperature_c':r'°?C','power_w':'W'}.get(field)
        if not unit_pattern:
            continue
        number=re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*('+unit_pattern+r')?\s*',value)
        if not number:
            continue
        units=set(re.findall(r'(?<![\w])('+unit_pattern+r')(?![\w])',key))
        if number[2]:
            units.add(number[2])
        if len(units)!=1:
            continue
        unit=units.pop()
        amount=float(number[1])
        if field.endswith('_bytes'):
            binary_powers={'B':1,'KiB':1024,'MiB':1024**2,'GiB':1024**3}
            decimal_powers={'KB':1000,'MB':1000**2,'GB':1000**3}
            amount=int(amount*{**binary_powers,**decimal_powers}[unit])
        if not field.endswith('_pct') or amount<=100:
            values[field]=amount
    return values


def sample_npu(command,seq,raw_file,stop, invoke=None):
    from agentmon.sampling.ascend_smi import run_npu_smi
    if not command:
        return dict(status='NPU_SMI_NOT_FOUND',inventory=[],detail=None)
    invoke = invoke or run_npu_smi

    def read(args):
        start=time.monotonic_ns()
        result=invoke([command,*args],timeout=2)
        finish=time.monotonic_ns()
        raw=result['stdout']
        offset=raw_file.tell()
        line=json.dumps(dict(seq=seq,argv=[command,*args],start_monotonic_ns=start,
            end_monotonic_ns=finish,stdout=raw[:131072],stdout_truncated=len(raw)>131072,
            stderr=result.get('stderr_excerpt',''),status=result['status'],
            returncode=result['returncode']),ensure_ascii=True).encode()+b'\n'
        raw_file.write(line)
        return dict(status=result['status'],returncode=result['returncode'],raw_offset=offset,
                    raw_line_bytes=len(line),raw_line_sha256=hashlib.sha256(line).hexdigest(),
                    start_monotonic_ns=start,end_monotonic_ns=finish),raw

    result,raw=read(['info'])
    targets=hardware_targets(raw) if result['status']=='supported' else []
    sample=dict(status=result['status'] if targets else 'INVENTORY_UNAVAILABLE',inventory=targets,
                inventory_query=result,detail=None)
    if not targets or stop.is_set():
        if targets:
            sample['status']='STOPPED_BEFORE_DETAIL'
        return sample
    card,chip,bus=targets[(seq-1)%len(targets)]
    detail,raw=read(['info','-t','usages','-i',card,'-c',chip])
    detail.update(card_id=card,chip_id=chip,pci_bus=bus,metrics={})
    if detail['status']=='supported':
        detail['metrics']=verified_device_metrics(raw)
        if not detail['metrics']:
            detail['status']='NO_VERIFIED_METRICS'
    sample['detail']=detail
    if detail['status']!='supported':
        sample['status']='DETAIL_UNAVAILABLE'
    return sample


def collect(out,identity,interval,stop,procroot=Path('/proc'),lookup=shutil.which,invoke=None):
    samples=0
    errors=Counter()
    devices={}
    inventory=set()
    started=time.time_ns()
    stream_hash=hashlib.sha256()
    command=lookup('npu-smi')
    with (out/'HARDWARE.jsonl').open('xb') as data,(out/'HARDWARE_RAW.jsonl').open('xb') as raw:
        while not stop.is_set():
            samples+=1
            t=time.monotonic_ns()
            rec=dict(seq=samples,utc_ns=time.time_ns(),start_monotonic_ns=t)
            try:
                rec['host']=sample_host(procroot)
                errors.update(rec['host']['missing_sources'])
            except Exception as exc:
                errors['host_exception']+=1
                rec['host_error']=type(exc).__name__+':'+str(exc)
            try:
                npu=rec['npu']=sample_npu(command,samples,raw,stop,invoke)
                inventory.update(tuple(row) for row in npu['inventory'])
                if npu['status']!='supported':
                    errors[npu['status']]+=1
                detail=npu.get('detail')
                if detail and detail['metrics']:
                    key=detail['card_id']+'/'+detail['chip_id']+'/'+detail['pci_bus']
                    counts=devices.setdefault(key,Counter())
                    counts.update(detail['metrics'].keys())
            except Exception as exc:
                errors['npu_exception']+=1
                rec['npu_error']=type(exc).__name__+':'+str(exc)
            rec['end_monotonic_ns']=time.monotonic_ns()
            encoded=json.dumps(rec,ensure_ascii=True,separators=(',',':')).encode()+b'\n'
            data.write(encoded)
            data.flush()
            stream_hash.update(encoded)
            raw.flush()
            stop.wait(max(0,interval-(time.monotonic_ns()-t)/1e9))
    missing_detail=sorted('/'.join(target) for target in inventory if '/'.join(target) not in devices)
    summary=dict(status='NO_SAMPLES' if not samples else 'PARTIAL' if errors or missing_detail else 'RECORDED',
        run_id=identity['run_id'],service_run_id=identity['service_run_id'],identity=identity,
        scope='NODE_HOST_BLOCK_DEVICES_AND_NPU_CHIPS',pid=os.getpid(),
        status_path=str(out/STATUS), inventory_targets=sorted(inventory), chips_without_detail=missing_detail,
        hostname=os.uname().nodename if hasattr(os,'uname') else os.environ.get('COMPUTERNAME'),
        interval_seconds=interval, samples=samples, coverage_errors=dict(errors),
        start_utc_ns=started,end_utc_ns=time.time_ns(),per_chip_field_sample_counts=devices,
        clock='HOST_MONOTONIC_NS_AND_WALL_UTC_NS; COMMAND START_END_BOUND EACH SAMPLE',
        npu_detail_sampling='ONE_IDENTIFIED_CHIP_ROTATING_PER_INTERVAL',
        diskstats_units='CUMULATIVE_PER_DEVICE; SECTORS_512_BYTES; TIMES_MS; IN_FLIGHT_DEPTH_GAUGE',
        memory_units='BYTES',psi_units='AVG10_60_300_PERCENT_AND_TOTAL_MICROSECONDS',
        raw_path='HARDWARE_RAW.jsonl', series_path='HARDWARE.jsonl',series_sha256=stream_hash.hexdigest())
    with (out/STATUS).open('x',encoding='utf-8') as f:
        json.dump(summary,f,indent=2)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--identity',required=True)
    parser.add_argument('--interval',type=int,required=True)
    args=parser.parse_args()
    if not 5<=args.interval<=300:
        parser.error('sampling interval must be 5..300 seconds')
    sys.path.insert(0,str(HERE/'_hardware_core'))
    stop=threading.Event()
    def watch_parent():
        sys.stdin.buffer.read(1)
        stop.set()
    threading.Thread(target=watch_parent,daemon=True).start()
    collect(args.out,json.loads(args.identity),args.interval,stop)


if __name__=='__main__':
    main()
