"""Bounded per-process JSONL sink; no disk wait on the model thread.

At high water the producer yields for 1 ms at most once per 64 events.
This deliberate, measurable overhead prevents CPU bursts starving the writer;
it cannot guarantee losslessness when storage stalls. Drops remain fatal to audit.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import uuid


class HookEventSink:
    build_id: str
    diagnostic_prefix: str

    def __init__(self, directory: Path, run_id: str, *, evidence_status='VERIFIED', capacity=512,
                 max_file_bytes=16*1024*1024):
        self.directory=Path(directory); self.run_id=run_id; self.evidence_status=evidence_status
        self.pid=os.getpid(); self.epoch=uuid.uuid4().hex
        self.path=self.directory/f'hook-{self.pid}-{self.epoch}.jsonl'
        self.health_path=self.directory/f'health-{self.pid}-{self.epoch}.json'
        # 上限限制待写“事件条数”，不是最多能跑多少请求；每个请求会产生多条事件。
        # 事件长度仍受 Block 快照等控制，条数上限本身不是精确的内存字节上限。
        self.queue=queue.Queue(maxsize=capacity); self.max_file_bytes=max_file_bytes
        self.seq=self.accepted=self.written=self.dropped=self.failures=0
        self.observer_failures=0; self.drop_reasons={}; self.queue_high_water=0
        self.cooperative_yields=0
        self.written_bytes=0; self.file_part=0; self.file_bytes=0
        self.lock=threading.Lock(); self.emit_lock=threading.Lock()
        self.closed=False; self.stopping=threading.Event()
        self.thread=threading.Thread(target=self._writer,name='agentmon-kv-jsonl',daemon=True)
        self.thread.start()

    def health(self):
        with self.lock:
            return {'schema_version':'1.0','pid':self.pid,'run_id':self.run_id,
                'engine_epoch':self.epoch,'hook_build_id':self.build_id,
                'accepted_events':self.accepted,'written_events':self.written,
                'dropped_events':self.dropped,'logger_failures':self.failures,
                'observer_failures':self.observer_failures,'last_seq_no':self.seq,
                'evidence_status':self.evidence_status,'drop_reasons':dict(self.drop_reasons),
                'queue_capacity':self.queue.maxsize,'queue_high_water':self.queue_high_water,
                'producer_yield_count':self.cooperative_yields,
                'producer_yield_seconds_per_call':0.001,'producer_yield_event_spacing':64,
                'queued_events':self.queue.qsize(),'written_bytes':self.written_bytes,
                'file_part':self.file_part,'max_file_bytes':self.max_file_bytes}

    def _drop(self, reason, count=1):
        # Caller holds the state lock.
        self.dropped+=count; self.drop_reasons[reason]=self.drop_reasons.get(reason,0)+count

    def failure(self):
        with self.lock:
            self.failures+=1; first=self.failures==1
        if first:
            try: print(self.diagnostic_prefix+'_OBSERVER_FAILURE pid='+str(self.pid)+'; no payload logged',file=sys.stderr)
            except Exception: pass

    def observer_failure(self):
        with self.lock:
            self.observer_failures+=1; first=self.observer_failures==1
        if first:
            try: print(self.diagnostic_prefix+'_METADATA_READ_FAILURE pid='+str(self.pid)+'; no payload logged',file=sys.stderr)
            except Exception: pass

    def emit(self,event,*,process_role,source_function,**fields):
        try:
            # Producer ordering and writer accounting use DIFFERENT locks. Holding
            # the writer's state lock while serializing thousands of Content
            # events starved the consumer and filled the 4096-event queue.
            with self.emit_lock:
                with self.lock:
                    if self.closed:
                        self._drop('SINK_CLOSED'); return False
                # 入队前分配序号：队列满时会留下缺号，可与 dropped 相互核对。
                # 连续的已写行与“未定义观测点也都覆盖”是两件不同的事。
                    self.seq+=1
                    seq=self.seq
                record={'schema_version':'1.0','event':event,'utc_ns':time.time_ns(),
                    'monotonic_ns':time.monotonic_ns(),'pid':self.pid,'process_role':process_role,
                    'engine_epoch':self.epoch,'engine_epoch_kind':'OBSERVER_PROCESS_ACTIVATION',
                    'run_id':self.run_id,'seq_no':seq,'hook_build_id':self.build_id,
                    'request_id':None,'external_request_id':None,'group_id':None,'block_ids':[],
                    'block_event_kind':None,'reuse_kind':None,'ref_count_if_provable':None,
                    'num_new_tokens_if_available':None,'num_computed_tokens_if_available':None,
                    'source_function':source_function,'evidence_status':self.evidence_status,
                    'generation':None,'generation_status':'UNRESOLVED','physical_allocation':None,
                    'physical_allocation_status':'NOT_OBSERVED','reason':{}}
                record.update(fields)
                line=json.dumps(record,ensure_ascii=False,separators=(',',':'),allow_nan=False)
                # A pure-Python producer can monopolize the GIL while the writer
                # needs it to join/encode batches. Yield the CPU once at the
                # high-water boundary; never wait for queue space or disk I/O.
                # This is measured observer overhead, NOT a lossless guarantee:
                # a stalled writer still produces explicit QUEUE_FULL failures.
                if seq % 64 == 0 and self.queue.qsize() >= max(1,self.queue.maxsize//2):
                    with self.lock:self.cooperative_yields+=1
                    time.sleep(0.001)
                try:
                    # 不因队列满等待磁盘写入；仍存在锁和 JSON 序列化开销，并非零成本。
                    self.queue.put_nowait(line)
                    with self.lock:
                        self.accepted+=1
                        self.queue_high_water=max(self.queue_high_water,1,self.queue.qsize())
                    return True
                except queue.Full:
                    # 推理优先：丢弃观测并明确计数。不能静默丢弃后继续标研究验收通过。
                    with self.lock:self._drop('QUEUE_FULL')
                    return False
        except Exception:
            self.failure(); return False

    def _save_health(self):
        temporary=self.health_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.health(),indent=2)+'\n',encoding='utf-8')
        temporary.replace(self.health_path)

    def _writer(self):
        # 当前写入器是模型进程内的后台线程，不是已解耦的独立采集服务。
        # 每次最多聚合 256 条，减少重复写入；生产速度持续超过写入速度仍会积压。
        handle=None; last_health=0.0
        try:
            self.directory.mkdir(parents=True,exist_ok=True)
            handle=self.path.open('x',encoding='utf-8',buffering=65536)
        except Exception: self.failure()
        while not self.stopping.is_set() or not self.queue.empty():
            batch=[]
            try:
                batch.append(self.queue.get(timeout=0.1))
                for _ in range(255):
                    try: batch.append(self.queue.get_nowait())
                    except queue.Empty: break
            except queue.Empty: pass
            if batch:
                try:
                    if handle is None:
                        with self.lock: self._drop('WRITER_UNAVAILABLE',len(batch))
                    else:
                        data='\n'.join(batch)+'\n'; size=len(data.encode('utf-8'))
                        if self.max_file_bytes and self.file_bytes and self.file_bytes+size>self.max_file_bytes:
                            # 达到阈值就开新卷，不删除旧卷；它不是总磁盘容量上限。
                            # 一个完整批次也可能大于阈值，所以是分卷目标而非严格文件限额。
                            handle.close(); handle=None
                            self.file_part+=1; self.file_bytes=0
                            part=self.path.with_name(self.path.stem+f'.part{self.file_part:06d}.jsonl')
                            handle=part.open('x',encoding='utf-8',buffering=65536)
                        handle.write(data); handle.flush()
                        self.file_bytes+=size
                        with self.lock:
                            self.written+=len(batch); self.written_bytes+=size
                except Exception:
                    self.failure()
                    with self.lock: self._drop('WRITE_FAILURE',len(batch))
                finally:
                    for _ in batch: self.queue.task_done()
            if time.monotonic()-last_health>=1 or self.stopping.is_set():
                try: self._save_health()
                except Exception: self.failure()
                last_health=time.monotonic()
        if handle:
            try: handle.close()
            except Exception: self.failure()
        try: self._save_health()
        except Exception: self.failure()

    def close(self):
        # 关闭前尝试排空，但退出等待有界；线程未排空时不能假设所有事件已持久化。
        # benchmark 另查 health/档案行数；这里不以无限等待阻塞服务退出。
        if self.closed: return
        self.emit('HOOK_HEALTH',process_role='observer',source_function='EventSink.close',health=self.health())
        with self.lock: self.closed=True
        self.stopping.set(); self.thread.join(timeout=2)
