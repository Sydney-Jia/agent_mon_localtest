"""Bounded asynchronous metadata writer with live, complete-line watermarks.

An offered sequence is consumed even on overflow. Thus missing critical events
cannot disappear behind renumbering. A seal never requires process shutdown.
"""
from __future__ import annotations
import hashlib
import json
import os
import queue
import threading
import time
from pathlib import Path


class WatermarkWriter:
    def __init__(self, root, identity, *, capacity=4096, rotate_bytes=16*1024*1024):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.identity = dict(identity)
        if not identity.get("writer_epoch") or not identity.get("process_epoch"):
            raise ValueError("WRITER_AND_PROCESS_EPOCH_REQUIRED")
        self.queue = queue.Queue(maxsize=capacity)
        self.rotate_bytes = rotate_bytes
        self.lock = threading.Lock()
        self.offered = self.accepted = self.written = self.dropped = self.errors = 0
        self.written_seq = 0
        self.closed = False
        self.accepting = True
        self.failure = None
        self.shards = []
        self._stop = object()
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def emit(self, event):
        with self.lock:
            if not self.accepting:
                raise ValueError("WRITER_CLOSED_TO_INPUT")
            self.offered += 1
            seq = self.offered
            value = dict(event, producer=self.identity, seq=seq)
            # Serialize on the caller so failures are attributed to this event.
            try:
                raw = (json.dumps(value, sort_keys=True, allow_nan=False)+"\n").encode()
                self.queue.put_nowait((seq, raw))
                self.accepted += 1
                return seq
            except queue.Full:
                self.dropped += 1
                self.failure = {"reason": "QUEUE_OVERFLOW", "seq": seq}
            except (TypeError, ValueError) as exc:
                self.errors += 1
                self.failure = {"reason": "SERIALIZATION_FAILED", "seq": seq, "error": str(exc)}
            return None

    def _write(self):
        stream = None
        try:
            while True:
                item = self.queue.get()
                try:
                    if item is self._stop:
                        break
                    seq, raw = item
                    with self.lock:
                        if stream is None or stream.tell()+len(raw) > self.rotate_bytes:
                            if stream:
                                stream.close()
                            name = "events-%06d.jsonl" % len(self.shards)
                            stream = (self.root/name).open("xb")
                            self.shards.append(dict(path=name, first_seq=seq, last_seq=seq-1, end_offset=0))
                        stream.write(raw)
                        stream.flush()
                        self.shards[-1].update(last_seq=seq, end_offset=stream.tell())
                        self.written += 1
                        self.written_seq = seq
                except OSError as exc:
                    with self.lock:
                        self.errors += 1
                        self.failure = dict(reason="DISK_WRITE_FAILED", seq=item[0], error=str(exc))
                finally:
                    self.queue.task_done()
        finally:
            if stream:
                stream.close()
            with self.lock:
                self.closed = True

    def health(self):
        with self.lock:
            return dict(identity=self.identity, accepted=self.accepted, offered=self.offered,
                        written=self.written, pending=self.accepted-self.written,
                        dropped=self.dropped, errors=self.errors, closed=self.closed,
                        written_seq=self.written_seq, failure=self.failure,
                        heartbeat_monotonic_ns=time.monotonic_ns())

    def seal(self, timeout=30):
        with self.lock:
            target = self.offered
        deadline = time.monotonic()+timeout
        while True:
            h = self.health()
            if h["dropped"] or h["errors"]:
                raise ValueError("WRITER_NOT_LOSSLESS:"+str(h["failure"]))
            if h["written_seq"] >= target:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("WRITER_WATERMARK_TIMEOUT")
            time.sleep(.002)
        with self.lock:
            rows = [dict(r) for r in self.shards if r["first_seq"] <= target]
        # Freeze each range to the target, even if this live producer continued.
        for row in rows:
            raw = (self.root/row["path"]).read_bytes()[:row["end_offset"]]
            lines = raw.splitlines(keepends=True)
            selected = []
            for line in lines:
                if not line.endswith(b"\n"):
                    raise ValueError("PARTIAL_LINE_INSIDE_WATERMARK")
                if json.loads(line)["seq"] <= target:
                    selected.append(line)
            data = b"".join(selected)
            row.update(end_offset=len(data), last_seq=json.loads(selected[-1])["seq"],
                       sha256=hashlib.sha256(data).hexdigest())
        return dict(status="SEALED_TO_WATERMARK", identity=self.identity,
                    target_seq=target, written_seq=h["written_seq"], shards=rows,
                    closed_required=False, health=h)

    def close(self):
        with self.lock:
            self.accepting = False
        self.queue.put(self._stop)
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise TimeoutError("WRITER_CLOSE_TIMEOUT")


def verify_seal(root, receipt):
    expected = 1
    for row in receipt["shards"]:
        path = (Path(root)/row["path"]).resolve()
        if path.parent != Path(root).resolve():
            raise ValueError("SHARD_PATH_OUTSIDE_ROOT")
        with path.open("rb") as f:
            raw = f.read(row["end_offset"])
        if len(raw) != row["end_offset"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("SHARD_MISSING_TRUNCATED_OR_CHANGED")
        for line in raw.splitlines(keepends=True):
            if not line.endswith(b"\n"):
                raise ValueError("PARTIAL_SEALED_LINE")
            value = json.loads(line)
            if value["producer"] != receipt["identity"] or value["seq"] != expected:
                raise ValueError("WRITER_IDENTITY_OR_SEQUENCE_GAP")
            expected += 1
    if expected-1 != receipt["target_seq"]:
        raise ValueError("WATERMARK_NOT_COVERED")
    return {"status": "VERIFIED", "events": expected-1}
