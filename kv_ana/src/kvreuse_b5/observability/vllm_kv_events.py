"""Passive observation of the native vLLM KV event protocol."""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass, field


def decode_frame(frames, rank, topic, unpack):
    """Turn a three-frame vLLM MessagePack publication into a safe JSONL record."""
    if len(frames) != 3 or frames[0] != topic or len(frames[1]) != 8:
        raise ValueError("Unexpected topic or frame shape")
    batch = unpack(frames[2])
    if not isinstance(batch, list) or len(batch) != 3:
        raise ValueError("Expected array-like KVEventBatch")
    timestamp, events, publisher_rank = batch
    if type(publisher_rank) is not int or publisher_rank != rank:
        raise ValueError("Event rank disagrees with the loopback socket")
    if type(timestamp) not in (float, int) or not math.isfinite(timestamp) or not isinstance(events, list):
        raise ValueError("Invalid batch timestamp or events")
    return {"rank": rank, "seq": int.from_bytes(frames[1], "big"), "ts": timestamp,
            "events": [project_event(event) for event in events]}


def project_event(raw):
    if not isinstance(raw, dict):
        raise ValueError("Expected a tagged KV cache event")
    kind = raw.get("type")
    if kind == "AllBlocksCleared":
        return {"type": kind}
    if kind not in ("BlockStored", "BlockRemoved"):
        raise ValueError("Unexpected native KV event type")
    hashes = raw.get("block_hashes")
    if not isinstance(hashes, list) or any(type(value) not in (bytes, int) for value in hashes):
        raise ValueError("Invalid block hash format")
    group = raw.get("group_idx")
    if group is not None and (type(group) is not int or group < 0):
        raise ValueError("Invalid KV cache group index")
    strings = ("medium", "locality", "ownership", "kv_cache_spec_kind")
    if any(raw.get(key) is not None and not isinstance(raw[key], str) for key in strings):
        raise ValueError("Invalid KV cache metadata")
    window = raw.get("kv_cache_spec_sliding_window")
    if window is not None and (type(window) is not int or window <= 0):
        raise ValueError("Invalid KV cache sliding window")
    return {"type": kind, "group_idx": group,
            "hashes": ["hex:" + value.hex() if isinstance(value, bytes) else f"int:{value}" for value in hashes],
            **{key: raw[key] for key in strings if raw.get(key) is not None},
            **({"kv_cache_spec_sliding_window": window} if window is not None else {})}


@dataclass
class Group:
    active: dict = field(default_factory=dict)
    kinds: set = field(default_factory=set)
    sliding_windows: set = field(default_factory=set)
    stored_hash_reports: int = 0
    repeated_store_reports: int = 0
    removed_hash_reports: int = 0
    unmatched_removals: int = 0
    invalidated_by_gaps: int = 0
    ended_by_reset: int = 0
    negative_age_samples: int = 0
    observed_ages_s: list = field(default_factory=list)

    def apply(self, event, timestamp):
        hashes = event["hashes"]
        if event["type"] == "BlockStored":
            self.stored_hash_reports += len(hashes)
            if "kv_cache_spec_kind" in event:
                self.kinds.add(event["kv_cache_spec_kind"])
            if "kv_cache_spec_sliding_window" in event:
                self.sliding_windows.add(event["kv_cache_spec_sliding_window"])
            for block_hash in hashes:
                self.repeated_store_reports += block_hash in self.active
                self.active.setdefault(block_hash, timestamp)
        else:
            self.removed_hash_reports += len(hashes)
            for block_hash in hashes:
                since = self.active.pop(block_hash, None)
                if since is None:
                    self.unmatched_removals += 1
                elif timestamp >= since:
                    self.observed_ages_s.append(timestamp - since)
                else:
                    self.negative_age_samples += 1

    def clear(self, reset=False):
        if reset:
            self.ended_by_reset += len(self.active)
        else:
            self.invalidated_by_gaps += len(self.active)
        self.active.clear()

    def report(self):
        ages = self.observed_ages_s
        return {
            "spec_kinds_observed": sorted(self.kinds), "sliding_windows_observed": sorted(self.sliding_windows),
            "stored_hash_reports": self.stored_hash_reports, "repeated_store_reports": self.repeated_store_reports,
            "removed_hash_reports": self.removed_hash_reports, "unmatched_removals": self.unmatched_removals,
            "ended_by_reset": self.ended_by_reset, "invalidated_by_gaps": self.invalidated_by_gaps,
            "tracked_at_last_observation": len(self.active), "negative_age_samples": self.negative_age_samples,
            "matched_store_to_remove": {"count": len(ages), "min_s": min(ages) if ages else None,
                                         "median_s": statistics.median(ages) if ages else None,
                                         "max_s": max(ages) if ages else None},
        }


@dataclass
class Rank:
    first_seq: int | None = None
    last_seq: int | None = None
    first_ts: float | None = None
    last_ts: float | None = None
    batches: int = 0
    native_events: int = 0
    missing_seq_numbers: int = 0
    nonincreasing_seq_frames: int = 0
    malformed_frames: int = 0
    resets: int = 0
    groups: dict = field(default_factory=dict)

    def apply(self, record):
        if record.get("malformed"):
            self.malformed_frames += 1
            self.clear_groups()
            return
        seq, timestamp, events = record["seq"], record["ts"], record["events"]
        if self.last_seq is not None and seq <= self.last_seq:
            self.nonincreasing_seq_frames += 1
            self.clear_groups()
            return
        expected = 0 if self.last_seq is None else self.last_seq + 1
        if seq > expected:
            self.missing_seq_numbers += seq - expected
            self.clear_groups()
        if self.first_seq is None:
            self.first_seq, self.first_ts = seq, timestamp
        self.last_seq, self.last_ts = seq, timestamp
        self.batches += 1
        self.native_events += len(events)
        for event in events:
            if event["type"] == "AllBlocksCleared":
                self.resets += 1
                self.clear_groups(reset=True)
            else:
                scope = tuple(event.get(key) for key in ("group_idx", "medium", "locality", "ownership"))
                self.groups.setdefault(scope, Group()).apply(event, timestamp)

    def clear_groups(self, reset=False):
        for group in self.groups.values():
            group.clear(reset)

    def report(self):
        if not self.batches:
            coverage = "no_batches_received"
        elif self.missing_seq_numbers or self.nonincreasing_seq_frames or self.malformed_frames:
            coverage = "incomplete_sequence"
        else:
            coverage = "contiguous_through_last_received_batch"
        seen_removals = self.resets or any(group.removed_hash_reports for group in self.groups.values())
        if seen_removals:
            evidence = "removal_or_reset_observed"
        elif coverage == "contiguous_through_last_received_batch" and self.native_events:
            evidence = "no_removal_observed"
        else:
            evidence = "indeterminate_due_to_missing_evidence"
        return {
            "coverage": coverage, "removal_evidence": evidence, "first_seq": self.first_seq,
            "last_seq": self.last_seq, "first_event_ts": self.first_ts, "last_event_ts": self.last_ts,
            "batches": self.batches, "native_events": self.native_events,
            "missing_seq_numbers": self.missing_seq_numbers, "nonincreasing_seq_frames": self.nonincreasing_seq_frames,
            "malformed_frames": self.malformed_frames, "reset_events": self.resets,
            "groups": [dict(zip(("group_idx", "medium", "locality", "ownership"), scope), **group.report())
                       for scope, group in self.groups.items()],
        }


def summarize(records, ranks):
    state = {rank: Rank() for rank in ranks}
    for record in records:
        if type(record.get("rank")) is not int or record["rank"] not in state:
            raise ValueError("Capture file contains an unexpected DP rank")
        state[record["rank"]].apply(record)
    return {"format": "vllm-kv-observation-v1", "protocol_reference_revision": "c4df37",
            "tail_loss_detectable": False, "ranks": {str(rank): data.report() for rank, data in state.items()}}


def write_summary(events, out, ranks):
    with events.open(encoding="utf-8") as source:
        result = summarize((json.loads(line) for line in source if line.strip()), ranks)
    with out.open("x", encoding="utf-8") as destination:
        json.dump(result, destination, indent=2, ensure_ascii=False)
        destination.write("\n")


def capture(out, seconds, base_port, ranks, topic):
    try:
        import msgspec
        import zmq
    except ImportError as error:
        raise RuntimeError("Live capture requires the optional kv-events extra") from error

    out.mkdir(parents=True)
    events_path = out / "events.jsonl"
    topic = topic.encode("utf-8")
    context = zmq.Context()
    poller = zmq.Poller()
    ranks_by_socket = {}
    try:
        for rank in ranks:
            receiver = context.socket(zmq.SUB)
            ranks_by_socket[receiver] = rank
            receiver.setsockopt(zmq.LINGER, 0)
            receiver.setsockopt(zmq.RCVHWM, 100_000)
            receiver.setsockopt(zmq.SUBSCRIBE, topic)
            receiver.bind(f"tcp://127.0.0.1:{base_port + rank}")
            poller.register(receiver, zmq.POLLIN)
        print(f"KV subscriber ready for ranks {ranks} on loopback port base {base_port}", flush=True)
        with events_path.open("x", encoding="utf-8") as events:
            deadline = time.monotonic() + seconds
            while (remaining := deadline - time.monotonic()) > 0:
                for receiver, mask in poller.poll(max(1, round(1000 * remaining))):
                    if not mask & zmq.POLLIN:
                        continue
                    rank = ranks_by_socket[receiver]
                    frames = receiver.recv_multipart()
                    try:
                        record = decode_frame(frames, rank, topic, msgspec.msgpack.decode)
                    except (ValueError, msgspec.DecodeError):
                        record = {"rank": rank, "malformed": True}
                    events.write(json.dumps(record, separators=(",", ":")) + "\n")
                    events.flush()
    except KeyboardInterrupt:
        pass
    finally:
        for receiver in ranks_by_socket:
            receiver.close()
        context.term()
        if events_path.is_file():
            write_summary(events_path, out / "summary.json", ranks)
