"""Transport-only KV evidence contract; no backend or analysis imports."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any

SCHEMA = "agentmon.kv-evidence/1"
MODES = ("AUDIT_MIN", "METADATA", "METADATA_TIMING")
UNKNOWN = "NOT_OBSERVED"


class EvidenceError(ValueError):
    def __init__(self, reason: str, *, module: str, stage: str, field=None,
                 identity=None, affected=None, recoverable=False):
        super().__init__(reason)
        self.detail = dict(reason=reason, module=module, stage=stage, field=field,
                           identity=identity, affected=affected, recoverable=recoverable)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def positive_int(value, name, allow_zero=False):
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError("INVALID_INTEGER:" + name)
    return value


@dataclass(frozen=True)
class Producer:
    run_id: str
    container_id: str
    started_at: str
    role: str
    rank: int | None
    pid: int
    process_epoch: str
    writer_epoch: str

    def validate(self):
        if len(self.container_id) != 64 or any(c not in "0123456789abcdef" for c in self.container_id):
            raise ValueError("FULL_CONTAINER_ID_REQUIRED")
        for key in ("run_id", "started_at", "role", "process_epoch", "writer_epoch"):
            if not getattr(self, key):
                raise ValueError("IDENTITY_REQUIRED:" + key)
        positive_int(self.pid, "pid")
        if self.rank is not None:
            positive_int(self.rank, "rank", True)
        return asdict(self)


def version_key(row):
    """Content, allocation and media identity are never interchangeable."""
    fields = ("run_id", "process_epoch", "pool_scope", "medium", "group_id",
              "key_digest", "slot_id", "lease")
    for k in fields:
        if row.get(k) is None:
            raise ValueError("VERSION_IDENTITY_MISSING:" + k)
    positive_int(row["lease"], "lease")
    return tuple(row[k] for k in fields)


def envelope(producer: Producer, seq: int, event: str, data: dict[str, Any],
             monotonic_ns: int, wall_ns: int, mode: str, mode_version: int):
    positive_int(seq, "seq")
    positive_int(mode_version, "mode_version", True)
    if mode not in MODES:
        raise ValueError("UNKNOWN_OBSERVATION_MODE")
    return dict(schema=SCHEMA, producer=producer.validate(), seq=seq, event=event,
                monotonic_ns=monotonic_ns, wall_ns=wall_ns,
                clock_scope=(producer.run_id, producer.pid, producer.process_epoch),
                mode=mode, mode_version=mode_version, data=data)


# This table is also consumed by release documentation and contract tests.
CONTRACTS = {
    "reuse": dict(events=["POOL_CHECKPOINT", "KEY_REGISTER", "REFERENCE", "KEY_REMOVE", "QUERY"],
                  fields=["version", "before_ref", "after_ref", "reason", "query_id", "eligible_ranges"],
                  consumer="reconstruction + competing_risks", failure="STATE_INTERVAL_INEXACT"),
    "bytes": dict(events=["LAYOUT", "DISPATCH", "WORKER_DISPATCH"],
                  fields=["layout_version", "storage", "shape", "stride", "offset", "block_ranges", "batch_id"],
                  consumer="capacity", failure="BYTE_WEIGHT_UNKNOWN"),
    "holding": dict(events=["REFERENCE", "DDR_READY", "KEY_REMOVE", "WINDOW_SEAL"],
                    fields=["version", "monotonic_ns", "clock_scope"],
                    consumer="reconstruction + holding", failure="HOLDING_INTERVAL_CENSORED"),
    "transfer": dict(events=["TRANSFER_PLAN", "TRANSFER_SUBMIT", "TRANSFER_COMPLETE", "MANAGER_ACK", "FORWARD"],
                     fields=["job_id", "transfer_id", "rank", "source", "destination", "descriptors", "batch_id"],
                     consumer="transfer_chain", failure="TRANSFER_CHAIN_INCOMPLETE"),
    "compute": dict(events=["REQUEST", "QUERY", "FORWARD", "DEVICE_INTERVAL"],
                    fields=["request_id", "input_identity", "token_range", "clock_scope", "timing_scope"],
                    consumer="path_costs", failure="COMMON_WORK_COMPUTE_COST_UNKNOWN"),
    "dependency": dict(events=["LAYOUT", "GROUP_MAP", "RESTORE_UNIT"],
                       fields=["scheduler_group", "canonical_group", "resources", "components", "valid_range"],
                       consumer="dependencies", failure="RESTORE_DEPENDENCY_UNPROVEN"),
    "correctness": dict(events=["INSTANCE", "TRANSFER_COMPLETE", "FORWARD", "OUTPUT"],
                        fields=["identity", "code_proof", "source", "destination", "request_id"],
                        consumer="correctness", failure="PARTIAL_CORRECTNESS_ONLY"),
}
