"""Per-producer evidence checks consumed by the existing longload return."""
from collections import Counter, defaultdict
import json
import hashlib
from pathlib import Path

from importance_pool_validate import QueueCheck
from importance_query_validate import EVENTS as QUERY_EVENTS, QueryCheck

QUEUE_EVENTS = ("IMPORTANCE_QUEUE_CHECKPOINT", "IMPORTANCE_QUEUE_REQUEST", "IMPORTANCE_QUEUE_EDIT")
COST_EVENTS = ("KV_IMPORTANCE_COMPUTE_BATCH", "KV_IMPORTANCE_FORWARD_COST", "RESEARCH_FORWARD_STREAM_TIME")


def hardware_window(experiment, meta):
    root = Path(experiment) / "observation"
    source = root / "HARDWARE_STATUS.json"
    if not source.is_file():
        return {"status": "NO_FINAL_HARDWARE_RECEIPT", "sample_stream_present": (root/"HARDWARE.jsonl").is_file()}
    try:
        raw = source.read_bytes()
        report = json.loads(raw)
        if not meta.get("service_run_id") or report.get("service_run_id") != meta["service_run_id"]:
            raise ValueError("HARDWARE_SERVICE_ID_MISMATCH")
        if report.get("run_id") != Path(experiment).name or report.get("series_path") != "HARDWARE.jsonl":
            raise ValueError("HARDWARE_EXPERIMENT_OR_FILE_MISMATCH")
        series = hashlib.sha256()
        with (root / "HARDWARE.jsonl").open("rb") as f:
            for part in iter(lambda: f.read(1 << 20), b""):
                series.update(part)
        if series.hexdigest() != report.get("series_sha256"):
            raise ValueError("HARDWARE_SERIES_SHA_MISMATCH")
        return dict(status=report["status"], receipt_sha256=hashlib.sha256(raw).hexdigest(),
                    receipt=report, scope="NODE_SAMPLES; PER_REQUEST_ATTRIBUTION_UNAVAILABLE")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return dict(status="UNVERIFIED", reason=str(exc))


def inspect_window(db, populations, contract=None, service_run=None):
    if not contract and not any(n.startswith(("KV_DEMAND_", "IMPORTANCE_QUEUE_", "KV_IMPORTANCE_"))
                                for n in populations):
        return {"status": "LEGACY_OR_NOT_OBSERVED"}
    pools, queries = {}, {}
    pending = defaultdict(Counter)
    sampled = set()
    invalid = Counter()
    gaps = Counter()
    admissions = set()
    first_outputs = set()
    table_sources = set()
    observed_tables = Counter()
    categories = Counter()
    unknown_table_count = 0
    final_decisions = {}
    group_results = {}
    selection_links = Counter()
    names = sorted(QUERY_EVENTS | set(QUEUE_EVENTS + COST_EVENTS) | {"RESEARCH_OBSERVABILITY_GAP",
                   "KV_IMPORTANCE_REQUEST_QUEUED", "KV_IMPORTANCE_FIRST_ACCEPTED",
                   "KV_WORKER_TABLE_EVIDENCE", "KV_WORKER_TABLE_SOURCE", "HBM_DDR_DECISION"})
    marks = ",".join("?" for _ in names)
    rows = db.execute(f"""SELECT e.run_id,e.pid,e.epoch,e.seq,e.name,p.payload_json
        FROM events e JOIN raw_payloads p ON p.event_id=e.id WHERE e.name IN ({marks})
        ORDER BY e.run_id,e.pid,e.epoch,e.seq,e.id""", names)
    try:
        for run, pid, epoch, seq, name, raw in rows:
            row = json.loads(raw)
            if service_run and service_run != run:
                invalid["FOREIGN_SERVICE_RUN"] += 1
                continue
            if run is None or pid is None or epoch is None or seq is None:
                invalid["MISSING_PRODUCER_IDENTITY"] += 1
                continue
            writer = run, pid, epoch
            if name in QUEUE_EVENTS:
                scope = row.get("pool_scope")
                if not scope:
                    invalid["MISSING_POOL_SCOPE"] += 1
                    continue
                key = (*writer, scope)
                if key not in pools:
                    pools[key] = QueueCheck()
                pools[key].consume(row)
            elif name in QUERY_EVENTS:
                if writer not in queries:
                    queries[writer] = QueryCheck()
                queries[writer].consume(row)
                if name == "HBM_ELIGIBILITY_END":
                    group_results[(*writer, row.get("request_id"), row.get("query_id"))] = (
                        row.get("group_ids"), row.get("hit_length"))
            elif name == "HBM_DDR_DECISION" and row.get("phase") == "FINAL_SELECTED":
                final_decisions[(*writer, row.get("request_id"), row.get("decision_id"))] = (
                    row.get("local_tokens"), row.get("external_tokens"))
            elif name == "RESEARCH_OBSERVABILITY_GAP":
                gaps[str(row.get("phase"))] += 1
            elif name == "KV_WORKER_TABLE_SOURCE":
                table_sources.add((*writer, row.get("rank"), row.get("layout_epoch"), row.get("source_id")))
            elif name == "KV_WORKER_TABLE_EVIDENCE":
                binding = (*writer, row.get("rank"), row.get("layout_epoch"), row.get("table_source_id"))
                observed_tables[binding] += 1
                if row.get("raw_cpu_active_count") is None:
                    unknown_table_count += 1
                for category, count in row.get("counts", {}).items():
                    if type(count) is int and count >= 0:
                        categories[category] += count
                    else:
                        invalid["INVALID_WORKER_TABLE_COUNT"] += 1
            elif name in ("KV_IMPORTANCE_REQUEST_QUEUED", "KV_IMPORTANCE_FIRST_ACCEPTED"):
                rid, segment = row.get("request_id"), row.get("segment")
                if type(rid) is not str or not rid or type(segment) is not int:
                    invalid["MISSING_REQUEST_SEGMENT"] += 1
                else:
                    target = admissions if name == "KV_IMPORTANCE_REQUEST_QUEUED" else first_outputs
                    key = (*writer, rid, segment)
                    if key in target:
                        invalid["DUPLICATE_REQUEST_SEGMENT_EVENT"] += 1
                    target.add(key)
            elif name == "KV_IMPORTANCE_COMPUTE_BATCH":
                if row.get("full_device_sample_requested"):
                    sampled.add((run, row.get("batch_id"), row.get("batch_sha256")))
                for request in row.get("requests", []):
                    decision = request.get("selection")
                    if decision is None:
                        selection_links["without_selection"] += 1
                        continue
                    prefix = (*writer, request.get("request_id"))
                    link = (*prefix, decision.get("prefix_decision_id"))
                    matched = (None not in link and link in final_decisions and final_decisions[link]
                               == (decision.get("local"), decision.get("external")))
                    selection_links["matched_final" if matched else "unmatched_final"] += 1
                    for group in decision.get("group_candidates", []):
                        key = (*prefix, group.get("query_id"))
                        match = (None not in key and key in group_results and group_results[key]
                                 == (group.get("group_ids"), group.get("hit_tokens")))
                        selection_links["matched_group" if match else "unmatched_group"] += 1
            elif name in COST_EVENTS:
                key = (*writer, row.get("batch_id"), row.get("batch_sha256"), row.get("rank"),
                       row.get("layout_epoch"), row.get("forward_id"))
                if any(v is None for v in key):
                    invalid["INCOMPLETE_FORWARD_IDENTITY"] += 1
                    continue
                pending[key][name] += 1
        queue = [v.summary() for v in pools.values()]
        query = [v.summary() for v in queries.values()]
    finally:
        for checker in queries.values():
            checker.close()
    ranks = defaultdict(Counter)
    matched = 0
    for (run, pid, epoch, batch, batch_sha, rank, layout, forward), counts in pending.items():
        b = (run, batch, batch_sha)
        if b in sampled and counts == {"KV_IMPORTANCE_FORWARD_COST": 1, "RESEARCH_FORWARD_STREAM_TIME": 1}:
            ranks[b][rank] += 1
            matched += 1
    complete = sum(ranks[b] == dict.fromkeys(range(8), 1) for b in sampled)
    observed = bool(pools or queries or pending or admissions or observed_tables or sampled)
    actors_ok = bool(first_outputs) and first_outputs <= admissions
    status = "NO_IMPORTANCE_EVENTS" if not observed else "PARTIAL"
    if (not invalid and not gaps and pools and queries and actors_ok and sampled and complete == len(sampled)
            and all(r["status"] == "MATCHED_THROUGH_CUT" for r in queue + query)
            and any(r["selected_slots"] for r in queue) and matched == len(pending)
            and selection_links["matched_group"] and selection_links["matched_final"]
            and not selection_links["unmatched_group"] and not selection_links["unmatched_final"]):
        status = "SELECTED_CHECKS_MATCH_THROUGH_CUT"
    return dict(status=status, queues=queue, demand=query, gaps=dict(gaps),
        selected_request_dispatch_rows=dict(selection_links),
        invalid_identities=dict(invalid), compute=dict(scheduled_sampled_batches=len(sampled),
            joined_host_and_stream_observations=matched, observed_worker_forward_identities=len(pending),
            batches_with_one_host_and_stream_pair_per_rank=complete,
            scheduler_first_accepted=len(first_outputs), scheduler_enqueues=len(admissions),
            segments_without_first_accepted=len(admissions - first_outputs),
            first_accepted_without_recorded_enqueue=len(first_outputs - admissions),
            device_timing_scope="SAMPLED_WHOLE_FORWARD_CURRENT_STREAM"),
        worker_tables=dict(sampled_rows=sum(observed_tables.values()),
            rows_without_source=sum(n for k,n in observed_tables.items() if k not in table_sources or None in k),
            rows_without_active_count=unknown_table_count, slot_comparisons=dict(categories),
            scope="HOST_TABLE_DIAGNOSTIC; DEVICE_READS_AND_NUMERICAL_KV_NOT_PROVEN"),
        coverage="BASELINE_AND_WINDOW_DISTINCT_EVENTS_TO_RECORDED_CUT; CHECK_WRITER_WINDOW_AND_SEAL_SEPARATELY",
        modeling_limits={"bounded_swa_replay":"UNAVAILABLE_IN_BENCH120_HBM_PROFILE",
                         "ddr":"DISABLED_IN_HBM_PROFILE", "storage_cost":"REQUIRES_MATCHED_MEASUREMENT",
                         "all_interpreter_or_device_protection":"NOT_OBSERVED_AS_COMPLETE_SET"})
