"""Offline review of one service-level snapshot; never a request/KV trace."""
from __future__ import annotations

import math
from agentmon.analysis.c1_observations import safe_delta
from agentmon.analysis.prometheus import parse_prometheus_text


COUNTERS = ("prefix_cache_queries_total", "prefix_cache_hits_total",
            "external_prefix_cache_queries_total", "external_prefix_cache_hits_total",
            "num_preemptions_total")
GAUGES = ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc")
WINDOW_COUNTERS = COUNTERS + (
    "prompt_tokens_total",
    "prompt_tokens_cached_total",
    "request_success_total",
    "request_prefill_time_seconds_count",
    "request_prefill_time_seconds_sum",
)


def review_metrics(text: str, *, model: str, engine: str, historical_block_size: int = 128,
                   max_model_len: int = 262144) -> dict:
    """Select exact model/engine series; absent/ambiguous/nonfinite remain unknown.

    Config labels are preserved strings, not interpreted as physical allocation.
    Ratios below are arithmetic only; counter units and windows require source.
    """
    if max_model_len <= 0 or historical_block_size <= 0:
        raise ValueError("reference sizes must be positive")
    parsed = parse_prometheus_text(text)
    samples = [s for s in parsed
               if s.labels.get("model_name") == model and s.labels.get("engine") == engine]
    issues = []

    def one(name):
        candidates = samples
        if name == "cache_config_info":
            # Current exporter omits model_name on config info; preserve engine-only scope.
            candidates = [s for s in parsed if s.labels.get("engine") == engine
                          and s.labels.get("model_name") in (None, model)]
        rows = [s for s in candidates if s.name == "vllm:" + name]
        if len(rows) != 1 or not math.isfinite(rows[0].value) or rows[0].value < 0:
            issues.append({"metric": name, "status": "UNRESOLVED", "matching_series": len(rows)})
            return None
        return rows[0]

    values = {}
    for name in COUNTERS + GAUGES:
        sample = one(name)
        values[name] = sample.value if sample else None
    sample = one("cache_config_info")
    config = dict(sample.labels) if sample and sample.value == 1 else {}
    if sample and sample.value != 1:
        issues.append({"metric": "cache_config_info", "status": "UNRESOLVED", "reason": "info_value_not_one"})

    def positive_integer(key):
        raw = config.get(key, "")
        return int(raw) if raw.isdigit() and int(raw) > 0 else None

    block_size, blocks, capacity = (positive_integer(k) for k in
                                    ("block_size", "num_gpu_blocks", "kv_cache_size_tokens"))
    queries, hits = values[COUNTERS[0]], values[COUNTERS[1]]
    ratio = hits / queries if queries and hits is not None and hits <= queries else None
    product = blocks * block_size if blocks and block_size else None
    return {
        "schema_version": "agentmon-kv-metrics-review-v1",
        "scope": {"model_name": model, "engine": engine},
        "evidence_level": "SERVICE_SNAPSHOT_ONLY", "values": values,
        "cache_config_labels": config, "issues": issues,
        "cache_config_scope": "ENGINE_ONLY" if config and "model_name" not in config else "MODEL_ENGINE" if config else "UNRESOLVED",
        "derived": {"prefix_counter_ratio": ratio, "counter_unit": "UNRESOLVED",
                    "counter_window_and_reset": "UNRESOLVED",
                    "num_gpu_blocks_times_block_size": product,
                    "capacity_matches_simple_product": product == capacity if product and capacity else None,
                    "capacity_div_reference_max_model_len": capacity / max_model_len if capacity else None,
                    "reference_max_model_len": max_model_len,
                    "actual_max_num_seqs": "UNRESOLVED"},
        "block_size_investigation": {"historical": historical_block_size, "metric_label": block_size,
                                     "discrepancy_observed": block_size != historical_block_size if block_size else None,
                                     "resolution": "UNRESOLVED", "physical_layout": "NOT_CAPTURED"},
        "request_binding": "NOT_CAPTURED", "request_kv_events": "NOT_CAPTURED",
        "tensor_capture": "NOT_CAPTURED", "kv_offload_observed": "NOT_PROVEN",
    }


def _by_engine(text: str, model: str) -> dict:
    engines = {}
    for sample in parse_prometheus_text(text):
        labels = sample.labels
        if labels.get("model_name") != model or "engine" not in labels:
            continue
        if not sample.name.startswith("vllm:"):
            continue
        name = sample.name.removeprefix("vllm:")
        extra = set(labels) - {"model_name", "engine"}
        if extra and (extra != {"finished_reason"} or name not in (
            "request_success_total", "request_success_created"
        )):
            continue
        series = engines.setdefault(labels["engine"], {}).setdefault(name, {})
        reason = labels.get("finished_reason")
        value = sample.value
        series[reason] = value if reason not in series and math.isfinite(value) and value >= 0 else None
    return engines


def _counter_window(before: dict, after: dict, name: str) -> dict:
    left, right = before.get(name, {}), after.get(name, {})
    if not left or not right:
        return {"delta": None, "reason": "missing_series"}
    if left.keys() != right.keys():
        return {"delta": None, "reason": "series_changed"}
    if None in left and len(left) > 1:
        return {"delta": None, "reason": "ambiguous_aggregate_and_breakdown"}
    created = name.rsplit("_", 1)[0] + "_created"
    start_epochs, end_epochs = before.get(created, {}), after.get(created, {})
    if start_epochs != end_epochs or None in start_epochs.values():
        return {"delta": None, "reason": "counter_epoch_changed_or_invalid"}
    deltas = [safe_delta(left[key], right[key])["delta"] for key in left]
    if None in deltas:
        return {"delta": None, "reason": "invalid_counter_or_reset"}
    return {"delta": sum(deltas), "reason": None}


def review_window(before: str, after: str, *, model: str, engine: str | None = None) -> dict:
    """Compare snapshots from one serving instance without merging DP replicas."""
    left, right = _by_engine(before, model), _by_engine(after, model)
    selected = [engine] if engine is not None else sorted(left.keys() | right.keys())
    engines = {}
    for label in selected:
        pre, post = left.get(label, {}), right.get(label, {})
        counters = {name: _counter_window(pre, post, name) for name in WINDOW_COUNTERS}
        queries, hits = (counters[name]["delta"] for name in COUNTERS[:2])
        prefix_valid = queries is not None and hits is not None and hits <= queries
        prefill_count = counters["request_prefill_time_seconds_count"]["delta"]
        prefill_sum = counters["request_prefill_time_seconds_sum"]["delta"]
        engines[label] = {
            "counters": counters,
            "gauges": {
                name: {"before": pre.get(name, {}).get(None), "after": post.get(name, {}).get(None)}
                for name in GAUGES
            },
            "prefix_counter_status": "VERIFIED" if prefix_valid else "UNRESOLVED",
            "prefix_counter_ratio": hits / queries if prefix_valid and queries else None,
            "mean_prefill_seconds": prefill_sum / prefill_count if prefill_count and prefill_sum is not None else None,
        }
    complete = bool(engines) and all(row["prefix_counter_status"] == "VERIFIED" for row in engines.values())
    return {
        "schema_version": "agentmon-kv-metrics-window-v1",
        "scope": {"model_name": model},
        "evidence_level": "ENGINE_WINDOW_ONLY",
        "status": "VERIFIED" if complete else "PARTIAL",
        "reset_detection": "snapshot_decrease_and_exported_created_values_only",
        "request_binding": "NOT_CAPTURED",
        "engines": engines,
    }
