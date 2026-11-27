from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from agentmon.analysis.kv_metrics_review import review_metrics, review_window
from agentmon.adapters.inference.vllm_018.version import is_supported_vllm_version

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT.parent / "docs/dsv41_metrics_review_20261008/evidence/user_metrics.prom"


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def review(text=None, **kwargs):
    return review_metrics(SNAPSHOT.read_text(encoding="utf-8") if text is None else text,
                          model="deepseek-v41", engine="0", **kwargs)


def test_observed_snapshot_does_not_promote_runtime_capture():
    result = review()
    assert not result["issues"]
    assert result["values"]["prefix_cache_hits_total"] == 256
    assert result["derived"]["prefix_counter_ratio"] == pytest.approx(256 / 781)
    assert result["block_size_investigation"]["resolution"] == "UNRESOLVED"
    assert result["request_kv_events"] == "NOT_CAPTURED"
    assert result["kv_offload_observed"] == "NOT_PROVEN"
    assert result["cache_config_labels"]["kv_offloading_size"] == "None"


def test_capacity_mismatch_is_arithmetic_not_layout_or_bug():
    result = review()
    assert result["derived"]["num_gpu_blocks_times_block_size"] == 430336
    assert result["derived"]["capacity_matches_simple_product"] is False
    assert result["derived"]["capacity_div_reference_max_model_len"] == pytest.approx(4.9641914368)
    assert result["derived"]["actual_max_num_seqs"] == "UNRESOLVED"
    assert result["block_size_investigation"]["physical_layout"] == "NOT_CAPTURED"


def test_exact_scope_no_other_model_or_engine_leak():
    text = SNAPSHOT.read_text(encoding="utf-8")
    other = text.replace('engine="0"', 'engine="1"').replace("781.0", "5000.0")
    assert review(text + other)["derived"]["prefix_counter_ratio"] == pytest.approx(256 / 781)
    assert review(other)["values"]["prefix_cache_queries_total"] is None


@pytest.mark.parametrize("replacement", ["NaN", "+Inf", "-1.0"])
def test_nonfinite_negative_counter_unknown(replacement):
    result = review(SNAPSHOT.read_text(encoding="utf-8").replace("781.0", replacement))
    assert result["derived"]["prefix_counter_ratio"] is None
    assert result["issues"]


def test_duplicate_series_and_zero_denominator_are_not_hit_rate():
    text = SNAPSHOT.read_text(encoding="utf-8")
    result = review(text + text)
    assert result["derived"]["prefix_counter_ratio"] is None
    assert result["cache_config_labels"] == {}
    assert review(text.replace("781.0", "0.0"))["derived"]["prefix_counter_ratio"] is None


def test_equal_block_label_still_requires_physical_validation():
    result = review(historical_block_size=32)
    assert result["block_size_investigation"]["discrepancy_observed"] is False
    assert result["block_size_investigation"]["resolution"] == "UNRESOLVED"


def test_invalid_info_sample_unknown():
    text = SNAPSHOT.read_text(encoding="utf-8").replace('} 1.0', '} 0.0')
    assert review(text)["cache_config_labels"] == {}


def test_offline_cli_preserves_bytes_unknown_time_and_refuses_overwrite(tmp_path):
    module = script("dsv41_metrics_review")
    out = tmp_path / "review"
    args = ["--metrics", str(SNAPSHOT), "--output-dir", str(out), "--provenance", "USER_PROVIDED_OUTPUT"]
    assert module.main(args) == 0
    assert (out / "metrics.prom").read_bytes() == SNAPSHOT.read_bytes()
    assert json.loads((out / "metrics_review.json").read_text())["provenance"]["collected_at_utc"] == "UNKNOWN"
    with pytest.raises(FileExistsError):
        module.main(args)


def window_sample(queries, hits, stop, length, *, engine=0, created=1, model="ds"):
    labels = f'engine="{engine}",model_name="{model}"'
    return "\n".join((
        f"vllm:prefix_cache_queries_total{{{labels}}} {queries}",
        f"vllm:prefix_cache_queries_created{{{labels}}} {created}",
        f"vllm:prefix_cache_hits_total{{{labels}}} {hits}",
        f'vllm:request_success_total{{{labels},finished_reason="stop"}} {stop}',
        f'vllm:request_success_total{{{labels},finished_reason="length"}} {length}',
        f"vllm:request_prefill_time_seconds_count{{{labels}}} {stop + length}",
        f"vllm:request_prefill_time_seconds_sum{{{labels}}} {(stop + length) / 2}",
        f'vllm:num_requests_waiting{{{labels},reason="capacity"}} 20',
        f"vllm:num_requests_waiting{{{labels}}} 0",
        "",
    ))


def test_window_keeps_engines_separate_and_sums_completion_deltas():
    before = window_sample(100, 20, 1, 2) + window_sample(100, 0, 1, 1, engine=1)
    after = window_sample(200, 90, 2, 3) + window_sample(100, 0, 1, 1, engine=1)
    after += window_sample(15, 2, 3, 2, model="unrelated", engine=2)
    result = review_window(before, after, model="ds")
    assert result["status"] == "VERIFIED"
    assert result["request_binding"] == "NOT_CAPTURED"
    active, idle = result["engines"]["0"], result["engines"]["1"]
    assert set(result["engines"]) == {"0", "1"}
    assert active["counters"]["request_success_total"]["delta"] == 2
    assert active["prefix_counter_ratio"] == 0.7
    assert active["mean_prefill_seconds"] == 0.5
    assert idle["prefix_counter_ratio"] is None
    assert active["gauges"]["num_requests_waiting"] == {"before": 0, "after": 0}


def test_window_rejects_reset_even_when_other_completion_series_grows():
    before = window_sample(100, 20, 1, 2)
    after = window_sample(200, 90, 4, 1, created=2)
    result = review_window(before, after, model="ds")
    assert result["status"] == "PARTIAL"
    counters = result["engines"]["0"]["counters"]
    assert counters["prefix_cache_queries_total"]["reason"] == "counter_epoch_changed_or_invalid"
    assert counters["request_success_total"]["reason"] == "invalid_counter_or_reset"


def test_window_handles_duplicates_missing_engines_and_invalid_ratio():
    snapshot = window_sample(100, 20, 1, 2)
    assert review_window(snapshot, snapshot + snapshot, model="ds")["status"] == "PARTIAL"
    assert review_window(snapshot, snapshot, model="ds", engine="1")["status"] == "PARTIAL"
    result = review_window(snapshot, window_sample(110, 40, 2, 2), model="ds")
    assert result["status"] == "PARTIAL"
    assert result["engines"]["0"]["prefix_counter_ratio"] is None


def test_window_cli_copies_both_snapshots_and_accepts_explicit_engine(tmp_path):
    before, after = tmp_path / "pre.prom", tmp_path / "post.prom"
    before.write_text(window_sample(100, 20, 1, 2, engine=1), encoding="utf-8")
    after.write_text(window_sample(200, 90, 2, 2, engine=1), encoding="utf-8")
    out = tmp_path / "review"
    assert script("dsv41_metrics_review").main([
        "--before", str(before), "--metrics", str(after), "--output-dir", str(out),
        "--provenance", "fixture", "--engine", "1", "--model", "ds",
    ]) == 0
    report = json.loads((out / "metrics_window.json").read_text(encoding="utf-8"))
    assert set(report["engines"]) == {"1"}
    assert (out / "before.prom").read_bytes() == before.read_bytes()
    assert (out / "metrics.prom").read_bytes() == after.read_bytes()


def test_source_ast_does_not_import_or_execute_target(tmp_path):
    module = script("dsv41_kv_source_snapshot")
    target = tmp_path / "scheduler.py"
    target.write_text("raise RuntimeError('must never run')\nclass Scheduler:\n def schedule(self):\n  self.manager.allocate_slots(None)\n", encoding="utf-8")
    row = module.read_file(target, "scheduler.py")
    assert row["status"] == "SOURCE_READ"
    assert row["definitions"][1]["name"] == "Scheduler.schedule"
    assert row["lexical_calls"][0]["line"] == 4
    assert row["lexical_calls"][0]["resolution"] == "LEXICAL_CALL_NOT_RUNTIME_DISPATCH"
    assert module.read_file(tmp_path / "missing.py", "missing.py")["status"] == "MISSING"
    target.write_text("invalid Python !!!", encoding="utf-8")
    assert module.read_file(target, "scheduler.py")["status"] == "SOURCE_ERROR"


def test_missing_source_not_fake_chain_and_render_preserves_output(tmp_path):
    module = script("dsv41_kv_source_snapshot")
    result = module.collect(tmp_path / "absent")
    assert all(f["status"] == "MISSING" for f in result["files"])
    assert result["runtime_dispatch"] == "UNVERIFIED"
    assert result["hook_installed"] is False
    out = tmp_path / "inventory"
    module.render(result, out)
    assert "MISSING" in (out / "source_inventory.md").read_text()
    with pytest.raises(FileExistsError):
        module.render(result, out)


def test_old_p2_version_guard_unchanged():
    assert is_supported_vllm_version("0.18.0+local")
    assert not is_supported_vllm_version("0.27.1+empty")
