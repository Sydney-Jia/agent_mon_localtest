"""Run package checks in a clean interpreter with only the selected archives."""
import contextvars
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from agentmon.adapters.inference.bench120_hbm import importance_query
from agentmon.kv_evidence.stream import WatermarkWriter, verify_seal

ROOT = Path(__file__).resolve().parents[2]


def test_archives_are_disjoint_and_runnable(tmp_path):
    out = tmp_path / "release"
    subprocess.run([sys.executable, ROOT / "packages/build.py", "--out", out], check=True)
    install = tmp_path / "install"
    seen = set()
    for part in ("online", "offline", "experiments"):
        with ZipFile(out / f"agentmon-{part}.zip") as archive:
            assert archive.testzip() is None
            names = {n for n in archive.namelist() if n.startswith("src/")}
            assert seen.isdisjoint(names)
            seen.update(names)
            for n in names:
                archive.extract(n, install)
        env = {**os.environ, "PYTHONPATH": str(install / "src"), "PYTHONNOUSERSITE": "1"}
        def run(code):
            subprocess.run([sys.executable, "-S", "-c", code], cwd=tmp_path,
                           env=env, check=True, timeout=60)
        if part == "online":
            run("import agentmon.capture_cli, importlib.util as u;"
                "from agentmon.adapters.inference.vllm_030 import hook;"
                "assert u.find_spec('agentmon.cli') is None;"
                "assert u.find_spec('agentmon.reporting.report') is None")
            if sys.platform.startswith("linux"):
                subprocess.run([sys.executable, "-S", "-m", "agentmon", "run", "--output-root",
                                str(tmp_path / "runs"), "--", sys.executable, "-c", "print(42)"],
                               env=env, cwd=tmp_path, check=True, timeout=90)
                manifests = list((tmp_path / "runs").glob("*/manifest.json"))
                assert len(manifests) == 1
                assert (manifests[0].parent / "events.jsonl").stat().st_size > 0
        if part == "offline":
            run("from agentmon.report_cli import main; from pathlib import Path;"
                "from tempfile import TemporaryDirectory; tmp=TemporaryDirectory();p=Path(tmp.name);"
                "(p/'manifest.json').write_text('{}');"
                "assert main([str(p)])==0 and (p/'report.txt').is_file()")
        if part == "experiments" and sys.platform.startswith("linux"):
            run("from agentmon.cli import main; assert main(['--version'])==0")
    expected = {"src/" + f.relative_to(ROOT / "src").as_posix()
                for f in (ROOT / "src/agentmon").rglob("*")
                if f.is_file() and f.suffix in {".py", ".typed", ".yaml", ".toml", ".json"}}
    assert seen == expected


def test_hbm_observation_distinguishes_miss_from_unvisited(monkeypatch):
    trace = []
    stack = contextvars.ContextVar("queries")
    q = dict(query_id="q", request_id="r", probes=0, group_ids=[0, 2, 3])
    pool = SimpleNamespace(_agentmon_research_scope="pool", _agentmon_research_leases={9: 7})
    stack.set((q,))
    monkeypatch.setattr(importance_query, "_research_queries", stack, raising=False)
    monkeypatch.setattr(importance_query, "key_fields",
                        lambda raw: {"key_digest": hashlib.sha256(raw).hexdigest()}, raising=False)
    monkeypatch.setattr(importance_query, "observe",
                        lambda e, **fields: trace.append((e, fields)), raising=False)

    def pool_lookup():
        importance_query.importance_group_result(pool, b"prefix", q["group_ids"],
                                                 [SimpleNamespace(block_id=9)], 2)
    def query(block_pool, _am_research_query):
        # The real hook binds the owning caller frame and active query object.
        pool_lookup()
    query(pool, q)
    event, row = trace[0]
    assert event == "KV_QUERY_GROUP_RESULTS"
    assert [item["result"] for item in row["visited"]] == ["HIT", "MISS"]
    assert [item["allocation_lease"] for item in row["visited"]] == [7, None]
    assert row["first_missing_group"] == 2 and row["unvisited_group_ids"] == [3]


def test_writer_seals_real_files_and_detects_tampering(tmp_path):
    root = tmp_path / "events"
    writer = WatermarkWriter(root, {"writer_epoch": "w", "process_epoch": "p"}, rotate_bytes=60)
    try:
        for i in range(5):
            writer.emit({"ordinal": i})
        seal = writer.seal()
        assert verify_seal(root, seal) == {"status": "VERIFIED", "events": 5}
    finally:
        writer.close()
    shard = root / seal["shards"][0]["path"]
    shard.write_bytes(b"x" + shard.read_bytes()[1:])
    with pytest.raises(ValueError, match="SHARD_MISSING_TRUNCATED_OR_CHANGED"):
        verify_seal(root, seal)


@pytest.mark.parametrize("version", ["0271", "030"])
def test_shared_sink_keeps_adapter_identity(version, tmp_path):
    package = f"agentmon.adapters.inference.vllm_{version}"
    expected = importlib.import_module(package).BUILD_ID
    sink = importlib.import_module(package + ".events").EventSink(tmp_path, "sample-run")
    try:
        sink.emit("TEST_EVENT", process_role="worker", source_function="test")
    finally:
        sink.close()
    rows = [json.loads(line) for line in sink.path.read_text(encoding="utf8").splitlines()]
    health = json.loads(sink.health_path.read_text(encoding="utf8"))
    assert len(rows) == health["written_events"] == health["accepted_events"]
    assert len(rows) >= 2 and health["dropped_events"] == 0
    assert health["hook_build_id"] == expected
    assert all(row["hook_build_id"] == expected and row["run_id"] == "sample-run" for row in rows)
