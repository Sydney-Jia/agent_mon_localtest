#!/usr/bin/env python3
"""Review one Prometheus snapshot or compare two snapshots offline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agentmon.analysis.kv_metrics_review import review_metrics, review_window


def _read_snapshot(path):
    with path.open("rb") as stream:
        return stream.read(16 * 1024 * 1024 + 1)


def main(argv=None):
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--before", type=Path, help="earlier snapshot from the same service")
    p.add_argument("--output-dir", type=Path, required=True, help="must not exist")
    p.add_argument("--model", default="deepseek-v41")
    p.add_argument("--engine", help="an engine label; default is 0 for one snapshot, all for a comparison")
    p.add_argument("--provenance", required=True, help="source description, not a runtime PASS assertion")
    p.add_argument("--collected-at-utc", default="UNKNOWN")
    p.add_argument("--historical-block-size", type=int, default=128)
    p.add_argument("--reference-max-model-len", type=int, default=262144)
    args = p.parse_args(argv)
    raw = _read_snapshot(args.metrics)
    before = _read_snapshot(args.before) if args.before is not None else None
    if len(raw) > 16 * 1024 * 1024 or (before is not None and len(before) > 16 * 1024 * 1024):
        p.error("snapshot exceeds 16 MiB")
    if before is not None:
        report = review_window(
            before.decode("utf-8"), raw.decode("utf-8"), model=args.model, engine=args.engine
        )
        report_name = "metrics_window.json"
        exit_code = 0 if report["status"] == "VERIFIED" else 2
    else:
        report = review_metrics(raw.decode("utf-8"), model=args.model,
                                engine="0" if args.engine is None else args.engine,
                                historical_block_size=args.historical_block_size,
                                max_model_len=args.reference_max_model_len)
        report_name = "metrics_review.json"
        exit_code = 2 if report["issues"] else 0
    report["provenance"] = {"source": args.provenance, "input": str(args.metrics),
                            "collected_at_utc": args.collected_at_utc,
                            "network_access": False, "model_executed": False}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "metrics.prom").write_bytes(raw)
    if before is not None:
        report["provenance"]["before_input"] = str(args.before)
        (args.output_dir / "before.prom").write_bytes(before)
    (args.output_dir / report_name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("OFFLINE_REVIEW_COMPLETE; REQUEST_KV_TRACE=NOT_CAPTURED; BLOCK_SIZE=UNRESOLVED")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
