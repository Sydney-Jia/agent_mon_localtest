#!/usr/bin/env python3
"""Capture native vLLM KV events or summarize a saved capture."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kvreuse_b5.observability.vllm_kv_events import capture, write_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listener = commands.add_parser("capture", help="Listen on local loopback for a bounded period")
    listener.add_argument("--out", type=Path, required=True, help="new capture directory")
    listener.add_argument("--seconds", type=int, default=180)
    listener.add_argument("--base-port", type=int, default=5557)
    listener.add_argument("--topic", default="vllm-kv-events")
    offline = commands.add_parser("summarize", help="Summarize projected JSONL without transport dependencies")
    offline.add_argument("--events", type=Path, required=True)
    offline.add_argument("--out", type=Path, required=True, help="new summary file")
    for command in (listener, offline):
        command.add_argument("--ranks", type=int, nargs="+", default=[0])
    args = parser.parse_args(argv)
    if any(rank < 0 for rank in args.ranks) or len(set(args.ranks)) != len(args.ranks):
        parser.error("ranks must be distinct nonnegative integers")
    if args.command == "summarize":
        write_summary(args.events, args.out, args.ranks)
        return
    if args.seconds <= 0 or not 1 <= args.base_port <= 65535 - max(args.ranks):
        parser.error("invalid capture duration or loopback port range")
    try:
        capture(args.out, args.seconds, args.base_port, args.ranks, args.topic)
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
