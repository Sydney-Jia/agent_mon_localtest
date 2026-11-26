"""Online collection CLI. Engine and Linux probes are loaded on use."""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .config import ConfigError, load_config


def add_capture_parsers(subparsers):
    doctor = subparsers.add_parser("doctor", help="probe environment capabilities")
    add_config_args(doctor)
    doctor.add_argument("--output", help="write doctor JSON to this path")

    run = subparsers.add_parser("run", help="run a command under P0 monitoring")
    add_config_args(run)
    run.add_argument("--label", help="optional run label")
    run.add_argument("--cwd", help="child process working directory")
    run.add_argument("--trajectory", help="explicit mini-SWE trajectory path")
    run.add_argument(
        "--defer-report",
        action="store_true",
        help="finish collection without generating reports; intended for batched post-processing",
    )
    run.add_argument("command_argv", nargs=argparse.REMAINDER)


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="path to agentmon TOML config")
    parser.add_argument(
        "--allow-unknown-config",
        action="store_true",
        help="ignore unknown config keys during migrations",
    )
    parser.add_argument("--output-root", help="override general.output_root")


def _load_cfg(args: argparse.Namespace):
    overrides = {}
    if getattr(args, "output_root", None):
        overrides["general.output_root"] = args.output_root
    return load_config(
        config_path=getattr(args, "config", None),
        allow_unknown_config=getattr(args, "allow_unknown_config", False),
        cli_overrides=overrides,
    )


def doctor_command(args: argparse.Namespace, probe) -> int:
    try:
        cfg = _load_cfg(args)
    except ConfigError as exc:
        print(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "command": "doctor",
                    "status": "misconfigured",
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2

    report, exit_code = probe(cfg, output=args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code


def run_command(args: argparse.Namespace, controller_type) -> int:
    try:
        cfg = _load_cfg(args)
    except ConfigError as exc:
        print(f"agentmon config error: {exc}", file=sys.stderr)
        return 2
    argv = list(args.command_argv)
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print("agentmon run requires a command after --", file=sys.stderr)
        return 64
    controller = controller_type(
        cfg,
        argv,
        label=args.label,
        cwd=args.cwd,
        trajectory=args.trajectory,
        write_final_report=not args.defer_report,
    )
    exit_code = controller.run()
    output = controller.run_dir / ("manifest.json" if args.defer_report else "report.txt")
    field = "manifest" if args.defer_report else "report"
    print(f"agentmon run_id={controller.run_id} {field}={output}", file=sys.stderr)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(prog="agentmon-capture")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    add_capture_parsers(commands)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        from .doctor import run_doctor
        return doctor_command(args, run_doctor)
    from .run_controller import RunController
    args.defer_report = True
    return run_command(args, RunController)


if __name__ == "__main__":
    raise SystemExit(main())
