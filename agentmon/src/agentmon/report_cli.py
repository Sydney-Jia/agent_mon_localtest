"""Generate reports for a previously recorded run."""
import argparse
from pathlib import Path

from .reporting.report import write_reports


def main(argv=None):
    parser = argparse.ArgumentParser(prog="agentmon-report")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    if not (args.run_dir / "manifest.json").is_file():
        parser.error("expected a run directory containing manifest.json")
    write_reports(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
