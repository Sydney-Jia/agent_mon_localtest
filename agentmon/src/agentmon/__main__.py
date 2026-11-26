from importlib.util import find_spec

if find_spec("agentmon.cli") is None:
    from .capture_cli import main
else:
    from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
