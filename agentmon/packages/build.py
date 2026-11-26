"""Build three non-overlapping installable ZIPs from the maintained package sources."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src/agentmon"
PARTS = ("online", "offline", "experiments")
ONLINE = {"adapters", "contracts", "context_compression", "kv_evidence", "sampling",
          "storage", "trajectory", "__init__.py", "__main__.py", "py.typed", "clock.py",
          "config.py", "doctor.py", "events.py", "paths.py", "security.py", "strategy.py",
          "capabilities.py", "capture_cli.py", "run_controller.py"}
OFFLINE = {"analysis", "reporting", "pipeline", "step_sequence", "report_cli.py", "profile_data.py"}
EXPERIMENTS = {"batch", "control", "experiments", "optimizer", "replay", "runtime",
               "scheduler", "server", "script_compat.py", "cli.py"}
SHARED_READERS = {"analysis/prometheus.py", "reporting/ascend.py",
                  "reporting/event_log.py", "runtime/p2_import.py"}


def owner(path):
    if path.as_posix() in SHARED_READERS:
        return "online"
    if path.name == "runtime_control.py" and path.parts[0] == "adapters":
        return "experiments"
    for name, roots in zip(PARTS, (ONLINE, OFFLINE, EXPERIMENTS)):
        if path.parts[0] in roots:
            return name
    raise ValueError(f"unassigned source file: {path}")


def sources():
    packages = {p: {} for p in PARTS}
    for f in sorted(SRC.rglob("*")):
        if f.is_file() and f.suffix in {".py", ".json", ".yaml", ".toml", ".typed"}:
            rel = f.relative_to(SRC)
            packages[owner(rel)]["src/agentmon/" + rel.as_posix()] = f
    online = packages["online"]
    online["hooks/legacy/sitecustomize.py"] = ROOT / "src/sitecustomize.py"
    online["docs/HOOKS.md"] = ROOT / "docs/HOOKS.md"
    for f in (ROOT / "hooks/v030").iterdir():
        if f.is_file() and f.suffix in {".md", ".py"}:
            online["hooks/v030/" + f.name] = f
    for part in PARTS:
        for f in (ROOT / "packages" / part).glob("*.py"):
            packages[part]["tools/" + f.name] = f
        packages[part]["README.md"] = ROOT / "packages" / part / "README.md"
    return packages


def metadata(part):
    dependencies = [f"agentmon-{p}==0.1.0" for p in PARTS[:PARTS.index(part)]]
    command = {"online": 'agentmon-capture = "agentmon.capture_cli:main"',
               "offline": 'agentmon-report = "agentmon.report_cli:main"',
               "experiments": 'agentmon = "agentmon.cli:main"'}[part]
    return f'''[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "agentmon-{part}"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = {json.dumps(dependencies)}

[project.scripts]
{command}

[tool.setuptools]
py-modules = []

[tool.setuptools.packages.find]
where = ["src"]
include = ["agentmon*"]
namespaces = true

[tool.setuptools.package-data]
"*" = ["*.typed", "*.yaml", "*.toml", "*.json"]
'''.encode()


def build(out):
    out.mkdir(parents=True, exist_ok=True)
    # Refuse to mix newly built packages with another delivery.
    if any(out.iterdir()):
        raise ValueError(f"output directory must be empty: {out}")
    manifest = {}
    for part, files in sources().items():
        name = f"agentmon-{part}.zip"
        dest = out / name
        with ZipFile(dest, "x") as archive:
            for path in sorted([*files, "pyproject.toml"]):
                data = metadata(part) if path == "pyproject.toml" else files[path].read_bytes()
                # All selected inputs are text. Source bindings use upstream LF bytes.
                data = data.replace(b"\r\n", b"\n")
                info = ZipInfo(path)
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data, compress_type=ZIP_DEFLATED, compresslevel=9)
        manifest[name] = {"bytes": dest.stat().st_size, "files": len(files) + 1,
                          "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
    (out / "SHA256SUMS").write_text(
        "".join(f"{record['sha256']}  {name}\n" for name, record in manifest.items()),
        encoding="ascii")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    print(json.dumps(build(parser.parse_args().out), indent=2))
