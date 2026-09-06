from __future__ import annotations

import subprocess
from pathlib import Path


DENIED_PARTS = {"local", "初赛数据集", ".venv", "__pycache__"}
DENIED_NAMES = {"data.local.yaml", "result.csv"}
DENIED_SUFFIXES = {
    ".cbm",
    ".csv",
    ".joblib",
    ".parquet",
    ".pickle",
    ".pkl",
    ".xls",
    ".xlsx",
    ".zip",
}


def main() -> int:
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
    )
    tracked = [Path(raw.decode()) for raw in completed.stdout.split(b"\0") if raw]
    violations: list[str] = []
    for path in tracked:
        synthetic = path.parts[:3] == ("tests", "fixtures", "synthetic")
        if set(path.parts) & DENIED_PARTS:
            violations.append(f"protected path: {path}")
        if path.name in DENIED_NAMES:
            violations.append(f"protected filename: {path}")
        if path.parent == Path("configs") and path.name.endswith(".local.yaml"):
            violations.append(f"protected local config: {path}")
        if path.suffix.lower() in DENIED_SUFFIXES and not synthetic:
            violations.append(f"protected extension: {path}")
        if path.is_file() and path.stat().st_size > 5 * 1024 * 1024:
            violations.append(f"tracked file exceeds 5 MiB: {path}")
    if violations:
        raise SystemExit("private-artifact guard failed:\n" + "\n".join(violations))
    print(f"private-artifact guard PASS ({len(tracked)} tracked files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
