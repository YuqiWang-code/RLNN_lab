#!/usr/bin/env python3
"""Publish newly completed v1 runs from the configured project server."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
NODE = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
OUT = ROOT / "outputs/v1"
WORKBOOK = ROOT / "docs/实验结果.xlsx"


def command(*args, capture=False):
    result = subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True,
                            text=True, encoding="utf-8", capture_output=capture)
    return result.stdout.strip() if capture else None


def result_paths():
    for path in sorted(OUT.glob("*/run.json")):
        if path.parent.name.startswith("smoke_"):
            continue
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if metadata["status"] != "completed":
            continue
        yield path.parent.relative_to(ROOT).as_posix()
        log = OUT / "queue" / (path.parent.name + ".log")
        if log.is_file() and not (path.parent / "train.log").exists():
            yield log.relative_to(ROOT).as_posix()


def publish():
    command(sys.executable, "tools/server.py", "pull", "--version", "v1")
    git = ("git", "-c", "core.quotepath=false")
    paths = list(result_paths())
    done = all((OUT / "queue" / f"{name}.done.txt").is_file() for name in ("frequency", "comparison"))
    if done:
        paths.append("outputs/v1/queue")
    changed = paths and command(*git, "status", "--porcelain", "--", *paths, capture=True)
    if changed:
        command(NODE, "tools/collect_v1.mjs")
        paths.append(WORKBOOK.relative_to(ROOT).as_posix())
        if done or not any((ROOT / "docs/temporary").glob("v1_*.txt")):
            snapshot = command(sys.executable, "tools/snapshot.py", "--version", "v1", capture=True).splitlines()[-1]
            paths.append(Path(snapshot).as_posix())
        command(*git, "add", "--", *paths)
        command(*git, "commit", "--only", "-m", "Record completed v1 MIS experiments", "--", *paths)
    if command(*git, "rev-list", "origin/main..HEAD", capture=True):
        command(sys.executable, "tools/server.py", "push")
        command(*git, "push", "origin", "main")
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="publish every three minutes until both queues finish")
    args = parser.parse_args()
    while True:
        try:
            if publish() or not args.watch:
                return
        except (OSError, subprocess.CalledProcessError) as error:
            if not args.watch:
                raise
            print(f"Publish will retry: {error}", flush=True)
        time.sleep(180)


if __name__ == "__main__":
    main()
