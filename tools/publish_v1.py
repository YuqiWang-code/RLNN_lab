#!/usr/bin/env python3
"""Publish newly completed MIS experiments from the configured project server."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
NODE = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
WORKBOOK = ROOT / "docs/实验结果.xlsx"


def command(*args, capture=False):
    result = subprocess.run([str(arg) for arg in args], cwd=ROOT, check=True,
                            text=True, encoding="utf-8", capture_output=capture)
    return result.stdout.strip() if capture else None


def result_paths(version, output):
    metadata_files = list(output.glob("*/run.json"))
    if version == "v2":
        metadata_files += list(output.glob("*/*/run.json"))
    for path in sorted(metadata_files):
        if any(part.startswith("smoke_") for part in path.relative_to(output).parts):
            continue
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if metadata["status"] != "completed":
            continue
        yield path.parent.relative_to(ROOT).as_posix()
        log = output / "queue" / (path.parent.name + ".log")
        if log.is_file() and not (path.parent / "train.log").exists():
            yield log.relative_to(ROOT).as_posix()


def publish(version):
    output = ROOT / "outputs" / version
    command(sys.executable, "tools/server.py", "pull", "--version", version)
    git = ("git", "-c", "core.quotepath=false")
    paths = set(result_paths(version, output))
    markers = ("frequency", "comparison") if version == "v1" else ("v2",)
    done = all((output / "queue" / f"{name}.done.txt").is_file() for name in markers)
    if done:
        if version == "v1":
            paths.add("outputs/v1/queue")
        else:
            for path in sorted((output / "queue").glob("*.txt")):
                if not path.name.startswith("smoke_"):
                    paths.add(path.relative_to(ROOT).as_posix())
    if version == "v2":
        for name in ("dataset_manifest.json", "decision.json"):
            if (output / name).is_file():
                paths.add(f"outputs/v2/{name}")
    paths = sorted(paths)
    changed = paths and command(*git, "status", "--porcelain", "--", *paths, capture=True)
    if changed:
        command(NODE, f"tools/collect_{version}.mjs")
        paths.append(WORKBOOK.relative_to(ROOT).as_posix())
        if done or not any((ROOT / "docs/temporary").glob(f"{version}_*.txt")):
            snapshot = command(sys.executable, "tools/snapshot.py", "--version", version, capture=True).splitlines()[-1]
            paths.append(Path(snapshot).as_posix())
        command(*git, "add", "--", *paths)
        command(*git, "commit", "--only", "-m", f"Record completed {version} MIS experiments", "--", *paths)
    if command(*git, "rev-list", "origin/main..HEAD", capture=True):
        command(sys.executable, "tools/server.py", "push")
        command(*git, "push", "origin", "main")
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--watch", action="store_true", help="publish every three minutes until the experiment finishes")
    args = parser.parse_args()
    while True:
        try:
            if publish(args.version) or not args.watch:
                return
        except (OSError, subprocess.CalledProcessError) as error:
            if not args.watch:
                raise
            print(f"Publish will retry: {error}", flush=True)
        time.sleep(180)


if __name__ == "__main__":
    main()
