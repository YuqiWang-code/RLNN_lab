#!/usr/bin/env python3
"""Download public PDFs and upstream source snapshots into the project library."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "docs" / "参考文献"
METHODS = ROOT / "reference_methods"
DATA_SUFFIXES = {
    ".ckpt", ".gpickle", ".h5", ".mat", ".npy", ".npz", ".pkl", ".pt", ".pth",
    ".safetensors", ".tfrecord", ".bin", ".onnx", ".pkl.gz", ".pickle", ".pyc", ".so", ".o",
    ".allx", ".tx", ".ally", ".ty", ".graph", ".zip",
}


def download(url):
    with urlopen(Request(url, headers={"User-Agent": "RLNN-Lab/1.0"}), timeout=30) as stream:
        return stream.read()


def paper(row):
    destination = LIBRARY / row["category"] / (row["method"] + ".pdf")
    if destination.is_file():
        return "present"
    errors = []
    for url in filter(None, row["pdf"].split("|")):
        try:
            payload = download(url)
            if not payload.startswith(b"%PDF-"):
                raise ValueError("response is not a PDF")
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged = destination.with_suffix(".part")
            staged.write_bytes(payload)
            staged.replace(destination)
            return url
        except (OSError, ValueError) as error:
            errors.append(str(error))
    return "; ".join(errors) if errors else ""


def code_file(path, size):
    if size > 10 * 1024**2 or "".join(path.suffixes[-2:]).lower() in DATA_SUFFIXES:
        return False
    if path.suffix.lower() in DATA_SUFFIXES or {"__pycache__", "few_shot_data", "Checkpoints"}.intersection(path.parts):
        return False
    return not ("raw" in path.parts and path.suffix.lower() == ".txt")


def source(row):
    destination = METHODS / row["method"]
    destination.mkdir(exist_ok=True)
    repo = row["repository"]
    if not repo:
        return "unavailable"
    pin = destination / "upstream.json"
    tree = destination / "source"
    if pin.exists():
        revision = json.loads(pin.read_text(encoding="utf-8"))["revision"]
        if tree.is_dir():
            return revision
    else:
        revision = subprocess.check_output(
            ["git", "ls-remote", f"https://github.com/{repo}.git", "HEAD"],
            text=True, timeout=30,
        ).split()[0]
    payload = download(f"https://codeload.github.com/{repo}/zip/{revision}")
    staging = destination / ".download"
    staging.mkdir(exist_ok=True)
    with ZipFile(io.BytesIO(payload)) as archive:
        for item in archive.infolist():
            path = Path(*Path(item.filename).parts[1:])
            if item.is_dir() or not code_file(path, item.file_size):
                continue
            target = staging / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as reader, target.open("wb") as writer:
                shutil.copyfileobj(reader, writer)
    staging.replace(tree)
    pin.write_text(json.dumps({"repository": f"https://github.com/{repo}", "revision": revision}, indent=2) + "\n")
    return revision


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=["code", "pdf", "all"], default="all", nargs="?")
    parser.add_argument("--method", action="append", help="fetch only selected entries")
    args = parser.parse_args()
    catalog = rows(METHODS / "catalog.tsv")
    tasks = []
    if args.scope != "pdf":
        tasks.extend((source, row) for row in catalog)
    if args.scope != "code":
        bibliography = LIBRARY / "bibliography.tsv"
        entries = catalog + (rows(bibliography) if bibliography.exists() else [])
        tasks.extend((paper, row) for row in entries if row["pdf"])

    def fetch(task):
        action, row = task
        try:
            result = action(row)
        except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
            result = f"ERROR {error}"
        print(f"{action.__name__}\t{row['method']}\t{result}", flush=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, (task for task in tasks if not args.method or task[1]["method"] in args.method)))


if __name__ == "__main__":
    main()
