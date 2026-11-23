"""Download a released RLD4CO MIS test split and preserve its numbered graph pairs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import pickle
import re
from time import sleep

import numpy as np
import requests
import torch


REVISION = "f273683b3fe6faa217de35ead96fbe0ee4a44a4a"
DATASET = "shengyuf/RLD4CO"
MODEL = "shengyuf/RLNN"
MODEL_REVISION = "0b96a439a11ac90dc69512720084d08784f02bf4"
SPLITS = {"RB_large": (1000, "rlnn_rb_large_mis"), "ER_small": (128, "rlnn_er_mis"),
          "ER_large": (16, "rlnn_er_mis")}


def get(url, stream=False):
    for attempt in range(8):
        try:
            reply = requests.get(url, stream=stream, timeout=(12, 45))
            reply.raise_for_status()
            return reply
        except requests.RequestException:
            if attempt == 7:
                raise
            sleep(min(3 * (attempt + 1), 20))


def save_download(url, path, size):
    if path.is_file() and path.stat().st_size == size:
        return
    for attempt in range(4):
        temporary = path.with_suffix(path.suffix + ".partial")
        try:
            with get(url, stream=True) as response, temporary.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=512 * 1024):
                    stream.write(chunk)
            if temporary.stat().st_size != size:
                raise OSError(f"Incomplete download of {path.name}")
            temporary.replace(path)
            return
        except (requests.RequestException, OSError):
            if attempt == 3:
                raise


def to_tensor(raw, target, idx, remote_name):
    with raw.open("rb") as stream:
        graph = pickle.load(stream)
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError(f"Unexpected graph type in {remote_name}")
    mapping = {node: i for i, node in enumerate(graph)}
    src = np.repeat(np.arange(len(graph), dtype=np.int64), [len(graph[node]) for node in graph])
    dst = np.fromiter((mapping[nbr] for node in graph for nbr in graph[node]),
                      count=len(src), dtype=np.int64)
    pairs = np.stack((src, dst))
    if np.any(src == dst):
        pairs = pairs[:, src != dst].copy()
    edges = torch.from_numpy(pairs)
    torch.save({"edge_index": edges, "num_nodes": len(graph)}, target)
    return {"source": raw.name, "official_filename": remote_name, "tensor": target.name,
            "graph_id": f"idx_{idx}", "num_nodes": len(graph), "undirected_edges": edges.shape[1] // 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=SPLITS)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    family = args.family
    endpoint = args.endpoint.rstrip("/")
    count, policy = SPLITS[family]
    data = Path("models/data")
    published = data / "author_rld4co"
    raw_dir = published / "MIS" / (family + "_test")
    cache_dir = data / "MIS_tensors" / (family + "_author_test")
    weight_dir = published / "checkpoints/erdoes" / policy
    for directory in (raw_dir, cache_dir, weight_dir):
        directory.mkdir(parents=True, exist_ok=True)
    model_files = get(f"{endpoint}/api/models/{MODEL}/tree/{MODEL_REVISION}/erdoes/{policy}").json()
    for model in model_files:
        if model["path"].endswith("/best.pt"):
            # The two ER splits share this checkpoint. Each invocation stages it privately.
            destination = weight_dir / "best.pt"
            if not destination.is_file():
                stage = weight_dir / (family + ".pt")
                save_download(f"{endpoint}/{MODEL}/resolve/{MODEL_REVISION}/{model['path']}", stage, model["size"])
                stage.replace(destination)
            break
    else:
        raise ValueError(f"Released weights missing for {policy}")
    split = "MIS/" + family + "_test"
    files = get(f"{endpoint}/api/datasets/{DATASET}/tree/{REVISION}/{split}?limit=1000").json()
    graph_ids = [int(re.search(r"_(\d+)\.gpickle$", item["path"]).group(1)) for item in files]
    if sorted(graph_ids) != list(range(count)):
        raise ValueError(f"Incomplete release listing for {family}")
    print(f"Fetching {family}: {count} graphs, {sum(item['size'] for item in files)} bytes", flush=True)

    def fetch(i, item):
        raw = raw_dir / f"idx_{i}.gpickle"
        save_download(f"{endpoint}/datasets/{DATASET}/resolve/{REVISION}/{item['path']}", raw, item["size"])
        return i, raw, item

    entries = [None] * count
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(fetch, i, item) for i, item in sorted(zip(graph_ids, files))]
        for completed, job in enumerate(as_completed(jobs), 1):
            i, raw, item = job.result()
            entries[i] = to_tensor(raw, cache_dir / f"graph_{i:06d}.pt", i, Path(item["path"]).name)
            if completed == 1 or completed % 25 == 0 or completed == count:
                print(f"{family} {completed}/{count}", flush=True)
    manifest = {"source_directory": str(raw_dir.resolve()), "repository": DATASET, "revision": REVISION,
                "split": split, "count": count, "checkpoint_repository": MODEL,
                "checkpoint_revision": MODEL_REVISION, "released_checkpoint": str(weight_dir), "graphs": entries}
    (cache_dir / "tensor_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Ready {cache_dir}: {min(g['num_nodes'] for g in entries)}–{max(g['num_nodes'] for g in entries)} vertices", flush=True)


if __name__ == "__main__":
    main()
