"""Save the source IDs for the graph-cache rows used by the MIS evaluator."""
import argparse
import json
from pathlib import Path
import pickle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source = Path(manifest["source_directory"])
    graphs = sorted(manifest["graphs"], key=lambda entry: int(Path(entry["source"]).stem.split("_")[-1]))
    with args.output.open("w", encoding="utf-8") as stream:
        for entry in graphs:
            filename = Path(entry["source"])
            with (source / filename).open("rb") as handle:
                graph = pickle.load(handle)
            nodes = [int(node) for node in graph.nodes()]
            if len(nodes) != entry["num_nodes"]:
                raise ValueError(f"{filename} does not match the tensor manifest")
            stream.write(json.dumps(dict(graph_id=filename.stem, original_node_ids=nodes),
                                    separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
