"""Prepare isolated upstream working copies for the common deadline harness."""
import argparse
from pathlib import Path
import re


def prepare_difusco(source):
    for name in ("pl_meta_model.py", "pl_mis_model.py", "models/gnn_encoder.py", "co_datasets/mis_dataset.py"):
        path = source / name
        text = path.read_text(encoding="utf-8")
        if name == "models/gnn_encoder.py":
            text = text.replace("from models.nn import", "from .nn import")
        else:
            text = re.sub(r"(?m)^from (models|utils|co_datasets|pl_meta_model)\b", r"from .\1", text)
        path.write_text(text.replace("import pickle5 as pickle", "import pickle"), encoding="utf-8")


def prepare_dgl(source):
    path = source / "treesearch.py"
    text = path.read_text(encoding="utf-8")
    line = "        for pmap in range(num_prob_maps):\n"
    guard = "            if time.perf_counter() >= getattr(self, 'deadline', float('inf')):\n                break\n"
    if guard not in text:
        if text.count(line) != 1:
            raise ValueError("Expected the upstream per-map loop")
        text = text.replace(line, line + guard)
    partial = "                solutions_to_append.append((result, num_unlabeled))\n"
    observer = "                if hasattr(self, 'record_partial'):\n                    self.record_partial(result)\n"
    if observer not in text:
        if text.count(partial) != 1:
            raise ValueError("Expected the upstream partial-solution queue insertion")
        text = text.replace(partial, observer + partial)
    path.write_text(text, encoding="utf-8")

    for name in ("treesearch.py", "train.py", "utils.py"):
        module = source / name
        text = module.read_text(encoding="utf-8")
        text = re.sub(r"(?m)^from (model|utils|reducelib)\b", r"from .\1", text)
        module.write_text(text, encoding="utf-8")

    path = source / "utils.py"
    text = path.read_text(encoding="utf-8")
    import_line = "from dgl.distributed import shared_mem_utils\n"
    if "\n" + import_line in text:
        helper = "def _copy_graph_to_shared_mem(g, graph_name = None):\n"
        text = text.replace("\n" + import_line, "\n").replace(helper, helper + "    " + import_line)
        path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("difusco", "dgl"))
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    if args.method == "difusco":
        prepare_difusco(args.source)
    else:
        prepare_dgl(args.source)


if __name__ == "__main__":
    main()
