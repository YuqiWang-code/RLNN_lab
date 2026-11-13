"""Merge per-shard evaluation CSVs into the canonical single-shard file.

The evaluation scripts write ``<name>_shard<i>.csv`` when run with
``--shards N``.  Downstream summarisers expect one file, so concatenate the
shards, drop exact duplicate rows (a shard that was restarted can repeat one
graph) and report the graph count.

Usage::

    python models/merge_v8_shards.py \
        --pattern 'outputs/v8_macro/author_test/capture_budget/capture_budget_ref0_shard*.csv' \
        --out outputs/v8_macro/author_test/capture_budget/capture_budget_ref0.csv
"""
import argparse
import csv
import glob
from pathlib import Path

KEY = ("graph_id", "budget", "cell", "search_seed", "repeat")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f"no shard files match {args.pattern}")

    rows, seen, fields = [], set(), None
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = fields or reader.fieldnames
            for row in reader:
                key = tuple(row[name] for name in KEY if name in row)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    rows.sort(key=lambda row: (row.get("budget", ""), row.get("cell", ""), row["graph_id"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    graphs = len({row["graph_id"] for row in rows})
    print(f"{args.out}: {len(paths)} shard files, {len(rows)} rows, {graphs} graphs")


if __name__ == "__main__":
    main()
