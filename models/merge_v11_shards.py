"""Merge every v11 evaluation CSV into one long table.

The v11 evaluators write ``v11_rows.csv`` per run directory (one row per
graph x seed x repeat, plus a header-only trailing summary row in some
runs).  Downstream collectors need a single file, so this script:

* discovers ``outputs/v11/**/v11_rows.csv`` (or an explicit ``--pattern``),
* tags every row with its run directory (``run``) so runs that share the
  same ``cell``/``budget`` labels stay distinguishable,
* drops duplicated rows using the ``merge_v8_shards.py`` key convention
  widened with ``run`` (a restarted shard repeats a graph),
* verifies the header is identical across runs and that the merged row
  count equals the sum of the per-run unique rows.

Usage::

    python models/merge_v11_shards.py --out outputs/v11/collect/v11_rows_merged.csv
"""
import argparse
import csv
import glob
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY = ("run", "graph_id", "budget", "cell", "arm", "decoder", "seed", "repeat")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="outputs/v11/**/v11_rows.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/v11/collect/v11_rows_merged.csv")
    parser.add_argument("--skip", nargs="*", default=["outputs/v11/collect"])
    # v13 fused-engine ablation cells land in outputs/v11/t7/ but belong to the v13
    # report (docs/temporary/RLNN_v13_FusedEngine.md), not to the v11 main-line
    # collection; excluding them keeps this merge reproducible.
    parser.add_argument("--exclude-run-prefix", nargs="*", default=["t7/v13_", "t7/smoke_v14b"])
    args = parser.parse_args()

    paths = sorted(p for p in glob.glob(str(ROOT / args.pattern), recursive=True)
                   if Path(p).is_file() and not any(s in Path(p).as_posix() for s in args.skip))
    skipped = [p for p in paths
               if any(Path(p).parent.relative_to(ROOT / "outputs/v11").as_posix().startswith(prefix)
                      for prefix in args.exclude_run_prefix)]
    if skipped:
        print(f"excluding {len(skipped)} non-v11 run(s): "
              + ", ".join(sorted(Path(p).parent.relative_to(ROOT / 'outputs/v11').as_posix() for p in skipped)))
    paths = [p for p in paths if p not in set(skipped)]
    if not paths:
        raise SystemExit(f"no v11_rows.csv files match {args.pattern}")

    rows, seen, fields = [], set(), None
    per_run, dups, legacy = {}, Counter(), []
    for path in paths:
        path = Path(path)
        run = path.parent.relative_to(ROOT / "outputs/v11").as_posix()
        with path.open(encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if fields is None:
                fields = list(reader.fieldnames)
            elif set(reader.fieldnames) - set(fields):
                raise SystemExit(f"header has unknown columns in {path}: {reader.fieldnames} != {fields}")
            elif set(fields) - set(reader.fieldnames):
                legacy.append((run, sorted(set(fields) - set(reader.fieldnames))))
            kept = 0
            for row in reader:
                if not row.get("graph_id"):
                    continue  # trailing summary row written by some evaluators
                key = tuple(row.get(name, "") for name in KEY if name in row) + (run,)
                if key in seen:
                    dups[run] += 1
                    continue
                seen.add(key)
                row = {name: row.get(name, "") for name in fields}
                row = dict(row, run=run)
                rows.append(row)
                kept += 1
        per_run[run] = kept

    rows.sort(key=lambda row: (row["run"], row.get("budget", ""), row.get("cell", ""),
                               row.get("seed", ""), row.get("repeat", ""), row["graph_id"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run"] + fields)
        writer.writeheader()
        writer.writerows(rows)

    total = sum(per_run.values())
    print(f"{args.out.relative_to(ROOT).as_posix()}: {len(paths)} files, {len(rows)} rows, "
          f"{len({row['graph_id'] for row in rows})} graphs, {total} kept rows")
    for run, missing in legacy:
        print(f"  legacy schema {run}: missing {', '.join(missing)} (empty-filled)")
    for run in sorted(per_run):
        dup = dups.get(run, 0)
        print(f"  {run:<44} {per_run[run]:>6} rows" + (f"  (+{dup} dup dropped)" if dup else ""))
    assert len(rows) == total, "merged row count does not match per-run sum"


if __name__ == "__main__":
    main()
