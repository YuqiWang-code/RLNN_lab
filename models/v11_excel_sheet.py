"""Append the v11 result cells to ``docs/实验结果.xlsx`` as a new long-format sheet.

The sheet follows the ``Results`` long-table fields
(version, run, task, dataset, split, epoch, step, metric, value, unit, source,
recorded_at, notes) and appends bootstrap / win-tie-loss columns after them, so
the numbers are reproducible from the CSVs alone:

* ``outputs/v11/collect/arm_summary.csv``    -- one row per arm x budget cell
* ``outputs/v11/collect/comparisons.csv``    -- paired / constant-reference deltas

Only this one sheet is touched; every pre-existing sheet is left untouched and
the workbook is re-read with openpyxl afterwards to verify sheet name, row count
and key cell values against the CSV.

Usage::

    python models/v11_excel_sheet.py            # write + verify
    python models/v11_excel_sheet.py --verify   # verify only
"""
import argparse
import csv
import json
import statistics as stats
from datetime import datetime
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "docs/实验结果.xlsx"
COLLECT = ROOT / "outputs/v11/collect"
SHEET = "V11 C4-FIFO主线"

BASE_FIELDS = ["version", "run", "task", "dataset", "split", "epoch", "step", "metric",
               "value", "unit", "source", "recorded_at", "notes"]
EXTRA_FIELDS = ["ci_low", "ci_high", "win", "tie", "loss", "n_graphs", "report_only", "csv_row"]
FIELDS = BASE_FIELDS + EXTRA_FIELDS

TASK_OF = {"A0": "v11-A0", "P0": "v11-P0", "curve": "v11-curve", "ladder": "v11-ladder",
           "C4": "v11-C4", "clock": "v11-A0-clock", "misconfig": "v11-A0-misconfig",
           "legacy": "v11-A0-legacy"}


def task_for(arm):
    for prefix, task in TASK_OF.items():
        if arm.startswith(prefix):
            return task
    return "v11"


def read_csv(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def build_rows():
    stats_payload = json.loads((COLLECT / "paired_stats.json").read_text(encoding="utf-8"))
    arms = read_csv(COLLECT / "arm_summary.csv")
    comps = read_csv(COLLECT / "comparisons.csv")
    stamp = datetime.now().replace(microsecond=0)
    rows = []

    for index, arm in enumerate(arms, start=2):  # row 1 is the header
        notes = json.dumps({
            "budget_s": float(arm["budget"]), "seed": arm["seed"], "n_graphs": int(arm["graphs"]),
            "dtype": arm["dtype"], "dense": arm["dense"], "fixed_steps": arm["fixed_steps"] or None,
            "mean_steps": arm["steps"] or None, "mean_ms_per_step": arm["ms_per_step"] or None,
            "report_only": arm["report_only"] == "True",
            "ci": "graph-level bootstrap 5000 reps, percentile 2.5/97.5",
        }, ensure_ascii=False)
        common = {
            "version": "v11", "run": arm["arm"], "task": task_for(arm["arm"]),
            "dataset": "RB200-300 n=1000", "split": "eval", "epoch": None,
            "step": (int(arm["fixed_steps"]) if arm["arm"].startswith("curve") and arm["fixed_steps"]
                     else None),
            "source": f"outputs/v11/collect/v11_rows_merged.csv (run={arm['run']})",
            "run_dir": arm["run"],
            "recorded_at": stamp, "notes": notes,
            "ci_low": float(arm["search_ci_low"]), "ci_high": float(arm["search_ci_high"]),
            "win": None, "tie": None, "loss": None, "n_graphs": int(arm["graphs"]),
            "report_only": arm["report_only"] == "True", "csv_row": f"arm_summary.csv:{index}",
        }
        rows.append({**common, "metric": "search_size", "value": float(arm["search"]),
                     "unit": "indep_nodes"})
        for field, metric, unit in (("final", "final_size", "indep_nodes"),
                                    ("steps", "steps", "steps"),
                                    ("ms_per_step", "ms_per_step", "ms")):
            if arm[field] in ("", "None"):
                continue
            rows.append({**common, "metric": metric, "value": float(arm[field]), "unit": unit,
                         "ci_low": None, "ci_high": None})

    for index, comp in enumerate(comps, start=2):
        ref = comp["reference"].split(" (paired arm)")[0]
        ref_key = ref.replace("=", "_").replace(" ", "_")
        notes = json.dumps({"kind": comp["kind"], "graphs": int(comp["graphs"]),
                            "reference": comp["reference"],
                            "delta_definition": "arm mean - reference mean over the same graphs "
                                                "(paired) or over the fixed constant (unpaired)",
                            "win": int(comp["win"]), "tie": int(comp["tie"]),
                            "loss": int(comp["loss"])}, ensure_ascii=False)
        rows.append({
            "version": "v11", "run": comp["arm"], "task": task_for(comp["arm"]),
            "dataset": "RB200-300 n=1000" if int(comp["graphs"]) == 1000 else "RB200-300 n=200",
            "split": "eval", "epoch": None, "step": None,
            "metric": f"search_size_delta_vs/{ref_key}", "value": float(comp["delta"]),
            "unit": "indep_nodes",
            "source": f"outputs/v11/collect/comparisons.csv (reference={comp['reference']})",
            "recorded_at": stamp, "notes": notes,
            "ci_low": float(comp["ci_low"]), "ci_high": float(comp["ci_high"]),
            "win": int(comp["win"]), "tie": int(comp["tie"]), "loss": int(comp["loss"]),
            "n_graphs": int(comp["graphs"]), "report_only": False,
            "csv_row": f"comparisons.csv:{index}",
        })
    return rows, stats_payload


def write(rows):
    workbook = openpyxl.load_workbook(WORKBOOK)
    if SHEET in workbook.sheetnames:  # idempotent re-run of the same v11 collection
        del workbook[SHEET]
    sheet = workbook.create_sheet(SHEET)
    header_fill = PatternFill("solid", fgColor="DDEBF7")
    sheet.append(FIELDS)
    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    for row in rows:
        sheet.append([row[field] for field in FIELDS])
    widths = {"run": 34, "task": 20, "dataset": 20, "metric": 34, "source": 52,
              "notes": 60, "csv_row": 22}
    for position, field in enumerate(FIELDS, start=1):
        sheet.column_dimensions[get_column_letter(position)].width = widths.get(field, 12)
    for row in sheet.iter_rows(min_row=2, min_col=10, max_col=10):
        for cell in row:
            cell.number_format = "0.0000"
    sheet.freeze_panes = "B2"
    sheet.auto_filter.ref = f"A1:{get_column_letter(len(FIELDS))}{sheet.max_row}"
    workbook.save(WORKBOOK)
    return len(rows)


def verify(rows, stats_payload):
    """Re-open the saved workbook and check the sheet against the CSV numbers."""
    workbook = openpyxl.load_workbook(WORKBOOK, read_only=True)
    problems = []
    if SHEET not in workbook.sheetnames:
        problems.append(f"sheet {SHEET!r} missing")
        return problems, {}
    sheet = workbook[SHEET]
    data = list(sheet.iter_rows(values_only=True))
    header, body = data[0], data[1:]
    if list(header) != FIELDS:
        problems.append(f"header mismatch: {header}")
    if len(body) != len(rows):
        problems.append(f"row count {len(body)} != expected {len(rows)}")
    if len(workbook.sheetnames) < 34:
        problems.append(f"expected >=34 sheets after append, got {len(workbook.sheetnames)}")

    # spot-check: the primary size of A0-1s and the paired delta vs the 20.0685 ledger
    a0 = next(arm for arm in read_csv(COLLECT / "arm_summary.csv") if arm["arm"] == "A0-1s")
    expected = {
        ("A0-1s", "search_size"): float(a0["search"]),
        ("A0-1s", "final_size"): float(a0["final"]),
        ("A0-1s", "steps"): float(a0["steps"]),
    }
    for (run, metric), want in expected.items():
        hits = [r for r in body if r[1] == run and r[7] == metric]
        if not hits:
            problems.append(f"no row for {run}/{metric}")
        elif abs(float(hits[0][8]) - want) > 1e-9:
            problems.append(f"{run}/{metric} = {hits[0][8]} != {want}")
    hist_rows = [r for r in body if str(r[7]) == "search_size_delta_vs/hist-20.0685"]
    if not hist_rows:
        problems.append("no hist-20.0685 delta rows")
    a0_delta = next((row["delta"] for row in stats_payload["comparisons"]
                     if row["arm"] == "A0-1s" and row["reference"] == "hist-20.0685"), None)
    a0_cell = next((r[8] for r in hist_rows if r[1] == "A0-1s"), None)
    if a0_delta is None or a0_cell is None or abs(float(a0_cell) - a0_delta) > 1e-9:
        problems.append(f"hist delta cell {a0_cell} != {a0_delta}")

    summary = {"sheet": SHEET, "rows": len(body), "sheets_total": len(workbook.sheetnames),
               "sheets": workbook.sheetnames,
               "checks": {
                   "A0-1s/search_size": next(r[8] for r in body if r[1] == "A0-1s" and r[7] == "search_size"),
                   "A0-1s/delta_vs_hist-20.0685": a0_cell,
                   "hist_delta_rows": len(hist_rows),
                   "P0-0.2s-s1017/search_size": next(r[8] for r in body
                                                    if r[1] == "P0-0.2s-s1017" and r[7] == "search_size"),
                   "C4-escape-deploy-h8/search_size": next(r[8] for r in body
                                                           if r[1] == "C4-escape-deploy-h8" and r[7] == "search_size"),
               }}
    return problems, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="verify an already written sheet")
    args = parser.parse_args()
    rows, stats_payload = build_rows()
    written = None if args.verify else write(rows)
    problems, summary = verify(rows, stats_payload)
    if written is not None:
        print(f"appended {written} rows to {SHEET!r} in {WORKBOOK.relative_to(ROOT).as_posix()}")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for problem in problems:
        print(f"FAIL {problem}")
    print("VERIFY PASS" if not problems else f"VERIFY FAIL ({len(problems)})")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
