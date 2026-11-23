"""Collect full-cohort, 200 ms MIS scores for the four-size paper table."""

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
from compare_mis_runs import compare


ROOT = Path(__file__).resolve().parents[1]
V11 = ROOT / "outputs/v11"
PRIMARY = V11 / "intel_ppo_deadline_20261120/comparison.csv"
TUNED_ONLINEMIS = ROOT / "outputs/v15/priority_mis_20261121/online_mis_tuned/test"
RESIDENT_RUNS = ROOT / "outputs/v16/anytime_author"
ER_LARGE_RESIDENT = ROOT / "outputs/v16/er_large_200ms"
ADDITIONAL_RUNS = ROOT / "outputs/v17/four_scale_comparators"
ADDITIONAL_METHODS = ("gurobi", "dimes")
APPENDIX_ONLY = ("online_mis", "author_rlnn_matched")
SPLITS = {"RB_small": 1000, "RB_large": 1000, "ER_small": 128, "ER_large": 16}
SEEDS = (1017, 1019, 1021)
METHODS = {
    "author_rlnn": "Released RLNN",
    "fld_eg": "FLD-EG (paper reconstruction)",
    "diffuco": "DiffUCO (released weights)",
    "ltft": "LTFT (RB retrained)",
    "difusco": "DIFUSCO (SATLIB transfer)",
    "dgl_treesearch": "DGL-TreeSearch (RB retrained)",
    "intel": "INTEL (released weights, GPU port)",
    "ppo_lwd": "PPO-LwD (RB retrained)",
    "isco_full_window": "DISCS iSCO (full window)",
    "rlsa": "RLSA",
    "isco": "DISCS iSCO (300 steps)",
    "fused": "Ours",
    "online_mis": "KaMIS OnlineMIS",
    "redumis": "KaMIS ReduMIS",
    "author_rlnn_matched": "Released RLNN (distribution-specific weights)",
}

LABELS = {
    "gurobi": "Gurobi (vertex MIP)",
    "redumis": "KaMIS (ReduMIS CLI)",
    "online_mis_tuned": "OnlineMIS (tuned CLI)",
    "online_mis_resident": "OnlineMIS (resident)",
    "author_rlnn": "Released RLNN", "fld_eg": "FLD-EG (recon.)",
    "diffuco": "DiffUCO", "ltft": "LTFT", "difusco": "DIFUSCO",
    "dgl_treesearch": "DGL-TreeSearch", "dimes": "DIMES (ER retrain)", "intel": "INTEL", "ppo_lwd": "PPO-LwD",
    "isco_full_window": "DISCS iSCO", "rlsa": "RLSA", "isco": "iSCO (300 steps)",
    "priority_degree": "Degree-only search",
    "priority_fused": r"\textbf{Fused RLNN (Ours)}",
}

SMALL_RB_DIRECTORIES = {
    "redumis": "reference_deadline_20261119/kamis",
    "online_mis": "reference_deadline_20261119/kamis_online",
    "author_rlnn": "reference_deadline_20261119_followup/author_rlnn",
    "author_rlnn_matched": "reference_deadline_20261119_followup/author_rlnn",
    "fld_eg": "reference_deadline_20261119_followup/fld_eg",
    "diffuco": "learned_deadline_20261119/diffuco",
    "ltft": "learned_deadline_20261119/ltft",
    "difusco": "supervised_deadline_20261119/difusco",
    "dgl_treesearch": "supervised_deadline_20261119/dgl_treesearch",
    "intel": "intel_ppo_deadline_20261120/intel",
    "ppo_lwd": "intel_ppo_deadline_20261120/ppo_lwd",
    "isco_full_window": "reference_deadline_20261119/isco_full_window",
    "rlsa": "reference_deadline_20261119/rlsa",
    "isco": "reference_deadline_20261119/isco",
}


def paper_cells(rows):
    cells = {(r["split"], r["method"]): r for r in rows if r["method"] in LABELS}
    for split, count in SPLITS.items():
        resident = ER_LARGE_RESIDENT if split == "ER_large" else RESIDENT_RUNS / split / "200"
        directories = {"priority_fused": resident / "learned", "priority_degree": resident / "degree",
                       "online_mis_resident": resident / "online_mis", "online_mis_tuned": TUNED_ONLINEMIS / split}
        directories.update({method: ADDITIONAL_RUNS / split / method for method in ADDITIONAL_METHODS})
        for method, directory in directories.items():
            source = directory / "per_graph.csv"
            diagnostic = directory / "status.json"
            if method in ADDITIONAL_METHODS and diagnostic.is_file():
                record = json.loads(diagnostic.read_text(encoding="utf-8"))
                cells[split, method] = dict(split=split, method=method, status=record["status"],
                                            reason=record["reason"], source=diagnostic.relative_to(ROOT).as_posix())
            elif source.is_file():
                cells[split, method] = dict(split=split, method=method, **summarize(source, count),
                                            source=source.relative_to(ROOT).as_posix())
    return cells


def write_table(rows, status, target, csv_target=None):
    rows = list(rows)
    cells = paper_cells(rows)
    if csv_target is not None:
        appendix = {(r["split"], r["method"]): r for r in rows if r["method"] in APPENDIX_ONLY}
        all_cells = {**cells, **appendix}
        methods = [*LABELS, *APPENDIX_ONLY]
        missing = [(split, method) for method in methods for split in SPLITS if (split, method) not in all_cells]
        if missing:
            raise ValueError(f"Missing 200 ms cohorts: {missing}")
        with csv_target.open("w", encoding="utf-8", newline="") as stream:
            fields = ("table_scope", "method", "split", "graphs", "budget_ms", "status", "runs",
                      "mean_size", "mean_wall_ms", "on_time", "source", "reason")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for method in methods:
                scope = "both" if method == "redumis" else ("appendix" if method in APPENDIX_ONLY else "main")
                for split, count in SPLITS.items():
                    writer.writerow({"status": "measured", **all_cells[split, method], "graphs": count,
                                     "budget_ms": 200, "table_scope": scope})
    if target is None:
        return
    best_size = dict.fromkeys(SPLITS, 0.0)
    for (split, method), row in cells.items():
        if "mean_size" in row:
            best_size[split] = max(best_size[split], float(row["mean_size"]))
    lines = [
        r"\begin{table*}[t]", r"\centering\footnotesize\setlength{\tabcolsep}{2.6pt}",
        r"\caption{Maximum independent set at 200\,ms per graph. Size averages the largest feasible set delivered to the host by the cutoff over all $N$ graph--seed pairs. A missed delivery scores zero. Ready counts timely deliveries. Time is the mean wall time per call in milliseconds, including overruns. DiffUCO skips inference on large RB and large ER after calibration. An em dash marks an unavailable measurement (Gurobi license details in Appendix~\ref{app:protocol}). Bold marks the highest mean size in each range.}",
        r"\label{tab:external-deadline}", r"\resizebox{\textwidth}{!}{\begin{tabular}{lrrrrrrrrrrrr}",
        r"\toprule",
        r"\paperHeader & \multicolumn{3}{c}{\paperHead{RB-[200--300]}} & \multicolumn{3}{c}{\paperHead{RB-[800--1200]}} & \multicolumn{3}{c}{\paperHead{ER-[700--800]}} & \multicolumn{3}{c}{\paperHead{ER-[9000--11000]}} \\",
        r"\paperHeader & \multicolumn{3}{c}{$N=3000$} & \multicolumn{3}{c}{$N=3000$} & \multicolumn{3}{c}{$N=384$} & \multicolumn{3}{c}{$N=48$} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
        r"\paperHeader \paperHead{Solver}" + r" & \paperHead{Size\,$\uparrow$} & \paperHead{Time\,$\downarrow$} & \paperHead{Ready}" * 4 + r" \\", r"\midrule",
    ]
    row_colors = {"online_mis_resident": "TableOrange", "priority_degree": "TableGreen", "priority_fused": "TableBlue"}
    for method, label in LABELS.items():
        columns = [label]
        if method == "priority_degree":
            lines.append(r"\midrule")
        if method in row_colors:
            lines.append(r"\rowcolor{" + row_colors[method] + "}")
        for split in SPLITS:
            r = cells.get((split, method))
            if r is None:
                reason = status.get(split, {}).get(method, r"\textemdash")
                columns.append(r"\multicolumn{3}{c}{" + reason + "}")
            elif "mean_size" not in r:
                columns.append(r"\multicolumn{3}{c}{\textemdash}")
            else:
                size = float(r["mean_size"])
                shown = f"{size:.4f}"
                if size > 0 and size == best_size[split]:
                    shown = r"\textbf{" + shown + "}"
                wall = float(r["mean_wall_ms"])
                time = r"$<0.01$" if 0 < wall < 0.01 else f"{wall:.2f}"
                columns.extend((shown, time, str(r["on_time"])))
        lines.append(" & ".join(columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}"])
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(path, graphs, *additional):
    rows = []
    for file in (path, *additional):
        with file.open(encoding="utf-8-sig", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    expected = {(f"idx_{i}", seed) for i in range(graphs) for seed in SEEDS}
    keys = {(r["graph_id"], int(r["seed"])) for r in rows}
    if len(rows) != len(expected) or keys != expected:
        raise ValueError(f"Incomplete graph/seed cohort in {path}")
    timely = 0
    for r in rows:
        ready = r["final_ready_ms"]
        if float(r["budget"]) != 0.2:
            raise ValueError(f"Wrong deadline in {path}")
        if ready and not (0 <= float(ready) <= 200):
            raise ValueError(f"Late incumbent in {path}")
        if not ready and float(r["final_size"]) != 0:
            raise ValueError(f"Missing incumbent has nonzero size in {path}")
        timely += bool(ready)
    return {"mean_size": fmean(float(r["final_size"]) for r in rows),
            "mean_wall_ms": fmean(float(r["wall_ms"]) for r in rows),
            "on_time": timely, "runs": len(rows)}


def collect(root):
    for split, count in SPLITS.items():
        for method in METHODS:
            if split == "RB_small" and method == "fused":
                paths = [V11 / f"fused_pave_02s_deadline/pave_off_s{seed}/v11_rows.csv" for seed in SEEDS]
            elif split == "RB_small":
                paths = [V11 / SMALL_RB_DIRECTORIES[method] / "per_graph.csv"]
            else:
                paths = [root / split / method / "per_graph.csv"]
                if not paths[0].is_file():
                    continue
            stats = summarize(paths[0], count, *paths[1:])
            yield dict(split=split, method=method, **stats,
                       source=";".join(p.relative_to(ROOT).as_posix() for p in paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--tex", type=Path, help="write the four-scale main table")
    parser.add_argument("--paper-csv", type=Path,
                        help="write all paper cohorts with per-graph sources, optionally alongside --tex")
    args = parser.parse_args()
    root = args.results.resolve()
    rows = list(collect(root))
    status_path = root / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    if args.paper_csv:
        write_table(rows, status, args.tex, args.paper_csv)
        return
    with (root / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*rows[0], "status"])
        writer.writeheader()
        writer.writerows(dict(row, status="measured") for row in rows)
        for split, failures in status.items():
            for method, reason in failures.items():
                writer.writerow(dict(split=split, method=method, status=reason,
                                     source=status_path.relative_to(ROOT).as_posix()))
    contrasts = {}
    for split in ("RB_large", "ER_small", "ER_large"):
        candidate = root / split / "fused/per_graph.csv"
        if not candidate.exists():
            continue
        for method in ("author_rlnn", "author_rlnn_matched", "rlsa", "online_mis"):
            reference = root / split / method / "per_graph.csv"
            if reference.exists():
                result = compare(candidate, reference)
                result["sources"].update(candidate=candidate.relative_to(ROOT).as_posix(),
                                         reference=reference.relative_to(ROOT).as_posix())
                contrasts.setdefault(split, {})[method] = result
    (root / "paired.json").write_text(json.dumps(contrasts, indent=2) + "\n", encoding="utf-8")
    if args.tex:
        write_table(rows, status, args.tex)


if __name__ == "__main__":
    main()
