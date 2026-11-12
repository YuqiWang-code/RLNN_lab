"""Join the three positioning runs into one table: wall clock against set size.

Inputs (all on the same 100 validation graphs, same search seed, frozen PAVE v3 H1):
  outputs/v8_macro/positioning/kamis/kamis_ref0.csv             KaMIS at four limits
  outputs/v8_macro/positioning/capture_budget/capture_budget_ref0.csv    eager vs captured, 0.5/1.0 s
  outputs/v8_macro/positioning_wall/capture_budget/capture_budget_ref0.csv captured only, 2.5/5.0 s

Outputs ``summary/positioning.md`` and ``summary/positioning_ref0.json``: the
budget curve of both methods, the paired drive-by-drive comparison at matched wall
clock, and the engine ablation (eager vs captured) that makes the small clocks
reachable at all.

    python -u models/summarize_v8_positioning.py --reference_seed 0
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import torch

ROOT = Path(__file__).resolve().parents[1]
POSITIONING = ROOT / "outputs/v8_macro/positioning"
WALL = ROOT / "outputs/v8_macro/positioning_wall"
BOOTSTRAP = 5000


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def pool(rows, key, value, extra=None):
    """Mean of ``value`` per ``key``, optionally pooling an ``extra`` column too."""
    buckets = {}
    for row in rows:
        buckets.setdefault(key(row), []).append(row)
    out = {}
    for bucket, items in buckets.items():
        entry = dict(n=len(items), value=mean(float(item[value]) for item in items))
        if extra:
            entry.update({name: mean(float(item[name]) for item in items) for name in extra})
        out[bucket] = entry
    return out


def bootstrap(values, resamples=BOOTSTRAP, seed=0):
    if not values:
        return dict(mean=0.0, low=0.0, high=0.0, wins=0, ties=0, losses=0, n=0)
    tensor = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(0, tensor.numel(), (resamples, tensor.numel()), generator=generator)
    means = tensor[index].mean(dim=1)
    return dict(mean=float(tensor.mean()), low=float(means.quantile(0.025)),
                high=float(means.quantile(0.975)), wins=sum(v > 0 for v in values),
                ties=sum(v == 0 for v in values), losses=sum(v < 0 for v in values),
                n=len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_seed", type=int, default=0)
    parser.add_argument("--positioning", type=Path, default=POSITIONING)
    parser.add_argument("--wall", type=Path, default=WALL)
    args = parser.parse_args()
    ref = args.reference_seed

    kamis_rows = read_rows(args.positioning / "kamis" / f"kamis_ref{ref}.csv")
    kamis = pool(kamis_rows, lambda r: float(r["budget"]), "size", extra=("wall_time_s",))
    kamis_graph = pool(kamis_rows, lambda r: (r["graph_id"], float(r["budget"])), "size")

    first = read_rows(args.positioning / "capture_budget" / f"capture_budget_ref{ref}.csv")
    second = read_rows(args.wall / "capture_budget" / f"capture_budget_ref{ref}.csv")
    arm_rows = pool(first, lambda r: (r["arm"], float(r["budget"])), "final_size",
                    extra=("steps", "solve_time_s", "ms_per_step", "prelude_s"))
    arm_graph = pool(first + second, lambda r: (r["arm"], r["graph_id"], float(r["budget"])),
                     "final_size")
    arms = sorted({r["arm"] for r in first})
    engine_rows, engine_graph = arm_rows, arm_graph
    curve = pool(first + second, lambda r: (r["arm"], float(r["budget"])), "final_size",
                 extra=("steps", "solve_time_s", "ms_per_step"))

    lines = []
    lines.append(f"# Positioning: wall clock against independent-set size\n")
    lines.append(f"100 validation graphs, search seed "
                 f"{sorted({int(r['search_seed']) for r in first})}, frozen backbone and frozen "
                 f"PAVE (horizon=1). Size = mean count of vertices in the returned independent "
                 f"set; wall = measured mean seconds per graph for one full solve.\n")
    lines.append("## Budget curve\n")
    lines.append("| method | clock | steps | size | wall s/graph |")
    lines.append("| --- | --- | --- | --- | --- |")
    for limit in sorted(kamis):
        entry = kamis[limit]
        lines.append(f"| KaMIS (ReduMIS) | {limit:g} s limit | - | {entry['value']:.4f} | "
                     f"{entry['wall_time_s']:.2f} |")
    for (arm, budget), entry in sorted(curve.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        lines.append(f"| RLNN {arm} | {budget:g} s | {entry['steps']:.0f} | "
                     f"{entry['value']:.4f} | {entry['solve_time_s']:.2f} |")
    lines.append("")

    lines.append("## Engine ablation at equal clock\n")
    lines.append("| clock | eager size | captured size | paired captured-eager [CI] | wins/ties/losses | steps ratio |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    ablation = {}
    for budget in sorted({b for (_, b) in engine_rows}):
        cell = {arm: engine_rows.get((arm, budget)) for arm in arms}
        if not all(cell.values()) or len(arms) < 2:
            continue
        diffs = []
        ratios = []
        for graph_id in sorted({r["graph_id"] for r in first}):
            eager = engine_graph.get(("eager", graph_id, budget))
            captured = engine_graph.get(("captured", graph_id, budget))
            if eager and captured:
                diffs.append(captured["value"] - eager["value"])
        for graph_id in sorted({r["graph_id"] for r in first}):
            eager = [r for r in first if r["graph_id"] == graph_id and r["arm"] == "eager"
                     and float(r["budget"]) == budget]
            captured = [r for r in first if r["graph_id"] == graph_id and r["arm"] == "captured"
                        and float(r["budget"]) == budget]
            if eager and captured:
                ratios.append(mean(float(r["steps"]) for r in captured)
                              / max(mean(float(r["steps"]) for r in eager), 1e-9))
        stats = bootstrap(diffs, seed=ref)
        ratio = mean(ratios) if ratios else 0.0
        ablation[str(budget)] = dict(eager=cell["eager"]["value"], captured=cell["captured"]["value"],
                                     paired=stats, step_ratio=ratio)
        lines.append(f"| {budget:g} s | {cell['eager']['value']:.4f} | {cell['captured']['value']:.4f} | "
                     f"{stats['mean']:+.4f} [{stats['low']:+.4f},{stats['high']:+.4f}] | "
                     f"{stats['wins']}/{stats['ties']}/{stats['losses']} | x{ratio:.2f} |")
    lines.append("")

    lines.append("## At matched wall clock\n")
    lines.append("KaMIS has a floor of about 1.9 s per graph even at a 0.1 s limit, so the small "
                 "clocks have no KaMIS entry.\n")
    lines.append("| wall s/graph | KaMIS size | RLNN captured size | paired captured-KaMIS [CI] | wins/ties/losses |")
    lines.append("| --- | --- | --- | --- | --- |")
    matched = {}
    for limit in sorted(kamis):
        wall = kamis[limit]["wall_time_s"]
        candidates = [(budget, entry) for (arm, budget), entry in curve.items() if arm == "captured"]
        budget, entry = min(candidates, key=lambda item: abs(item[1]["solve_time_s"] - wall))
        diffs = []
        for graph_id in sorted({r["graph_id"] for r in kamis_rows}):
            ours = arm_graph.get(("captured", graph_id, budget))
            theirs = kamis_graph.get((graph_id, limit))
            if ours and theirs:
                diffs.append(ours["value"] - theirs["value"])
        stats = bootstrap(diffs, seed=ref)
        ours_wall = entry["solve_time_s"]
        matched[str(limit)] = dict(kamis_size=kamis[limit]["value"], kamis_wall=wall,
                                   ours_budget=budget, ours_size=entry["value"],
                                   ours_wall=ours_wall, paired=stats)
        lines.append(f"| {wall:.2f} (KaMIS {limit:g} s) vs {ours_wall:.2f} (ours {budget:g} s) | "
                     f"{kamis[limit]['value']:.4f} | {entry['value']:.4f} | "
                     f"{stats['mean']:+.4f} [{stats['low']:+.4f},{stats['high']:+.4f}] | "
                     f"{stats['wins']}/{stats['ties']}/{stats['losses']} |")
    lines.append("")
    texts = "\n".join(lines) + "\n"

    summary = dict(ref=ref, graphs=len({r["graph_id"] for r in kamis_rows}),
                   kamis={str(k): v for k, v in kamis.items()},
                   curve={f"{arm}@{b:g}": v for (arm, b), v in curve.items()},
                   ablation=ablation, matched_wall=matched)
    out = args.positioning / "summary"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"positioning_ref{ref}.md").write_text(texts, encoding="utf-8")
    (out / f"positioning_ref{ref}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(texts)
    print(f"wrote {out / f'positioning_ref{ref}.md'}")


if __name__ == "__main__":
    main()
