"""v11 M1' / C4 probe: does a node-identity FIFO make the plateau walk actually walk?

Direction lock §3.4 defines two measured quantities on the *same* search; both
definitions live in ``models/mis_v11_tabu.py`` (module docstring) and are copied
into the result JSON so a reader never has to guess the denominator:

* **ER** (elimination rate) ``1 - |pool_tabu| / |pool_plain|``, paired on the same
  ``(graph, state, outer round)``.  Denominator = number of outer rounds that
  actually built a candidate pool (``er_rounds``).  ``ER ~ 0`` means the FIFO never
  hit a candidate.  ``er_pooled = 1 - sum(pool_tabu)/sum(pool_plain)`` is reported
  next to it so a single large pool cannot dominate the mean.
* **AR** (acceptance rate) fraction of the ``outer_rounds`` whose end solution has
  the **same size**, a **different mask** (Hamming > 0) and **no archive refresh**.
  ``refreshed_rounds`` is reported next to it: "walked but never refreshed in 8
  rounds" is the second half of the C4 sentence.

Two things this probe now measures that the first version did not (captain review
2026-11-17 14:40, m01061): the real archive **gain** (was hard-coded to 0) and the
number of rounds that **refreshed** the archive best.

Both tabu conventions are swept (``--tabu_mode cross,same``): ``cross`` is L04
Alg.1 (a just-evicted node may not be re-added, a just-added node may not be
evicted); ``same`` is the literal same-set reading.  h = 0 is always the no-tabu
control.

CPU only, no training, no GPU: every number here is a property of the search, not of
the policy.  The policy (``--rankers``) is accepted for interface compatibility with
the locked command line but is not needed: the probe uses the frozen rule chooser
(``coverage``), so the walk can be measured without a GPU.

Usage (locked form, direction doc §3.5)::

    python -u models/probe_tabu_walk.py \\
      --test_path models/data/MIS_tensors/RB_small_author_test \\
      --graphs 200 --outer_rounds 8 --h 8,16,32 --budget 0 \\
      --result_dir outputs/v11/tabu_walk
"""
import argparse
import csv
import json
from math import inf
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from eval_v2 import dataset_at
from mis_v3_search import SearchGraph
from mis_v11_tabu import TabuList, platform_search_tabu

ER_FORMULA = "ER = 1 - |pool_tabu| / |pool_plain|, paired per (graph, state, outer round)"
AR_FORMULA = ("AR = |{rounds: size_after == size_before, mask changed, archive not refreshed}| "
              "/ outer_rounds")


def probe_parser():
    parser = argparse.ArgumentParser(description="v11 C4/L04 node-identity FIFO tabu probe")
    parser.add_argument("--test_path", default="models/data/MIS_tensors/RB_small_author_test")
    parser.add_argument("--graphs", type=int, default=200)
    parser.add_argument("--outer_rounds", type=int, default=8)
    parser.add_argument("--h", default="8,16,32,64,128", help="FIFO depths, comma separated (0 = no tabu)")
    parser.add_argument("--tabu_mode", default="cross,same", help="comma separated: cross,same")
    parser.add_argument("--budget", type=float, default=0.0,
                        help="seconds per graph (0 = unlimited, the deterministic setting)")
    parser.add_argument("--repair_rounds", type=int, default=4)
    parser.add_argument("--initial_rounds", type=int, default=3)
    parser.add_argument("--candidate_cap", type=int, default=32)
    parser.add_argument("--improving_cap", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--choose", default="coverage",
                        help="frozen rule chooser used for the CPU probe")
    parser.add_argument("--rankers", default=None,
                        help="accepted for CLI compatibility; the CPU probe does not load a policy")
    parser.add_argument("--seed", type=int, default=1017)
    parser.add_argument("--result_dir", default="outputs/v11/tabu_walk")
    return parser


def greedy_mis(graph):
    """Minimum-degree greedy independent set: the input a converged search starts from."""
    order = sorted(range(graph.num_nodes),
                   key=lambda node: (graph.neighbors[node].bit_count(), graph.priority[node]))
    selected = 0
    for node in order:
        if not graph.neighbors[node] & selected:
            selected |= 1 << node
    return selected


def walk(graph, selected, *, h, mode, rounds, budget, args):
    """One (graph, h, mode) run; returns the result, per-round records and the FIFO."""
    # An *empty* FIFO is the h=0 control: it forbids nothing, so it reproduces the
    # numbers this probe already reported, while `tabu=None` would now delegate to
    # the frozen search (mis_v11_tabu.platform_search_tabu) and drop on_round.
    tabu = TabuList(h, mode=mode)
    records = []
    deadline = (perf_counter() + budget) if budget > 0 else inf
    result = platform_search_tabu(
        graph, selected, tabu=tabu, choose=args.choose, outer_rounds=rounds,
        initial_rounds=args.initial_rounds, horizon=args.horizon,
        candidate_cap=args.candidate_cap, improving_cap=args.improving_cap,
        repair_rounds=args.repair_rounds, deadline=deadline,
        on_round=records.append)
    return result, records, tabu


def summarise(rows, rounds, start_size, best_size):
    """ER / AR / refresh / gain aggregates from the per-round rows."""
    macro = [row for row in rows if row["kind"] == "macro" and row["pool_plain"]]
    ratios = [1.0 - row["pool_tabu"] / row["pool_plain"] for row in macro]
    pooled_plain = sum(row["pool_plain"] for row in macro)
    pooled_tabu = sum(row["pool_tabu"] for row in macro)
    er_pooled = (1.0 - pooled_tabu / pooled_plain) if pooled_plain else float("nan")
    walked = [row for row in rows
              if row["size_after"] == row["size_before"] and row["changed"] and not row["refreshed"]]
    refreshed = [row for row in rows if row["refreshed"]]
    rejected = sum(row["rejected"] or 0 for row in macro)
    return dict(er=mean(ratios) if ratios else float("nan"),
                er_pooled=er_pooled,
                er_rounds=len(macro),
                er_rejected=rejected,
                er_plain=pooled_plain,
                er_tabu=pooled_tabu,
                ar_rounds=len(walked), ar=len(walked) / rounds,
                refreshed_rounds=len(refreshed),
                accept_rows=len(rows),
                gain=best_size - start_size)


def feasible(graph, mask):
    """Independent-set assertion (hard gate: the walk must never break feasibility)."""
    nodes = mask
    while nodes:
        bit = nodes & -nodes
        nodes ^= bit
        index = bit.bit_length() - 1
        if graph.neighbors[index] & (mask ^ bit):
            return False
    return True


def main(argv=None):
    args = probe_parser().parse_args(argv)
    depths = [int(part) for part in str(args.h).split(",") if part.strip() != ""]
    if 0 not in depths:
        depths = [0] + depths                      # always run the no-tabu control
    modes = [part.strip() for part in str(args.tabu_mode).split(",") if part.strip()]
    dataset = dataset_at(args.test_path, torch.float32, args.graphs)
    out = Path(args.result_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    fifo_stats = {}
    started = perf_counter()
    for mode in modes:
        for h in depths:
            tabu_checks = tabu_hits = 0
            sample = None
            for index in range(len(dataset)):
                graph_obj = dataset[index]
                graph = SearchGraph(graph_obj.num_nodes, graph_obj.edge_index, seed=args.seed)
                start = greedy_mis(graph)
                result, records, tabu = walk(graph, start, h=h, mode=mode,
                                             rounds=args.outer_rounds, budget=args.budget, args=args)
                stats = summarise(records, args.outer_rounds, start.bit_count(),
                                  result.best.bit_count())
                if not feasible(graph, result.best):
                    raise SystemExit(f"illegal set produced by the walk: graph {index}, h={h}")
                if tabu is not None:
                    tabu_checks += tabu.checks
                    tabu_hits += tabu.hits
                    sample = sample or tabu.sample
                rows.append(dict(mode=mode, h=h, graph_id=index, nodes=graph.num_nodes,
                                 start_size=start.bit_count(), end_size=result.work.bit_count(),
                                 best_size=result.best.bit_count(), gain=stats["gain"],
                                 rounds=len(records), exchanges=result.exchanges,
                                 deadline_hit=int(result.deadline_hit),
                                 er=stats["er"], er_pooled=stats["er_pooled"],
                                 er_rounds=stats["er_rounds"], er_rejected=stats["er_rejected"],
                                 er_plain=stats["er_plain"], er_tabu=stats["er_tabu"],
                                 ar_rounds=stats["ar_rounds"], ar=stats["ar"],
                                 refreshed_rounds=stats["refreshed_rounds"]))
            fifo_stats[f"{mode}:{h}"] = dict(checks=tabu_checks, hits=tabu_hits, sample=sample)
            subset = [row for row in rows if row["h"] == h and row["mode"] == mode]
            ers = [row["er"] for row in subset if row["er"] == row["er"]]
            print(f"mode={mode:5s} h={h:3d} graphs={len(subset)} "
                  f"start={mean(r['start_size'] for r in subset):.4f} "
                  f"best={mean(r['best_size'] for r in subset):.4f} "
                  f"gain={mean(r['gain'] for r in subset):+.4f} "
                  f"AR={mean(r['ar'] for r in subset):.4f} "
                  f"refresh={mean(r['refreshed_rounds'] for r in subset):.3f} "
                  f"ER={mean(ers) if ers else float('nan'):.4f} "
                  f"hits={tabu_hits}/{tabu_checks}", flush=True)
    summary = dict(graphs=len(dataset), outer_rounds=args.outer_rounds, budget=args.budget,
                   choose=args.choose, seed=args.seed, tabu_modes=modes,
                   formulas=dict(er=ER_FORMULA, ar=AR_FORMULA),
                   wall_s=round(perf_counter() - started, 1), per_h={})
    for mode in modes:
        for h in depths:
            key = f"{mode}:{h}"
            subset = [row for row in rows if row["h"] == h and row["mode"] == mode]
            ers = [row["er"] for row in subset if row["er"] == row["er"]]
            summary["per_h"][key] = dict(
                mode=mode, h=h, graphs=len(subset),
                start_size=mean(row["start_size"] for row in subset),
                best_size=mean(row["best_size"] for row in subset),
                gain=mean(row["gain"] for row in subset),
                ar=mean(row["ar"] for row in subset),
                refreshed_rounds=mean(row["refreshed_rounds"] for row in subset),
                er=mean(ers) if ers else None,
                er_pooled=mean(row["er_pooled"] for row in subset if row["er_pooled"] == row["er_pooled"]),
                er_rejected=sum(row["er_rejected"] for row in subset),
                er_rounds=mean(row["er_rounds"] for row in subset),
                fifo_checks=fifo_stats[key]["checks"], fifo_hits=fifo_stats[key]["hits"],
                fifo_sample=fifo_stats[key]["sample"],
                deadline_hits=sum(row["deadline_hit"] for row in subset))
    tabu_keys = [key for key in summary["per_h"] if not key.endswith(":0")]
    best_ar = max((summary["per_h"][key]["ar"] for key in tabu_keys), default=0.0)
    best_refresh = max((summary["per_h"][key]["refreshed_rounds"] for key in tabu_keys), default=0.0)
    best_hits = max((summary["per_h"][key]["fifo_hits"] for key in tabu_keys), default=0)
    # §3.5 success is a conjunction: AR >= 1/8 AND ER in [0.05, 0.6].  AR alone would
    # call the probe passed even when the FIFO never hits anything (ER = 0), which is
    # exactly the inert case, so both are required to say "pass".
    er_window = (0.05, 0.6)
    er_ok = any(summary["per_h"][key]["er"] is not None
                and er_window[0] <= summary["per_h"][key]["er"] <= er_window[1]
                for key in tabu_keys)
    passed = bool(best_ar >= 1.0 / args.outer_rounds and er_ok)
    kill_reason = None
    if not passed:
        if best_hits == 0:
            kill_reason = "tabu_never_fired"
        elif best_ar == 0:
            kill_reason = "ar_zero"
        elif best_refresh == 0:
            kill_reason = "no_refresh"
        else:
            kill_reason = "er_out_of_window"
    summary["verdict"] = dict(
        best_ar=best_ar,
        best_refreshed_rounds=best_refresh,
        best_fifo_hits=best_hits,
        ar_threshold=1.0 / args.outer_rounds,
        er_window=list(er_window),
        er_ok=bool(er_ok),
        passed=passed,
        kill_reason=kill_reason)
    (out / "tabu_walk.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / "tabu_walk.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary["verdict"], indent=2))
    print(f"wrote {out / 'tabu_walk.json'} and {out / 'tabu_walk.csv'}")
    if not summary["verdict"]["passed"]:
        if kill_reason == "ar_zero":
            print("kill: AR == 0/8 for every h -> stop this line, do not retune h (§3.5)")
        elif kill_reason == "no_refresh":
            print("kill: the walk moves (AR>0) but never refreshes the archive in 8 rounds "
                  "-> memory is not the bottleneck, the improvement direction does not exist")
        elif kill_reason == "tabu_never_fired":
            print("kill: the FIFO never rejected a candidate (hits == 0) -> the key space "
                  "dwarfs the window; stop, do not retune h (§3.5)")
        else:
            print("kill: ER outside [0.05,0.6] -> stop, do not retune h (§3.5)")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
