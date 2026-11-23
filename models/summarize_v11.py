"""Summarise two v11 evaluation runs: paired, per-budget, pooled over seeds.

Direction lock §2.7 defines the success criterion on **graph-level paired** diffs,
pooled over three evaluation seeds (5000 bootstrap resamples, CI must exclude 0 and
the point estimate must reach +0.020 at 1.0 s / +0.030 at 0.2 s), with a robustness
clause (>= 2/3 seeds same sign) and a downgrade clause (any seed < -0.010 => unstable).

Usage::

    python models/summarize_v11.py --a outputs/v11/A0/v11_rows.csv \
                                   --b outputs/v11/A1/v11_rows.csv --boot 5000

The v11 CSV is what ``models/eval_v8_padded.py --v11_csv`` writes, so this script
never parses the legacy schema; the legacy CSV/summary stay untouched for A0.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from statistics import mean  # noqa: E402

import torch  # noqa: E402

def paired_bootstrap(diffs, resamples=5000, seed=0):
    """Verbatim copy of ``eval_v8_capture_budget.paired_bootstrap`` (torch.float64).

    t16 (t13-F6): the reference implementation sits behind ``eval_v8_capture_budget``
    -> ``eval_v2`` -> ``solvers`` -> ``torch_scatter``/``tqdm``, none of which a bare
    local interpreter has, so this CLI could not be exercised without hand-made
    stubs.  The copy is op-for-op identical (same dtype, same ``torch.Generator``
    seed, same resample matrix), which makes the numbers bit-identical rather than
    merely equivalent; ``--check_reference`` re-verifies that on a machine where the
    heavy import works.
    """
    if not diffs:
        return 0.0, 0.0, 0.0
    values = torch.tensor(diffs, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(0, values.numel(), (resamples, values.numel()), generator=generator)
    means = values[index].mean(dim=1)
    return float(values.mean()), float(means.quantile(0.025)), float(means.quantile(0.975))

S1_TARGET = {1.0: 0.020, 0.2: 0.030}
MIN_SEED = -0.010


def load(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def per_key(rows, key=("graph_id", "budget")):
    """(graph, budget) -> mean final_size over repeats, per seed: {seed: {key: size}}."""
    table = defaultdict(lambda: defaultdict(list))
    for row in rows:
        # t16: the v11 artefact writes ``search_seed`` (the eval seed); ``seed`` is
        # accepted too so hand-built tables keep working.  Reading only ``seed``
        # silently collapsed every row onto seed 0 and made the three-seed
        # robustness clause unreachable.
        seed = int(float(row.get("seed") or row.get("search_seed") or 0))
        table[seed][tuple(float(row[field]) if field == "budget" else row[field]
                          for field in key)].append(float(row["final_size"]))
    return {seed: {k: mean(v) for k, v in keys.items()} for seed, keys in table.items()}


def compare(a_rows, b_rows, boot):
    a, b = per_key(a_rows), per_key(b_rows)
    seeds = sorted(set(a) & set(b))
    out = {}
    for budget in sorted({key[1] for table in (a, b) for key in table[seeds[0]]}):
        pooled, per_seed = [], {}
        for seed in seeds:
            keys = sorted({k for k in a[seed] if k[1] == budget})
            diffs = [b[seed][k] - a[seed][k] for k in keys if k in b[seed]]
            per_seed[seed] = diffs
            pooled.extend(diffs)
        if not pooled:
            continue
        gap, low, high = paired_bootstrap(pooled, boot) if pooled else (0.0, 0.0, 0.0)
        seed_means = {seed: mean(diffs) for seed, diffs in per_seed.items() if diffs}
        positive = sum(1 for value in seed_means.values() if value > 0)
        out[budget] = dict(gap=gap, low=low, high=high, pairs=len(pooled),
                           per_seed=seed_means, positive=positive, seeds=len(seed_means),
                           unstable=any(value < MIN_SEED for value in seed_means.values()))
    return out


def check_reference():
    """Compare the local copy with the reference implementation, bit for bit."""
    from eval_v8_capture_budget import paired_bootstrap as reference
    cases = [[], [0.5], [-0.25, 0.5, 0.125], [1.0, -2.0, 3.5, 0.0, -0.5, 2.0],
             [round(0.001 * index % 7 - 2, 4) for index in range(137)]]
    failures = []
    for case in cases:
        ours = paired_bootstrap(case, 5000, 1017)
        theirs = reference(case, 5000, 1017)
        if ours != theirs:
            failures.append(dict(case=case[:5], ours=ours, reference=theirs))
    report = dict(cases=len(cases), identical=not failures, failures=failures)
    print(json.dumps(report, indent=1))
    return 0 if not failures else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, default=None, help="baseline v11_rows.csv (A0)")
    parser.add_argument("--b", type=Path, default=None, help="treatment v11_rows.csv (A1)")
    parser.add_argument("--check_reference", action="store_true",
                        help="compare the local bootstrap copy with the reference")
    parser.add_argument("--boot", type=int, default=5000)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()
    if args.check_reference:
        return check_reference()
    if args.a is None or args.b is None:
        parser.error("--a and --b are required unless --check_reference is given")
    result = compare(load(args.a), load(args.b), args.boot)
    label = args.label or f"{args.b.stem} - {args.a.stem}"
    print(f"# {label}\n")
    print("| budget | pairs | Δ | 95% CI | per-seed Δ | +seeds | verdict |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    verdicts = {}
    for budget, stats in result.items():
        target = S1_TARGET.get(budget)
        passed = (stats["low"] > 0 and stats["gap"] >= target) if target else (stats["low"] > 0)
        robust = stats["positive"] >= (2 * stats["seeds"] + 2) // 3
        # t11 / R1: fixed priority unstable > fail > signal-only > pass.  A result
        # whose CI clears the target but whose seeds disagree is *not* a claim
        # ("signal-only", Discussion), and it is never counted for S1.
        if stats["unstable"]:
            verdict = "unstable"
        elif not passed:
            verdict = "fail"
        elif not robust:
            verdict = "signal-only"
        else:
            verdict = "pass"
        verdicts[budget] = verdict
        seeds = " ".join(f"{seed}:{value:+.4f}" for seed, value in sorted(stats["per_seed"].items()))
        print(f"| {budget:g} s | {stats['pairs']} | {stats['gap']:+.4f} | "
              f"[{stats['low']:+.4f},{stats['high']:+.4f}] | {seeds} | "
              f"{stats['positive']}/{stats['seeds']} | {verdict} |")
    notes = [budget for budget, verdict in verdicts.items() if verdict == "signal-only"]
    if notes:
        print("\n" + "; ".join(
            f"{budget:g} s signal-only" for budget in notes))
        print("signal-only = 有信号未达判据（Discussion only）: pooled CI excludes 0 and "
              "clears the target, but fewer than 2/3 seeds share the sign")
    s1 = any(verdicts.get(budget) == "pass" for budget in (1.0, 0.2))
    print(f"\nS1 ({(1.0, 0.2)} any) = {s1}; verdicts = {verdicts}")


if __name__ == "__main__":
    main()
