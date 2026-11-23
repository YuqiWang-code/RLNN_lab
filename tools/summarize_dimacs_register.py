"""Score the external DIMACS REG cohort and certify per-graph clique-cover bounds."""
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from statistics import fmean, median


def read_graphs(dataset):
    entries = json.loads((dataset / "tensor_manifest.json").read_text())["graphs"]
    graphs = {}
    for entry in entries:
        filename = entry["source"]
        adjacency = [set() for _ in range(entry["num_nodes"])]
        for line in (dataset / "raw" / filename).read_text().splitlines():
            if line.startswith("e "):
                a, b = (int(v) - 1 for v in line.split()[1:])
                adjacency[a].add(b)
                adjacency[b].add(a)
        groups = []
        for vertex in sorted(range(len(adjacency)), key=lambda v: (len(adjacency[v]), v)):
            for clique in groups:
                if all(member in adjacency[vertex] for member in clique):
                    clique.append(vertex)
                    break
            else:
                groups.append([vertex])
        if sorted(v for group in groups for v in group) != list(range(len(adjacency))):
            raise ValueError(f"Incomplete clique partition for {filename}")
        if any(b not in adjacency[a] for group in groups for a, b in combinations(group, 2)):
            raise ValueError(f"Invalid clique upper bound for {filename}")
        graphs[Path(filename).stem] = (adjacency, groups)
    return graphs


def check_arm(root, graphs, seeds, deadline_ms):
    with (root / "per_graph.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(graph, seed) for graph in graphs for seed in seeds}
    records = {(row["graph_id"], int(row["seed"])): row for row in rows}
    if set(records) != expected or len(records) != len(rows):
        raise ValueError(f"Incomplete or repeated graph/seed rows in {root}")
    stats = {}
    for line in (root / "solutions.jsonl").read_text().splitlines():
        event = json.loads(line)
        key = (event["graph_id"], int(event["seed"]))
        row = records[key]
        adjacency, cliques = graphs[key[0]]
        sizes, times = [], []
        for update in event["updates"]:
            nodes = set(update["nodes"])
            if (len(nodes) != len(update["nodes"]) or any(v < 0 or v >= len(adjacency) for v in nodes)
                    or any(adjacency[v] & nodes for v in nodes)):
                raise ValueError(f"{root} {key}: infeasible witness")
            sizes.append(len(nodes))
            times.append(float(update["ready_ms"]))
        first_size = sizes[0] if sizes else 0
        size = sizes[-1] if sizes else 0
        if (key in stats or size != int(row["final_size"]) or max(sizes, default=0) != size
                or size > len(cliques) or times != sorted(times)
                or any(t < 0 or t > deadline_ms for t in times)
                or bool(times) != bool(row["final_ready_ms"])
                or (times and (times[0] != float(row["first_ready_ms"]) or times[-1] != float(row["final_ready_ms"])))
                or 1000 * float(row["budget"]) != deadline_ms):
            raise ValueError(f"{root} {key}: inconsistent times, size or upper bound")
        stats[key] = dict(size=size, first_size=first_size, wall_ms=float(row["wall_ms"]),
                          ready=int(bool(times)), first_ready_ms=times[0] if times else None,
                          optimal=int(bool(times) and size == len(cliques)),
                          optimal_ms=next((t for t, s in zip(times, sizes) if s == len(cliques)), None),
                          first_optimal=int(bool(times) and first_size == len(cliques)))
    if set(stats) != expected:
        raise ValueError(f"Missing witness rows in {root}")
    return stats


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    config = json.loads((args.results / "protocol.json").read_text())
    graphs = read_graphs(args.dataset)
    seeds = config["seeds"]
    methods = {method: check_arm(args.results / method, graphs, seeds, config["budget_ms"])
               for method in config["methods"]}
    graph_rows = []
    certificates = {}
    for graph, (adjacency, cover) in graphs.items():
        row = dict(graph=graph, vertices=len(adjacency), edges=sum(map(len, adjacency)) // 2,
                   isolates=sum(not x for x in adjacency), clique_upper_bound=len(cover))
        certificates[graph] = cover
        for method, records in methods.items():
            sample = [records[graph, seed] for seed in seeds]
            row[method + "_first_size"] = fmean(r["first_size"] for r in sample)
            row[method + "_size"] = fmean(r["size"] for r in sample)
            row[method + "_ready"] = sum(r["ready"] for r in sample)
            row[method + "_optimal"] = sum(r["optimal"] for r in sample)
        row["all_methods_tied_pairs"] = sum(len({a[graph, seed]["size"] for a in methods.values()}) == 1
                                            for seed in seeds)
        graph_rows.append(row)
    write_csv(args.results / "by_graph.csv", graph_rows)
    (args.results / "clique_covers.json").write_text(json.dumps(certificates, separators=(",", ":")) + "\n")
    scores = []
    for method, records in methods.items():
        sample = list(records.values())
        scores.append(dict(method=method, pairs=len(sample), mean_size=fmean(r["size"] for r in sample),
                           on_time=sum(r["ready"] for r in sample), mean_wall_ms=fmean(r["wall_ms"] for r in sample),
                           returned_on_time=sum(r["wall_ms"] <= config["budget_ms"] for r in sample),
                           first_ready_median_ms=median(r["first_ready_ms"] for r in sample if r["ready"]),
                           first_optimum_median_ms=median(r["optimal_ms"] for r in sample if r["optimal"]),
                           mean_first_size=fmean(r["first_size"] for r in sample),
                           improved_pairs=sum(r["size"] > r["first_size"] for r in sample),
                           optimal_pairs=sum(r["optimal"] for r in sample),
                           first_optimal_pairs=sum(r["first_optimal"] for r in sample)))
    write_csv(args.results / "summary.csv", scores)


if __name__ == "__main__":
    main()
