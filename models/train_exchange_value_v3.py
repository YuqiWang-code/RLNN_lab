"""Train state-balanced action rankings on fixed MIS continuation labels."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
from torch.nn import functional as F

from mis_v3_value import MODES, STATS_DIM, PAVEV3, move_state


def non_tied_pairs(values):
    first, second = torch.triu_indices(len(values), len(values), offset=1)
    distinct = values[first] != values[second]
    first, second = first[distinct], second[distinct]
    first_better = values[first] > values[second]
    return torch.stack((torch.where(first_better, first, second),
                        torch.where(first_better, second, first)))


def load_states(path, split):
    states = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(states, list):
        raise ValueError("Label files must contain a list of state dictionaries")
    for state in states:
        if state["split"] != split:
            raise ValueError(f"{path} contains split {state['split']}, expected {split}")
        state["stats"] = torch.as_tensor(state["stats"], dtype=torch.float32).reshape(-1, STATS_DIM)
        state["values"] = torch.as_tensor(state["values"], dtype=torch.float32).reshape(-1)
        if len(state["stats"]) != len(state["values"]):
            raise ValueError(f"Candidate features and values differ for {state['graph_id']}")
        state["_pairs"] = non_tied_pairs(state["values"])
    return states


def rule_value(state, rule):
    if len(state["values"]) == 0:
        return 0.0
    return float(state["values"][state["rule_indices"][rule]])


def strongest_rule(states):
    names = ("coverage", "degree", "random")
    gains = {name: mean(rule_value(state, name) for state in states) for name in names}
    return max(names, key=gains.get), gains


def average_ranks(values):
    return 1 + (values[:, None] > values).sum(1).float() + 0.5 * ((values[:, None] == values).sum(1) - 1)


def state_pair_loss(scores, pairs, max_pairs, generator):
    if pairs.shape[1] > max_pairs:
        chosen = torch.randperm(pairs.shape[1], generator=generator)[:max_pairs].to(pairs.device)
        pairs = pairs[:, chosen]
    better, worse = pairs
    return F.softplus(scores[worse] - scores[better]).mean()


@torch.no_grad()
def validate(model, states, rule):
    model.eval()
    rows = []
    if model.mode == "stats":
        sizes = [len(state["values"]) for state in states]
        all_scores = model({"stats": torch.cat([state["stats"] for state in states])}).cpu()
        predictions = all_scores.split(sizes)
    else:
        predictions = (model(state).cpu() for state in states)
    for state, scores in zip(states, predictions):
        values = state["values"].cpu()
        pairs = state["_pairs"].cpu()
        row = dict(graph_id=state["graph_id"], origin=state.get("origin", "initial"),
                   deploy_stop=bool(state.get("deploy_stop", False)),
                   candidates=len(values), informative=bool(pairs.shape[1]),
                   selected_gain=0.0, rule_gain=rule_value(state, rule), oracle_gain=0.0,
                   selected_index=-1, pair_accuracy=None, top1=None, spearman=None)
        if len(values):
            selected = int(scores.argmax())
            upper = float(values.max())
            row.update(selected_gain=float(values[selected]), oracle_gain=upper, selected_index=selected)
            if pairs.shape[1]:
                order = scores[pairs[0]] - scores[pairs[1]]
                row["pair_accuracy"] = float((order.gt(0).float() + 0.5 * order.eq(0)).mean())
                row["top1"] = float(values[selected] == upper)
                rank_score, rank_value = average_ranks(scores), average_ranks(values)
                rank_score -= rank_score.mean()
                rank_value -= rank_value.mean()
                norm = rank_score.norm() * rank_value.norm()
                if float(norm) > 0:
                    row["spearman"] = float((rank_score * rank_value).sum() / norm)
        row["delta_vs_rule"] = row["selected_gain"] - row["rule_gain"]
        row["regret"] = row["oracle_gain"] - row["selected_gain"]
        rows.append(row)

    def summarize(subset):
        result = dict(states=len(subset), candidate_states=sum(row["candidates"] > 0 for row in subset),
                      informative_states=sum(row["informative"] for row in subset),
                      informative_graphs=len({row["graph_id"] for row in subset if row["informative"]}))
        for key in ("selected_gain", "rule_gain", "oracle_gain", "delta_vs_rule", "regret"):
            result[key] = mean(row[key] for row in subset)
        result["informative_coverage"] = result["informative_states"] / len(subset)
        positive = [row for row in subset if row["oracle_gain"] > 0]
        result["positive_oracle_coverage"] = len(positive) / len(subset)
        result["positive_hit_rate"] = mean(row["selected_gain"] > 0 for row in positive) if positive else None
        for key in ("pair_accuracy", "top1", "spearman"):
            measured = [row[key] for row in subset if row[key] is not None]
            result[f"informative_{key}"] = mean(measured) if measured else None
        return result

    metrics = summarize(rows)
    by_origin = {origin: summarize([row for row in rows if row["origin"] == origin])
                 for origin in sorted({row["origin"] for row in rows})}
    return metrics, by_origin, rows


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--valid-data", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, default="stats")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    started = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    result = args.result_dir
    result.mkdir(parents=True, exist_ok=True)
    train, valid = load_states(args.train_data, "train"), load_states(args.valid_data, "validation")
    if not train or not valid:
        raise ValueError("Training and validation label files must both contain deployment states")
    overlap = {state["graph_id"] for state in train} & {state["graph_id"] for state in valid}
    if overlap:
        raise ValueError(f"Training and validation share {len(overlap)} graph identifiers")
    rule, rule_gains = strongest_rule(valid)
    informative = [state for state in train if state["_pairs"].shape[1]]
    metadata = dict(version="v3", mode="value_train", model=args.mode, train_seed=args.seed,
                    config=vars(args), started_at=started_at, train_states=len(train), val_states=len(valid),
                    train_informative=len(informative),
                    train_informative_graphs=len({state["graph_id"] for state in informative}),
                    val_informative=sum(bool(state["_pairs"].shape[1]) for state in valid),
                    strongest_rule=rule, validation_rule_gains=rule_gains)
    if not informative or not metadata["val_informative"]:
        write_json(result / "run.json", dict(metadata, status="skipped", reason="no informative ranking states"))
        return
    model = PAVEV3(args.mode).to(args.device)
    learning_rate = args.lr if args.lr is not None else (1e-3 if args.mode == "stats" else 3e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # The small statistics tensors fit on one device; graph encoders transfer one state at a time.
    if args.mode == "stats":
        informative = [move_state(state, args.device) for state in informative]
        valid = [move_state(state, args.device) for state in valid]
    checkpoints = result / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    history, best, best_key = [], None, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        order = torch.randperm(len(informative), generator=generator)
        for batch in order.split(args.batch_size):
            losses = []
            batch_states = [informative[index] for index in batch.tolist()]
            if args.mode == "stats":
                all_scores = model({"stats": torch.cat([state["stats"] for state in batch_states])})
                predictions = all_scores.split([len(state["values"]) for state in batch_states])
            else:
                predictions = (model(state) for state in batch_states)
            for state, scores in zip(batch_states, predictions):
                pairs = state["_pairs"].to(scores.device)
                losses.append(state_pair_loss(scores, pairs, args.max_pairs, generator))
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.extend(float(value.detach()) for value in losses)
        metrics, by_origin, rows = validate(model, valid, rule)
        row = dict(epoch=epoch, loss=mean(epoch_losses), elapsed_s=perf_counter() - started, **metrics)
        history.append(row)
        key = metrics["delta_vs_rule"], -metrics["regret"]
        if best_key is None or key > best_key:
            best_key, best = key, row
            torch.save(dict(mode=args.mode, stats_dim=STATS_DIM, model_state=model.state_dict(),
                            strongest_rule=rule, seed=args.seed, epoch=epoch, metrics=metrics),
                       checkpoints / "best.pt")
            write_json(result / "best_validation.json", dict(metrics=metrics, by_origin=by_origin))
            write_csv(result / "best_validation_per_state.csv", rows)
        write_csv(result / "metrics.csv", history)
        print(f"Epoch {epoch}: loss {row['loss']:.4f}, gain-vs-{rule} {metrics['delta_vs_rule']:+.5f}, "
              f"regret {metrics['regret']:.5f}, informative {metrics['informative_states']}/{metrics['states']}",
              flush=True)
    write_json(result / "run.json", dict(metadata, status="completed", best=best,
               params=sum(parameter.numel() for parameter in model.parameters()), learning_rate=learning_rate,
               finished_at=datetime.now(timezone.utc).isoformat(), wall_time_s=perf_counter() - started))


if __name__ == "__main__":
    main()
