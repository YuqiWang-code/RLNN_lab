"""Decide whether fixed-continuation labels support formal v3 ranking training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def read_run(path):
    path = path / "run.json" if path.is_dir() else path
    return path.resolve(), json.loads(path.read_text(encoding="utf-8"))


def decide(validation_path, variant, training_path=None, validation_data=None):
    validation_path, validation = read_run(validation_path)
    summary = validation["summaries"][variant]
    validation_reasons = []
    if validation["status"] != "completed":
        validation_reasons.append("validation census is not completed")
    if validation["split"] != "validation" or validation["mode"] != "opportunities":
        validation_reasons.append("validation input must be the validation-start census")
    if summary["states"] != 300 or summary["graphs"] != 100:
        validation_reasons.append("validation requires all 300 initial states from 100 graphs")
    if summary["informative_graphs"] < 10:
        validation_reasons.append("validation informative_graphs < 10")
    if round(summary["oracle_gap"], 12) < 0.03:
        validation_reasons.append("validation all-state oracle_gap < 0.03")
    validation_pass = not validation_reasons
    training_metrics = None
    training_pass = None
    training_reasons = []
    if training_path is not None:
        training_path, training = read_run(training_path)
        training_metrics = dict(run_json=str(training_path),
                                states_path=str(training_path.parent / "states.pt"),
                                states=training["state_count"], samples=training["samples"],
                                informative_states=training["informative_states"],
                                informative_graphs=training["informative_graphs"],
                                horizon=training["config"]["horizon"],
                                anchored=training["config"]["anchored"])
        if training["status"] != "completed":
            training_reasons.append("training label collection is not completed")
        if training["split"] != "train" or training["mode"] != "labels":
            training_reasons.append("training input must be labels collected from the training split")
        horizon = 1 if variant == "D2_neutral_h1" else 2
        if training["config"]["horizon"] != horizon or bool(training["config"]["anchored"]) != (variant == "D4_anchored_h2"):
            training_reasons.append("training action type or horizon differs from the validation variant")
        if training["informative_states"] < 100:
            training_reasons.append("training informative_states < 100")
        if training["informative_graphs"] < 50:
            training_reasons.append("training informative_graphs < 50")
        training_pass = not training_reasons
    reasons = validation_reasons + training_reasons
    if training_path is None:
        reasons.append("training labels have not been evaluated")
    admitted = validation_pass and training_pass if training_path is not None else None
    return dict(version="v3", mode="training_admission", variant=variant,
                decided_at=datetime.now(timezone.utc).isoformat(),
                thresholds=dict(validation_states=300, validation_graphs=100,
                                validation_informative_graphs=10, validation_oracle_gap=0.03,
                                training_informative_states=100, training_informative_graphs=50),
                validation=dict(run_json=str(validation_path),
                                states_path=str(validation_data.resolve() if validation_data is not None
                                                else validation_path.parent / "states.pt"), **summary),
                training=training_metrics, validation_pass=validation_pass,
                training_pass=training_pass, train=admitted,
                **{"pass": admitted, "reasons": reasons})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--validation-data", type=Path,
                        help="Validation states used by the trainer, including collected intermediate states")
    parser.add_argument("--variant", choices=("D2_neutral_h1", "D3_neutral_h2", "D4_anchored_h2"), default="D3_neutral_h2")
    parser.add_argument("--output", type=Path, default=Path("outputs/v3/decision.json"))
    args = parser.parse_args()
    decision = decide(args.validation, args.variant, args.training, args.validation_data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(decision, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
