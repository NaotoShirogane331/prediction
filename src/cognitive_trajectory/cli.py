"""Command line entrypoints for the runnable research prototype."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parser():
    parser = argparse.ArgumentParser(
        prog="ctp", description="Cognitive trajectory prediction research prototype"
    )
    parser.add_argument("--version", action="version", version="0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Generate clearly labeled synthetic data")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--n-people", type=int, default=120)
    generate.add_argument("--seed", type=int, default=2026)
    prepare = sub.add_parser("prepare", help="Normalize and align CSV records and media indices")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("check", help="Report missing/duplicate/unmatched records")
    check.add_argument("--data", type=Path, required=True)
    train = sub.add_parser("train", help="Train the model with person-disjoint partitions")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--seed", type=int, default=2026)
    predict = sub.add_parser("predict", help="Predict without reading future scores")
    predict.add_argument("--data", type=Path, required=True)
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--split", choices=["train", "validation", "test"])
    evaluate = sub.add_parser("evaluate", help="Evaluate predictions against supplied future scores")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    demo = sub.add_parser(
        "demo", help="Run generation, training, reload, test prediction, evaluation and plotting"
    )
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--n-people", type=int, default=120)
    demo.add_argument("--epochs", type=int, default=30)
    demo.add_argument("--seed", type=int, default=2026)
    return parser


def run_demo(output: Path, n_people=120, epochs=30, seed=2026) -> dict:
    from .data import prepare_data
    from .pipeline import empty_output, evaluate, predict, train_model, write_json
    from .plotting import plot_trajectory
    from .synthetic import generate_data

    if epochs <= 0 or n_people < 10:
        raise ValueError("Demo needs epochs > 0 and n_people >= 10")
    output = empty_output(output)
    generate_data(output / "raw", n_people=n_people, seed=seed)
    prepare_data(output / "raw", output / "prepared")
    training = train_model(output / "prepared", output / "model", epochs=epochs, seed=seed)
    predict(output / "prepared", output / "model", output / "predictions.csv", split="test")
    metrics = evaluate(output / "prepared", output / "predictions.csv", output / "metrics.json")
    plot_trajectory(output / "prepared", output / "predictions.csv", output / "trajectory.png")
    summary = dict(
        synthetic=True,
        seed=seed,
        n_people=n_people,
        training=training,
        metrics=metrics,
        artifacts=dict(
            model="model/model.pt",
            predictions="predictions.csv",
            metrics="metrics.json",
            plot="trajectory.png",
        ),
    )
    write_json(output / "summary.json", summary)
    return summary


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate":
            from .synthetic import generate_data

            result = generate_data(args.output, n_people=args.n_people, seed=args.seed)
        elif args.command == "prepare":
            from .data import prepare_data

            result = prepare_data(args.input, args.output)
        elif args.command == "check":
            from .data import check_data

            result = check_data(args.data)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 2 if result["errors"] else 0
        elif args.command == "train":
            from .pipeline import train_model

            result = train_model(
                args.data,
                args.output,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
            )
        elif args.command == "predict":
            from .pipeline import predict

            frame = predict(args.data, args.model, args.output, split=args.split)
            result = {"predictions": len(frame), "output": str(args.output)}
        elif args.command == "evaluate":
            from .pipeline import evaluate

            result = evaluate(args.data, args.predictions, args.output)
        else:
            result = run_demo(args.output, n_people=args.n_people, epochs=args.epochs, seed=args.seed)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f"ctp: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
