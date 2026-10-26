"""Command line interface: `sentiment download | train | evaluate | predict`."""

import argparse
import sys
from pathlib import Path

from sentiment import data
from sentiment.baseline import Baseline
from sentiment.metrics import Metrics, classification_metrics
from sentiment.neural import CNNClassifier, pick_device

MODELS = {"baseline": "TF-IDF + logistic regression", "cnn": "1D CNN (PyTorch)"}
FILES = {"baseline": "baseline.joblib", "cnn": "cnn.pt"}
VAL_FRACTION = 0.1


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.command(args)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def parser() -> argparse.ArgumentParser:
    data_dir = argparse.ArgumentParser(add_help=False)
    data_dir.add_argument("--data-dir", type=Path, default=Path("data"), help="default: data")
    model_dir = argparse.ArgumentParser(add_help=False)
    model_dir.add_argument("--model-dir", type=Path, default=Path("models"), help="default: models")

    p = argparse.ArgumentParser(prog="sentiment", description="IMDB movie review sentiment.")
    sub = p.add_subparsers(required=True, metavar="command")

    d = sub.add_parser("download", parents=[data_dir], help="download the dataset (about 41 MB)")
    d.set_defaults(command=download)

    t = sub.add_parser("train", parents=[data_dir, model_dir], help="train on the training split")
    t.add_argument("--model", choices=[*MODELS, "all"], default="all")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--epochs", type=int, default=8, help="CNN epochs (default: 8)")
    t.add_argument(
        "--max-len", type=int, default=400, help="CNN tokens kept per review (default: 400)"
    )
    t.add_argument("--device", default="auto", help="auto, cpu, mps or cuda (default: auto)")
    t.set_defaults(command=train)

    e = sub.add_parser("evaluate", parents=[data_dir, model_dir], help="score on the test split")
    e.add_argument("--device", default="auto", help="auto, cpu, mps or cuda (default: auto)")
    e.set_defaults(command=evaluate)

    pr = sub.add_parser("predict", parents=[model_dir], help="classify one or more texts")
    pr.add_argument("texts", nargs="+", metavar="text")
    pr.add_argument("--device", default="cpu", help="auto, cpu, mps or cuda (default: cpu)")
    pr.set_defaults(command=predict)
    return p


def download(args) -> int:
    data.download(args.data_dir)
    return 0


def train(args) -> int:
    train_set, val_set = data.train_val_split(
        data.load_split(args.data_dir, "train"), VAL_FRACTION, args.seed
    )
    print(f"{len(train_set)} training and {len(val_set)} validation reviews")
    for name in MODELS if args.model == "all" else [args.model]:
        print(f"Training {MODELS[name]}")
        if name == "baseline":
            model = Baseline.fit(train_set)
        else:
            device = pick_device(args.device)
            print(f"device: {device}")
            model = CNNClassifier.fit(
                train_set,
                val_set,
                seed=args.seed,
                device=device,
                epochs=args.epochs,
                max_len=args.max_len,
            )
        accuracy = score(model, val_set).accuracy
        print(f"{name}: val accuracy {accuracy:.4f}, trained in {model.train_seconds:.0f} s")
        model.save(args.model_dir / FILES[name])
    return 0


def evaluate(args) -> int:
    test = data.load_split(args.data_dir, "test")
    print(f"IMDB test split, {len(test)} reviews\n")
    print("| Model | Accuracy | Precision | Recall | F1 | Train time |")
    print("|---|---|---|---|---|---|")
    for name, model in load_models(args.model_dir, pick_device(args.device)).items():
        m = score(model, test)
        print(
            f"| {MODELS[name]} | {m.accuracy:.4f} | {m.precision:.4f} | {m.recall:.4f} "
            f"| {m.f1:.4f} | {model.train_seconds:.0f} s |"
        )
    return 0


def score(model, reviews: list[data.Review]) -> Metrics:
    probs = model.predict_proba([r.text for r in reviews])
    return classification_metrics([r.label for r in reviews], probs >= 0.5)


def predict(args) -> int:
    models = load_models(args.model_dir, pick_device(args.device))
    width = max(map(len, models))
    for text in args.texts:
        print(text)
        for name, model in models.items():
            p = float(model.predict_proba([text])[0])
            label = "positive" if p >= 0.5 else "negative"
            print(f"  {name:<{width}}  {label}  (P(positive) = {p:.3f})")
    return 0


def load_models(model_dir: Path, device) -> dict:
    """Load every trained model found in model_dir."""
    models = {}
    if (model_dir / FILES["baseline"]).exists():
        models["baseline"] = Baseline.load(model_dir / FILES["baseline"])
    if (model_dir / FILES["cnn"]).exists():
        models["cnn"] = CNNClassifier.load(model_dir / FILES["cnn"], device)
    if not models:
        raise FileNotFoundError(f"no trained models in {model_dir}; run `sentiment train` first")
    return models


if __name__ == "__main__":
    sys.exit(main())
