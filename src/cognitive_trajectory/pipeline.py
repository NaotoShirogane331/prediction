"""CPU training, safe checkpoint reload, label-free prediction and evaluation."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import (
    Preprocessor,
    _date,
    _required,
    _strings,
    build_samples,
    check_data,
    fit_preprocessor,
    load_tables,
    make_splits,
)
from .losses import pinball_loss
from .model import CognitiveTrajectoryModel
from .schema import QUANTILE_COLUMNS, QUANTILES, SCHEMA_VERSION


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def empty_output(path: Path) -> Path:
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output must be an empty directory, already exists: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _output_file(path: Path) -> Path:
    path = Path(path)
    if path.exists():
        raise ValueError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _collate(samples: list[dict]) -> dict:
    lengths = torch.tensor([len(s["x"]) for s in samples], dtype=torch.long)
    x = torch.zeros(len(samples), int(lengths.max()), samples[0]["x"].shape[-1])
    missing = torch.ones_like(x, dtype=torch.bool)
    dt = torch.zeros(x.shape[:2])
    for i, s in enumerate(samples):
        n = len(s["x"])
        x[i, :n] = torch.as_tensor(s["x"], dtype=torch.float32)
        missing[i, :n] = torch.as_tensor(s["missing"], dtype=torch.bool)
        dt[i, :n] = torch.as_tensor(s["dt_days"], dtype=torch.float32)
    batch = dict(
        x=x,
        missing=missing,
        dt_days=dt,
        lengths=lengths,
        baseline=torch.tensor([s["baseline"] for s in samples], dtype=torch.float32),
        horizon_days=torch.tensor([s["horizon_days"] for s in samples], dtype=torch.float32),
    )
    if "target_score" in samples[0]:
        batch["target_score"] = torch.tensor([s["target_score"] for s in samples], dtype=torch.float32)
    return batch


def _forward(model, batch):
    return model(**{k: v for k, v in batch.items() if k != "target_score"})


def _validate_data(data_dir):
    report = check_data(Path(data_dir))
    if report["errors"]:
        raise ValueError("Invalid data: " + "; ".join(report["errors"]))
    return load_tables(Path(data_dir))


def _data_kind(directory: Path) -> str:
    provenance = directory / "provenance.json"
    if provenance.exists() and json.loads(provenance.read_text()).get("synthetic") is True:
        return "synthetic"
    return "user_supplied_unverified"


def train_model(
    data_dir: Path, output_dir: Path, epochs=30, patience=5, batch_size=16, learning_rate=0.001, seed=2026
) -> dict:
    if min(epochs, patience, batch_size) <= 0 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs, patience, batch_size and learning_rate must be positive")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output must be empty: {output_dir}")
    visits, _, targets = _validate_data(data_dir)
    splits = make_splits(visits, seed=seed)
    preprocessor = fit_preprocessor(visits, targets, splits["train"])
    samples = {
        name: build_samples(visits, targets, preprocessor, ids, include_labels=True)
        for name, ids in splits.items()
        if name in ("train", "validation")
    }
    if any(not s for s in samples.values()):
        raise ValueError("Training and validation partitions both need labeled samples")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = CognitiveTrajectoryModel()
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loaders = {
        name: DataLoader(
            s, batch_size=batch_size, shuffle=(name == "train"), collate_fn=_collate, num_workers=0
        )
        for name, s in samples.items()
    }
    history, best, best_state, best_epoch, stale = [], float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        losses = {}
        for partition, loader in loaders.items():
            model.train(partition == "train")
            total, count = 0.0, 0
            with torch.set_grad_enabled(partition == "train"):
                for batch in loader:
                    loss = pinball_loss(_forward(model, batch), batch["target_score"])
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite training/validation loss")
                    if partition == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                    total += float(loss.detach()) * len(batch["baseline"])
                    count += len(batch["baseline"])
            losses[partition] = total / count
        history.append(
            dict(epoch=epoch, train_pinball=losses["train"], validation_pinball=losses["validation"])
        )
        if losses["validation"] < best:
            best, best_state, best_epoch, stale = (
                losses["validation"],
                copy.deepcopy(model.state_dict()),
                epoch,
                0,
            )
        else:
            stale += 1
        if stale >= patience:
            break
    output_dir = empty_output(output_dir)
    torch.save(best_state, output_dir / "model.pt")
    write_json(output_dir / "preprocessor.json", preprocessor.to_dict())
    write_json(output_dir / "splits.json", splits)
    config = dict(
        schema_version=SCHEMA_VERSION,
        model={"rgcn_hidden": 32, "temporal_hidden": 64},
        quantiles=list(QUANTILES),
        seed=seed,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device="cpu",
        torch_version=torch.__version__,
        data_kind=_data_kind(Path(data_dir)),
    )
    write_json(output_dir / "config.json", config)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    summary = dict(
        epochs_completed=len(history),
        best_epoch=best_epoch,
        best_validation_pinball=best,
        parameters_changed=sum(not torch.equal(initial[k], v) for k, v in best_state.items()),
        train_samples=len(samples["train"]),
        validation_samples=len(samples["validation"]),
        people_per_split={k: len(v) for k, v in splits.items()},
        data_kind=config["data_kind"],
        device="cpu",
    )
    write_json(output_dir / "training_summary.json", summary)
    return summary


def predict(data_dir: Path, model_dir: Path, output_path: Path, split=None) -> pd.DataFrame:
    # Inference intentionally does not run label validation or read target_score.
    visits, _, targets = load_tables(Path(data_dir))
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported model schema version")
    preprocessor = Preprocessor.from_dict(json.loads((model_dir / "preprocessor.json").read_text()))
    ids = None
    if split is not None:
        partitions = json.loads((model_dir / "splits.json").read_text())
        if split not in partitions:
            raise ValueError(f"Unknown split: {split}")
        ids = partitions[split]
    samples = build_samples(visits, targets, preprocessor, person_ids=ids, include_labels=False)
    if not samples:
        raise ValueError("No prediction requests match the selected people/split")
    model = CognitiveTrajectoryModel(**config["model"])
    model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True, map_location="cpu"))
    model.eval()
    torch.set_num_threads(1)
    values = []
    with torch.inference_mode():
        for batch in DataLoader(samples, batch_size=64, collate_fn=_collate, shuffle=False):
            values.extend(_forward(model, batch).tolist())
    rows = []
    for sample, qs in zip(samples, values, strict=True):
        row = {
            k: sample[k] for k in ("person_id", "origin_visit_id", "origin_date", "target_date", "baseline")
        }
        row["horizon_days"] = sample["horizon_days"]
        row.update(zip(QUANTILE_COLUMNS, qs, strict=True))
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(_output_file(output_path), index=False)
    return frame


def evaluate(data_dir: Path, predictions_path: Path, output_path: Path) -> dict:
    _, _, targets = load_tables(Path(data_dir))
    predictions = pd.read_csv(
        predictions_path, dtype={"person_id": str, "origin_visit_id": str}, keep_default_na=False
    )
    keys = ["person_id", "origin_visit_id", "target_date"]
    for frame, name in [(predictions, "predictions"), (targets, "targets")]:
        _required(frame, keys, name)
        _strings(frame, ["person_id", "origin_visit_id"], name)
        frame["target_date"] = frame["target_date"].map(lambda value: _date(value, "target_date"))
        if frame.duplicated(keys).any():
            raise ValueError(f"duplicate {name} keys")
    if predictions.empty:
        raise ValueError("No predictions to evaluate")
    if "target_score" not in targets:
        raise ValueError("Evaluation requires target_score labels")
    joined = predictions.merge(targets[keys + ["target_score"]], on=keys, how="left", validate="one_to_one")
    labels = pd.to_numeric(joined["target_score"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(labels).all():
        raise ValueError("All evaluated predictions require finite target_score labels")
    qs = joined[list(QUANTILE_COLUMNS)].to_numpy(dtype=float)
    baseline = joined["baseline"].to_numpy(dtype=float)
    if not np.isfinite(qs).all() or not np.isfinite(baseline).all() or (np.diff(qs, axis=1) < 0).any():
        raise ValueError("Predictions must be finite with ordered quantiles and finite baseline")
    error = labels[:, None] - qs
    levels = np.asarray(QUANTILES)[None, :]
    metrics = dict(
        data_kind=_data_kind(Path(data_dir)),
        n_predictions=len(joined),
        median_mae=float(np.abs(qs[:, 2] - labels).mean()),
        pinball=float(np.maximum(levels * error, (levels - 1) * error).mean()),
        interval_90_coverage=float(((labels >= qs[:, 0]) & (labels <= qs[:, 4])).mean()),
        interval_90_mean_width=float((qs[:, 4] - qs[:, 0]).mean()),
        baseline_mae=float(np.abs(baseline - labels).mean()),
        note="Empirical metrics only; synthetic results do not establish real-data performance or calibration.",
    )
    write_json(_output_file(output_path), metrics)
    return metrics
