import json

import numpy as np
import pandas as pd
import pytest
import torch


def test_train_reload_and_unlabeled_predictions(tmp_path):
    from cognitive_trajectory.model import CognitiveTrajectoryModel
    from cognitive_trajectory.pipeline import evaluate, predict, train_model
    from cognitive_trajectory.synthetic import generate_data

    data = tmp_path / "data"
    generate_data(data, n_people=12, seed=2026)
    model_dir = tmp_path / "model"
    summary = train_model(data, model_dir, epochs=2, batch_size=6)
    assert summary["epochs_completed"] == 2
    assert np.isfinite(summary["best_validation_pinball"])
    assert summary["parameters_changed"] > 0
    torch.manual_seed(2026)
    initial = CognitiveTrajectoryModel().state_dict()
    trained = torch.load(model_dir / "model.pt", weights_only=True)
    assert any(not torch.equal(initial[k], v) for k, v in trained.items())
    p1 = predict(data, model_dir, tmp_path / "p1.csv", split="test")
    p2 = predict(data, model_dir, tmp_path / "p2.csv", split="test")
    np.testing.assert_allclose(p1.filter(regex="^q").to_numpy(), p2.filter(regex="^q").to_numpy())
    metrics = evaluate(data, tmp_path / "p1.csv", tmp_path / "metrics.json")
    assert metrics["n_predictions"] == len(p1)
    assert metrics["data_kind"] == "synthetic"
    assert 0 <= metrics["interval_90_coverage"] <= 1
    assert metrics["interval_90_mean_width"] > 0
    assert metrics["baseline_mae"] >= 0
    targets = pd.read_csv(data / "targets.csv", dtype={"person_id": str, "origin_visit_id": str})
    targets["target_score"] = "not used for prediction"
    targets.to_csv(data / "targets.csv", index=False)
    p3 = predict(data, model_dir, tmp_path / "p3.csv", split="test")
    np.testing.assert_allclose(p1.filter(regex="^q").to_numpy(), p3.filter(regex="^q").to_numpy())
    targets.drop(columns="target_score").to_csv(data / "targets.csv", index=False)
    p4 = predict(data, model_dir, tmp_path / "p4.csv", split="test")
    np.testing.assert_allclose(p1.filter(regex="^q").to_numpy(), p4.filter(regex="^q").to_numpy())
    with pytest.raises(ValueError, match="label|score"):
        evaluate(data, tmp_path / "p1.csv", tmp_path / "unlabeled_metrics.json")
    with pytest.raises(ValueError, match="empty|exist"):
        train_model(data, model_dir, epochs=1)


def test_demo_cli_creates_real_outputs_and_protects_existing(tmp_path):
    from cognitive_trajectory.cli import main

    output = tmp_path / "demo"
    assert main(["demo", "--output", str(output), "--n-people", "12", "--epochs", "1"]) == 0
    for file in ["model/model.pt", "predictions.csv", "metrics.json", "trajectory.png", "summary.json"]:
        assert (output / file).stat().st_size > 0
    assert (output / "trajectory.png").read_bytes().startswith(b"\x89PNG")
    summary = json.loads((output / "summary.json").read_text())
    assert summary["synthetic"] is True
    assert main(["demo", "--output", str(output), "--n-people", "12", "--epochs", "1"]) != 0


def test_evaluate_rejects_duplicate_prediction_keys(tmp_path):
    from cognitive_trajectory.pipeline import evaluate
    from cognitive_trajectory.synthetic import generate_data

    data = tmp_path / "data"
    generate_data(data, n_people=10)
    targets = pd.read_csv(data / "targets.csv", dtype=str).iloc[[0, 0]].copy()
    for col, value in zip(["q05", "q25", "q50", "q75", "q95"], [10, 15, 20, 25, 29]):
        targets[col] = value
    targets["baseline"] = 20.0
    targets.to_csv(tmp_path / "predictions.csv", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(data, tmp_path / "predictions.csv", tmp_path / "metrics.json")


def test_literal_na_identifiers_survive_prediction_and_evaluation(tmp_path):
    from cognitive_trajectory.pipeline import evaluate, predict, train_model
    from cognitive_trajectory.synthetic import generate_data

    data = tmp_path / "data"
    generate_data(data, n_people=10)
    for filename in ["visits.csv", "assets.csv", "targets.csv"]:
        frame = pd.read_csv(data / filename, dtype=str, keep_default_na=False)
        frame["person_id"] = frame.person_id.replace({"0001": "NA"})
        frame.to_csv(data / filename, index=False)
    train_model(data, tmp_path / "model", epochs=1)
    result = predict(data, tmp_path / "model", tmp_path / "pred.csv")
    assert "NA" in set(result.person_id)
    metrics = evaluate(data, tmp_path / "pred.csv", tmp_path / "metrics.json")
    assert metrics["n_predictions"] == 30


def test_evaluation_uses_same_canonical_dates_as_prediction(tmp_path):
    from cognitive_trajectory.pipeline import evaluate, predict, train_model
    from cognitive_trajectory.synthetic import generate_data

    data = tmp_path / "data"
    generate_data(data, n_people=10)
    train_model(data, tmp_path / "model", epochs=1)
    targets = pd.read_csv(data / "targets.csv", dtype=str, keep_default_na=False)
    targets["target_date"] = targets.target_date.map(
        lambda value: "-".join(str(int(piece)) for piece in value.split("-"))
    )
    targets.to_csv(data / "targets.csv", index=False)
    result = predict(data, tmp_path / "model", tmp_path / "pred.csv")
    metrics = evaluate(data, tmp_path / "pred.csv", tmp_path / "metrics.json")
    assert metrics["n_predictions"] == len(result)
