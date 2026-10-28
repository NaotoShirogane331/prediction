"""Data boundary and leakage regression tests using independently checked fixtures."""

import functools
import operator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cognitive_trajectory import data, synthetic
from cognitive_trajectory.schema import FEATURE_COLUMNS


def visit(person="001", identity="v1", date="2020-01-01", score=10.0, available=None, **features):
    row = dict.fromkeys(FEATURE_COLUMNS, 0.0)
    row.update(
        person_id=person,
        visit_id=identity,
        visit_date=date,
        session_id="s1",
        score=score,
        score_available_date=available or (date if score is not None else ""),
    )
    row.update(features)
    return row


def target(person="001", origin="v2", date="2020-04-10", score=12.0):
    return {"person_id": person, "origin_visit_id": origin, "target_date": date, "target_score": score}


def frames():
    return (
        pd.DataFrame([visit(action_1=0.0), visit(identity="v2", date="2020-01-11", score=20, action_1=2.0)]),
        pd.DataFrame([target()]),
    )


def write_tables(path: Path, visits, targets, assets=None):
    path.mkdir(parents=True, exist_ok=True)
    visits.to_csv(path / "visits.csv", index=False)
    targets.to_csv(path / "targets.csv", index=False)
    if assets is None:
        assets = pd.DataFrame(columns=["person_id", "visit_id", "modality", "path"])
    assets.to_csv(path / "assets.csv", index=False)


def test_generator_is_deterministic_fictional_and_has_irregular_visits(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    counts = synthetic.generate_data(a, n_people=10)
    synthetic.generate_data(b, n_people=10)
    for name in ["visits.csv", "targets.csv", "assets.csv", "provenance.json"]:
        assert (a / name).read_bytes() == (b / name).read_bytes()
    visits, assets, targets = data.load_tables(a)
    assert counts["people"] == 10
    assert visits.groupby("person_id").size().between(6, 10).all()
    assert targets.groupby("person_id").size().eq(3).all()
    assert visits["score"].dropna().between(0, 30).all()
    assert assets["path"].str.startswith("fictional/").all()
    assert not data.check_data(a)["errors"]
    assert set(a.iterdir()) == {a / x for x in ["visits.csv", "targets.csv", "assets.csv", "provenance.json"]}
    with pytest.raises(ValueError, match="10"):
        synthetic.generate_data(tmp_path / "bad", n_people=9)
    with pytest.raises(ValueError, match="empty"):
        synthetic.generate_data(a, n_people=10)


def test_split_is_person_disjoint_deterministic_and_preserves_ids():
    visits = pd.DataFrame({"person_id": [f"{i:03d}" for i in range(120)] * 2})
    result = data.make_splits(visits)
    assert result == data.make_splits(visits.sample(frac=1, random_state=2))
    assert [len(result[k]) for k in ["train", "validation", "test"]] == [84, 18, 18]
    assert set(result["train"]).isdisjoint(result["test"])
    assert set(result["train"]).isdisjoint(result["validation"])
    assert set(result["test"]).isdisjoint(result["validation"])
    assert len(set(functools.reduce(operator.iadd, result.values(), []))) == 120


def test_prepare_sorts_dates_joins_assets_and_preserves_leading_zeroes(tmp_path):
    visits, targets = frames()
    assets = pd.DataFrame(
        [
            {
                "person_id": "001",
                "visit_date": "2020-1-11",
                "session_id": "s1",
                "modality": "audio",
                "path": "fictional/a.wav",
            }
        ]
    )
    write_tables(tmp_path / "raw", visits.iloc[::-1], targets, assets)
    report = data.prepare_data(tmp_path / "raw", tmp_path / "prepared")
    assert not report["errors"]
    actual, actual_assets, _ = data.load_tables(tmp_path / "prepared")
    assert actual["visit_id"].tolist() == ["v1", "v2"]
    assert actual["person_id"].tolist() == ["001", "001"]
    assert actual_assets.iloc[0]["visit_id"] == "v2"
    assert (tmp_path / "prepared" / "preparation_report.json").exists()


@pytest.mark.parametrize(
    "change,match",
    [
        ("duplicate_visit", "duplicate"),
        ("duplicate_session", "session"),
        ("unmatched_asset", "match"),
        ("ambiguous_asset", "ambiguous"),
        ("unsafe_path", "path"),
        ("absolute_path", "path"),
        ("unknown_origin", "origin"),
        ("past_target", "after"),
        ("score_range", "score"),
        ("score_date", "score_available"),
        ("infinite_feature", "finite"),
        ("duplicate_target", "duplicate"),
    ],
)
def test_prepare_rejects_conflicts_and_check_reports_them(tmp_path, change, match):
    visits, targets = frames()
    assets = pd.DataFrame([{"person_id": "001", "visit_id": "v1", "modality": "audio", "path": "a.wav"}])
    if change == "duplicate_visit":
        visits = pd.concat([visits, visits.iloc[:1]], ignore_index=True)
    elif change == "duplicate_session":
        visits.loc[1, "visit_date"] = "2020-01-01"
    elif change == "unmatched_asset":
        assets.loc[0, "visit_id"] = "missing"
    elif change == "ambiguous_asset":
        visits.loc[1, ["visit_date", "session_id"]] = ["2020-01-01", "s2"]
        assets = pd.DataFrame(
            [{"person_id": "001", "visit_date": "2020-01-01", "modality": "audio", "path": "a.wav"}]
        )
    elif change == "unsafe_path":
        assets.loc[0, "path"] = "../secret.wav"
    elif change == "absolute_path":
        assets.loc[0, "path"] = "/tmp/a.wav"
    elif change == "unknown_origin":
        targets.loc[0, "origin_visit_id"] = "missing"
    elif change == "past_target":
        targets.loc[0, "target_date"] = "2020-01-11"
    elif change == "score_range":
        visits.loc[0, "score"] = 31
    elif change == "score_date":
        visits.loc[1, "score_available_date"] = "2020-01-01"
    elif change == "infinite_feature":
        visits.loc[0, "action_1"] = np.inf
    elif change == "duplicate_target":
        targets = pd.concat([targets, targets], ignore_index=True)
    write_tables(tmp_path / "raw", visits, targets, assets)
    report = data.check_data(tmp_path / "raw")
    assert report["errors"]
    with pytest.raises(ValueError, match=match):
        data.prepare_data(tmp_path / "raw", tmp_path / "normalized")
    assert not (tmp_path / "normalized" / "visits.csv").exists()


def test_preprocessor_uses_only_train_historical_values_and_roundtrips():
    visits, targets = frames()
    visits = pd.concat(
        [
            visits,
            pd.DataFrame(
                [
                    visit(identity="future", date="2021-01-01", score=30, action_1=999),
                    visit(person="002", identity="held", score=30, action_1=999),
                ]
            ),
        ],
        ignore_index=True,
    )
    pre = data.fit_preprocessor(visits, targets, ["001"])
    assert pre.means[0] == 1.0
    assert pre.scales[0] == 1.0
    assert pre.means[6] == 15.0
    restored = data.Preprocessor.from_dict(pre.to_dict())
    np.testing.assert_array_equal(pre.means, restored.means)
    np.testing.assert_array_equal(pre.scales, restored.scales)
    sample = data.build_samples(visits, targets, restored)[0]
    np.testing.assert_array_equal(sample["x"][:, 0], [-1, 1])
    np.testing.assert_array_equal(sample["dt_days"], [0, 10])
    assert sample["baseline"] == 20.0
    assert sample["horizon_days"] == 90.0
    assert sample["x"].dtype == np.float32
    assert sample["missing"].dtype == np.bool_


def test_future_and_target_labels_cannot_contaminate_inference():
    visits, targets = frames()
    pre = data.fit_preprocessor(visits, targets, ["001"])
    expected = data.build_samples(visits, targets, pre, include_labels=False)[0]
    visits = pd.concat(
        [visits, pd.DataFrame([visit(identity="future", date="2021-01-01", score=30, action_1=999)])],
        ignore_index=True,
    )
    targets["target_score"] = ["not a number; never read this during inference"]
    actual = data.build_samples(visits, targets, pre, include_labels=False)[0]
    for key in ["x", "missing", "dt_days"]:
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual["baseline"] == expected["baseline"]
    assert "target_score" not in actual
    targets = targets.drop(columns="target_score")
    assert len(data.build_samples(visits, targets, pre, include_labels=False)) == 1
    with pytest.raises(ValueError, match="label|target_score"):
        data.build_samples(visits, targets, pre)


def test_delayed_scores_are_masked_and_cannot_change_stats_or_baseline():
    visits, targets = frames()
    visits.loc[1, "score_available_date"] = "2020-02-01"
    pre = data.fit_preprocessor(visits, targets, ["001"])
    assert pre.means[6] == 10.0
    sample = data.build_samples(visits, targets, pre)[0]
    assert sample["baseline"] == 10.0
    assert sample["missing"][:, 6].tolist() == [False, True]
    assert sample["x"][1, 6] == 0.0
    visits.loc[1, "score"] = 29.0
    changed = data.build_samples(visits, targets, pre)[0]
    np.testing.assert_array_equal(sample["x"], changed["x"])
    visits.loc[0, "score_available_date"] = "2020-02-01"
    with pytest.raises(ValueError, match="available score"):
        data.build_samples(visits, targets, pre)


def test_missing_is_distinct_from_zero_and_unlabeled_tables_are_valid(tmp_path):
    visits, targets = frames()
    visits.loc[1, "action_1"] = np.nan
    targets = targets.drop(columns="target_score")
    write_tables(tmp_path, visits, targets)
    assert not data.check_data(tmp_path)["errors"]
    pre = data.fit_preprocessor(visits, targets, ["001"])
    sample = data.build_samples(visits, targets, pre, include_labels=False)[0]
    assert sample["missing"][:, 0].tolist() == [False, True]
    assert sample["x"][:, 0].tolist() == [0.0, 0.0]


def test_later_origins_fit_unique_known_observations_without_horizon_weighting():
    visits, targets = frames()
    targets = pd.concat([targets, pd.DataFrame([target(origin="v1", date="2020-04-01")])])
    pre = data.fit_preprocessor(visits, targets, ["001"])
    assert pre.means[6] == 15.0
    assert pre.scales[6] == 5.0
    assert len(data.build_samples(visits, targets, pre)) == 2


def test_templates_are_valid_unlabeled_safe_metadata():
    report = data.check_data(Path(__file__).resolve().parents[1] / "examples" / "templates")
    assert report["errors"] == []
    assert report["counts"]["visits"] > 0


@pytest.mark.parametrize(
    "field,value", [("scales", [0.0] * 14), ("means", [0.0]), ("means", [float("nan")] * 14)]
)
def test_preprocessor_serde_rejects_invalid_state(field, value):
    visits, targets = frames()
    payload = data.fit_preprocessor(visits, targets, ["001"]).to_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        data.Preprocessor.from_dict(payload)


def test_prepare_protects_existing_output_and_check_does_not_mutate(tmp_path):
    visits, targets = frames()
    write_tables(tmp_path / "raw", visits, targets)
    original = (tmp_path / "raw" / "visits.csv").read_bytes()
    assert not data.check_data(tmp_path / "raw")["errors"]
    assert (tmp_path / "raw" / "visits.csv").read_bytes() == original
    data.prepare_data(tmp_path / "raw", tmp_path / "out")
    with pytest.raises(ValueError, match="empty"):
        data.prepare_data(tmp_path / "raw", tmp_path / "out")


def test_origin_day_other_sessions_are_excluded_without_inventing_chronology():
    visits, targets = frames()
    other = visit(identity="a_earlier_lexically", date="2020-01-11", score=29, action_1=999)
    other["session_id"] = "s0"
    visits = pd.concat([visits, pd.DataFrame([other])], ignore_index=True)
    pre = data.fit_preprocessor(visits, targets, ["001"])
    assert pre.means[0] == 1.0
    sample = data.build_samples(visits, targets, pre)[0]
    assert len(sample["x"]) == 2
    assert sample["baseline"] == 20.0
