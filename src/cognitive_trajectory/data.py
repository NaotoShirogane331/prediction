"""CSV validation and origin-aware preprocessing for the fictional research demo."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd

from .schema import DEFAULT_SEED, FEATURE_COLUMNS, SCHEMA_VERSION

VISIT_COLUMNS = [
    "person_id",
    "visit_id",
    "visit_date",
    "session_id",
    "score_available_date",
    *FEATURE_COLUMNS,
]
ASSET_COLUMNS = ["person_id", "visit_id", "modality", "path"]
TARGET_COLUMNS = ["person_id", "origin_visit_id", "target_date"]
SCORE_INDEX = FEATURE_COLUMNS.index("score")


def _required(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")


def _strings(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    for column in columns:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"{name}: {column} must not be empty")


def _date(value: object, name: str, allow_empty: bool = False) -> str:
    if pd.isna(value) or str(value).strip() == "":
        if allow_empty:
            return ""
        raise ValueError(f"{name}: date must not be empty")
    value = str(value).strip()
    if not re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", value):
        raise ValueError(f"{name}: expected an ISO date, got {value!r}")
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except ValueError as error:
        raise ValueError(f"{name}: invalid date {value!r}") from error


def _numeric(series: pd.Series, name: str) -> pd.Series:
    blank = series.isna() | series.astype(str).str.strip().eq("")
    converted = pd.to_numeric(series.where(~blank, np.nan), errors="coerce").astype(float)
    if ((~blank & converted.isna()) | (converted.notna() & ~np.isfinite(converted))).any():
        raise ValueError(f"{name}: entries must be finite numbers or blank")
    return converted


def load_tables(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read CSVs without coercing identifiers or interpreting target labels.

    Target values are deliberately left untouched: an inference caller can use an
    unlabeled table, or a table containing arbitrary labels that it never reads.
    """
    directory = Path(directory)
    tables = []
    for name in ["visits", "assets", "targets"]:
        path = directory / f"{name}.csv"
        if not path.is_file():
            raise ValueError(f"Missing table: {path.name}")
        try:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
            raise ValueError(f"{path.name}: {error}") from error
        tables.append(frame)
    visits, assets, targets = tables
    for column in FEATURE_COLUMNS:
        if column in visits:
            visits[column] = _numeric(visits[column], f"visits.{column}")
    return visits, assets, targets


def _visits(frame: pd.DataFrame) -> pd.DataFrame:
    _required(frame, VISIT_COLUMNS, "visits")
    frame = frame.copy()
    _strings(frame, ["person_id", "visit_id", "session_id"], "visits")
    if frame.empty:
        raise ValueError("visits: at least one visit is required")
    frame["visit_date"] = frame["visit_date"].map(lambda x: _date(x, "visit_date"))
    frame["score_available_date"] = frame["score_available_date"].map(
        lambda x: _date(x, "score_available_date", allow_empty=True)
    )
    for column in FEATURE_COLUMNS:
        frame[column] = _numeric(frame[column], f"visits.{column}")
    if frame.duplicated(["person_id", "visit_id"]).any():
        raise ValueError("visits: duplicate person_id/visit_id")
    if frame.duplicated(["person_id", "visit_date", "session_id"]).any():
        raise ValueError("visits: duplicate same-day session; provide distinct session_id values")
    present = frame["score"].notna()
    if (~frame.loc[present, "score"].between(0, 30)).any():
        raise ValueError("visits: score must be within the fictional range 0..30")
    if (present & frame["score_available_date"].eq("")).any():
        raise ValueError("visits: present score requires score_available_date")
    if (~present & frame["score_available_date"].ne("")).any():
        raise ValueError("visits: missing score requires empty score_available_date")
    if (present & (frame["score_available_date"] < frame["visit_date"])).any():
        raise ValueError("visits: score_available_date must be on or after visit_date")
    return frame.sort_values(["person_id", "visit_date", "session_id", "visit_id"]).reset_index(drop=True)


def _targets(frame: pd.DataFrame, visits: pd.DataFrame, validate_labels: bool) -> pd.DataFrame:
    _required(frame, TARGET_COLUMNS, "targets")
    frame = frame.copy()
    _strings(frame, ["person_id", "origin_visit_id"], "targets")
    frame["target_date"] = frame["target_date"].map(lambda x: _date(x, "target_date"))
    if frame.duplicated(TARGET_COLUMNS).any():
        raise ValueError("targets: duplicate person/origin/target_date")
    origins = visits.set_index(["person_id", "visit_id"])["visit_date"]
    for row in frame[TARGET_COLUMNS].itertuples(index=False):
        key = (row.person_id, row.origin_visit_id)
        if key not in origins.index:
            raise ValueError(f"targets: unknown origin {key}")
        if row.target_date <= origins.loc[key]:
            raise ValueError(f"targets: target_date must be after origin date for {key}")
    if validate_labels and "target_score" in frame:
        frame["target_score"] = _numeric(frame["target_score"], "targets.target_score")
        labels = frame["target_score"].dropna()
        if (~labels.between(0, 30)).any():
            raise ValueError("targets: target_score must be within the fictional range 0..30")
    return frame.sort_values(TARGET_COLUMNS).reset_index(drop=True)


def _safe_path(value: str) -> bool:
    return bool(
        value
        and "\\" not in value
        and "\x00" not in value
        and not PurePosixPath(value).is_absolute()
        and not PureWindowsPath(value).is_absolute()
        and not PureWindowsPath(value).drive
        and ".." not in PurePosixPath(value).parts
        and PurePosixPath(value).parts
    )


def _assets(frame: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    _required(frame, ["person_id", "modality", "path"], "assets")
    frame = frame.copy()
    _strings(frame, ["person_id", "modality", "path"], "assets")
    if "visit_id" not in frame:
        frame["visit_id"] = ""
    frame["visit_id"] = frame["visit_id"].fillna("").astype(str).str.strip()
    for index, row in frame.iterrows():
        if not _safe_path(row["path"]):
            raise ValueError(f"assets: unsafe relative path {row['path']!r}")
        candidates = visits.loc[visits["person_id"].eq(row["person_id"])]
        if row["visit_id"]:
            candidates = candidates.loc[candidates["visit_id"].eq(row["visit_id"])]
        elif "visit_date" not in frame or pd.isna(row["visit_date"]) or not str(row["visit_date"]).strip():
            raise ValueError("assets: missing visit_id requires visit_date for an exact match")
        if "visit_date" in frame and pd.notna(row["visit_date"]) and str(row["visit_date"]).strip():
            date = _date(row["visit_date"], "assets.visit_date")
            frame.loc[index, "visit_date"] = date
            candidates = candidates.loc[candidates["visit_date"].eq(date)]
        if "session_id" in frame and pd.notna(row["session_id"]) and str(row["session_id"]).strip():
            candidates = candidates.loc[candidates["session_id"].eq(str(row["session_id"]).strip())]
        if len(candidates) == 0:
            raise ValueError(f"assets: no matching visit for row {index}")
        if len(candidates) > 1:
            raise ValueError(f"assets: ambiguous visit match for row {index}; provide session_id")
        frame.loc[index, "visit_id"] = candidates.iloc[0]["visit_id"]
    if frame.duplicated(["person_id", "visit_id", "modality", "path"]).any():
        raise ValueError("assets: duplicate index entry")
    return frame.sort_values(ASSET_COLUMNS).reset_index(drop=True)


def _report(visits, assets, targets, directory: Path) -> dict:
    found = 0
    for value in assets["path"]:
        candidate = directory / value
        # A symlink escaping the input directory is not an indexed local asset.
        if candidate.resolve().is_relative_to(directory.resolve()) and candidate.is_file():
            found += 1
    missing = len(assets) - found
    return {
        "errors": [],
        "warnings": (
            [(f"{missing} indexed asset files do not exist locally; index validity is checked separately.")]
            if missing
            else []
        ),
        "counts": {
            "people": int(visits["person_id"].nunique()),
            "visits": len(visits),
            "assets": len(assets),
            "targets": len(targets),
            "asset_files_found": found,
            "asset_files_missing": missing,
        },
    }


def check_data(directory: Path) -> dict:
    """Return structural errors separately from missing-media warnings; never write."""
    directory = Path(directory)
    try:
        visits, assets, targets = load_tables(directory)
        visits = _visits(visits)
        targets = _targets(targets, visits, validate_labels=True)
        assets = _assets(assets, visits)
        return _report(visits, assets, targets, directory)
    except (ValueError, OSError) as error:
        return {"errors": [str(error)], "warnings": [], "counts": {}}


def prepare_data(source: Path, output: Path) -> dict:
    """Validate before writing; never guess a join or overwrite existing output."""
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    visits, assets, targets = load_tables(source)
    visits = _visits(visits)
    targets = _targets(targets, visits, validate_labels=True)
    assets = _assets(assets, visits)
    report = _report(visits, assets, targets, source)
    output.mkdir(parents=True, exist_ok=True)
    visits.to_csv(output / "visits.csv", index=False)
    assets.to_csv(output / "assets.csv", index=False)
    targets.to_csv(output / "targets.csv", index=False)
    if (source / "provenance.json").is_file():
        shutil.copyfile(source / "provenance.json", output / "provenance.json")
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def make_splits(visits: pd.DataFrame, seed: int = DEFAULT_SEED) -> dict[str, list[str]]:
    ids = sorted(visits["person_id"].astype(str).unique().tolist())
    if len(ids) < 3:
        raise ValueError("At least three people are required for disjoint splits")
    shuffled = np.random.default_rng(seed).permutation(ids).tolist()
    n_train = min(len(ids) - 2, max(1, int(len(ids) * 0.70)))
    n_validation = max(1, int(len(ids) * 0.15))
    return {
        "train": sorted(shuffled[:n_train]),
        "validation": sorted(shuffled[n_train : n_train + n_validation]),
        "test": sorted(shuffled[n_train + n_validation :]),
    }


@dataclass
class Preprocessor:
    means: np.ndarray
    scales: np.ndarray

    def __post_init__(self):
        self.means = np.asarray(self.means, dtype=np.float64)
        self.scales = np.asarray(self.scales, dtype=np.float64)
        shape = (len(FEATURE_COLUMNS),)
        if (
            self.means.shape != shape
            or self.scales.shape != shape
            or not np.isfinite(self.means).all()
            or not np.isfinite(self.scales).all()
            or (self.scales <= 0).any()
        ):
            raise ValueError("Preprocessor needs 14 finite means and positive finite scales")

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "feature_columns": list(FEATURE_COLUMNS),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Preprocessor:
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("feature_columns") != FEATURE_COLUMNS
        ):
            raise ValueError("Preprocessor schema/version does not match this package")
        return cls(payload["means"], payload["scales"])


def _history(visits: pd.DataFrame, person_id: str, origin_id: str) -> tuple[pd.DataFrame, str]:
    person = visits.loc[visits["person_id"].eq(person_id)]
    origin = person.loc[person["visit_id"].eq(origin_id)]
    if len(origin) != 1:
        raise ValueError(f"Unknown or duplicate origin {(person_id, origin_id)}")
    date = origin.iloc[0]["visit_date"]
    # Date-only data cannot establish the ordering of other origin-day sessions.
    history = person.loc[person["visit_date"].lt(date) | person["visit_id"].eq(origin_id)].copy()
    return history, date


def _known_values(history: pd.DataFrame, origin_date: str) -> np.ndarray:
    values = history[FEATURE_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    unavailable = history["score_available_date"].eq("") | history["score_available_date"].gt(origin_date)
    values[unavailable.to_numpy(), SCORE_INDEX] = np.nan
    return values


def fit_preprocessor(visits: pd.DataFrame, targets: pd.DataFrame, train_ids) -> Preprocessor:
    """Fit unique train observations known at at least one training origin."""
    visits = _visits(visits)
    targets = _targets(targets.drop(columns=["target_score"], errors="ignore"), visits, False)
    train_ids = set(map(str, train_ids))
    train_visits = visits.loc[visits["person_id"].isin(train_ids)]
    known = np.zeros((len(visits), len(FEATURE_COLUMNS)), dtype=bool)
    for row in targets.loc[targets["person_id"].isin(train_ids)].itertuples(index=False):
        history, date = _history(train_visits, row.person_id, row.origin_visit_id)
        known[history.index] |= np.isfinite(_known_values(history, date))
    values = visits[FEATURE_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    values[~known] = np.nan
    if not np.isfinite(values).any():
        raise ValueError("No known training observations at the supplied origins")
    counts = np.isfinite(values).sum(axis=0)
    means = np.divide(np.nansum(values, axis=0), counts, out=np.zeros(len(FEATURE_COLUMNS)), where=counts > 0)
    squared = np.nansum((values - means) ** 2, axis=0)
    scales = np.sqrt(np.divide(squared, counts, out=np.zeros(len(FEATURE_COLUMNS)), where=counts > 0))
    scales[scales < 1e-8] = 1.0
    return Preprocessor(means, scales)


def build_samples(
    visits: pd.DataFrame,
    targets: pd.DataFrame,
    preprocessor: Preprocessor,
    person_ids=None,
    include_labels: bool = True,
) -> list[dict]:
    """Build each horizon independently, dropping labels before any inference work."""
    if not include_labels:
        targets = targets.drop(columns=["target_score"], errors="ignore")
    visits = _visits(visits)
    targets = _targets(targets, visits, validate_labels=include_labels)
    if include_labels and ("target_score" not in targets or targets["target_score"].isna().any()):
        raise ValueError("Labeled samples require a finite target_score for every target")
    if person_ids is not None:
        targets = targets.loc[targets["person_id"].isin(set(map(str, person_ids)))]
    samples = []
    for row in targets.itertuples(index=False):
        history, origin_date = _history(visits, row.person_id, row.origin_visit_id)
        values = _known_values(history, origin_date)
        available_scores = values[:, SCORE_INDEX]
        available_scores = available_scores[np.isfinite(available_scores)]
        if not len(available_scores):
            raise ValueError(f"No available score at origin {(row.person_id, row.origin_visit_id)}")
        missing = np.isnan(values)
        normalized = (values - preprocessor.means) / preprocessor.scales
        normalized[missing] = 0.0
        dates = pd.to_datetime(history["visit_date"])
        dt_days = dates.diff().dt.days.fillna(0).to_numpy(dtype=np.float32)
        sample = {
            "x": normalized.astype(np.float32),
            "missing": missing.astype(np.bool_),
            "dt_days": dt_days,
            "baseline": float(available_scores[-1]),
            "horizon_days": float((pd.Timestamp(row.target_date) - pd.Timestamp(origin_date)).days),
            "person_id": row.person_id,
            "origin_visit_id": row.origin_visit_id,
            "origin_date": origin_date,
            "target_date": row.target_date,
        }
        if include_labels:
            sample["target_score"] = float(row.target_score)
        samples.append(sample)
    return samples
