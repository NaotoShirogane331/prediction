"""Deterministic fictional records for exercising the pipeline, not clinical data."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import DEFAULT_SEED, FEATURE_COLUMNS, HORIZONS


def generate_data(output: Path, n_people: int = 120, seed: int = DEFAULT_SEED) -> dict:
    if not isinstance(n_people, int) or isinstance(n_people, bool) or n_people < 10:
        raise ValueError("n_people must be an integer >= 10")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    rng = np.random.default_rng(seed)
    visits, assets, targets = [], [], []
    for index in range(n_people):
        person_id = f"{index + 1:04d}"
        n_visits = int(rng.integers(6, 11))
        intervals = np.r_[0, rng.integers(25, 125, size=n_visits - 1)]
        offsets = intervals.cumsum()
        start = date(2020, 1, 1) + timedelta(days=int(rng.integers(0, 366)))
        level = float(rng.uniform(15, 27))
        annual_trend = float(rng.normal(-0.7, 0.6))
        age, education = float(rng.integers(50, 86)), float(rng.integers(6, 21))
        background = [age, education, float(rng.integers(0, 2))]
        condition = rng.uniform(0, 1, size=2)
        for visit_index, offset in enumerate(offsets):
            visit_date = start + timedelta(days=int(offset))
            latent = level + annual_trend * float(offset) / 365
            score = float(np.clip(latent + rng.normal(0, 0.65), 0, 30))
            identity = f"v{visit_index + 1:03d}"
            actions = rng.normal(latent / 30, 0.15, size=6)
            assessments = [float(rng.normal(latent / 30, 0.10)), float(rng.normal(latent / 15, 0.15))]
            values = [
                *actions,
                score,
                *assessments,
                *background,
                *np.clip(condition + rng.normal(0, 0.05, size=2), 0, 1),
            ]
            # Always retain the first score so every synthetic origin has a baseline.
            missing = rng.random(len(FEATURE_COLUMNS)) < 0.06
            if visit_index == 0:
                missing[6] = False
            record = dict(
                zip(
                    FEATURE_COLUMNS,
                    [None if absent else round(float(value), 6) for value, absent in zip(values, missing)],
                )
            )
            available = visit_date + timedelta(days=int(rng.choice([0, 0, 0, 2, 7])))
            record.update(
                person_id=person_id,
                visit_id=identity,
                visit_date=visit_date.isoformat(),
                session_id="session_1",
                score_available_date="" if missing[6] else available.isoformat(),
            )
            visits.append(record)
            for modality, extension in [("audio", "wav"), ("video", "mp4")]:
                assets.append(
                    {
                        "person_id": person_id,
                        "visit_id": identity,
                        "modality": modality,
                        "path": f"fictional/{person_id}/{identity}/{modality}.{extension}",
                    }
                )
        origin = start + timedelta(days=int(offsets[-1]))
        for horizon in HORIZONS:
            score = level + annual_trend * (float(offsets[-1]) + horizon) / 365 + rng.normal(0, 0.8)
            targets.append(
                {
                    "person_id": person_id,
                    "origin_visit_id": f"v{n_visits:03d}",
                    "target_date": (origin + timedelta(days=horizon)).isoformat(),
                    "target_score": round(float(np.clip(score, 0, 30)), 6),
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    columns = ["person_id", "visit_id", "visit_date", "session_id", "score_available_date", *FEATURE_COLUMNS]
    pd.DataFrame(visits)[columns].to_csv(output / "visits.csv", index=False)
    pd.DataFrame(assets).to_csv(output / "assets.csv", index=False)
    pd.DataFrame(targets).to_csv(output / "targets.csv", index=False)
    provenance = {
        "synthetic": True,
        "seed": seed,
        "n_people": n_people,
        "scale": "synthetic_demo_0_30",
        "description": "Entirely fictional latent linear trends and noisy features; "
        "not MMSE, MoCA, clinical evidence, or real media.",
        "assets": "Fictional metadata only; no media files are created.",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return {"people": n_people, "visits": len(visits), "assets": len(assets), "targets": len(targets)}
