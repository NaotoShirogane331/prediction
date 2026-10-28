"""Plot saved prediction values without recomputing or beautifying forecasts."""

import json
import os
from pathlib import Path

import pandas as pd


def plot_trajectory(data_dir: Path, predictions_path: Path, output: Path) -> None:
    # Keep Matplotlib's writable cache inside the user's selected output directory.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(output).parent / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    from .data import load_tables

    visits, _, _ = load_tables(Path(data_dir))
    predictions = pd.read_csv(predictions_path, dtype={"person_id": str, "origin_visit_id": str})
    first = predictions.iloc[0]
    person, origin_id = first["person_id"], first["origin_visit_id"]
    future = predictions[
        (predictions.person_id == person) & (predictions.origin_visit_id == origin_id)
    ].sort_values("target_date")
    origin_date = pd.Timestamp(first["origin_date"])
    dates = pd.to_datetime(visits["visit_date"])
    available = pd.to_datetime(visits["score_available_date"], errors="coerce")
    hist = visits[
        (visits.person_id == person)
        & ((dates < origin_date) | (visits.visit_id == origin_id))
        & (available <= origin_date)
    ].copy()
    hist["score"] = pd.to_numeric(hist["score"], errors="coerce")
    hist = hist.dropna(subset=["score"]).sort_values("visit_date")
    x = pd.to_datetime(future["target_date"])
    fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
    ax.plot(pd.to_datetime(hist.visit_date), hist.score, "o-", color="#2457bf", label="Available history")
    ax.fill_between(x, future.q05, future.q95, color="#4f86f7", alpha=0.15, label="5–95% quantiles")
    ax.fill_between(x, future.q25, future.q75, color="#4f86f7", alpha=0.25, label="25–75% quantiles")
    ax.plot(x, future.q50, "o-", color="#ef6c35", label="Predicted median")
    ax.axhline(float(first["baseline"]), color="#7a8595", ls=":", lw=1.4, label="Last-score baseline")
    ax.axvline(origin_date, color="#8993a4", ls="--", lw=1)
    marker = Path(data_dir) / "provenance.json"
    synthetic = marker.exists() and json.loads(marker.read_text()).get("synthetic") is True
    ax.set_title(
        f"{'SYNTHETIC DEMO' if synthetic else 'RESEARCH OUTPUT'} | {person}\n"
        "Quantile predictions — empirical coverage is reported separately",
        loc="left",
        fontsize=13,
    )
    ax.set_ylabel("Synthetic demonstration score" if synthetic else "Configured score")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(output, dpi=160)
    plt.close(fig)
