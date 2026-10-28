"""Versioned feature order shared by data preparation and the model."""

SCHEMA_VERSION = 1
FEATURE_GROUPS = {
    "action": [f"action_{i}" for i in range(1, 7)],
    "assessment": ["score", "assessment_1", "assessment_2"],
    "background": [f"background_{i}" for i in range(1, 4)],
    "condition": [f"condition_{i}" for i in range(1, 3)],
}
FEATURE_COLUMNS = [column for group in FEATURE_GROUPS.values() for column in group]
GROUP_DIMS = tuple(map(len, FEATURE_GROUPS.values()))
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
QUANTILE_COLUMNS = ("q05", "q25", "q50", "q75", "q95")
HORIZONS = (90, 180, 365)
DEFAULT_SEED = 2026
