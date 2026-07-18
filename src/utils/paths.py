"""Project path resolution and experiment constants.

The project root is the first ancestor directory that contains
``data/raw/agregatenew.csv``. This makes the modules work both from
the repository itself (once the raw data is copied into its
``data/raw``) and from the original ``Data`` folder layout.
"""

from __future__ import annotations

from pathlib import Path

# ============================================================
# EXPERIMENT CONSTANTS (shared across all models)
# ============================================================

CONTEXT_LENGTH = 168
FORECAST_HORIZON = 24

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

RAW_WIND_FILENAME = "agregatenew.csv"
RAW_OPERATIONAL_FILENAME = "Renewable_energy_dataset.csv"


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from `start` (default: cwd) to locate the data
    root.

    Preference order:
    1. a folder containing the confidential raw dataset
       (``data/raw/agregatenew.csv``) — the full original layout;
    2. a folder containing the distributed derived inputs
       (``data/optimization_input/pgu_parameters.csv``) — a GitHub
       clone of this repository, where the raw data is absent but
       the entire optimization study remains reproducible.
    """
    current = (start or Path.cwd()).resolve()
    candidates = [current, *current.parents]

    for candidate in candidates:
        if (candidate / "data" / "raw" / RAW_WIND_FILENAME).exists():
            return candidate
    for candidate in candidates:
        if (candidate / "data" / "optimization_input"
                / "pgu_parameters.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the project root: no ancestor of "
        f"{current} contains data/raw/{RAW_WIND_FILENAME} or "
        "data/optimization_input/pgu_parameters.csv."
    )


def get_paths(start: Path | None = None) -> dict[str, Path]:
    """Return the standard project directories as a dictionary."""
    root = find_project_root(start)
    paths = {
        "PROJECT_ROOT": root,
        "RAW_DIR": root / "data" / "raw",
        "PROCESSED_DIR": root / "data" / "processed",
        "OPT_INPUT_DIR": root / "data" / "optimization_input",
        "OPT_OUTPUT_DIR": root / "data" / "optimization_output",
        "MODELS_DIR": root / "models",
        "FIGURES_DIR": root / "figures",
    }
    return paths
