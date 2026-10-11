"""
Save figures and tables to reports/.

Paths are anchored to this file's location rather than the working
directory, so notebooks (run from notebooks/), tests, and scripts (run
from the project root) all write to the same reports/figures and
reports/tables.
"""

import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"


def save_figure(fig, name, directory=FIGURES_DIR, dpi=300):
    """Save fig as <directory>/<name>.png and return the path."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_table(df, name, directory=TABLES_DIR, index=True):
    """Save df as <directory>/<name>.csv and return the path."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    df.to_csv(path, index=index)
    return path


def save_json(obj, name, directory=TABLES_DIR):
    """Save obj as <directory>/<name>.json and return the path.

    numpy scalars (e.g. from scipy.stats results) are converted to
    plain Python values so they serialize.
    """

    def to_builtin(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Not JSON serializable: {type(value).__name__}")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=to_builtin) + "\n")
    return path
