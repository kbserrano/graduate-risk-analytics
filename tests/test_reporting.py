"""
Tests for src/reporting.py — output paths must resolve to the project's
reports/ directory regardless of the working directory (notebooks run
from notebooks/, tests and scripts from the project root).

Run with: pytest tests/test_reporting.py -v
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import reporting
from src.reporting import save_figure, save_table, save_json


def test_report_dirs_are_anchored_to_project_root():
    assert reporting.PROJECT_ROOT == PROJECT_ROOT.resolve()
    assert reporting.FIGURES_DIR == PROJECT_ROOT.resolve() / "reports" / "figures"
    assert reporting.TABLES_DIR == PROJECT_ROOT.resolve() / "reports" / "tables"


def test_report_dirs_ignore_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert reporting.FIGURES_DIR.is_absolute()
    assert reporting.FIGURES_DIR.parent.parent == PROJECT_ROOT.resolve()


def test_save_figure_writes_png(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = save_figure(fig, "example", directory=tmp_path / "figs")
    plt.close(fig)

    assert path == tmp_path / "figs" / "example.png"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_save_table_writes_csv(tmp_path):
    df = pd.DataFrame({"pr_auc": [0.4, 0.3]}, index=["A1", "B1"])

    path = save_table(df, "summary", directory=tmp_path)
    assert path == tmp_path / "summary.csv"
    assert pd.read_csv(path, index_col=0).equals(df)

    path = save_table(df.reset_index(), "flat", directory=tmp_path, index=False)
    assert list(pd.read_csv(path).columns) == ["index", "pr_auc"]


def test_save_json_handles_numpy_scalars(tmp_path):
    result = {"statistic": np.float64(36.4), "significant": np.bool_(True)}

    path = save_json(result, "friedman", directory=tmp_path)
    assert path == tmp_path / "friedman.json"
    assert json.loads(path.read_text()) == {"statistic": 36.4, "significant": True}
