"""Kiểm thử nhanh cho GĐ1/GĐ1b. Chạy: pytest -q"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import load_config  # noqa: E402
from s1_data import EXCLUDE_FROM_MODEL, FEATURES_CAT, FEATURES_NUM, snapshot_schema  # noqa: E402


def test_leakage_tests_all_pass():
    t = pd.read_csv(ROOT / "outputs/tables/s1_leakage_tests.csv")
    assert t["pass"].all(), t[~t["pass"]]


def test_snapshot_contract():
    snap = pd.read_parquet(ROOT / "data/processed/snapshots.parquet")
    snapshot_schema(load_config()).validate(snap, lazy=True)


def test_no_label_or_key_in_features():
    assert not set(EXCLUDE_FROM_MODEL) & set(FEATURES_NUM + FEATURES_CAT)


def test_logged_has_no_true_y0():
    cols = pd.read_parquet(ROOT / "data/processed/logged.parquet").columns
    assert "y" not in cols and not any(c.startswith(("y_a", "profit_a", "tau_")) for c in cols)


def test_oracle_not_read_in_training_code():
    """Luật cứng #1: chỉ hàm evaluate_/oracle_ được đọc oracle.parquet."""
    for f in (ROOT / "src").glob("s[2-8]*.py"):
        lines = f.read_text(encoding="utf-8").splitlines()
        func = ""
        for ln in lines:
            s = ln.strip()
            if s.startswith("def "):
                func = s[4:].split("(")[0]
            if "oracle.parquet" in s and not s.startswith("#"):
                assert func.startswith(("evaluate_", "oracle_")), f"{f.name}: đọc oracle trong {func or 'module'}"
