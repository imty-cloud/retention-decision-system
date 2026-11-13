"""Kiểm thử GĐ9 (Hillstrom). Chạy: .venv/bin/python -m pytest -q tests/test_s9.py"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import common  # noqa: E402
import s9_external_hillstrom as s9  # noqa: E402

TABLES = ROOT / "outputs/tables"


@pytest.fixture(scope="module")
def df():
    d, f = s9.load_hillstrom()
    d.attrs["file"] = f
    return d


def test_raw_file_hash_matches_recorded(df):
    rec = pd.read_csv(TABLES / "s9_data_sha256.csv").iloc[0]
    assert common.sha256_file(df.attrs["file"]) == rec.sha256
    assert rec.so_dong == len(df) == 64000


def test_split_stratified_deterministic_disjoint(df):
    a1, b1 = s9.split_train_test(df)
    a2, b2 = s9.split_train_test(df)
    assert (a1 == a2).all() and (b1 == b2).all()
    assert len(np.intersect1d(a1, b1)) == 0 and len(a1) + len(b1) == len(df)
    assert abs(len(b1) / len(df) - 0.30) < 0.001
    tr, te = df.iloc[a1], df.iloc[b1]
    for a in s9.ARMS:
        assert abs((tr.A == a).mean() - (df.A == a).mean()) < 0.001
        assert abs((te.A == a).mean() - (df.A == a).mean()) < 0.001


def test_known_propensity_and_covariate_balance(df):
    shares = df.A.value_counts(normalize=True)
    assert (abs(shares - s9.H["propensity"]) < 0.01).all()
    smd, tests = s9.balance_table(df)
    assert smd.SMD.abs().max() < 0.05
    assert tests.p_value.min() > 0.001


def test_no_test_outcome_in_fit_inputs(df):
    """Mô hình chỉ được khớp trên train: vector quyết định chọn learner không dùng chỉ số hàng của test."""
    tr_idx, te_idx = s9.split_train_test(df)
    X_tr = s9.design_matrix(df.iloc[tr_idx])
    X_te = s9.design_matrix(df.iloc[te_idx], list(X_tr.columns))
    assert list(X_tr.columns) == list(X_te.columns)
    assert not set(s9.H["outcomes"]) & set(X_tr.columns) and "A" not in X_tr.columns


def test_milp_matches_bruteforce_and_greedy():
    rng = np.random.default_rng(0)
    V = rng.normal(0, 1, (7, 2))
    cap = 3
    best = -1e9
    for pol in itertools.product([0, 1, 2], repeat=7):
        pol = np.array(pol)
        if (pol > 0).sum() <= cap:
            best = max(best, s9.value_vec(V, pol).sum())
    pm, info = s9.policy_milp(V, cap, time_limit=10)
    pg = s9.policy_greedy(V, cap)
    assert info["trang_thai"] == "Optimal"
    assert (pm > 0).sum() <= cap and (pg > 0).sum() <= cap
    assert s9.value_vec(V, pm).sum() == pytest.approx(best)
    assert s9.value_vec(V, pg).sum() == pytest.approx(best)


def test_ipw_is_unbiased_on_synthetic_rct():
    rng = np.random.default_rng(1)
    n = 300_000
    A = rng.integers(0, 3, n)
    tau = np.array([0.0, 2.0, -1.0])
    spend = rng.normal(5, 1, n) + tau[A]
    pol = np.ones(n, dtype=int)                                    # gửi email nam cho tất cả
    est = s9.ipw_contrib(pol, A, spend, 0.3, 0.1).mean()
    assert est == pytest.approx(0.3 * (5 + 2.0) - 0.1, abs=0.03)
    est0 = s9.ipw_contrib(np.zeros(n, dtype=int), A, spend, 0.3, 0.1).mean()
    assert est0 == pytest.approx(0.3 * 5, abs=0.03)


def test_no_oracle_word_in_s9_source():
    assert "oracle" not in (ROOT / "src/s9_external_hillstrom.py").read_text(encoding="utf-8").lower()


def test_outputs_policy_respects_send_cap_and_manifest():
    pol = pd.read_csv(TABLES / "s9_policy_value_test.csv")
    ctrl = pol[pol.chinh_sach.str.startswith(("MILP", "Greedy", "Gửi ngẫu nhiên"))]
    assert (ctrl.ty_le_gui <= s9.H["max_send_share"] + 1e-9).all()
    assert (pol[pol.chinh_sach == "Không gửi"].ty_le_gui == 0).all()
    assert (ROOT / "outputs/manifests/s9_manifest.json").exists()
    ate = pd.read_csv(TABLES / "s9_ate_rct.csv")
    assert set(ate.outcome) == {"conversion", "visit", "spend"}
