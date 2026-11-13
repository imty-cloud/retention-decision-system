"""Kiểm thử Router v2.1: bước sửa khả thi cuối (trần a2 ≤ 25 % số voucher phát) + kiểm lại độ phủ."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import s7_router as R  # noqa: E402
from test_s7 import make_period, proposal  # noqa: E402

T = ROOT / "outputs/tables"
CAP = {"a0": 0, "a1": 30, "a2": 60, "a3": 30}


def r21():
    return R.ActionRouterV2(1e9, "test-model", R.POLICY_VERSION_V21)


def ins_for(d, B=1e9, K=60):
    return {"is_uk": np.arange(len(d)) < int(0.9 * len(d)), "B": B, "K": K}


def a2_share(out):
    nv = (out.final_action != "a0").sum()
    return (out.final_action == "a2").sum() / max(nv, 1)


def test_repair_enforces_a2_cap_downgrading_lowest_vhat():
    d = make_period(n=400, seed=21)
    d["V_hat_a2"] = np.linspace(1, 100, len(d))                               # V̂(a2) tăng dần theo mã khách
    w = proposal(len(d), 100, action=2)                                       # 100 a2, 0 a1/a3 ⇒ vi phạm trần 25 % nặng
    out = r21().route(d, w, "valid", ins=ins_for(d))
    assert a2_share(out) <= 0.25 + 1e-9
    d_ = out.merge(d[["customer_id", "V_hat_a2"]], on="customer_id")
    kept = d_[(d_.proposed_action == "a2") & (d_.final_action == "a2")]
    moved = d_[d_.repair_status != "none"]
    assert len(moved) > 0 and kept.V_hat_a2.min() >= moved.V_hat_a2.max() - 1e-9   # chỉ hạ các a2 có V̂ thấp nhất
    assert (moved[moved.repair_status == "downgrade_a1"].final_action == "a1").all()       # V̂(a1)=10 > 0 ⇒ xuống a1
    assert (moved.router_reason_code.isin(["A2_CAP_REPAIR_A1", "A2_CAP_REPAIR_A0"])).all()


def test_repair_to_a0_when_vhat_a1_not_positive_and_not_backfilled():
    d = make_period(n=400, seed=22)
    d["V_hat_a1"] = -5.0                                                      # không có a1 dương ⇒ phải về a0
    r = r21()
    out = r.route(d, proposal(len(d), 60, action=2), "valid", ins=ins_for(d))
    assert (out.final_action == "a2").sum() == 0 or a2_share(out) <= 0.25 + 1e-9
    dropped = out[out.repair_status == "drop_a0"]
    assert len(dropped) > 0 and (dropped.final_action == "a0").all()
    assert (~dropped.in_randomization_set).all() and (~dropped.control_group).all()
    assert (dropped.propensity_a0 == 1.0).all() and (dropped.propensity_logged == 1.0).all()
    assert not dropped.is_backfill.any() and (out.is_backfill & out.customer_id.isin(dropped.customer_id)).sum() == 0
    assert r.last_repair["sua_a2_ve_a0"] == len(dropped)


def test_repair_noop_when_already_feasible_and_v20_unchanged():
    d = make_period(n=400, seed=23)
    w = proposal(len(d), 100, action=1)
    o21 = r21().route(d, w, "valid", ins=ins_for(d))
    assert (o21.repair_status == "none").all()
    o20 = R.ActionRouterV2(1e9, "test-model").route(d, w, "valid", ins=ins_for(d))
    assert "repair_status" not in o20.columns                                  # v2.0 không có bước sửa
    assert (o20.final_action.to_numpy() == o21.final_action.to_numpy()).all()


def test_budget_calls_a2_after_repair_and_coverage_backfill():
    d = make_period(n=600, seed=24)
    w = proposal(len(d), 60, action=2)                                         # 60 a2 = £3 600 ≤ B = £4 000 (đề xuất khả thi)
    w[60:70] = 0; w[60:70, 3] = 1                                             # 10 a3 = K
    r = r21()
    out = r.route(d, w, "valid", ins=ins_for(d, B=4000, K=10))
    assert out.final_action.map(CAP).sum() <= 4000 and (out.final_action == "a3").sum() <= 10
    assert a2_share(out) <= 0.25 + 1e-9
    b = out[out.is_backfill]
    assert (b.final_action == "a1").all() and (b.p_control == 0).all() and not b.control_group.any()


def test_repair_idempotent_and_schema():
    r = r21()
    d = make_period(n=400, seed=25)
    w = proposal(len(d), 100, action=2)
    o1 = r.route(d, w, "valid", ins=ins_for(d))
    n = len(r.log)
    o2 = r.route(d, w, "valid", ins=ins_for(d))
    assert len(r.log) == n and (o1.decision_id.to_numpy() == o2.decision_id.to_numpy()).all()
    R.DECISION_SCHEMA_V21.validate(o1)
    bad = o1.copy()
    i = bad.index[bad.repair_status == "drop_a0"]
    if len(i):
        bad.loc[i[0], "in_randomization_set"] = True
        with pytest.raises(Exception):
            R.DECISION_SCHEMA_V21.validate(bad)


# ---------------------------------------------------------------- kết quả thật
@pytest.fixture(scope="module")
def logs():
    return {"MILP": pd.read_parquet(ROOT / "data/processed/decisions_s7_v21.parquet"),
            "Luật cũ": pd.read_parquet(ROOT / "data/processed/decisions_s7_v21_legacy.parquet")}


def test_real_zero_violations_every_flow():
    a = pd.read_csv(T / "s7_constraint_audit_v21.csv")
    v = a[a.phien_ban == "v2.1"]
    assert len(v) == 14
    cols = ["vp_ngan_sach", "vp_cuoc_goi", "vp_ty_le_a2", "vp_do_phu_UK", "vp_do_phu_nonUK", "vp_gui_khach_bi_chan"]
    assert not v[cols].any().any(), v[v[cols].any(axis=1)]


def test_real_final_log_properties(logs):
    for name, d in logs.items():
        R.DECISION_SCHEMA_V21.validate(d)
        assert d.policy_version.str.startswith(R.POLICY_VERSION_V21).all() and set(d.split) <= {"valid", "calib"}
        g = d.groupby("t0")
        sent = d.final_action != "a0"
        assert (sent.groupby(d.t0).sum() * 0 + (d.final_action == "a2").groupby(d.t0).sum()
                <= 0.25 * sent.groupby(d.t0).sum() + 1e-9).all()
        assert (d.final_action.map(CAP).groupby(d.t0).sum() <= 8000 + 1e-6).all()
        assert ((d.final_action == "a3").groupby(d.t0).sum() <= 60).all()
        b = d[d.is_backfill]
        assert (b.p_control == 0).all() and not b.control_group.any()
        assert (d[d.repair_status == "drop_a0"].propensity_a0 == 1.0).all()
        s = d[sent].sort_values("t0")
        assert (s.groupby("customer_id").t0.diff().dropna() >= pd.Timedelta(days=R.FREQ_CAP_DAYS)).all()
        assert d.idempotency_key.is_unique


def test_v2_records_untouched():
    v2 = pd.read_parquet(ROOT / "data/processed/decisions_s7_v2.parquet")
    assert (v2.policy_version == R.POLICY_VERSION_V2).all() and "repair_status" not in v2.columns
