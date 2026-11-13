"""Kiểm thử GĐ7 (Router). Router được kiểm bằng dữ liệu tổng hợp (không cần mô hình); bản ghi thật kiểm theo file đầu ra."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import s7_router as R  # noqa: E402

T = ROOT / "outputs/tables"
TREATED = ["a1", "a2", "a3"]


def make_period(n=400, t0="2011-02-14", seed=0, cancel_rows=(), tau_lo=0.05):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"customer_id": np.arange(1, n + 1), "t0": pd.Timestamp(t0),
                      "cancel_rate_365": 0.0, "value_365": rng.uniform(10, 5000, n),
                      "s_hat_mean": rng.uniform(100, 400, n), "mu0_cal": rng.uniform(0.1, 0.4, n)})
    d["s_lo"], d["s_hi"] = d.s_hat_mean / 2.5, d.s_hat_mean * 2.5
    for a in TREATED:
        d[f"tau_hat_{a}"] = 0.10
        d[f"tau_lo_{a}"] = tau_lo
        d[f"tau_hi_{a}"] = 0.15
        d[f"V_hat_{a}"] = 10.0
    d.loc[list(cancel_rows), "cancel_rate_365"] = 0.5
    return d


def proposal(n, k=150, action=1, start=0):
    w = np.zeros((n, 4)); w[:, 0] = 1
    for i in range(start, start + k):
        w[i] = 0; w[i, action] = 1
    return w


def new_router(s_wide=1e9):
    return R.ActionRouter(s_wide, "test-model")


def test_contract_rejects_bad_input():
    d = make_period()
    r = new_router()
    r.validate_input(d)
    bad = d.copy(); bad.loc[0, "mu0_cal"] = 1.7
    with pytest.raises(Exception):
        r.validate_input(bad)
    dup = pd.concat([d, d.iloc[:1]])
    with pytest.raises(Exception):
        r.validate_input(dup)


def test_hard_guardrail_order_and_no_reallocation():
    d = make_period(cancel_rows=range(0, 150))                 # 150 đề xuất đầu đều có cancel_rate > 0.3
    r = new_router()
    w = proposal(len(d), 150)
    out = r.route(d, w, "valid")
    assert len(out) == len(d) and out.decision_id.is_unique
    p = out[out.proposed_action != "a0"]
    # opt-out đứng trước cancel: khách vừa opt-out vừa cancel được ghi G1
    opt = np.array([R.hash_unit("optout", R.SALT, c) < R.OPT_OUT_PCT / 100 for c in p.customer_id])
    assert (p.guardrail_triggered[opt] == "G1_opt_out").all()
    assert (p.guardrail_triggered[~opt] == "G2_cancel_rate").all()
    assert (p[p.guardrail_triggered != "none"].final_action == "a0").all()
    assert out.final_action.ne("a0").sum() == 0                    # mọi đề xuất bị chặn → không ai được tái phân bổ
    assert (out[out.proposed_action == "a0"].final_action == "a0").all()


def test_router_never_adds_vouchers_and_budget_only_decreases():
    d = make_period(seed=1)
    out = new_router().route(d, proposal(len(d), 200, action=2), "valid")
    assert (out.final_action != "a0").sum() <= (out.proposed_action != "a0").sum()
    cap = {"a0": 0, "a1": 30, "a2": 60, "a3": 30}
    assert out.final_action.map(cap).sum() <= out.proposed_action.map(cap).sum()


def test_frequency_cap_blocks_next_period_but_not_after_28_days():
    r = new_router()
    d1 = make_period(t0="2011-02-14", seed=2)
    o1 = r.route(d1, proposal(len(d1), 120), "valid")
    sent = set(o1[o1.final_action != "a0"].customer_id)
    assert len(sent) > 0
    d2 = make_period(t0="2011-02-28", seed=2)                      # 14 ngày sau → còn trong cap
    o2 = r.route(d2, proposal(len(d2), 120), "valid")
    blocked = o2[o2.customer_id.isin(sent) & (o2.proposed_action != "a0")]
    assert (blocked.guardrails_all.str.contains("G3_frequency_cap")).all()
    assert (blocked.final_action == "a0").all()
    d3 = make_period(t0="2011-03-14", seed=2)                      # đúng 28 ngày sau kỳ 1 → hết cap
    o3 = r.route(d3, proposal(len(d3), 120), "valid")
    again = o3[o3.customer_id.isin(sent) & (o3.proposed_action != "a0")]
    assert not again.guardrails_all.str.contains("G3_frequency_cap").any() or \
        set(again[again.guardrails_all.str.contains("G3_frequency_cap")].customer_id) <= set(o2[o2.final_action != "a0"].customer_id)


def test_control_group_and_propensity():
    d = make_period(n=4000, seed=3)
    out = new_router().route(d, proposal(len(d), 3000), "valid")
    rnd = out[out.in_randomization_set]
    share = rnd.control_group.mean()
    assert 0.07 < share < 0.13                                      # ≈ 10 %
    assert (rnd.propensity_a0 == 0.1).all()
    assert np.isclose(rnd[rnd.control_group].propensity_logged, 0.1).all()
    assert np.isclose(rnd[~rnd.control_group].propensity_logged, 0.9).all()
    assert (out[~out.in_randomization_set].propensity_logged == 1.0).all()
    assert (out[out.control_group].final_action == "a0").all()
    # tái lập: cùng đầu vào → cùng nhóm đối chứng
    out2 = new_router().route(d, proposal(len(d), 3000), "valid")
    assert (out.control_group.to_numpy() == out2.control_group.to_numpy()).all()


def test_hitl_when_ci_contains_zero_and_reviewer_rule():
    d = make_period(seed=4, tau_lo=-0.05)
    out = new_router().route(d, proposal(len(d), 100), "valid")
    p = out[(out.proposed_action != "a0") & (out.guardrail_triggered == "none")]
    assert p.hitl_flag.all() and (p.hitl_trigger == "H1_CI_tau_chua_0").all()
    assert set(p.hitl_outcome) <= {"approved", "downgraded", "rejected"}
    # τ_mid = 0.10 − 0.5·0.15 = 0.025 → V(τ_mid) = 0.3·S·0.025 − 0.1·S·(μ0+0.025) − 0.02 < 0 khi μ0 lớn → có từ chối
    d2 = make_period(seed=4, tau_lo=0.05)
    out2 = new_router().route(d2, proposal(len(d2), 100), "valid")
    assert not out2.hitl_flag.any()                                 # CI không chứa 0, Ŝ hẹp, không VIP a2/a3


def test_idempotent_rerun_returns_same_records():
    r = new_router()
    d = make_period(seed=5)
    o1 = r.route(d, proposal(len(d), 100), "valid")
    n = len(r.log)
    o2 = r.route(d, proposal(len(d), 100), "valid")                 # chạy lại cùng kỳ
    assert len(r.log) == n
    assert (o1.decision_id.to_numpy() == o2.decision_id.to_numpy()).all()


def test_decision_schema_rejects_invalid_log():
    d = make_period(seed=6)
    out = new_router().route(d, proposal(len(d), 100), "valid")
    R.DECISION_SCHEMA.validate(out)
    bad = out.copy()
    i = bad.index[bad.control_group][0] if bad.control_group.any() else bad.index[0]
    bad.loc[bad.index[bad.proposed_action == "a0"][0], "final_action"] = "a1"      # Router tự thêm voucher
    with pytest.raises(Exception):
        R.DECISION_SCHEMA.validate(bad)


# ---------------------------------------------------------------- kiểm theo file đầu ra thật
def test_real_log_properties():
    d = pd.read_parquet(ROOT / "data/processed/decisions_s7.parquet")
    R.DECISION_SCHEMA.validate(d)
    assert set(d.split) <= {"valid", "calib"}                       # không locked/oot
    assert d.idempotency_key.is_unique and d.decision_id.is_unique
    assert d.groupby("t0").size().min() > 3000                      # toàn bộ khách mỗi kỳ
    assert (d.final_action != "a0").sum() <= (d.proposed_action != "a0").sum()
    sent = d[d.final_action != "a0"].sort_values("t0")             # frequency cap không bao giờ bị vi phạm
    gap = sent.groupby("customer_id").t0.diff().dropna()
    assert (gap >= pd.Timedelta(days=R.FREQ_CAP_DAYS)).all()
    rnd = d[d.in_randomization_set]
    assert 0.07 < rnd.control_group.mean() < 0.13
    assert not d.reason_1.fillna("").str.startswith(("T0_MONTH|", "DAYS_TO_XMAS|")).any()


def test_real_outputs_consistent():
    c = pd.read_csv(T / "s7_scoring_consistency.csv")
    assert c[["max_lech_V_hat", "max_lech_tau", "max_lech_tau_lo", "max_lech_mu0_cal", "max_lech_s_hat_mean"]].max().max() < 1e-6
    v = pd.read_csv(T / "s7_policy_value_summary.csv").set_index("chinh_sach")
    assert v.loc["MILP thuần (đề xuất)", "tong_vi_pham_rang_buoc"] == 0
    i = pd.read_csv(T / "s7_idempotency.csv").iloc[0]
    assert i.chay_lai_tra_ban_ghi_cu and i.khoa_chong_lap_duy_nhat and i.so_ban_ghi_truoc == i.so_ban_ghi_sau_chay_lai_ky_dau
    p = pd.read_csv(T / "s7_latency_percentiles.csv").set_index("khau")
    assert p.loc["tong", "P50"] <= p.loc["tong", "P95"] <= p.loc["tong", "P99"]
