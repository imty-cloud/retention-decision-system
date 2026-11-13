"""Kiểm thử AMENDMENT v2 của GĐ7 (router-v2.0). (a) cap trong miền MILP · (b) ràng buộc trên bản ghi cuối · (c) khách bù
có propensity đối chứng = 0 · (d) chạy lại không tạo bản ghi mới · (e) schema pandera với cột mã lý do mới."""
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
LOG = ROOT / "data/processed/decisions_s7_v2.parquet"


def ins_for(d, B=1e9, K=60, uk_share=0.9):
    n = len(d)
    is_uk = np.arange(n) < int(uk_share * n)
    return {"is_uk": is_uk, "B": B, "K": K}


def router():
    return R.ActionRouterV2(1e9, "test-model")


# ---------------------------------------------------------------- tổng hợp
def test_backfill_fills_each_segment_with_a1_and_no_control():
    d = make_period(n=600, seed=11)
    ins = ins_for(d)
    out = router().route(d, proposal(len(d), 0), "valid", ins=ins)          # MILP không đề xuất ai ⇒ phải bù toàn bộ
    req = R.coverage_requirement(ins["is_uk"], np.ones(len(d), bool) & ~np.array(
        [R.hash_unit("optout", R.SALT, c) < R.OPT_OUT_PCT / 100 for c in d.customer_id]))
    b = out[out.is_backfill]
    assert (b.final_action == "a1").all() and (b.router_reason_code == R.BACKFILL_CODE).all()
    assert int((b.customer_id.isin(d.customer_id[ins["is_uk"]])).sum()) == req[0]
    assert int((b.customer_id.isin(d.customer_id[~ins["is_uk"]])).sum()) == req[1]
    # (c) propensity đối chứng = 0, không thuộc tập ngẫu nhiên, không vào đối chứng
    assert (b.p_control == 0).all() and (b.propensity_a0 == 0).all() and (~b.in_randomization_set).all() and (~b.control_group).all()
    # khách bù không bao giờ là khách opt-out
    opt = np.array([R.hash_unit("optout", R.SALT, c) < R.OPT_OUT_PCT / 100 for c in b.customer_id])
    assert not opt.any()


def test_backfill_ranked_by_vhat_a1_and_requires_nonneg():
    d = make_period(n=600, seed=12)
    d["V_hat_a1"] = np.linspace(-50, 50, len(d))                             # một nửa V̂(a1) < 0
    ins = ins_for(d)
    out = router().route(d, proposal(len(d), 0), "valid", ins=ins)
    b = out[out.is_backfill].merge(d[["customer_id", "V_hat_a1"]], on="customer_id")
    assert (b.V_hat_a1 >= 0).all()
    uk = b[b.customer_id.isin(d.customer_id[ins["is_uk"]])]
    elig = d[ins["is_uk"] & (d.V_hat_a1 >= 0)].copy()
    elig = elig[[R.hash_unit("optout", R.SALT, c) >= R.OPT_OUT_PCT / 100 for c in elig.customer_id]]
    top = elig.sort_values("V_hat_a1", ascending=False).customer_id.head(len(uk))
    assert set(uk.customer_id) == set(top)


def test_backfill_respects_budget_and_records_shortfall():
    d = make_period(n=600, seed=13)
    ins = ins_for(d, B=90)                                                   # chỉ đủ 3 voucher a1
    r = router()
    out = r.route(d, proposal(len(d), 0), "valid", ins=ins)
    assert out.is_backfill.sum() == 3
    cap = out.final_action.map({"a0": 0, "a1": 30, "a2": 60, "a3": 30}).sum()
    assert cap <= 90
    assert (out.backfill_status == "ung_vien_bu_vuot_nguon_luc").any()      # có ứng viên bị bỏ vì hết ngân sách
    cov = r.last_coverage
    assert cov["gui_UK"] + cov["gui_nonUK"] == 3 and cov["gui_UK"] < cov["req_UK"]   # ghi nhận vi phạm, không nới điều kiện


def test_backfill_candidates_go_through_hitl():
    d = make_period(n=600, seed=14, tau_lo=-0.2)                             # CI a1 chứa 0 ⇒ mọi ứng viên bù đi qua HITL
    out = router().route(d, proposal(len(d), 0), "valid", ins=ins_for(d))
    b = out[out.is_backfill]
    assert b.hitl_flag.all() and (b.hitl_outcome == "approved").all() and (b.hitl_trigger == "H1_CI_tau_chua_0").all()
    rej = out[out.backfill_status == "ung_vien_bu_bi_HITL_tu_choi"]
    assert (rej.final_action == "a0").all() and (rej.hitl_outcome == "rejected").all()


def test_cap_flags_from_v2_history():
    r = router()
    d1 = make_period(t0="2011-02-14", seed=15)
    o1 = r.route(d1, proposal(len(d1), 100), "valid", ins=ins_for(d1))
    sent = set(o1[o1.final_action != "a0"].customer_id)
    d2 = make_period(t0="2011-02-28", seed=15)
    fl = r.domain_flags(d2)
    assert set(d2.customer_id[fl["incap"]]) == sent                         # cờ cap = đúng tập đã GỬI ở kỳ trước (gồm khách bù)


def test_idempotent_v2_rerun():                                              # (d)
    r = router()
    d = make_period(seed=16)
    ins = ins_for(d)
    o1 = r.route(d, proposal(len(d), 50), "valid", ins=ins)
    n = len(r.log)
    o2 = r.route(d, proposal(len(d), 50), "valid", ins=ins)
    assert len(r.log) == n and (o1.decision_id.to_numpy() == o2.decision_id.to_numpy()).all()


def test_schema_v2_rejects_bad_backfill():                                   # (e)
    d = make_period(n=600, seed=17)
    out = router().route(d, proposal(len(d), 0), "valid", ins=ins_for(d))
    R.DECISION_SCHEMA_V2.validate(out)
    bad = out.copy()
    i = bad.index[bad.is_backfill][0]
    bad.loc[i, "p_control"] = 0.1
    with pytest.raises(Exception):
        R.DECISION_SCHEMA_V2.validate(bad)
    bad2 = out.copy()
    bad2.loc[i, "router_reason_code"] = "MILP_AUTO"
    with pytest.raises(Exception):
        R.DECISION_SCHEMA_V2.validate(bad2)


# ---------------------------------------------------------------- trên kết quả thật
@pytest.fixture(scope="module")
def log():
    return pd.read_parquet(LOG)


def test_real_milp_v2_never_proposes_capped(log):                            # (a)
    log = log.sort_values("t0")
    last = {}
    for t0, g in log.groupby("t0", sort=True):
        prop = g[g.proposed_action != "a0"]
        inc = [c for c in prop.customer_id if c in last and (t0 - last[c]).days < R.FREQ_CAP_DAYS]
        assert not inc, f"{t0}: MILP đề xuất {len(inc)} khách đang trong cap"
        opt = np.array([R.hash_unit("optout", R.SALT, c) < R.OPT_OUT_PCT / 100 for c in prop.customer_id])
        assert not opt.any()
        for c in g.customer_id[g.final_action != "a0"]:
            last[c] = t0
    assert (log.guardrail_triggered == "none").all()                         # lớp kiểm tra thứ hai không bao giờ phải chặn


def test_real_final_log_resources(log):                                      # (b)
    a = pd.read_csv(T / "s7_constraint_audit_v2.csv")
    v2 = a[a.phien_ban == "v2"]
    assert len(v2) == 14
    m = v2[v2.chinh_sach == "MILP qua Router v2"]                            # bản ghi cuối chính: sạch mọi ràng buộc tài nguyên
    assert len(m) == 7 and not m[["vp_ngan_sach", "vp_cuoc_goi", "vp_ty_le_a2", "vp_gui_khach_bi_chan"]].any().any()
    # luồng luật cũ: ngân sách / cuộc gọi / khách bị chặn luôn sạch; trần a2 có thể vượt do HITL từ chối a1 (không do bước bù) → chỉ báo cáo
    assert not v2[["vp_ngan_sach", "vp_cuoc_goi", "vp_gui_khach_bi_chan"]].any().any()
    cap = log.final_action.map({"a0": 0, "a1": 30, "a2": 60, "a3": 30})
    assert (cap.groupby(log.t0).sum() <= 8000 + 1e-6).all()
    assert ((log.final_action == "a3").groupby(log.t0).sum() <= 60).all()


def test_real_backfill_propensity(log):                                      # (c)
    b = log[log.is_backfill]
    assert (b.p_control == 0).all() and (b.propensity_a0 == 0).all() and not b.control_group.any()
    assert (b.router_reason_code == R.BACKFILL_CODE).all() and (b.final_action == "a1").all()
    rnd = log[log.in_randomization_set]
    assert 0.06 < rnd.control_group.mean() < 0.14


def test_real_idempotency_and_schema(log):                                   # (d) + (e)
    i = pd.read_csv(T / "s7_idempotency_v2.csv").iloc[0]
    assert i.chay_lai_tra_ban_ghi_cu and i.khoa_chong_lap_duy_nhat and i.so_ban_ghi_truoc == i.so_ban_ghi_sau
    R.DECISION_SCHEMA_V2.validate(log)
    assert set(log.split) <= {"valid", "calib"} and (log.policy_version == R.POLICY_VERSION_V2).all()
    assert log.reason_1.notna().mean() > 0.99                                 # sửa lỗi v1: mã lý do đã có trong bản ghi
    assert not log.reason_1.fillna("").str.startswith(("T0_MONTH|", "DAYS_TO_XMAS|")).any()


def test_v1_outputs_untouched():
    v1 = pd.read_parquet(ROOT / "data/processed/decisions_s7.parquet")
    assert (v1.policy_version == R.POLICY_VERSION).all() and "is_backfill" not in v1.columns
