"""Luật cứng #1: oracle.parquet chỉ được đọc trong hàm có tên bắt đầu bằng evaluate_ / oracle_."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
EXEMPT = {"s1_data.py", "s1b_intervention.py", "common.py"}  # s1b là nơi SINH oracle


def _violations(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = []

    def visit(node, fn):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = fn + [node.name]
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "oracle" in node.value.lower():
            if not any(f.startswith(("evaluate_", "oracle_")) for f in fn):
                bad.append((node.lineno, node.value[:60], fn))
        for ch in ast.iter_child_nodes(node):
            visit(ch, fn)

    visit(tree, [])
    return bad


def test_oracle_only_in_evaluate_functions():
    bad = {}
    for f in SRC.glob("*.py"):
        if f.name in EXEMPT:
            continue
        v = [x for x in _violations(f)]
        if v:
            bad[f.name] = v
    assert not bad, f"Đọc/nhắc oracle ngoài hàm evaluate_/oracle_: {bad}"


def test_pred_s2_schema_and_oof():
    import numpy as np
    import pandas as pd
    p = pd.read_parquet(SRC.parent / "data/processed/pred_s2.parquet")
    for c in ["mu0_raw", "mu0_cal", "mu0_is_oof", "s_hat", "s_hat_mean", "s_lo", "s_hi"]:
        assert c in p.columns
    assert p.mu0_is_oof.eq(p.split == "train").all()
    assert p[["mu0_raw", "mu0_cal", "s_hat_mean"]].notna().all().all()
    assert (p.s_lo <= p.s_hat).all() and (p.s_hat <= p.s_hi).all()
    assert p.mu0_cal.between(0, 1).all()


def test_s3_outputs():
    import pandas as pd
    r = pd.read_parquet(SRC.parent / "data/processed/reasons_s3.parquet")
    assert set(r.split) <= {"train", "valid", "calib"}          # GĐ3 không chạm locked/oot
    for i in (1, 2, 3):
        assert r[f"reason_{i}_code"].str.contains(r"\|(UP|DOWN)$").all()
    assert (r.reason_1_shap.abs() >= r.reason_2_shap.abs()).all()
    assert (r.reason_2_shap.abs() >= r.reason_3_shap.abs()).all()
    assert not r.duplicated(["customer_id", "t0"]).any()
    for i in (1, 2, 3):                                           # mã lý do không lấy từ đặc trưng lịch
        assert not r[f"reason_{i}_code"].str.startswith(("T0_MONTH|", "DAYS_TO_XMAS|")).any()


def test_s5_constraints_and_verification():
    import pandas as pd
    T = SRC.parent / "outputs/tables"
    for f in ["s5_constraint_check.csv", "s5_constraint_check_mo_rong.csv"]:
        if (T / f).exists():
            assert pd.read_csv(T / f).tong_vi_pham.sum() == 0, f
    v = pd.read_csv(T / "s5_verification_6_summary.csv", index_col=0).gia_tri
    for k in ["max_lech_z_sao", "max_lech_spo", "max_lech_spo_plus"]:
        assert abs(float(v[k])) < 1e-9, k
    assert float(v["ty_le_spo_plus_ge_spo"]) == 1.0
    assert float(v["ty_le_giam_khi_di_nguoc_subgradient (đúng dấu)"]) == 1.0
    cx = pd.read_csv(T / "s5_solver_crosscheck.csv")
    assert cx.chenh_lech.abs().max() < 1e-6


def test_s5b_dr_diagnostics():
    import pandas as pd
    T = SRC.parent / "outputs/tables"
    c = pd.read_csv(T / "s5b_consistency_s5.csv").iloc[0]
    assert c.so_lo_so_khop == c.so_lo_ky_vong and c.max_lech_gia_tri_hien_thuc < 1e-6   # cùng instance/quyết định với s5
    e = pd.read_csv(T / "s5b_estimates.csv")
    assert e.groupby(["split", "quy_mo", "chinh_sach", "don_vi_tinh"]).bien_the.nunique().eq(7).all()
    p = e[e.bien_the == "AIPW (chính)"]
    assert ((p.uoc_luong - p.phan_mo_hinh_DM - p.phan_hieu_chinh_IPW).abs() < 1e-6).all()   # phân rã cộng đúng
    assert (p.ci_lo <= p.ci_hi).all()
    assert pd.read_csv(T / "s5b_full_t0_milp.csv").n_vi_pham.sum() == 0


def test_s6_milp():
    import pandas as pd
    T = SRC.parent / "outputs/tables"
    m = pd.read_csv(T / "s6_main_comparison.csv")
    assert m.n_vi_pham.sum() == 0
    mi = m[m.phuong_phap == "MILP trên V̂"]
    assert mi.lech_PuLP_vs_highspy.abs().max() < 0.05 * 1 and mi.gap_solver.max() <= 1e-3 + 1e-9
    g = m[m.phuong_phap == "Greedy (V̂/mệnh giá)"].set_index(["split", "t0"]).gia_tri_ke_hoach_V_hat
    assert (mi.set_index(["split", "t0"]).gia_tri_ke_hoach_V_hat >= g - 1e-6).all()       # MILP ≥ Greedy trên mục tiêu V̂
    d = pd.read_csv(T / "s6_shadow_prices.csv")
    assert (d.gia_bong_ngan_sach_GBP_tren_GBP_menh_gia >= 0).all() and (d.gia_bong_cuoc_goi_GBP_tren_cuoc >= 0).all()
    assert pd.read_csv(T / "s6_sensitivity_BK.csv").vi_pham.fillna(0).sum() == 0
