"""GĐ6 – MILP toàn quy mô mỗi t0 (Chương 6).

Dùng lại (không viết lại) từ s5_spo.py: build_instance, MILP (highspy), greedy, check_feasible, evaluate_load_costs;
từ s5b_dr_diagnostics.py: load_data, legacy_scores. Instance = TOÀN BỘ khách của một t0 (B, K, độ phủ như config, không co).

Thiết kế (ghi trước khi chạy):
 1. So sánh chính (B = 8 000, K = 60 theo config) trên 7 t0 của valid + calib: MILP (PuLP + HiGHS, giới hạn 60 s, gap 0.1 %,
    theo CLAUDE.md) vs Greedy (V̂/mệnh giá) vs luật RFM cũ (MILP trên điểm luật cũ, cùng miền khả thi). Đối chiếu giá trị mục tiêu
    PuLP với highspy trực tiếp (gap 0.01 %). Báo: giá trị kế hoạch (V̂), giá trị thực (đáp án, hiện thực A và kỳ vọng B),
    khoảng cách tối ưu (solver, so với cận LP), tài nguyên dùng, thời gian dựng/giải.
 2. Giá bóng từ LP relaxation: ngân sách (GBP giá trị / GBP mệnh giá), công suất gọi (GBP / cuộc), trần a2, độ phủ UK / non-UK;
    quy ước dấu đã kiểm chứng bằng sai phân hữu hạn của LP trong thử nghiệm riêng, và độ dốc trung bình của lưới B, K khớp giá bóng.
 3. Độ nhạy B ∈ {2k, 4k, 8k, 16k, 32k} × K ∈ {30, 60, 120} (7 t0, mọi phương pháp), kèm biến thể không ràng buộc độ phủ
    (đo "giá" của ràng buộc phủ; B = 2k không khả thi nếu giữ độ phủ vì 3 % × ~3.3k khách UK × £30 mệnh giá tối thiểu ≈ £3k).
 4. Độ nhạy biên lợi nhuận m ∈ {0.20, 0.25, 0.30, 0.35, 0.40}: V̂ tính lại theo m (τ̂ là tác động lên xác suất mua nên không
    đổi theo m), giá trị thật cũng tính lại theo m. Ma trận "giả định m̂ × thật m" cho thấy mất bao nhiêu nếu giả định sai.
 5. Thời gian giải theo n (1k … 50k khách, B, K tỷ lệ theo n, mẫu không hoàn lại từ train+valid+calib).
 6. Thí nghiệm đinh: μ̂0 chưa hiệu chỉnh (mu0_raw) vs đã hiệu chỉnh (mu0_cal) ⇒ MILP riêng cho từng loại V̂ ⇒ chi phí chiết khấu
    kế hoạch [Σ w·min(d·Ŝ, cap)·(μ̂0+τ̂)] vs thực tế (hiện thực và kỳ vọng theo kết quả tiềm năng). Mệnh giá phát hành giống nhau
    (ràng buộc cố định) – cái thay đổi là chi phí chiết khấu THỰC TRẢ và lợi nhuận.
Không dùng locked/oot. Đáp án chỉ đọc trong evaluate_*.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import time

import highspy
import matplotlib
import numpy as np
import pandas as pd
import pulp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
import s5_spo as S5
from s5b_dr_diagnostics import legacy_scores, load_data

CFG = common.load_config()
SEED = CFG["project"]["seed"]
TABLES, FIGS = Path(CFG["paths"]["tables"]), Path(CFG["paths"]["figures"])
BIZ = CFG["business"]
TREATED = S5.TREATED
CAP, DISC, CONTACT = S5.CAP, S5.DISC, S5.CONTACT
M0 = BIZ["gross_margin_rate"]
B0, K0 = BIZ["budget_face_value_per_t0"], BIZ["call_capacity_per_t0"]
B_GRID, K_GRID = [2000, 4000, 8000, 16000, 32000], [30, 60, 120]
M_GRID = [0.20, 0.25, 0.30, 0.35, 0.40]
N_GRID = [1000, 2000, 5000, 10000, 20000, 50000]
TIME_LIMIT, GAP = 60.0, 1e-3
METHODS = ["MILP trên V̂", "Greedy (V̂/mệnh giá)", "Luật RFM cũ"]
HI = highspy.HighsModelStatus


# ============================================================================ V̂ theo biên lợi nhuận m và mu0 tùy chọn
def value_hat_m(tau: np.ndarray, mu0: np.ndarray, S: np.ndarray, m: float) -> np.ndarray:
    """Công thức CLAUDE.md mục 1 (giống s4_causal.value_hat) nhưng m và μ̂0 là tham số."""
    V = np.zeros_like(tau)
    for j, a in enumerate(TREATED):
        p = np.clip(mu0 + tau[:, j], 0, 1)
        V[:, j] = m * S * tau[:, j] - np.minimum(DISC[j + 1] * S, CAP[j + 1]) * p - CONTACT[j + 1]
    return V


# ============================================================================ solver: highspy (nhanh) và PuLP (theo đặc tả)
def solve_hs(ins: dict, c: np.ndarray, gap: float = GAP, limit: float = TIME_LIMIT, m: S5.MILP | None = None) -> dict:
    """MILP bằng highspy (class của s5). Trả lại cả trạng thái, gap, cận dưới/trên, thời gian."""
    c = S5.cmat(np.asarray(c, float))              # n×3 (a1..a3) → n×4 (a0 = 0)
    t = time.perf_counter()
    m = m or S5.MILP(ins)
    t_build = time.perf_counter() - t
    m.h.setOptionValue("mip_rel_gap", gap)
    m.h.setOptionValue("time_limit", limit)
    m.h.changeColsCost(m.N, m.cols, -c.ravel())
    t = time.perf_counter()
    m.h.run()
    t_solve = time.perf_counter() - t
    st, info = m.h.getModelStatus(), m.h.getInfo()
    ok = info.primal_solution_status == 2
    w = np.rint(np.array(m.h.getSolution().col_value)).reshape(ins["n"], 4) if ok else None
    return dict(w=w, status=st.name if hasattr(st, "name") else str(st), gap=float(info.mip_gap) if ok else np.nan,
                obj=float((c * w).sum()) if ok else np.nan, bound=-float(info.mip_dual_bound) if ok else np.nan,
                t_build=t_build, t_solve=t_solve)


def solve_pulp(ins: dict, c: np.ndarray, gap: float = GAP, limit: float = TIME_LIMIT) -> dict:
    """Cùng bài toán bằng PuLP + HiGHS (đặc tả GĐ6; CBC không có trong PuLP 4.2 nên không có phương án dự phòng)."""
    n = ins["n"]
    t = time.perf_counter()
    prob = pulp.LpProblem("milp_toan_t0", pulp.LpMaximize)
    x = [[prob.add_variable(f"x_{i}_{a}", cat="Binary") for a in range(4)] for i in range(n)]
    prob += pulp.lpSum(float(c[i, a - 1]) * x[i][a] for i in range(n) for a in (1, 2, 3))
    for i in range(n):
        prob += pulp.lpSum(x[i]) == 1
        if ins["blocked"][i]:
            for a in (1, 2, 3):
                prob += x[i][a] == 0
    prob += pulp.lpSum(CAP[a] * x[i][a] for i in range(n) for a in (1, 2, 3)) <= ins["B"]
    prob += pulp.lpSum(x[i][3] for i in range(n)) <= ins["K"]
    prob += (3 * pulp.lpSum(x[i][2] for i in range(n)) - pulp.lpSum(x[i][1] + x[i][3] for i in range(n))) <= 0
    prob += pulp.lpSum(x[i][a] for i in range(n) if ins["is_uk"][i] for a in (1, 2, 3)) >= ins["req_uk"]
    prob += pulp.lpSum(x[i][a] for i in range(n) if not ins["is_uk"][i] for a in (1, 2, 3)) >= ins["req_non"]
    t_build = time.perf_counter() - t
    t = time.perf_counter()
    stats = prob.solve(pulp.HiGHS(msg=False, gapRel=gap, timeLimit=limit, threads=1))
    t_solve = time.perf_counter() - t
    w = np.array([[round(v.value() or 0) for v in row] for row in x], float)
    obj = float((c * w[:, 1:]).sum()) if stats.has_solution else np.nan
    bd = float(stats.best_bound) if stats.best_bound is not None else np.nan
    return dict(w=w, status=stats.status.name, obj=obj, bound=bd, gap=abs(bd - obj) / max(abs(obj), 1e-9),
                t_build=t_build, t_solve=t_solve)


def lp_relaxation(ins: dict, c: np.ndarray) -> dict:
    """LP relaxation (bỏ tính nguyên): giá trị tối ưu và GIÁ BÓNG (đối ngẫu) của từng ràng buộc – thang GBP giá trị / đơn vị."""
    m = S5.MILP(ins)
    m.h.changeColsIntegrality(m.N, m.cols, np.array([highspy.HighsVarType.kContinuous] * m.N))
    m.h.setOptionValue("time_limit", 120.0)
    m.h.changeColsCost(m.N, m.cols, -S5.cmat(np.asarray(c, float)).ravel())
    m.h.run()
    n = ins["n"]
    if m.h.getModelStatus() != HI.kOptimal:
        return dict(lp=np.nan, pi_B=np.nan, pi_K=np.nan, pi_a2=np.nan, cost_cover_uk=np.nan, cost_cover_non=np.nan)
    d = np.array(m.h.getSolution().row_dual)
    # bài toán nội bộ là MIN(−c·x): ràng buộc ≤ có dual ≤ 0 ⇒ giá bóng (max) = −dual; ràng buộc ≥ có dual ≥ 0 ⇒ chi phí = dual
    return dict(lp=-float(m.h.getInfo().objective_function_value), pi_B=-d[n], pi_K=-d[n + 1], pi_a2=-d[n + 2],
                cost_cover_uk=d[n + 3], cost_cover_non=d[n + 4])


def lp_value(ins: dict, c: np.ndarray) -> float:
    return lp_relaxation(ins, c)["lp"]


# ============================================================================ ĐÁNH GIÁ (được đọc đáp án)
def evaluate_truth(keys: pd.DataFrame) -> dict:
    """Kết quả tiềm năng thật của từng dòng: Y(a), Y(0), giỏ S, τ điều kiện, chi phí chiết khấu hiện thực."""
    orc = pd.read_parquet(common.p("data/processed/oracle.parquet"))
    t = keys[["customer_id", "t0"]].merge(orc.drop(columns=["split"]), on=["customer_id", "t0"], how="left",
                                          validate="one_to_one")
    assert t.y_a0.notna().all()
    acts = ["a0", "a1", "a2", "a3"]
    return dict(S=t.basket_value_S.to_numpy(), y0=t.y_a0.to_numpy().astype(float),
                Y=np.column_stack([t[f"y_{a}"] for a in acts]).astype(float),
                tau=np.column_stack([t[f"tau_cond_{a}"] for a in TREATED]),
                dc_real=np.column_stack([t[f"discount_cost_{a}"] for a in acts]),
                profit=np.column_stack([t[f"profit_{a}"] for a in acts]))


def evaluate_utilities(T: dict, m: float) -> tuple[np.ndarray, np.ndarray]:
    """Lợi nhuận gia tăng so với a0 theo biên m: U_real (hiện thực, A) và U_exp (kỳ vọng theo U, B). n × 4 (cột 0 = 0)."""
    S, y0, Y, tau = T["S"], T["y0"], T["Y"], T["tau"]
    n = len(S)
    Ur, Ue = np.zeros((n, 4)), np.zeros((n, 4))
    for k in (1, 2, 3):
        dcap = np.minimum(DISC[k] * S, CAP[k])
        Ur[:, k] = m * S * (Y[:, k] - Y[:, 0]) - dcap * Y[:, k] - CONTACT[k]
        Ue[:, k] = m * S * tau[:, k - 1] - dcap * (y0 + tau[:, k - 1]) - CONTACT[k]
    return Ur, Ue


def evaluate_policy(T: dict, idx: np.ndarray, w: np.ndarray, m: float = M0) -> dict:
    Ur, Ue = evaluate_utilities({k: v[idx] for k, v in T.items()}, m)
    return dict(gia_tri_hien_thuc=float((w * Ur).sum()), gia_tri_ky_vong=float((w * Ue).sum()))


def evaluate_oracle_policy(T: dict, ins: dict, m: float = M0) -> dict:
    """Chính sách biết trước (cùng miền khả thi) – mốc tham chiếu regret."""
    idx = ins["idx"]
    Ur, Ue = evaluate_utilities({k: v[idx] for k, v in T.items()}, m)
    zr, ze = solve_hs(ins, Ur[:, 1:])["obj"], solve_hs(ins, Ue[:, 1:])["obj"]
    return dict(z_sao_hien_thuc=zr, z_sao_ky_vong=ze)


def evaluate_discount_costs(T: dict, idx: np.ndarray, w: np.ndarray) -> dict:
    """Chi phí chiết khấu THỰC TRẢ: hiện thực (đúng Y(a) đã xảy ra) và kỳ vọng theo kết quả tiềm năng; kèm tỷ lệ mua thật."""
    S, y0, tau, dcr = T["S"][idx], T["y0"][idx], T["tau"][idx], T["dc_real"][idx]
    exp = sum((w[:, k] * np.minimum(DISC[k] * S, CAP[k]) * (y0 + tau[:, k - 1])).sum() for k in (1, 2, 3))
    n_v = w[:, 1:].sum()
    pbuy = sum((w[:, k] * (y0 + tau[:, k - 1])).sum() for k in (1, 2, 3)) / max(n_v, 1)
    return dict(chiet_khau_thuc_hien_thuc=float((w * dcr).sum()), chiet_khau_thuc_ky_vong=float(exp),
                ty_le_mua_that_TB_nguoi_nhan=float(pbuy))


# ============================================================================ tiện ích kết quả
def resources(ins: dict, w: np.ndarray) -> dict:
    ck = S5.check_feasible(ins, w)
    return dict(ngan_sach_dung=ck["ngan_sach_dung"], so_voucher=ck["so_voucher"], so_cuoc_goi=ck["so_cuoc_goi"],
                so_a2=ck["so_a2"], n_vi_pham=ck["n_vi_pham"])


def make_ins(df, meta, split, t0, **kw) -> dict:
    idx = np.where((df.split == split) & (df.t0 == t0))[0]
    return S5.build_instance(idx, meta, len(idx), split, t0, **kw)


def periods(df) -> list[tuple[str, pd.Timestamp]]:
    return [(s, t) for s in ("valid", "calib") for t in np.sort(df[df.split == s].t0.unique())]


# ============================================================================ chính
def main() -> None:
    np.random.seed(SEED)
    t_start = time.time()
    files: list[str] = []
    pd.set_option("display.width", 250)
    df = load_data()
    p2 = pd.read_parquet(common.p("data/processed/pred_s2.parquet"), columns=["customer_id", "t0", "mu0_raw", "mu0_cal"])
    df = df.merge(p2, on=["customer_id", "t0"], how="left", validate="one_to_one")
    meta = df[["split", "t0", S5.SEG_COL, "cancel_rate_365"]].copy()
    tau = df[[f"tau_hat_{a}" for a in TREATED]].to_numpy()
    Ssh = df.s_hat_mean.to_numpy()
    V = df[[f"V_hat_{a}" for a in TREATED]].to_numpy()
    legacy = legacy_scores(df)
    ev = df.split.isin(["valid", "calib"]).to_numpy()
    assert np.allclose(value_hat_m(tau, df.mu0_base.to_numpy(), Ssh, M0)[ev], V[ev], atol=1e-6)   # khớp V̂ của GĐ4
    T = evaluate_truth(df)
    assert np.allclose(evaluate_utilities(T, M0)[0], S5.evaluate_load_costs(df)[0], atol=1e-6)      # khớp U_real của s5
    assert np.allclose(evaluate_utilities(T, M0)[1], S5.evaluate_load_costs(df)[1], atol=1e-6)      # khớp U_exp của s5
    per = periods(df)
    print(f"{len(per)} kỳ (valid+calib); dữ liệu sẵn sàng ({time.time() - t_start:.0f}s)")

    # ---------------------------------------------------------------- 1. so sánh chính + 2. giá bóng
    rows, duals = [], []
    for split, t0 in per:
        ins = make_ins(df, meta, split, t0)
        idx = ins["idx"]
        orc = evaluate_oracle_policy(T, ins)
        costs = {"MILP trên V̂": V[idx], "Luật RFM cũ": legacy[idx]}
        lpv = lp_relaxation(ins, V[idx])
        duals.append({"split": split, "t0": t0, "n_khach": len(idx), "LP_toi_uu_V_hat": lpv["lp"],
                      "gia_bong_ngan_sach_GBP_tren_GBP_menh_gia": lpv["pi_B"], "gia_bong_cuoc_goi_GBP_tren_cuoc": lpv["pi_K"],
                      "gia_bong_tran_a2": lpv["pi_a2"], "chi_phi_do_phu_UK_GBP_tren_khach": lpv["cost_cover_uk"],
                      "chi_phi_do_phu_nonUK_GBP_tren_khach": lpv["cost_cover_non"]})
        for meth in METHODS:
            if meth == "MILP trên V̂":
                r = solve_pulp(ins, costs[meth])
                chk = solve_hs(ins, costs[meth], gap=1e-4)
                extra = dict(obj_highspy_truc_tiep=chk["obj"], lech_PuLP_vs_highspy=r["obj"] - chk["obj"],
                             gap_solver=r["gap"], trang_thai=r["status"], giay_dung_mo_hinh=r["t_build"],
                             giay_giai=r["t_solve"], cap_LP=lpv["lp"])
                w = r["w"]
            elif meth == "Luật RFM cũ":
                r = solve_pulp(ins, costs[meth])
                extra = dict(gap_solver=r["gap"], trang_thai=r["status"], giay_dung_mo_hinh=r["t_build"],
                             giay_giai=r["t_solve"])
                w = r["w"]
            else:
                t = time.perf_counter()
                w = S5.greedy(ins, V[idx])
                extra = dict(giay_giai=time.perf_counter() - t, trang_thai="heuristic")
            plan = float((w[:, 1:] * V[idx]).sum())
            ev_ = evaluate_policy(T, idx, w)
            rows.append({"split": split, "t0": t0, "n_khach": len(idx), "phuong_phap": meth, "B": ins["B"], "K": ins["K"],
                         **resources(ins, w), "gia_tri_ke_hoach_V_hat": plan, **ev_, **orc,
                         "regret_cn_hien_thuc": (orc["z_sao_hien_thuc"] - ev_["gia_tri_hien_thuc"]) / abs(orc["z_sao_hien_thuc"]),
                         "regret_cn_ky_vong": (orc["z_sao_ky_vong"] - ev_["gia_tri_ky_vong"]) / abs(orc["z_sao_ky_vong"]),
                         **extra})
        print(f"  {split} {pd.Timestamp(t0).date()} xong ({time.time() - t_start:.0f}s)")
    main_tab = pd.DataFrame(rows)
    # khoảng cách tối ưu của từng phương pháp trên mục tiêu V̂ (so với MILP cùng kỳ và với cận LP)
    ref = main_tab[main_tab.phuong_phap == "MILP trên V̂"].set_index(["split", "t0"]).gia_tri_ke_hoach_V_hat
    main_tab["gap_so_voi_MILP_V_hat"] = [(ref[(r.split, r.t0)] - r.gia_tri_ke_hoach_V_hat) / abs(ref[(r.split, r.t0)])
                                         if r.phuong_phap != "Luật RFM cũ" else np.nan for r in main_tab.itertuples()]
    main_tab.to_csv(common.p(TABLES / "s6_main_comparison.csv"), index=False)
    dual_tab = pd.DataFrame(duals)
    dual_tab.to_csv(common.p(TABLES / "s6_shadow_prices.csv"), index=False)
    files += [str(TABLES / "s6_main_comparison.csv"), str(TABLES / "s6_shadow_prices.csv")]
    show = ["split", "t0", "phuong_phap", "so_voucher", "so_cuoc_goi", "so_a2", "ngan_sach_dung", "gia_tri_ke_hoach_V_hat",
            "gia_tri_hien_thuc", "gia_tri_ky_vong", "regret_cn_hien_thuc", "regret_cn_ky_vong", "gap_solver",
            "giay_dung_mo_hinh", "giay_giai", "n_vi_pham"]
    print(main_tab[show].round(3).to_string())
    print(dual_tab.round(4).to_string())
    summ = main_tab.groupby("phuong_phap").agg(
        n_ky=("t0", "size"), V_ke_hoach_TB=("gia_tri_ke_hoach_V_hat", "mean"), thuc_hien_thuc_TB=("gia_tri_hien_thuc", "mean"),
        thuc_ky_vong_TB=("gia_tri_ky_vong", "mean"), regret_A_TB=("regret_cn_hien_thuc", "mean"),
        regret_B_TB=("regret_cn_ky_vong", "mean"), ngan_sach_dung_TB=("ngan_sach_dung", "mean"),
        so_voucher_TB=("so_voucher", "mean"), giay_giai_TB=("giay_giai", "mean"), vi_pham=("n_vi_pham", "sum")).reset_index()
    # cộng dồn qua các kỳ (đúng cách GĐ8 sẽ tính G4): cộng tử số và mẫu số rồi mới chia
    for s_, lab in [(None, "tat_ca"), ("valid", "valid"), ("calib", "calib")]:
        g = main_tab if s_ is None else main_tab[main_tab.split == s_]
        a = g.groupby("phuong_phap").apply(lambda d: pd.Series({
            f"regret_A_cong_don_{lab}": (d.z_sao_hien_thuc.sum() - d.gia_tri_hien_thuc.sum()) / abs(d.z_sao_hien_thuc.sum()),
            f"regret_B_cong_don_{lab}": (d.z_sao_ky_vong.sum() - d.gia_tri_ky_vong.sum()) / abs(d.z_sao_ky_vong.sum())}),
            include_groups=False).reset_index()
        summ = summ.merge(a, on="phuong_phap")
    # số kỳ MILP thắng Greedy (x/7): theo V̂ kế hoạch, theo giá trị thật hiện thực (A) và kỳ vọng (B)
    mi = main_tab[main_tab.phuong_phap == "MILP trên V̂"].set_index(["split", "t0"])
    gr = main_tab[main_tab.phuong_phap == "Greedy (V̂/mệnh giá)"].set_index(["split", "t0"])
    n_ky = len(mi)
    for col, nm in [("gia_tri_ke_hoach_V_hat", "V_hat"), ("gia_tri_hien_thuc", "hien_thuc_A"), ("gia_tri_ky_vong", "ky_vong_B")]:
        summ[f"ky_MILP_thang_Greedy_{nm}"] = ["" if p != "MILP trên V̂" else f"{int((mi[col] > gr[col]).sum())}/{n_ky}"
                                              for p in summ.phuong_phap]
    summ.to_csv(common.p(TABLES / "s6_main_summary.csv"), index=False)
    files.append(str(TABLES / "s6_main_summary.csv"))
    print(summ.round(3).to_string())

    # ---------------------------------------------------------------- 3. độ nhạy B × K (+ biến thể không ràng buộc phủ)
    srows = []
    for split, t0 in per:
        base = make_ins(df, meta, split, t0)
        idx = base["idx"]
        for B in B_GRID:
            for K in K_GRID:
                for cover in (True, False):
                    ins = make_ins(df, meta, split, t0, B=B, K=K, req=None if cover else (0, 0))
                    for meth, c in (("MILP trên V̂", V[idx]), ("Luật RFM cũ", legacy[idx])):
                        if not cover and meth != "MILP trên V̂":
                            continue
                        r = solve_hs(ins, c)
                        if r["w"] is None:
                            srows.append({"split": split, "t0": t0, "B": B, "K": K, "phu_UK_nonUK": cover,
                                          "phuong_phap": meth, "trang_thai": r["status"], "kha_thi": False})
                            continue
                        ev_ = evaluate_policy(T, idx, r["w"])
                        lpv = lp_relaxation(ins, c) if meth == "MILP trên V̂" else {}
                        srows.append({"split": split, "t0": t0, "B": B, "K": K, "phu_UK_nonUK": cover, "phuong_phap": meth,
                                      "trang_thai": r["status"], "kha_thi": True, "gap": r["gap"], "giay_giai": r["t_solve"],
                                      **resources(ins, r["w"]), "gia_tri_ke_hoach_V_hat": float((r["w"][:, 1:] * V[idx]).sum()),
                                      **ev_, **{f"{k}": v for k, v in lpv.items()}})
                    if cover:
                        w = S5.greedy(ins, V[idx])
                        ck = resources(ins, w)
                        srows.append({"split": split, "t0": t0, "B": B, "K": K, "phu_UK_nonUK": cover,
                                      "phuong_phap": "Greedy (V̂/mệnh giá)", "trang_thai": "heuristic", "kha_thi": ck["n_vi_pham"] == 0,
                                      **ck, "gia_tri_ke_hoach_V_hat": float((w[:, 1:] * V[idx]).sum()),
                                      **evaluate_policy(T, idx, w)})
        print(f"  độ nhạy B×K: {split} {pd.Timestamp(t0).date()} xong ({time.time() - t_start:.0f}s)")
    sens = pd.DataFrame(srows)
    sens.to_csv(common.p(TABLES / "s6_sensitivity_BK_instances.csv"), index=False)
    agg = sens[sens.kha_thi].groupby(["B", "K", "phu_UK_nonUK", "phuong_phap"]).agg(
        n_ky=("t0", "size"), V_ke_hoach_TB=("gia_tri_ke_hoach_V_hat", "mean"), thuc_hien_thuc_TB=("gia_tri_hien_thuc", "mean"),
        thuc_ky_vong_TB=("gia_tri_ky_vong", "mean"), ngan_sach_dung_TB=("ngan_sach_dung", "mean"),
        so_voucher_TB=("so_voucher", "mean"), so_cuoc_goi_TB=("so_cuoc_goi", "mean"), so_a2_TB=("so_a2", "mean"),
        gia_bong_ngan_sach_TB=("pi_B", "mean"), gia_bong_cuoc_goi_TB=("pi_K", "mean"),
        giay_giai_TB=("giay_giai", "mean"), gap_max=("gap", "max"), vi_pham=("n_vi_pham", "sum")).reset_index()
    nfe = sens.groupby(["B", "K", "phu_UK_nonUK", "phuong_phap"]).kha_thi.agg(lambda s: int((~s).sum())).rename("so_ky_khong_kha_thi")
    agg = agg.merge(nfe.reset_index(), on=["B", "K", "phu_UK_nonUK", "phuong_phap"], how="outer")
    agg.to_csv(common.p(TABLES / "s6_sensitivity_BK.csv"), index=False)
    # số kỳ MILP thắng Greedy theo từng (B, K) (x/7) – chỉ các kỳ khả thi cho cả hai phương pháp, giữ độ phủ
    sm_, sg_ = (sens[(sens.phuong_phap == p) & sens.phu_UK_nonUK & sens.kha_thi].set_index(["B", "K", "split", "t0"])
                for p in ("MILP trên V̂", "Greedy (V̂/mệnh giá)"))
    common_idx = sm_.index.intersection(sg_.index)
    wrows = []
    for B_, K_ in sorted({(i[0], i[1]) for i in common_idx}):
        ids = [i for i in common_idx if i[0] == B_ and i[1] == K_]
        row = {"B": B_, "K": K_, "so_ky_so_sanh": len(ids)}
        for col, nm in [("gia_tri_ke_hoach_V_hat", "V_hat"), ("gia_tri_hien_thuc", "hien_thuc_A"), ("gia_tri_ky_vong", "ky_vong_B")]:
            wins = sum(sm_.at[i, col] > sg_.at[i, col] for i in ids)
            row[f"ky_MILP_thang_Greedy_{nm}"] = f"{int(wins)}/{len(ids)}"
        wrows.append(row)
    pd.DataFrame(wrows).to_csv(common.p(TABLES / "s6_milp_vs_greedy_wins.csv"), index=False)
    files.append(str(TABLES / "s6_milp_vs_greedy_wins.csv"))
    files += [str(TABLES / "s6_sensitivity_BK_instances.csv"), str(TABLES / "s6_sensitivity_BK.csv")]
    print(agg[agg.phuong_phap == "MILP trên V̂"].round(2).to_string())

    # ---------------------------------------------------------------- 4. độ nhạy biên lợi nhuận m
    mrows, dec = [], {}
    for split, t0 in per:
        ins = make_ins(df, meta, split, t0)
        idx = ins["idx"]
        for mh in M_GRID:
            Vm = value_hat_m(tau[idx], df.mu0_base.to_numpy()[idx], Ssh[idx], mh)
            dec[(t0, mh, "MILP")] = solve_hs(ins, Vm)["w"]
            dec[(t0, mh, "Greedy")] = S5.greedy(ins, Vm)
        dec[(t0, None, "Luật")] = solve_hs(ins, legacy[idx])["w"]
    for split, t0 in per:
        ins = make_ins(df, meta, split, t0)
        idx = ins["idx"]
        for mt in M_GRID:
            Ur, Ue = evaluate_utilities({k: v[idx] for k, v in T.items()}, mt)
            zr, ze = solve_hs(ins, Ur[:, 1:])["obj"], solve_hs(ins, Ue[:, 1:])["obj"]
            for mh in M_GRID:
                for meth in ("MILP", "Greedy"):
                    w = dec[(t0, mh, meth)]
                    mrows.append({"split": split, "t0": t0, "m_that": mt, "m_gia_dinh": mh, "phuong_phap": meth,
                                  "gia_tri_hien_thuc": float((w * Ur).sum()), "gia_tri_ky_vong": float((w * Ue).sum()),
                                  "z_sao_hien_thuc": zr, "z_sao_ky_vong": ze, "so_voucher": int(w[:, 1:].sum())})
            w = dec[(t0, None, "Luật")]
            mrows.append({"split": split, "t0": t0, "m_that": mt, "m_gia_dinh": np.nan, "phuong_phap": "Luật cũ",
                          "gia_tri_hien_thuc": float((w * Ur).sum()), "gia_tri_ky_vong": float((w * Ue).sum()),
                          "z_sao_hien_thuc": zr, "z_sao_ky_vong": ze, "so_voucher": int(w[:, 1:].sum())})
    msens = pd.DataFrame(mrows)
    msens.to_csv(common.p(TABLES / "s6_sensitivity_margin_instances.csv"), index=False)
    mag = msens.groupby(["m_that", "m_gia_dinh", "phuong_phap"], dropna=False).agg(
        gia_tri_hien_thuc_TB=("gia_tri_hien_thuc", "mean"), gia_tri_ky_vong_TB=("gia_tri_ky_vong", "mean"),
        z_sao_ky_vong_TB=("z_sao_ky_vong", "mean"), so_voucher_TB=("so_voucher", "mean")).reset_index()
    ok = mag[(mag.phuong_phap == "MILP") & (mag.m_that == mag.m_gia_dinh)].set_index("m_that").gia_tri_ky_vong_TB
    mag["mat_gia_tri_ky_vong_so_voi_gia_dinh_dung"] = [
        ok[r.m_that] - r.gia_tri_ky_vong_TB if r.phuong_phap == "MILP" else np.nan for r in mag.itertuples()]
    mag.to_csv(common.p(TABLES / "s6_sensitivity_margin.csv"), index=False)
    files += [str(TABLES / "s6_sensitivity_margin_instances.csv"), str(TABLES / "s6_sensitivity_margin.csv")]
    print(mag[mag.m_that == mag.m_gia_dinh].round(2).to_string())
    print(mag[(mag.phuong_phap == "MILP")].pivot(index="m_gia_dinh", columns="m_that", values="gia_tri_ky_vong_TB").round(1))

    # ---------------------------------------------------------------- 5. thời gian giải theo n (tới 50k)
    pool = np.where(df.split.isin(["train", "valid", "calib"]))[0]
    rng = np.random.default_rng(SEED)
    trows = []
    for n in N_GRID:
        for rep in range(3 if n <= 10000 else 1):
            idx = np.sort(rng.choice(pool, n, replace=False))
            ins = S5.build_instance(idx, meta, 3600, "pool", None, B=B0 * n / 3600, K=K0 * n / 3600)
            r = solve_hs(ins, V[idx])
            t = time.perf_counter(); wg = S5.greedy(ins, V[idx]); tg = time.perf_counter() - t
            row = {"n": n, "lan": rep, "B": ins["B"], "K": ins["K"], "highspy_trang_thai": r["status"], "highspy_gap": r["gap"],
                   "highspy_giay_dung": r["t_build"], "highspy_giay_giai": r["t_solve"], "greedy_giay": tg,
                   "V_MILP": r["obj"], "V_greedy": float((wg[:, 1:] * V[idx]).sum()),
                   "gap_greedy_vs_MILP": (r["obj"] - float((wg[:, 1:] * V[idx]).sum())) / abs(r["obj"]) if r["w"] is not None else np.nan}
            if n <= 10000:
                pr = solve_pulp(ins, V[idx])
                row.update(pulp_trang_thai=pr["status"], pulp_giay_dung=pr["t_build"], pulp_giay_giai=pr["t_solve"],
                           pulp_V=pr["obj"])
            trows.append(row)
            print(f"  n={n} lan {rep}: highspy {r['t_solve']:.2f}s ({r['status']}) greedy {tg:.3f}s ({time.time() - t_start:.0f}s)")
    tim = pd.DataFrame(trows)
    tim.to_csv(common.p(TABLES / "s6_scaling.csv"), index=False)
    files.append(str(TABLES / "s6_scaling.csv"))
    print(tim.groupby("n")[["highspy_giay_dung", "highspy_giay_giai", "greedy_giay", "highspy_gap", "gap_greedy_vs_MILP"]].mean().round(3))

    # ---------------------------------------------------------------- 6. thí nghiệm đinh: μ̂0 thô vs hiệu chỉnh
    crows = []
    mu_raw, mu_cal = df.mu0_raw.to_numpy(), df.mu0_cal.to_numpy()
    for split, t0 in per:
        ins = make_ins(df, meta, split, t0)
        idx = ins["idx"]
        for tag, mu in (("μ̂0 thô (chưa hiệu chỉnh)", mu_raw), ("μ̂0 đã hiệu chỉnh (Platt)", mu_cal)):
            Vp = value_hat_m(tau[idx], mu[idx], Ssh[idx], M0)
            w = solve_hs(ins, Vp)["w"]
            plan_cost = sum((w[:, k] * np.minimum(DISC[k] * Ssh[idx], CAP[k]) *
                             np.clip(mu[idx] + tau[idx, k - 1], 0, 1)).sum() for k in (1, 2, 3))
            plan_pbuy = sum((w[:, k] * np.clip(mu[idx] + tau[idx, k - 1], 0, 1)).sum() for k in (1, 2, 3)) / max(w[:, 1:].sum(), 1)
            act = evaluate_discount_costs(T, idx, w)
            ev_ = evaluate_policy(T, idx, w)
            crows.append({"split": split, "t0": t0, "loai_mu0": tag, "so_voucher": int(w[:, 1:].sum()),
                          "menh_gia_phat_hanh": float((w @ CAP).sum()), "chiet_khau_ke_hoach": float(plan_cost),
                          **act, "gia_tri_ke_hoach_V_hat": float((w[:, 1:] * Vp).sum()), **ev_,
                          "ty_le_mua_ke_hoach_TB_nguoi_nhan": float(plan_pbuy),
                          "mu0_TB_toan_ky": float(mu[idx].mean()), "ty_le_mua_a0_that_toan_ky": float(T["y0"][idx].mean())})
    ctab = pd.DataFrame(crows)
    ctab["thuc_tren_ke_hoach_ky_vong"] = ctab.chiet_khau_thuc_ky_vong / ctab.chiet_khau_ke_hoach
    ctab["thuc_tren_ke_hoach_hien_thuc"] = ctab.chiet_khau_thuc_hien_thuc / ctab.chiet_khau_ke_hoach
    ctab.to_csv(common.p(TABLES / "s6_calibration_discount_cost.csv"), index=False)
    cs = ctab.groupby("loai_mu0").agg(
        n_ky=("t0", "size"), so_voucher_TB=("so_voucher", "mean"), menh_gia_TB=("menh_gia_phat_hanh", "mean"),
        chiet_khau_ke_hoach_TB=("chiet_khau_ke_hoach", "mean"), chiet_khau_thuc_ky_vong_TB=("chiet_khau_thuc_ky_vong", "mean"),
        chiet_khau_thuc_hien_thuc_TB=("chiet_khau_thuc_hien_thuc", "mean"),
        ty_le_mua_ke_hoach=("ty_le_mua_ke_hoach_TB_nguoi_nhan", "mean"), ty_le_mua_that=("ty_le_mua_that_TB_nguoi_nhan", "mean"),
        V_ke_hoach_TB=("gia_tri_ke_hoach_V_hat", "mean"), gia_tri_that_hien_thuc_TB=("gia_tri_hien_thuc", "mean"),
        gia_tri_that_ky_vong_TB=("gia_tri_ky_vong", "mean")).reset_index()
    cs["thuc_tren_ke_hoach_ky_vong"] = cs.chiet_khau_thuc_ky_vong_TB / cs.chiet_khau_ke_hoach_TB
    cs["thuc_tren_ke_hoach_hien_thuc"] = cs.chiet_khau_thuc_hien_thuc_TB / cs.chiet_khau_ke_hoach_TB
    cs["lech_V_ke_hoach_vs_that_ky_vong"] = cs.V_ke_hoach_TB - cs.gia_tri_that_ky_vong_TB
    cs.to_csv(common.p(TABLES / "s6_calibration_discount_cost_summary.csv"), index=False)
    files += [str(TABLES / "s6_calibration_discount_cost.csv"), str(TABLES / "s6_calibration_discount_cost_summary.csv")]
    print(cs.round(3).T.to_string())

    files += plots(main_tab, dual_tab, agg, mag, tim, ctab)
    common.write_manifest("s6", files, extra={"B_grid": B_GRID, "K_grid": K_GRID, "m_grid": M_GRID, "n_grid": N_GRID,
                                              "time_limit": TIME_LIMIT, "gap": GAP, "runtime_s": round(time.time() - t_start)},
                          packages=["pandas", "numpy", "highspy", "PuLP", "matplotlib", "scikit-learn"])
    print(f"XONG GĐ6 ({time.time() - t_start:.0f}s)")


# ============================================================================ hình
def plots(main_tab, dual_tab, agg, mag, tim, ctab) -> list[str]:
    out = []
    # (1) so sánh chính
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    lab = [f"{pd.Timestamp(t).date()}\n({s})" for s, t in main_tab[main_tab.phuong_phap == METHODS[0]][["split", "t0"]].to_numpy()]
    x = np.arange(len(lab))
    for ax, col, ttl in [(axes[0], "gia_tri_ke_hoach_V_hat", "Giá trị KẾ HOẠCH (Σ V̂)"),
                         (axes[1], "gia_tri_ky_vong", "Giá trị THẬT kỳ vọng (đáp án)")]:
        for i, meth in enumerate(METHODS):
            d = main_tab[main_tab.phuong_phap == meth]
            ax.bar(x + (i - 1) * 0.27, d[col], 0.27, label=meth)
        ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7); ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("GBP / kỳ (toàn bộ khách của kỳ)"); ax.set_title(ttl); ax.legend(fontsize=8)
    fig.suptitle("MILP vs Greedy vs luật cũ trên toàn bộ khách mỗi t0 (B = 8 000, K = 60)")
    fig.tight_layout(); fig.savefig(common.p(FIGS / "fig_s6_main_comparison.png"), dpi=160); plt.close(fig)
    out.append(str(FIGS / "fig_s6_main_comparison.png"))
    # (2) độ nhạy B × K
    m = agg[(agg.phuong_phap == "MILP trên V̂") & agg.phu_UK_nonUK]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, col, ttl in [(axes[0], "V_ke_hoach_TB", "Giá trị kế hoạch Σ V̂ (GBP/kỳ)"),
                         (axes[1], "thuc_ky_vong_TB", "Giá trị thật kỳ vọng (GBP/kỳ)"),
                         (axes[2], "gia_bong_ngan_sach_TB", "Giá bóng ngân sách (GBP / GBP mệnh giá)")]:
        pv = m.pivot(index="K", columns="B", values=col)
        im = ax.imshow(pv.values, aspect="auto", cmap="viridis", origin="lower")
        ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns)
        ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = pv.values[i, j]
                ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}", ha="center", va="center",
                        color="w", fontsize=8)
        ax.set_xlabel("B (mệnh giá phát hành / kỳ)"); ax.set_ylabel("K (cuộc gọi / kỳ)"); ax.set_title(ttl)
        fig.colorbar(im, ax=ax)
    fig.suptitle("Độ nhạy MILP theo ngân sách B và công suất gọi K (trung bình 7 kỳ valid+calib; n/a = không khả thi)")
    fig.tight_layout(); fig.savefig(common.p(FIGS / "fig_s6_sensitivity_BK.png"), dpi=160); plt.close(fig)
    out.append(str(FIGS / "fig_s6_sensitivity_BK.png"))
    # (3) thời gian giải theo n
    g = tim.groupby("n").mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(g.index, g.highspy_giay_giai, "o-", label="MILP – HiGHS (giải)")
    ax.plot(g.index, g.highspy_giay_dung + g.highspy_giay_giai, "s--", label="MILP – HiGHS (dựng + giải)")
    gp = g.dropna(subset=["pulp_giay_giai"])
    ax.plot(gp.index, gp.pulp_giay_dung + gp.pulp_giay_giai, "^-", label="MILP – PuLP+HiGHS (dựng + giải)")
    ax.plot(g.index, g.greedy_giay, "d-", label="Greedy")
    ax.axhline(TIME_LIMIT, color="r", ls=":", label="giới hạn 60 s")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Số khách n"); ax.set_ylabel("Giây"); ax.legend(fontsize=8)
    ax.set_title("Thời gian giải theo quy mô (B, K ∝ n; gap 0.1 %)")
    fig.tight_layout(); fig.savefig(common.p(FIGS / "fig_s6_scaling.png"), dpi=160); plt.close(fig)
    out.append(str(FIGS / "fig_s6_scaling.png"))
    # (4) biên lợi nhuận
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    d = mag[(mag.phuong_phap == "MILP")].pivot(index="m_gia_dinh", columns="m_that", values="gia_tri_ky_vong_TB")
    for mt in d.columns:
        axes[0].plot(d.index, d[mt], "o-", label=f"m thật = {mt:.2f}")
    axes[0].set_xlabel("Biên m GIẢ ĐỊNH khi tính V̂"); axes[0].set_ylabel("Giá trị thật kỳ vọng TB (GBP/kỳ)")
    axes[0].set_title("Quyết định theo m̂ nhưng thế giới có m thật"); axes[0].legend(fontsize=8)
    sm = mag[mag.m_that == mag.m_gia_dinh]
    for meth, c in [("MILP", "tab:blue"), ("Greedy", "tab:green")]:
        q = sm[sm.phuong_phap == meth]
        axes[1].plot(q.m_that, q.gia_tri_ky_vong_TB, "o-", color=c, label=meth + " (m đúng)")
    q = mag[(mag.phuong_phap == "Luật cũ")].groupby("m_that").gia_tri_ky_vong_TB.mean()
    axes[1].plot(q.index, q.values, "s-", color="tab:red", label="Luật cũ")
    q = mag[(mag.phuong_phap == "MILP") & (mag.m_that == mag.m_gia_dinh)]
    axes[1].plot(q.m_that, q.z_sao_ky_vong_TB, "k--", label="Biết trước (kỳ vọng)")
    axes[1].set_xlabel("Biên lợi nhuận m (giả định = thật)"); axes[1].set_ylabel("Giá trị thật kỳ vọng TB (GBP/kỳ)")
    axes[1].set_title("Độ nhạy theo biên lợi nhuận 0.20–0.40"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(common.p(FIGS / "fig_s6_sensitivity_margin.png"), dpi=160); plt.close(fig)
    out.append(str(FIGS / "fig_s6_sensitivity_margin.png"))
    # (5) thí nghiệm đinh
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    labs = list(ctab.loai_mu0.unique())
    g = ctab.groupby("loai_mu0")[["chiet_khau_ke_hoach", "chiet_khau_thuc_ky_vong", "chiet_khau_thuc_hien_thuc"]].mean().loc[labs]
    xx = np.arange(len(labs))
    for i, (col, nm) in enumerate([("chiet_khau_ke_hoach", "Kế hoạch"), ("chiet_khau_thuc_ky_vong", "Thực tế (kỳ vọng)"),
                                   ("chiet_khau_thuc_hien_thuc", "Thực tế (hiện thực)")]):
        axes[0].bar(xx + (i - 1) * 0.27, g[col], 0.27, label=nm)
    axes[0].set_xticks(xx); axes[0].set_xticklabels(labs, fontsize=8); axes[0].set_ylabel("Chiết khấu trả cho người mua (GBP/kỳ)")
    axes[0].set_title("Chi phí chiết khấu: kế hoạch vs thực tế"); axes[0].legend(fontsize=8)
    g2 = ctab.groupby("loai_mu0")[["ty_le_mua_ke_hoach_TB_nguoi_nhan", "ty_le_mua_that_TB_nguoi_nhan"]].mean().loc[labs]
    for i, (col, nm) in enumerate([("ty_le_mua_ke_hoach_TB_nguoi_nhan", "Kế hoạch μ̂0+τ̂"),
                                   ("ty_le_mua_that_TB_nguoi_nhan", "Thật y0+τ")]):
        axes[1].bar(xx + (i - 0.5) * 0.35, g2[col], 0.35, label=nm)
    axes[1].set_xticks(xx); axes[1].set_xticklabels(labs, fontsize=8)
    axes[1].set_ylabel("Xác suất mua TB của người nhận voucher"); axes[1].set_title("Xác suất mua: kế hoạch vs thật"); axes[1].legend(fontsize=8)
    fig.suptitle("Thí nghiệm đinh: hiệu chỉnh μ̂0 và chi phí chiết khấu (7 kỳ valid+calib)")
    fig.tight_layout(); fig.savefig(common.p(FIGS / "fig_s6_calibration_discount_cost.png"), dpi=160); plt.close(fig)
    out.append(str(FIGS / "fig_s6_calibration_discount_cost.png"))
    return out


if __name__ == "__main__":
    from s6_milp import main as _main
    _main()
