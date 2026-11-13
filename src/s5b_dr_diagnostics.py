"""GĐ5b – Chẩn đoán bộ ước lượng DR/AIPW cho giá trị chính sách (chuẩn bị GĐ8).

ĐĂNG KÝ TRƯỚC (CLAUDE.md mục 6b – ghi trước khi chạy):
  Bộ ước lượng CHÍNH = AIPW KHÔNG winsorize: Γ_ia = ψ_a − ψ_0, ψ_k = μ̂_k + 1{A=k}(Y − μ̂_k)/e_k, Y = profit_obs,
  e = propensity GHI LOG, μ̂_k như s5_spo.py (LightGBM; ngoài fold trên train, học trên toàn train khi chấm valid/calib).
  Giá trị chính sách = Σ w·Γ; CI 95% bootstrap KHỐI theo customer_id (percentile, 2000 lần, multinomial trên khách).
  Lý do chọn: không chệch khi propensity đúng (propensity ghi log là đúng theo thiết kế) – KHÔNG chọn theo độ gần đáp án.
  Biến thể CHỈ để phân tích độ nhạy:
    · Winsorize Γ: cắt Γ theo phân vị 0.5%/99.5% của train (đúng ngưỡng s5_spo.py);
    · Winsorize phần dư: cắt Y − μ̂_k theo phân vị 0.5%/99.5% phần dư train của nhánh k;
    · Self-normalized: phần hiệu chỉnh của nhánh k chia cho Σ|g|·1{A=k}/e_k rồi nhân Σ|g| (kiểu Hájek);
    · Cắt trọng số: 1/e_k ≤ phân vị 99% của 1/e_k trên train (dự kiến gần như không đổi vì max 1/e = 7.2).
  Tham chiếu phân rã: DM (chỉ phần mô hình Σ w·(μ̂_a − μ̂_0)) và IPW thuần (Σ w·(1{A=a}Y/e_a − 1{A=0}Y/e_0)).
  Phân rã: ước lượng = phần mô hình (DM) + phần hiệu chỉnh IPW (ước lượng − DM).

Chính sách thử (VALID trước, xác nhận trên CALIB; không dùng locked/oot):
  · cấp dòng (không ràng buộc): "ai cũng a_k" (k = 1..3; DR phải ≈ ATE lợi nhuận thật), ngẫu nhiên đều (π = 1/4) –
    cho toàn tập và từng t0 (dùng làm đơn vị kiểm tra chệch);
  · cấp lô 100 khách: luật RFM cũ và MILP trên V̂ – ĐÚNG instance/MILP của s5_spo.py (5 seed), giá trị / lô;
  · quy mô GĐ8: luật cũ và MILP trên V̂ trên TOÀN BỘ khách của từng t0 (B, K không co; valid ≈ cỡ locked), giá trị / kỳ;
  · chênh lệch MILP V̂ − luật cũ, và MILP V̂ − 1.10 × luật cũ (biên của G4a).
Đáp án chỉ đọc trong evaluate_* để báo chệch, độ phủ CI – không dùng để chọn gì.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import time

import joblib
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor

import common
import s5_spo as S5
from s4_causal import FEATS_C, NUIS, gkf

CFG = common.load_config()
SEED = CFG["project"]["seed"]
TABLES, FIGS = Path(CFG["paths"]["tables"]), Path(CFG["paths"]["figures"])
ACTIONS, TREATED = S5.ACTIONS, S5.TREATED
B_BOOT = 2000
VARIANTS = ["AIPW (chính)", "Winsorize Γ", "Winsorize phần dư", "Self-normalized", "Cắt trọng số (p99)",
            "Tham chiếu: DM (chỉ mô hình)", "Tham chiếu: IPW thuần"]
COMP = ["DM", "C", "CR", "CW", "GW", "IPW"] + [f"{p}{k}" for p in "CWS" for k in range(4)]
FULL_T0_GAP = 1e-4      # MILP toàn t0: gap 0.01% (chặt hơn mức 0.1% của GĐ6)


# ============================================================================ dữ liệu & nuisance (y hệt s5_spo.py)
def load_data() -> pd.DataFrame:
    lg = pd.read_parquet(common.p("data/processed/logged.parquet"))
    lg = lg[lg.split.isin(["train", "valid", "calib"])].reset_index(drop=True)          # không đụng locked/oot
    p4 = pd.read_parquet(common.p("data/processed/pred_s4.parquet"))
    keep = ["customer_id", "t0"] + [f"{p}_{a}" for p in ("V_hat", "tau_hat") for a in TREATED] + ["mu0_base",
                                                                                                     "s_hat_mean"]
    df = lg.merge(p4[keep], on=["customer_id", "t0"], how="left", validate="one_to_one")
    assert df.V_hat_a1.notna().all()
    return df


def nuisance(df: pd.DataFrame) -> dict:
    """μ̂_k(x) của lợi nhuận: ngoài fold (GroupKFold theo khách) trên train; học trên toàn train cho valid/calib."""
    tr = df.split.eq("train").to_numpy()
    pre = joblib.load(common.p("models/s4_propensity.joblib"))["pre"]
    X = pre.transform(df[FEATS_C])
    A = df.A_logged.map({a: k for k, a in enumerate(ACTIONS)}).to_numpy()
    P = df[[f"propensity_{a}" for a in ACTIONS]].to_numpy()
    y = df.profit_obs.to_numpy(float)
    Mu = np.zeros((len(df), 4))
    itr = np.where(tr)[0]
    for f_tr, f_te in gkf(df.customer_id.to_numpy()[itr]):
        a_tr, a_te = itr[f_tr], itr[f_te]
        for k in range(4):
            fit = a_tr[A[a_tr] == k]
            Mu[a_te, k] = LGBMRegressor(**NUIS).fit(X[fit], y[fit]).predict(X[a_te])
    for k in range(4):
        Mu[~tr, k] = LGBMRegressor(**NUIS).fit(X[itr[A[itr] == k]], y[itr[A[itr] == k]]).predict(X[~tr])

    n = len(df)
    r = y - Mu[np.arange(n), A]                          # phần dư của nhánh đã ghi log
    w = 1.0 / P[np.arange(n), A]
    psi = np.column_stack([Mu[:, k] + (A == k) * (y - Mu[:, k]) / P[:, k] for k in range(4)])
    G = psi[:, 1:] - psi[:, [0]]
    g_lo, g_hi = np.percentile(G[tr], 0.5, axis=0), np.percentile(G[tr], 99.5, axis=0)   # = ngưỡng s5_spo.py
    r_lo, r_hi = np.zeros(4), np.zeros(4)
    w_cap = np.zeros(4)
    for k in range(4):
        rk = r[tr & (A == k)]
        r_lo[k], r_hi[k] = np.percentile(rk, 0.5), np.percentile(rk, 99.5)
        w_cap[k] = np.quantile(1.0 / P[tr, k], 0.99)
    thr = pd.DataFrame({"hanh_dong": ACTIONS, "nguong_phan_du_duoi": r_lo, "nguong_phan_du_tren": r_hi,
                        "tran_trong_so_p99": w_cap, "max_1_tren_e_train": (1 / P[tr]).max(0),
                        "nguong_Gamma_duoi": np.r_[np.nan, g_lo], "nguong_Gamma_tren": np.r_[np.nan, g_hi]})
    return dict(A=A, P=P, y=y, Mu=Mu, r=r, w=w, G=G, Gw=np.clip(G, g_lo, g_hi),
                r_cl=np.clip(r, r_lo[A], r_hi[A]), w_cl=np.minimum(w, w_cap[A]), thr=thr)


# ============================================================================ chính sách
def legacy_scores(df: pd.DataFrame) -> np.ndarray:
    L = CFG["intervention_dgp"]["legacy_logits"]
    r = df.groupby(["split", "t0"]).recency_days.rank(pct=True).to_numpy()
    m = df.groupby(["split", "t0"]).aov_365.rank(pct=True).to_numpy()
    return np.column_stack([L["a1_r"] * r - L["a0"], L["a2_mr"] * m * r - L["a0"], L["a3_m"] * m + L["a3_bias"] - L["a0"]])


def batch_policies(df: pd.DataFrame, legacy: np.ndarray, V: np.ndarray) -> tuple[dict, list]:
    """Tái tạo ĐÚNG instance của s5_spo.run_seed (cùng seed, cùng thứ tự sinh: train → valid → calib) và quyết định MILP."""
    meta = df[["split", "t0", S5.SEG_COL, "cancel_rate_365"]].copy()
    n = len(df)
    Q = {(s, p): np.zeros((n, 4)) for s in ("valid", "calib") for p in ("Luật RFM cũ", "MILP trên V̂ (GĐ4)")}
    n_inst = {"valid": 0, "calib": 0}
    log = []
    for seed in S5.SEEDS:
        rng = np.random.default_rng(seed)
        insts = {s: S5.make_instances(meta, s, S5.N_INST[s], rng) for s in ["train", "valid", "calib"]}
        for s in ("valid", "calib"):
            for j, ins in enumerate(insts[s]):
                m = S5.MILP(ins)
                for p, sc in (("Luật RFM cũ", legacy), ("MILP trên V̂ (GĐ4)", V)):
                    wdec = m.solve(S5.cmat(sc[ins["idx"]]))
                    Q[(s, p)][ins["idx"]] += wdec
                    log.append({"split": s, "seed": seed, "instance": j, "phuong_phap": p, "idx": ins["idx"], "w": wdec})
                n_inst[s] += 1
    for (s, p) in Q:
        Q[(s, p)] /= n_inst[s]                          # giá trị = trung bình / lô (giống cột gia_tri_DR của s5)
    return Q, log


def full_t0_policies(df: pd.DataFrame, legacy: np.ndarray, V: np.ndarray, split: str) -> tuple[dict, pd.DataFrame]:
    """Quy mô GĐ8: MILP trên TOÀN BỘ khách của mỗi t0 (B, K, độ phủ như s5 nhưng không co). Giá trị = trung bình / kỳ."""
    meta = df[["split", "t0", S5.SEG_COL, "cancel_rate_365"]].copy()
    t0s = np.sort(df[df.split == split].t0.unique())
    Q = {p: np.zeros((len(df), 4)) for p in ("Luật RFM cũ", "MILP trên V̂ (GĐ4)")}
    rows = []
    for t0 in t0s:
        idx = np.where((df.split == split) & (df.t0 == t0))[0]
        ins = S5.build_instance(idx, meta, len(idx), split, t0)
        m = S5.MILP(ins)
        m.h.setOptionValue("mip_rel_gap", FULL_T0_GAP)
        m.h.setOptionValue("time_limit", 120.0)
        for p, sc in (("Luật RFM cũ", legacy), ("MILP trên V̂ (GĐ4)", V)):
            t = time.perf_counter()
            wdec = m.solve(S5.cmat(sc[idx]))
            chk = S5.check_feasible(ins, wdec)
            rows.append({"split": split, "t0": t0, "phuong_phap": p, "n_khach": len(idx), "giay_giai": time.perf_counter() - t,
                         **{k: chk[k] for k in ("ngan_sach_dung", "ngan_sach_B", "so_voucher", "so_cuoc_goi", "so_a2",
                                                "n_vi_pham")}})
            Q[p][idx] = wdec / len(t0s)
    return Q, pd.DataFrame(rows)


def row_policies(df: pd.DataFrame, split: str) -> dict:
    """Chính sách cấp dòng: ai cũng a_k (k=1..3) và ngẫu nhiên đều – cho toàn tập và từng t0. Giá trị = TB / dòng."""
    out = {}
    for unit, mask in [("toàn tập", df.split.eq(split).to_numpy())] + \
            [(str(pd.Timestamp(t).date()), (df.split.eq(split) & df.t0.eq(t)).to_numpy())
             for t in np.sort(df[df.split == split].t0.unique())]:
        nrow = mask.sum()
        for k, a in enumerate(TREATED, start=1):
            Q = np.zeros((len(df), 4)); Q[mask, k] = 1.0 / nrow
            out[(unit, f"Ai cũng {a}")] = Q
        Q = np.zeros((len(df), 4)); Q[mask, :] = 0.25 / nrow
        out[(unit, "Ngẫu nhiên đều")] = Q
    return out


# ============================================================================ ước lượng + bootstrap khối
def components(Q: np.ndarray, d: dict) -> np.ndarray:
    """Đóng góp từng dòng vào các tổng cần cho mọi biến thể (cột theo COMP)."""
    n = len(Q)
    q = Q[:, 1:]
    g = np.column_stack([-q.sum(1), q])                  # tương phản so với a0
    A = d["A"]
    gA = g[np.arange(n), A]
    cols = {"DM": (g * d["Mu"]).sum(1), "C": gA * d["r"] * d["w"], "CR": gA * d["r_cl"] * d["w"],
            "CW": gA * d["r"] * d["w_cl"], "GW": (q * d["Gw"]).sum(1), "IPW": gA * d["y"] * d["w"]}
    for k in range(4):
        cols[f"C{k}"] = np.where(A == k, gA * d["r"] * d["w"], 0.0)
        cols[f"W{k}"] = np.where(A == k, np.abs(gA) * d["w"], 0.0)
        cols[f"S{k}"] = np.abs(g[:, k])
    return np.column_stack([cols[c] for c in COMP])


def variant_values(s: dict) -> dict:
    sn = s["DM"] + sum(np.where(s[f"W{k}"] != 0, s[f"C{k}"] * s[f"S{k}"] / np.where(s[f"W{k}"] != 0, s[f"W{k}"], 1), 0)
                       for k in range(4))
    return {"AIPW (chính)": s["DM"] + s["C"], "Winsorize Γ": s["GW"], "Winsorize phần dư": s["DM"] + s["CR"],
            "Self-normalized": sn, "Cắt trọng số (p99)": s["DM"] + s["CW"], "Tham chiếu: DM (chỉ mô hình)": s["DM"],
            "Tham chiếu: IPW thuần": s["IPW"]}


def estimate_split(df, d, split: str, pols: dict, rng) -> tuple[pd.DataFrame, dict, np.ndarray, list]:
    """pols: tên → (quy_mo, đơn vị, Q). Bootstrap khối: lấy lại khách (multinomial) – cùng trọng số cho mọi chính sách."""
    m_split = df.split.eq(split).to_numpy()
    rows_idx = np.where(m_split)[0]
    codes, uniq = pd.factorize(df.customer_id.to_numpy()[rows_idx])
    nc = len(uniq)
    Ind = sp.csr_matrix((np.ones(len(rows_idx)), (codes, np.arange(len(rows_idx)))), shape=(nc, len(rows_idx)))
    names = list(pols)
    R = np.column_stack([components(pols[k][2][rows_idx], {kk: (v[rows_idx] if isinstance(v, np.ndarray) else v)
                                                             for kk, v in d.items() if kk != "thr"}) for k in names])
    Mc = np.asarray(Ind @ R)                                             # nc × (|COMP|·|pols|)
    W = rng.multinomial(nc, np.full(nc, 1.0 / nc), size=B_BOOT).astype(np.float32)
    point, boot = Mc.sum(0), W @ Mc.astype(np.float32)
    nC = len(COMP)
    vals, out = {}, []
    for j, k in enumerate(names):
        sl = slice(j * nC, (j + 1) * nC)
        vp = variant_values(dict(zip(COMP, point[sl])))
        vb = variant_values(dict(zip(COMP, boot[:, sl].T.astype(float))))
        vals[k] = (vp, vb)
    return vals, Mc, codes, names


def summarize(split, pols, vals) -> pd.DataFrame:
    rows = []
    for k, (vp, vb) in vals.items():
        quy_mo, don_vi = pols[k][0], pols[k][1]
        for v in VARIANTS:
            rows.append({"split": split, "quy_mo": quy_mo, "chinh_sach": k[1] if isinstance(k, tuple) else k,
                         "don_vi_tinh": k[0] if isinstance(k, tuple) else "", "don_vi": don_vi, "bien_the": v,
                         "uoc_luong": vp[v], "ci_lo": np.percentile(vb[v], 2.5), "ci_hi": np.percentile(vb[v], 97.5),
                         "se_bootstrap": float(np.std(vb[v], ddof=1)),
                         "phan_mo_hinh_DM": vp["Tham chiếu: DM (chỉ mô hình)"],
                         "phan_hieu_chinh_IPW": vp[v] - vp["Tham chiếu: DM (chỉ mô hình)"]})
    return pd.DataFrame(rows)


def add_contrasts(split, pols, vals, scale_key, label) -> list[dict]:
    """Chênh lệch MILP V̂ − luật cũ và MILP V̂ − 1.10 × luật cũ (biên G4a), cùng trọng số bootstrap."""
    a, b = (scale_key, "MILP trên V̂ (GĐ4)"), (scale_key, "Luật RFM cũ")
    rows = []
    for nm, f in [("MILP V̂ − luật cũ", 1.0), ("MILP V̂ − 1.10 × luật cũ (biên G4a)", 1.1)]:
        for v in VARIANTS:
            pe = vals[a][0][v] - f * vals[b][0][v]
            be = vals[a][1][v] - f * vals[b][1][v]
            rows.append({"split": split, "quy_mo": label, "chinh_sach": nm, "don_vi_tinh": scale_key,
                         "don_vi": pols[a][1], "bien_the": v, "uoc_luong": pe, "ci_lo": np.percentile(be, 2.5),
                         "ci_hi": np.percentile(be, 97.5), "se_bootstrap": float(np.std(be, ddof=1)),
                         "phan_mo_hinh_DM": vals[a][0]["Tham chiếu: DM (chỉ mô hình)"] -
                         f * vals[b][0]["Tham chiếu: DM (chỉ mô hình)"],
                         "phan_hieu_chinh_IPW": pe - (vals[a][0]["Tham chiếu: DM (chỉ mô hình)"] -
                                                      f * vals[b][0]["Tham chiếu: DM (chỉ mô hình)"])})
    return rows


def heavy_tail(split, Mc, names, pols) -> list[dict]:
    """Đuôi dày (không dùng đáp án): tỷ trọng |phần hiệu chỉnh| do 10 khách lớn nhất đóng góp."""
    nC, iC = len(COMP), COMP.index("C")
    rows = []
    for j, k in enumerate(names):
        c = Mc[:, j * nC + iC]
        a = np.sort(np.abs(c))[::-1]
        tot = a.sum()
        if tot == 0:
            continue
        rows.append({"split": split, "quy_mo": pols[k][0], "chinh_sach": k[1], "don_vi_tinh": k[0],
                     "so_khach_dong_gop": int((a > 0).sum()), "ty_trong_top10_khach": a[:10].sum() / tot,
                     "ty_trong_top1pct_khach": a[:max(1, int(0.01 * (a > 0).sum()))].sum() / tot,
                     "dong_gop_lon_nhat": float(c[np.argmax(np.abs(c))]), "tong_hieu_chinh": float(c.sum()),
                     "tong_abs_hieu_chinh": float(tot)})
    return rows


# ============================================================================ ĐÁNH GIÁ (được phép đọc đáp án)
def evaluate_truth(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return S5.evaluate_load_costs(df)                                     # U_real (A), U_exp (B) – n × 4


def evaluate_attach_truth(est: pd.DataFrame, pols_all: dict, U_real, U_exp) -> pd.DataFrame:
    def tv(Q, U):
        return float((Q[:, 1:] * U[:, 1:]).sum())
    out = est.copy()
    tA, tB = [], []
    for r in out.itertuples():
        if r.chinh_sach.startswith("MILP V̂ −"):
            f = 1.1 if "1.10" in r.chinh_sach else 1.0
            qa, qb = pols_all[(r.split, (r.don_vi_tinh, "MILP trên V̂ (GĐ4)"))], pols_all[(r.split, (r.don_vi_tinh,
                                                                                                   "Luật RFM cũ"))]
            tA.append(tv(qa, U_real) - f * tv(qb, U_real)); tB.append(tv(qa, U_exp) - f * tv(qb, U_exp))
        else:
            Q = pols_all[(r.split, (r.don_vi_tinh, r.chinh_sach))]
            tA.append(tv(Q, U_real)); tB.append(tv(Q, U_exp))
    out["gia_tri_that_A_hien_thuc"], out["gia_tri_that_B_ky_vong"] = tA, tB
    for s in ("A_hien_thuc", "B_ky_vong"):
        t = out[f"gia_tri_that_{s}"]
        out[f"ci_phu_{s}"] = (out.ci_lo <= t) & (t <= out.ci_hi)
        out[f"z_{s}"] = (out.uoc_luong - t) / out.se_bootstrap
    out["sai_lech_A"] = out.uoc_luong - out.gia_tri_that_A_hien_thuc
    out["DM_tru_that_A"] = out.phan_mo_hinh_DM - out.gia_tri_that_A_hien_thuc
    return out


def evaluate_bias_units(tab: pd.DataFrame) -> pd.DataFrame:
    """Đơn vị kiểm tra chệch: chính sách cấp dòng theo TỪNG t0 (4 chính sách × 7 kỳ valid+calib)."""
    u = tab[(tab.quy_mo == "cấp dòng") & (tab.don_vi_tinh != "toàn tập")]
    rows = []
    for (s, v), g in u.groupby(["split", "bien_the"], sort=False):
        rows.append({"split": s, "bien_the": v, "so_don_vi": len(g),
                     "do_phu_CI_A": g.ci_phu_A_hien_thuc.mean(), "do_phu_CI_B": g.ci_phu_B_ky_vong.mean(),
                     "z_TB_A": g.z_A_hien_thuc.mean(), "z_sd_A": g.z_A_hien_thuc.std(ddof=1),
                     "z_TB_B": g.z_B_ky_vong.mean(), "sai_lech_TB_GBP_dong": g.sai_lech_A.mean(),
                     "se_TB_GBP_dong": g.se_bootstrap.mean()})
    return pd.DataFrame(rows)


def evaluate_consistency_s5(log: list, U_real) -> pd.DataFrame:
    """Kiểm tra cùng instance & cùng quyết định với s5_spo.py (calib): giá trị hiện thực từng lô phải trùng s5."""
    r5 = pd.read_csv(TABLES / "s5_regret_instances.csv")
    mine = pd.DataFrame([{"seed": e["seed"], "instance": e["instance"], "phuong_phap": e["phuong_phap"],
                          "z_moi": float((e["w"] * U_real[e["idx"]]).sum())} for e in log if e["split"] == "calib"])
    m = mine.merge(r5[["seed", "instance", "phuong_phap", "z_hien_thuc"]], on=["seed", "instance", "phuong_phap"])
    return pd.DataFrame([{"so_lo_so_khop": len(m), "so_lo_ky_vong": len(mine),
                          "max_lech_gia_tri_hien_thuc": float((m.z_moi - m.z_hien_thuc).abs().max())}])


def evaluate_plot(tab: pd.DataFrame, path: Path) -> None:
    keep = tab[(tab.don_vi_tinh.isin(["toàn tập", "lô", "kỳ"])) & tab.bien_the.isin(VARIANTS[:5] + VARIANTS[6:])]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), sharey=True)
    pol_order = list(dict.fromkeys(keep.quy_mo + " · " + keep.chinh_sach))
    cmap = plt.get_cmap("tab10")
    for ax, s in zip(axes, ["valid", "calib"]):
        d = keep[keep.split == s]
        for vi, v in enumerate(VARIANTS[:5] + VARIANTS[6:]):
            dv = d[d.bien_the == v]
            y = [pol_order.index(a) + (vi - 2.5) * 0.12 for a in dv.quy_mo + " · " + dv.chinh_sach]
            ax.plot(dv.z_A_hien_thuc.clip(-8, 8), y, "o", ms=5, color=cmap(vi), label=v)
        ax.axvspan(-1.96, 1.96, color="0.9", zorder=0)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{s}: z = (ước lượng − giá trị thật A) / SE bootstrap")
        ax.set_xlabel("z (vùng xám = CI 95% phủ giá trị thật; cắt ở ±8)")
    axes[0].set_yticks(range(len(pol_order))); axes[0].set_yticklabels(pol_order, fontsize=8)
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Chẩn đoán DR/AIPW: ước lượng có nằm trong CI quanh giá trị thật không (đáp án chỉ dùng để đánh giá)")
    fig.tight_layout()
    fig.savefig(path, dpi=160); plt.close(fig)


# ============================================================================ chính
def main() -> None:
    np.random.seed(SEED)
    t_start = time.time()
    files: list[str] = []
    pd.set_option("display.width", 260)
    df = load_data()
    d = nuisance(df)
    d["thr"].to_csv(common.p(TABLES / "s5b_thresholds.csv"), index=False)
    files.append(str(TABLES / "s5b_thresholds.csv"))
    print(d["thr"].round(3).to_string())
    legacy = legacy_scores(df)
    V = df[[f"V_hat_{a}" for a in TREATED]].to_numpy()

    Qb, log = batch_policies(df, legacy, V)
    print(f"tái tạo instance s5 xong ({time.time() - t_start:.0f}s)")
    est_rows, pols_all, tail, solves = [], {}, [], []
    rng = np.random.default_rng(SEED)
    for split in ("valid", "calib"):
        pols = {k: ("cấp dòng", "GBP / dòng (khách × kỳ)", Q) for k, Q in row_policies(df, split).items()}
        for p in ("Luật RFM cũ", "MILP trên V̂ (GĐ4)"):
            pols[("lô", p)] = ("cấp lô 100 khách (instance s5)", "GBP / lô", Qb[(split, p)])
        Qf, sv = full_t0_policies(df, legacy, V, split)
        solves.append(sv)
        for p, Q in Qf.items():
            pols[("kỳ", p)] = ("quy mô GĐ8 (toàn bộ t0)", "GBP / kỳ", Q)
        vals, Mc, codes, names = estimate_split(df, d, split, pols, rng)
        est = summarize(split, pols, vals)
        est = pd.concat([est, pd.DataFrame(add_contrasts(split, pols, vals, "lô", "cấp lô 100 khách (instance s5)") +
                                           add_contrasts(split, pols, vals, "kỳ", "quy mô GĐ8 (toàn bộ t0)"))])
        est_rows.append(est)
        tail += heavy_tail(split, Mc, names, pols)
        for k, (_, _, Q) in pols.items():
            pols_all[(split, k)] = Q
        print(f"{split} xong ({time.time() - t_start:.0f}s)")
    est = pd.concat(est_rows, ignore_index=True)
    est.to_csv(common.p(TABLES / "s5b_estimates.csv"), index=False)
    sv = pd.concat(solves, ignore_index=True)
    sv.to_csv(common.p(TABLES / "s5b_full_t0_milp.csv"), index=False)
    tl = pd.DataFrame(tail)
    tl.to_csv(common.p(TABLES / "s5b_heavy_tail.csv"), index=False)
    files += [str(TABLES / f) for f in ("s5b_estimates.csv", "s5b_full_t0_milp.csv", "s5b_heavy_tail.csv")]
    print(sv.round(3).to_string())
    print(tl[tl.quy_mo != "cấp dòng"].round(3).to_string())

    # ---- đánh giá bằng đáp án (chỉ evaluate_*)
    U_real, U_exp = evaluate_truth(df)
    tab = evaluate_attach_truth(est, pols_all, U_real, U_exp)
    tab.to_csv(common.p(TABLES / "s5b_vs_truth.csv"), index=False)
    bias = evaluate_bias_units(tab)
    bias.to_csv(common.p(TABLES / "s5b_bias_units.csv"), index=False)
    cons = evaluate_consistency_s5(log, U_real)
    cons.to_csv(common.p(TABLES / "s5b_consistency_s5.csv"), index=False)
    evaluate_plot(tab, common.p(FIGS / "fig_s5b_dr_z_scores.png"))
    files += [str(TABLES / f) for f in ("s5b_vs_truth.csv", "s5b_bias_units.csv", "s5b_consistency_s5.csv")]
    files.append(str(FIGS / "fig_s5b_dr_z_scores.png"))
    show = ["split", "quy_mo", "chinh_sach", "bien_the", "uoc_luong", "ci_lo", "ci_hi", "phan_mo_hinh_DM",
            "phan_hieu_chinh_IPW", "gia_tri_that_A_hien_thuc", "gia_tri_that_B_ky_vong", "ci_phu_A_hien_thuc",
            "z_A_hien_thuc"]
    main_tab = tab[(tab.don_vi_tinh.isin(["toàn tập", "lô", "kỳ"]))]
    print(main_tab[main_tab.bien_the == "AIPW (chính)"][show].round(2).to_string())
    print(bias.round(3).to_string())
    print(cons.to_string())

    common.write_manifest("s5b", files, extra={"B_boot": B_BOOT, "seeds_s5": S5.SEEDS, "n_inst_s5": S5.N_INST,
                                               "full_t0_gap": FULL_T0_GAP, "runtime_s": round(time.time() - t_start)},
                          packages=["pandas", "numpy", "scikit-learn", "lightgbm", "highspy", "scipy", "matplotlib"])
    print(f"XONG GĐ5b ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    from s5b_dr_diagnostics import main as _main
    _main()
