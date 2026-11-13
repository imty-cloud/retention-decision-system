"""GĐ5 – Predict-Then-Optimize & SPO+ (Chương 5).

ĐĂNG KÝ TRƯỚC (ghi trước khi chạy, không chọn lại sau khi xem regret):
 1. Instance = lô 100 khách lấy ngẫu nhiên (không hoàn lại) trong CÙNG một t0 của một split. B, K co theo 100/n_t0
    (n_t0 = số khách của split đó tại t0). Ràng buộc (CLAUDE.md mục 1): mỗi khách đúng 1 hành động; Σ cap_a·x ≤ B;
    Σ x_a3 ≤ K; số a2 ≤ 25% số voucher; độ phủ tối thiểu mỗi phân khúc UK/non-UK = ceil(min_coverage_per_segment × số
    khách), không vượt số khách đủ điều kiện; x = 0 cho mọi voucher nếu cancel_rate_365 > guardrail_cancel_rate_max.
    Miền khả thi chỉ phụ thuộc thiết kế/đặc trưng, KHÔNG phụ thuộc dự báo ⇒ đúng giả định của SPO+.
 2. MỘT solver cho mọi phương pháp: HiGHS (highspy gọi trực tiếp, chỉ đổi hệ số mục tiêu giữa các lần giải).
    Đối chiếu với PuLP + CBC (mã ràng buộc viết độc lập) trên mẫu instance. Greedy là heuristic nhưng được kiểm tra
    khả thi bằng cùng bộ kiểm tra ràng buộc.
 3. Vector chi phí huấn luyện: pseudo-outcome AIPW của lợi nhuận gia tăng so với a0,
      Γ_ia = μ̂_a − μ̂_0 + 1{A=a}(Y − μ̂_a)/e_a − 1{A=a0}(Y − μ̂_0)/e_0,  Y = profit_obs, e = propensity GHI LOG,
    μ̂_a: LightGBM cross-fit GroupKFold(5) theo khách trên train (ngoài fold). Valid/calib: μ̂_a học trên toàn train
    (không khớp gì trên valid/calib). Γ winsorize ở phân vị 0.5%/99.5% của train (cùng ngưỡng cho PTO và SPO+).
 4. Lớp giả thuyết tuyến tính ĉ_ia = θ_aᵀ z_i (a1..a3; a0 ≡ 0), z = đặc trưng GĐ4 (không lịch) + đầu ra GĐ4
    (V̂, τ̂, μ̂0 nền, log Ŝ), chuẩn hóa trên train.
      PTO tuyến tính  : ridge (MSE) trên mọi dòng train.          PTO LightGBM: đối chứng phi tuyến (MSE).
      SPO+ tuyến tính : cùng lớp, khởi tạo từ PTO tuyến tính, Adam, minibatch 10 instance, L2 = 1e-3.
    Bài toán là MAX nên subgradient theo ĉ của SPO+ là 2(w*(2ĉ − c) − w*(c)), w* = argmax; đây chính là
    2(w*(c) − w*(2ĉ − c)) của bài toán MIN với chi phí −c, đổi dấu một lần khi quay về thang max.
 5. Chọn siêu tham số trên VALID theo tiêu chí riêng của từng phương pháp (không dùng đáp án):
      PTO → MSE với Γ (winsorize); SPO+ → SPO+ loss trung bình với Γ (lưới lr × epoch 1..15).
    Báo cáo trên CALIB (chưa dùng để chọn). Không dùng locked/oot. 5 seed (lấy mẫu instance + thứ tự SGD).
 6. Đánh giá CHỈ trong evaluate_*: z(w) = Σ w·(lợi nhuận gia tăng so với a0); regret chuẩn hóa = (z* − z_π)/|z*|
      (A) theo lợi nhuận HIỆN THỰC profit_a − profit_a0 (đặc tả);
      (B) theo lợi nhuận KỲ VỌNG có điều kiện – lấy kỳ vọng theo số ngẫu nhiên U của cơ chế can thiệp (phần may rủi
          không mô hình nào biết trước), báo cáo kèm để tách nhiễu không thể giảm.
    Kèm giá trị ước lượng DR (không dùng đáp án) cho mỗi chính sách.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import itertools
import math
import os
import time

import highspy
import joblib
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pulp
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

import common
from s4_causal import FEATS_C, NUIS, gkf

CFG = common.load_config()
SEED = CFG["project"]["seed"]
TABLES, FIGS = Path(CFG["paths"]["tables"]), Path(CFG["paths"]["figures"])
BIZ = CFG["business"]
ACTIONS = ["a0", "a1", "a2", "a3"]
TREATED = ACTIONS[1:]
CAP = np.array([BIZ["actions"][a]["face_value_cap"] for a in ACTIONS], float)
DISC = np.array([BIZ["actions"][a]["discount"] for a in ACTIONS], float)
CONTACT = np.array([BIZ["actions"][a]["contact_cost"] for a in ACTIONS], float)
MARGIN = BIZ["gross_margin_rate"]
COV = BIZ["min_coverage_per_segment"]
SEG_COL = BIZ["coverage_segment_col"]
GR_CANCEL = BIZ["guardrail_cancel_rate_max"]
A2_SHARE = BIZ["max_share_deep_voucher"]
BATCH = 100
N_INST = {"train": 300, "valid": 100, "calib": 120}
SEEDS = [SEED + k for k in range(5)]
LRS = [0.003, 0.01, 0.03]
EPOCHS, MB, L2 = 15, 10, 1e-3
RIDGE_ALPHAS = [1, 10, 100, 1e3, 1e4, 1e5]
LGBM_GRID = [(4, 200), (8, 300), (16, 300)]
if os.environ.get("S5_NHANH") == "1":       # chế độ kiểm thử nhanh (worker loky kế thừa biến môi trường)
    N_INST, SEEDS, EPOCHS, LRS = {"train": 20, "valid": 10, "calib": 12}, [SEED, SEED + 1], 1, [0.01]
LUOI = os.environ.get("S5_LUOI", "goc")    # "goc" = lưới gốc; "mo_rong" = lưới mở rộng theo quy tắc biên (CLAUDE.md GĐ5)
assert LUOI in ("goc", "mo_rong")
SFX = "" if LUOI == "goc" else "_mo_rong"
PATIENCE = 3
METHODS = ["Luật RFM cũ", "Nhắm theo rủi ro (μ̂0 thấp)", "Greedy (V̂/mệnh giá)", "MILP trên V̂ (GĐ4)",
           "PTO tuyến tính (MSE)", "PTO LightGBM (MSE)", "SPO+ tuyến tính"]


# ============================================================================ instance & ràng buộc
def build_instance(idx, meta: pd.DataFrame, n_t0: int | None, split="", t0=None, B=None, K=None, req=None) -> dict:
    is_uk = meta[SEG_COL].to_numpy()[idx] == 1
    blocked = meta.cancel_rate_365.fillna(0).to_numpy()[idx] > GR_CANCEL
    n = len(idx)
    B = BIZ["budget_face_value_per_t0"] * n / n_t0 if B is None else B
    K = BIZ["call_capacity_per_t0"] * n / n_t0 if K is None else K
    if req is None:
        req = []
        for seg in (True, False):
            ns, el = int((is_uk == seg).sum()), int(((is_uk == seg) & ~blocked).sum())
            req.append(min(math.ceil(COV * ns), el) if ns else 0)
    return dict(split=split, t0=t0, idx=np.asarray(idx), n=n, B=float(B), K=float(K), is_uk=is_uk, blocked=blocked,
                req_uk=int(req[0]), req_non=int(req[1]))


def make_instances(meta: pd.DataFrame, split: str, n_inst: int, rng) -> list[dict]:
    sub = meta[meta.split == split]
    t0s = np.sort(sub.t0.unique())
    pools = {t: sub.index[sub.t0 == t].to_numpy() for t in t0s}
    out = []
    for k in range(n_inst):
        t0 = t0s[k % len(t0s)]
        idx = rng.choice(pools[t0], BATCH, replace=False)
        out.append(build_instance(idx, meta, len(pools[t0]), split, t0))
    return out


def check_feasible(inst: dict, w: np.ndarray) -> dict:
    """Bộ kiểm tra ràng buộc ĐỘC LẬP với solver – áp cho mọi phương pháp, kể cả Greedy."""
    w = np.rint(w)
    nv, na2 = w[:, 1:].sum(), w[:, 2].sum()
    vp = {"vp_mot_hanh_dong": not np.all(w.sum(1) == 1), "vp_ngan_sach": (w @ CAP).sum() > inst["B"] + 1e-6,
          "vp_cuoc_goi": w[:, 3].sum() > inst["K"] + 1e-6, "vp_ty_le_a2": na2 > A2_SHARE * nv + 1e-9,
          "vp_do_phu_UK": w[inst["is_uk"], 1:].sum() < inst["req_uk"],
          "vp_do_phu_nonUK": w[~inst["is_uk"], 1:].sum() < inst["req_non"],
          "vp_guardrail": w[inst["blocked"], 1:].sum() > 0}
    return {"ngan_sach_dung": (w @ CAP).sum(), "ngan_sach_B": inst["B"], "so_cuoc_goi": w[:, 3].sum(),
            "so_voucher": nv, "so_a2": na2, "n_vi_pham": int(sum(vp.values())), **vp}


class MILP:
    """max Σ c_ia x_ia trên miền khả thi CỐ ĐỊNH của instance; HiGHS; chỉ đổi hệ số mục tiêu giữa các lần giải."""

    def __init__(self, inst: dict):
        n = inst["n"]
        self.n, self.N = n, 4 * n
        r, c, v = [], [], []
        for i in range(n):
            for a in range(4):
                r.append(i); c.append(4 * i + a); v.append(1.0)                    # mỗi khách đúng 1 hành động
        rb, rk, rs, ru, rn = n, n + 1, n + 2, n + 3, n + 4
        for i in range(n):
            for a in (1, 2, 3):
                r.append(rb); c.append(4 * i + a); v.append(CAP[a])               # ngân sách mệnh giá
                r.append(rs); c.append(4 * i + a); v.append(3.0 if a == 2 else -1.0)  # 3·x2 − x1 − x3 ≤ 0 (a2 ≤ 25%)
                r.append(ru if inst["is_uk"][i] else rn); c.append(4 * i + a); v.append(1.0)  # độ phủ phân khúc
            r.append(rk); c.append(4 * i + 3); v.append(1.0)                       # năng lực gọi điện
        M = sp.csc_matrix((v, (r, c)), shape=(n + 5, self.N))
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = self.N, n + 5
        lp.col_cost_ = np.zeros(self.N)
        lp.col_lower_ = np.zeros(self.N)
        ub = np.ones((n, 4))
        ub[inst["blocked"], 1:] = 0.0                                               # guardrail
        lp.col_upper_ = ub.ravel()
        inf = highspy.kHighsInf
        lp.row_lower_ = np.r_[np.ones(n), -inf, -inf, -inf, inst["req_uk"], inst["req_non"]]
        lp.row_upper_ = np.r_[np.ones(n), inst["B"], inst["K"], 0.0, inf, inf]
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = M.indptr, M.indices, M.data
        lp.integrality_ = [highspy.HighsVarType.kInteger] * self.N
        h = highspy.Highs()
        for k, val in [("output_flag", False), ("mip_rel_gap", 1e-7), ("threads", 1), ("time_limit", 30.0)]:
            h.setOptionValue(k, val)
        h.passModel(lp)
        self.h, self.cols = h, np.arange(self.N, dtype=np.int32)

    def solve(self, c: np.ndarray) -> np.ndarray:
        self.h.changeColsCost(self.N, self.cols, -np.asarray(c, float).ravel())
        self.h.run()
        assert self.h.getModelStatus() == highspy.HighsModelStatus.kOptimal, self.h.getModelStatus()
        return np.rint(np.array(self.h.getSolution().col_value)).reshape(self.n, 4)


def cmat(c3: np.ndarray) -> np.ndarray:
    return np.column_stack([np.zeros(len(c3)), c3])


def greedy(inst: dict, V: np.ndarray) -> np.ndarray:
    """Greedy theo V̂/mệnh giá: (1) phủ tối thiểu từng phân khúc bằng cặp có tỷ số cao nhất; (2) thêm các cặp V̂ > 0."""
    n, B, K = inst["n"], inst["B"], inst["K"]
    w = np.zeros((n, 4)); w[:, 0] = 1
    st = {"bud": 0.0, "calls": 0, "nv": 0, "na2": 0}
    ratio = V / CAP[1:]

    def can(i, a):
        na2 = st["na2"] + (a == 2)
        return (w[i, 0] == 1 and not inst["blocked"][i] and st["bud"] + CAP[a] <= B + 1e-9
                and (a != 3 or st["calls"] + 1 <= K + 1e-9) and na2 <= A2_SHARE * (st["nv"] + 1) + 1e-9)

    def add(i, a):
        w[i] = 0; w[i, a] = 1
        st["bud"] += CAP[a]; st["calls"] += a == 3; st["nv"] += 1; st["na2"] += a == 2

    for seg, req in [(True, inst["req_uk"]), (False, inst["req_non"])]:
        members = np.where(inst["is_uk"] == seg)[0]
        cand = sorted(((ratio[i, a - 1], i, a) for i in members for a in (1, 2, 3)), reverse=True)
        got = 0
        for _, i, a in cand:
            if got >= req:
                break
            if can(i, a):
                add(i, a); got += 1
    for _, i, a in sorted(((ratio[i, a - 1], i, a) for i in range(n) for a in (1, 2, 3) if V[i, a - 1] > 0),
                          reverse=True):
        if can(i, a):
            add(i, a)
    return w


# ============================================================================ SPO+
def spo_plus(chat: np.ndarray, c: np.ndarray, milp: MILP, w_c: np.ndarray) -> tuple[float, np.ndarray]:
    """SPO+ cho bài toán MAX: ℓ = max_w (2ĉ − c)ᵀw − 2ĉᵀw*(c) + z*(c);  ∂ℓ/∂ĉ = 2(w*(2ĉ − c) − w*(c))."""
    w_t = milp.solve(2 * chat - c)
    loss = float(((2 * chat - c) * w_t).sum() - 2 * (chat * w_c).sum() + (c * w_c).sum())
    return loss, 2 * (w_t - w_c)


def spo_set_metrics(theta, Z, C, W, insts, milps, G) -> tuple[float, float]:
    """SPO+ loss trung bình / khách và giá trị DR (Γ không winsorize) của quyết định w*(ĉ) – không dùng đáp án."""
    ls, dr = [], []
    for c, wc, ins, m in zip(C, W, insts, milps):
        chat = cmat(Z[ins["idx"]] @ theta)
        ls.append(spo_plus(chat, c, m, wc)[0] / BATCH)
        dr.append((m.solve(chat) * cmat(G[ins["idx"]])).sum() / BATCH)
    return float(np.mean(ls)), float(np.mean(dr))


def train_spo(theta0, Z, G, Ctr, Wtr, Itr, Mtr, Cva, Wva, Iva, Mva, lr, rng, epochs=EPOCHS,
              patience=None) -> list[dict]:
    """patience=None: chạy đủ `epochs` (lưới gốc). patience=k: dừng sớm khi SPO+ loss valid không cải thiện k epoch."""
    th = theta0.copy()
    m1, m2, t, b1, b2 = np.zeros_like(th), np.zeros_like(th), 0, 0.9, 0.999
    vl, vd = spo_set_metrics(th, Z, Cva, Wva, Iva, Mva, G)
    hist = [dict(lr=lr, epoch=0, train_spo_plus=np.nan, valid_spo_plus=vl, valid_gia_tri_DR=vd, theta=th.copy())]
    best_vl, best_ep = np.inf, 0
    for ep in range(1, epochs + 1):
        order, tl = rng.permutation(len(Ctr)), []
        for b in range(0, len(order), MB):
            g = np.zeros_like(th)
            js = order[b:b + MB]
            for j in js:
                Zi = Z[Itr[j]["idx"]]
                loss, grad = spo_plus(cmat(Zi @ th), Ctr[j], Mtr[j], Wtr[j])
                tl.append(loss / BATCH)
                g += Zi.T @ grad[:, 1:]
            g /= len(js) * BATCH
            g[1:] += L2 * th[1:]
            t += 1
            m1 = b1 * m1 + (1 - b1) * g
            m2 = b2 * m2 + (1 - b2) * g ** 2
            th -= lr * (m1 / (1 - b1 ** t)) / (np.sqrt(m2 / (1 - b2 ** t)) + 1e-8)
        vl, vd = spo_set_metrics(th, Z, Cva, Wva, Iva, Mva, G)
        hist.append(dict(lr=lr, epoch=ep, train_spo_plus=float(np.mean(tl)), valid_spo_plus=vl, valid_gia_tri_DR=vd,
                         theta=th.copy()))
        if vl < best_vl:
            best_vl, best_ep = vl, ep
        elif patience is not None and ep - best_ep >= patience:
            break
    return hist


# ============================================================================ một seed
def run_seed(seed: int, D: dict) -> dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    meta, Z, Gw, G = D["meta"], D["Z"], D["Gw"], D["G"]
    insts = {s: make_instances(meta, s, N_INST[s], rng) for s in ["train", "valid", "calib"]}
    milps = {s: [MILP(i) for i in insts[s]] for s in insts}
    C = {s: [cmat(Gw[i["idx"]]) for i in insts[s]] for s in ["train", "valid"]}
    W = {s: [m.solve(c) for m, c in zip(milps[s], C[s])] for s in C}

    hist = []
    for lr in D["lrs"]:
        hist += train_spo(D["theta_pto"], Z, G, C["train"], W["train"], insts["train"], milps["train"],
                          C["valid"], W["valid"], insts["valid"], milps["valid"], lr, rng,
                          epochs=D["epochs"], patience=D["patience"])
    cand = [h for h in hist if h["epoch"] >= 1]
    best = min(cand, key=lambda h: h["valid_spo_plus"])
    theta_spo = best["theta"]

    preds = {"Luật RFM cũ": D["legacy"], "Nhắm theo rủi ro (μ̂0 thấp)": D["risk"], "MILP trên V̂ (GĐ4)": D["V"],
             "PTO tuyến tính (MSE)": Z @ D["theta_pto"], "PTO LightGBM (MSE)": D["lgbm"], "SPO+ tuyến tính": Z @ theta_spo}
    dec, chk = {k: [] for k in METHODS}, []
    for j, (ins, m) in enumerate(zip(insts["calib"], milps["calib"])):
        for meth in METHODS:
            if meth == "Greedy (V̂/mệnh giá)":
                w = greedy(ins, D["V"][ins["idx"]])
            else:
                w = m.solve(cmat(preds[meth][ins["idx"]]))
            dec[meth].append(w.astype(np.int8))
            chk.append({"seed": seed, "instance": j, "phuong_phap": meth, **check_feasible(ins, w),
                        "gia_tri_DR_uoc_luong": float((w * cmat(G[ins["idx"]])).sum())})
    for h in hist:
        h["seed"] = seed
        h["duoc_chon"] = h is best
    return dict(seed=seed, insts=insts["calib"], dec=dec, chk=chk, theta_spo=theta_spo,
                hist=[{k: v for k, v in h.items() if k != "theta"} for h in hist],
                spo_pred_calib=preds["SPO+ tuyến tính"], runtime=time.time() - t0)


# ============================================================================ kiểm chứng cài đặt
def verify_small_instance(meta: pd.DataFrame, rng, n_pairs: int = 300) -> tuple[pd.DataFrame, dict]:
    """Instance 6 khách: liệt kê TOÀN BỘ 4^6 nghiệm, so z*, SPO, SPO+ và subgradient với solver."""
    sub = meta[(meta.split == "calib")]
    t0 = np.sort(sub.t0.unique())[0]
    pool = sub.index[(sub.t0 == t0) & (sub.cancel_rate_365.fillna(0) <= GR_CANCEL)].to_numpy()
    uk = pool[meta[SEG_COL].to_numpy()[pool] == 1]
    non = pool[meta[SEG_COL].to_numpy()[pool] == 0]
    idx = np.r_[rng.choice(uk, 4, replace=False), rng.choice(non, 2, replace=False)]
    # B = 150 để a2 khả thi (1 a2 cần ≥ 4 voucher: 60 + 3×30); K = 1; phủ ≥ 1 khách mỗi phân khúc
    inst = build_instance(idx, meta, None, "calib", t0, B=150.0, K=1.0, req=(1, 1))
    allw = np.array(list(itertools.product(range(4), repeat=6)))
    Wall = np.eye(4)[allw]                                               # (4096, 6, 4)
    feas = np.array([check_feasible(inst, w)["n_vi_pham"] == 0 for w in Wall])
    Wf = Wall[feas].reshape(feas.sum(), -1)
    milp = MILP(inst)

    def enum_argmax(c):
        v = Wf @ c.ravel()
        k = int(np.argmax(v))
        return Wf[k].reshape(6, 4), float(v[k])

    def spo_plus_enum(chat, c):
        wc, zc = enum_argmax(c)
        _, zt = enum_argmax(2 * chat - c)
        return zt - 2 * (chat * wc).sum() + zc

    rows = []
    for r in range(n_pairs):
        c = cmat(rng.normal(-2, 20, (6, 3)))
        chat = cmat(c[:, 1:] + rng.normal(0, rng.uniform(0, 40), (6, 3)))
        wc_e, zc_e = enum_argmax(c)
        wc_s = milp.solve(c)
        wh_e, _ = enum_argmax(chat)
        wh_s = milp.solve(chat)
        spo_e, spo_s = zc_e - (c * wh_e).sum(), (c * wc_s).sum() - (c * wh_s).sum()
        sp_e = spo_plus_enum(chat, c)
        sp_s, g = spo_plus(chat, c, milp, wc_s)
        d = cmat(rng.normal(size=(6, 3)))
        eps = 1e-5
        fd = (spo_plus_enum(chat + eps * d, c) - spo_plus_enum(chat - eps * d, c)) / (2 * eps)
        eta = 0.01
        down = spo_plus_enum(chat - eta * g, c) < sp_e - 1e-12 if np.abs(g).sum() > 0 else True
        up_wrong = spo_plus_enum(chat + eta * g, c) < sp_e - 1e-12 if np.abs(g).sum() > 0 else False
        rows.append({"cap": r, "z_sao_liet_ke": zc_e, "z_sao_solver": float((c * wc_s).sum()),
                     "spo_liet_ke": spo_e, "spo_solver": spo_s, "spo_plus_liet_ke": sp_e, "spo_plus_solver": sp_s,
                     "dao_ham_huong_sai_phan": fd, "dao_ham_huong_subgradient": float((g * d).sum()),
                     "giam_khi_di_nguoc_subgradient": bool(down), "giam_khi_di_theo_dau_sai": bool(up_wrong),
                     "subgradient_bang_0": bool(np.abs(g).sum() == 0)})
    v = pd.DataFrame(rows)
    nz = ~v.subgradient_bang_0
    summ = {"so_nghiem_kha_thi": int(feas.sum()), "tong_so_nghiem": 4 ** 6,
            "max_lech_z_sao": float((v.z_sao_liet_ke - v.z_sao_solver).abs().max()),
            "max_lech_spo": float((v.spo_liet_ke - v.spo_solver).abs().max()),
            "max_lech_spo_plus": float((v.spo_plus_liet_ke - v.spo_plus_solver).abs().max()),
            "ty_le_spo_plus_ge_spo": float((v.spo_plus_liet_ke >= v.spo_liet_ke - 1e-9).mean()),
            "min_spo": float(v.spo_liet_ke.min()),
            "ty_le_dao_ham_khop_sai_phan": float(((v.dao_ham_huong_sai_phan - v.dao_ham_huong_subgradient).abs()
                                                  < 1e-4).mean()),
            "ty_le_giam_khi_di_nguoc_subgradient (đúng dấu)": float(v.giam_khi_di_nguoc_subgradient[nz].mean()),
            "ty_le_giam_khi_dung_dau_sai": float(v.giam_khi_di_theo_dau_sai[nz].mean()),
            "spo_plus_tai_chat_bang_c": spo_plus_enum(c, c)}
    return v, summ


def crosscheck_pulp(insts: list[dict], C: np.ndarray, n_check: int = 20) -> pd.DataFrame:
    """Viết lại ràng buộc độc lập bằng PuLP và giải bằng CBC; so giá trị tối ưu với HiGHS."""
    rows = []
    for j, ins in enumerate(insts[:n_check]):
        c = cmat(C[ins["idx"]])
        n = ins["n"]
        prob = pulp.LpProblem("kiem_tra", pulp.LpMaximize)
        x = [[prob.add_variable(f"x_{i}_{a}", cat="Binary") for a in range(4)] for i in range(n)]  # API PuLP 4.x
        prob += pulp.lpSum(float(c[i, a]) * x[i][a] for i in range(n) for a in range(4))
        for i in range(n):
            prob += pulp.lpSum(x[i]) == 1
            if ins["blocked"][i]:
                for a in (1, 2, 3):
                    prob += x[i][a] == 0
        prob += pulp.lpSum(CAP[a] * x[i][a] for i in range(n) for a in (1, 2, 3)) <= ins["B"]
        prob += pulp.lpSum(x[i][3] for i in range(n)) <= ins["K"]
        prob += pulp.lpSum(x[i][2] for i in range(n)) <= A2_SHARE * pulp.lpSum(x[i][a] for i in range(n)
                                                                               for a in (1, 2, 3))
        prob += pulp.lpSum(x[i][a] for i in range(n) if ins["is_uk"][i] for a in (1, 2, 3)) >= ins["req_uk"]
        prob += pulp.lpSum(x[i][a] for i in range(n) if not ins["is_uk"][i] for a in (1, 2, 3)) >= ins["req_non"]
        t = time.perf_counter()
        stats = prob.solve(pulp.HiGHS(msg=False, gapRel=1e-7))      # PuLP 4.2 chỉ còn HiGHS (không kèm CBC)
        t_p = time.perf_counter() - t
        t = time.perf_counter()
        w = MILP(ins).solve(c)
        t_h = time.perf_counter() - t
        z_p = pulp.value(prob.objective)
        rows.append({"instance": j, "z_highspy_truc_tiep": float((c * w).sum()), "z_PuLP_HiGHS": z_p,
                     "chenh_lech": float((c * w).sum()) - z_p, "trang_thai_PuLP": stats.status.name,
                     "giay_highspy(gồm dựng mô hình)": t_h, "giay_PuLP(gồm dựng mô hình)": t_p})
    return pd.DataFrame(rows)


# ============================================================================ ĐÁNH GIÁ (được phép đọc đáp án)
def evaluate_load_costs(keys: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """U_real: lợi nhuận gia tăng HIỆN THỰC so với a0; U_exp: kỳ vọng theo U (điều kiện trên Y(0) và giỏ S)."""
    orc = pd.read_parquet(common.p("data/processed/oracle.parquet"))
    t = keys[["customer_id", "t0"]].merge(orc.drop(columns=["split"]), on=["customer_id", "t0"], how="left",
                                          validate="one_to_one")
    assert t.profit_a0.notna().all()
    S, y0 = t.basket_value_S.to_numpy(), t.y_a0.to_numpy()
    U_real, U_exp = np.zeros((len(t), 4)), np.zeros((len(t), 4))
    for k, a in enumerate(ACTIONS[1:], start=1):
        U_real[:, k] = t[f"profit_{a}"] - t["profit_a0"]
        tc = t[f"tau_cond_{a}"].to_numpy()
        dc = np.minimum(DISC[k] * S, CAP[k])
        U_exp[:, k] = np.where(y0 == 0, tc * (MARGIN * S - dc) - CONTACT[k],
                               (1 + tc) * (MARGIN * S - dc) - MARGIN * S - CONTACT[k])
    return U_real, U_exp


def evaluate_regret(results: list[dict], U_real, U_exp) -> pd.DataFrame:
    rows = []
    for res in results:
        for j, ins in enumerate(res["insts"]):
            m = MILP(ins)
            ur, ue = U_real[ins["idx"]], U_exp[ins["idx"]]
            w_or, w_oe = m.solve(ur), m.solve(ue)
            zr, ze = (ur * w_or).sum(), (ue * w_oe).sum()
            decs = {**{k: res["dec"][k][j] for k in METHODS},
                    "Đáp án (biết trước, hiện thực)": w_or, "Đáp án (kỳ vọng theo U)": w_oe}
            for meth, w in decs.items():
                vr, ve = (ur * w).sum(), (ue * w).sum()
                rows.append({"seed": res["seed"], "instance": j, "t0": ins["t0"], "phuong_phap": meth,
                             "z_sao_hien_thuc": zr, "z_hien_thuc": vr, "regret_hien_thuc": zr - vr,
                             "regret_chuan_hoa_hien_thuc": (zr - vr) / abs(zr) if abs(zr) > 1e-9 else np.nan,
                             "z_sao_ky_vong": ze, "z_ky_vong": ve, "regret_ky_vong": ze - ve,
                             "regret_chuan_hoa_ky_vong": (ze - ve) / abs(ze) if abs(ze) > 1e-9 else np.nan,
                             "so_voucher": int(w[:, 1:].sum()), "so_a2": int(w[:, 2].sum()),
                             "so_a3": int(w[:, 3].sum())})
    return pd.DataFrame(rows)


def evaluate_summary(reg: pd.DataFrame, chk: pd.DataFrame) -> pd.DataFrame:
    per_seed = reg.groupby(["phuong_phap", "seed"]).apply(lambda d: pd.Series({
        "regret_cn_hien_thuc": d.regret_chuan_hoa_hien_thuc.mean(),
        "regret_cn_ky_vong": d.regret_chuan_hoa_ky_vong.mean(),
        "regret_tong_hien_thuc": d.regret_hien_thuc.sum() / d.z_sao_hien_thuc.abs().sum(),
        "regret_tong_ky_vong": d.regret_ky_vong.sum() / d.z_sao_ky_vong.abs().sum(),
        "gia_tri_hien_thuc_TB": d.z_hien_thuc.mean(), "gia_tri_ky_vong_TB": d.z_ky_vong.mean(),
        "so_voucher_TB": d.so_voucher.mean()}), include_groups=False).reset_index()
    dr = chk.groupby(["phuong_phap", "seed"]).gia_tri_DR_uoc_luong.mean().rename("gia_tri_DR_TB").reset_index()
    per_seed = per_seed.merge(dr, on=["phuong_phap", "seed"], how="left")
    cols = [c for c in per_seed.columns if c not in ("phuong_phap", "seed")]
    g = per_seed.groupby("phuong_phap")[cols]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd")).reset_index()
    order = METHODS + ["Đáp án (kỳ vọng theo U)", "Đáp án (biết trước, hiện thực)"]
    out["_o"] = out.phuong_phap.map({m: i for i, m in enumerate(order)})
    return out.sort_values("_o").drop(columns="_o")


def evaluate_paired(reg: pd.DataFrame, a="SPO+ tuyến tính", b="PTO tuyến tính (MSE)") -> pd.DataFrame:
    rows = []
    for col in ["regret_chuan_hoa_hien_thuc", "regret_chuan_hoa_ky_vong"]:
        p = reg.pivot_table(index=["seed", "instance"], columns="phuong_phap", values=col)
        d = p[a] - p[b]
        ps = d.groupby(level="seed").mean()
        rows.append({"chi_so": col, "so_sanh": f"{a} − {b}", "chenh_lech_TB": ps.mean(), "sd_qua_seed": ps.std(ddof=1),
                     "ty_le_instance_SPO_tot_hon": float((d < -1e-9).mean()),
                     "ty_le_instance_bang_nhau": float((d.abs() <= 1e-9).mean()),
                     "ty_le_instance_SPO_kem_hon": float((d > 1e-9).mean()),
                     "so_seed_SPO_tot_hon": int((ps < 0).sum())})
    return pd.DataFrame(rows)


def evaluate_mse_vs_regret(insts, V, U_real, U_exp, method_preds: dict, rng) -> pd.DataFrame:
    """Thí nghiệm đinh: bơm nhiễu / co giãn / tịnh tiến V̂ → MSE và regret (trên các instance calib của seed 0)."""
    idx_all = np.concatenate([i["idx"] for i in insts])
    sdV = V[idx_all].std()
    milps = [MILP(i) for i in insts]
    oracle_r = [(U_real[i["idx"]] * m.solve(U_real[i["idx"]])).sum() for i, m in zip(insts, milps)]
    oracle_e = [(U_exp[i["idx"]] * m.solve(U_exp[i["idx"]])).sum() for i, m in zip(insts, milps)]

    def run(pred, kind, level, rep=0):
        rr, re = [], []
        for i, m, zr, ze in zip(insts, milps, oracle_r, oracle_e):
            w = m.solve(cmat(pred[i["idx"]]))
            rr.append((zr - (U_real[i["idx"]] * w).sum()) / abs(zr))
            re.append((ze - (U_exp[i["idx"]] * w).sum()) / abs(ze))
        P = pred[idx_all]
        return {"loai": kind, "muc": level, "lan": rep,
                "MSE_hien_thuc": float(((P - U_real[idx_all, 1:]) ** 2).mean()),
                "MSE_ky_vong": float(((P - U_exp[idx_all, 1:]) ** 2).mean()),
                "regret_cn_hien_thuc": float(np.mean(rr)), "regret_cn_ky_vong": float(np.mean(re))}

    rows = []
    for s in [0, 0.25, 0.5, 1, 2, 4, 8]:
        for rep in range(3 if s > 0 else 1):
            rows.append(run(V + rng.normal(0, s * sdV, V.shape), "Nhiễu cộng (σ × sd(V̂))", s, rep))
    for k in [0.25, 0.5, 2, 4]:
        rows.append(run(V * k, "Co giãn (V̂ × k)", k))
    for dlt in [-20, -10, 10, 20]:
        rows.append(run(V + dlt, "Tịnh tiến (V̂ + δ GBP)", dlt))
    for name, pred in method_preds.items():
        rows.append(run(pred, "Phương pháp", name))
    rows.append(run(U_exp[:, 1:], "Phương pháp", "Kỳ vọng thật (đáp án)"))
    return pd.DataFrame(rows)


def evaluate_plots(summ: pd.DataFrame, mvr: pd.DataFrame) -> list[str]:
    out = []
    s = summ[summ.phuong_phap.isin(METHODS)].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(s))
    ax.bar(x - 0.2, s.regret_cn_hien_thuc_mean, 0.4, yerr=s.regret_cn_hien_thuc_sd, capsize=3,
           label="(A) theo lợi nhuận hiện thực (đặc tả)")
    ax.bar(x + 0.2, s.regret_cn_ky_vong_mean, 0.4, yerr=s.regret_cn_ky_vong_sd, capsize=3,
           label="(B) theo lợi nhuận kỳ vọng theo U")
    ax.set_xticks(x); ax.set_xticklabels(s.phuong_phap, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Regret chuẩn hóa TB (mean ± sd qua 5 seed)")
    ax.set_title("Regret chuẩn hóa trên calib – 120 instance × 5 seed, cùng miền khả thi & cùng solver (HiGHS)")
    ax.axhline(0, color="k", lw=0.8); ax.legend(fontsize=8)
    fig.tight_layout()
    p = common.p(FIGS / f"fig_s5_regret_by_method{SFX}.png"); fig.savefig(p, dpi=160); plt.close(fig)
    out.append(str(FIGS / f"fig_s5_regret_by_method{SFX}.png"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, sfx, lab in [(axes[0], "hien_thuc", "hiện thực"), (axes[1], "ky_vong", "kỳ vọng theo U")]:
        for kind, mk, col in [("Nhiễu cộng (σ × sd(V̂))", "o-", "tab:blue"), ("Co giãn (V̂ × k)", "s", "tab:green"),
                              ("Tịnh tiến (V̂ + δ GBP)", "^", "tab:orange")]:
            d = mvr[mvr.loai == kind].groupby("muc")[[f"MSE_{sfx}", f"regret_cn_{sfx}"]].mean().reset_index()
            ax.plot(d[f"MSE_{sfx}"], d[f"regret_cn_{sfx}"], mk, color=col, label=kind, ms=6)
            for r in d.itertuples():
                ax.annotate(str(r.muc), (getattr(r, f"MSE_{sfx}"), getattr(r, f"regret_cn_{sfx}")), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
        d = mvr[mvr.loai == "Phương pháp"]
        ax.plot(d[f"MSE_{sfx}"], d[f"regret_cn_{sfx}"], "*", color="tab:red", ms=12, label="Phương pháp (seed 0)")
        for r in d.itertuples():
            ax.annotate(str(r.muc), (getattr(r, f"MSE_{sfx}"), getattr(r, f"regret_cn_{sfx}")), fontsize=7,
                        xytext=(4, -10), textcoords="offset points", color="tab:red")
        ax.set_xscale("log")
        ax.set_xlabel(f"MSE của vector chi phí dự báo (so với lợi nhuận gia tăng {lab}, thang log)")
        ax.set_ylabel(f"Regret chuẩn hóa TB ({lab})")
        ax.set_title(f"MSE vs Regret – {lab}")
        ax.legend(fontsize=7)
    fig.suptitle("Thí nghiệm đinh: MSE thấp hơn KHÔNG đồng nghĩa quyết định tốt hơn (calib, seed 0)")
    fig.tight_layout()
    p = common.p(FIGS / f"fig_s5_mse_vs_regret{SFX}.png"); fig.savefig(p, dpi=160); plt.close(fig)
    out.append(str(FIGS / f"fig_s5_mse_vs_regret{SFX}.png"))
    return out


# ============================================================================ quy tắc mở rộng biên lưới
def _scale_ext(v: float, grid: list) -> tuple[list, str]:
    if v == max(grid):
        return [v * 3, v * 10], "biên trên"
    if v == min(grid):
        return [v / 3, v / 10], "biên dưới"
    return [], "trong lưới"


def _int_ext(v: int, vals: list, lo: int) -> tuple[list, str]:
    if v == max(vals):
        return [v * 3, v * 10], "biên trên"
    if v == min(vals):
        return sorted({max(lo, round(v / 3)), max(lo, round(v / 10))} - {v}), "biên dưới"
    return [], "trong lưới"


def grid_extension() -> tuple[dict, pd.DataFrame]:
    """Áp quy tắc đăng ký trước (CLAUDE.md, GĐ5) lên kết quả CHỌN TRÊN VALID của lưới gốc – không dùng calib/đáp án.
    Biên = giá trị chọn nằm ở đầu/cuối lưới gốc ở ≥ 1 seed; mở rộng MỘT lần cho mọi seed."""
    sel = pd.read_csv(TABLES / "s5_pto_selection.csv")
    hist = pd.read_csv(TABLES / "s5_spo_training.csv")
    rows = []
    g = dict(alphas=list(RIDGE_ALPHAS), lgbm=list(LGBM_GRID), lrs=list(LRS), epochs=EPOCHS, patience=None)

    a_sel = float(sel[sel.duoc_chon & sel.mo_hinh.str.startswith("PTO tuyến tính")].sieu_tham_so.iloc[0].split("=")[1])
    ext, pos = _scale_ext(a_sel, RIDGE_ALPHAS)
    g["alphas"] += ext
    rows.append({"phuong_phap": "PTO tuyến tính", "tham_so": "alpha", "luoi_goc": str(RIDGE_ALPHAS),
                 "gia_tri_chon_valid": a_sel, "vi_tri": pos, "them_vao": str(ext)})

    cfg = sel[sel.duoc_chon & sel.mo_hinh.eq("PTO LightGBM")].sieu_tham_so.iloc[0]
    nl_sel, ne_sel = [int(p.split("=")[1]) for p in cfg.split(", ")]
    nl_ext, nl_pos = _int_ext(nl_sel, sorted({c[0] for c in LGBM_GRID}), lo=2)
    ne_ext, ne_pos = _int_ext(ne_sel, sorted({c[1] for c in LGBM_GRID}), lo=1)
    new = [(e, ne_sel) for e in nl_ext] + [(nl_sel, e) for e in ne_ext]
    g["lgbm"] += [c for c in dict.fromkeys(new) if c not in LGBM_GRID]
    rows += [{"phuong_phap": "PTO LightGBM", "tham_so": "num_leaves", "luoi_goc": str(sorted({c[0] for c in LGBM_GRID})),
              "gia_tri_chon_valid": nl_sel, "vi_tri": nl_pos, "them_vao": str([(e, ne_sel) for e in nl_ext])},
             {"phuong_phap": "PTO LightGBM", "tham_so": "n_estimators", "luoi_goc": str(sorted({c[1] for c in LGBM_GRID})),
              "gia_tri_chon_valid": ne_sel, "vi_tri": ne_pos, "them_vao": str([(nl_sel, e) for e in ne_ext])}]

    ch = hist[hist.duoc_chon]
    lr_pos = []
    if (ch.lr == max(LRS)).any():
        g["lrs"] += [max(LRS) * 3, max(LRS) * 10]; lr_pos.append("biên trên")
    if (ch.lr == min(LRS)).any():
        g["lrs"] = [min(LRS) / 10, min(LRS) / 3] + g["lrs"]; lr_pos.append("biên dưới")
    rows.append({"phuong_phap": "SPO+", "tham_so": "lr", "luoi_goc": str(LRS),
                 "gia_tri_chon_valid": str(dict(zip(ch.seed, ch.lr))), "vi_tri": ", ".join(lr_pos) or "trong lưới",
                 "them_vao": str([v for v in g["lrs"] if v not in LRS])})
    ep_hit = bool((ch.epoch == EPOCHS).any())
    if ep_hit:
        g["epochs"], g["patience"] = EPOCHS * 3, PATIENCE
    rows.append({"phuong_phap": "SPO+", "tham_so": "epoch tối đa", "luoi_goc": f"1..{EPOCHS}",
                 "gia_tri_chon_valid": str(dict(zip(ch.seed, ch.epoch))),
                 "vi_tri": "biên trên" if ep_hit else "trong lưới",
                 "them_vao": f"tối đa {g['epochs']} epoch, early stopping patience {g['patience']}" if ep_hit else "[]"})
    return g, pd.DataFrame(rows)


def evaluate_compare_grids() -> list[str]:
    """Ghép kết quả lưới gốc và lưới mở rộng (bảng regret, so sánh cặp, đường SPO+ loss valid theo epoch)."""
    out = []
    s0 = pd.read_csv(TABLES / "s5_regret_summary.csv").assign(luoi="gốc")
    s1 = pd.read_csv(TABLES / "s5_regret_summary_mo_rong.csv").assign(luoi="mở rộng")
    r0 = pd.read_csv(TABLES / "s5_regret_instances.csv")
    r1 = pd.read_csv(TABLES / "s5_regret_instances_mo_rong.csv")
    k = ["seed", "instance", "phuong_phap"]
    m = r0.merge(r1, on=k, suffixes=("_goc", "_mr"))
    same_inst = bool(np.allclose(m.z_sao_hien_thuc_goc, m.z_sao_hien_thuc_mr)
                     and np.allclose(m.z_sao_ky_vong_goc, m.z_sao_ky_vong_mr))
    fixed = ["Luật RFM cũ", "Nhắm theo rủi ro (μ̂0 thấp)", "Greedy (V̂/mệnh giá)", "MILP trên V̂ (GĐ4)"]
    mf = m[m.phuong_phap.isin(fixed)]
    same_fixed = bool(np.allclose(mf.z_hien_thuc_goc, mf.z_hien_thuc_mr))
    both = pd.concat([s0, s1], ignore_index=True)
    both["cung_instance_hai_luoi"] = same_inst
    both["pp_khong_hoc_trung_khop"] = same_fixed
    both.to_csv(common.p(TABLES / "s5_regret_summary_hai_luoi.csv"), index=False)
    p0 = pd.read_csv(TABLES / "s5_spo_vs_pto_paired.csv").assign(luoi="gốc")
    p1 = pd.read_csv(TABLES / "s5_spo_vs_pto_paired_mo_rong.csv").assign(luoi="mở rộng")
    pd.concat([p0, p1]).to_csv(common.p(TABLES / "s5_spo_vs_pto_paired_hai_luoi.csv"), index=False)
    out += [str(TABLES / "s5_regret_summary_hai_luoi.csv"), str(TABLES / "s5_spo_vs_pto_paired_hai_luoi.csv")]
    print(f"Cùng instance ở hai lưới: {same_inst} · phương pháp không học trùng khớp: {same_fixed}")

    h0 = pd.read_csv(TABLES / "s5_spo_training.csv").assign(luoi="gốc")
    h1 = pd.read_csv(TABLES / "s5_spo_training_mo_rong.csv").assign(luoi="mở rộng")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, h, ttl in [(axes[0], h0, "Lưới gốc (15 epoch, không dừng sớm)"),
                       (axes[1], h1, "Lưới mở rộng (≤ 45 epoch, dừng sớm patience 3)")]:
        cmap = plt.get_cmap("viridis", h.lr.nunique() + 1)
        for c, (lr, d) in enumerate(h.groupby("lr")):
            for sd, ds in d.groupby("seed"):
                ax.plot(ds.epoch, ds.valid_spo_plus, "-", color=cmap(c), lw=0.8, alpha=0.6,
                        label=f"lr={lr:g}" if sd == d.seed.min() else None)
            sel = d[d.duoc_chon]
            ax.plot(sel.epoch, sel.valid_spo_plus, "*", color="tab:red", ms=11,
                    label="được chọn (theo seed)" if c == 0 else None)
        ax.set_title(ttl)
        ax.set_xlabel("epoch (0 = khởi tạo từ PTO tuyến tính)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("SPO+ loss TB / khách trên valid (Γ)")
    fig.suptitle("Đường SPO+ loss trên valid theo epoch – 5 seed (mỗi đường = 1 seed × 1 lr)")
    fig.tight_layout()
    fig.savefig(common.p(FIGS / "fig_s5_valid_loss_hai_luoi.png"), dpi=160); plt.close(fig)
    out.append(str(FIGS / "fig_s5_valid_loss_hai_luoi.png"))

    meth = METHODS
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(meth))
    for off, (lab, s, col) in zip([-0.2, 0.2], [("gốc", s0, "tab:blue"), ("mở rộng", s1, "tab:orange")]):
        s = s.set_index("phuong_phap").loc[meth]
        ax.bar(x + off, s.regret_cn_ky_vong_mean, 0.4, yerr=s.regret_cn_ky_vong_sd, capsize=3, color=col,
               label=f"Lưới {lab}")
    ax.set_xticks(x); ax.set_xticklabels(meth, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Regret chuẩn hóa TB (B: kỳ vọng theo U), mean ± sd 5 seed")
    ax.set_title("Regret trên calib: lưới gốc vs lưới mở rộng (cùng instance, cùng solver)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(common.p(FIGS / "fig_s5_regret_hai_luoi.png"), dpi=160); plt.close(fig)
    out.append(str(FIGS / "fig_s5_regret_hai_luoi.png"))
    return out


# ============================================================================ chính
def main() -> None:
    np.random.seed(SEED)
    t_start = time.time()
    files: list[str] = []
    pd.set_option("display.width", 250)

    lg = pd.read_parquet(common.p("data/processed/logged.parquet"))
    lg = lg[lg.split.isin(["train", "valid", "calib"])].reset_index(drop=True)          # không đụng locked/oot
    p4 = pd.read_parquet(common.p("data/processed/pred_s4.parquet"))
    keep = ["customer_id", "t0"] + [f"{p}_{a}" for p in ("V_hat", "tau_hat") for a in TREATED] + ["mu0_base",
                                                                                                     "s_hat_mean"]
    df = lg.merge(p4[keep], on=["customer_id", "t0"], how="left", validate="one_to_one")
    assert df.V_hat_a1.notna().all()
    tr = df.split.eq("train").to_numpy()
    va = df.split.eq("valid").to_numpy()
    meta = df[["split", "t0", SEG_COL, "cancel_rate_365"]].copy()

    # ---- 1. pseudo-outcome AIPW Γ của lợi nhuận gia tăng (propensity ghi log, μ̂_a ngoài fold)
    pre = joblib.load(common.p("models/s4_propensity.joblib"))["pre"]
    X = pre.transform(df[FEATS_C])
    A = df.A_logged.map({a: k for k, a in enumerate(ACTIONS)}).to_numpy()
    P = df[[f"propensity_{a}" for a in ACTIONS]].to_numpy()
    y = df.profit_obs.to_numpy(float)
    Mu = np.zeros((len(df), 4))
    itr = np.where(tr)[0]
    g_tr = df.customer_id.to_numpy()[itr]
    for f_tr, f_te in gkf(g_tr):
        a_tr, a_te = itr[f_tr], itr[f_te]
        for k in range(4):
            fit = a_tr[A[a_tr] == k]
            Mu[a_te, k] = LGBMRegressor(**NUIS).fit(X[fit], y[fit]).predict(X[a_te])
    for k in range(4):
        Mu[~tr, k] = LGBMRegressor(**NUIS).fit(X[itr[A[itr] == k]], y[itr[A[itr] == k]]).predict(X[~tr])
    psi = np.column_stack([Mu[:, k] + (A == k) * (y - Mu[:, k]) / P[:, k] for k in range(4)])
    G = psi[:, 1:] - psi[:, [0]]
    lo, hi = np.percentile(G[tr], 0.5, axis=0), np.percentile(G[tr], 99.5, axis=0)
    Gw = np.clip(G, lo, hi)
    gsum = pd.DataFrame([{"hanh_dong": a, "split": s, "TB_Gamma": G[m, j].mean(), "sd_Gamma": G[m, j].std(),
                          "TB_Gamma_winsor": Gw[m, j].mean(), "sd_Gamma_winsor": Gw[m, j].std(),
                          "nguong_duoi": lo[j], "nguong_tren": hi[j], "ty_le_bi_cat": float(((G[m, j] < lo[j]) |
                                                                                           (G[m, j] > hi[j])).mean())}
                         for j, a in enumerate(TREATED) for s, m in [("train", tr), ("valid", va)]])
    gsum.to_csv(common.p(TABLES / f"s5_gamma_summary{SFX}.csv"), index=False)
    files.append(str(TABLES / f"s5_gamma_summary{SFX}.csv"))
    print(gsum.round(3).to_string())
    print(f"Γ xong ({time.time() - t_start:.0f}s)")

    # ---- 2. lưới siêu tham số (gốc, hoặc mở rộng theo quy tắc biên đăng ký trước) + đặc trưng z + mô hình PTO
    if LUOI == "mo_rong":
        grids, rule = grid_extension()
        rule.to_csv(common.p(TABLES / "s5_grid_extension_rule.csv"), index=False)
        files.append(str(TABLES / "s5_grid_extension_rule.csv"))
        print("QUY TẮC MỞ RỘNG BIÊN:\n", rule.to_string())
    else:
        grids = dict(alphas=list(RIDGE_ALPHAS), lgbm=list(LGBM_GRID), lrs=list(LRS), epochs=EPOCHS, patience=None)
    extra = np.column_stack([df[[f"V_hat_{a}" for a in TREATED]].to_numpy(), df[[f"tau_hat_{a}" for a in TREATED]],
                             df.mu0_base, np.log(df.s_hat_mean)])
    Zraw = np.column_stack([X, extra])
    scaler = StandardScaler().fit(Zraw[tr])
    Z = np.column_stack([np.ones(len(df)), scaler.transform(Zraw)])
    pen = np.eye(Z.shape[1]); pen[0, 0] = 0
    ZtZ, ZtG = Z[tr].T @ Z[tr], Z[tr].T @ Gw[tr]
    sel = []
    best = (np.inf, None, None)
    for al in grids["alphas"]:
        th = np.linalg.solve(ZtZ + al * pen, ZtG)
        mse = float(((Z[va] @ th - Gw[va]) ** 2).mean())
        sel.append({"mo_hinh": "PTO tuyến tính (ridge)", "sieu_tham_so": f"alpha={al:g}", "MSE_valid": mse,
                    "MSE_train": float(((Z[tr] @ th - Gw[tr]) ** 2).mean())})
        if mse < best[0]:
            best = (mse, al, th)
    theta_pto, alpha_pto = best[2], best[1]
    bestg = (np.inf, None, None)
    for nl, ne in grids["lgbm"]:
        kw = {**NUIS, "num_leaves": nl, "n_estimators": ne, "min_child_samples": 200}
        ms = [LGBMRegressor(**kw).fit(Z[tr, 1:], Gw[tr, j]) for j in range(3)]
        pr = np.column_stack([m.predict(Z[:, 1:]) for m in ms])
        mse = float(((pr[va] - Gw[va]) ** 2).mean())
        sel.append({"mo_hinh": "PTO LightGBM", "sieu_tham_so": f"num_leaves={nl}, n_estimators={ne}",
                    "MSE_valid": mse, "MSE_train": float(((pr[tr] - Gw[tr]) ** 2).mean())})
        if mse < bestg[0]:
            bestg = (mse, (nl, ne), (ms, pr))
    lgbm_models, lgbm_pred = bestg[2]
    Vh = df[[f"V_hat_{a}" for a in TREATED]].to_numpy()
    sel.append({"mo_hinh": "Tham chiếu: V̂ GĐ4 (không huấn luyện lại)", "sieu_tham_so": "-",
                "MSE_valid": float(((Vh[va] - Gw[va]) ** 2).mean()), "MSE_train": float(((Vh[tr] - Gw[tr]) ** 2).mean())})
    sel = pd.DataFrame(sel)
    sel["duoc_chon"] = sel.sieu_tham_so.isin([f"alpha={alpha_pto:g}",
                                              f"num_leaves={bestg[1][0]}, n_estimators={bestg[1][1]}"])
    sel.to_csv(common.p(TABLES / f"s5_pto_selection{SFX}.csv"), index=False)
    files.append(str(TABLES / f"s5_pto_selection{SFX}.csv"))
    print(sel.round(3).to_string())

    # điểm luật cũ (logit so với a0, hạng phần trăm trong (split, t0)) và điểm rủi ro
    L = CFG["intervention_dgp"]["legacy_logits"]
    r = df.groupby(["split", "t0"]).recency_days.rank(pct=True).to_numpy()
    mm = df.groupby(["split", "t0"]).aov_365.rank(pct=True).to_numpy()
    legacy = np.column_stack([L["a1_r"] * r - L["a0"], L["a2_mr"] * mm * r - L["a0"],
                              L["a3_m"] * mm + L["a3_bias"] - L["a0"]])
    risk = np.column_stack([1 - df.mu0_base.to_numpy(), -np.ones(len(df)), -np.ones(len(df))])

    # ---- 3. kiểm chứng cài đặt: instance 6 khách + đối chiếu solver
    s6 = None                     # không phụ thuộc lưới siêu tham số → chỉ chạy ở lưới gốc
    if LUOI == "goc":
        v6, s6 = verify_small_instance(meta, np.random.default_rng(SEED))
        v6.to_csv(common.p(TABLES / f"s5_verification_6{SFX}.csv"), index=False)
        pd.DataFrame([s6]).T.rename(columns={0: "gia_tri"}).to_csv(
            common.p(TABLES / f"s5_verification_6_summary{SFX}.csv"))
        files += [str(TABLES / f"s5_verification_6{SFX}.csv"), str(TABLES / f"s5_verification_6_summary{SFX}.csv")]
        print("KIỂM CHỨNG 6 KHÁCH:", {k: (round(v, 8) if isinstance(v, float) else v) for k, v in s6.items()})

    # ---- 4. nhiều seed song song
    D = dict(meta=meta, Z=Z, Gw=Gw, G=G, theta_pto=theta_pto, lgbm=lgbm_pred, V=Vh, legacy=legacy, risk=risk,
             lrs=grids["lrs"], epochs=grids["epochs"], patience=grids["patience"])
    results = Parallel(n_jobs=len(SEEDS), backend="loky")(delayed(run_seed)(s, D) for s in SEEDS)
    print("seed xong:", {r["seed"]: round(r["runtime"]) for r in results}, f"({time.time() - t_start:.0f}s)")
    hist = pd.DataFrame([h for r in results for h in r["hist"]])
    hist.to_csv(common.p(TABLES / f"s5_spo_training{SFX}.csv"), index=False)
    files.append(str(TABLES / f"s5_spo_training{SFX}.csv"))
    print(hist[hist.duoc_chon].round(4).to_string())
    chk = pd.DataFrame([c for r in results for c in r["chk"]])
    cc = chk.groupby("phuong_phap").agg(n_quyet_dinh=("n_vi_pham", "size"), tong_vi_pham=("n_vi_pham", "sum"),
                                        ngan_sach_dung_TB=("ngan_sach_dung", "mean"), ngan_sach_B_TB=("ngan_sach_B", "mean"),
                                        so_voucher_TB=("so_voucher", "mean"), so_cuoc_goi_TB=("so_cuoc_goi", "mean"),
                                        so_a2_TB=("so_a2", "mean")).reset_index()
    cc.to_csv(common.p(TABLES / f"s5_constraint_check{SFX}.csv"), index=False)
    files.append(str(TABLES / f"s5_constraint_check{SFX}.csv"))
    print(cc.round(2).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    h0 = hist[hist.seed == SEEDS[0]]
    for lr in grids["lrs"]:
        d = h0[h0.lr == lr]
        axes[0].plot(d.epoch, d.valid_spo_plus, "o-", ms=3, label=f"lr={lr}")
        axes[1].plot(d.epoch, d.valid_gia_tri_DR, "o-", ms=3, label=f"lr={lr}")
    axes[0].set_ylabel("SPO+ loss TB / khách (valid, Γ)"); axes[1].set_ylabel("Giá trị DR TB / khách (valid, GBP)")
    for ax in axes:
        ax.set_xlabel("epoch (0 = khởi tạo từ PTO tuyến tính)"); ax.legend(fontsize=8)
    fig.suptitle(f"Huấn luyện SPO+ (seed {SEEDS[0]}) – tiêu chí chọn: SPO+ loss trên valid")
    fig.tight_layout()
    fig.savefig(common.p(FIGS / f"fig_s5_spo_training{SFX}.png"), dpi=160); plt.close(fig)
    files.append(str(FIGS / f"fig_s5_spo_training{SFX}.png"))

    if LUOI == "goc":
        cx = crosscheck_pulp(results[0]["insts"], Vh)
        cx.to_csv(common.p(TABLES / f"s5_solver_crosscheck{SFX}.csv"), index=False)
        files.append(str(TABLES / f"s5_solver_crosscheck{SFX}.csv"))
        print("ĐỐI CHIẾU: max |highspy − PuLP| =", cx.chenh_lech.abs().max(), "| trạng thái PuLP:",
              cx.trang_thai_PuLP.unique())

    # ---- 5. đánh giá bằng đáp án (chỉ trong evaluate_*)
    U_real, U_exp = evaluate_load_costs(df)
    reg = evaluate_regret(results, U_real, U_exp)
    reg.to_csv(common.p(TABLES / f"s5_regret_instances{SFX}.csv"), index=False)
    summ = evaluate_summary(reg, chk)
    summ.to_csv(common.p(TABLES / f"s5_regret_summary{SFX}.csv"), index=False)
    pair = evaluate_paired(reg)
    pair.to_csv(common.p(TABLES / f"s5_spo_vs_pto_paired{SFX}.csv"), index=False)
    files += [str(TABLES / f"{b}{SFX}.csv") for b in ["s5_regret_instances", "s5_regret_summary", "s5_spo_vs_pto_paired"]]
    print(summ[["phuong_phap", "regret_cn_hien_thuc_mean", "regret_cn_hien_thuc_sd", "regret_cn_ky_vong_mean",
                "regret_cn_ky_vong_sd", "regret_tong_ky_vong_mean", "gia_tri_ky_vong_TB_mean",
                "gia_tri_hien_thuc_TB_mean", "gia_tri_DR_TB_mean", "so_voucher_TB_mean"]].round(3).to_string())
    print(pair.round(4).to_string())

    r0 = results[0]
    mvr = evaluate_mse_vs_regret(r0["insts"], Vh, U_real, U_exp,
                                 {"V̂ GĐ4": Vh, "PTO tuyến tính": Z @ theta_pto, "PTO LightGBM": lgbm_pred,
                                  "SPO+ tuyến tính": r0["spo_pred_calib"]}, np.random.default_rng(SEED))
    mvr.to_csv(common.p(TABLES / f"s5_mse_vs_regret{SFX}.csv"), index=False)
    files.append(str(TABLES / f"s5_mse_vs_regret{SFX}.csv"))
    print(mvr.groupby(["loai", "muc"], sort=False)[["MSE_hien_thuc", "MSE_ky_vong", "regret_cn_hien_thuc",
                                                   "regret_cn_ky_vong"]].mean().round(3).to_string())
    files += evaluate_plots(summ, mvr)

    joblib.dump({"scaler": scaler, "theta": theta_pto, "alpha": alpha_pto, "z_cols": "X(s4 pre) + V̂, τ̂, μ̂0 nền, log Ŝ"},
                common.p(f"models/s5_pto_linear{SFX}.joblib"))
    joblib.dump({"models": lgbm_models, "config": bestg[1]}, common.p(f"models/s5_pto_lgbm{SFX}.joblib"))
    joblib.dump({"scaler": scaler, "theta_by_seed": {r["seed"]: r["theta_spo"] for r in results}},
                common.p(f"models/s5_spo_linear{SFX}.joblib"))
    files += [f"models/s5_pto_linear{SFX}.joblib", f"models/s5_pto_lgbm{SFX}.joblib", f"models/s5_spo_linear{SFX}.joblib"]
    if LUOI == "mo_rong":
        files += evaluate_compare_grids()
    common.write_manifest("s5" + SFX, files, extra={"seeds": SEEDS, "n_inst": N_INST, "batch": BATCH, "lrs": grids["lrs"],
                                              "epochs": grids["epochs"], "patience": grids["patience"], "luoi": LUOI, "alpha_pto": alpha_pto, "lgbm_pto": bestg[1],
                                              "verification_6": s6, "runtime_s": round(time.time() - t_start)},
                          packages=["pandas", "numpy", "scikit-learn", "lightgbm", "highspy", "PuLP", "matplotlib"])
    print(f"XONG GĐ5 ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    from s5_spo import main as _main
    _main()
