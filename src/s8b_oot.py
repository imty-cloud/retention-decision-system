"""GĐ8 – BƯỚC 1b: phát lại OOT (vận hành), giám sát MLOps, kịch bản trôi, khuyết nhãn có chọn lọc, thiết kế thí điểm.

Chế độ chạy
  python src/s8b_oot.py --oot-rehearsal   Diễn tập: valid = kỳ tham chiếu, calib ĐÓNG VAI OOT → bảng s8_oot_rehearsal_*.csv, nhãn THỬ.
                                          Không đọc locked/oot (mọi đọc qua s8_mlops.read_rows).
  python src/s8b_oot.py --pilot           Thiết kế thí điểm (cỡ mẫu cho G3 và cho DR) từ ước lượng GĐ4/GĐ5b/GĐ7 trên valid/calib.
  python src/s8b_oot.py --make-drift      Sinh file đáp án kịch bản trôi (oot_persuasion_scale = 0.6) bằng s1b.generate; chỉ ghi file MỚI,
                                          không ghi đè; không đọc / đánh giá file vừa sinh. Chạy TRƯỚC --freeze.
  python src/s8b_oot.py --oot             Phát lại OOT THẬT (BƯỚC 2): chỉ chạy khi file khóa s8_locked_opened.lock ĐÃ tồn tại VÀ verify_freeze
                                          khớp tuyệt đối; nếu không thì từ chối trước khi đọc dữ liệu.

ĐỊNH NGHĨA (viết TRƯỚC khi có dữ liệu OOT; nằm trong phạm vi đóng băng):
 • Current = μ0 + Platt GĐ2, Ŝ, CausalForestDML, MILP + Router v2.1, chạy TUẦN TỰ từng t0 (frequency cap theo bản ghi của chính router).
 • Ngày phát lại = từng t0 của OOT, cộng thêm "ngày chốt" = t0 cuối + 30 ngày. Nhãn của kỳ t0 TRƯỞNG THÀNH tại ngày d nếu t0 + 30 ≤ d.
   Brier/ECE (và mọi chỉ số cần nhãn) CHỈ tính cho kỳ đã trưởng thành, tại ngày phát lại đầu tiên mà kỳ đó trưởng thành.
 • Challenger (6b) = Current + Platt khớp lại tại ngày quyết định d trên dòng a0 (logged) của 3 kỳ GẦN NHẤT đã trưởng thành tại d
   (3 kỳ = cùng cỡ cửa sổ calib của Current). Kho nhãn: thật = calib + locked + oot; diễn tập = valid + calib.
 • Challenger-2 (PHỤ) = huấn luyện lại μ0 (cùng pipeline/siêu tham số) trên dòng a0 của train+valid+calib+locked, Platt theo quy tắc Challenger.
   Ở chế độ thật CHỈ chạy khi đã có bảng gate locked (outputs/tables/s8_locked_gates.csv). Diễn tập: train+valid (calib là OOT).
 • PSI: 10 bin theo phân vị của TRAIN (NaN là một bin riêng; biến ít giá trị/hạng mục dùng từng giá trị), ε = 1e-4.
   Ngưỡng `monitoring`: < 0.10 ổn định · 0.10–0.25 cảnh báo · ≥ 0.25 báo động. Điểm số: μ̂0 hiệu chỉnh, Ŝ, τ̂_a, V̂_a.
 • Action TVD = ½ Σ_a |p_kỳ(a) − p_tham_chiếu(a)| trên hành động CUỐI của Current; tham chiếu thật = valid+calib (decisions_s7_v21),
   diễn tập = valid.
 • Giá trị thực hiện (thước đo GIÁM SÁT, không chấm gate – 6b): AIPW trên dữ liệu do Router sinh ra. Với khách thuộc tập ngẫu nhiên
   (π = hành động trước khi chọn đối chứng, P(π) = 0.9, P(a0) = 0.1): Γ = μ̂_π − μ̂_0 + 1{A=π}(Y−μ̂_π)/0.9 − 1{A=a0}(Y−μ̂_0)/0.1;
   khách bù độ phủ (không ngẫu nhiên): chỉ phần mô hình μ̂_1 − μ̂_0. Y = lợi nhuận hiện thực dưới hành động thực tế; μ̂_k = LightGBM (NUIS
   của GĐ4) học trên profit_obs của train theo nhánh ghi log. CI 95 % bootstrap cụm khách (multinomial). Đặt cạnh giá trị theo đáp án.
 • Phản hồi của thế giới (kết quả dưới hành động Router đã làm) lấy từ đáp án trong hàm đánh giá riêng – đó là mô phỏng "điều quan sát được
   sau khi triển khai", không dùng để chọn hay chỉnh gì của Current.
 • Kịch bản trôi: τ thật ở OOT nhân 0.6 (s1b, cùng số ngẫu nhiên). So sánh CÓ guardrail (MILP qua Router v2.1) với KHÔNG guardrail
   (MILP thuần trên miền v1) và Đề xuất MILP miền v2.1; giá trị theo đáp án trôi và không trôi; vi phạm quy định bắt buộc; số lượt bị chặn;
   tổn thất tránh được = giá trị(CÓ) − giá trị(KHÔNG). Diễn tập: cặp đáp án (không trôi / trôi) sinh TRONG BỘ NHỚ bằng s1b.generate trên
   snapshot train+valid+calib, calib gắn nhãn "oot" (khác số ngẫu nhiên với đáp án chính thức – chỉ để diễn tập code).
 • Khuyết nhãn có chọn lọc: (i) ước lượng tác động chương trình trên nhóm được chọn: so thô (được gửi vs không được chọn) / đối chứng
   ngẫu nhiên / DR; (ii) Challenger hiệu chỉnh μ0 học từ dữ liệu Router: KHÔNG đối chứng (chỉ dòng a0 không được chọn) vs CÓ đối chứng + IPW
   (đối chứng trọng số 1/0.1). Độ chệch đo so với đáp án trên nhóm được chọn.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from scipy.stats import norm
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

import common
import s1b_intervention as S1B
import s2_predict as S2
import s3_explain as S3
import s4_causal as S4
import s5_spo as S5
import s6_milp as S6
import s7_router as S7
import s8_mlops as S8
from s1_data import FEATURES_CAT, FEATURES_NUM
from s5b_dr_diagnostics import legacy_scores  # noqa: F401  (giữ API giống s8)

CFG = common.load_config()
SEED = CFG["project"]["seed"]
MON = CFG["monitoring"]
H = CFG["decision"]["horizon_days"]
ROOT = common.ROOT
TABLES, FIGS = S8.TABLES, S8.FIGS
ACTIONS, TREATED = S5.ACTIONS, S5.TREATED
M0 = S8.M0
FEATS = FEATURES_NUM + FEATURES_CAT
CAL_FEATS = ["t0_month", "days_to_xmas"]
N_BOOT = 1000
RECAL_WINDOW = 3
P_CTRL = CFG["business"]["control_holdout_pct"] / 100
DRIFT_SCALE = 0.6
Z80 = norm.ppf(0.975) + norm.ppf(0.80)

ROLES = {
    "rehearsal": dict(tag="oot_rehearsal", oot="calib", ref=["valid"], pool=["valid", "calib"], ch2_train=["train", "valid"],
                      nhan="THỬ (valid = tham chiếu, calib đóng vai OOT)"),
    "real": dict(tag="oot", oot="oot", ref=["valid", "calib"], pool=["calib", "locked", "oot"],
                 ch2_train=["train", "valid", "calib", "locked"], nhan="VẬN HÀNH (OOT thật)"),
}


# ============================================================================ phiên OOT thật (có rào)
def open_oot(lock_path: Path = S8.LOCK_PATH, freeze_path: Path = S8.FREEZE_PATH, root: Path = ROOT, runner=None):
    """Chỉ mở OOT khi file khóa locked ĐÃ tồn tại VÀ đóng băng còn khớp tuyệt đối. Từ chối TRƯỚC khi đọc dữ liệu."""
    lock_path = Path(lock_path)
    if not lock_path.exists():
        raise S8.FreezeMismatch("Từ chối --oot: chưa có file khóa s8_locked_opened.lock (locked chưa được mở/chấm).")
    mism = S8.verify_freeze(freeze_path, root)
    if mism:
        raise S8.FreezeMismatch("Từ chối --oot: lệch manifest đóng băng:\n  " + "\n  ".join(mism[:30]))
    S8._STATE.update(oot_open=True, lock_path=lock_path)
    try:
        return (runner or (lambda: replay("real")))()
    finally:
        S8._STATE["oot_open"] = False


# ============================================================================ tiện ích
def psi(ref: np.ndarray, cur: np.ndarray, bins: int = MON["psi_bins"], eps: float = 1e-4) -> float:
    ref, cur = pd.Series(ref), pd.Series(cur)
    num = pd.api.types.is_numeric_dtype(ref) and ref.nunique(dropna=True) > bins
    if num:
        r = ref.dropna().to_numpy(float)
        edges = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        def dist(s):
            v = s.to_numpy(float)
            h = np.histogram(v[~np.isnan(v)], edges)[0].astype(float)
            return np.r_[h, np.isnan(v).sum()] / max(len(v), 1)
        p, q = dist(ref), dist(cur)
    else:
        cats = sorted(set(ref.astype(str)) | set(cur.astype(str)))
        p = ref.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        q = cur.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def psi_status(v: float) -> str:
    return "báo động" if v >= MON["psi_alert"] else ("cảnh báo" if v >= MON["psi_warn"] else "ổn định")


def score_light(scorer: S7.Scorer, df: pd.DataFrame) -> pd.DataFrame:
    """Điểm số của Current (không cần CI) – dùng cho PSI điểm số và kho hiệu chỉnh."""
    X, Xc = scorer.features(df)
    raw = np.clip(scorer.mu0.predict_proba(X)[:, 1], S2.EPS, 1 - S2.EPS)
    cal = scorer.cal.predict(raw)
    s_mean = np.exp(scorer.basket.predict(X)) * scorer.smear
    tau = S4.predict_tau(scorer.bundle["learner"], Xc)
    V = S6.value_hat_m(tau, cal, s_mean, M0)
    out = pd.DataFrame({"mu0_raw": raw, "mu0_cal": cal, "s_hat_mean": s_mean}, index=df.index)
    for j, a in enumerate(TREATED):
        out[f"tau_hat_{a}"], out[f"V_hat_{a}"] = tau[:, j], V[:, j]
    return out


def platt_fit(raw: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> LogisticRegression:
    return LogisticRegression(C=1e6, max_iter=1000).fit(S2._logit(raw).reshape(-1, 1), y, sample_weight=w)


def platt_apply(m: LogisticRegression, raw: np.ndarray) -> np.ndarray:
    return np.clip(m.predict_proba(S2._logit(raw).reshape(-1, 1))[:, 1], S2.EPS, 1 - S2.EPS)


def recal_for_date(pool: pd.DataFrame, d: pd.Timestamp, raw_col: str = "mu0_raw") -> tuple[LogisticRegression | None, list]:
    """Platt trên dòng a0 của RECAL_WINDOW kỳ gần nhất đã trưởng thành tại ngày d (t0 + H ≤ d)."""
    mat = np.sort(pool.loc[pool.t0 + pd.Timedelta(days=H) <= d, "t0"].unique())[-RECAL_WINDOW:]
    if len(mat) == 0:
        return None, []
    sub = pool[pool.t0.isin(mat)]
    return platt_fit(sub[raw_col].to_numpy(), sub.y_obs.to_numpy().astype(int)), [str(pd.Timestamp(t).date()) for t in mat]


def boot_sum(groups: np.ndarray, vals: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap cụm khách (multinomial) cho tổng của từng cột vals (n × k) → n_boot × k."""
    codes, uniq = pd.factorize(groups)
    per = np.zeros((len(uniq), vals.shape[1]))
    np.add.at(per, codes, vals)
    W = np.random.default_rng(seed).multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=n_boot).astype(float)
    return W @ per


def fit_mu(train: pd.DataFrame) -> tuple:
    pre = joblib.load(common.p("models/s4_propensity.joblib"))["pre"]
    X = pre.transform(train[S4.FEATS_C])
    A = train.A_logged.map({a: k for k, a in enumerate(ACTIONS)}).to_numpy()
    y = train.profit_obs.to_numpy(float)
    return pre, [LGBMRegressor(**S4.NUIS).fit(X[A == k], y[A == k]) for k in range(4)]


def predict_mu(mu: tuple, df: pd.DataFrame) -> np.ndarray:
    pre, ms = mu
    X = pre.transform(df[S4.FEATS_C])
    return np.column_stack([m.predict(X) for m in ms])


def pre_control_action(dec: pd.DataFrame) -> np.ndarray:
    """π = hành động khách nhận nếu KHÔNG bị chọn vào đối chứng (đối chứng: a1 nếu HITL hạ xuống a1, ngược lại đề xuất)."""
    fin = dec.final_action.to_numpy().astype(object)
    alt = np.where(dec.hitl_outcome.to_numpy() == "downgraded", "a1", dec.proposed_action.to_numpy())
    return np.where(dec.control_group.to_numpy(), alt, fin)


def code(a) -> np.ndarray:
    return pd.Categorical(np.asarray(a, dtype=object), categories=ACTIONS).codes.astype(int)


def dr_program(dec: pd.DataFrame, Mu: np.ndarray, Y: np.ndarray, groups: np.ndarray, t0: np.ndarray, n_boot: int) -> tuple[pd.DataFrame, dict]:
    """AIPW giá trị chương trình của Router so với a0 (định nghĩa ở docstring). Trả bảng theo kỳ + gộp, và các mảng theo dòng."""
    n = len(dec)
    rand, bf = dec.in_randomization_set.to_numpy(bool), dec.is_backfill.to_numpy(bool)
    A, pi = code(dec.final_action), code(pre_control_action(dec))
    pc = np.where(rand, dec.p_control.to_numpy(float), np.nan)
    idx = np.arange(n)
    DM = np.where(rand, Mu[idx, pi] - Mu[:, 0], np.where(bf, Mu[:, 1] - Mu[:, 0], 0.0))
    corr = np.where(rand, (A == pi) * (Y - Mu[idx, pi]) / (1 - pc) - (A == 0) * (Y - Mu[:, 0]) / pc, 0.0)
    G = DM + np.nan_to_num(corr)
    cols = np.column_stack([G, DM])
    rows = []
    for lab, m in [("Gộp mọi kỳ", np.ones(n, bool))] + [(str(pd.Timestamp(t).date()), t0 == t) for t in np.sort(np.unique(t0))]:
        bs = boot_sum(groups[m], cols[m], n_boot, SEED)
        rows.append({"ky": lab, "n_tap_ngau_nhien": int((rand & m).sum()), "n_doi_chung": int((rand & m & (A == 0)).sum()),
                     "n_bu_do_phu": int((bf & m).sum()), "DR_uoc_luong": G[m].sum(), "DR_ci_lo": np.percentile(bs[:, 0], 2.5),
                     "DR_ci_hi": np.percentile(bs[:, 0], 97.5), "DR_se": float(bs[:, 0].std(ddof=1)), "DM_chi_mo_hinh": DM[m].sum(),
                     "phan_hieu_chinh_IPW": G[m].sum() - DM[m].sum()})
    return pd.DataFrame(rows), dict(G=G, DM=DM, A=A, pi=pi, rand=rand, bf=bf)


# ============================================================================ đáp án / phản hồi mô phỏng (luật cứng #1: chỉ trong oracle_* / evaluate_*)
def oracle_drift_name() -> str:
    return "oracle_oot_drift"


def oracle_feedback(keys: pd.DataFrame, splits: list[str], actions: np.ndarray) -> dict:
    """Kết quả 'quan sát được sau triển khai' dưới hành động Router đã làm (lợi nhuận, mua) + Y(0) (chỉ dùng đánh giá)."""
    T = S8.evaluate_truth_rows(keys, splits)
    a = code(actions)
    i = np.arange(len(a))
    return dict(profit=T["profit"][i, a], y=T["Y"][i, a], y0=T["y0"], T=T)


def oracle_truth_frame(keys: pd.DataFrame, orc: pd.DataFrame) -> dict:
    t = keys[["customer_id", "t0"]].merge(orc, on=["customer_id", "t0"], how="left", validate="one_to_one")
    assert t.y_a0.notna().all()
    return dict(S=t.basket_value_S.to_numpy(), y0=t.y_a0.to_numpy().astype(float),
                Y=np.column_stack([t[f"y_{a}"] for a in ACTIONS]).astype(float),
                tau=np.column_stack([t[f"tau_cond_{a}"] for a in TREATED]),
                dc_real=np.column_stack([t[f"discount_cost_{a}"] for a in ACTIONS]),
                profit=np.column_stack([t[f"profit_{a}"] for a in ACTIONS]))


def oracle_rehearsal_pair(keys: pd.DataFrame, oot_role: str) -> tuple[dict, dict]:
    """DIỄN TẬP kịch bản trôi: sinh TRONG BỘ NHỚ cặp đáp án (không trôi / trôi 0.6) bằng s1b.generate trên snapshot train+valid+calib,
    split `oot_role` gắn nhãn 'oot'. Không ghi file; khác số ngẫu nhiên với đáp án chính thức (chỉ để chạy thử code)."""
    snap = S8.read_rows("snapshots", list(S8.MODEL_SPLITS))
    snap["split"] = np.where(snap.split.astype(str) == oot_role, "oot", snap.split.astype(str))
    out = []
    for sc in (1.0, DRIFT_SCALE):
        cfg = copy.deepcopy(CFG)
        cfg["intervention_dgp"]["oot_persuasion_scale"] = sc
        _, orc = S1B.generate(snap, cfg)
        out.append(oracle_truth_frame(keys, orc[orc.split == "oot"].drop(columns=["split"])))
    return out[0], out[1]


def oracle_drift_truth(keys: pd.DataFrame) -> tuple[dict, dict]:
    """THẬT (bước 2): đáp án không trôi = oracle.parquet; trôi = oracle_oot_drift.parquet – đều qua read_rows (split oot)."""
    T0 = S8.evaluate_truth_rows(keys, ["oot"])
    T1 = oracle_truth_frame(keys, S8.read_rows(oracle_drift_name(), ["oot"]).drop(columns=["split"]))
    return T0, T1


def oracle_make_drift_file(out_dir: Path = S8.DATA_DIR) -> pd.DataFrame:
    """Sinh đáp án kịch bản trôi (chỉ dòng oot) bằng s1b.generate với oot_persuasion_scale = 0.6, CÙNG số ngẫu nhiên với GĐ1b.
    Đọc snapshots.parquet TOÀN BỘ (bắt buộc để tái tạo đúng dòng số ngẫu nhiên) nhưng KHÔNG xem/đánh giá kết quả: chỉ kiểm tính nhất quán
    nội bộ (ngoài oot, hai lần sinh 1.0 và 0.6 phải giống hệt) rồi ghi file MỚI. Ghi bảng băm trước/sau mọi file data/processed khác."""
    out_dir = Path(out_dir)
    target = out_dir / f"{oracle_drift_name()}.parquet"
    if target.exists():
        raise FileExistsError(f"{target.name} đã tồn tại – không ghi đè.")
    others = sorted(f for f in out_dir.glob("*.parquet") if f.name != target.name)
    before = {f.name: common.sha256_file(f) for f in others}
    snap = pd.read_parquet(common.p(CFG["paths"]["snapshots"]))
    gens = {}
    for sc in (1.0, DRIFT_SCALE):
        cfg = copy.deepcopy(CFG)
        cfg["intervention_dgp"]["oot_persuasion_scale"] = sc
        gens[sc] = S1B.generate(snap, cfg)[1]
    a, b = gens[1.0], gens[DRIFT_SCALE]
    non = (a.split != "oot").to_numpy()
    cols = [c for c in a.columns if c != "oracle_best_action_unconstrained"]
    same = bool(a.loc[non, cols].reset_index(drop=True).equals(b.loc[non, cols].reset_index(drop=True)))
    assert same, "ngoài oot hai lần sinh phải giống hệt (cùng số ngẫu nhiên)"
    b[b.split == "oot"].reset_index(drop=True).to_parquet(target, index=False)
    after = {f.name: common.sha256_file(f) for f in others}
    tab = pd.DataFrame([{"file": k, "sha256_truoc": before[k], "sha256_sau": after[k], "khong_doi": before[k] == after[k]} for k in before])
    tab.loc[len(tab)] = {"file": target.name, "sha256_truoc": "(chưa có)", "sha256_sau": common.sha256_file(target), "khong_doi": None}
    tab.to_csv(common.p(TABLES / "s8_drift_generation_hashes.csv"), index=False)
    assert tab.khong_doi.iloc[:-1].all(), "một file data/processed hiện có đã bị thay đổi"
    common.write_manifest("s8_drift", [target.relative_to(ROOT), TABLES / "s8_drift_generation_hashes.csv"],
                          extra={"oot_persuasion_scale": DRIFT_SCALE, "ngoai_oot_giong_het": same, "chi_ghi_dong_split": "oot"})
    print(f"Đã sinh {target.name} (scale {DRIFT_SCALE}); {int(tab.khong_doi.iloc[:-1].sum())}/{len(tab) - 1} file hiện có không đổi.")
    return tab


def evaluate_values(T: dict, W: dict, masks: dict | None = None) -> pd.DataFrame:
    Ur, Ue = S6.evaluate_utilities(T, M0)
    rows = []
    for p, w in W.items():
        rows.append({"chinh_sach": p, "gia_tri_hien_thuc": float((w * Ur).sum()), "gia_tri_ky_vong": float((w * Ue).sum()),
                     "so_voucher": int(w[:, 1:].sum())})
    return pd.DataFrame(rows)


def evaluate_program_truth(T: dict, info: dict) -> np.ndarray:
    """Giá trị THẬT (hiện thực) theo dòng của chương trình Router so với a0: π trên tập ngẫu nhiên, a1 cho khách bù."""
    Ur, _ = S6.evaluate_utilities(T, M0)
    i = np.arange(len(info["pi"]))
    return np.where(info["rand"], Ur[i, info["pi"]], np.where(info["bf"], Ur[:, 1], 0.0))


def evaluate_selective(T: dict, fb: dict, dec: pd.DataFrame, info: dict, mu0_raw: np.ndarray, mu0_cal: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(i) tác động chương trình trên nhóm được chọn; (ii) Challenger hiệu chỉnh học từ dữ liệu Router – độ chệch so với đáp án."""
    rand, A, pi = info["rand"], info["A"], info["pi"]
    Y = fb["profit"]
    treated, ctrl = rand & (A == pi) & (A > 0), rand & (A == 0)
    nonsel = (dec.proposed_action.to_numpy() == "a0") & ~dec.is_backfill.to_numpy() & (A == 0)
    truth_row = evaluate_program_truth(T, info)
    truth_mean = truth_row[rand].mean()
    est = {"So thô: được gửi − không được chọn (KHÔNG đối chứng)": Y[treated].mean() - Y[nonsel].mean(),
           "Đối chứng ngẫu nhiên: được gửi − đối chứng": Y[treated].mean() - Y[ctrl].mean() if ctrl.any() else np.nan,
           "DR/AIPW (có đối chứng + IPW)": info["G"][rand].mean()}
    t1 = pd.DataFrame([{"phuong_phap": k, "uoc_luong_GBP_moi_khach_duoc_chon": v, "that_GBP_moi_khach_duoc_chon": truth_mean,
                        "do_chech": v - truth_mean, "n_duoc_gui": int(treated.sum()), "n_doi_chung": int(ctrl.sum()),
                        "n_khong_duoc_chon": int(nonsel.sum())} for k, v in est.items()])
    y0 = fb["y0"]
    obs_a0 = (A == 0)
    fits = {"Current (Platt GĐ2, không học lại)": None,
            "Challenger KHÔNG đối chứng (chỉ a0 không được chọn)": (obs_a0 & ~ctrl, np.ones(len(A))),
            "Challenger CÓ đối chứng + IPW (đối chứng × 1/0.1)": (obs_a0, np.where(ctrl, 1 / P_CTRL, 1.0))}
    rows = []
    for k, f in fits.items():
        if f is None:
            p = mu0_cal
        else:
            m, w = f
            p = platt_apply(platt_fit(mu0_raw[m], y0[m].astype(int), w[m]), mu0_raw)
        for grp, g in (("nhóm được chọn (tập ngẫu nhiên)", rand), ("toàn bộ", np.ones(len(A), bool))):
            rows.append({"phuong_phap": k, "nhom": grp, "n": int(g.sum()), "p_du_bao_TB": p[g].mean(), "Y0_that_TB": y0[g].mean(),
                         "do_chech": p[g].mean() - y0[g].mean(), "ece_vs_Y0_that": S2.ece_equal_mass(y0[g], p[g]),
                         "brier_vs_Y0_that": float(np.mean((p[g] - y0[g]) ** 2))})
    return t1, pd.DataFrame(rows)


def evaluate_drift(T0: dict, T1: dict, W: dict, dec: pd.DataFrame, audits: dict) -> pd.DataFrame:
    rows = []
    for sc, T in (("không trôi", T0), (f"trôi (τ × {DRIFT_SCALE})", T1)):
        v = evaluate_values(T, W).set_index("chinh_sach")
        for p in W:
            a = audits[p]
            rows.append({"kich_ban": sc, "chinh_sach": p, **v.loc[p].to_dict(), "vi_pham_quy_dinh": int(a.n_vi_pham.sum()),
                         "gui_khach_opt_out_cap_huy": int(a.vp_gui_khach_bi_chan.sum()),
                         "luot_bi_chan_cung": int((dec.guardrail_triggered != "none").sum()) if p.startswith("CÓ") else 0,
                         "HITL_tu_choi_ha_cap": int(dec.hitl_outcome.isin(["rejected", "downgraded"]).sum()) if p.startswith("CÓ") else 0})
    t = pd.DataFrame(rows)
    for sc in t.kich_ban.unique():
        m = t.kich_ban == sc
        base = t.loc[m & (t.chinh_sach == "KHÔNG guardrail: MILP thuần (miền v1)"), "gia_tri_hien_thuc"].iloc[0]
        base_e = t.loc[m & (t.chinh_sach == "KHÔNG guardrail: MILP thuần (miền v1)"), "gia_tri_ky_vong"].iloc[0]
        t.loc[m, "ton_that_tranh_duoc_hien_thuc"] = t.loc[m, "gia_tri_hien_thuc"] - base
        t.loc[m, "ton_that_tranh_duoc_ky_vong"] = t.loc[m, "gia_tri_ky_vong"] - base_e
    return t


# ============================================================================ phát lại
def replay(mode: str, out_root: Path = ROOT, n_lat: int = S8.N_LAT, n_boot: int = N_BOOT, manifest: bool = True) -> dict:
    R = ROLES["rehearsal" if mode == "rehearsal" else "real"]
    oot, tag, nhan = R["oot"], R["tag"], R["nhan"]
    out_root = Path(out_root)
    tdir, fdir, ddir = out_root / TABLES, out_root / FIGS, out_root / "data/processed"
    for d in (tdir, fdir, ddir):
        d.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    t_start = time.time()
    files: list[Path] = []

    lg = S8.read_rows("logged", [oot])
    train = S8.read_rows("logged", ["train"])
    p2tr = S8.read_rows("pred_s2", ["train"], columns=["s_lo", "s_hi"])
    s_wide = float(np.percentile(p2tr.s_hi - p2tr.s_lo, 95))
    assert abs(s_wide - S8.S_WIDE_EXPECTED) < 0.01
    ref_dec = S8.read_rows("decisions_s7_v21", R["ref"], columns=["final_action"])
    pool = S8.read_rows("logged", R["pool"], columns=FEATS + ["customer_id", "t0", "A_logged", "y_obs"])
    pool = pool[pool.A_logged == "a0"].reset_index(drop=True)
    meta = lg[["split", "t0", S5.SEG_COL, "cancel_rate_365"]].copy()
    scorer = S7.Scorer()
    t0s = np.sort(lg.t0.unique())
    n = len(lg)
    print(f"[{nhan}] {n} dòng · {len(t0s)} kỳ · {scorer.version}")

    # ---------------- tuần tự từng t0 qua Router v2.1 (Current)
    rm = S7.ActionRouterV2(s_wide, scorer.version, S7.POLICY_VERSION_V21)
    POL_PURE, POL_PROP, POL_R = "KHÔNG guardrail: MILP thuần (miền v1)", "Đề xuất MILP (miền v2.1, trước Router)", "CÓ guardrail: MILP qua Router v2.1"
    W = {p: np.zeros((n, 4)) for p in (POL_PURE, POL_PROP, POL_R)}
    sc_all = {k: np.full(n, np.nan) for k in ("mu0_raw", "mu0_cal")}
    snaps, decs, aud = {}, [], {p: [] for p in W}
    for t0 in t0s:
        ins0 = S6.make_ins(lg, meta, oot, t0)
        idx = ins0["idx"]
        base = lg.iloc[idx].reset_index(drop=True)
        rc = S3.reason_codes(scorer.mu0, base[S3.FEATS])
        for k in (1, 2, 3):
            base[f"reason_{k}_code"] = rc[f"reason_{k}_code"].to_numpy()
        X, Xc = scorer.features(base)
        sc = scorer.score(X, Xc)
        frame = rm.validate_input(S7.scored_frame_v2(base, sc))
        sc_all["mu0_raw"][idx], sc_all["mu0_cal"][idx] = sc["mu0_raw"], sc["mu0_cal"]
        snaps[("MILP", t0)] = dict(rm.history)
        fl = rm.domain_flags(frame)
        ins = S7.make_ins_v2(lg, meta, oot, t0, fl)
        W[POL_PURE][idx] = S6.solve_hs(ins0, sc["V"])["w"]
        W[POL_PROP][idx] = S6.solve_hs(ins, sc["V"])["w"]
        dec = rm.route(frame, W[POL_PROP][idx], oot, ins=ins)
        W[POL_R][idx] = S8._onehot(dec.final_action)
        decs.append(dec.assign(_row=idx))
        for p in W:
            a = S7.audit_final(ins, pd.DataFrame({"final_action": np.array(ACTIONS)[W[p][idx].argmax(1)]}), fl)
            a["n_vi_pham"] = int(sum(bool(v) for k_, v in a.items() if k_.startswith("vp_")))
            aud[p].append({"t0": pd.Timestamp(t0), **a})
        print(f"  {pd.Timestamp(t0).date()}: gửi {int(W[POL_R][idx, 1:].sum())} · đối chứng {int(dec.control_group.sum())} ({time.time() - t_start:.0f}s)")
    dec = pd.concat(decs, ignore_index=True).sort_values("_row").reset_index(drop=True)
    assert (dec._row.to_numpy() == np.arange(n)).all()
    dl = S7.DECISION_SCHEMA_V21.validate(dec.drop(columns="_row"), lazy=True)
    dl.to_parquet(ddir / f"decisions_s8_{tag}.parquet", index=False)
    files.append(ddir / f"decisions_s8_{tag}.parquet")
    audits = {p: pd.DataFrame(v) for p, v in aud.items()}
    out = {}

    # ---------------- (a) độ trễ
    lt = S8.latency(lg, meta, oot, t0s, scorer, snaps, s_wide, n_lat)
    stages = ["t_dac_trung", "t_ma_ly_do", "t_cham_diem", "t_hop_dong_va_mien", "t_milp", "t_router", "tong"]
    lat = []
    for lab, d in [("Gộp mọi kỳ", lt)] + [(str(pd.Timestamp(t).date()), lt[lt.t0 == t]) for t in t0s]:
        for s in stages:
            lat.append({"ky": lab, "khau": s, "so_luot": len(d), "P50": d[s].quantile(.5), "P95": d[s].quantile(.95), "P99": d[s].quantile(.99)})
    out["latency"] = pd.DataFrame(lat)

    # ---------------- (b) PSI đặc trưng & điểm số so với train
    tr_sc = score_light(scorer, train)
    oot_sc = score_light(scorer, lg)
    rows = []
    for kind, cols, ref, cur in (("đặc trưng", FEATS, train, lg), ("điểm số", list(tr_sc.columns.drop("mu0_raw")), tr_sc, oot_sc)):
        for c in cols:
            for lab, m in [("Gộp mọi kỳ", np.ones(n, bool))] + [(str(pd.Timestamp(t).date()), (lg.t0 == t).to_numpy()) for t in t0s]:
                v = psi(ref[c], cur[c][m] if isinstance(cur[c], pd.Series) else cur[c][m])
                rows.append({"loai": kind, "bien": c, "ky": lab, "psi": v, "trang_thai": psi_status(v),
                             "ghi_chu": "đặc trưng lịch – trôi theo thiết kế (mùa khác train)" if c in CAL_FEATS else ""})
    out["psi"] = pd.DataFrame(rows)

    # ---------------- (b) Action TVD so với tham chiếu
    ref_p = ref_dec.final_action.value_counts(normalize=True).reindex(ACTIONS, fill_value=0)
    tv = []
    for t in t0s:
        q = dec.final_action[(lg.t0 == t).to_numpy()].value_counts(normalize=True).reindex(ACTIONS, fill_value=0)
        v = 0.5 * float((q - ref_p).abs().sum())
        tv.append({"ky": str(pd.Timestamp(t).date()), **{f"p_{a}": q[a] for a in ACTIONS}, **{f"p_tham_chieu_{a}": ref_p[a] for a in ACTIONS},
                   "action_tvd": v, "nguong": MON["action_tvd_alert"], "canh_bao": v >= MON["action_tvd_alert"]})
    out["action_tvd"] = pd.DataFrame(tv)

    # ---------------- (b)(c)(e) nhãn trưởng thành: Brier/ECE Current vs Challenger (+ Challenger-2) ; bảng độ trễ nhãn
    dates = [pd.Timestamp(t) for t in t0s] + [pd.Timestamp(t0s[-1]) + pd.Timedelta(days=H)]
    pool["mu0_raw"] = np.clip(scorer.mu0.predict_proba(pool[FEATS])[:, 1], S2.EPS, 1 - S2.EPS)
    ch2 = challenger2(mode, pool, train, scorer)
    a0 = (lg.A_logged == "a0").to_numpy()
    y = lg.y_obs.to_numpy().astype(float)
    cal_rows, delay_rows = [], []
    for t in t0s:
        t = pd.Timestamp(t)
        d_eval = next(d for d in dates if t + pd.Timedelta(days=H) <= d)        # ngày phát lại ĐẦU TIÊN mà kỳ t đã trưởng thành
        m = (lg.t0 == t).to_numpy() & a0
        ch, win = recal_for_date(pool, t)
        preds = {"Current": sc_all["mu0_cal"][m]}
        preds["Challenger (Platt theo kỳ)"] = platt_apply(ch, sc_all["mu0_raw"][m]) if ch is not None else np.full(m.sum(), np.nan)
        if ch2 is not None:
            raw2 = np.clip(ch2["model"].predict_proba(lg.loc[m, FEATS])[:, 1], S2.EPS, 1 - S2.EPS)
            c2, _ = recal_for_date(ch2["pool"], t, "mu0_raw2")
            preds["Challenger-2 (μ0 học lại, PHỤ)"] = platt_apply(c2, raw2) if c2 is not None else raw2
        for k, p in preds.items():
            ok = np.isfinite(p).all() and m.sum() > 0
            cal_rows.append({"ky": str(t.date()), "phien_ban": k, "ngay_tinh": str(d_eval.date()), "tre_so_voi_quyet_dinh_ngay": (d_eval - t).days,
                             "cua_so_hieu_chinh": ", ".join(win) if k.startswith("Challenger") else "calib (GĐ2)", "n_a0": int(m.sum()),
                             "ty_le_mua": y[m].mean(), "p_TB": p.mean() if ok else np.nan,
                             "brier": float(np.mean((p - y[m]) ** 2)) if ok else np.nan, "ece": S2.ece_equal_mass(y[m], p) if ok else np.nan,
                             "canh_bao_ece": bool(S2.ece_equal_mass(y[m], p) >= MON["ece_alert"]) if ok else None})
    for d in dates:
        decided = [t for t in t0s if pd.Timestamp(t) <= d]
        mat = [t for t in t0s if pd.Timestamp(t) + pd.Timedelta(days=H) <= d]
        md = lg.t0.isin(decided).to_numpy(); mm = lg.t0.isin(mat).to_numpy()
        delay_rows.append({"ngay_phat_lai": str(d.date()), "so_ky_da_quyet_dinh": len(decided), "so_ky_nhan_truong_thanh": len(mat),
                           "so_dong_da_quyet_dinh": int(md.sum()), "so_dong_nhan_truong_thanh": int(mm.sum()),
                           "so_dong_a0_co_nhan": int((mm & a0).sum()),
                           "ty_le_dong_co_nhan": mm.sum() / max(md.sum(), 1)})
    out["calibration_matured"] = pd.DataFrame(cal_rows)
    out["label_delay_by_date"] = pd.DataFrame(delay_rows)
    first_delay = [(next(d for d in dates if pd.Timestamp(t) + pd.Timedelta(days=H) <= d) - pd.Timestamp(t)).days for t in t0s]
    out["label_delay_by_metric"] = pd.DataFrame([
        {"chi_so": c, "can_nhan": need, "tre_toi_thieu_ngay": mn, "tre_thuc_te_TB_ngay": float(np.mean(first_delay)) if need else 0.0,
         "tre_thuc_te_max_ngay": int(max(first_delay)) if need else 0, "ghi_chu": note}
        for c, need, mn, note in [
            ("Độ trễ xử lý P95 (G6)", False, 0, "đo ngay tại t0"),
            ("PSI đặc trưng / điểm số", False, 0, "đặc trưng có tại cutoff = t0 − 1 ngày"),
            ("Action TVD", False, 0, "có ngay sau khi Router ra quyết định"),
            ("Vi phạm ràng buộc (G5)", False, 0, "kiểm toán ngay trên bản ghi quyết định"),
            ("Brier / ECE (Current, Challenger)", True, H, "chờ hết cửa sổ [t0, t0+30) rồi tới ngày phát lại kế tiếp (lịch 2 tuần)"),
            ("Khớp lại Challenger (Platt)", True, H, "chỉ dùng kỳ đã trưởng thành ⇒ áp cho kỳ sau với độ trễ ≥ 30 ngày"),
            ("Giá trị thực hiện DR / đối chứng", True, H, "cần lợi nhuận trong cửa sổ kết quả"),
            ("Khuyết nhãn có chọn lọc (độ chệch)", True, H, "cần Y(0) của nhóm đối chứng")]])

    # ---------------- (d) giá trị thực hiện DR + (g) khuyết nhãn + (f) trôi
    mu = fit_mu(train)
    Mu = predict_mu(mu, lg)
    fb = oracle_feedback(lg, [oot], dec.final_action.to_numpy())
    dr, info = dr_program(dec, Mu, fb["profit"], lg.customer_id.to_numpy(), lg.t0.to_numpy(), n_boot)
    truth_row = evaluate_program_truth(fb["T"], info)
    dr["that_dap_an_hien_thuc"] = [truth_row.sum()] + [truth_row[(lg.t0 == t).to_numpy()].sum() for t in t0s]
    dr["ci_phu_that"] = (dr.DR_ci_lo <= dr.that_dap_an_hien_thuc) & (dr.that_dap_an_hien_thuc <= dr.DR_ci_hi)
    dr["ci_chua_0"] = (dr.DR_ci_lo <= 0) & (0 <= dr.DR_ci_hi)
    out["dr_value"] = dr
    out["selective_effect"], out["selective_calibration"] = evaluate_selective(fb["T"], fb, dec, info, sc_all["mu0_raw"], sc_all["mu0_cal"])
    T0, T1 = oracle_rehearsal_pair(lg, oot) if mode == "rehearsal" else oracle_drift_truth(lg)
    out["drift_guardrail"] = evaluate_drift(T0, T1, W, dec, audits)
    out["playbook"] = playbook()

    for k, v in out.items():
        v.insert(0, "nhan", nhan)
        f = tdir / f"s8_{tag}_{k}.csv"
        v.to_csv(f, index=False)
        files.append(f)
    files.append(plot_replay(fdir / f"fig_s8_{tag}.png", out, nhan))
    if manifest:
        rel = [f.relative_to(ROOT) if f.is_relative_to(ROOT) else f for f in files]
        common.write_manifest(f"s8_{tag}", rel, extra={"mode": mode, "nhan": nhan, "oot_vai": oot, "tham_chieu": R["ref"], "kho_nhan": R["pool"],
                                                       "model_version": scorer.version, "n_lat": n_lat, "n_boot": n_boot,
                                                       "runtime_s": round(time.time() - t_start)}, packages=S8.KEY_PKGS)
    print(f"XONG phát lại – {nhan} ({time.time() - t_start:.0f}s)")
    return out


def challenger2(mode: str, pool: pd.DataFrame, train: pd.DataFrame, scorer) -> dict | None:
    """μ0 học lại (PHỤ). Thật: chỉ khi đã có bảng gate locked; diễn tập: train+valid."""
    if mode != "rehearsal" and not (ROOT / TABLES / "s8_locked_gates.csv").exists():
        print("Challenger-2: bỏ qua – chưa có s8_locked_gates.csv (chỉ chạy SAU khi đã chấm gate).")
        return None
    splits = ROLES["rehearsal" if mode == "rehearsal" else "real"]["ch2_train"]
    d = S8.read_rows("logged", splits, columns=FEATS + ["A_logged", "y_obs"])
    d = d[d.A_logged == "a0"]
    model = clone(scorer.mu0).fit(d[FEATS], d.y_obs.astype(int))
    p = pool.copy()
    p["mu0_raw2"] = np.clip(model.predict_proba(p[FEATS])[:, 1], S2.EPS, 1 - S2.EPS)
    return {"model": model, "pool": p, "splits": splits}


# ============================================================================ playbook
def playbook() -> pd.DataFrame:
    G = CFG["release_gates"]
    rows = [
        ("PSI đặc trưng (không tính đặc trưng lịch)", f"≥ {MON['psi_warn']}", f"≥ {MON['psi_alert']}", "monitoring.psi_warn / psi_alert", "mỗi t0",
         "0 ngày", "Data Scientist (MLOps)", "Cảnh báo: soát pipeline dữ liệu, so phân phối; Báo động ≥ 3 đặc trưng: chuyển Challenger-2 sang đánh giá",
         "Báo động PSI điểm μ̂0 hoặc V̂ ≥ 0.25 hai kỳ liên tiếp ⇒ quay lui về Luật cũ qua Router"),
        ("PSI điểm số (μ̂0, Ŝ, τ̂, V̂)", f"≥ {MON['psi_warn']}", f"≥ {MON['psi_alert']}", "monitoring.psi_warn / psi_alert", "mỗi t0", "0 ngày",
         "Data Scientist (MLOps)", "Kiểm tra trôi đầu vào; tạm giảm ngân sách B nếu V̂ trôi lên", "như trên"),
        ("ECE μ̂0 trên kỳ đã trưởng thành", f"> {G['G2_ece_max']}", f"≥ {MON['ece_alert']}", "release_gates.G2_ece_max / monitoring.ece_alert",
         "khi nhãn trưởng thành", "≥ 30 ngày (thực tế 42)", "Data Scientist", "Báo động: dùng Challenger (Platt theo kỳ) cho kế hoạch ngân sách; giữ Current để ra quyết định cho tới khi duyệt",
         "Chi phí chiết khấu thực tế vượt kế hoạch > 20 % hai kỳ liên tiếp ⇒ chuyển hẳn sang Challenger"),
        ("Action TVD so với tham chiếu", "—", f"≥ {MON['action_tvd_alert']}", "monitoring.action_tvd_alert", "mỗi t0", "0 ngày", "Trưởng nhóm CRM",
         "Rà soát hành động bất thường (vd a2 tăng vọt); duyệt thủ công đợt gửi", "TVD ≥ 0.20 kèm PSI báo động ⇒ quay lui Luật cũ qua Router"),
        ("Độ trễ P95 đầu–cuối", "> 15 s", f"> {G['G6_latency_p95_seconds_max']} s", "release_gates.G6_latency_p95_seconds_max", "mỗi t0", "0 ngày",
         "MLOps", "Solver quá thời gian ⇒ Greedy (dự phòng Router)", "Quá ngưỡng 2 kỳ ⇒ chạy Luật cũ qua Router"),
        ("Vi phạm ràng buộc trên quyết định gửi", "—", f"> {G['G5_constraint_violations_max']}", "release_gates.G5_constraint_violations_max",
         "mỗi t0", "0 ngày", "Tài chính + MLOps", "CHẶN gửi đợt đó; sửa và chạy lại", "Bất kỳ vi phạm nào ⇒ dừng gửi tự động (HITL 100 %)"),
        ("Giá trị thực hiện DR (đối chứng 10 %)", "CI chứa 0", "CI < 0", "6b (thước đo giám sát, không chấm gate)", "khi nhãn trưởng thành",
         "≥ 30 ngày", "Trưởng nhóm CRM + Tài chính", "CI < 0: thu hẹp ngân sách, tăng tỷ lệ đối chứng", "CI < 0 hai kỳ trưởng thành liên tiếp ⇒ quay lui Luật cũ qua Router"),
        ("Tải HITL", "> 150 ca/kỳ", "> 250 ca/kỳ", "giả định 6b (≈117 ca/kỳ)", "mỗi t0", "0 ngày", "CSKH", "Bổ sung người duyệt / nới hàng đợi",
         "Không duyệt kịp ⇒ ca chưa duyệt về a0"),
        ("Tỷ lệ bị chặn cứng", "tăng > 2× tham chiếu", "—", "so với valid+calib", "mỗi t0", "0 ngày", "Trưởng nhóm CRM",
         "Soát danh sách opt-out / cap / hủy đơn", "—"),
    ]
    return pd.DataFrame(rows, columns=["canh_bao", "nguong_canh_bao", "nguong_bao_dong", "nguon_nguong", "tan_suat", "do_tre_du_lieu",
                                       "chu_so_huu", "hanh_dong", "dieu_kien_quay_lui"])


# ============================================================================ thiết kế thí điểm (cỡ mẫu)
def pilot(out_root: Path = ROOT, n_boot: int = N_BOOT, manifest: bool = True) -> dict:
    tdir, fdir = Path(out_root) / TABLES, Path(out_root) / FIGS
    tdir.mkdir(parents=True, exist_ok=True); fdir.mkdir(parents=True, exist_ok=True)
    N_L = 12947
    r_arm = CFG["intervention_dgp"]["random_arm_pct"] / 100
    # --- G3: từ s4_power_g3 (ước lượng GĐ4 trên valid/calib), công thức n = n0·((z.975+z.80)·se0/q)²
    pw = pd.read_csv(ROOT / TABLES / "s4_power_g3.csv")
    pw = pw[(pw.quan_the == "locked toàn bộ") & pw.kich_ban.str.startswith(("S1", "S2"))]
    g3 = []
    for x in pw.itertuples():
        n_need = x.n_cap_valid_calib * (Z80 * x.se_valid_calib / x.qini_gia_dinh) ** 2
        n_pair_locked = N_L * r_arm * 2 / 4
        g3.append({"kich_ban": x.kich_ban, "hanh_dong": x.hanh_dong, "qini_gia_dinh": x.qini_gia_dinh, "se_valid_calib": x.se_valid_calib,
                   "n_cap_valid_calib": x.n_cap_valid_calib, "n_cap_can_power80": n_need, "n_moi_nhanh_can": n_need / 2,
                   "n_cap_locked_thiet_ke": n_pair_locked, "boi_so_locked": n_need / n_pair_locked,
                   "ty_le_nhanh_ngau_nhien_can_tren_locked": n_need / (N_L * 2 / 4),
                   "so_ky_can_o_20pct_ngau_nhien": 3 * n_need / n_pair_locked,
                   "kha_thi_trong_locked": bool(n_need / (N_L * 2 / 4) <= 1),
                   "khop_s4_power": bool(abs(n_need - x.n_cap_can_cho_power_80) / x.n_cap_can_cho_power_80 < 1e-3)})   # lệch chỉ do làm tròn z
    g3 = pd.DataFrame(g3)
    # --- DR (A): thiết kế ghi log 20 % ngẫu nhiên (s5b) – MILP V̂ − luật cũ, SE một kỳ = se(TB kỳ)·√k
    e = pd.read_csv(ROOT / TABLES / "s5b_estimates.csv")
    e = e[(e.bien_the == "AIPW (chính)") & (e.chinh_sach == "MILP V̂ − luật cũ") & (e.don_vi_tinh == "kỳ")]
    k_per = {"valid": 4, "calib": 3}
    drA = []
    for x in e.itertuples():
        se1 = x.se_bootstrap * np.sqrt(k_per[x.split])
        for lab, delta in (("Δ = £2 000/kỳ", 2000.0), ("Δ = £4 500/kỳ (6b: chênh thật ≈ £4–5k)", 4500.0),
                           (f"Δ = ước lượng DR {x.split} (£{x.uoc_luong:,.0f})", x.uoc_luong)):
            drA.append({"thiet_ke": "Ghi log 20 % ngẫu nhiên (s5b) – MILP V̂ − luật cũ", "nguon_se": f"s5b {x.split}", "se_mot_ky": se1,
                        "kich_ban_hieu_ung": lab, "hieu_ung": delta, "so_ky_can_power80": (Z80 * se1 / delta) ** 2 if delta > 0 else np.nan,
                        "ghi_chu": "" if delta > 0 else "ước lượng âm ⇒ không dùng làm hiệu ứng mục tiêu"})
    # --- DR (B): thiết kế Router (đối chứng 10 % trong nhóm được chọn) – giá trị chương trình Current so với a0
    vc = S8.read_rows("logged", ["valid", "calib"])
    train = S8.read_rows("logged", ["train"])
    dv = S8.read_rows("decisions_s7_v21", ["valid", "calib"])
    vc["_cid"] = vc.customer_id.astype(int)
    dv = vc[["_cid", "t0"]].merge(dv, left_on=["_cid", "t0"], right_on=["customer_id", "t0"], how="left", validate="one_to_one")
    assert dv.final_action.notna().all()
    Mu = predict_mu(fit_mu(train), vc)
    fb = oracle_feedback(vc, ["valid", "calib"], dv.final_action.to_numpy())
    dr, info = dr_program(dv, Mu, fb["profit"], vc.customer_id.to_numpy(), vc.t0.to_numpy(), n_boot)
    per = dr[dr.ky != "Gộp mọi kỳ"]
    se1 = float(np.sqrt((per.DR_se ** 2).mean()))
    plan = float(dv.V_hat_proposed[dv.in_randomization_set | dv.is_backfill].sum() / len(per))
    est = float(dr.loc[dr.ky == "Gộp mọi kỳ", "DR_uoc_luong"].iloc[0] / len(per))
    drB = []
    for pc in (0.10, 0.20, 0.30, 0.50):
        se_p = se1 * np.sqrt(((1 - pc) / pc) / ((1 - P_CTRL) / P_CTRL))
        for lab, delta in ((f"Δ = V̂ kế hoạch (£{plan:,.0f}/kỳ, lạc quan)", plan), (f"Δ = ước lượng DR valid+calib (£{est:,.0f}/kỳ)", est)):
            drB.append({"thiet_ke": "Router: đối chứng trong nhóm được chọn – Current so với không làm gì", "nguon_se": "valid+calib (s7 v2.1, 7 kỳ)",
                        "ty_le_doi_chung": pc, "se_mot_ky": se_p, "kich_ban_hieu_ung": lab, "hieu_ung": delta,
                        "so_ky_can_power80": (Z80 * se_p / delta) ** 2 if delta > 0 else np.nan,
                        "ghi_chu": "SE ở tỷ lệ khác 10 % là xấp xỉ theo (1−p)/p" if pc != P_CTRL else "SE đo trực tiếp (bootstrap cụm)"})
    drv = pd.concat([pd.DataFrame(drA), pd.DataFrame(drB)], ignore_index=True)
    out = {"pilot_g3": g3, "pilot_dr": drv, "pilot_dr_router_valid_calib": dr}
    files = []
    for k, v in out.items():
        f = tdir / f"s8_{k}.csv"
        v.to_csv(f, index=False); files.append(f)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    mult = np.linspace(0.25, 40, 400)
    for x in g3[g3.kich_ban.str.startswith("S1")].itertuples():
        se_m = x.se_valid_calib * np.sqrt(x.n_cap_valid_calib / (x.n_cap_locked_thiet_ke * mult))
        ax[0].plot(mult, norm.cdf(x.qini_gia_dinh / se_m - 1.96), label=f"{x.hanh_dong} (Qini {x.qini_gia_dinh:.3f})")
    ax[0].axhline(0.8, color="k", ls="--", lw=0.8); ax[0].axvline(1, color="tab:red", ls=":", label="cỡ locked hiện có")
    ax[0].set_xscale("log"); ax[0].set_xlabel("Bội số cỡ mẫu nhánh ngẫu nhiên so với locked"); ax[0].set_ylabel("P(CI 95 % của Qini > 0)")
    ax[0].set_title("G3 – power theo cỡ mẫu (kịch bản S1, calib riêng)"); ax[0].legend(fontsize=8)
    b = drv[drv.thiet_ke.str.startswith("Router")]
    for lab in b.kich_ban_hieu_ung.unique():
        s = b[b.kich_ban_hieu_ung == lab]
        ax[1].plot(s.ty_le_doi_chung * 100, s.so_ky_can_power80, "o-", label=lab)
    ax[1].set_yscale("log"); ax[1].set_xlabel("Tỷ lệ đối chứng (%)"); ax[1].set_ylabel("Số kỳ cần (power 80 %)")
    ax[1].set_title("DR – số kỳ để CI giá trị chương trình loại trừ 0"); ax[1].legend(fontsize=8)
    fig.suptitle("Thiết kế thí điểm – cỡ mẫu (ước lượng từ valid/calib)")
    fig.tight_layout(); f = fdir / "fig_s8_pilot_sample_size.png"; fig.savefig(f, dpi=160); plt.close(fig); files.append(f)
    if manifest:
        common.write_manifest("s8_pilot", [x.relative_to(ROOT) if x.is_relative_to(ROOT) else x for x in files],
                              extra={"z_power80": Z80, "N_locked": N_L, "random_arm_pct": r_arm}, packages=S8.KEY_PKGS)
    return out


# ============================================================================ hình phát lại
def plot_replay(path: Path, out: dict, nhan: str) -> Path:
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    p = out["psi"]
    g = p[p.ky == "Gộp mọi kỳ"].sort_values("psi", ascending=False).head(15)
    ax[0, 0].barh(g.bien + " (" + g.loai + ")", g.psi, color=["tab:red" if v >= MON["psi_alert"] else ("tab:orange" if v >= MON["psi_warn"] else "tab:green") for v in g.psi])
    ax[0, 0].axvline(MON["psi_warn"], color="tab:orange", ls="--"); ax[0, 0].axvline(MON["psi_alert"], color="tab:red", ls="--")
    ax[0, 0].invert_yaxis(); ax[0, 0].set_xlabel("PSI so với train"); ax[0, 0].set_title("PSI – 15 biến trôi nhiều nhất"); ax[0, 0].tick_params(axis="y", labelsize=7)
    c = out["calibration_matured"]
    for k in c.phien_ban.unique():
        s = c[c.phien_ban == k]
        ax[0, 1].plot(s.ky, s.ece, "o-", label=k)
    ax[0, 1].axhline(MON["ece_alert"], color="tab:red", ls="--", label="ngưỡng báo động ECE"); ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_ylabel("ECE (dòng a0, khi nhãn đã trưởng thành)"); ax[0, 1].set_title("Current vs Challenger")
    d = out["drift_guardrail"]
    piv = d.pivot(index="chinh_sach", columns="kich_ban", values="gia_tri_hien_thuc")
    piv.plot.barh(ax=ax[1, 0]); ax[1, 0].axvline(0, color="k", lw=0.8)
    ax[1, 0].set_xlabel("Lợi nhuận gia tăng hiện thực (GBP, cộng dồn)"); ax[1, 0].set_title("Guardrail khi dữ liệu bị trôi"); ax[1, 0].tick_params(axis="y", labelsize=7)
    r = out["dr_value"]
    xs = np.arange(len(r))
    ax[1, 1].errorbar(xs, r.DR_uoc_luong, yerr=[r.DR_uoc_luong - r.DR_ci_lo, r.DR_ci_hi - r.DR_uoc_luong], fmt="o", capsize=4, label="DR (CI 95 %)")
    ax[1, 1].plot(xs, r.that_dap_an_hien_thuc, "rx", ms=9, label="đáp án (hiện thực)")
    ax[1, 1].axhline(0, color="k", lw=0.8); ax[1, 1].set_xticks(xs); ax[1, 1].set_xticklabels(r.ky, fontsize=8)
    ax[1, 1].set_ylabel("Giá trị chương trình so với a0 (GBP)"); ax[1, 1].set_title("Giá trị thực hiện – DR trên nhóm đối chứng"); ax[1, 1].legend(fontsize=8)
    fig.suptitle(f"Phát lại OOT – {nhan}")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return path


# ============================================================================ CLI
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="GĐ8 bước 1b – phát lại OOT, giám sát, thí điểm")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--oot-rehearsal", action="store_true")
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--make-drift", action="store_true")
    g.add_argument("--oot", action="store_true")
    a = ap.parse_args(argv)
    if a.oot_rehearsal:
        replay("rehearsal")
    elif a.pilot:
        o = pilot()
        pd.set_option("display.width", 250)
        print(o["pilot_g3"].round(3).to_string(index=False)); print(o["pilot_dr"].round(2).to_string(index=False))
    elif a.make_drift:
        oracle_make_drift_file()
    else:
        try:
            open_oot()
        except S8.FreezeMismatch as e:
            print(e)
            sys.exit(2)


if __name__ == "__main__":
    main()
