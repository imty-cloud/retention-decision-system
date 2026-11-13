"""GĐ4 – Nhân quả & uplift (Chương 4): propensity, ATE (naive / IPW / DML), CATE đa nhánh, Qini, V̂.

QUYẾT ĐỊNH ĐĂNG KÝ TRƯỚC (ghi trước khi chạy, không chọn lại sau khi xem kết quả):
  1. Mô hình propensity, DML và CATE KHÔNG dùng t0_month, days_to_xmas. Lý do: GĐ3 cho thấy hai đặc trưng này
     không ngoại suy được (train chỉ có tháng 6–12; valid là tháng 1–3, days_to_xmas valid 286–328 so với 5–201
     ở train) và chỉ đóng vai trò hằng số cộng lên log-odds, không phân biệt khách.
  2. ATE chính ước lượng trên train (cross-fitting 5 fold theo khách). Trên valid+calib chỉ CHẤM ĐIỂM bằng các
     mô hình nuisance đã học trên train (AIPW ngoài mẫu) – không khớp mô hình nào trên valid/calib.
     Sai số chuẩn: cụm theo khách (một khách xuất hiện ở nhiều t0); SE độc lập được ghi kèm để so sánh.
  3. DML: DoubleMLAPOS (mở rộng nhiều mức điều trị của điểm số IRM) cho ATE toàn quần thể của từng hành động
     so với a0. DoubleMLIRM theo cặp (chỉ dòng A∈{a0,a}) được báo cáo thêm; ước lượng lượng của nó là ATE trên
     tập con đó, nên được so với ATE thật của CHÍNH tập con đó.
  4. Chọn CATE learner đưa vào pred_s4: hệ số Qini trung bình (a1..a3) cao nhất trên NHÁNH NGẪU NHIÊN của VALID.
     Báo cáo Qini trên valid, calib và valid+calib (gộp, theo đặc tả).
  5. Khoảng tin cậy τ̂ (tau_lo/tau_hi) luôn lấy từ CausalForestDML, mức 95%, kể cả khi learner được chọn khác CF
     (cột tau_cf_a* lưu điểm của CF để đối chiếu). Router GĐ7 dùng khoảng này cho luật "CI vắt qua 0".
  6. Mức nền trong V̂: valid/calib dùng mu0_cal (GĐ2); train dùng mu0_raw NGOÀI FOLD (mu0_cal của train bị kéo
     về mức calib). Ŝ = s_hat_mean (kỳ vọng, smearing Duan). τ̂ của dòng train là dự báo ngoài fold.

Công thức (CLAUDE.md mục 1): V[i,a] = m·Ŝ·τ̂_a − min(d_a·Ŝ, cap_a)·(μ̂0+τ̂_a) − contact_a.
Chốt chặn số học duy nhất: (μ̂0+τ̂_a) được cắt vào [0,1] vì là một xác suất; số dòng bị cắt được ghi lại.

Cột data/processed/pred_s4.parquet (CHỈ train/valid/calib; locked/oot sẽ chấm ở GĐ7/GĐ8 bằng predict_cate()):
  tau_hat_a1..a3   τ̂ trên purchase từ learner được chọn – dùng trong V̂, Greedy, MILP.
  tau_lo/hi_a*     khoảng 95% của CausalForestDML – dùng cho hậu kiểm HITL (GĐ7).
  tau_cf_a*        điểm của CausalForestDML (đối chiếu).
  V_hat_a1..a3     giá trị gia tăng kỳ vọng so với a0 (GBP) – vector chi phí cho GĐ5/GĐ6.
  mu0_base, mu0_base_nguon, s_hat_mean   đầu vào của V̂.  tau_is_oof  True với dòng train.

Hàm đọc file đáp án chỉ nằm trong các hàm evaluate_*; kết quả của chúng không đi vào bất kỳ lựa chọn nào ở trên.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import time

import doubleml as dml
import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from econml.dml import CausalForestDML
from econml.dr import DRLearner
from econml.metalearners import TLearner, XLearner
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklift.metrics import qini_auc_score, qini_curve, uplift_auc_score

import common
from s1_data import FEATURES_CAT, FEATURES_NUM

CFG = common.load_config()
SEED = CFG["project"]["seed"]
TABLES, FIGS = Path(CFG["paths"]["tables"]), Path(CFG["paths"]["figures"])
CAL_FEATS = ["t0_month", "days_to_xmas"]
NUM_C = [f for f in FEATURES_NUM if f not in CAL_FEATS]
FEATS_C = NUM_C + FEATURES_CAT
ARMS = ["a0", "a1", "a2", "a3"]
TREATED = ARMS[1:]
ARM_LABEL = {"a1": "a1 email voucher 10%", "a2": "a2 email voucher 20%", "a3": "a3 gọi điện + voucher 10%"}
OUTCOMES = {"purchase": "y_obs", "profit": "profit_obs"}
OUT_LABEL = {"purchase": "Mua lại (xác suất)", "profit": "Lợi nhuận (GBP)"}
LEARNERS = ["T-learner", "X-learner", "DR-learner", "CausalForestDML"]
N_BOOT = 300
NUIS = dict(num_leaves=8, n_estimators=300, learning_rate=0.03, min_child_samples=50, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, n_jobs=4, verbose=-1)
FINAL = dict(num_leaves=4, n_estimators=200, learning_rate=0.03, min_child_samples=200, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0, random_state=SEED, n_jobs=4, verbose=-1)


# ----------------------------------------------------------------------------- tiện ích
def make_pre() -> Pipeline:
    """Tiền xử lý dùng chung cho mọi mô hình nhân quả: median + cờ thiếu, one-hot top-10 quốc gia. Chỉ fit trên train."""
    return Pipeline([("ct", ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), NUM_C),
        ("cat", OneHotEncoder(max_categories=11, handle_unknown="infrequent_if_exist", sparse_output=False),
         FEATURES_CAT)]))])


def gkf(groups) -> list:
    return list(GroupKFold(n_splits=5).split(np.zeros(len(groups)), groups=groups))


def se_cluster(IF: np.ndarray, groups) -> tuple[float, float]:
    """SE độc lập và SE cụm theo khách từ hàm ảnh hưởng (đã quy tâm)."""
    n = len(IF)
    IF = IF - IF.mean()
    se_iid = np.sqrt(np.sum(IF ** 2)) / n
    s = pd.Series(IF).groupby(np.asarray(groups)).sum().to_numpy()
    return float(se_iid), float(np.sqrt(np.sum(s ** 2)) / n)


def poisson_cluster_weights(groups, B=N_BOOT, seed=SEED) -> np.ndarray:
    """Trọng số bootstrap Poisson theo cụm khách (B × n)."""
    codes, uniq = pd.factorize(np.asarray(groups))
    W = np.random.default_rng(seed).poisson(1.0, size=(B, len(uniq))).astype(np.float32)
    return W[:, codes]


def cluster_resampler(groups, seed=SEED):
    """Sinh chỉ số dòng cho bootstrap theo cụm (lấy lại khách có hoàn lại)."""
    codes, uniq = pd.factorize(np.asarray(groups))
    order = np.argsort(codes, kind="stable")
    rows_by_c = np.split(order, np.cumsum(np.bincount(codes))[:-1])
    rng = np.random.default_rng(seed)

    def draw():
        pick = rng.integers(0, len(uniq), len(uniq))
        return np.concatenate([rows_by_c[c] for c in pick])
    return draw


# ----------------------------------------------------------------------------- propensity (giả vờ không biết)
def fit_propensity(X, A, groups):
    """LightGBM đa lớp dự báo hành động đã ghi log; trả dự báo ngoài fold trên train + mô hình học trên toàn train."""
    oof = np.zeros((len(A), 4))
    for tr, te in gkf(groups):
        oof[te] = LGBMClassifier(**NUIS).fit(X[tr], A[tr]).predict_proba(X[te])
    return oof, LGBMClassifier(**NUIS).fit(X, A)


def propensity_compare(P_est, P_log, A, split_name) -> list[dict]:
    rows = []
    ll_est = -np.mean(np.log(np.clip(P_est[np.arange(len(A)), A], 1e-9, 1)))
    ll_log = -np.mean(np.log(P_log[np.arange(len(A)), A]))
    for k, a in enumerate(ARMS):
        rows.append({"tap": split_name, "hanh_dong": a, "TB_uoc_luong": P_est[:, k].mean(),
                     "TB_ghi_log": P_log[:, k].mean(), "MAE": np.abs(P_est[:, k] - P_log[:, k]).mean(),
                     "tuong_quan_pearson": np.corrcoef(P_est[:, k], P_log[:, k])[0, 1],
                     "min_uoc_luong": P_est[:, k].min(), "min_ghi_log": P_log[:, k].min(),
                     "ty_le_uoc_luong_duoi_0.05": (P_est[:, k] < 0.05).mean(),
                     "log_loss_A_uoc_luong": ll_est, "log_loss_A_ghi_log": ll_log})
    return rows


# ----------------------------------------------------------------------------- ATE
def ate_naive_ipw(y, A, groups, P=None) -> dict:
    """P=None → chênh lệch trung bình thô; P (n×4) → IPW chuẩn hóa (Hájek). CI 95% bootstrap Poisson theo cụm khách."""
    W = poisson_cluster_weights(groups)
    wts = np.ones(len(y)) if P is None else None

    def arm(k):
        m = A == k
        w = wts[m] if P is None else 1.0 / P[m, k]
        return np.sum(w * y[m]) / np.sum(w), (W[:, m] @ (w * y[m])) / (W[:, m] @ w)

    p0, b0 = arm(0)
    out = {}
    for k in (1, 2, 3):
        pk, bk = arm(k)
        bb = bk - b0
        out[ARMS[k]] = dict(uoc_luong=pk - p0, ci_lo=np.percentile(bb, 2.5), ci_hi=np.percentile(bb, 97.5),
                            se_cum=float(np.std(bb, ddof=1)), n=len(y))
    return out


def dml_apos(X, y, A, groups) -> dict:
    """DoubleMLAPOS: E[Y(a)] cho a0..a3 với cross-fitting GroupKFold(5) theo khách; tương phản a − a0, SE cụm."""
    xc = [f"x{j}" for j in range(X.shape[1])]
    d = pd.DataFrame(X, columns=xc)
    d["y"], d["A"] = y, A
    ap = dml.DoubleMLAPOS(dml.DoubleMLData(d, "y", "A", xc), ml_g=LGBMRegressor(**NUIS),
                          ml_m=LGBMClassifier(**NUIS), treatment_levels=[0, 1, 2, 3], n_folds=5,
                          draw_sample_splitting=False)
    ap.set_sample_splitting([gkf(groups)])
    ap.fit()
    IFs = [-(m.psi[:, 0, 0] / m.psi_elements["psi_a"][:, 0, 0].mean()) for m in ap.modellist]
    out = {}
    for k in (1, 2, 3):
        est = float(ap.coef[k] - ap.coef[0])
        se_i, se_c = se_cluster(IFs[k] - IFs[0], groups)
        out[ARMS[k]] = dict(uoc_luong=est, ci_lo=est - 1.96 * se_c, ci_hi=est + 1.96 * se_c, se_cum=se_c,
                            se_doc_lap=se_i, n=len(y))
    return out


def dml_irm_pair(X, y, A, groups, k) -> tuple[dict, np.ndarray]:
    """DoubleMLIRM cho cặp (a0, a_k) trên tập con A∈{a0,a_k}; ước lượng lượng = ATE trên tập con đó."""
    m = (A == 0) | (A == k)
    xc = [f"x{j}" for j in range(X.shape[1])]
    d = pd.DataFrame(X[m], columns=xc)
    d["y"], d["D"] = y[m], (A[m] == k).astype(int)
    g = np.asarray(groups)[m]
    irm = dml.DoubleMLIRM(dml.DoubleMLData(d, "y", "D", xc), ml_g=LGBMRegressor(**NUIS),
                          ml_m=LGBMClassifier(**NUIS), n_folds=5, score="ATE", draw_sample_splitting=False)
    irm.set_sample_splitting([gkf(g)])
    irm.fit()
    IF = -(irm.psi[:, 0, 0] / irm.psi_elements["psi_a"][:, 0, 0].mean())
    est = float(irm.coef[0])
    se_i, se_c = se_cluster(IF, g)
    return dict(uoc_luong=est, ci_lo=est - 1.96 * se_c, ci_hi=est + 1.96 * se_c, se_cum=se_c, se_doc_lap=se_i,
                n=int(m.sum())), m


class OutOfSampleAIPW:
    """AIPW chấm điểm trên tập mới bằng μ̂_a và ê_a đã học trên train (không khớp gì trên tập mới)."""

    def fit(self, X, y, A):
        self.mu = [LGBMRegressor(**NUIS).fit(X[A == k], y[A == k]) for k in range(4)]
        return self

    def ate(self, X, y, A, P, groups) -> dict:
        P = np.clip(P, 0.01, 1)
        Mu = np.column_stack([m.predict(X) for m in self.mu])
        psi = np.column_stack([Mu[:, k] + (A == k) * (y - Mu[:, k]) / P[:, k] for k in range(4)])
        out = {}
        for k in (1, 2, 3):
            IF = psi[:, k] - psi[:, 0]
            est = float(IF.mean())
            se_i, se_c = se_cluster(IF, groups)
            out[ARMS[k]] = dict(uoc_luong=est, ci_lo=est - 1.96 * se_c, ci_hi=est + 1.96 * se_c, se_cum=se_c,
                                se_doc_lap=se_i, n=len(y))
        return out


# ----------------------------------------------------------------------------- CATE
def make_learner(name: str):
    if name == "T-learner":
        return TLearner(models=LGBMRegressor(**NUIS), categories=ARMS)
    if name == "X-learner":
        return XLearner(models=LGBMRegressor(**NUIS), propensity_model=LGBMClassifier(**NUIS), categories=ARMS)
    if name == "DR-learner":
        return DRLearner(model_propensity=LGBMClassifier(**NUIS), model_regression=LGBMRegressor(**NUIS),
                         model_final=LGBMRegressor(**FINAL), categories=ARMS, cv=5, min_propensity=0.01,
                         random_state=SEED)
    if name == "CausalForestDML":
        return CausalForestDML(model_y=LGBMRegressor(**NUIS), model_t=LGBMClassifier(**NUIS), discrete_treatment=True,
                               categories=ARMS, cv=5, n_estimators=400, min_samples_leaf=50, max_samples=0.45,
                               random_state=SEED, n_jobs=-1)
    raise ValueError(name)


def fit_learner(name, X, y, A_str, groups):
    L = make_learner(name)
    if name in ("DR-learner", "CausalForestDML"):
        L.fit(y, A_str, X=X, groups=np.asarray(groups))      # cross-fitting nội bộ giữ nguyên cụm khách
    else:
        L.fit(y, A_str, X=X)
    return L


def predict_tau(L, X) -> np.ndarray:
    return np.column_stack([np.ravel(L.effect(X, T0="a0", T1=a)) for a in TREATED])


def predict_tau_interval(cf, X, alpha=0.05) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = zip(*[cf.effect_interval(X, T0="a0", T1=a, alpha=alpha) for a in TREATED])
    return np.column_stack([np.ravel(v) for v in lo]), np.column_stack([np.ravel(v) for v in hi])


def predict_cate(df: pd.DataFrame) -> pd.DataFrame:
    """Chấm τ̂ cho bảng đặc trưng bất kỳ (không cần nhãn) bằng mô hình đã lưu – dùng ở GĐ7/GĐ8 cho locked/oot."""
    bundle = joblib.load(common.p("models/s4_cate_bundle.joblib"))
    X = bundle["pre"].transform(df[FEATS_C])
    tau = predict_tau(bundle["learner"], X)
    lo, hi = predict_tau_interval(bundle["cf"], X)
    out = pd.DataFrame(index=df.index)
    for j, a in enumerate(TREATED):
        out[f"tau_hat_{a}"], out[f"tau_lo_{a}"], out[f"tau_hi_{a}"] = tau[:, j], lo[:, j], hi[:, j]
    return out


def qini_table(d: pd.DataFrame, tau: np.ndarray, label: str, learner: str, boot: bool) -> list[dict]:
    """Qini/AUUC (scikit-uplift) trên NHÁNH NGẪU NHIÊN: với mỗi a, so dòng A=a với A=a0 (xác suất 1/2 – như RCT)."""
    rows = []
    for j, a in enumerate(TREATED):
        m = ((d.is_random_arm == 1) & d.A_logged.isin(["a0", a])).to_numpy()
        y, t, s = d.y_obs.to_numpy()[m], (d.A_logged.to_numpy()[m] == a).astype(int), tau[m, j]
        r = {"tap": label, "learner": learner, "hanh_dong": a, "n": int(m.sum()), "n_dieu_tri": int(t.sum()),
             "qini": qini_auc_score(y, s, t), "auuc": uplift_auc_score(y, s, t)}
        if boot:
            draw = cluster_resampler(d.customer_id.to_numpy()[m], seed=SEED + j)   # cùng hạt giống cho mọi learner
            qs = []
            for _ in range(N_BOOT):
                idx = draw()
                if t[idx].min() == t[idx].max():
                    continue
                qs.append(qini_auc_score(y[idx], s[idx], t[idx]))
            r["qini_ci_lo"], r["qini_ci_hi"] = np.percentile(qs, 2.5), np.percentile(qs, 97.5)
        rows.append(r)
    return rows


# ----------------------------------------------------------------------------- V̂
def value_hat(tau: np.ndarray, mu0: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, int]:
    biz = CFG["business"]
    m = biz["gross_margin_rate"]
    V, n_clip = np.zeros_like(tau), 0
    for j, a in enumerate(TREATED):
        spec = biz["actions"][a]
        p_raw = mu0 + tau[:, j]
        n_clip += int(((p_raw < 0) | (p_raw > 1)).sum())
        V[:, j] = m * S * tau[:, j] - np.minimum(spec["discount"] * S, spec["face_value_cap"]) * np.clip(p_raw, 0, 1) \
            - spec["contact_cost"]
    return V, n_clip


# ----------------------------------------------------------------------------- ĐÁNH GIÁ (được phép đọc đáp án)
def evaluate_load_truth(keys: pd.DataFrame) -> pd.DataFrame:
    orc = pd.read_parquet(common.p("data/processed/oracle.parquet"))
    out = keys[["customer_id", "t0"]].merge(orc.drop(columns=["split"]), on=["customer_id", "t0"], how="left",
                                            validate="one_to_one")
    assert out[["y_a0", "profit_a0"]].notna().all().all()
    return out.set_index(keys.index)


def evaluate_true_ate(keys: pd.DataFrame, mask_by_arm: dict | None = None) -> dict:
    """ATE thật trên đúng các dòng được truyền vào; mask_by_arm cho phép tập con riêng từng hành động (IRM cặp)."""
    t = evaluate_load_truth(keys)
    out = {}
    for a in TREATED:
        tt = t if mask_by_arm is None else t[mask_by_arm[a]]
        out[a] = {"purchase": float((tt[f"y_{a}"] - tt["y_a0"]).mean()),
                  "profit": float((tt[f"profit_{a}"] - tt["profit_a0"]).mean())}
    return out


def evaluate_ate_table(ate: pd.DataFrame, truths: dict) -> pd.DataFrame:
    """Ghép ATE ước lượng với ATE thật; truths[(tap, phuong_phap)] → dict a → outcome → giá trị."""
    rows = []
    for _, r in ate.iterrows():
        tr = truths.get((r.tap, r.phuong_phap), truths.get((r.tap, "*")))
        true = tr[r.hanh_dong][r.ket_qua]
        rows.append({**r.to_dict(), "ate_that": true, "sai_lech": r.uoc_luong - true,
                     "ci_phu_ate_that": bool(r.ci_lo <= true <= r.ci_hi)})
    return pd.DataFrame(rows)


def evaluate_forest_plot(tab: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))
    for ax, outc in zip(axes, ["purchase", "profit"]):
        sub = tab[(tab.ket_qua == outc) & (tab.tap == "train")]
        methods = list(dict.fromkeys(sub.phuong_phap))
        ypos, labels = 0, []
        for a in TREATED:
            for mth in methods:
                r = sub[(sub.hanh_dong == a) & (sub.phuong_phap == mth)].iloc[0]
                ax.errorbar(r.uoc_luong, ypos, xerr=[[r.uoc_luong - r.ci_lo], [r.ci_hi - r.uoc_luong]], fmt="o",
                            color="tab:blue" if "DML" in mth else "tab:gray", capsize=3)
                ax.plot(r.ate_that, ypos, "D", color="tab:red", ms=6,
                        label="ATE thật (đáp án)" if ypos == 0 else None)
                labels.append(f"{a} · {mth}")
                ypos -= 1
            ypos -= 0.6
        yt = []
        y = 0
        for a in TREATED:
            for _ in methods:
                yt.append(y)
                y -= 1
            y -= 0.6
        ax.set_yticks(yt)
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(f"ATE so với a0 – {OUT_LABEL[outc]}")
        ax.set_title(f"ATE trên train: {OUT_LABEL[outc]} (CI 95%, SE cụm theo khách)")
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_cate_vs_truth(keys: pd.DataFrame, taus: dict[str, np.ndarray], label: str) -> list[dict]:
    """Tương quan hạng τ̂ với tau_cond (đáp án) – trên mọi dòng của tập (không chỉ nhánh ngẫu nhiên)."""
    t = evaluate_load_truth(keys)
    rows = []
    for name, tau in taus.items():
        for j, a in enumerate(TREATED):
            tc = t[f"tau_cond_{a}"].to_numpy()
            rows.append({"tap": label, "learner": name, "hanh_dong": a,
                         "spearman_tau_hat_vs_tau_cond": spearmanr(tau[:, j], tc).statistic,
                         "TB_tau_hat": tau[:, j].mean(), "TB_tau_cond": tc.mean(),
                         "TB_tac_dong_mua_that": float((t[f"y_{a}"] - t["y_a0"]).mean())})
    return rows


def evaluate_cate_calibration_plot(keys: pd.DataFrame, tau: np.ndarray, learner: str, path: Path) -> None:
    t = evaluate_load_truth(keys)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
    for j, (ax, a) in enumerate(zip(axes, TREATED)):
        q = pd.qcut(tau[:, j], 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"q": q, "hat": tau[:, j], "true": t[f"tau_cond_{a}"].to_numpy()}).groupby("q").mean()
        ax.plot(g.hat, g.true, "o-", label="Thập phân vị τ̂")
        lim = [min(g.hat.min(), g.true.min()), max(g.hat.max(), g.true.max())]
        ax.plot(lim, lim, "k--", lw=1, label="y = x")
        ax.set_title(ARM_LABEL[a])
        ax.set_xlabel("τ̂ trung bình trong nhóm")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("tau_cond trung bình (đáp án)")
    fig.suptitle(f"Hiệu chỉnh CATE ({learner}) trên valid+calib – so với tác động thật")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_vhat_vs_truth(keys: pd.DataFrame, V: np.ndarray, label: str) -> list[dict]:
    t = evaluate_load_truth(keys)
    rows = []
    for j, a in enumerate(TREATED):
        up = t[f"uplift_profit_{a}"].to_numpy()
        pos = V[:, j] > 0
        rows.append({"tap": label, "hanh_dong": a, "spearman_V_hat_vs_uplift_profit": spearmanr(V[:, j], up).statistic,
                     "TB_V_hat": V[:, j].mean(), "TB_uplift_profit_that": up.mean(),
                     "ty_le_V_hat_duong": pos.mean(), "ty_le_uplift_that_duong": (up > 0).mean(),
                     "uplift_that_TB_khi_V_hat_duong": up[pos].mean() if pos.any() else np.nan,
                     "uplift_that_TB_khi_V_hat_am": up[~pos].mean() if (~pos).any() else np.nan})
    return rows


# ----------------------------------------------------------------------------- chính
def main() -> None:
    np.random.seed(SEED)
    t_start = time.time()
    files: list[str] = []
    df = pd.read_parquet(common.p("data/processed/logged.parquet"))
    df = df[df.split.isin(["train", "valid", "calib"])].reset_index(drop=True)   # không đụng locked/oot
    A_code = df.A_logged.map({a: k for k, a in enumerate(ARMS)}).to_numpy()
    tr_m = (df.split == "train").to_numpy()
    vc_m = df.split.isin(["valid", "calib"]).to_numpy()
    tr, vc = df[tr_m].reset_index(drop=True), df[vc_m].reset_index(drop=True)
    A_tr, A_vc = A_code[tr_m], A_code[vc_m]
    g_tr, g_vc = tr.customer_id.to_numpy(), vc.customer_id.to_numpy()

    pre = make_pre().fit(tr[FEATS_C])
    X_tr, X_vc = pre.transform(tr[FEATS_C]), pre.transform(vc[FEATS_C])
    P_log_tr = tr[[f"propensity_{a}" for a in ARMS]].to_numpy()
    P_log_vc = vc[[f"propensity_{a}" for a in ARMS]].to_numpy()
    print(f"train {len(tr)} dòng · valid+calib {len(vc)} dòng · {X_tr.shape[1]} cột đặc trưng (không có lịch)")

    # ---- 1. Propensity ước lượng vs ghi log
    P_oof_tr, prop_model = fit_propensity(X_tr, A_tr, g_tr)
    P_est_vc = prop_model.predict_proba(X_vc)
    prop = pd.DataFrame(propensity_compare(P_oof_tr, P_log_tr, A_tr, "train (ngoài fold)") +
                        propensity_compare(P_est_vc, P_log_vc, A_vc, "valid+calib"))
    prop.to_csv(common.p(TABLES / "s4_propensity_compare.csv"), index=False)
    files.append(str(TABLES / "s4_propensity_compare.csv"))
    print(prop.round(4).to_string())

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for k, a in enumerate(ARMS):
        ax = axes[0, k]
        for A_val, c, lab in [(k, "tab:blue", f"A = {a}"), (None, "tab:orange", f"A ≠ {a}")]:
            msk = A_tr == k if A_val is not None else A_tr != k
            ax.hist(P_oof_tr[msk, k], bins=40, alpha=0.55, color=c, density=True, label=lab)
        ax.axvline(0.05, color="k", ls=":", lw=1)
        ax.set_title(f"Chồng lấp ê({a}|x) – train")
        ax.set_xlabel("propensity ước lượng (ngoài fold)")
        ax.legend(fontsize=8)
        ax = axes[1, k]
        ax.scatter(P_log_tr[:, k], P_oof_tr[:, k], s=2, alpha=0.15)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.8)
        ax.set_xlabel(f"propensity ghi log ({a})"); ax.set_ylabel("propensity ước lượng")
        r = prop[(prop.tap.str.startswith("train")) & (prop.hanh_dong == a)].iloc[0]
        ax.set_title(f"MAE={r.MAE:.3f} · r={r.tuong_quan_pearson:.3f}")
    fig.suptitle("Propensity: ước lượng bằng mô hình (giả vờ không biết) so với propensity ghi log")
    fig.tight_layout()
    fig.savefig(common.p(FIGS / "fig_s4_propensity_overlap.png"), dpi=160); plt.close(fig)
    files.append(str(FIGS / "fig_s4_propensity_overlap.png"))

    # ---- 2. ATE: naive / IPW (log, ước lượng) / DML-APOS / DML-IRM cặp – trên train
    ate_rows, irm_masks = [], {}
    for outc, col in OUTCOMES.items():
        y = tr[col].to_numpy(float)
        res = {"Naive (chênh lệch thô)": ate_naive_ipw(y, A_tr, g_tr),
               "IPW (propensity ghi log)": ate_naive_ipw(y, A_tr, g_tr, P_log_tr),
               "IPW (propensity ước lượng)": ate_naive_ipw(y, A_tr, g_tr, np.clip(P_oof_tr, 0.01, 1)),
               "DML-APOS (cross-fit 5)": dml_apos(X_tr, y, A_tr, g_tr)}
        pair = {}
        for k in (1, 2, 3):
            pair[ARMS[k]], irm_masks[ARMS[k]] = dml_irm_pair(X_tr, y, A_tr, g_tr, k)
        res["DML-IRM cặp (tập con a0∪a)"] = pair
        for mth, d in res.items():
            for a, r in d.items():
                ate_rows.append({"tap": "train", "ket_qua": outc, "hanh_dong": a, "phuong_phap": mth, **r})
        # valid+calib: ngoài mẫu, không khớp gì trên tập này
        y_vc = vc[col].to_numpy(float)
        aipw = OutOfSampleAIPW().fit(X_tr, y, A_tr)
        res_vc = {"Naive (chênh lệch thô)": ate_naive_ipw(y_vc, A_vc, g_vc),
                  "IPW (propensity ghi log)": ate_naive_ipw(y_vc, A_vc, g_vc, P_log_vc),
                  "AIPW ngoài mẫu (nuisance từ train)": aipw.ate(X_vc, y_vc, A_vc, P_est_vc, g_vc)}
        for mth, d in res_vc.items():
            for a, r in d.items():
                ate_rows.append({"tap": "valid+calib", "ket_qua": outc, "hanh_dong": a, "phuong_phap": mth, **r})
        print(f"ATE {outc} xong ({time.time() - t_start:.0f}s)")
    ate = pd.DataFrame(ate_rows)
    ate.to_csv(common.p(TABLES / "s4_ate.csv"), index=False)
    files.append(str(TABLES / "s4_ate.csv"))

    truths = {("train", "*"): evaluate_true_ate(tr),
              ("train", "DML-IRM cặp (tập con a0∪a)"): evaluate_true_ate(tr, irm_masks),
              ("valid+calib", "*"): evaluate_true_ate(vc)}
    ate_eval = evaluate_ate_table(ate, truths)
    ate_eval.to_csv(common.p(TABLES / "s4_ate_vs_truth.csv"), index=False)
    files.append(str(TABLES / "s4_ate_vs_truth.csv"))
    print(ate_eval[["tap", "ket_qua", "hanh_dong", "phuong_phap", "uoc_luong", "ci_lo", "ci_hi", "ate_that",
                    "ci_phu_ate_that"]].round(4).to_string())
    evaluate_forest_plot(ate_eval, common.p(FIGS / "fig_s4_ate_forest.png"))
    files.append(str(FIGS / "fig_s4_ate_forest.png"))

    # ---- 3. Chẩn đoán: placebo (xáo A trong từng t0) + ổn định theo tập con
    rng = np.random.default_rng(SEED)
    A_shuf = A_tr.copy()
    for _, idx in tr.groupby("t0").indices.items():
        A_shuf[idx] = rng.permutation(A_tr[idx])
    plc = []
    for outc, col in OUTCOMES.items():
        for a, r in dml_apos(X_tr, tr[col].to_numpy(float), A_shuf, g_tr).items():
            plc.append({"ket_qua": outc, "hanh_dong": a, **r, "ci_phu_0": bool(r["ci_lo"] <= 0 <= r["ci_hi"])})
    plc = pd.DataFrame(plc)
    plc.to_csv(common.p(TABLES / "s4_placebo.csv"), index=False)
    files.append(str(TABLES / "s4_placebo.csv"))
    print("PLACEBO\n", plc.round(4).to_string())

    t0s = np.sort(tr.t0.unique())
    cust = np.unique(g_tr)
    half_a = set(np.random.default_rng(SEED + 1).choice(cust, len(cust) // 2, replace=False))
    subsets = {"t0 nửa đầu (7 kỳ)": tr.t0.isin(t0s[:7]).to_numpy(),
               "t0 nửa sau (8 kỳ)": tr.t0.isin(t0s[7:]).to_numpy(),
               "khách nhóm A": np.isin(g_tr, list(half_a)), "khách nhóm B": ~np.isin(g_tr, list(half_a))}
    stab, stab_truth = [], {}
    for sname, sm in subsets.items():
        stab_truth[(sname, "*")] = evaluate_true_ate(tr[sm])
        for outc, col in OUTCOMES.items():
            for a, r in dml_apos(X_tr[sm], tr[col].to_numpy(float)[sm], A_tr[sm], g_tr[sm]).items():
                stab.append({"tap": sname, "ket_qua": outc, "hanh_dong": a, "phuong_phap": "DML-APOS", **r})
    stab = evaluate_ate_table(pd.DataFrame(stab), stab_truth)
    stab.to_csv(common.p(TABLES / "s4_ate_stability.csv"), index=False)
    files.append(str(TABLES / "s4_ate_stability.csv"))
    print("ỔN ĐỊNH\n", stab[["tap", "ket_qua", "hanh_dong", "uoc_luong", "ci_lo", "ci_hi", "ate_that",
                            "ci_phu_ate_that"]].round(4).to_string())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, outc in zip(axes, OUTCOMES):
        s = stab[stab.ket_qua == outc].reset_index(drop=True)
        for i, r in s.iterrows():
            ax.errorbar(r.uoc_luong, -i, xerr=[[r.uoc_luong - r.ci_lo], [r.ci_hi - r.uoc_luong]], fmt="o",
                        color="tab:blue", capsize=3)
            ax.plot(r.ate_that, -i, "D", color="tab:red", ms=5, label="ATE thật" if i == 0 else None)
        ax.set_yticks(-np.arange(len(s)))
        ax.set_yticklabels([f"{r.hanh_dong} · {r.tap}" for r in s.itertuples()], fontsize=8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(f"ATE DML-APOS – {OUT_LABEL[outc]}")
        ax.legend(fontsize=8)
    fig.suptitle("Ổn định ATE theo tập con của train (CI 95%, SE cụm)")
    fig.tight_layout()
    fig.savefig(common.p(FIGS / "fig_s4_ate_stability.png"), dpi=160); plt.close(fig)
    files.append(str(FIGS / "fig_s4_ate_stability.png"))
    print(f"chẩn đoán xong ({time.time() - t_start:.0f}s)")

    # ---- 4. CATE đa nhánh trên purchase: học trên toàn train → valid/calib; ngoài fold cho train
    y_tr = tr.y_obs.to_numpy(float)
    A_str_tr = tr.A_logged.to_numpy().astype(str)   # kiểu chuỗi Unicode (EconML metalearner từ chối dtype object)
    full, tau_vc, tau_oof = {}, {}, {n: np.zeros((len(tr), 3)) for n in LEARNERS}
    cf_lo_oof, cf_hi_oof = np.zeros((len(tr), 3)), np.zeros((len(tr), 3))
    for name in LEARNERS:
        full[name] = fit_learner(name, X_tr, y_tr, A_str_tr, g_tr)
        tau_vc[name] = predict_tau(full[name], X_vc)
    for f, (itr, ite) in enumerate(gkf(g_tr)):
        for name in LEARNERS:
            L = fit_learner(name, X_tr[itr], y_tr[itr], A_str_tr[itr], g_tr[itr])
            tau_oof[name][ite] = predict_tau(L, X_tr[ite])
            if name == "CausalForestDML":
                cf_lo_oof[ite], cf_hi_oof[ite] = predict_tau_interval(L, X_tr[ite])
        print(f"  CATE fold {f + 1}/5 xong ({time.time() - t_start:.0f}s)")

    qrows = []
    for name in LEARNERS:
        for lab, msk in [("valid", vc.split.eq("valid")), ("calib", vc.split.eq("calib"))]:
            qrows += qini_table(vc[msk.to_numpy()], tau_vc[name][msk.to_numpy()], lab, name, boot=False)
        qrows += qini_table(vc, tau_vc[name], "valid+calib", name, boot=True)
        qrows += qini_table(tr, tau_oof[name], "train (ngoài fold)", name, boot=False)
    qini = pd.DataFrame(qrows)
    sel_score = qini[qini.tap == "valid"].groupby("learner").qini.mean()
    chosen = sel_score.idxmax()
    qini["duoc_chon"] = qini.learner.eq(chosen)
    qini["ghi_chu"] = qini.tap.map({
        "valid": "Tập dùng để CHỌN learner → lạc quan (thiên lệch chọn mô hình) cho learner được chọn",
        "calib": "KHÔNG dùng để chọn learner → ước lượng KHÔNG bị thiên lệch do chọn mô hình (nên dùng để dự báo locked)",
        "valid+calib": "Gộp theo đặc tả; chứa phần valid đã dùng để chọn → hơi lạc quan",
        "train (ngoài fold)": "Ngoài fold trên train (cùng giai đoạn với dữ liệu học) – chỉ để tham khảo"})
    qini.to_csv(common.p(TABLES / "s4_qini.csv"), index=False)
    files.append(str(TABLES / "s4_qini.csv"))
    print(qini.round(4).to_string())
    print("Qini TB trên nhánh ngẫu nhiên valid (quy tắc chọn):", sel_score.round(4).to_dict(), "→", chosen)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for j, (ax, a) in enumerate(zip(axes, TREATED)):
        m = ((vc.is_random_arm == 1) & vc.A_logged.isin(["a0", a])).to_numpy()
        yy, tt = vc.y_obs.to_numpy()[m], (vc.A_logged.to_numpy()[m] == a).astype(int)
        for name in LEARNERS:
            xq, yq = qini_curve(yy, tau_vc[name][m, j], tt)
            q = qini[(qini.tap == "valid+calib") & (qini.learner == name) & (qini.hanh_dong == a)].qini.iloc[0]
            ax.plot(xq / m.sum(), yq, lw=2 if name == chosen else 1.2, label=f"{name} (Qini={q:.3f})")
        ax.plot([0, 1], [0, yq[-1]], "k--", lw=1, label="Ngẫu nhiên")
        ax.set_title(ARM_LABEL[a])
        ax.set_xlabel("Tỷ lệ khách được nhắm (xếp theo τ̂ giảm dần)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Số lượt mua tăng thêm (Qini)")
    fig.suptitle("Đường Qini trên NHÁNH NGẪU NHIÊN của valid+calib (ngoài mẫu)")
    fig.tight_layout()
    fig.savefig(common.p(FIGS / "fig_s4_qini_curves.png"), dpi=160); plt.close(fig)
    files.append(str(FIGS / "fig_s4_qini_curves.png"))

    cvt = pd.DataFrame(evaluate_cate_vs_truth(vc, tau_vc, "valid+calib") +
                       evaluate_cate_vs_truth(tr, tau_oof, "train (ngoài fold)"))
    cvt.to_csv(common.p(TABLES / "s4_cate_vs_truth.csv"), index=False)
    files.append(str(TABLES / "s4_cate_vs_truth.csv"))
    print(cvt.round(4).to_string())
    evaluate_cate_calibration_plot(vc, tau_vc[chosen], chosen, common.p(FIGS / "fig_s4_cate_calibration_truth.png"))
    files.append(str(FIGS / "fig_s4_cate_calibration_truth.png"))

    # ổn định CATE: learner được chọn học trên hai nửa khách → tương quan hạng τ̂ trên valid+calib
    hA = np.isin(g_tr, list(half_a))
    LA = fit_learner(chosen, X_tr[hA], y_tr[hA], A_str_tr[hA], g_tr[hA])
    LB = fit_learner(chosen, X_tr[~hA], y_tr[~hA], A_str_tr[~hA], g_tr[~hA])
    tA, tB = predict_tau(LA, X_vc), predict_tau(LB, X_vc)
    cstab = []
    for j, a in enumerate(TREATED):
        topA = tA[:, j] >= np.quantile(tA[:, j], 0.8)
        topB = tB[:, j] >= np.quantile(tB[:, j], 0.8)
        cstab.append({"learner": chosen, "hanh_dong": a, "spearman_nuaA_vs_nuaB": spearmanr(tA[:, j], tB[:, j]).statistic,
                      "trung_top20_jaccard": (topA & topB).sum() / (topA | topB).sum(),
                      "spearman_nuaA_vs_toan_bo": spearmanr(tA[:, j], tau_vc[chosen][:, j]).statistic})
    cstab = pd.DataFrame(cstab)
    cstab.to_csv(common.p(TABLES / "s4_cate_stability.csv"), index=False)
    files.append(str(TABLES / "s4_cate_stability.csv"))
    print(cstab.round(4).to_string())

    # ---- 5. V̂ và pred_s4
    p2 = pd.read_parquet(common.p("data/processed/pred_s2.parquet"))
    p2 = df[["customer_id", "t0"]].merge(p2[["customer_id", "t0", "mu0_raw", "mu0_cal", "mu0_is_oof", "s_hat_mean"]],
                                         on=["customer_id", "t0"], how="left", validate="one_to_one")
    assert p2.mu0_is_oof.eq(df.split.eq("train")).all()
    mu0_base = np.where(df.split.eq("train"), p2.mu0_raw, p2.mu0_cal)
    S = p2.s_hat_mean.to_numpy()
    tau_all = np.zeros((len(df), 3))
    tau_all[tr_m], tau_all[vc_m] = tau_oof[chosen], tau_vc[chosen]
    cf_all, lo_all, hi_all = np.zeros_like(tau_all), np.zeros_like(tau_all), np.zeros_like(tau_all)
    cf_all[tr_m], cf_all[vc_m] = tau_oof["CausalForestDML"], tau_vc["CausalForestDML"]
    lo_vc, hi_vc = predict_tau_interval(full["CausalForestDML"], X_vc)
    lo_all[tr_m], hi_all[tr_m], lo_all[vc_m], hi_all[vc_m] = cf_lo_oof, cf_hi_oof, lo_vc, hi_vc
    V, n_clip = value_hat(tau_all, mu0_base, S)
    out = df[["customer_id", "t0", "split", "A_logged", "is_random_arm"]].copy()
    for j, a in enumerate(TREATED):
        out[f"tau_hat_{a}"], out[f"tau_lo_{a}"], out[f"tau_hi_{a}"] = tau_all[:, j], lo_all[:, j], hi_all[:, j]
        out[f"tau_cf_{a}"], out[f"V_hat_{a}"] = cf_all[:, j], V[:, j]
    out["mu0_base"] = mu0_base
    out["mu0_base_nguon"] = np.where(df.split.eq("train"), "mu0_raw_ngoai_fold", "mu0_cal")
    out["s_hat_mean"] = S
    out["tau_is_oof"] = tr_m
    out["cate_learner"] = chosen
    out.to_parquet(common.p("data/processed/pred_s4.parquet"), index=False)
    files.append("data/processed/pred_s4.parquet")
    print(f"V̂: {n_clip} cặp (dòng, hành động) có μ̂0+τ̂ ngoài [0,1] bị cắt")

    # kiểm tra V̂ không dùng đáp án: lợi nhuận gia tăng trên nhánh ngẫu nhiên khi nhắm top-k% theo V̂
    vrows = []
    vco = out[vc_m].reset_index(drop=True)
    for j, a in enumerate(TREATED):
        m = ((vco.is_random_arm == 1) & vco.A_logged.isin(["a0", a])).to_numpy()
        v, pr, t_ = vco[f"V_hat_{a}"].to_numpy()[m], vc.profit_obs.to_numpy()[m], (vco.A_logged.to_numpy()[m] == a)
        for k in [0.1, 0.2, 0.5, 1.0]:
            top = v >= np.quantile(v, 1 - k) if k < 1 else np.ones(len(v), bool)
            vrows.append({"hanh_dong": a, "top_k": k, "n": int(top.sum()), "TB_V_hat": v[top].mean(),
                          "uplift_LN_quan_sat": pr[top & t_].mean() - pr[top & ~t_].mean()})
        vrows.append({"hanh_dong": a, "top_k": "V_hat>0", "n": int((v > 0).sum()), "TB_V_hat": v[v > 0].mean(),
                      "uplift_LN_quan_sat": pr[(v > 0) & t_].mean() - pr[(v > 0) & ~t_].mean()})
    vtop = pd.DataFrame(vrows)
    vtop.to_csv(common.p(TABLES / "s4_vhat_topk_random_arm.csv"), index=False)
    files.append(str(TABLES / "s4_vhat_topk_random_arm.csv"))
    print(vtop.round(3).to_string())
    vt = pd.DataFrame(evaluate_vhat_vs_truth(vc, V[vc_m], "valid+calib") + evaluate_vhat_vs_truth(tr, V[tr_m], "train"))
    vt.to_csv(common.p(TABLES / "s4_vhat_vs_truth.csv"), index=False)
    files.append(str(TABLES / "s4_vhat_vs_truth.csv"))
    print(vt.round(4).to_string())

    joblib.dump({"pre": pre, "learner": full[chosen], "learner_name": chosen, "cf": full["CausalForestDML"],
                 "features": FEATS_C}, common.p("models/s4_cate_bundle.joblib"))
    joblib.dump({"pre": pre, "propensity": prop_model}, common.p("models/s4_propensity.joblib"))
    files += ["models/s4_cate_bundle.joblib", "models/s4_propensity.joblib"]

    # ---- 6. Tiêu chí chấp nhận
    dml_tr = ate_eval[(ate_eval.tap == "train") & ate_eval.phuong_phap.str.startswith("DML")]
    q_sel = qini[(qini.tap == "valid+calib") & qini.duoc_chon]
    acc = pd.DataFrame([
        {"tieu_chi": "CI DML-APOS (train) phủ ATE thật", "gia_tri":
            f"{int(dml_tr[dml_tr.phuong_phap.str.contains('APOS')].ci_phu_ate_that.sum())}/6",
         "dat": bool(dml_tr[dml_tr.phuong_phap.str.contains("APOS")].ci_phu_ate_that.all())},
        {"tieu_chi": "CI DML-IRM cặp (train) phủ ATE thật của tập con", "gia_tri":
            f"{int(dml_tr[dml_tr.phuong_phap.str.contains('IRM')].ci_phu_ate_that.sum())}/6",
         "dat": bool(dml_tr[dml_tr.phuong_phap.str.contains("IRM")].ci_phu_ate_that.all())},
        {"tieu_chi": f"Qini > 0 cho >= 2 hành động ({chosen}, nhánh ngẫu nhiên valid+calib)",
         "gia_tri": f"{int((q_sel.qini > 0).sum())}/3", "dat": bool((q_sel.qini > 0).sum() >= 2)},
        {"tieu_chi": f"Qini có CI 95% > 0 cho >= 2 hành động (chuẩn G3, {chosen})",
         "gia_tri": f"{int((q_sel.qini_ci_lo > 0).sum())}/3", "dat": bool((q_sel.qini_ci_lo > 0).sum() >= 2)},
        {"tieu_chi": "Placebo: CI DML chứa 0 (6 phép thử)", "gia_tri": f"{int(plc.ci_phu_0.sum())}/6",
         "dat": bool(plc.ci_phu_0.all())}])
    acc.to_csv(common.p(TABLES / "s4_acceptance.csv"), index=False)
    files.append(str(TABLES / "s4_acceptance.csv"))
    print(acc.to_string())

    common.write_manifest("s4", files, extra={"cate_learner": chosen, "selection_qini_valid": sel_score.to_dict(),
                                              "features": FEATS_C, "n_clip_prob": n_clip,
                                              "runtime_s": round(time.time() - t_start)},
                          packages=["pandas", "numpy", "pyarrow", "scikit-learn", "lightgbm", "econml", "DoubleML",
                                    "scikit-uplift", "matplotlib", "joblib"])
    print(f"XONG GĐ4 ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    from s4_causal import main as _main     # pickle dưới tên module s4_causal (nạp lại được ở GĐ7/GĐ8)
    _main()
