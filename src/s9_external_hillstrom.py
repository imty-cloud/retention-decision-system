"""GĐ9 – Kiểm định ngoài trên RCT thật (Hillstrom MineThatData E-Mail).

GHI CHÚ LÝ THUYẾT: với MỘT ràng buộc đếm duy nhất (số khách được gửi ≤ 30%), Greedy theo giá trị V̂ là nghiệm tối ưu
(ma trận ràng buộc unimodular), nên MILP = Greedy là KỲ VỌNG LÝ THUYẾT, không phải kết quả bất ngờ. Khác biệt chỉ có thể
xuất hiện khi có thêm ràng buộc ngân sách/năng lực như GĐ5–6.

KẾT QUẢ CHÍNH / PHỤ: `conversion` là kết quả CHÍNH đã đăng ký trước; `visit` là kết quả PHỤ đã khai báo từ đầu
(H["outcomes"]) và KHÔNG thay thế kết quả chính. Qini visit dùng đúng learner đã chọn bằng conversion, không chọn lại
(`python src/s9_external_hillstrom.py --visit-only`, ghi s9_qini_visit_test.csv).

MỤC ĐÍCH VÀ GIỚI HẠN (đọc trước khi trích kết quả):
  • Đây là kiểm định PHƯƠNG PHÁP (learner CATE → Qini → chính sách có ràng buộc → giá trị chính sách ước lượng
    không chệch bằng IPW) trên dữ liệu có ngẫu nhiên hóa thật. KHÔNG phải kiểm định toàn pipeline: Hillstrom
    không có dự báo xác suất nền/hiệu chỉnh, không có chi phí chiết khấu, không có Router/HITL/MLOps, không có
    trôi theo thời gian, và chỉ có 2 hành động email (+ không gửi) thay vì 4 hành động của đề tài.
  • Tỷ lệ chuyển đổi chỉ ~0,9% (≈ 580 đơn/64.000) nên Qini/uplift theo conversion rất nhiễu; CI phản ánh đúng điều đó.
  • Đơn vị tiền của `spend` là USD gốc; ta dùng "đơn vị tiền" chung và đặt phí gửi 0,10/email theo đặc tả.
  • Tập này độc lập hoàn toàn với Online Retail: không đọc/ghi data/processed, không import GĐ khác.

ĐĂNG KÝ TRƯỚC (ghi trước khi chạy):
  1. Chia 70/30 phân tầng theo hành động, seed cố định (HILLSTROM["seed"]).
  2. Propensity đã biết = 1/3 (RCT cân bằng). Mọi learner dùng propensity CỐ ĐỊNH 1/k (không ước lượng) – lớp KnownPropensity.
  3. Chọn siêu tham số và chọn learner chỉ bằng Qini trung bình (2 loại email, outcome conversion) trên 5-fold CV của TRAIN.
     Learner thắng được dùng cho chính sách, với mô hình CATE của `spend` (cùng lớp learner, cùng siêu tham số). Test chỉ để báo cáo.
  4. Giá trị chính sách = biên × uplift chi tiêu − phí gửi; ràng buộc ≤ 30% khách được gửi; mỗi khách đúng 1 hành động.
  5. Giá trị chính sách trên test: IPW với e=1/3, CI bootstrap theo dòng (khách độc lập trong RCT).
  6. Kết quả xấu (CATE không hơn ngẫu nhiên, chính sách không hơn gửi ngẫu nhiên) được giữ nguyên và nêu rõ.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pulp
from econml.dml import CausalForestDML
from econml.dr import DRLearner
from econml.metalearners import TLearner, XLearner
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklift.metrics import qini_auc_score

import common

# =============================================================================================
# KHỐI THAM SỐ HILLSTROM (toàn bộ tham số của GĐ9 nằm ở đây; không đụng configs/config.yaml)
# =============================================================================================
HILLSTROM = dict(
    raw_dir="data/raw/hillstrom",
    seed=42,
    test_size=0.30,
    propensity=1 / 3,
    arms={"a0": "No E-Mail", "a1": "Mens E-Mail", "a2": "Womens E-Mail"},
    arm_vi={"a0": "Không gửi", "a1": "Email nam", "a2": "Email nữ"},
    outcomes=("conversion", "visit", "spend"),
    primary="conversion",
    value_outcome="spend",
    margin=0.30,
    margin_sens=(0.20, 0.30, 0.40),
    email_cost=0.10,
    max_send_share=0.30,
    cv_folds=5,
    n_boot_qini=300,
    n_boot_policy=1000,
    n_random_policies=200,
    n_random_score=200,
    n_threads=4,
    milp_time_limit=60,
    milp_gap=0.001,
    decile_groups=10,
    # lưới nhỏ cho từng learner (đăng ký trước)
    grid={
        "T-learner": [{"num_leaves": 4}, {"num_leaves": 8}],
        "X-learner": [{"num_leaves": 4}, {"num_leaves": 8}],
        "DR-learner": [{"num_leaves": 4}, {"num_leaves": 8}],
        "CausalForestDML": [{"min_samples_leaf": 50}, {"min_samples_leaf": 200}],
    },
    cf_trees_cv=200,
    cf_trees_final=400,
)
H = HILLSTROM
CFG = common.load_config()
SEED = H["seed"]
TABLES, FIGS = Path(CFG["paths"]["tables"]), Path(CFG["paths"]["figures"])
ARMS = list(H["arms"])
TREATED = ARMS[1:]
CODES = [0, 1, 2]                      # econml metalearner cần nhãn hành động số: 0=a0, 1=a1, 2=a2
LEARNERS = list(H["grid"])
FEATS_NUM = ["recency", "history", "mens", "womens", "newbie"]
FEATS_CAT = ["history_segment", "zip_code", "channel"]
OUT_VI = {"conversion": "Chuyển đổi (xác suất)", "visit": "Truy cập (xác suất)", "spend": "Chi tiêu"}
NUIS = dict(n_estimators=200, learning_rate=0.05, min_child_samples=100, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=5.0, random_state=SEED, num_threads=H["n_threads"], verbose=-1)
FINAL = dict(num_leaves=4, n_estimators=150, learning_rate=0.03, min_child_samples=300, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0, random_state=SEED,
             num_threads=H["n_threads"], verbose=-1)


# ----------------------------------------------------------------------------- dữ liệu
def load_hillstrom() -> tuple[pd.DataFrame, Path]:
    """Tải (nếu chưa có) về data/raw/hillstrom/, trả về bảng gộp 3 outcome + hành động và đường dẫn file gốc."""
    from sklift.datasets import fetch_hillstrom
    raw = common.p(Path(H["raw_dir"]) / ".keep").parent
    d = fetch_hillstrom(target_col="all", data_home=str(raw), dest_subdir=".")
    df = d.data.copy()
    for c in H["outcomes"]:
        df[c] = d.target[c].to_numpy()
    inv = {v: k for k, v in H["arms"].items()}
    df["A"] = d.treatment.map(inv).to_numpy()
    assert df.A.notna().all(), "Có nhãn hành động lạ"
    files = sorted(raw.glob("*.csv.gz"))
    assert files, "Không thấy file Hillstrom đã tải"
    return df.reset_index(drop=True), files[0]


def split_train_test(df: pd.DataFrame, seed: int = SEED, test_size: float = H["test_size"]):
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=test_size, random_state=seed,
                                      stratify=df["A"].to_numpy())
    return np.sort(idx_tr), np.sort(idx_te)


def design_matrix(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    X = pd.get_dummies(df[FEATS_NUM + FEATS_CAT], columns=FEATS_CAT, dtype=float).astype(float)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


# ----------------------------------------------------------------------------- propensity đã biết
class KnownPropensity(BaseEstimator, ClassifierMixin):
    """Bộ phân loại 'giả' trả về propensity cố định 1/k (RCT cân bằng): thay mô hình propensity trong learner."""

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        k = len(self.classes_)
        return np.full((len(X), k), 1.0 / k)

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


# ----------------------------------------------------------------------------- cân bằng hiệp biến & ATE
def balance_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SMD (so với No E-Mail) cho từng cột one-hot + kiểm định toàn cục (ANOVA / chi-bình-phương)."""
    X = design_matrix(df)
    rows = []
    c = df.A.to_numpy() == "a0"
    for col in X.columns:
        for a in TREATED:
            t = df.A.to_numpy() == a
            m1, m0 = X.loc[t, col].mean(), X.loc[c, col].mean()
            sd = np.sqrt((X.loc[t, col].var() + X.loc[c, col].var()) / 2)
            rows.append(dict(bien=col, hanh_dong=a, trung_binh_gui=m1, trung_binh_doi_chung=m0,
                             SMD=(m1 - m0) / sd if sd > 0 else 0.0))
    smd = pd.DataFrame(rows)
    tests = []
    for col in FEATS_NUM + FEATS_CAT:
        if col in FEATS_CAT or df[col].nunique() <= 2:
            p = stats.chi2_contingency(pd.crosstab(df[col], df.A))[1]
            kind = "chi-bình-phương"
        else:
            p = stats.f_oneway(*[df.loc[df.A == a, col] for a in ARMS])[1]
            kind = "ANOVA"
        tests.append(dict(bien_goc=col, kiem_dinh=kind, p_value=p))
    return smd, pd.DataFrame(tests)


def ate_diff_means(df: pd.DataFrame, label: str) -> list[dict]:
    """ATE = chênh lệch trung bình (đáp án thật nhờ ngẫu nhiên hóa) + CI 95% (Welch)."""
    rows = []
    c = df.A.to_numpy() == "a0"
    for out in H["outcomes"]:
        y0 = df.loc[c, out].to_numpy(float)
        for a in TREATED:
            y1 = df.loc[df.A.to_numpy() == a, out].to_numpy(float)
            est = y1.mean() - y0.mean()
            se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
            p = stats.ttest_ind(y1, y0, equal_var=False)[1]
            rows.append(dict(tap=label, outcome=out, hanh_dong=a, n_gui=len(y1), n_doi_chung=len(y0),
                             trung_binh_doi_chung=y0.mean(), ATE=est, se=se, ci_lo=est - 1.96 * se,
                             ci_hi=est + 1.96 * se, p_value=p))
    return rows


# ----------------------------------------------------------------------------- learner CATE
def make_learner(name: str, hp: dict, final: bool):
    nu = dict(NUIS, num_leaves=hp.get("num_leaves", 8))
    if name == "T-learner":
        return TLearner(models=LGBMRegressor(**nu), categories=CODES)
    if name == "X-learner":
        return XLearner(models=LGBMRegressor(**nu), propensity_model=KnownPropensity(), categories=CODES)
    if name == "DR-learner":
        return DRLearner(model_propensity=KnownPropensity(), model_regression=LGBMRegressor(**nu),
                         model_final=LGBMRegressor(**FINAL), categories=CODES, cv=5, min_propensity=0.01,
                         random_state=SEED)
    if name == "CausalForestDML":
        return CausalForestDML(model_y=LGBMRegressor(**dict(NUIS, num_leaves=8)), model_t=KnownPropensity(),
                               discrete_treatment=True, categories=CODES, cv=5,
                               n_estimators=H["cf_trees_final"] if final else H["cf_trees_cv"],
                               min_samples_leaf=hp["min_samples_leaf"], max_samples=0.45,
                               random_state=SEED, n_jobs=H["n_threads"])
    raise ValueError(name)


def fit_predict_tau(name: str, hp: dict, X_fit, y_fit, A_fit, X_new, final: bool) -> np.ndarray:
    """Khớp learner trên (X_fit, y_fit, A_fit) rồi trả τ̂ (n_new × 2) cho [email nam, email nữ] so với không gửi."""
    L = make_learner(name, hp, final)
    code = lambda A: pd.Series(A).map({a: k for k, a in enumerate(ARMS)}).to_numpy()
    L.fit(y_fit, code(A_fit), X=X_fit)
    return np.column_stack([np.ravel(L.effect(X_new, T0=0, T1=k)) for k in (1, 2)])


def _is_binary(y) -> bool:
    return set(np.unique(y)) <= {0, 1}


def qini_curve_any(y, score, t):
    """Đường Qini tích lũy Q(k) = Σy_T − Σy_C·(n_T/n_C) khi nhắm k khách điểm cao nhất; dùng được cho y liên tục (spend)."""
    o = np.argsort(-score, kind="stable")
    y, t = np.asarray(y, float)[o], np.asarray(t)[o]
    nt, nc = np.cumsum(t), np.cumsum(1 - t)
    q = np.cumsum(y * t) - np.cumsum(y * (1 - t)) * nt / np.maximum(nc, 1)
    return np.arange(1, len(y) + 1), q


def qini_coef(y, score, t) -> float:
    """Hệ số Qini. y nhị phân → scikit-uplift (chuẩn hóa như GĐ4). y liên tục → diện tích giữa đường Qini và đường
    ngẫu nhiên, chia cho n (đơn vị: chi tiêu/khách; KHÔNG chuẩn hóa nên không so trực tiếp với Qini nhị phân)."""
    if _is_binary(y):
        return float(qini_auc_score(y, score, t))
    k, q = qini_curve_any(y, score, t)
    f = k / len(k)
    gap = q - f * q[-1]
    return float(np.trapezoid(gap, f) / len(y))


def qini_value(y, A, tau_col, arm) -> float:
    m = np.isin(A, ["a0", arm])
    t = (A[m] == arm).astype(int)
    return qini_coef(y[m], tau_col[m], t)


def cv_qini(name, hp, X, y, A) -> list[float]:
    """Qini trung bình (2 loại email) trên từng fold của CV 5-fold phân tầng theo hành động."""
    skf = StratifiedKFold(H["cv_folds"], shuffle=True, random_state=SEED)
    out = []
    for tr, va in skf.split(X, A):
        tau = fit_predict_tau(name, hp, X.iloc[tr], y[tr], A[tr], X.iloc[va], final=False)
        out.append(float(np.mean([qini_value(y[va], A[va], tau[:, j], a) for j, a in enumerate(TREATED)])))
    return out


def bootstrap_qini(y, A, tau, B, seed) -> list[dict]:
    """Qini + CI bootstrap 95% theo dòng, cho từng loại email. Cùng hạt giống cho mọi learner (ghép cặp)."""
    rows = []
    for j, arm in enumerate(TREATED):
        m = np.isin(A, ["a0", arm])
        yy, tt, ss = y[m], (A[m] == arm).astype(int), tau[m, j]
        rng = np.random.default_rng(seed + j)
        qs = []
        for _ in range(B):
            idx = rng.integers(0, len(yy), len(yy))
            if tt[idx].min() == tt[idx].max():
                continue
            qs.append(qini_coef(yy[idx], ss[idx], tt[idx]))
        rows.append(dict(hanh_dong=arm, n=int(m.sum()), qini=float(qini_coef(yy, ss, tt)),
                         ci_lo=float(np.percentile(qs, 2.5)), ci_hi=float(np.percentile(qs, 97.5)),
                         ty_le_boot_qini_duong=float(np.mean(np.array(qs) > 0))))
    return rows


def random_score_qini(y, A, B, seed) -> dict:
    """Qini kỳ vọng của điểm ngẫu nhiên (mốc so sánh) và khoảng 95% của phân phối đó."""
    out = {}
    for j, arm in enumerate(TREATED):
        m = np.isin(A, ["a0", arm])
        yy, tt = y[m], (A[m] == arm).astype(int)
        rng = np.random.default_rng(seed + 100 + j)
        qs = [qini_coef(yy, rng.random(len(yy)), tt) for _ in range(B)]
        out[arm] = (float(np.mean(qs)), float(np.percentile(qs, 2.5)), float(np.percentile(qs, 97.5)))
    return out


# ----------------------------------------------------------------------------- chính sách
def policy_greedy(V: np.ndarray, cap: int) -> np.ndarray:
    """Greedy: hành động tốt nhất mỗi khách (nếu V>0), rồi lấy `cap` khách có V lớn nhất. Trả mã hành động 0/1/2."""
    best = V.argmax(1)
    bv = V.max(1)
    order = np.argsort(-bv, kind="stable")[:cap]
    pol = np.zeros(len(V), dtype=int)
    keep = order[bv[order] > 0]
    pol[keep] = best[keep] + 1
    return pol


def policy_milp(V: np.ndarray, cap: int, time_limit: int = H["milp_time_limit"], gap: float = H["milp_gap"]):
    """MILP: x[i,a] nhị phân (a=0 là không gửi, giá trị 0); mỗi khách đúng 1 hành động; Σ gửi ≤ cap. PuLP + HiGHS."""
    n, k = V.shape
    prob = pulp.LpProblem("hillstrom_policy", pulp.LpMaximize)
    x = {(i, a): prob.add_variable(f"x_{i}_{a}", cat="Binary") for i in range(n) for a in range(k + 1)}
    prob += pulp.lpSum(float(V[i, a - 1]) * x[i, a] for i in range(n) for a in range(1, k + 1))
    for i in range(n):
        prob += pulp.lpSum(x[i, a] for a in range(k + 1)) == 1
    prob += pulp.lpSum(x[i, a] for i in range(n) for a in range(1, k + 1)) <= cap
    t0 = time.perf_counter()
    res = prob.solve(pulp.HiGHS(msg=False, timeLimit=time_limit, gapRel=gap, threads=H["n_threads"]))
    dt = time.perf_counter() - t0
    pol = np.zeros(n, dtype=int)
    for i in range(n):
        for a in range(1, k + 1):
            if (x[i, a].value() or 0) > 0.5:
                pol[i] = a
    return pol, dict(trang_thai=res.status.name, thoi_gian_giai_s=dt, gap_tuong_doi=res.gap_rel)


def value_vec(V: np.ndarray, pol: np.ndarray) -> np.ndarray:
    """V̂ (mô hình) của từng khách dưới chính sách pol (0 nếu không gửi)."""
    out = np.zeros(len(pol))
    m = pol > 0
    out[m] = V[np.where(m)[0], pol[m] - 1]
    return out


def ipw_contrib(pol: np.ndarray, A_code: np.ndarray, spend: np.ndarray, margin: float, cost: float) -> np.ndarray:
    """Đóng góp IPW từng khách: 1{A=π}/e × (biên·chi tiêu − phí gửi nếu π≠không gửi). Không chệch trong RCT với e=1/3."""
    e = H["propensity"]
    return (A_code == pol) / e * (margin * spend - cost * (pol > 0))


def boot_means(contribs: np.ndarray, B: int, seed: int) -> np.ndarray:
    """contribs: n × P. Trả (B × P) trung bình bootstrap; cùng chỉ số cho mọi chính sách (ghép cặp)."""
    n = contribs.shape[0]
    rng = np.random.default_rng(seed)
    return np.stack([contribs[rng.integers(0, n, n)].mean(0) for _ in range(B)])


def random_policy_contrib(rnd: np.ndarray, A_code, spend, margin, cost) -> np.ndarray:
    return np.mean([ipw_contrib(rnd[:, d], A_code, spend, margin, cost) for d in range(rnd.shape[1])], 0)


# ----------------------------------------------------------------------------- đồ họa
def savefig(fig, name: str, files: list[str]) -> None:
    path = FIGS / name
    fig.tight_layout()
    fig.savefig(common.p(path), dpi=160)
    plt.close(fig)
    files.append(str(path))


def save_table(df: pd.DataFrame, name: str, files: list[str]) -> None:
    path = TABLES / name
    df.to_csv(common.p(path), index=False)
    files.append(str(path))


# ----------------------------------------------------------------------------- main
def main() -> None:
    t_start = time.time()
    np.random.seed(SEED)
    files: list[str] = []
    df, raw_file = load_hillstrom()
    sha = common.sha256_file(raw_file)
    save_table(pd.DataFrame([dict(file=str(raw_file.relative_to(common.ROOT)), bytes=raw_file.stat().st_size,
                                  sha256=sha, so_dong=len(df), nguon="sklift.datasets.fetch_hillstrom")]),
               "s9_data_sha256.csv", files)
    print(f"Hillstrom: {len(df)} dòng · SHA-256 {sha[:16]}…")

    # ---- 1. chia 70/30 phân tầng + kiểm tra cân bằng
    i_tr, i_te = split_train_test(df)
    tr, te = df.iloc[i_tr].reset_index(drop=True), df.iloc[i_te].reset_index(drop=True)
    share = pd.DataFrame([dict(tap=nm, n=len(d), **{f"ty_le_{a}": (d.A == a).mean() for a in ARMS})
                          for nm, d in [("toàn bộ", df), ("train", tr), ("test", te)]])
    save_table(share, "s9_split_summary.csv", files)
    smd, tests = balance_table(df)
    smd_tr, _ = balance_table(tr)
    smd["tap"], smd_tr["tap"] = "toàn bộ", "train"
    save_table(pd.concat([smd, smd_tr]), "s9_balance_smd.csv", files)
    save_table(tests, "s9_balance_tests.csv", files)
    print(share.round(4).to_string())
    print(f"max |SMD| toàn bộ = {smd.SMD.abs().max():.4f} (ngưỡng thường dùng 0,1) · "
          f"p-value nhỏ nhất kiểm định cân bằng = {tests.p_value.min():.3f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    for a, mk in zip(TREATED, ["o", "s"]):
        s = smd[smd.hanh_dong == a]
        ax.scatter(s.SMD, range(len(s)), marker=mk, label=f"{H['arm_vi'][a]} so với không gửi")
    ax.set_yticks(range(len(smd[smd.hanh_dong == "a1"])))
    ax.set_yticklabels(smd[smd.hanh_dong == "a1"].bien, fontsize=8)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(-0.1, color="gray", ls=":")
    ax.axvline(0.1, color="gray", ls=":")
    ax.set_xlabel("Chênh lệch trung bình chuẩn hóa (SMD)")
    ax.set_title("Hillstrom: cân bằng hiệp biến sau ngẫu nhiên hóa (đường chấm: ±0,1)")
    ax.legend(fontsize=8)
    savefig(fig, "fig_s9_balance.png", files)

    # ---- 2. ATE thật (chênh lệch trung bình)
    ate = pd.DataFrame(ate_diff_means(df, "toàn bộ") + ate_diff_means(tr, "train") + ate_diff_means(te, "test"))
    save_table(ate, "s9_ate_rct.csv", files)
    print(ate[ate.tap == "toàn bộ"].round(4).to_string())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, out in zip(axes, H["outcomes"]):
        sub = ate[ate.outcome == out]
        labels, k = [], 0
        for tp, c in [("toàn bộ", "tab:blue"), ("train", "tab:green"), ("test", "tab:orange")]:
            for a in TREATED:
                r = sub[(sub.tap == tp) & (sub.hanh_dong == a)].iloc[0]
                ax.errorbar(r.ATE, k, xerr=[[r.ATE - r.ci_lo], [r.ci_hi - r.ATE]], fmt="o", color=c, capsize=3)
                labels.append(f"{tp} · {H['arm_vi'][a]}")
                k += 1
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(range(k))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"ATE – {OUT_VI[out]}")
        ax.set_xlabel("chênh lệch trung bình so với không gửi (CI 95%)")
    savefig(fig, "fig_s9_ate_forest.png", files)

    # ---- 3. CATE: CV trên train → chọn learner (quy tắc đăng ký trước)
    X_tr = design_matrix(tr)
    X_te = design_matrix(te, list(X_tr.columns))
    A_tr, A_te = tr.A.to_numpy(), te.A.to_numpy()
    y_cv = tr[H["primary"]].to_numpy(float)
    grid_rows = []
    for name in LEARNERS:
        for hp in H["grid"][name]:
            t1 = time.time()
            sc = cv_qini(name, hp, X_tr, y_cv, A_tr)
            grid_rows.append(dict(learner=name, sieu_tham_so=str(hp), qini_cv_tb=np.mean(sc),
                                  qini_cv_sd=np.std(sc, ddof=1), **{f"fold{k + 1}": v for k, v in enumerate(sc)},
                                  giay=time.time() - t1))
            print(f"  CV {name} {hp}: Qini={np.mean(sc):+.4f} ± {np.std(sc, ddof=1):.4f} ({time.time() - t1:.0f}s)", flush=True)
    grid = pd.DataFrame(grid_rows)
    grid["hp_obj"] = [hp for n in LEARNERS for hp in H["grid"][n]]
    best = grid.loc[grid.groupby("learner").qini_cv_tb.idxmax()].set_index("learner")
    chosen = best.qini_cv_tb.idxmax()
    grid["tot_nhat_trong_learner"] = grid.apply(lambda r: r.sieu_tham_so == best.loc[r.learner, "sieu_tham_so"], axis=1)
    grid["learner_thang"] = grid.learner == chosen
    save_table(grid.drop(columns="hp_obj"), "s9_cv_grid.csv", files)
    print(f"Learner thắng theo quy tắc đăng ký trước (Qini CV trên train): {chosen}", flush=True)

    # ---- 4. khớp lần cuối trên toàn train, chấm test (conversion và spend)
    taus: dict[tuple[str, str], np.ndarray] = {}
    for name in LEARNERS:
        hp = best.loc[name, "hp_obj"]
        for out in (H["primary"], H["value_outcome"]):
            t1 = time.time()
            taus[(name, out)] = fit_predict_tau(name, hp, X_tr, tr[out].to_numpy(float), A_tr, X_te, final=True)
            print(f"  fit cuối {name} / {out}: {time.time() - t1:.0f}s", flush=True)

    qrows = []
    for out in (H["primary"], H["value_outcome"]):
        y_te = te[out].to_numpy(float)
        ref = random_score_qini(y_te, A_te, H["n_random_score"], SEED)
        for name in LEARNERS:
            for r in bootstrap_qini(y_te, A_te, taus[(name, out)], H["n_boot_qini"], SEED):
                a = r["hanh_dong"]
                qrows.append(dict(outcome=out, learner=name, **r, qini_ngau_nhien_tb=ref[a][0],
                                  ngau_nhien_lo=ref[a][1], ngau_nhien_hi=ref[a][2], la_learner_thang=name == chosen,
                                  ci_khong_chua_0=bool(r["ci_lo"] > 0 or r["ci_hi"] < 0)))
    qini = pd.DataFrame(qrows)
    save_table(qini, "s9_qini_test.csv", files)
    print(qini.round(4).to_string())

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    cols = dict(zip(LEARNERS, ["tab:blue", "tab:orange", "tab:green", "tab:red"]))
    for r_i, out in enumerate((H["primary"], H["value_outcome"])):
        y_te = te[out].to_numpy(float)
        for c_j, arm in enumerate(TREATED):
            ax = axes[r_i, c_j]
            m = np.isin(A_te, ["a0", arm])
            tt = (A_te[m] == arm).astype(int)
            for name in LEARNERS:
                xs, ys = qini_curve_any(y_te[m], taus[(name, out)][m, c_j], tt)
                ax.plot(xs, ys, color=cols[name], lw=2 if name == chosen else 1,
                        label=name + (" (thắng CV)" if name == chosen else ""))
            xs, ys = qini_curve_any(y_te[m], np.random.default_rng(SEED).random(m.sum()), tt)
            ax.plot([0, xs[-1]], [0, ys[-1]], "k--", lw=1, label="ngẫu nhiên")
            ax.set_title(f"{H['arm_vi'][arm]} · {OUT_VI[out]}")
            ax.set_xlabel("số khách được nhắm (test)")
            ax.set_ylabel("uplift tích lũy")
            ax.legend(fontsize=7)
    fig.suptitle("Hillstrom (RCT thật): đường Qini trên tập test")
    savefig(fig, "fig_s9_qini_curves.png", files)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, out in zip(axes, (H["primary"], H["value_outcome"])):
        sub = qini[qini.outcome == out]
        w = 0.2
        for k, name in enumerate(LEARNERS):
            for j, arm in enumerate(TREATED):
                r = sub[(sub.learner == name) & (sub.hanh_dong == arm)].iloc[0]
                ax.errorbar(j + (k - 1.5) * w, r.qini, yerr=[[r.qini - r.ci_lo], [r.ci_hi - r.qini]], fmt="o",
                            color=cols[name], capsize=3, label=name if j == 0 else None)
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(range(2))
        ax.set_xticklabels([H["arm_vi"][a] for a in TREATED])
        ax.set_ylabel("hệ số Qini (test)")
        ax.set_title(f"Qini ± CI bootstrap 95% – {OUT_VI[out]}")
        ax.legend(fontsize=8)
    savefig(fig, "fig_s9_qini_ci.png", files)

    # độ hiệu chỉnh CATE theo decile τ̂ (learner thắng)
    dec_rows = []
    for out in (H["primary"], H["value_outcome"]):
        y_te = te[out].to_numpy(float)
        tau = taus[(chosen, out)]
        for j, arm in enumerate(TREATED):
            g = pd.qcut(pd.Series(tau[:, j]).rank(method="first"), H["decile_groups"], labels=False)
            for q in range(H["decile_groups"]):
                mq = (g == q).to_numpy()
                y1, y0 = y_te[mq & (A_te == arm)], y_te[mq & (A_te == "a0")]
                se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
                dec_rows.append(dict(outcome=out, hanh_dong=arm, decile=q + 1, tau_hat_tb=tau[mq, j].mean(),
                                     uplift_quan_sat=y1.mean() - y0.mean(), se=se, n=int(mq.sum())))
    dec = pd.DataFrame(dec_rows)
    save_table(dec, "s9_cate_deciles_test.csv", files)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for r_i, out in enumerate((H["primary"], H["value_outcome"])):
        for c_j, arm in enumerate(TREATED):
            ax = axes[r_i, c_j]
            s = dec[(dec.outcome == out) & (dec.hanh_dong == arm)]
            ax.errorbar(s.tau_hat_tb, s.uplift_quan_sat, yerr=1.96 * s.se, fmt="o", capsize=3)
            lim = [min(s.tau_hat_tb.min(), s.uplift_quan_sat.min()), max(s.tau_hat_tb.max(), s.uplift_quan_sat.max())]
            ax.plot(lim, lim, "k--", lw=1, label="y = x")
            ax.set_xlabel(f"τ̂ trung bình theo decile ({chosen})")
            ax.set_ylabel("uplift quan sát (RCT, test)")
            ax.set_title(f"{H['arm_vi'][arm]} · {OUT_VI[out]}")
            ax.legend(fontsize=8)
    fig.suptitle("Độ hiệu chỉnh CATE trên test: τ̂ so với uplift quan sát theo decile")
    savefig(fig, "fig_s9_cate_deciles.png", files)

    # ---- 5. chính sách có ràng buộc
    n_te = len(te)
    cap = int(np.floor(H["max_send_share"] * n_te))
    A_code = pd.Series(A_te).map({a: k for k, a in enumerate(ARMS)}).to_numpy()
    spend_te = te[H["value_outcome"]].to_numpy(float)
    tau_sp = taus[(chosen, H["value_outcome"])]
    ate_tr_spend = {a: ate[(ate.tap == "train") & (ate.outcome == "spend") & (ate.hanh_dong == a)].ATE.iloc[0]
                    for a in TREATED}
    best_single = max(TREATED, key=lambda a: ate_tr_spend[a])            # chọn trên train, không nhìn test
    rng = np.random.default_rng(SEED)
    rnd = np.zeros((n_te, H["n_random_policies"]), dtype=int)
    for d in range(H["n_random_policies"]):
        sel = rng.choice(n_te, cap, replace=False)
        rnd[sel, d] = rng.integers(1, 3, cap)

    V = H["margin"] * tau_sp - H["email_cost"]
    pg = policy_greedy(V, cap)
    pm, info = policy_milp(V, cap)
    pm_val, pg_val = value_vec(V, pm).sum(), value_vec(V, pg).sum()
    print(f"MILP {info['trang_thai']} {info['thoi_gian_giai_s']:.1f}s · V̂ MILP={pm_val:.2f} Greedy={pg_val:.2f} · "
          f"gửi MILP={int((pm > 0).sum())}/{cap}", flush=True)

    pol_rows, contribs, names = [], [], []

    def add(name, pol, model_total, note=""):
        if pol.ndim == 1:
            c = ipw_contrib(pol, A_code, spend_te, H["margin"], H["email_cost"])
        else:
            c = random_policy_contrib(pol, A_code, spend_te, H["margin"], H["email_cost"])
        contribs.append(c)
        names.append(name)
        share = float((pol > 0).mean())
        pol_rows.append(dict(chinh_sach=name, ty_le_gui=share, n_gui_tb=share * n_te,
                             gia_tri_mo_hinh_tb=(model_total / n_te if model_total is not None else np.nan),
                             ghi_chu=note))

    add("MILP (learner thắng CV)", pm, pm_val, f"HiGHS {info['trang_thai']}, {info['thoi_gian_giai_s']:.1f}s")
    add("Greedy (learner thắng CV)", pg, pg_val, "xếp hạng theo V̂ lớn nhất")
    add("Gửi ngẫu nhiên 30%", rnd, None, f"trung bình {H['n_random_policies']} lần bốc; email nam/nữ ngẫu nhiên")
    add(f"Gửi hết ({H['arm_vi'][best_single]}, chọn theo ATE train)", np.full(n_te, ARMS.index(best_single)), None,
        "KHÔNG tuân thủ trần 30% – chỉ là mốc tham chiếu")
    add("Không gửi", np.zeros(n_te, dtype=int), 0.0, "")
    for name in LEARNERS:                                               # learner khác – đối chiếu (Greedy)
        if name == chosen:
            continue
        Vn = H["margin"] * taus[(name, H["value_outcome"])] - H["email_cost"]
        pn = policy_greedy(Vn, cap)
        add(f"Greedy ({name}, không phải learner thắng)", pn, value_vec(Vn, pn).sum(),
            "đối chiếu, không dùng làm kết luận chính")

    C = np.column_stack(contribs)
    bm = boot_means(C, H["n_boot_policy"], SEED)
    k_none, k_rand = names.index("Không gửi"), names.index("Gửi ngẫu nhiên 30%")
    pol_df = pd.DataFrame(pol_rows)
    pol_df["gia_tri_ipw_tb"] = C.mean(0)
    pol_df["ci_lo"], pol_df["ci_hi"] = np.percentile(bm, 2.5, axis=0), np.percentile(bm, 97.5, axis=0)
    d_none, d_rand = bm - bm[:, [k_none]], bm - bm[:, [k_rand]]
    pol_df["tang_them_so_voi_khong_gui"] = C.mean(0) - C[:, k_none].mean()
    pol_df["tang_them_ci_lo"], pol_df["tang_them_ci_hi"] = np.percentile(d_none, 2.5, axis=0), np.percentile(d_none, 97.5, axis=0)
    pol_df["hon_gui_ngau_nhien"] = C.mean(0) - C[:, k_rand].mean()
    pol_df["hon_ngau_nhien_ci_lo"], pol_df["hon_ngau_nhien_ci_hi"] = np.percentile(d_rand, 2.5, axis=0), np.percentile(d_rand, 97.5, axis=0)
    pol_df["learner_thang"] = chosen
    save_table(pol_df, "s9_policy_value_test.csv", files)
    print(pol_df.drop(columns="ghi_chu").round(4).to_string())

    # độ nhạy theo biên lợi nhuận
    sens = []
    zero = np.zeros(n_te, dtype=int)
    for mg in H["margin_sens"]:
        Vm = mg * tau_sp - H["email_cost"]
        pgm, (pmm, im) = policy_greedy(Vm, cap), policy_milp(Vm, cap)
        pols = {"MILP": pmm, "Greedy": pgm, "Không gửi": zero}
        cs = [ipw_contrib(p_, A_code, spend_te, mg, H["email_cost"]) for p_ in pols.values()]
        cs.append(random_policy_contrib(rnd, A_code, spend_te, mg, H["email_cost"]))
        cs = np.column_stack(cs)
        b2 = boot_means(cs, H["n_boot_policy"], SEED + 7)
        dd = b2 - b2[:, [3]]
        for k, nm in enumerate(list(pols) + ["Gửi ngẫu nhiên 30%"]):
            sens.append(dict(bien=mg, chinh_sach=nm, gia_tri_ipw_tb=cs[:, k].mean(), ci_lo=np.percentile(b2[:, k], 2.5),
                             ci_hi=np.percentile(b2[:, k], 97.5), hon_ngau_nhien=cs[:, k].mean() - cs[:, 3].mean(),
                             hon_ngau_nhien_ci_lo=np.percentile(dd[:, k], 2.5), hon_ngau_nhien_ci_hi=np.percentile(dd[:, k], 97.5),
                             ty_le_gui=float((list(pols.values())[k] > 0).mean()) if k < 3 else H["max_send_share"],
                             milp_trang_thai=im["trang_thai"]))
    sens = pd.DataFrame(sens)
    save_table(sens, "s9_policy_margin_sensitivity.csv", files)

    same = bool(np.isclose(pm_val, pg_val, rtol=1e-6))
    save_table(pd.DataFrame([dict(gia_tri_mo_hinh_milp=pm_val, gia_tri_mo_hinh_greedy=pg_val, tuong_duong=same,
                                  n_gui_milp=int((pm > 0).sum()), n_gui_greedy=int((pg > 0).sum()), tran_gui=cap,
                                  trang_thai=info["trang_thai"], thoi_gian_milp_s=info["thoi_gian_giai_s"],
                                  chu_thich="Chỉ có ràng buộc lực lượng (ma trận unimodular) nên MILP và Greedy trùng nghiệm; "
                                            "khác biệt chỉ xuất hiện khi có thêm ràng buộc ngân sách/năng lực như GĐ5–6.")]),
               "s9_milp_vs_greedy.csv", files)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    show = [i for i, nm in enumerate(names) if "không phải learner thắng" not in nm]
    ys = [pol_df.gia_tri_ipw_tb[i] for i in show]
    ax.barh(range(len(show)), ys, color="tab:blue", alpha=0.7)
    ax.errorbar(ys, range(len(show)), xerr=[[pol_df.gia_tri_ipw_tb[i] - pol_df.ci_lo[i] for i in show],
                                           [pol_df.ci_hi[i] - pol_df.gia_tri_ipw_tb[i] for i in show]], fmt="none",
                ecolor="k", capsize=3)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels([names[i] for i in show], fontsize=8)
    ax.set_xlabel("lợi nhuận kỳ vọng trên mỗi khách (IPW, e=1/3; CI bootstrap 95%)")
    ax.set_title("Hillstrom (RCT thật): giá trị chính sách trên tập test")
    savefig(fig, "fig_s9_policy_value.png", files)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for nm, c in [("MILP", "tab:blue"), ("Greedy", "tab:green"), ("Gửi ngẫu nhiên 30%", "tab:gray"), ("Không gửi", "k")]:
        s = sens[sens.chinh_sach == nm]
        ax.errorbar(s.bien, s.gia_tri_ipw_tb, yerr=[s.gia_tri_ipw_tb - s.ci_lo, s.ci_hi - s.gia_tri_ipw_tb],
                    marker="o", capsize=3, label=nm, color=c)
    ax.set_xlabel("biên lợi nhuận giả định")
    ax.set_ylabel("lợi nhuận trên mỗi khách (IPW)")
    ax.set_title("Độ nhạy theo biên lợi nhuận (phí gửi cố định 0,10)")
    ax.legend(fontsize=8)
    savefig(fig, "fig_s9_margin_sensitivity.png", files)

    # ---- 6. tổng kết trung thực + manifest
    mr = pol_df.iloc[0]
    q_ch = qini[(qini.learner == chosen)]
    verdict = pd.DataFrame([
        dict(muc="Learner thắng (Qini CV, train)", gia_tri=chosen),
        dict(muc="Qini conversion/test: số email có CI hoàn toàn > 0",
             gia_tri=f"{int(((q_ch.outcome == 'conversion') & (q_ch.ci_lo > 0)).sum())}/2"),
        dict(muc="Qini chi tiêu/test: số email có CI hoàn toàn > 0",
             gia_tri=f"{int(((q_ch.outcome == 'spend') & (q_ch.ci_lo > 0)).sum())}/2"),
        dict(muc="MILP hơn gửi ngẫu nhiên 30% (CI 95% hoàn toàn > 0)?",
             gia_tri="Có" if mr.hon_ngau_nhien_ci_lo > 0 else "Không (CI chứa 0 hoặc < 0)"),
        dict(muc="MILP hơn không gửi (CI 95% hoàn toàn > 0)?",
             gia_tri="Có" if mr.tang_them_ci_lo > 0 else "Không (CI chứa 0 hoặc < 0)"),
        dict(muc="Giới hạn", gia_tri="Kiểm định phương pháp trên RCT thật, không phải toàn pipeline Online Retail"),
    ])
    save_table(verdict, "s9_acceptance.csv", files)
    print(verdict.to_string())
    man = common.write_manifest("s9", files + [str(raw_file.relative_to(common.ROOT))],
                                extra=dict(hillstrom=H, hillstrom_sha256=sha, learner_thang=chosen,
                                           thoi_gian_chay_s=round(time.time() - t_start, 1)),
                                packages=["pandas", "numpy", "scikit-learn", "econml", "lightgbm", "scikit-uplift",
                                          "PuLP", "highspy", "matplotlib", "scipy"])
    print(f"Xong GĐ9 trong {time.time() - t_start:.0f}s · manifest {man}")


def main_visit_only() -> None:
    """Kết quả PHỤ: Qini/AUUC cho `visit` bằng learner đã chọn ở main() (đọc từ s9_cv_grid.csv, không chọn lại)."""
    import ast
    import json
    from sklift.metrics import uplift_auc_score
    grid = pd.read_csv(TABLES / "s9_cv_grid.csv")
    row = grid[grid.learner_thang & grid.tot_nhat_trong_learner].iloc[0]
    name, hp = row.learner, ast.literal_eval(row.sieu_tham_so)
    df, raw_file = load_hillstrom()
    assert common.sha256_file(raw_file) == pd.read_csv(TABLES / "s9_data_sha256.csv").sha256.iloc[0]
    i_tr, i_te = split_train_test(df)
    tr, te = df.iloc[i_tr].reset_index(drop=True), df.iloc[i_te].reset_index(drop=True)
    X_tr = design_matrix(tr)
    X_te = design_matrix(te, list(X_tr.columns))
    A_te, y = te.A.to_numpy(), te["visit"].to_numpy(float)
    tau = fit_predict_tau(name, hp, X_tr, tr["visit"].to_numpy(float), tr.A.to_numpy(), X_te, final=True)
    ref = random_score_qini(y, A_te, H["n_random_score"], SEED)
    rows = []
    for j, arm in enumerate(TREATED):
        m = np.isin(A_te, ["a0", arm])
        yy, tt, ss = y[m], (A_te[m] == arm).astype(int), tau[m, j]
        rng = np.random.default_rng(SEED + j)
        qs, us = [], []
        for _ in range(H["n_boot_qini"]):
            idx = rng.integers(0, len(yy), len(yy))
            if tt[idx].min() == tt[idx].max():
                continue
            qs.append(qini_auc_score(yy[idx], ss[idx], tt[idx]))
            us.append(uplift_auc_score(yy[idx], ss[idx], tt[idx]))
        rows.append(dict(
            vai_tro="PHỤ (đã khai báo từ đầu; conversion mới là kết quả CHÍNH đăng ký trước, visit không thay thế)",
            outcome="visit", learner=name, sieu_tham_so=str(hp), hanh_dong=arm, n=int(m.sum()),
            qini=float(qini_auc_score(yy, ss, tt)), qini_ci_lo=float(np.percentile(qs, 2.5)),
            qini_ci_hi=float(np.percentile(qs, 97.5)), auuc=float(uplift_auc_score(yy, ss, tt)),
            auuc_ci_lo=float(np.percentile(us, 2.5)), auuc_ci_hi=float(np.percentile(us, 97.5)),
            qini_ngau_nhien_tb=ref[arm][0], ngau_nhien_lo=ref[arm][1], ngau_nhien_hi=ref[arm][2],
            ty_le_boot_qini_duong=float(np.mean(np.array(qs) > 0)),
            ghi_chu="learner lấy từ s9_cv_grid.csv (chọn theo Qini conversion trên CV train), không chọn lại"))
    out = pd.DataFrame(rows)
    files: list[str] = []
    save_table(out, "s9_qini_visit_test.csv", files)
    print(out.drop(columns=["vai_tro", "ghi_chu"]).round(4).to_string())
    old = json.loads((common.ROOT / "outputs/manifests/s9_manifest.json").read_text(encoding="utf-8"))
    keep = [f["file"] for f in old["files"] if f["file"] != files[0]]
    common.write_manifest("s9", keep + files, extra=old["extra"] | {"ghi_chu_visit": "bổ sung s9_qini_visit_test.csv (kết quả phụ)"},
                          packages=["pandas", "numpy", "scikit-learn", "econml", "lightgbm", "scikit-uplift",
                                    "PuLP", "highspy", "matplotlib", "scipy"])


if __name__ == "__main__":
    main_visit_only() if "--visit-only" in sys.argv else main()
