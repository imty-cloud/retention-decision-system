"""GĐ2 – Dự báo nền μ0(x), hiệu chỉnh xác suất, giá trị giỏ Ŝ với khoảng split-conformal (Chương 2).

Cột trong data/processed/pred_s2.parquet và nơi dùng:
  mu0_raw      xác suất mua lại chưa hiệu chỉnh (chẩn đoán / so sánh trước-sau).
  mu0_cal      xác suất đã hiệu chỉnh – DÙNG cho V̂, chi phí chiết khấu (μ̂0+τ̂), ngân sách ở GĐ4–GĐ8.
  mu0_is_oof   True nếu dòng thuộc train và mu0_* là dự báo NGOÀI FOLD (GroupKFold theo khách); các split khác = False
               (mô hình chưa từng thấy các dòng đó). GĐ4/GĐ5 dùng μ̂0 cho train phải dùng cột này.
  s_hat        Ŝ trung vị = exp(dự báo log) – chỉ để vẽ/chẩn đoán và làm tâm khoảng conformal.
  s_hat_mean   Ŝ kỳ vọng = s_hat × hệ số smearing Duan (khớp trên calib) – DÙNG trong công thức V (m·Ŝ·τ̂, d·Ŝ).
  s_lo, s_hi   khoảng split-conformal 90% (thang log đối xứng quanh s_hat) – dùng cho độ bất định / guardrail GĐ7.
  Lưu ý: s_* của dòng train là dự báo trong mẫu (mô hình Ŝ không được làm OOF).

Luật cứng: KHÔNG đọc file đáp án (luật cứng #1). μ0 học trên train ∧ A_logged=='a0'; chọn mô hình trên valid;
bộ hiệu chỉnh/conformal chỉ khớp trên calib.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import itertools

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import common
from s1_data import FEATURES_CAT, FEATURES_NUM

EPS = 1e-6
CFG = common.load_config()
SEED = CFG["project"]["seed"]
FEATS = FEATURES_NUM + FEATURES_CAT


# ----------------------------------------------------------------------------- mô hình
def _onehot() -> OneHotEncoder:
    # country: giữ top-10 nước, còn lại gộp vào nhóm "infrequent" (max_categories = 10 + 1)
    return OneHotEncoder(max_categories=11, handle_unknown="infrequent_if_exist", sparse_output=False)


def make_logit(C: float, num: list[str] | None = None) -> Pipeline:
    num = num or FEATURES_NUM
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), num),
        ("cat", _onehot(), FEATURES_CAT)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=C, max_iter=2000))])


def make_lgbm(num_leaves: int, n_estimators: int, task: str = "clf", num: list[str] | None = None) -> Pipeline:
    num = num or FEATURES_NUM
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), num),
        ("cat", _onehot(), FEATURES_CAT)])
    kw = dict(num_leaves=num_leaves, n_estimators=n_estimators, learning_rate=0.03, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
              random_state=SEED, n_jobs=4, verbose=-1)
    est = lgb.LGBMClassifier(**kw) if task == "clf" else lgb.LGBMRegressor(**kw)
    return Pipeline([("pre", pre), ("est", est)])


def proba(m: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return np.clip(m.predict_proba(X)[:, 1], EPS, 1 - EPS)


# ----------------------------------------------------------------------------- chỉ số
def ece_equal_mass(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    order = np.argsort(p)
    ece, n = 0.0, len(y)
    for idx in np.array_split(order, bins):
        if len(idx):
            ece += len(idx) / n * abs(y[idx].mean() - p[idx].mean())
    return float(ece)


def metrics(y, p) -> dict:
    y, p = np.asarray(y), np.clip(np.asarray(p), EPS, 1 - EPS)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p), "ece": ece_equal_mass(y, p)}


def boot_ci(df: pd.DataFrame, ycol: str, pcol: str, n_boot: int = 300, seed: int = SEED) -> dict:
    """Bootstrap theo cụm khách hàng (một khách xuất hiện nhiều kỳ) → CI 95%."""
    rng = np.random.default_rng(seed)
    groups = {c: g[[ycol, pcol]].to_numpy() for c, g in df.groupby("customer_id")}
    keys = np.array(list(groups))
    out = {k: [] for k in ["roc_auc", "pr_auc", "log_loss", "brier", "ece"]}
    for _ in range(n_boot):
        arr = np.vstack([groups[k] for k in rng.choice(keys, len(keys))])
        if arr[:, 0].min() == arr[:, 0].max():
            continue
        for k, v in metrics(arr[:, 0], arr[:, 1]).items():
            out[k].append(v)
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in out.items()}


# ----------------------------------------------------------------------------- hiệu chỉnh
def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Platt:
    def fit(self, p, y):
        self.m = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p):
        return np.clip(self.m.predict_proba(_logit(p).reshape(-1, 1))[:, 1], EPS, 1 - EPS)


class Isotonic:
    def fit(self, p, y):
        self.m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return self

    def predict(self, p):
        return np.clip(self.m.predict(p), 0.005, 1 - 0.005)  # chặn 0/1 tuyệt đối để log loss hữu hạn


class Identity:
    def fit(self, p, y):
        return self

    def predict(self, p):
        return np.clip(p, EPS, 1 - EPS)


CALIBRATORS = {"none": Identity, "platt": Platt, "isotonic": Isotonic}


# ----------------------------------------------------------------------------- hình
def reliability(ax, y, p, label, color, bins=15):
    order = np.argsort(p)
    xs, ys = [], []
    for idx in np.array_split(order, bins):
        xs.append(p[idx].mean())
        ys.append(y[idx].mean())
    ax.plot(xs, ys, "o-", color=color, label=label, ms=4)


def plot_reliability(path, y, praw, pcal, title_cal):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    hi = max(praw.max(), pcal.max(), y.mean()) * 1.05
    for ax, p, t, c in [(axes[0], praw, "Trước hiệu chỉnh", "tab:red"), (axes[1], pcal, title_cal, "tab:green")]:
        reliability(ax, y, p, t, c)
        ax.plot([0, hi], [0, hi], "k--", lw=1, label="Hiệu chỉnh hoàn hảo")
        ax.set_title(f"{t}\nECE={ece_equal_mass(y, p):.4f} · Brier={brier_score_loss(y, p):.4f}")
        ax.set_xlabel("Xác suất dự báo trung bình trong bin (15 bin cùng khối lượng)")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Tần suất mua lại quan sát")
    fig.suptitle("Biểu đồ độ tin cậy μ̂0 trên tập valid (chỉ dòng a0)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- quy trình μ0 cho một bộ đặc trưng
def run_mu0(tr, va, ca, num, tag):
    """Cùng một quy trình cho mọi bộ đặc trưng: CV lưới nhỏ (train∧a0) → chọn họ mô hình trên valid →
    chọn bộ hiệu chỉnh bằng CV trong calib → khớp lại trên toàn calib. Không dùng locked."""
    feats = num + FEATURES_CAT
    va = va.copy()
    gkf = GroupKFold(n_splits=5)
    cands = [("logit", dict(C=C)) for C in [0.01, 0.1, 1.0]] + \
            [("lgbm", dict(num_leaves=nl, n_estimators=ne)) for nl, ne in itertools.product([4, 8, 16], [150, 400])]
    rows = []
    for kind, hp in cands:
        oof = np.zeros(len(tr))
        for itr, ite in gkf.split(tr, groups=tr.customer_id):
            m = make_logit(num=num, **hp) if kind == "logit" else make_lgbm(num=num, **hp)
            m.fit(tr.iloc[itr][feats], tr.y_obs.iloc[itr])
            oof[ite] = proba(m, tr.iloc[ite][feats])
        rows.append({"model": kind, **hp, "cv_log_loss": log_loss(tr.y_obs, oof),
                     "cv_pr_auc": average_precision_score(tr.y_obs, oof)})
    cv = pd.DataFrame(rows)
    models, hps, val_rows = {}, {}, []
    for kind in ["logit", "lgbm"]:
        best = cv[cv.model == kind].sort_values("cv_log_loss").iloc[0]
        hp = {k: (int(best[k]) if k in ("num_leaves", "n_estimators") else float(best[k]))
              for k in (["C"] if kind == "logit" else ["num_leaves", "n_estimators"])}
        m = make_logit(num=num, **hp) if kind == "logit" else make_lgbm(num=num, **hp)
        m.fit(tr[feats], tr.y_obs)
        models[kind], hps[kind] = m, hp
        val_rows.append({"model": kind, "hp": str(hp), **metrics(va.y_obs, proba(m, va[feats]))})
    vsel = pd.DataFrame(val_rows)
    chosen = vsel.sort_values("log_loss").iloc[0]["model"]
    mu0 = models[chosen]
    ca_p = proba(mu0, ca[feats])
    cal_rows = []
    for name, cls in CALIBRATORS.items():
        oof = np.zeros(len(ca))
        for itr, ite in GroupKFold(n_splits=5).split(ca, groups=ca.customer_id):
            oof[ite] = cls().fit(ca_p[itr], ca.y_obs.values[itr]).predict(ca_p[ite])
        cal_rows.append({"method": name, **metrics(ca.y_obs, oof)})
    calcv = pd.DataFrame(cal_rows)
    best_cal = calcv[calcv.method != "none"].sort_values("log_loss").iloc[0]["method"]
    calibrators = {k: CALIBRATORS[k]().fit(ca_p, ca.y_obs.values) for k in ["platt", "isotonic"]}
    va["p_raw"] = proba(mu0, va[feats])
    for k, c in calibrators.items():
        va[f"p_{k}"] = c.predict(va.p_raw.values)
    va["p_cal"] = va[f"p_{best_cal}"]
    print(f"[{tag}] mô hình={chosen} {hps[chosen]} · hiệu chỉnh={best_cal}")
    return dict(num=num, feats=feats, mu0=mu0, hp=hps[chosen], chosen=chosen, models=models, best_cal=best_cal,
                calibrators=calibrators, cal=calibrators[best_cal], va=va,
                tables={"s2_cv_grid.csv": cv, "s2_model_selection_valid.csv": vsel,
                        "s2_calibration_cv_calib.csv": calcv})


def oof_train_mu0(df, num, chosen_kind, hp):
    """Dự báo μ0 thô NGOÀI FOLD cho mọi dòng train: GroupKFold theo khách; mô hình mỗi fold học trên
    a0 của các khách ngoài fold, rồi chấm điểm mọi dòng của khách trong fold."""
    feats = num + FEATURES_CAT
    trn = df[df.split == "train"]
    out = pd.Series(np.nan, index=trn.index)
    for itr, ite in GroupKFold(n_splits=5).split(trn, groups=trn.customer_id):
        fit_rows = trn.iloc[itr]
        fit_rows = fit_rows[fit_rows.A_logged == "a0"]
        m = make_logit(num=num, **hp) if chosen_kind == "logit" else make_lgbm(num=num, **hp)
        m.fit(fit_rows[feats], fit_rows.y_obs)
        out.iloc[ite] = proba(m, trn.iloc[ite][feats])
    return out


# ----------------------------------------------------------------------------- chính
def main() -> None:
    np.random.seed(SEED)
    T, F = common.p(CFG["paths"]["tables"]), None
    tables = Path(CFG["paths"]["tables"])
    figs = Path(CFG["paths"]["figures"])
    files: list[str] = []

    df = pd.read_parquet(common.p("data/processed/logged.parquet"))
    a0 = df["A_logged"] == "a0"
    tr = df[(df.split == "train") & a0].reset_index(drop=True)
    va = df[(df.split == "valid") & a0].reset_index(drop=True)
    ca = df[(df.split == "calib") & a0].reset_index(drop=True)
    print(f"a0 – train {len(tr)}, valid {len(va)}, calib {len(ca)}")

    # ---- 1-4. Huấn luyện μ0 + hiệu chỉnh cho 2 bộ đặc trưng (đối chiếu lịch), chọn theo log loss valid
    CAL_FEATS = ["t0_month", "days_to_xmas"]
    sets = {"day_du": FEATURES_NUM, "bo_dac_trung_lich": [c for c in FEATURES_NUM if c not in CAL_FEATS]}
    runs = {k: run_mu0(tr, va, ca, v, k) for k, v in sets.items()}
    abl = []
    for k, r in runs.items():
        v = r["va"]
        for stage, col in [("chua_hieu_chinh", "p_raw"), ("da_hieu_chinh", "p_cal")]:
            ci = boot_ci(v, "y_obs", col)
            row = {"bo_dac_trung": k, "n_dac_trung_so": len(sets[k]), "mo_hinh": r["chosen"],
                   "bo_hieu_chinh": r["best_cal"], "giai_doan": stage, "n_valid": len(v),
                   "xac_suat_TB_du_bao": v[col].mean(), "ty_le_that": v.y_obs.mean(),
                   "lech_TB": v[col].mean() - v.y_obs.mean(), **metrics(v.y_obs, v[col])}
            for m_, (lo, hi) in ci.items():
                row[f"{m_}_ci_lo"], row[f"{m_}_ci_hi"] = lo, hi
            abl.append(row)
    abl = pd.DataFrame(abl)
    abl["duoc_chon"] = False
    sel_ll = abl[abl.giai_doan == "da_hieu_chinh"].set_index("bo_dac_trung").log_loss
    feat_choice = sel_ll.idxmin()
    abl.loc[abl.bo_dac_trung == feat_choice, "duoc_chon"] = True
    abl["ly_do_chon"] = ""
    abl.loc[abl.duoc_chon, "ly_do_chon"] = (
        f"log loss valid (xác suất đã hiệu chỉnh) thấp nhất: " +
        ", ".join(f"{k}={v:.4f}" for k, v in sel_ll.items()) +
        "; valid là tập chọn mô hình theo CLAUDE.md; locked không dùng.")
    abl.to_csv(common.p(tables / "s2_ablation_calendar.csv"), index=False)
    files.append(str(tables / "s2_ablation_calendar.csv"))
    print(abl[["bo_dac_trung", "giai_doan", "log_loss", "brier", "ece", "pr_auc",
               "xac_suat_TB_du_bao", "ty_le_that", "duoc_chon"]].to_string())
    print("Bộ đặc trưng được chọn:", feat_choice)

    R = runs[feat_choice]
    num_cols, mu0, cal, best_cal, chosen, va = R["num"], R["mu0"], R["cal"], R["best_cal"], R["chosen"], R["va"]
    for name, t in R["tables"].items():
        t.to_csv(common.p(tables / name), index=False)
        files.append(str(tables / name))
    joblib.dump(mu0, common.p("models/s2_mu0_model.joblib"))
    joblib.dump(R["models"], common.p("models/s2_mu0_candidates.joblib"))
    joblib.dump({"method": best_cal, "calibrator": cal, "all": R["calibrators"], "features_num": num_cols},
                common.p("models/s2_calibrator.joblib"))

    rep = []
    segs = {"toan_bo": np.ones(len(va), bool), "UK": va.is_uk.values == 1, "non_UK": va.is_uk.values == 0,
            "one_time": va.is_one_time.values == 1, "repeat": va.is_one_time.values == 0}
    for seg, mask in segs.items():
        sub = va[mask]
        for col, lab in [("p_raw", "chua_hieu_chinh"), ("p_platt", "platt"), ("p_isotonic", "isotonic")]:
            if seg == "toan_bo" or lab in ("chua_hieu_chinh", best_cal):
                ci = boot_ci(sub, "y_obs", col)
                r = {"phan_khuc": seg, "phuong_phap": lab, "n": len(sub), "ty_le_duong": sub.y_obs.mean(),
                     **metrics(sub.y_obs, sub[col])}
                for k, (lo, hi) in ci.items():
                    r[f"{k}_ci_lo"], r[f"{k}_ci_hi"] = lo, hi
                rep.append(r)
    rep = pd.DataFrame(rep)
    rep.to_csv(common.p(tables / "s2_metrics_valid.csv"), index=False)
    files.append(str(tables / "s2_metrics_valid.csv"))
    print(rep[["phan_khuc", "phuong_phap", "n", "roc_auc", "pr_auc", "log_loss", "brier", "ece"]].to_string())

    plot_reliability(common.p(figs / "fig_s2_reliability_valid.png"), va.y_obs.values, va.p_raw.values,
                     va.p_cal.values, f"Sau hiệu chỉnh ({best_cal})")
    files.append(str(figs / "fig_s2_reliability_valid.png"))

    # ---- 5. Giá trị giỏ Ŝ: LightGBM trên log(value_obs) với value_obs>0, split-conformal 90%
    pos = df[df.value_obs > 0]
    str_ = pos[pos.split == "train"].reset_index(drop=True)
    sva = pos[pos.split == "valid"].reset_index(drop=True)
    sca = pos[pos.split == "calib"].reset_index(drop=True)
    best_s, best_mae = None, np.inf
    srows = []
    for nl, ne in itertools.product([4, 8, 16], [150, 400]):
        oof = np.zeros(len(str_))
        for itr, ite in GroupKFold(n_splits=5).split(str_, groups=str_.customer_id):
            m = make_lgbm(nl, ne, "reg").fit(str_.iloc[itr][FEATS], np.log(str_.value_obs.iloc[itr]))
            oof[ite] = m.predict(str_.iloc[ite][FEATS])
        mae = np.mean(np.abs(oof - np.log(str_.value_obs)))
        srows.append({"num_leaves": nl, "n_estimators": ne, "cv_mae_log": mae})
        if mae < best_mae:
            best_s, best_mae = (nl, ne), mae
    pd.DataFrame(srows).to_csv(common.p(tables / "s2_basket_cv_grid.csv"), index=False)
    files.append(str(tables / "s2_basket_cv_grid.csv"))
    smod = make_lgbm(*best_s, "reg").fit(str_[FEATS], np.log(str_.value_obs))
    joblib.dump(smod, common.p("models/s2_basket_model.joblib"))
    alpha = 0.10
    resid = np.abs(np.log(sca.value_obs) - smod.predict(sca[FEATS]))
    k = int(np.ceil((len(resid) + 1) * (1 - alpha)))
    q = float(np.sort(resid)[min(k, len(resid)) - 1])
    smear = float(np.mean(np.exp(np.log(sca.value_obs) - smod.predict(sca[FEATS]))))  # Duan, khớp trên calib
    lv = smod.predict(sva[FEATS])
    covered = (np.log(sva.value_obs) >= lv - q) & (np.log(sva.value_obs) <= lv + q)
    # độ phủ có điều kiện theo nhóm UK/non-UK
    cov_rows = [{"nhom": "toan_bo", "n": len(sva), "coverage": covered.mean(), "muc_tieu": 1 - alpha,
                 "q_log": q, "smearing_duan": smear,
                 "ky_vong_that_valid": float(sva.value_obs.mean()),
                 "TB_s_hat_median_valid": float(np.exp(lv).mean()),
                 "TB_s_hat_mean_valid": float(np.exp(lv).mean() * smear),
                 "do_rong_trung_vi_ty_le": float(np.median(np.exp(2 * q))),
                 "mae_log_valid": float(np.mean(np.abs(np.log(sva.value_obs) - lv))),
                 "r2_log_valid": float(1 - np.sum((np.log(sva.value_obs) - lv) ** 2) /
                                        np.sum((np.log(sva.value_obs) - np.log(sva.value_obs).mean()) ** 2))}]
    for nm, msk in [("UK", sva.is_uk == 1), ("non_UK", sva.is_uk == 0)]:
        cov_rows.append({"nhom": nm, "n": int(msk.sum()), "coverage": covered[msk.values].mean(),
                         "muc_tieu": 1 - alpha})
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(common.p(tables / "s2_basket_conformal_valid.csv"), index=False)
    files.append(str(tables / "s2_basket_conformal_valid.csv"))
    print(cov.to_string())

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(np.exp(lv), sva.value_obs, s=6, alpha=0.4)
    lim = [min(np.exp(lv).min(), sva.value_obs.min()), max(np.exp(lv).max(), sva.value_obs.max())]
    ax.plot(lim, lim, "k--", lw=1, label="y = ŷ")
    ax.plot(lim, np.array(lim) * np.exp(q), "r:", lw=1, label=f"Biên conformal 90% (×{np.exp(q):.2f})")
    ax.plot(lim, np.array(lim) / np.exp(q), "r:", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Giá trị giỏ dự báo Ŝ (GBP, thang log)"); ax.set_ylabel("Giá trị giỏ quan sát (GBP, thang log)")
    ax.set_title(f"Ŝ trên valid – độ phủ conformal {covered.mean():.3f} (mục tiêu 0.90)")
    ax.legend(); fig.tight_layout()
    fig.savefig(common.p(figs / "fig_s2_basket_conformal.png"), dpi=160); plt.close(fig)
    files.append(str(figs / "fig_s2_basket_conformal.png"))

    # ---- 6. Lưu dự báo cho MỌI dòng (train: ngoài fold)
    out = df[["customer_id", "t0", "split", "A_logged"]].copy()
    out["mu0_raw"] = proba(mu0, df[R["feats"]])
    is_tr = (df.split == "train").values
    out.loc[is_tr, "mu0_raw"] = oof_train_mu0(df, num_cols, chosen, R["hp"]).values
    out["mu0_is_oof"] = is_tr
    out["mu0_cal"] = cal.predict(out["mu0_raw"].values)
    lp = smod.predict(df[FEATS])
    out["s_hat"], out["s_lo"], out["s_hi"] = np.exp(lp), np.exp(lp - q), np.exp(lp + q)
    out["s_hat_mean"] = out["s_hat"] * smear
    pred_path = common.p("data/processed/pred_s2.parquet")
    out.to_parquet(pred_path, index=False)
    files.append("data/processed/pred_s2.parquet")
    files += ["models/s2_mu0_model.joblib", "models/s2_mu0_candidates.joblib", "models/s2_calibrator.joblib",
              "models/s2_basket_model.joblib"]

    # ---- 7. Tiêu chí chấp nhận
    g = rep[(rep.phan_khuc == "toan_bo")].set_index("phuong_phap")
    after, before = g.loc[best_cal], g.loc["chua_hieu_chinh"]
    acc = pd.DataFrame([
        {"tieu_chi": "ECE sau hiệu chỉnh <= 0.03 (valid)", "gia_tri": after.ece, "dat": bool(after.ece <= 0.03)},
        {"tieu_chi": "Brier sau <= trước (valid)", "gia_tri": after.brier - before.brier,
         "dat": bool(after.brier <= before.brier)},
        {"tieu_chi": "Coverage conformal Ŝ trong [0.85,0.95] (valid)", "gia_tri": float(covered.mean()),
         "dat": bool(0.85 <= covered.mean() <= 0.95)}])
    acc.to_csv(common.p(tables / "s2_acceptance.csv"), index=False)
    files.append(str(tables / "s2_acceptance.csv"))
    print(acc.to_string())

    common.write_manifest("s2", files, extra={"mu0_model": chosen, "calibrator": best_cal,
                                              "basket_hp": best_s, "conformal_q_log": q,
                                              "features_calendar_choice": feat_choice, "smearing_duan": smear},
                          packages=["pandas", "numpy", "pyarrow", "scikit-learn", "lightgbm", "matplotlib", "joblib"])
    print("XONG GĐ2")


if __name__ == "__main__":
    # chạy qua import để lớp Platt/Isotonic được pickle dưới tên module s2_predict (nạp lại được ở GĐ8)
    from s2_predict import main as _main
    _main()
