"""GĐ3 – Giải thích & độ tin cậy cho μ0 (Chương 3).

Giải thích mô hình μ0 đã chọn ở GĐ2 (LightGBM, bộ đặc trưng ĐẦY ĐỦ gồm t0_month, days_to_xmas) trên tập valid
(chỉ các dòng a0, nơi μ0 có nhãn hợp lệ). Không đọc file đáp án, không dùng locked.

CẢNH BÁO DIỄN GIẢI: PFI và SHAP mô tả cách MÔ HÌNH dùng đặc trưng, KHÔNG phải quan hệ nhân quả.
SHAP ở thang log-odds của mô hình thô (chưa hiệu chỉnh); hiệu chỉnh Platt là đơn điệu nên thứ hạng không đổi.

Mã lý do (reason codes): `reason_codes(df)` dùng được cho bất kỳ bảng đặc trưng nào (không cần nhãn). File
reasons_s3.parquet CHỈ chứa train/valid/calib; locked/oot sẽ được Router/GĐ8 gọi `reason_codes` khi tới lượt,
để GĐ3 không chạm vào hai tập đó.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import itertools
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import log_loss, roc_auc_score

import common
from s1_data import FEATURES_CAT, FEATURES_NUM
from s2_predict import EPS, make_lgbm

warnings.filterwarnings("ignore", message="LightGBM binary classifier")
CFG = common.load_config()
SEED = CFG["project"]["seed"]
FEATS = FEATURES_NUM + FEATURES_CAT
TABLES, FIGS = Path(CFG["paths"]["tables"]), Path(CFG["paths"]["figures"])

# ----------------------------------------------------------------------------- từ điển nghiệp vụ
LABEL = {
    "recency_days": ("Số ngày từ đơn gần nhất", "RFM"), "tenure_days": ("Thâm niên khách", "RFM"),
    "n_orders_all": ("Số đơn toàn lịch sử", "RFM"), "value_all": ("Doanh số toàn lịch sử", "RFM"),
    "n_orders_30": ("Số đơn 30 ngày", "RFM"), "n_orders_90": ("Số đơn 90 ngày", "RFM"),
    "n_orders_180": ("Số đơn 180 ngày", "RFM"), "n_orders_365": ("Số đơn 365 ngày", "RFM"),
    "value_30": ("Doanh số 30 ngày", "RFM"), "value_90": ("Doanh số 90 ngày", "RFM"),
    "value_180": ("Doanh số 180 ngày", "RFM"), "value_365": ("Doanh số 365 ngày", "RFM"),
    "aov_365": ("Giá trị đơn TB 365 ngày", "RFM"), "aov_all": ("Giá trị đơn TB toàn lịch sử", "RFM"),
    "units_per_order_365": ("Số lượng hàng / đơn", "RFM"), "lines_per_order_365": ("Số dòng hàng / đơn", "RFM"),
    "distinct_products_365": ("Số mã hàng đã mua", "RFM"), "is_one_time": ("Khách mua một lần", "RFM"),
    "value_trend_90_365": ("Xu hướng doanh số 90/365 ngày", "RFM"),
    "ipt_mean_days": ("Chu kỳ mua TB", "Nhịp mua"), "ipt_std_days": ("Độ lệch chu kỳ mua", "Nhịp mua"),
    "overdue_ratio": ("Mức quá chu kỳ mua", "Nhịp mua"),
    "n_cancels_365": ("Số hóa đơn hủy 365 ngày", "Hủy đơn"), "cancel_value_365": ("Giá trị hủy 365 ngày", "Hủy đơn"),
    "cancel_rate_365": ("Tỷ lệ hủy 365 ngày", "Hủy đơn"), "days_since_last_cancel": ("Ngày từ lần hủy gần nhất", "Hủy đơn"),
    "t0_month": ("Tháng của kỳ quyết định", "Lịch"), "days_to_xmas": ("Số ngày đến Giáng sinh", "Lịch"),
    "is_uk": ("Khách ở Anh", "Quốc gia"), "country": ("Quốc gia", "Quốc gia"),
}
GROUPS = ["RFM", "Nhịp mua", "Hủy đơn", "Lịch", "Quốc gia"]
SHORT = {f: f.upper() for f in FEATS}


# ----------------------------------------------------------------------------- SHAP gộp theo đặc trưng gốc
def _source_feature(name: str) -> str:
    """Tên cột sau tiền xử lý → đặc trưng gốc (cộng gộp one-hot country và cờ thiếu dữ liệu)."""
    n = name.split("__", 1)[1]
    if n.startswith("missingindicator_"):
        return n[len("missingindicator_"):]
    if n.startswith("country_"):
        return "country"
    return n


def shap_by_feature(model, X: pd.DataFrame):
    """Trả (ma trận SHAP log-odds n×F theo FEATS, giá trị nền). Cộng gộp SHAP của one-hot/cờ thiếu về đặc trưng gốc."""
    pre, est = model.named_steps["pre"], model.named_steps["est"]
    Z = pre.transform(X)
    names = [_source_feature(c) for c in pre.get_feature_names_out()]
    expl = shap.TreeExplainer(est)
    sv = expl.shap_values(Z)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    out = np.zeros((len(X), len(FEATS)))
    for j, nm in enumerate(names):
        out[:, FEATS.index(nm)] += sv[:, j]
    base = float(np.ravel(expl.expected_value)[-1])
    return out, base


# ----------------------------------------------------------------------------- mã lý do
CAL_FEATS = ["t0_month", "days_to_xmas"]   # đặc trưng lịch: không ngoại suy được (GĐ3) → không làm lý do


def reason_codes(model, X: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Top-k đặc trưng theo |SHAP| cho mỗi dòng, CHỈ xét đặc trưng KHÔNG phải lịch (t0_month, days_to_xmas
    hoạt động như hằng số cộng lên log-odds, không phải thông tin hành động được về khách).
    Mã = <ĐẶC_TRƯNG>|UP (đẩy xác suất mua lại lên) hoặc |DOWN. Không cần nhãn.
    Router (GĐ7) dùng cột reason_i_code; reason_i_text là câu tiếng Việt cho người duyệt."""
    sv, _ = shap_by_feature(model, X)
    sv = sv.copy()
    sv[:, [FEATS.index(c) for c in CAL_FEATS]] = 0.0          # loại đặc trưng lịch khỏi xếp hạng lý do
    order = np.argsort(-np.abs(sv), axis=1)[:, :k]
    out = {}
    for r in range(k):
        j = order[:, r]
        val = sv[np.arange(len(sv)), j]
        out[f"reason_{r+1}_code"] = [f"{SHORT[FEATS[a]]}|{'UP' if v > 0 else 'DOWN'}" for a, v in zip(j, val)]
        out[f"reason_{r+1}_text"] = [f"{LABEL[FEATS[a]][0]} {'làm tăng' if v > 0 else 'làm giảm'} khả năng mua lại"
                                     for a, v in zip(j, val)]
        out[f"reason_{r+1}_shap"] = val
    return pd.DataFrame(out, index=X.index)


# ----------------------------------------------------------------------------- PFI
def pfi(model, X, y, groups: dict[str, list[str]], n_rep=10, seed=SEED):
    """PFI theo log loss: tăng log loss khi hoán vị cột (nhóm cột hoán vị CÙNG một hoán vị hàng)."""
    rng = np.random.default_rng(seed)
    base = log_loss(y, np.clip(model.predict_proba(X)[:, 1], EPS, 1 - EPS))
    rows = []
    for name, cols in groups.items():
        d = []
        for _ in range(n_rep):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            d.append(log_loss(y, np.clip(model.predict_proba(Xp)[:, 1], EPS, 1 - EPS)) - base)
        rows.append({"muc": name, "pfi_mean": np.mean(d), "pfi_sd": np.std(d, ddof=1),
                     "pfi_ci_lo": np.mean(d) - 1.96 * np.std(d, ddof=1) / np.sqrt(n_rep),
                     "pfi_ci_hi": np.mean(d) + 1.96 * np.std(d, ddof=1) / np.sqrt(n_rep)})
    return pd.DataFrame(rows).sort_values("pfi_mean", ascending=False), base


# ----------------------------------------------------------------------------- faithfulness
def deletion_test(model, X, y, sv, baseline, ks=(1, 3, 5, 10), seed=SEED):
    """Xóa top-k đặc trưng theo |SHAP| từng dòng (thay bằng giá trị nền của train) so với xóa k đặc trưng NGẪU NHIÊN.
    Mô hình trung thực ⇒ xóa top-k làm xác suất thay đổi nhiều hơn xóa ngẫu nhiên."""
    rng = np.random.default_rng(seed)
    p0 = np.clip(model.predict_proba(X)[:, 1], EPS, 1 - EPS)
    lg0 = np.log(p0 / (1 - p0))
    order = np.argsort(-np.abs(sv), axis=1)
    rows = []
    for k in ks:
        for mode in ["top_k_shap", "ngau_nhien"]:
            mask = np.zeros(sv.shape, bool)
            if mode == "top_k_shap":
                mask[np.arange(len(sv))[:, None], order[:, :k]] = True
            else:
                for i in range(len(sv)):
                    mask[i, rng.choice(len(FEATS), k, replace=False)] = True
            Xm = X.copy()
            for j, f in enumerate(FEATS):
                Xm[f] = np.where(mask[:, j], baseline[f], X[f])
            p = np.clip(model.predict_proba(Xm)[:, 1], EPS, 1 - EPS)
            lg = np.log(p / (1 - p))
            rows.append({"k": k, "cach_xoa": mode, "do_lech_logit_tuyet_doi_TB": float(np.mean(np.abs(lg - lg0))),
                         "roc_auc": roc_auc_score(y, p), "log_loss": log_loss(y, p)})
    return pd.DataFrame(rows)


def main() -> None:
    np.random.seed(SEED)
    files: list[str] = []
    df = pd.read_parquet(common.p("data/processed/logged.parquet"))
    a0 = df.A_logged == "a0"
    tr = df[(df.split == "train") & a0].reset_index(drop=True)
    va = df[(df.split == "valid") & a0].reset_index(drop=True)
    model = joblib.load(common.p("models/s2_mu0_model.joblib"))
    used = set(model.named_steps["pre"].feature_names_in_)
    assert {"t0_month", "days_to_xmas"} <= used, "GĐ3 yêu cầu mô hình đủ đặc trưng (có t0_month, days_to_xmas)"
    Xv, yv = va[FEATS], va.y_obs.values

    # ---- 1. PFI từng đặc trưng + theo nhóm
    grp = {g: [f for f in FEATS if LABEL[f][1] == g] for g in GROUPS}
    p_f, base_ll = pfi(model, Xv, yv, {f: [f] for f in FEATS})
    p_f["nhan_nghiep_vu"] = p_f.muc.map(lambda f: LABEL[f][0])
    p_f["nhom"] = p_f.muc.map(lambda f: LABEL[f][1])
    p_f.to_csv(common.p(TABLES / "s3_pfi_feature.csv"), index=False)
    p_g, _ = pfi(model, Xv, yv, grp)
    p_g.rename(columns={"muc": "nhom"}).to_csv(common.p(TABLES / "s3_pfi_group.csv"), index=False)
    files += [str(TABLES / "s3_pfi_feature.csv"), str(TABLES / "s3_pfi_group.csv")]
    print(f"log loss nền (valid, mô hình thô) = {base_ll:.4f}")
    print(p_f.head(12).to_string()); print(p_g.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw={"width_ratios": [1.6, 1]})
    t = p_f.head(15).iloc[::-1]
    axes[0].barh(t.nhan_nghiep_vu, t.pfi_mean, xerr=1.96 * t.pfi_sd, color="tab:blue")
    axes[0].set_xlabel("Mức tăng log loss khi hoán vị (10 lần, thanh sai số = ±1.96 sd)")
    axes[0].set_title("PFI theo đặc trưng (top 15) – tập valid")
    t = p_g.iloc[::-1]
    axes[1].barh(t.muc, t.pfi_mean, xerr=1.96 * t.pfi_sd, color="tab:orange")
    axes[1].set_xlabel("Mức tăng log loss khi hoán vị cả nhóm")
    axes[1].set_title("PFI theo nhóm đặc trưng")
    fig.text(0.5, 0.005, "PFI mô tả mức mô hình dựa vào đặc trưng, KHÔNG phải quan hệ nhân quả.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(common.p(FIGS / "fig_s3_pfi.png"), dpi=160); plt.close(fig)
    files.append(str(FIGS / "fig_s3_pfi.png"))

    # ---- 2. SHAP (thang log-odds) trên valid
    sv, base_val = shap_by_feature(model, Xv)
    z = model.named_steps["pre"].transform(Xv)
    p_raw = model.predict_proba(Xv)[:, 1]
    chk = np.abs(base_val + sv.sum(1) - np.log(p_raw / (1 - p_raw))).max()
    print(f"kiểm tra cộng tính SHAP: max|nền+ΣSHAP − logit| = {chk:.2e}")
    assert chk < 1e-3
    shap_tab = pd.DataFrame({"dac_trung": FEATS, "nhan_nghiep_vu": [LABEL[f][0] for f in FEATS],
                             "nhom": [LABEL[f][1] for f in FEATS],
                             "mean_abs_shap_logodds": np.abs(sv).mean(0),
                             "mean_shap_logodds": sv.mean(0)}).sort_values("mean_abs_shap_logodds", ascending=False)
    shap_tab["hang"] = np.arange(1, len(shap_tab) + 1)
    shap_tab.to_csv(common.p(TABLES / "s3_shap_importance.csv"), index=False)
    files.append(str(TABLES / "s3_shap_importance.csv"))
    print(shap_tab.head(12).to_string())

    Xn = Xv.copy()
    Xn["country"] = Xn["country"].astype("category").cat.codes.replace(-1, np.nan).astype(float)
    expl = shap.Explanation(values=sv, base_values=np.full(len(sv), base_val), data=Xn[FEATS].to_numpy(float),
                            feature_names=[LABEL[f][0] for f in FEATS])
    plt.figure(figsize=(9, 7.5))
    shap.plots.beeswarm(expl, max_display=15, show=False)
    plt.title("SHAP beeswarm của μ̂0 – thang LOG-ODDS (mô hình thô), tập valid\n"
              "(Quốc gia: màu theo mã số thứ tự, không có ý nghĩa thứ bậc; SHAP không phải nhân quả)", fontsize=10)
    plt.tight_layout(); plt.savefig(common.p(FIGS / "fig_s3_shap_beeswarm.png"), dpi=160, bbox_inches="tight"); plt.close()
    files.append(str(FIGS / "fig_s3_shap_beeswarm.png"))

    # waterfall: 3 khách tiêu biểu (xác suất thấp / trung vị / cao ở nhóm khách quay lại; chọn theo分位 của p_raw)
    rep_idx = {}
    rp = va.is_one_time.values == 0
    order_rep = np.where(rp)[0][np.argsort(p_raw[rp])]
    rep_idx["Rủi ro rời bỏ cao (p thấp)"] = order_rep[int(0.05 * len(order_rep))]
    rep_idx["Trung vị"] = order_rep[len(order_rep) // 2]
    rep_idx["Khả năng mua lại cao (p cao)"] = order_rep[int(0.95 * len(order_rep))]
    wf_rows = []
    for n, (title, i) in enumerate(rep_idx.items(), 1):
        plt.figure(figsize=(8.5, 5.5))
        shap.plots.waterfall(expl[i], max_display=10, show=False)
        plt.title(f"{title} – khách {va.customer_id.iloc[i]}, t0={pd.Timestamp(va.t0.iloc[i]).date()}, "
                  f"μ̂0 thô={p_raw[i]:.3f}\n(thang log-odds)", fontsize=10)
        plt.tight_layout(); plt.savefig(common.p(FIGS / f"fig_s3_shap_waterfall_{n}.png"), dpi=160, bbox_inches="tight"); plt.close()
        files.append(str(FIGS / f"fig_s3_shap_waterfall_{n}.png"))
        wf_rows.append({"hinh": n, "mo_ta": title, "customer_id": va.customer_id.iloc[i], "t0": va.t0.iloc[i],
                        "mu0_raw": p_raw[i]})
    pd.DataFrame(wf_rows).to_csv(common.p(TABLES / "s3_waterfall_customers.csv"), index=False)
    files.append(str(TABLES / "s3_waterfall_customers.csv"))

    # ---- 3. Từ điển đặc trưng → nhãn nghiệp vụ
    dd = pd.DataFrame([{"dac_trung": f, "ma_ly_do": SHORT[f], "nhan_nghiep_vu": LABEL[f][0], "nhom": LABEL[f][1]}
                       for f in FEATS])
    dd.to_csv(common.p(TABLES / "s3_feature_business_labels.csv"), index=False)
    files.append(str(TABLES / "s3_feature_business_labels.csv"))

    # ---- 4. Mã lý do top-3 cho train/valid/calib (KHÔNG locked/oot – xem docstring)
    sub = df[df.split.isin(["train", "valid", "calib"])]
    rc = reason_codes(model, sub[FEATS])
    rc.insert(0, "customer_id", sub.customer_id.values)
    rc.insert(1, "t0", sub.t0.values)
    rc.insert(2, "split", sub.split.values)
    rc.reset_index(drop=True).to_parquet(common.p("data/processed/reasons_s3.parquet"), index=False)
    files.append("data/processed/reasons_s3.parquet")
    freq = pd.concat([rc[f"reason_{r}_code"] for r in (1, 2, 3)]).value_counts().rename_axis("ma_ly_do").reset_index(name="so_lan")
    freq.to_csv(common.p(TABLES / "s3_reason_code_freq.csv"), index=False)
    files.append(str(TABLES / "s3_reason_code_freq.csv"))
    print(freq.head(10).to_string())

    # ---- 5. Stability: 5 lần bootstrap huấn luyện lại (theo khách) → tương quan hạng mean|SHAP| trên valid
    cvs = pd.read_csv(TABLES / "s2_cv_grid.csv")
    best = cvs[cvs.model == "lgbm"].sort_values("cv_log_loss").iloc[0]
    nl, ne = int(best.num_leaves), int(best.n_estimators)
    custs = tr.customer_id.unique()
    rng = np.random.default_rng(SEED)
    ranks = {"mo_hinh_goc": shap_tab.set_index("dac_trung").mean_abs_shap_logodds.reindex(FEATS).values}
    for b in range(5):
        pick = rng.choice(custs, len(custs))
        bs = pd.concat([tr[tr.customer_id == c] for c in pd.unique(pick)]) if False else \
            tr.merge(pd.DataFrame({"customer_id": pick}), on="customer_id")
        m = make_lgbm(nl, ne).fit(bs[FEATS], bs.y_obs)
        s_b, _ = shap_by_feature(m, Xv)
        ranks[f"bootstrap_{b+1}"] = np.abs(s_b).mean(0)
    R = pd.DataFrame(ranks, index=FEATS)
    pairs = []
    for a, b in itertools.combinations(R.columns, 2):
        pairs.append({"cap": f"{a} vs {b}", "spearman": spearmanr(R[a], R[b]).statistic})
    pairs = pd.DataFrame(pairs)
    vs_ref = pairs[pairs.cap.str.startswith("mo_hinh_goc")]
    stab = {"spearman_TB_cac_cap": pairs.spearman.mean(), "spearman_min": pairs.spearman.min(),
            "spearman_TB_so_voi_mo_hinh_goc": vs_ref.spearman.mean()}
    # độ ổn định top-5
    top5 = [set(R[c].sort_values(ascending=False).head(5).index) for c in R.columns]
    stab["top5_chung_ca_6_mo_hinh"] = sorted(set.intersection(*top5))
    pairs.to_csv(common.p(TABLES / "s3_shap_stability.csv"), index=False)
    R.assign(hang_goc=R["mo_hinh_goc"].rank(ascending=False)).to_csv(common.p(TABLES / "s3_shap_stability_importance.csv"))
    files += [str(TABLES / "s3_shap_stability.csv"), str(TABLES / "s3_shap_stability_importance.csv")]
    print(stab)

    # ---- 6. Faithfulness: xóa top-k vs xóa ngẫu nhiên
    base_vals = {f: (tr[f].median() if f in FEATURES_NUM else tr[f].mode().iloc[0]) for f in FEATS}
    ft = deletion_test(model, Xv, yv, sv, base_vals)
    ft.to_csv(common.p(TABLES / "s3_faithfulness_deletion.csv"), index=False)
    files.append(str(TABLES / "s3_faithfulness_deletion.csv"))
    print(ft.to_string())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for mode, c, lab in [("top_k_shap", "tab:red", "Xóa top-k theo |SHAP|"), ("ngau_nhien", "tab:gray", "Xóa k ngẫu nhiên")]:
        s_ = ft[ft.cach_xoa == mode]
        ax[0].plot(s_.k, s_.do_lech_logit_tuyet_doi_TB, "o-", color=c, label=lab)
        ax[1].plot(s_.k, s_.roc_auc, "o-", color=c, label=lab)
    ax[0].set_ylabel("Độ lệch log-odds tuyệt đối trung bình"); ax[1].set_ylabel("ROC-AUC trên valid")
    for a in ax:
        a.set_xlabel("k – số đặc trưng bị thay bằng giá trị nền của train"); a.legend()
    fig.suptitle("Kiểm tra faithfulness của SHAP (xóa đặc trưng)"); fig.tight_layout()
    fig.savefig(common.p(FIGS / "fig_s3_faithfulness.png"), dpi=160); plt.close(fig)
    files.append(str(FIGS / "fig_s3_faithfulness.png"))

    # ---- 7. Đặc trưng lịch: quan trọng đến đâu và có ngoại suy được không
    tm_train = sorted(tr.t0_month.unique().astype(int)); tm_valid = sorted(va.t0_month.unique().astype(int))
    unseen = sorted(set(tm_valid) - set(tm_train))
    cal_feats = ["t0_month", "days_to_xmas"]
    cal_pfi = p_g.set_index("muc").loc["Lịch"]
    cal_shap = shap_tab[shap_tab.dac_trung.isin(cal_feats)]
    sv_cal = sv[:, [FEATS.index(c) for c in cal_feats]].sum(1)
    cal_row = pd.DataFrame([{
        "thang_train": str([int(x) for x in tm_train]), "thang_valid": str([int(x) for x in tm_valid]),
        "thang_valid_chua_thay_o_train": str([int(x) for x in unseen]),
        "ty_le_dong_valid_o_thang_chua_thay": float(va.t0_month.isin(unseen).mean()),
        "days_to_xmas_train_min": float(tr.days_to_xmas.min()), "days_to_xmas_train_max": float(tr.days_to_xmas.max()),
        "days_to_xmas_valid_min": float(va.days_to_xmas.min()), "days_to_xmas_valid_max": float(va.days_to_xmas.max()),
        "shap_sd_lich_trong_valid": float(sv_cal.std()), "n_gia_tri_days_to_xmas_valid": int(va.days_to_xmas.nunique()),
        "pfi_nhom_lich": cal_pfi.pfi_mean, "pfi_nhom_lich_ci_lo": cal_pfi.pfi_ci_lo, "pfi_nhom_lich_ci_hi": cal_pfi.pfi_ci_hi,
        "hang_pfi_nhom_lich": int((p_g.pfi_mean > cal_pfi.pfi_mean).sum() + 1),
        "shap_abs_TB_lich": float(np.abs(sv_cal).mean()), "shap_TB_lich_co_dau": float(sv_cal.mean()),
        "ty_le_shap_lich_tren_tong": float(np.abs(sv_cal).mean() / np.abs(sv).sum(1).mean())}])
    cal_row.to_csv(common.p(TABLES / "s3_calendar_features.csv"), index=False)
    files.append(str(TABLES / "s3_calendar_features.csv"))
    print(cal_row.T.to_string())

    common.write_manifest("s3", files, extra={"stability": {k: (v if not isinstance(v, float) else round(v, 4))
                                                            for k, v in stab.items()},
                                              "base_logloss_valid": base_ll},
                          packages=["pandas", "numpy", "pyarrow", "scikit-learn", "lightgbm", "shap", "matplotlib"])
    print("XONG GĐ3")


if __name__ == "__main__":
    main()
