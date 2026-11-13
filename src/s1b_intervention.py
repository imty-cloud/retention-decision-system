"""
GĐ1b – Lớp can thiệp bán tổng hợp (semi-synthetic intervention layer).

Vì sao cần: Online Retail II không có thí nghiệm voucher. Ta giữ Y(0) là hành vi mua lại THẬT,
rồi sinh Y(a) cho a ∈ {a1, a2, a3} từ một cơ chế đã biết. Nhờ vậy:
  (1) DML phải khử nhiễu thật vì chính sách lịch sử bị nhiễu bởi RFM;
  (2) có nhánh ngẫu nhiên 20% để đánh giá Qini/OPE không thiên lệch;
  (3) có "đáp án" (potential outcomes + lợi nhuận thực hiện của mọi hành động) để tính Regret thật.

Đầu ra:
  data/processed/logged.parquet   – DỮ LIỆU MÔ HÌNH ĐƯỢC THẤY: snapshot + A_logged, propensity, y_obs, value_obs
  data/processed/oracle.parquet   – ĐÁP ÁN, CHỈ dùng để ĐÁNH GIÁ (regret, CATE thật). Cấm đọc khi huấn luyện.

Giả định đơn giản hóa (ghi vào phần Hạn chế): mỗi (khách, t0) độc lập; can thiệp không làm thay đổi
lịch sử ở các t0 sau (không có hiệu ứng mang sang); giá trị giỏ của người mới chuyển đổi là mô phỏng.

Chạy:  python src/s1b_intervention.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config, p, write_manifest  # noqa: E402

ACTIONS = ["a0", "a1", "a2", "a3"]


def pct_rank_within_t0(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("t0")[col].rank(pct=True, method="average")


def response_probs(df: pd.DataFrame, g: dict) -> pd.DataFrame:
    """c_a: P(Y(a)=1 | Y(0)=0, x) ; d_a: P(Y(a)=0 | Y(0)=1, x). Chỉ dùng đặc trưng tại t0."""
    r = pct_rank_within_t0(df, "recency_days")
    f = pct_rank_within_t0(df, "n_orders_365")
    m = pct_rank_within_t0(df, "aov_365")
    w = pct_rank_within_t0(df, "units_per_order_365")
    lapse = np.exp(-((r - g["lapse_center"]) / g["lapse_width"]) ** 2)
    scale = np.where(df["split"].eq("oot"), g["oot_persuasion_scale"], 1.0)
    c1 = (g["c1_scale"] * lapse * (1 - 0.6 * f) + g["c_floor"]) * scale
    c2 = (g["c2_mult"] * c1 + g["c2_price_sens"] * (1 - m) * lapse * scale)
    c3 = (g["c3_mult"] * c1 + g["c3_value"] * m * lapse * scale)
    out = pd.DataFrame({
        "c_a1": c1, "c_a2": c2, "c_a3": c3,
        "d_a1": g["d1"] * f, "d_a2": g["d2"] * f, "d_a3": g["d3"] * f * w,
        "_r": r, "_f": f, "_m": m, "_w": w}, index=df.index)
    return out.clip(0, 1)


def legacy_propensities(rk: pd.DataFrame, g: dict) -> np.ndarray:
    """Luật RFM cũ (gây nhiễu): khách 'đã lâu chưa mua' nhận email, khách giá trị cao nhận voucher sâu/cuộc gọi."""
    L = g["legacy_logits"]
    logits = np.column_stack([
        np.full(len(rk), L["a0"]),
        L["a1_r"] * rk["_r"],
        L["a2_mr"] * rk["_m"] * rk["_r"],
        L["a3_m"] * rk["_m"] + L["a3_bias"]])
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def generate(snap: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = cfg["intervention_dgp"]
    biz = cfg["business"]
    rng = np.random.default_rng(g["seed"])
    df = snap.reset_index(drop=True).copy()
    n = len(df)
    rk = response_probs(df, g)

    # --- Kết quả tiềm năng: dùng chung số ngẫu nhiên U (common random numbers) => đơn điệu, so sánh công bằng
    U = rng.uniform(size=n)
    y0 = df["y"].to_numpy()
    Y = {"a0": y0}
    for a in ["a1", "a2", "a3"]:
        conv = (y0 == 0) & (U < rk[f"c_{a}"].to_numpy())
        lost = (y0 == 1) & (U < rk[f"d_{a}"].to_numpy())
        Y[a] = np.where(conv, 1, np.where(lost, 0, y0)).astype("int8")

    # --- Giá trị giỏ nếu mua: người vốn mua dùng giá trị THẬT; người mới chuyển đổi dùng AOV * nhiễu lognormal
    sim_basket = df["aov_365"].to_numpy() * rng.lognormal(0, g["basket_noise_sd"], size=n)
    S = np.where(y0 == 1, df["y_value_h"].to_numpy(), sim_basket)

    margin = biz["gross_margin_rate"]
    profit, disc_cost = {}, {}
    for a in ACTIONS:
        spec = biz["actions"][a]
        dc = np.minimum(spec["discount"] * S, spec["face_value_cap"]) * Y[a]
        disc_cost[a] = dc
        profit[a] = margin * S * Y[a] - dc - spec["contact_cost"]

    # CATE thật (kỳ vọng theo U, điều kiện trên Y(0) đã hiện thực) và lợi nhuận tăng thêm kỳ vọng
    oracle = pd.DataFrame({"customer_id": df["customer_id"], "t0": df["t0"], "split": df["split"],
                           "basket_value_S": S})
    for a in ACTIONS:
        oracle[f"y_{a}"] = Y[a]
        oracle[f"profit_{a}"] = profit[a]
        oracle[f"discount_cost_{a}"] = disc_cost[a]
    for a in ["a1", "a2", "a3"]:
        # tác động trên xác suất mua, xét theo Y(0) đã hiện thực của đơn vị đó
        oracle[f"tau_cond_{a}"] = np.where(y0 == 0, rk[f"c_{a}"], -rk[f"d_{a}"])
        oracle[f"uplift_profit_{a}"] = profit[a] - profit["a0"]       # giá trị tăng thêm ĐÃ HIỆN THỰC
    best = np.column_stack([profit[a] for a in ACTIONS]).argmax(axis=1)
    oracle["oracle_best_action_unconstrained"] = np.array(ACTIONS)[best]

    # --- Chính sách lịch sử (logging policy)
    pi_legacy = legacy_propensities(rk, g)
    k = g["random_arm_pct"] / 100
    pi_marg = k * 0.25 + (1 - k) * pi_legacy                      # propensity biên dùng cho IPW/DR
    is_random = rng.uniform(size=n) < k
    u2 = rng.uniform(size=n)
    pick_legacy = (u2[:, None] > pi_legacy.cumsum(axis=1)).sum(axis=1)
    pick_random = rng.integers(0, 4, size=n)
    A = np.where(is_random, pick_random, pick_legacy).clip(0, 3)

    logged = df.copy()
    logged["A_logged"] = np.array(ACTIONS)[A]
    logged["is_random_arm"] = is_random.astype("int8")
    for j, a in enumerate(ACTIONS):
        logged[f"propensity_{a}"] = pi_marg[:, j]
    logged["propensity_logged"] = pi_marg[np.arange(n), A]
    Ymat = np.column_stack([Y[a] for a in ACTIONS])
    Pmat = np.column_stack([profit[a] for a in ACTIONS])
    Dmat = np.column_stack([disc_cost[a] for a in ACTIONS])
    logged["y_obs"] = Ymat[np.arange(n), A].astype("int8")
    logged["value_obs"] = np.where(logged["y_obs"] == 1, S, 0.0)     # doanh thu quan sát được
    logged["discount_cost_obs"] = Dmat[np.arange(n), A]
    logged["profit_obs"] = Pmat[np.arange(n), A]
    # y gốc là Y(0) THẬT -> chỉ đúng cho đơn vị A=a0; đổi tên để không ai dùng nhầm làm nhãn quan sát
    logged = logged.rename(columns={"y": "y0_real_DO_NOT_USE_AS_LABEL",
                                    "y_value_h": "y0_value_real_DO_NOT_USE"})
    logged = logged.drop(columns=["y0_real_DO_NOT_USE_AS_LABEL", "y0_value_real_DO_NOT_USE"])
    return logged, oracle


def summarize(logged: pd.DataFrame, oracle: pd.DataFrame) -> dict[str, pd.DataFrame]:
    keep = logged["split"].isin(["train", "valid", "calib", "locked", "oot"])
    lg, orc = logged[keep], oracle[keep]
    # 1) ATE thật theo tập
    ate = orc.groupby("split")[[f"tau_cond_{a}" for a in ["a1", "a2", "a3"]] +
                               [f"uplift_profit_{a}" for a in ["a1", "a2", "a3"]]].mean().round(4)
    # 2) Nhiễu: so sánh chênh lệch thô (naive) với ATE thật trên train
    tr = lg[lg["split"] == "train"]
    otr = orc[orc["split"] == "train"]
    rows = []
    base = tr.loc[tr["A_logged"] == "a0", "y_obs"].mean()
    for a in ["a1", "a2", "a3"]:
        naive = tr.loc[tr["A_logged"] == a, "y_obs"].mean() - base
        rnd = tr[tr["is_random_arm"] == 1]
        rct = rnd.loc[rnd["A_logged"] == a, "y_obs"].mean() - rnd.loc[rnd["A_logged"] == "a0", "y_obs"].mean()
        true = (otr[f"y_{a}"] - otr["y_a0"]).mean()
        rows.append({"action": a, "true_ATE_purchase": round(true, 4), "naive_diff_all_logged": round(naive, 4),
                     "diff_random_arm_only": round(rct, 4), "naive_bias": round(naive - true, 4)})
    bias = pd.DataFrame(rows)
    # 3) Phân phối hành động & độ chồng lấp
    mix = (lg.groupby(["split", "A_logged"]).size().unstack(fill_value=0)
           .pipe(lambda d: d.div(d.sum(axis=1), axis=0)).round(3))
    ov = lg[[f"propensity_{a}" for a in ACTIONS]].describe(percentiles=[.01, .05, .5, .95, .99]).T.round(4)
    # 4) Tỷ trọng nhóm phản ứng (khung khái niệm Chương 4.5) theo hành động a1 – từ ĐÁP ÁN
    seg = []
    for a in ["a1", "a2", "a3"]:
        y0, ya = orc["y_a0"], orc[f"y_{a}"]
        seg.append({"action": a, "persuadables": ((y0 == 0) & (ya == 1)).mean(),
                    "sure_things": ((y0 == 1) & (ya == 1)).mean(), "lost_causes": ((y0 == 0) & (ya == 0)).mean(),
                    "sleeping_dogs": ((y0 == 1) & (ya == 0)).mean(),
                    "share_positive_profit_uplift": (orc[f"uplift_profit_{a}"] > 0).mean()})
    segs = pd.DataFrame(seg).round(4)
    return {"s1b_true_ate_by_split": ate.reset_index(), "s1b_confounding_check": bias,
            "s1b_action_mix_by_split": mix.reset_index(), "s1b_propensity_overlap": ov.reset_index(),
            "s1b_response_segments": segs}


def main() -> None:
    cfg = load_config()
    snap = pd.read_parquet(ROOT / cfg["paths"]["snapshots"])
    logged, oracle = generate(snap, cfg)
    logged.to_parquet(p("data/processed/logged.parquet"), index=False)
    oracle.to_parquet(p("data/processed/oracle.parquet"), index=False)
    tables = summarize(logged, oracle)
    files = ["data/processed/logged.parquet", "data/processed/oracle.parquet"]
    for name, t in tables.items():
        out = f"outputs/tables/{name}.csv"
        t.to_csv(p(out), index=False)
        files.append(out)
        print(f"\n== {name}\n{t.to_string(index=False)}")
    write_manifest("s1b", files, extra={"intervention_dgp": cfg["intervention_dgp"],
                                        "business": cfg["business"]})
    print("\nGĐ1b hoàn tất.")


if __name__ == "__main__":
    main()
