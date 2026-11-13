"""GĐ4b – Phân tích độ đủ cỡ mẫu (power) cho cổng G3 trên locked – DỰ BÁO ĐĂNG KÝ TRƯỚC.

G3 (config release_gates): số hành động có hệ số Qini > 0 VỚI CI bootstrap 95% > 0 (nhánh ngẫu nhiên) ≥ 2.

Cách làm (không mở locked, không chọn lại learner, không đổi ngưỡng):
  - Chỉ đọc CỘT `split` và `is_entity_holdout` của logged.parquet để đếm số dòng locked; KHÔNG đọc nhãn/hành động locked.
  - Cỡ mẫu nhánh ngẫu nhiên kỳ vọng ở locked theo THIẾT KẾ: n_cặp = N_locked × random_arm_pct% × 2/4
    (nhánh ngẫu nhiên chia đều 4 hành động; mỗi phép so sánh Qini dùng a0 ∪ a).
  - SE Qini ở valid+calib suy từ CI bootstrap 95%: se = (hi − lo) / (2·1.96); ngoại suy se_L = se·sqrt(n_vc / n_L).
  - Giả định Q̂_L ~ N(q, se_L²) ⇒ P(CI dưới > 0) = Φ(q/se_L − 1.96).
    P(G3) = P(≥ 2/3 hành động đạt). Độc lập (ρ=0) tính chính xác; ρ=0.3 (ba phép so sánh dùng chung nhóm a0) bằng Monte Carlo.
  - Kịch bản q (tác động thật của bộ xếp hạng ở locked):
      S1 "calib riêng" – CHÍNH: calib không dùng để chọn learner ⇒ không thiên lệch do chọn mô hình.
      S2 "valid+calib gộp" – lạc quan (chứa phần valid đã dùng để chọn learner).
      S3 "train ngoài fold" – tham khảo (cùng giai đoạn với dữ liệu học).
  - Báo cáo riêng: toàn bộ locked, khách thường, khách entity-holdout (theo CLAUDE.md).
Hạn chế: locked là tháng 7–8 (mùa khác valid/calib) – độ lớn tác động có thể khác; xấp xỉ chuẩn cho Qini bootstrap.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import itertools

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

import common

CFG = common.load_config()
SEED = CFG["project"]["seed"]
TABLES, FIGS = Path(CFG["paths"]["tables"]), Path(CFG["paths"]["figures"])
Z = 1.96
TREATED = ["a1", "a2", "a3"]
SCEN = {"S1 calib riêng (chính, không thiên lệch chọn)": "calib",
        "S2 valid+calib gộp (lạc quan)": "valid+calib",
        "S3 train ngoài fold (tham khảo)": "train (ngoài fold)"}


def p_at_least_2(p: np.ndarray, rho: float, rng, n_mc: int = 200_000) -> float:
    """P(ít nhất 2 trong 3 hành động đạt). rho=0: công thức đóng; rho>0: Monte Carlo trên thang z."""
    if rho == 0:
        return float(sum(np.prod([p[i] if i in s else 1 - p[i] for i in range(3)])
                         for k in (2, 3) for s in itertools.combinations(range(3), k)))
    cov = np.full((3, 3), rho) + (1 - rho) * np.eye(3)
    zz = rng.multivariate_normal(np.zeros(3), cov, size=n_mc)
    thr = norm.ppf(1 - np.clip(p, 1e-12, 1 - 1e-12))      # đạt ⇔ z > ngưỡng (biên đúng bằng p)
    return float(((zz > thr).sum(axis=1) >= 2).mean())


def main() -> None:
    rng = np.random.default_rng(SEED)
    files: list[str] = []
    q = pd.read_csv(TABLES / "s4_qini.csv")
    chosen = q.loc[q.duoc_chon, "learner"].iloc[0]
    q = q[q.learner == chosen]
    vc = q[q.tap == "valid+calib"].set_index("hanh_dong")
    se_vc = (vc.qini_ci_hi - vc.qini_ci_lo) / (2 * Z)

    meta = pd.read_parquet(common.p("data/processed/logged.parquet"), columns=["split", "is_entity_holdout"])
    lk = meta[meta.split == "locked"]
    frac = CFG["intervention_dgp"]["random_arm_pct"] / 100 * 2 / 4
    pops = {"locked toàn bộ": len(lk), "locked khách thường": int((lk.is_entity_holdout == 0).sum()),
            "locked entity-holdout": int((lk.is_entity_holdout == 1).sum())}

    rows, summ = [], []
    for pop, N in pops.items():
        n_L = N * frac
        for scen, tap in SCEN.items():
            qs = q[q.tap == tap].set_index("hanh_dong").qini
            ps = []
            for a in TREATED:
                se_L = se_vc[a] * np.sqrt(vc.loc[a, "n"] / n_L)
                p_pass = norm.cdf(qs[a] / se_L - Z)
                n80 = vc.loc[a, "n"] * (se_vc[a] * (Z + norm.ppf(0.8)) / qs[a]) ** 2 if qs[a] > 0 else np.inf
                ps.append(p_pass)
                rows.append({"quan_the": pop, "N_dong_locked": N, "n_cap_ngau_nhien_ky_vong": round(n_L),
                             "kich_ban": scen, "hanh_dong": a, "learner": chosen, "qini_gia_dinh": qs[a],
                             "se_valid_calib": se_vc[a], "n_cap_valid_calib": int(vc.loc[a, "n"]), "se_locked_du_bao": se_L,
                             "do_rong_CI95_du_bao": 2 * Z * se_L, "P_qini_duong": norm.cdf(qs[a] / se_L),
                             "P_CI_duoi_duong": p_pass, "n_cap_can_cho_power_80": n80,
                             "boi_so_n_locked_can": n80 / n_L})
            ps = np.array(ps)
            summ.append({"quan_the": pop, "kich_ban": scen, "learner": chosen,
                         "P_G3_dat_rho0": p_at_least_2(ps, 0, rng), "P_G3_dat_rho0.3": p_at_least_2(ps, 0.3, rng),
                         "so_hanh_dong_ky_vong_dat": ps.sum()})
    det = pd.DataFrame(rows)
    sm = pd.DataFrame(summ)
    det.to_csv(common.p(TABLES / "s4_power_g3.csv"), index=False)
    sm.to_csv(common.p(TABLES / "s4_power_g3_summary.csv"), index=False)
    files += [str(TABLES / "s4_power_g3.csv"), str(TABLES / "s4_power_g3_summary.csv")]
    pd.set_option("display.width", 250)
    print(det[["quan_the", "kich_ban", "hanh_dong", "qini_gia_dinh", "se_locked_du_bao", "P_CI_duoi_duong",
               "boi_so_n_locked_can"]].round(3).to_string())
    print(sm.round(3).to_string())

    # hình: P(G3) theo cỡ mẫu nhánh ngẫu nhiên mỗi cặp
    grid = np.linspace(500, 12000, 80)
    fig, ax = plt.subplots(figsize=(8, 5))
    for scen, tap in SCEN.items():
        qs = q[q.tap == tap].set_index("hanh_dong").qini
        pg = [p_at_least_2(np.array([norm.cdf(qs[a] / (se_vc[a] * np.sqrt(vc.loc[a, "n"] / n)) - Z)
                                     for a in TREATED]), 0, rng) for n in grid]
        ax.plot(grid, pg, label=scen)
    for pop, N in pops.items():
        ax.axvline(N * frac, ls=":", color="gray")
        ax.text(N * frac, 0.97, pop, rotation=90, va="top", ha="right", fontsize=8)
    ax.axvline(vc.n.mean(), ls="--", color="k", lw=1)
    ax.text(vc.n.mean(), 0.5, "valid+calib hiện có", rotation=90, va="center", ha="right", fontsize=8)
    ax.set_xlabel("Số dòng nhánh ngẫu nhiên cho mỗi cặp (a0 ∪ a)")
    ax.set_ylabel("P(G3 đạt) – giả định độc lập")
    ax.set_title(f"Dự báo xác suất đạt G3 theo cỡ mẫu ({chosen}) – đăng ký trước khi mở locked")
    ax.set_ylim(0, 1); ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(common.p(FIGS / "fig_s4b_power_g3.png"), dpi=160); plt.close(fig)
    files.append(str(FIGS / "fig_s4b_power_g3.png"))
    common.write_manifest("s4b", files, extra={"learner": chosen, "random_arm_frac_per_pair": frac, "pops": pops})


if __name__ == "__main__":
    main()
