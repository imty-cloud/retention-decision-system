"""Phụ lục VIII + phần III: (1) bảng hợp đồng dữ liệu từ các schema pandera CÓ SẴN, (2) Graphic Abstract.

CHỈ ĐỌC: import schema từ src/, đọc CSV trong outputs/tables; chỉ GHI vào outputs/ (không đụng src/ configs/ models/ CLAUDE.md data/processed).
Mọi con số trong hình đọc từ CSV; khi chạy in bảng "con số → file nguồn" (và ghi ra outputs/tables/fig_graphic_abstract_numbers.csv).

Chạy:  python tools/build_appendix_extras.py [--contracts] [--figure]
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

import common
import s1_data
import s7_router as R
import s8_mlops as S8

TB, FG = ROOT / "outputs/tables", ROOT / "outputs/figures"


# ============================================================================ 1. hợp đồng dữ liệu
def contracts() -> list[tuple[str, str, object, str]]:
    cfg = common.load_config()
    return [
        ("Hợp đồng snapshot (GĐ1: đặc trưng point-in-time + nhãn)", "snapshot", s1_data.snapshot_schema(cfg), "src/s1_data.py:snapshot_schema"),
        ("Hợp đồng dữ liệu ghi log (GĐ1b/GĐ8: propensity, nhãn quan sát, cutoff, độ trễ nhãn)", "logged", S8.LOGGED_SCHEMA, "src/s8_mlops.py:LOGGED_SCHEMA"),
        ("Hợp đồng đầu vào của Router (GĐ7)", "dau_vao_router", R.INPUT_SCHEMA, "src/s7_router.py:_schemas → INPUT_SCHEMA"),
        ("Hợp đồng bản ghi quyết định – Router v1 (GĐ7, DECISION_SCHEMA)", "quyet_dinh_v1", R.DECISION_SCHEMA, "src/s7_router.py:_schemas → DECISION_SCHEMA"),
        ("Hợp đồng bản ghi quyết định – Router v2.1 (đang triển khai; mở rộng từ DECISION_SCHEMA)", "quyet_dinh_v21", R.DECISION_SCHEMA_V21,
         "src/s7_router.py:_schema_v21 → DECISION_SCHEMA_V21"),
    ]


def check_text(c) -> str:
    """Mô tả một Check bằng thông điệp lỗi (nếu có) hoặc tên + tham số."""
    if getattr(c, "error", None) and not str(c.error).startswith(str(c.name)):
        return str(c.error)
    st = getattr(c, "statistics", None) or {}
    st = {k: v for k, v in st.items() if v is not None and k != "raise_warning"}
    return f"{c.name}({', '.join(f'{k}={v}' for k, v in st.items())})" if st else str(c.name)


def contract_rows() -> pd.DataFrame:
    dd = pd.read_csv(TB / "s1_data_dictionary.csv").set_index("column").description.to_dict()
    rows = []
    for title, key, sch, src in contracts():
        for name, col in sch.columns.items():
            rows.append(dict(hop_dong=key, cot=name, kieu="(không ràng buộc kiểu)" if col.dtype is None else str(col.dtype),
                             nullable=bool(col.nullable), unique=bool(col.unique),
                             rang_buoc="; ".join(check_text(c) for c in col.checks), mo_ta=dd.get(name, ""), nguon=src))
        if sch.unique:
            rows.append(dict(hop_dong=key, cot="(cấp bảng)", kieu="", nullable="", unique="", rang_buoc=f"duy nhất theo ({', '.join(sch.unique)})", mo_ta="",
                             nguon=src))
        for c in sch.checks:
            rows.append(dict(hop_dong=key, cot="(cấp bảng)", kieu="", nullable="", unique="", rang_buoc=check_text(c), mo_ta="", nguon=src))
        rows.append(dict(hop_dong=key, cot="(cấp bảng)", kieu="", nullable="", unique="",
                         rang_buoc=f"strict={sch.strict}; coerce={sch.coerce}", mo_ta="cột thừa: " + ("cấm" if sch.strict is True else "cho phép"), nguon=src))
    return pd.DataFrame(rows)


def build_contracts() -> pd.DataFrame:
    df = contract_rows()
    df.to_csv(TB / "appx_data_contract.csv", index=False)
    L = ["# Phụ lục VIII – Hợp đồng dữ liệu (trích tự động từ schema pandera)\n",
         "> Sinh bởi `tools/build_appendix_extras.py` từ các đối tượng schema có sẵn trong `src/`; `mo_ta` lấy từ `outputs/tables/s1_data_dictionary.csv` "
         "(chỉ có cho cột của snapshot/logged; cột của Router để trống). Kiểm tra dạng hàm (lambda) hiển thị bằng thông điệp lỗi khai báo trong mã.\n"]
    for title, key, _, src in contracts():
        d = df[df.hop_dong == key].drop(columns=["hop_dong", "nguon"])
        d = d.assign(nullable=d.nullable.astype(str), unique=d.unique.astype(str))
        L.append(f"\n## {title}\n\n*Nguồn:* `{src}` · {int((d.cot != '(cấp bảng)').sum())} cột, {int((d.cot == '(cấp bảng)').sum())} ràng buộc cấp bảng\n")
        L.append("| " + " | ".join(d.columns) + " |\n|" + "|".join("---" for _ in d.columns) + "|")
        for r in d.itertuples(index=False):
            L.append("| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in r) + " |")
    (TB / "appx_data_contract.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return df


# ============================================================================ 2. Graphic Abstract
NUM: list[tuple[str, str, str]] = []


def n(label: str, value, src: str, fmt: str = "{}") -> str:
    s = fmt.format(value)
    NUM.append((label, s, src))
    return s


def csv(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(TB / name, **kw)


def box(ax, x, y, w, h, fc, ec, lw=0.8, r=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def wrap(s: str, w: int) -> str:
    return "\n".join(textwrap.wrap(s, w))


def build_figure() -> Path:
    NUM.clear()
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path"})
    sp = csv("s1_split_summary.csv").set_index("split")
    emb = csv("s1_embargo_check.csv")
    leak = csv("s1_leakage_tests.csv")
    m2 = csv("s2_metrics_valid.csv")
    m2 = m2[m2.phan_khuc == "toan_bo"].set_index("phuong_phap")
    pg = csv("s3_pfi_group.csv").set_index("nhom")
    st = csv("s3_shap_stability.csv")
    qn = csv("s4_qini.csv")
    qn = qn[(qn.tap == "valid+calib") & (qn.learner == "CausalForestDML")].set_index("hanh_dong")
    r5 = csv("s5_regret_summary.csv").set_index("phuong_phap")
    sc6 = csv("s6_scaling.csv")
    cd = csv("s6_calibration_discount_cost_summary.csv").set_index("loai_mu0")
    pv = csv("s7_policy_value_summary_v21.csv").set_index("chinh_sach")
    gs = csv("s8_locked_gate_summary_corrected.csv", keep_default_na=False)
    gs = gs[gs.quan_the.str.startswith("Toàn bộ")].iloc[0]
    g1 = csv("s8_locked_g1_contract.csv")
    g2 = csv("s8_locked_g2_calibration.csv").set_index("quan_the").loc["toan_bo"]
    g3 = csv("s8_locked_g3_qini.csv")
    g3 = g3[g3.quan_the == "toan_bo"]
    g4a = csv("s8_locked_g4a_comparison.csv")
    g4a = g4a[(g4a.quan_the == "toan_bo") & g4a.so_sanh.str.startswith("CHÍNH")].iloc[0]
    g4 = csv("s8_locked_g4_policy_value.csv")
    g4 = g4[(g4.quan_the == "toan_bo") & (g4.chinh_sach == "MILP qua Router v2.1")].iloc[0]
    g5 = csv("s8_locked_g5_constraint_audit.csv")
    g5 = g5[g5.chinh_sach == "MILP qua Router v2.1"]
    g6 = csv("s8_locked_g6_latency_percentiles.csv").set_index("khau")
    g7 = csv("s8_locked_g7_segments.csv")
    g7 = g7[(g7.quan_the == "toan_bo") & g7.so_sanh.str.startswith("CHÍNH")].set_index("phan_khuc")
    cm = csv("s8_oot_calibration_matured.csv")
    cm = cm[cm.phien_ban == "Current"].sort_values("ky")
    dg = csv("s8_oot_drift_guardrail.csv")
    dg = dg[dg.kich_ban == "không trôi"].set_index("chinh_sach")

    GREEN, RED, BLUE, GREY, INK = "#2e7d32", "#c62828", "#1f4e79", "#f2f4f7", "#222222"
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 117); ax.set_ylim(0, 83); ax.axis("off")
    ax.text(58.5, 80.2, "Hệ thống ra quyết định khép kín: phân bổ ngân sách khuyến mãi và điều phối chiến dịch giữ chân khách hàng đa kênh",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(58.5, 77.4, "Online Retail II · lớp can thiệp bán tổng hợp · chu trình 8 giai đoạn · Locked Test mở 1 lần", ha="center", va="center", fontsize=7.5, color="#555555")

    # ------------------------------------------------------------ TRÁI: dữ liệu
    X0, W0 = 2, 30
    ax.text(X0 + W0 / 2, 73.6, "DỮ LIỆU VÀ THIẾT KẾ", ha="center", fontsize=8.5, fontweight="bold", color=BLUE)
    box(ax, X0, 63.6, W0, 8.2, GREY, BLUE)
    ax.text(X0 + 1.2, 70.4, "Online Retail II (UCI, CC BY 4.0)", fontsize=7.3, fontweight="bold", va="center")
    ax.text(X0 + 1.2, 67.5, wrap("Đơn vị: khách × t0 (2 tuần/lần). Đặc trưng chỉ dùng sự kiện < t0 − 1 ngày; kết quả: mua lại trong 30 ngày.", 58),
            fontsize=6.2, va="center", linespacing=1.25)
    leak_s = n("kiểm thử rò rỉ đạt", f"{int(leak['pass'].sum())}/{len(leak)}", "s1_leakage_tests.csv")
    ax.text(X0 + 1.2, 64.5, f"{leak_s} kiểm thử rò rỉ đạt", fontsize=6.2, color=GREEN, va="center", fontweight="bold")
    box(ax, X0, 49.2, W0, 13.2, GREY, BLUE)
    ax.text(X0 + 1.2, 60.8, "Lớp can thiệp bán tổng hợp", fontsize=7.3, fontweight="bold", va="center")
    ax.text(X0 + 1.2, 56.0, wrap("Y(0) = hành vi mua lại THẬT; Y(a) sinh từ cơ chế đã biết cho 3 voucher (a1 10 %, a2 20 %, a3 gọi + 10 %). "
                                 "Log: 20 % ngẫu nhiên + 80 % luật RFM (gây nhiễu); đáp án chỉ dùng để đánh giá.", 55), fontsize=6.2, va="center", linespacing=1.25)
    ax.text(X0 + W0 / 2, 47.2, "Chia tập theo thời gian (số dòng · số kỳ)", ha="center", fontsize=7, fontweight="bold")
    cols = {"train": "#9ecae1", "valid": "#a1d99b", "calib": "#fdd0a2", "locked": "#f4a6a6", "oot": "#c7b3e6"}
    names = {"train": "train – huấn luyện", "valid": "valid – chọn mô hình", "calib": "calib – hiệu chỉnh", "locked": "locked – mở 1 lần",
             "oot": "oot – phát lại vận hành"}
    y = 44.0
    em_days = int(emb.gap_days.min())
    n("embargo nhỏ nhất (ngày)", em_days, "s1_embargo_check.csv")
    for i, s in enumerate(["train", "valid", "calib", "locked", "oot"]):
        rows = int(sp.loc[s, "rows"]); k = int(sp.loc[s, "n_t0"])
        n(f"{s}: dòng / kỳ", f"{rows:,} / {k}", "s1_split_summary.csv")
        ax.add_patch(Rectangle((X0, y - 5.6), W0, 5.6, fc=cols[s], ec="#555555", lw=0.6))
        ax.text(X0 + 1.2, y - 2.0, names[s], fontsize=6.6, fontweight="bold", va="center")
        ax.text(X0 + 1.2, y - 4.1, f"{rows:,} dòng · {k} kỳ · {sp.loc[s, 't0_first'][:7]} → {sp.loc[s, 't0_last'][:7]}", fontsize=6, va="center")
        y -= 5.6
        if i < 4:
            ax.text(X0 + W0 / 2, y - 1.0, f"embargo ≥ {em_days} ngày", fontsize=5.6, ha="center", va="center", color="#777777", style="italic")
            y -= 2.1
    n("locked: dòng khách thường / entity-holdout", f"{int(sp.loc['locked', 'rows']) - int(sp.loc['locked', 'entity_holdout_rows']):,} / {int(sp.loc['locked', 'entity_holdout_rows']):,}",
      "s1_split_summary.csv")
    ax.text(X0 + W0 / 2, 4.3, f"locked gồm {int(sp.loc['locked', 'entity_holdout_rows']):,} dòng entity-holdout (khách chưa từng thấy)", ha="center", fontsize=5.8, color="#555555")

    # ------------------------------------------------------------ GIỮA: 8 khối
    BX, BW = 41, 35
    ax.text(BX + BW / 2, 73.6, "CHU TRÌNH 8 GIAI ĐOẠN", ha="center", fontsize=8.5, fontweight="bold", color=BLUE)
    mu_lo, mu_hi = float(m2.loc["chua_hieu_chinh", "ece"]), float(m2.loc["platt", "ece"])
    pp = r5.loc["SPO+ tuyến tính", "regret_cn_hien_thuc_mean"], r5.loc["PTO tuyến tính (MSE)", "regret_cn_hien_thuc_mean"]
    big = sc6[sc6.n == sc6.n.max()].iloc[0]
    blocks = [
        ("1 · Dữ liệu và chống rò rỉ", "snapshot point-in-time, data contract, embargo, entity holdout",
         f"{leak_s} kiểm thử rò rỉ đạt; embargo ≥ {em_days} ngày"),
        ("2 · Dự báo, hiệu chỉnh, conformal", "LightGBM → Platt; khoảng conformal 90 % cho giá trị giỏ",
         f"ECE valid {n('ECE thô', mu_lo, 's2_metrics_valid.csv', '{:.3f}')} → {n('ECE Platt', mu_hi, 's2_metrics_valid.csv', '{:.3f}')} · ROC-AUC {n('ROC-AUC', m2.loc['platt', 'roc_auc'], 's2_metrics_valid.csv', '{:.3f}')}"),
        ("3 · Giải thích và độ tin cậy", "PFI, SHAP (log-odds), mã lý do top-3, ổn định bootstrap",
         f"PFI nhóm RFM {n('PFI RFM', pg.loc['RFM', 'pfi_mean'], 's3_pfi_group.csv', '{:.3f}')} · Spearman SHAP {n('Spearman SHAP', st.spearman.mean(), 's3_shap_stability.csv', '{:.2f}')}"),
        ("4 · Nhân quả: DML, CATE, Qini", "CausalForestDML; Qini trên nhánh ngẫu nhiên (valid+calib)",
         f"Qini a1 {n('Qini a1', qn.loc['a1', 'qini'], 's4_qini.csv', '{:.3f}')}; CI > 0 chỉ {int((qn.qini_ci_lo > 0).sum())}/3 hành động"),
        ("5 · PTO và SPO+", "hồi quy MSE → MILP so với SPO+ (lô 100 khách, 5 seed)",
         f"regret SPO+ {n('regret SPO+', pp[0], 's5_regret_summary.csv', '{:.3f}')} vs PTO {n('regret PTO', pp[1], 's5_regret_summary.csv', '{:.3f}')}: SPO+ không thắng"),
        ("6 · MILP toàn quy mô", "HiGHS: ngân sách mệnh giá, cuộc gọi, trần a2, độ phủ",
         f"{int(big.n):,} khách: {n('MILP 50k (s)', big.highspy_giay_giai, 's6_scaling.csv', '{:.1f}')} s · chi thực/kế hoạch {n('chi thực/kế hoạch (μ̂0 thô)', cd.iloc[0].thuc_tren_ke_hoach_ky_vong, 's6_calibration_discount_cost_summary.csv', '{:.2f}')} → {n('chi thực/kế hoạch (μ̂0 Platt)', cd.iloc[1].thuc_tren_ke_hoach_ky_vong, 's6_calibration_discount_cost_summary.csv', '{:.2f}')}"),
        ("7 · Action Router", "guardrail cứng, HITL mô phỏng, đối chứng 10 %, bù độ phủ",
         f"giá trị 7 kỳ: MILP {n('MILP qua Router', pv.loc['MILP qua Router v2.1', 'thuc_hien_thuc_cong_don'], 's7_policy_value_summary_v21.csv', '£{:,.0f}')} vs luật cũ {n('luật cũ qua Router', pv.loc['Luật cũ qua Router v2.1', 'thuc_hien_thuc_cong_don'], 's7_policy_value_summary_v21.csv', '£{:,.0f}')}"),
        ("8 · Triển khai và MLOps", "khóa Locked 1 lần, phát lại OOT, PSI · ECE · Action TVD",
         f"Locked: {n('gate đạt', sum(gs[k] == 'ĐẠT' for k in ['G1', 'G2', 'G3', 'G4a', 'G4b', 'G5', 'G6', 'G7']), 's8_locked_gate_summary_corrected.csv')}/8 gate đạt"),
    ]
    pitch, bh = 8.25, 6.7
    ys = []
    for i, (t, m, num) in enumerate(blocks):
        y0 = 70.0 - (i + 1) * pitch + (pitch - bh)
        ys.append(y0 + bh / 2)
        last = i == 7
        box(ax, BX, y0, BW, bh, "#fdecea" if last else "#eaf2fb", RED if last else BLUE, lw=1.0)
        ax.text(BX + 1.0, y0 + bh - 1.35, t, fontsize=7.3, fontweight="bold", va="center", color=INK)
        ax.text(BX + 1.0, y0 + bh / 2 - 0.1, wrap(m, 66), fontsize=5.9, va="center", color="#333333")
        ax.text(BX + 1.0, y0 + 1.3, num, fontsize=6.3, va="center", color=RED if last else BLUE, fontweight="bold")
        if i < 7:
            ax.annotate("", xy=(BX + BW / 2, y0 - (pitch - bh) + 0.15), xytext=(BX + BW / 2, y0 - 0.05),
                        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.9, mutation_scale=7))
    # phản hồi trái: GĐ8 → GĐ2 (Challenger)
    xl = BX - 2.2
    ax.plot([BX, xl, xl], [ys[7], ys[7], ys[1]], color=GREEN, lw=1.2)
    ax.annotate("", xy=(BX, ys[1]), xytext=(xl, ys[1]), arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.2, mutation_scale=8))
    ax.text(xl - 1.15, (ys[7] + ys[1]) / 2, "Phản hồi: Challenger = Platt hiệu chỉnh lại theo kỳ (chỉ nhãn đã trưởng thành)", rotation=90, ha="center", va="center", fontsize=5.8, color=GREEN)
    # phản hồi phải: GĐ8 → MILP/Router (playbook)
    xr = BX + BW + 2.2
    ax.plot([BX + BW, xr, xr], [ys[7], ys[7], ys[5]], color="#b26a00", lw=1.2)
    ax.annotate("", xy=(BX + BW, ys[5]), xytext=(xr, ys[5]), arrowprops=dict(arrowstyle="-|>", color="#b26a00", lw=1.2, mutation_scale=8))
    ax.annotate("", xy=(BX + BW, ys[6]), xytext=(xr, ys[6]), arrowprops=dict(arrowstyle="-|>", color="#b26a00", lw=1.2, mutation_scale=8))
    ax.text(xr + 1.15, (ys[7] + ys[5]) / 2, "Playbook: PSI, ECE, Action TVD → quay lui luật cũ", rotation=90, ha="center", va="center", fontsize=5.8, color="#b26a00")

    # ------------------------------------------------------------ PHẢI: kết quả
    RX, RW = 84.5, 30.5
    ax.text(RX + RW / 2, 73.6, "KẾT QUẢ", ha="center", fontsize=8.5, fontweight="bold", color=BLUE)
    ax.text(RX + RW / 2, 71.3, f"Release Gates trên Locked ({int(sp.loc['locked', 'rows']):,} dòng, mở 1 lần)", ha="center", fontsize=6.9, fontweight="bold")
    g2v = n("G2 ECE", g2.ece_hieu_chinh, "s8_locked_g2_calibration.csv", "{:.4f}")
    ng3 = int(g3.ci_duong.sum())
    rows_g = [
        ("G1", gs["G1"], f"hợp đồng dữ liệu {n('G1', f'{int(g1.dat.sum())}/{len(g1)}', 's8_locked_g1_contract.csv')}"),
        ("G2", gs["G2"], f"ECE {g2v} (≤ 0.03)"),
        ("G3", gs["G3"], f"Qini CI > 0: {n('G3', f'{ng3}/3', 's8_locked_g3_qini.csv')} hành động (≥ 2)"),
        ("G4a", gs["G4a"], f"chênh {n('G4a chênh', g4a.chenh_lech_tuyet_doi, 's8_locked_g4a_comparison.csv', '£{:,.0f}')} (cần ≥ {n('G4a ngưỡng', g4a.nguong_010_abs_doi_chung, 's8_locked_g4a_comparison.csv', '£{:,.0f}')})"),
        ("G4b", gs["G4b"], f"regret {n('G4b', g4.regret_A, 's8_locked_g4_policy_value.csv', '{:.3f}')} (≤ 0.35)"),
        ("G5", gs["G5"], f"{n('G5 vi phạm', int(g5.n_vi_pham.sum()), 's8_locked_g5_constraint_audit.csv')} vi phạm ràng buộc"),
        ("G6", gs["G6"], f"P95 {n('G6 P95', g6.loc['tong', 'P95'], 's8_locked_g6_latency_percentiles.csv', '{:.2f}')} s (≤ 30 s)"),
        ("G7", gs["G7"], f"UK {n('G7 UK', g7.loc['UK', 'chenh_tuong_doi'] * 100, 's8_locked_g7_segments.csv', '{:+.1f}')} % · non-UK {n('G7 non-UK', g7.loc['non-UK', 'chenh_tuong_doi'] * 100, 's8_locked_g7_segments.csv', '{:+.1f}')} %"),
    ]
    yy = 69.3
    for g, res, txt in rows_g:
        ok = res == "ĐẠT"
        box(ax, RX, yy - 3.7, RW, 3.9, "#ffffff", "#bbbbbb", lw=0.5, r=0.5)
        box(ax, RX + 0.5, yy - 3.2, 5.0, 2.9, GREEN if ok else RED, GREEN if ok else RED, lw=0.5, r=0.5)
        ax.text(RX + 3.0, yy - 1.75, g, color="white", fontsize=6.6, fontweight="bold", ha="center", va="center")
        ax.text(RX + 6.3, yy - 1.0, res, fontsize=6.4, fontweight="bold", color=GREEN if ok else RED, va="center")
        ax.text(RX + 6.3, yy - 2.7, txt, fontsize=5.7, va="center", color="#333333")
        yy -= 4.3
    ok_all = all(gs[k] == "ĐẠT" for k in ["G1", "G2", "G3", "G4a", "G4b", "G5", "G6", "G7"])
    box(ax, RX, yy - 5.6, RW, 6.0, "#fdecea" if not ok_all else "#e8f5e9", RED if not ok_all else GREEN, lw=1.4)
    ax.text(RX + RW / 2, yy - 1.5, "KHÔNG PHÁT HÀNH" if not ok_all else "ĐỦ ĐIỀU KIỆN PHÁT HÀNH", ha="center", va="center", fontsize=8.3, fontweight="bold",
            color=RED if not ok_all else GREEN)
    ax.text(RX + RW / 2, yy - 3.9, wrap(f"G4b trượt: regret {g4.regret_A:.3f} > 0.35; ngay cả chính sách biết kỳ vọng thật cũng có regret {n('regret đáp án B (lô 100)', r5.loc['Đáp án (kỳ vọng theo U)', 'regret_cn_hien_thuc_mean'], 's5_regret_summary.csv', '{:.2f}')}. G3 đạt trái dự báo 4–7 %", 60), ha="center", va="center",
            fontsize=5.7, color="#333333", linespacing=1.2)
    yy -= 7.4
    ax.text(RX + RW / 2, yy, "Hai phát hiện khi phát lại OOT (5 kỳ)", ha="center", fontsize=6.9, fontweight="bold")
    yy -= 1.4
    e0, e1 = cm.ece.iloc[0], cm.ece.iloc[-1]
    box(ax, RX, yy - 8.0, RW, 8.0, GREY, "#999999", lw=0.6)
    ax.text(RX + 1.0, yy - 1.5, "① Hiệu chỉnh trôi theo mùa", fontsize=6.6, fontweight="bold", va="center")
    ax.text(RX + 1.0, yy - 4.1, f"ECE của Current {n('ECE kỳ đầu', e0, 's8_oot_calibration_matured.csv', '{:.3f}')} → {n('ECE kỳ cuối', e1, 's8_oot_calibration_matured.csv', '{:.3f}')} (báo động ≥ 0.05)", fontsize=6.1,
            va="center", color=RED, fontweight="bold")
    ax.text(RX + 1.0, yy - 6.3, wrap("→ chuyển sang Challenger (Platt theo kỳ)", 58), fontsize=5.7, va="center", color="#333333")
    yy -= 8.9
    v_no, v_co = int(dg.loc["KHÔNG guardrail: MILP thuần (miền v1)", "vi_pham_quy_dinh"]), int(dg.loc["CÓ guardrail: MILP qua Router v2.1", "vi_pham_quy_dinh"])
    box(ax, RX, yy - 8.0, RW, 8.0, GREY, "#999999", lw=0.6)
    ax.text(RX + 1.0, yy - 1.5, "② Guardrail chặn vi phạm quy định", fontsize=6.6, fontweight="bold", va="center")
    ax.text(RX + 1.0, yy - 4.1, f"Vi phạm: {n('vi phạm không guardrail', v_no, 's8_oot_drift_guardrail.csv')} (không guardrail) so với {n('vi phạm có guardrail', v_co, 's8_oot_drift_guardrail.csv')} (có guardrail)",
            fontsize=6.1, va="center", color=GREEN, fontweight="bold")
    loss = dg.loc["CÓ guardrail: MILP qua Router v2.1", "ton_that_tranh_duoc_hien_thuc"]
    ax.text(RX + 1.0, yy - 6.3, wrap(f"đổi lại giảm giá trị hiện thực {n('mất giá trị', abs(loss), 's8_oot_drift_guardrail.csv', '£{:,.0f}')} (không trôi)", 58), fontsize=5.7, va="center", color="#333333")

    ax.text(58.5, 1.6, "Mọi con số đọc từ outputs/tables/*.csv (bảng nguồn: fig_graphic_abstract_numbers.csv). Bán tổng hợp: Y(a) mô phỏng; HITL là quy tắc mô phỏng.",
            ha="center", fontsize=5.8, color="#666666")
    FG.mkdir(exist_ok=True)
    png = FG / "fig_graphic_abstract.png"
    fig.savefig(png, dpi=300)
    fig.savefig(FG / "fig_graphic_abstract.svg")
    plt.close(fig)
    num = pd.DataFrame(NUM, columns=["con_so", "gia_tri", "file_nguon"])
    num.to_csv(TB / "fig_graphic_abstract_numbers.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    print(num.to_string(index=False))
    return png


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contracts", action="store_true")
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args()
    every = not (a.contracts or a.figure)
    if every or a.contracts:
        df = build_contracts()
        print(df.groupby("hop_dong").size().rename("so_dong").to_string())
    if every or a.figure:
        print("→", build_figure().relative_to(ROOT))


if __name__ == "__main__":
    main()
