"""Đóng gói sau GĐ8 (CHỈ ĐỌC kết quả có sẵn; không chạy lại s1–s8, không sửa src/ configs/ models/ CLAUDE.md data/processed).

Sinh:
  outputs/tables/s8_sha256_appendix.csv      phụ lục SHA-256 (lấy từ s8_freeze_manifest.json)
  outputs/tables/summary_for_essay.md        ánh xạ kết quả → mục tiểu luận → CLO → file nguồn (mọi số đọc từ CSV)
  notebooks/PTDLNC_full_pipeline.ipynb       notebook GĐ1 → GĐ8 + s9 (mặc định ĐỌC kết quả có sẵn; RUN_FULL = False)

Chạy:  python tools/build_packaging.py [--appendix] [--summary] [--notebook]   (mặc định: cả ba)
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TB = ROOT / "outputs/tables"
CLO = "[CLO: điền]"      # repo CHƯA có danh mục CLO của học phần ⇒ không tự suy diễn; điền theo đề cương


def tab(name: str) -> pd.DataFrame:
    return pd.read_csv(TB / name)


def sha(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def utc_mtime(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def val(df: pd.DataFrame, where: dict, col: str):
    m = df
    for k, v in where.items():
        m = m[m[k] == v]
    assert len(m) == 1, (where, len(m))
    return m[col].iloc[0]


def f(x, nd=3):
    return f"{x:.{nd}f}"


def gbp(x, nd=0):
    return f"−£{abs(x):,.{nd}f}" if x < 0 else f"£{x:,.{nd}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in r) + " |")
    return "\n".join(out)


# ============================================================================ 1. phụ lục SHA-256
def build_appendix() -> Path:
    man = json.loads((ROOT / "outputs/manifests/s8_freeze_manifest.json").read_text(encoding="utf-8"))
    rows = [{"nhom": "mã/mô hình/config/CLAUDE.md (trong phạm vi băm)", **e, "ghi_chu": ""} for e in man["files"]]
    rows += [{"nhom": "dữ liệu data/processed (trong phạm vi băm, băm theo byte)", **e, "ghi_chu": ""} for e in man["data_files"]]
    rows += [
        {"nhom": "TỔNG HỢP", "file": "(aggregate) mã/mô hình/config/CLAUDE.md", "bytes": "", "sha256": man["aggregate_sha256"],
         "ghi_chu": f"s8_freeze_manifest.json, tạo lúc {man['created_utc']}; = SHA-256 của các dòng 'file:sha256' nối bằng \\n"},
        {"nhom": "TỔNG HỢP", "file": "(aggregate) dữ liệu data/processed", "bytes": "", "sha256": man["data_aggregate_sha256"],
         "ghi_chu": "cùng cách tính; loại decisions_s8_* (đầu ra của chính s8)"}]
    lock = ROOT / "outputs/manifests/s8_locked_opened.lock"
    content = json.dumps(json.loads(lock.read_text(encoding="utf-8")), ensure_ascii=False, sort_keys=True)
    rows.append({"nhom": "FILE KHÓA", "file": "outputs/manifests/s8_locked_opened.lock", "bytes": lock.stat().st_size, "sha256": sha(lock),
                 "ghi_chu": "NỘI DUNG: " + content + ". freeze_aggregate_sha256 trong khóa = hash tổng hợp mã ở trên"})
    s1 = json.loads((ROOT / "outputs/manifests/s1_manifest.json").read_text(encoding="utf-8"))
    for e in s1["files"]:
        if e["file"].startswith("data/raw/"):
            rows.append({"nhom": "dữ liệu thô (NGOÀI phạm vi băm; s1_manifest.json)", **e,
                         "ghi_chu": "Online Retail II – CC BY 4.0"})
    h = tab("s9_data_sha256.csv").iloc[0]
    rows.append({"nhom": "dữ liệu thô (NGOÀI phạm vi băm; s9_data_sha256.csv)", "file": h["file"], "bytes": h["bytes"], "sha256": h["sha256"],
                 "ghi_chu": "Hillstrom – KHÔNG nêu giấy phép; không đưa vào repo, tải bằng sklift.datasets.fetch_hillstrom"})
    for t in ("tests/test_s8b.py", "tests/test_s8.py"):
        p = ROOT / t
        rows.append({"nhom": "SỬA SAU ĐÓNG BĂNG (tests/ nằm NGOÀI phạm vi băm)", "file": t, "bytes": p.stat().st_size, "sha256": sha(p),
                     "ghi_chu": {"tests/test_s8b.py": f"sửa lần cuối {utc_mtime(p)} – thêm test_real_oot_path_preflight (diễn tập nhánh OOT 'real' bằng "
                                                       "dữ liệu thay thế, TRƯỚC khi mở locked; freeze 14:31:07 UTC, khóa locked 14:40:19 UTC). Không đổi mã trong src/.",
                                 "tests/test_s8.py": f"sửa lần cuối {utc_mtime(p)} – sau khi mở locked: hai test từng khẳng định 'khóa chưa tồn tại' "
                                                     "đổi thành 'khóa và s8_locked_gates.csv THẬT không bị thay đổi'. Không đổi mã trong src/."}[t]})
    out = TB / "s8_sha256_appendix.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


# ============================================================================ 2. summary_for_essay.md
def build_summary() -> Path:
    L: list[str] = []
    A = L.append
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    A("# Ánh xạ kết quả → tiểu luận (PTDLNC – hệ thống ra quyết định giữ chân khách hàng đa kênh)\n")
    A(f"> Sinh tự động ngày {today} bởi `tools/build_packaging.py` từ các CSV trong `outputs/tables/`; **mọi con số đều được đọc trực tiếp từ file nguồn ghi ở cột "
      "cuối** (không gõ tay). Bản này chỉ tổng hợp kết quả đã có; không chạy lại pipeline.\n")
    A("**Cần người dùng điền / xác nhận:**\n")
    A(f"- Cột CLO ghi `{CLO}`: repo không có danh mục CLO của học phần nên không tự suy diễn. Điền theo đề cương.")
    A("- Đánh số mục I–VI là suy từ yêu cầu đóng gói (IV = kết quả, V = thảo luận, VI = playbook) và CLAUDE.md §5 (III = thiết kế dữ liệu, VIII = phụ lục). "
      "Mục I, II, VII chưa có nội dung kết quả cụ thể nên không liệt kê.\n")
    A("**Kết luận tổng quát:** hệ thống **KHÔNG đạt đủ cổng phát hành** trên Locked Test vì G4b (regret chuẩn hóa) trượt; bảy tiêu chí còn lại đạt ở mức toàn bộ "
      "(xem IV.6). Entity-holdout (chỉ mô tả) trượt G2, G3, G4a, G7. Xem thêm mục \"Công khai và sai lệch\" ở cuối.\n")

    # ------------------------------------------------------------------ III
    A("\n## III. Thiết kế bài toán và dữ liệu\n")
    sp = tab("s1_split_summary.csv").set_index("split")
    lk = tab("s1_leakage_tests.csv")
    em = tab("s1_embargo_check.csv")
    r = [("Quy mô các tập (dòng khách × t0 / số kỳ): " + "; ".join(f"{s} {int(sp.loc[s, 'rows']):,} / {int(sp.loc[s, 'n_t0'])}" for s in
                                                                   ("train", "valid", "calib", "locked", "oot")), "III", "s1_split_summary.csv"),
         (f"Kiểm thử rò rỉ: {int(lk['pass'].sum())}/{len(lk)} phép thử đạt (point-in-time, bất biến khi nhiễu tương lai, tính lại nhãn)", "III", "s1_leakage_tests.csv"),
         (f"Embargo giữa các khối ≥ độ trễ nhãn 30 ngày: khoảng cách nhỏ nhất {int(em.gap_days.min())} ngày, {int(em['pass'].sum())}/{len(em)} đạt", "III", "s1_embargo_check.csv"),
         ("Tỷ lệ mua lại gốc trôi theo mùa (mọi dòng): " + " → ".join(f"{s} {sp.loc[s, 'base_rate']:.3f}" for s in ("train", "valid", "calib", "locked", "oot")), "III/V",
          "s1_split_summary.csv; fig_s1_base_rate_by_t0.png"),
         ("Sơ đồ chia tập theo thời gian", "III (Graphic Abstract)", "fig_s1_split_timeline.png"),
         ("Lớp can thiệp bán tổng hợp: so thô lệch khỏi ATE thật (train, mua lại) a1 " + f(val(tab('s1b_confounding_check.csv'), {'action': 'a1'}, 'naive_bias'), 4) +
          ", a2 " + f(val(tab('s1b_confounding_check.csv'), {'action': 'a2'}, 'naive_bias'), 4) + ", a3 " +
          f(val(tab('s1b_confounding_check.csv'), {'action': 'a3'}, 'naive_bias'), 4), "III/V", "s1b_confounding_check.csv")]
    A(table_rows(r))

    # ------------------------------------------------------------------ IV
    A("\n## IV. Kết quả\n")
    A("### IV.1 Dự báo nền và hiệu chỉnh xác suất (GĐ2)\n")
    m = tab("s2_metrics_valid.csv")
    raw_, pl = (m[(m.phan_khuc == "toan_bo") & (m.phuong_phap == k)].iloc[0] for k in ("chua_hieu_chinh", "platt"))
    acc = tab("s2_acceptance.csv")
    cf = tab("s2_basket_conformal_valid.csv").set_index("nhom")
    g2 = tab("s8_locked_g2_calibration.csv").set_index("quan_the")
    A(table_rows([
        (f"Valid: ROC-AUC {f(pl.roc_auc)}, PR-AUC {f(pl.pr_auc)}; Brier {f(raw_.brier, 4)} → {f(pl.brier, 4)} sau Platt; ECE {f(raw_.ece, 4)} → {f(pl.ece, 4)}", "IV", "s2_metrics_valid.csv"),
        (f"Tiêu chí chấp nhận GĐ2: ECE ≤ 0.03 trên valid **không đạt** ({f(acc.iloc[0].gia_tri, 4)}); Brier không tệ hơn: đạt; coverage conformal Ŝ {f(cf.loc['toan_bo', 'coverage'], 4)} "
         "(mục tiêu 0.90, trong [0.85, 0.95])", "IV", "s2_acceptance.csv; s2_basket_conformal_valid.csv"),
        (f"Locked (G2, dòng a0, n = {int(g2.loc['toan_bo', 'n_a0']):,}): ECE {f(g2.loc['toan_bo', 'ece_tho'], 4)} (thô) → {f(g2.loc['toan_bo', 'ece_hieu_chinh'], 4)} (Platt); "
         f"Brier {f(g2.loc['toan_bo', 'brier_tho'], 4)} → {f(g2.loc['toan_bo', 'brier_hieu_chinh'], 4)}", "IV", "s8_locked_g2_calibration.csv"),
        ("Biểu đồ tin cậy trước/sau hiệu chỉnh; khoảng conformal của giá trị giỏ", "IV", "fig_s2_reliability_valid.png; fig_s2_basket_conformal.png")]))

    A("\n### IV.2 Giải thích và độ tin cậy (GĐ3)\n")
    pg = tab("s3_pfi_group.csv").set_index("nhom")
    fa = tab("s3_faithfulness_deletion.csv")
    st = tab("s3_shap_stability.csv")
    A(table_rows([
        (f"PFI theo nhóm (valid, neg log-loss): RFM {f(pg.loc['RFM', 'pfi_mean'], 4)}, Nhịp mua {f(pg.loc['Nhịp mua', 'pfi_mean'], 4)}, Hủy đơn {f(pg.loc['Hủy đơn', 'pfi_mean'], 4)}, "
         f"Lịch {f(pg.loc['Lịch', 'pfi_mean'], 4)}, Quốc gia {f(pg.loc['Quốc gia', 'pfi_mean'], 4)}", "IV", "s3_pfi_group.csv; fig_s3_pfi.png"),
        (f"Ổn định SHAP qua 5 lần bootstrap: Spearman trung bình {f(st.spearman.mean(), 3)} (nhỏ nhất {f(st.spearman.min(), 3)})", "IV", "s3_shap_stability.csv"),
        (f"Faithfulness: xóa top-1 theo SHAP làm lệch logit {f(val(fa, {'k': 1, 'cach_xoa': 'top_k_shap'}, 'do_lech_logit_tuyet_doi_TB'), 3)} so với {f(val(fa, {'k': 1, 'cach_xoa': 'ngau_nhien'}, 'do_lech_logit_tuyet_doi_TB'), 3)} khi xóa ngẫu nhiên",
         "IV", "s3_faithfulness_deletion.csv"),
        ("SHAP/PFI **không phải nhân quả** (nhắc trong báo cáo); mã lý do top-3 cho Router", "IV/V", "fig_s3_shap_beeswarm.png; reasons_s3.parquet")]))

    A("\n### IV.3 Nhân quả và uplift (GĐ4)\n")
    s4a = tab("s4_acceptance.csv").set_index("tieu_chi")
    ate = tab("s4_ate_vs_truth.csv")
    qn = tab("s4_qini.csv")
    q = qn[(qn.tap == "valid+calib") & (qn.learner == "CausalForestDML")].set_index("hanh_dong")
    A(table_rows([
        ("So thô lệch ATE thật; IPW và DML khử nhiễu: ATE mua lại a1 thật " + f(val(ate, {'tap': 'train', 'ket_qua': 'purchase', 'hanh_dong': 'a1', 'phuong_phap': 'Naive (chênh lệch thô)'}, 'ate_that'), 4) +
         ", thô " + f(val(ate, {'tap': 'train', 'ket_qua': 'purchase', 'hanh_dong': 'a1', 'phuong_phap': 'Naive (chênh lệch thô)'}, 'uoc_luong'), 4) + ", IPW " +
         f(val(ate, {'tap': 'train', 'ket_qua': 'purchase', 'hanh_dong': 'a1', 'phuong_phap': 'IPW (propensity ghi log)'}, 'uoc_luong'), 4), "IV", "s4_ate_vs_truth.csv; fig_s4_ate_forest.png"),
        (f"CI DML phủ ATE thật: {s4a.iloc[0].gia_tri} (APOS), {s4a.iloc[1].gia_tri} (IRM); placebo (xáo A) CI chứa 0: {s4a.iloc[4].gia_tri}", "IV", "s4_acceptance.csv; s4_placebo.csv"),
        ("Qini CausalForestDML trên nhánh ngẫu nhiên valid+calib: " + ", ".join(f"{a} {f(q.loc[a, 'qini'])} [{f(q.loc[a, 'qini_ci_lo'])}; {f(q.loc[a, 'qini_ci_hi'])}]" for a in ("a1", "a2", "a3")) +
         f"; Qini > 0 cho {s4a.iloc[2].gia_tri}, CI > 0 cho {s4a.iloc[3].gia_tri}", "IV", "s4_qini.csv; fig_s4_qini_curves.png"),
        ("Dự báo power cho G3 (đăng ký trước)", "IV/V", "s4_power_g3_summary.csv; fig_s4b_power_g3.png")]))

    A("\n### IV.4 Tối ưu hóa: SPO+, MILP (GĐ5–GĐ6)\n")
    r5 = tab("s5_regret_summary.csv").set_index("phuong_phap")
    pp = tab("s5_spo_vs_pto_paired.csv").iloc[0]
    m6 = tab("s6_main_summary.csv").set_index("phuong_phap")
    cd = tab("s6_calibration_discount_cost_summary.csv").set_index("loai_mu0")
    sc = tab("s6_scaling.csv")
    big = sc[sc.n == sc.n.max()].iloc[0]
    A(table_rows([
        ("Regret chuẩn hóa hiện thực (lô 100 khách, calib, 5 seed): " + "; ".join(
            f"{k} {f(r5.loc[k, 'regret_cn_hien_thuc_mean'], 3)} ± {f(r5.loc[k, 'regret_cn_hien_thuc_sd'], 3)}" for k in
            ("Luật RFM cũ", "Greedy (V̂/mệnh giá)", "MILP trên V̂ (GĐ4)", "PTO tuyến tính (MSE)", "PTO LightGBM (MSE)", "SPO+ tuyến tính")), "IV", "s5_regret_summary.csv; fig_s5_regret_by_method.png"),
        (f"SPO+ − PTO tuyến tính (regret hiện thực): {pp.chenh_lech_TB:+.4f} ± {pp.sd_qua_seed:.4f}; SPO+ tốt hơn ở {int(pp.so_seed_SPO_tot_hon)}/5 seed – **SPO+ không thắng PTO** (giữ nguyên kết quả)",
         "IV/V", "s5_spo_vs_pto_paired.csv; s5_regret_summary_hai_luoi.csv (lưới gốc vs mở rộng)"),
        (f"Quy mô toàn t0 (7 kỳ valid+calib, cộng dồn): regret A MILP {f(m6.loc['MILP trên V̂', 'regret_A_cong_don_tat_ca'], 4)}, Greedy {f(m6.loc['Greedy (V̂/mệnh giá)', 'regret_A_cong_don_tat_ca'], 4)}, "
         f"luật cũ {f(m6.loc['Luật RFM cũ', 'regret_A_cong_don_tat_ca'], 4)}; MILP ≥ Greedy trên V̂ ở {m6.loc['MILP trên V̂', 'ky_MILP_thang_Greedy_V_hat']} kỳ", "IV", "s6_main_summary.csv; fig_s6_main_comparison.png"),
        (f"Hiệu chỉnh xác suất ↔ chi phí chiết khấu: tỷ lệ chi thực/kế hoạch (kỳ vọng) {f(cd.loc['μ̂0 thô (chưa hiệu chỉnh)', 'thuc_tren_ke_hoach_ky_vong'], 3)} khi μ̂0 thô so với "
         f"{f(cd.loc['μ̂0 đã hiệu chỉnh (Platt)', 'thuc_tren_ke_hoach_ky_vong'], 3)} khi đã hiệu chỉnh", "IV/V", "s6_calibration_discount_cost_summary.csv; fig_s6_calibration_discount_cost.png"),
        ("Giá bóng ngân sách / cuộc gọi (LP relaxation); độ nhạy B, K, biên lợi nhuận m", "IV", "s6_shadow_prices.csv; s6_sensitivity_BK.csv; s6_sensitivity_margin.csv")]))

    A("\n### IV.5 Action Router, guardrail, HITL (GĐ7, Router v2.1, valid+calib 7 kỳ)\n")
    pv = tab("s7_policy_value_summary_v21.csv").set_index("chinh_sach")
    fl = tab("s7_decision_flow_v21.csv").set_index("chinh_sach")
    mr, lr = fl.loc["MILP qua Router v2.1"], fl.loc["Luật cũ qua Router v2.1"]
    A(table_rows([
        (f"Giá trị hiện thực cộng dồn: MILP qua Router {gbp(pv.loc['MILP qua Router v2.1', 'thuc_hien_thuc_cong_don'])}, luật cũ qua Router {gbp(pv.loc['Luật cũ qua Router v2.1', 'thuc_hien_thuc_cong_don'])}, "
         f"luật cũ thuần {gbp(pv.loc['Luật cũ thuần (miền v1)', 'thuc_hien_thuc_cong_don'])}", "IV", "s7_policy_value_summary_v21.csv; fig_s7_policy_value.png"),
        (f"Luồng quyết định MILP: đề xuất {int(mr.de_xuat_voucher):,} → HITL {int(mr.chuyen_HITL_de_xuat):,} → đối chứng {int(mr.doi_chung):,} → gửi thực tế {int(mr.gui_thuc_te):,}; "
         f"luật cũ: gửi {int(lr.gui_thuc_te):,}, bù độ phủ {int(lr.khach_bu):,}", "IV", "s7_decision_flow_v21.csv; fig_s7_decision_flow.png"),
        ("Người duyệt HITL là **quy tắc mô phỏng**, không phải người thật (6b)", "V", "CLAUDE.md §6b; s7_hitl_reviewer_value.csv")]))

    # IV.6 Release gates
    A("\n### IV.6 Bảng Release Gates (Locked Test, mở 1 lần)\n")
    g = tab("s8_locked_gates.csv")
    gt = g[g.quan_the == "toan_bo"].set_index("gate")
    verdict = {"G1": "—", "G2": "dự báo chỉ nêu khả năng 'không hợp'; không xảy ra", "G3": "**SAI** (dự báo ≈ 4–7 %, thực tế đạt)", "G4a": "đúng", "G4a-phụ": "đúng",
               "G4b": "đúng", "G4b-phụ": "đúng", "G5": "đúng", "G6": "đúng", "G7": "đúng", "G7-phụ": "—"}
    rows = []
    for k in ["G1", "G2", "G3", "G4a", "G4a-phụ", "G4b", "G4b-phụ", "G5", "G6", "G7", "G7-phụ"]:
        x = gt.loc[k]
        rows.append({"Gate": k, "Kết quả": "ĐẠT" if bool(x.dat) else "TRƯỢT", "Giá trị": x.gia_tri, "Ngưỡng": x.nguong, "Dự báo đăng ký trước (6b)": x.du_bao_dang_ky_truoc,
                     "Dự báo so với thực tế": verdict[k]})
    A(md_table(pd.DataFrame(rows)))
    A("\n*Nguồn:* `s8_locked_gates.csv` (toàn bộ locked: 12 947 dòng, 3 kỳ). **G4 = G4a ∧ G4b → TRƯỢT ⇒ không đạt đủ cổng.** Hình: `fig_s8_locked_gates.png`.\n")
    cs = pd.read_csv(TB / "s8_locked_gate_summary_corrected.csv", keep_default_na=False).drop(columns="ghi_chu")
    A("Theo quần thể (bảng tổng hợp đã sửa, lập từ `s8_locked_gates.csv`):\n")
    A(md_table(cs))
    A("\n*Nguồn:* `s8_locked_gate_summary_corrected.csv`. Entity-holdout (2 005 dòng) chỉ mang tính mô tả.\n")
    q3 = tab("s8_locked_g3_qini.csv")
    A("Qini trên nhánh ngẫu nhiên của locked (G3):\n")
    A(md_table(q3[["quan_the", "hanh_dong", "n", "qini", "qini_ci_lo", "qini_ci_hi", "ci_duong"]].round(3)))
    A("\n*Nguồn:* `s8_locked_g3_qini.csv`.\n")

    # IV.7 OOT
    A("\n### IV.7 Phát lại OOT (5 kỳ, 21 693 dòng, mô hình Current đóng băng)\n")
    lat = tab("s8_oot_latency.csv")
    gl = lat[lat.ky == "Gộp mọi kỳ"][["khau", "so_luot", "P50", "P95", "P99"]].round(3)
    A("**Độ trễ từng khâu (giây / một t0, ≥ 30 lượt/kỳ; gộp 5 kỳ)** – P50 / P95 / P99:\n")
    A(md_table(gl))
    tot = lat[(lat.khau == "tong") & (lat.ky != "Gộp mọi kỳ")][["ky", "so_luot", "P50", "P95", "P99"]].round(3)
    A("\nTổng đầu–cuối theo kỳ:\n")
    A(md_table(tot))
    A("\n*Nguồn:* `s8_oot_latency.csv` (locked: `s8_locked_g6_latency_percentiles.csv`).\n")
    ps = tab("s8_oot_psi.csv")
    pg_ = ps[ps.ky == "Gộp mọi kỳ"]
    top = pg_.sort_values("psi", ascending=False).head(8)[["loai", "bien", "psi", "trang_thai"]].round(3)
    A(f"**PSI so với train (gộp 5 kỳ):** {int((pg_.trang_thai == 'ổn định').sum())} ổn định, {int((pg_.trang_thai == 'cảnh báo').sum())} cảnh báo "
      f"(≥ 0.10), {int((pg_.trang_thai == 'báo động').sum())} báo động (≥ 0.25) trên {len(pg_)} biến/điểm. 8 giá trị lớn nhất (đặc trưng lịch `t0_month`, `days_to_xmas` trôi theo thiết kế mùa vụ):\n")
    A(md_table(top))
    A("\n*Nguồn:* `s8_oot_psi.csv`.\n")
    cm = tab("s8_oot_calibration_matured.csv")
    A("**Brier / ECE chỉ trên kỳ có nhãn đã trưởng thành** (t0 + 30 ngày ≤ ngày tính), dòng a0; Current vs Challenger (Platt theo kỳ) vs Challenger-2 (μ0 học lại, PHỤ):\n")
    pv_ = cm.pivot(index="ky", columns="phien_ban", values=["ece", "brier"]).round(4)
    pv_.columns = [f"{a} – {b}" for a, b in pv_.columns]
    pv_.insert(0, "ngày tính", cm[cm.phien_ban == "Current"].set_index("ky").ngay_tinh)
    pv_.insert(1, "trễ (ngày)", cm[cm.phien_ban == "Current"].set_index("ky").tre_so_voi_quyet_dinh_ngay)
    A(md_table(pv_.reset_index()))
    A(f"\nNgưỡng báo động ECE = 0.05 (`monitoring.ece_alert`). *Nguồn:* `s8_oot_calibration_matured.csv`. Ghi chú: tỷ lệ mua thực tế tăng từ "
      f"{f(cm[cm.phien_ban == 'Current'].ty_le_mua.iloc[0], 3)} lên {f(cm[cm.phien_ban == 'Current'].ty_le_mua.iloc[-1], 3)} trong khi p̂ trung bình chỉ ≈ "
      f"{f(cm[cm.phien_ban == 'Current'].p_TB.mean(), 2)} ⇒ Current bị hụt (mùa Q4).\n")
    tv = tab("s8_oot_action_tvd.csv")
    A(f"**Action TVD** so với phân phối hành động cuối của Current trên valid+calib: {f(tv.action_tvd.min(), 4)} – {f(tv.action_tvd.max(), 4)} (ngưỡng {tv.nguong.iloc[0]}; cảnh báo: "
      f"{int(tv.canh_bao.sum())}/{len(tv)} kỳ). *Nguồn:* `s8_oot_action_tvd.csv`.\n")
    ld = tab("s8_oot_label_delay_by_date.csv").drop(columns="nhan")
    A("**Độ trễ nhãn – tại mỗi ngày phát lại, bao nhiêu kỳ/dòng đã có nhãn trưởng thành:**\n")
    A(md_table(ld.round(3)))
    lm = tab("s8_oot_label_delay_by_metric.csv").drop(columns="nhan")
    A("\nMỗi chỉ số giám sát bị trễ bao lâu so với lúc ra quyết định:\n")
    A(md_table(lm.round(1)))
    A("\n*Nguồn:* `s8_oot_label_delay_by_date.csv`, `s8_oot_label_delay_by_metric.csv`.\n")
    dr = tab("s8_oot_dr_value.csv")
    d0 = dr[dr.ky == "Gộp mọi kỳ"].iloc[0]
    A(f"**Giá trị thực hiện (DR/AIPW trên nhóm đối chứng 10 %, thước đo giám sát – KHÔNG chấm gate):** gộp 5 kỳ {gbp(d0.DR_uoc_luong)} [CI 95 % {gbp(d0.DR_ci_lo)}; {gbp(d0.DR_ci_hi)}], "
      f"đáp án (hiện thực) {gbp(d0.that_dap_an_hien_thuc)}; CI {'phủ' if d0.ci_phu_that else 'không phủ'} đáp án và {'chứa' if d0.ci_chua_0 else 'không chứa'} 0; chỉ {int(d0.n_doi_chung)} khách đối chứng. "
      f"Theo kỳ, CI không phủ đáp án ở {int((~dr[dr.ky != 'Gộp mọi kỳ'].ci_phu_that).sum())}/{len(dr) - 1} kỳ. *Nguồn:* `s8_oot_dr_value.csv`; hình `fig_s8_oot.png`.\n")

    A("\n### IV.8 Kiểm định ngoài trên RCT thật – Hillstrom (s9)\n")
    s9 = tab("s9_acceptance.csv").set_index("muc")
    A(table_rows([(f"{k}: {v}", "IV/V", "s9_acceptance.csv; s9_qini_test.csv; s9_policy_value_test.csv; fig_s9_*.png") for k, v in s9.gia_tri.items() if not k.startswith("Giới hạn")] +
                 [("Giới hạn: kiểm định phương pháp trên RCT thật, **không** phải toàn pipeline Online Retail", "V", "s9_acceptance.csv")]))

    # ------------------------------------------------------------------ V
    A("\n## V. Thảo luận\n")
    A("### V.1 Khuyết nhãn có chọn lọc (OOT)\n")
    se = tab("s8_oot_selective_effect.csv")
    scb = tab("s8_oot_selective_calibration.csv")
    scb = scb[scb.nhom.str.startswith("nhóm")]
    A("Router chỉ quan sát kết quả của khách nó chọn gửi. Ước lượng tác động chương trình trên nhóm được chọn (thật = "
      f"{f(se.that_GBP_moi_khach_duoc_chon.iloc[0], 2)} £/khách):\n")
    A(md_table(se[["phuong_phap", "uoc_luong_GBP_moi_khach_duoc_chon", "do_chech", "n_doi_chung"]].round(2)))
    A("\nHiệu chỉnh μ̂0 học từ dữ liệu do Router sinh ra (nhóm được chọn): có vs không có nhóm đối chứng + IPW:\n")
    A(md_table(scb[["phuong_phap", "p_du_bao_TB", "Y0_that_TB", "do_chech", "ece_vs_Y0_that"]].round(4)))
    A("\n*Nguồn:* `s8_oot_selective_effect.csv`, `s8_oot_selective_calibration.csv`. Cả ba ước lượng tác động đều **chệch dương** so với đáp án (đáp án chỉ dùng để đo độ chệch). "
      f"Học lại hiệu chỉnh từ dữ liệu Router giảm độ chệch của p̂ trên nhóm được chọn từ {f(scb.iloc[0].do_chech, 3)} (Current) xuống ≈ {f(scb.iloc[1].do_chech, 3)}; thêm nhóm đối chứng + IPW chỉ cải thiện thêm "
      f"≈ {abs(scb.iloc[1].do_chech) - abs(scb.iloc[2].do_chech):.3f} – **không đáng kể** với chỉ {int(se.n_doi_chung.iloc[0])} khách đối chứng (phương sai lớn).\n")
    A("### V.2 Độ trễ hạ tầng và chi phí bộ giải\n")
    l6 = tab("s8_locked_g6_latency_percentiles.csv").set_index("khau")
    A(table_rows([
        (f"Locked (G6): P95 đầu–cuối {f(l6.loc['tong', 'P95'], 3)} s (ngưỡng 30 s); khâu chấm điểm mô hình chiếm P50 {f(l6.loc['t_cham_diem', 'P50'], 3)} s / {f(l6.loc['tong', 'P50'], 3)} s; "
         f"MILP P95 {f(l6.loc['t_milp', 'P95'], 3)} s", "V", "s8_locked_g6_latency_percentiles.csv"),
        (f"Độ trễ nhãn (hạ tầng dữ liệu) chi phối giám sát: Brier/ECE chỉ có sau tối thiểu 30 ngày, thực tế {f(val(tab('s8_oot_label_delay_by_metric.csv'), {'chi_so': 'Brier / ECE (Current, Challenger)'}, 'tre_thuc_te_TB_ngay'), 1)} ngày (trung bình)", "V", "s8_oot_label_delay_by_metric.csv"),
        (f"Chi phí bộ giải theo quy mô: n = {int(big.n):,} khách → HiGHS {f(big.highspy_giay_giai, 2)} s (gap {big.highspy_gap:g}), Greedy {f(big.greedy_giay, 3)} s; "
         f"Greedy kém MILP trên V̂ {sc.gap_greedy_vs_MILP.min():.1%}–{sc.gap_greedy_vs_MILP.max():.1%} (các cỡ n)", "V", "s6_scaling.csv; fig_s6_scaling.png"),
        (f"MILP một t0 (~3.6k khách): {f(m6.loc['MILP trên V̂', 'giay_giai_TB'], 3)} s trung bình", "V", "s6_main_summary.csv"),
        ("Hillstrom: chỉ có 1 ràng buộc đếm ⇒ MILP = Greedy (ma trận unimodular)", "V", "s9_milp_vs_greedy.csv")]))
    A("\n### V.3 Giả định nhận dạng nhân quả\n")
    pc = tab("s4_propensity_compare.csv")
    pct = pc[pc.tap.str.startswith("train")]
    A(table_rows([
        ("Không nhiễu chưa quan sát (ignorability): chính sách lịch sử chỉ phụ thuộc đặc trưng đã quan sát (20 % ngẫu nhiên + 80 % luật RFM) – đúng **theo thiết kế bán tổng hợp**, không kiểm định được trên dữ liệu thật", "V", "CLAUDE.md §2, §3.2; s1b_intervention.py"),
        (f"Chồng lấp (overlap): propensity ghi log nhỏ nhất {f(pct.min_ghi_log.min(), 3)} trên train; tỷ lệ propensity ước lượng < 0.05 bằng {pct['ty_le_uoc_luong_duoi_0.05'].max():.0f}", "V", "s4_propensity_compare.csv; fig_s4_propensity_overlap.png"),
        ("Kiểm tra mô hình hóa propensity: ước lượng vs ghi log sai lệch tuyệt đối trung bình ≤ " + f(pc.MAE.max(), 3), "V", "s4_propensity_compare.csv"),
        ("Placebo (xáo A): ATE ≈ 0 ở 6/6 phép thử", "V", "s4_placebo.csv"),
        ("SUTVA / không lan truyền: không có hiệu ứng mang sang giữa các t0, mỗi (khách, t0) độc lập – **giới hạn đã biết**; giỏ của người mới chuyển đổi là mô phỏng; biên lợi nhuận 30 % là giả định", "V", "CLAUDE.md §6"),
        ("Ở vận hành, Router gần tất định: chỉ nhóm đối chứng 10 % trong tập ngẫu nhiên nhận dạng được tác động (propensity a0 = 0 cho khách bù độ phủ)", "V", "CLAUDE.md §6b; s8_oot_dr_value.csv")]))
    A("\n### V.4 Quan hệ phi tuyến giữa sai số dự báo và regret\n")
    mv = tab("s5_mse_vs_regret.csv")
    gm = mv.groupby(["loai", "muc"], as_index=False)[["MSE_hien_thuc", "regret_cn_hien_thuc"]].mean()
    noise = gm[gm.loai.str.startswith("Nhiễu cộng")].copy()
    noise["muc"] = noise.muc.astype(float)
    noise = noise.sort_values("muc")
    A(f"Bơm nhiễu cộng vào V̂ (σ × sd V̂): MSE tăng gấp ≈ {noise.MSE_hien_thuc.iloc[-1] / noise.MSE_hien_thuc.iloc[0]:.1f} lần nhưng regret chỉ tăng chậm (và không đơn điệu ở mức nhiễu nhỏ) – quan hệ **phi tuyến**:\n")
    A(md_table(noise.assign(MSE_hien_thuc=noise.MSE_hien_thuc.round(1), regret_cn_hien_thuc=noise.regret_cn_hien_thuc.round(4))[["muc", "MSE_hien_thuc", "regret_cn_hien_thuc"]].rename(columns={"muc": "σ"})))
    scl = gm[gm.loai.str.startswith("Co giãn")]
    mth = gm[gm.loai == "Phương pháp"].set_index("muc")
    A(f"\nCo giãn V̂ × k đổi MSE từ {f(scl.MSE_hien_thuc.min(), 0)} đến {f(scl.MSE_hien_thuc.max(), 0)} nhưng regret **không đổi** ({f(scl.regret_cn_hien_thuc.iloc[0], 4)}) vì thứ hạng không đổi. "
      f"Giữa các phương pháp: SPO+ có MSE cao nhất ({f(mth.loc['SPO+ tuyến tính', 'MSE_hien_thuc'], 0)}) nhưng regret {f(mth.loc['SPO+ tuyến tính', 'regret_cn_hien_thuc'], 3)} thấp hơn PTO LightGBM "
      f"(MSE {f(mth.loc['PTO LightGBM', 'MSE_hien_thuc'], 0)}, regret {f(mth.loc['PTO LightGBM', 'regret_cn_hien_thuc'], 3)}). *Nguồn:* `s5_mse_vs_regret.csv`; hình `fig_s5_mse_vs_regret.png`. "
      "(MSE thấp ≠ regret thấp; regret phụ thuộc thứ hạng, không phụ thuộc thang đo.)\n")
    A("### V.5 Hiệu quả rào chắn nghiệp vụ (guardrail) khi dữ liệu bị trôi (τ thật × 0.6 ở OOT)\n")
    dg = tab("s8_oot_drift_guardrail.csv")
    A(md_table(dg[["kich_ban", "chinh_sach", "gia_tri_hien_thuc", "gia_tri_ky_vong", "vi_pham_quy_dinh", "HITL_tu_choi_ha_cap", "ton_that_tranh_duoc_hien_thuc"]].round(0)))
    nd, tr = dg[dg.kich_ban == "không trôi"], dg[dg.kich_ban != "không trôi"]
    g_nd, g_tr = nd[nd.chinh_sach.str.startswith("CÓ")].iloc[0], tr[tr.chinh_sach.str.startswith("CÓ")].iloc[0]
    n_nd, n_tr = nd[nd.chinh_sach.str.startswith("KHÔNG")].iloc[0], tr[tr.chinh_sach.str.startswith("KHÔNG")].iloc[0]
    A(f"\nCó guardrail (MILP qua Router) {gbp(g_nd.gia_tri_hien_thuc)} / {gbp(g_tr.gia_tri_hien_thuc)} (không trôi / trôi) so với không guardrail {gbp(n_nd.gia_tri_hien_thuc)} / {gbp(n_tr.gia_tri_hien_thuc)}: "
      f"guardrail **làm mất giá trị ở cả hai kịch bản** ({gbp(g_nd.ton_that_tranh_duoc_hien_thuc)} / {gbp(g_tr.ton_that_tranh_duoc_hien_thuc)}), đổi lại không vi phạm quy định bắt buộc "
      f"(opt-out, frequency cap, hủy đơn): không guardrail vi phạm {int(n_nd.vi_pham_quy_dinh)} lần, có guardrail {int(g_nd.vi_pham_quy_dinh)} lần. Ở router v2.1, opt-out/cap/hủy đơn nằm trong miền MILP "
      "nên không còn \"lượt bị chặn cứng\" sau khi giải; tác động nằm ở HITL (từ chối/hạ cấp). *Nguồn:* `s8_oot_drift_guardrail.csv`; đáp án trôi: `oracle_oot_drift.parquet` (chỉ đọc trong bước OOT).\n")

    # ------------------------------------------------------------------ VI
    A("\n## VI. Playbook vận hành\n")
    pb = tab("s8_oot_playbook.csv").drop(columns=["nhan", "nguon_nguong", "tan_suat", "do_tre_du_lieu"])
    A(md_table(pb))
    A("\n*Nguồn:* `s8_oot_playbook.csv` (ngưỡng PSI/ECE/TVD lấy từ `configs/config.yaml` mục `monitoring`; ba ngưỡng cảnh báo P95 > 15 s, tải HITL > 150/250 ca/kỳ, tỷ lệ chặn tăng > 2× "
      "là **ngưỡng tự đặt**, không thuộc `monitoring`).\n")

    # ------------------------------------------------------------------ thí điểm
    A("\n## Phụ: thiết kế thí điểm (cỡ mẫu)\n")
    pg3 = tab("s8_pilot_g3.csv")
    s1 = pg3[pg3.kich_ban.str.startswith("S1")]
    A("G3 – cỡ mẫu nhánh ngẫu nhiên để đạt power 80 % (Qini giả định = ước lượng calib):\n")
    A(md_table(s1[["hanh_dong", "qini_gia_dinh", "n_cap_can_power80", "boi_so_locked"]].round(3)))
    pdr = tab("s8_pilot_dr.csv")
    rt = pdr[pdr.thiet_ke.str.startswith("Router") & pdr.ty_le_doi_chung.isin([0.1, 0.5])][["ty_le_doi_chung", "kich_ban_hieu_ung", "se_mot_ky", "so_ky_can_power80"]].round(1)
    A("\nDR – số kỳ cần để CI giá trị chương trình loại trừ 0 (thiết kế Router):\n")
    A(md_table(rt))
    A("\n*Nguồn:* `s8_pilot_g3.csv`, `s8_pilot_dr.csv`, `fig_s8_pilot_sample_size.png`.\n")

    # ------------------------------------------------------------------ công khai
    A("\n## Công khai và sai lệch so với đăng ký trước\n")
    g3t = tab("s8_locked_g3_qini.csv").set_index(["quan_the", "hanh_dong"])
    pgt = tab("s8_pilot_g3.csv")
    a2p = pgt[pgt.kich_ban.str.startswith("S1") & (pgt.hanh_dong == "a2")].iloc[0]
    sp_ = tab("s1_split_summary.csv").set_index("split")
    A("**(i) Số liệu tổng hợp trên locked đã được xem ở GĐ2.** Theo lời người dùng, Claude (trong chat) đã xem số liệu tổng hợp trên locked ở GĐ2. Repo không có nhật ký phiên chat nên "
      "*không xác minh được chính xác số liệu nào*. Bằng chứng trong repo xác nhận tỷ lệ mua lại tổng hợp của locked và oot đã được biết ở GĐ2: CLAUDE.md §6 ghi tỷ lệ mua lại a0 "
      f"locked 0.187 và oot 0.289 (\"phát hiện ở GĐ2\"); `s1_split_summary.csv` cho tỷ lệ gốc locked {f(sp_.loc['locked', 'base_rate'], 4)}, oot {f(sp_.loc['oot', 'base_rate'], 4)}. Hệ quả: "
      "(a) đăng ký trước ở 6b (\"Platt có thể không hợp locked/oot\") được viết khi đã biết xu hướng mức nền; (b) kết quả G2 trên locked không phải kiểm định *mù hoàn toàn* về mức nền, "
      "dù mô hình, bộ hiệu chỉnh và ngưỡng đều đã được đóng băng trước khi mở và không được chỉnh theo nhãn locked. Nhãn cấp dòng, Qini và giá trị chính sách của locked chỉ được tính ở bước 2 "
      "(theo các handoff 1a/1b; không kiểm chứng được cho các phiên trước đó). Theo cách đọc nghiêm ngặt, điều này lệch khỏi luật cứng #4 (\"locked mở đúng 1 lần ở GĐ8\").\n")
    A("**(ii) Lỗi bảng tổng hợp gate.** `s8_locked_gate_summary.csv` in toàn `n/a` vì mã so sánh `v is True` với `numpy.bool_` (ở dry-run cột `dat` kiểu object nên hiển thị đúng; ở locked cột kiểu bool). "
      "Chỉ ảnh hưởng bảng tổng hợp; `s8_locked_gates.csv` (cột `dat`) đúng. Không sửa được vì `src/` đã đóng băng và locked đã mở. Bảng đúng: `s8_locked_gate_summary_corrected.csv`, "
      "lập từ `s8_locked_gates.csv`. Diễn tập đường locked không bắt được lỗi này vì không kiểm nội dung bảng tổng hợp.\n")
    A("**(iii) G3 dự báo ≈ 4–7 % nhưng đạt.** Toàn bộ locked: a2 Qini "
      f"{f(g3t.loc[('toan_bo', 'a2'), 'qini'])} [CI {f(g3t.loc[('toan_bo', 'a2'), 'qini_ci_lo'])}; {f(g3t.loc[('toan_bo', 'a2'), 'qini_ci_hi'])}], a3 {f(g3t.loc[('toan_bo', 'a3'), 'qini'])} "
      f"[{f(g3t.loc[('toan_bo', 'a3'), 'qini_ci_lo'])}; {f(g3t.loc[('toan_bo', 'a3'), 'qini_ci_hi'])}] ⇒ 2/3 hành động có CI > 0 (cận dưới chỉ {f(g3t.loc[('toan_bo', 'a2'), 'qini_ci_lo'])} và "
      f"{f(g3t.loc[('toan_bo', 'a3'), 'qini_ci_lo'])}, sát 0). Cần đọc thận trọng: (1) **không hiệu chỉnh đa kiểm định** (3 hành động, mỗi hành động một CI 95 %); (2) **khách thường chỉ đạt 1/3** "
      f"(a2 {f(g3t.loc[('khach_thuong', 'a2'), 'qini'])}; a3 CI [{f(g3t.loc[('khach_thuong', 'a3'), 'qini_ci_lo'])}; {f(g3t.loc[('khach_thuong', 'a3'), 'qini_ci_hi'])}] chứa 0) – "
      "nghĩa là a3 chỉ qua ngưỡng nhờ 2 005 dòng entity-holdout; entity-holdout riêng 0/3; (3) **mâu thuẫn với bảng thí điểm**: bảng thí điểm lấy Qini a2 = "
      f"{f(a2p.qini_gia_dinh, 3)} (ước lượng calib) và kết luận cần ≈ {a2p.boi_so_locked:,.0f}× cỡ locked để đạt power 80 %, trong khi locked cho {f(g3t.loc[('toan_bo', 'a2'), 'qini'])} (gấp ≈ "
      f"{g3t.loc[('toan_bo', 'a2'), 'qini'] / a2p.qini_gia_dinh:.0f}×). Hai kết quả chỉ hòa giải được nếu hiệu ứng thật khác nhau theo mùa (calib tháng 4–5, locked tháng 7–8) hoặc ước lượng locked "
      "là một lần rút may mắn (hiệu ứng lớn bị phóng đại khi chỉ báo cáo cái vượt ngưỡng). Dữ liệu hiện có không phân biệt được hai khả năng; **không nên diễn giải 0.098 như độ lớn hiệu ứng thật**. "
      "G3 vẫn được ghi là ĐẠT theo định nghĩa đã đăng ký (toàn bộ locked).\n")
    A("**(iv) Các chỗ tự quyết định** (chi tiết: `outputs/manifests/s8_handoff_1a.md`, `s8_handoff_1b.md`). *Bước 1a:* G4b chấm trên MILP qua Router v2.1, MILP thuần là G4b-phụ; G6 gồm khâu mã lý do "
      "SHAP top-3; quần thể chấm chính thức = toàn bộ; G1 là bộ kiểm tra lắp ghép (từng trượt ở dry-run đầu do lỗi mã kiểm tra propensity, đã sửa) – bốn điểm này đã được người dùng đồng ý và ghi vào 6b "
      "(\"Làm rõ trước khi mở locked\"); G6 dùng P95 gộp mọi lượt; frequency cap bắt đầu rỗng ở tập chấm; G2 trên calib ở dry-run là trong mẫu; không làm phân rã đáp án B/C; phạm vi băm dữ liệu loại "
      "`decisions_s8_*`; lệch phiên bản thư viện bị coi là lệch đóng băng. *Bước 1b:* cửa sổ Challenger = 3 kỳ gần nhất đã trưởng thành (chỉ ghi trong docstring `s8b_oot.py`, không vào 6b); "
      "\"phân định MILP với Current\" hiểu theo hai nghĩa (MILP − luật cũ dưới log 20 % ngẫu nhiên; Current so với không làm gì dưới đối chứng của Router); MILP qua Router vs luật cũ qua Router "
      "không nhận dạng được từ log Router (chưa tính); phản hồi sau triển khai lấy từ đáp án trong hàm `oracle_*`; \"không guardrail\" = MILP thuần miền v1; diễn tập trôi dùng cặp đáp án sinh trong bộ nhớ; "
      "ba ngưỡng playbook tự đặt; π của khách đối chứng = a1 nếu HITL hạ cấp (bỏ qua ảnh hưởng của bước sửa trần a2). *Bước đóng gói:* không có tài liệu CLO nên không điền CLO; "
      "repo không đưa bản sao Hillstrom (xem README).\n")
    sp_t = TB.parent.parent / "tests"
    A("**(v) Test sửa sau đóng băng** (thư mục `tests/` nằm ngoài phạm vi băm nên `verify_freeze` không bao quát): "
      f"`tests/test_s8b.py` sửa lúc {utc_mtime(sp_t / 'test_s8b.py')} (freeze `s8_freeze` lúc 14:31:07 UTC; khóa locked lúc 14:40:19 UTC) – thêm `test_real_oot_path_preflight`, diễn tập nhánh OOT \"real\" bằng dữ liệu thay thế "
      f"*trước* khi mở locked. `tests/test_s8.py` sửa lúc {utc_mtime(sp_t / 'test_s8.py')} (sau khi mở locked) – hai test (`test_dry_run_does_not_touch_locked_oot`, `test_locked_path_rehearsal`) từng khẳng định \"khóa chưa tồn tại\", "
      "đổi thành \"khóa và `s8_locked_gates.csv` thật không bị thay đổi\" (ý định kiểm tra không đổi). Phạm vi băm không thay đổi (`verify_freeze` = 0 lệch sau các lần sửa test); chỉ phát sinh thêm các file đầu ra `decisions_s8_*` và bảng/hình mới. "
      "Lưu ý nhỏ: `tests/test_s8b.py` dòng `assert not S8.LOCK_PATH.exists() or True` là khẳng định vô hiệu (luôn đúng), không ảnh hưởng kết quả. Hash hiện tại của hai file nằm trong `s8_sha256_appendix.csv`.\n")
    A("\n## Phụ lục tái lập\n")
    A("- `outputs/tables/s8_sha256_appendix.csv` – SHA-256 từng file trong phạm vi băm, hash tổng hợp, nội dung file khóa, ghi chú sửa sau băng.\n"
      "- `outputs/manifests/s8_freeze_manifest.json`, `requirements_lock.txt` (Python 3.14.7), `outputs/tables/s8_compat_matrix.csv`.\n"
      "- `outputs/tables/s8_checklist_1a.csv`, `outputs/manifests/s8_handoff_1a.md`, `s8_handoff_1b.md`.\n")
    out = TB / "summary_for_essay.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def table_rows(rows: list[tuple]) -> str:
    df = pd.DataFrame([{"Kết quả": a, "Mục": b, "CLO": CLO, "Nguồn": f"`{c}`" if "`" not in c else c} for a, b, c in rows])
    return md_table(df)


# ============================================================================ 3. notebook
def cell_md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def cell_code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


STAGES = [
    ("GĐ1 – Thiết kế bài toán và dữ liệu, chống rò rỉ (s1, s1b)",
     "Định nghĩa đơn vị khách × t0, chặn dữ liệu cutoff = t0 − 1 ngày, nhãn mua lại 30 ngày, chia tập có embargo và entity holdout; sinh lớp can thiệp bán tổng hợp.",
     "Làm sạch Online Retail II → snapshot point-in-time → data contract (pandera) → 6 kiểm thử rò rỉ → lớp can thiệp (Y(0) thật, Y(a) từ cơ chế đã biết, 20 % nhánh ngẫu nhiên).",
     "Chương 1 · mục III",
     ["python src/s1_data.py", "python src/s1b_intervention.py"],
     [("table", "s1_split_summary.csv"), ("table", "s1_leakage_tests.csv", 6), ("table", "s1b_confounding_check.csv"), ("fig", "fig_s1_split_timeline.png"), ("fig", "fig_s1_base_rate_by_t0.png")]),
    ("GĐ2 – Dự báo nền và hiệu chỉnh xác suất (s2)",
     "μ0(x) = P(mua lại | a0, x) và giá trị giỏ Ŝ có khoảng conformal; hiệu chỉnh để chi phí chiết khấu không bị lệch.",
     "Logistic và LightGBM (GroupKFold theo khách) → Platt/Isotonic khớp trên calib → ROC-AUC, PR-AUC, log loss, Brier, ECE 15 bin; conformal 90 % cho Ŝ.",
     "Chương 2 · mục IV.1",
     ["python src/s2_predict.py"],
     [("table", "s2_metrics_valid.csv", 6), ("table", "s2_acceptance.csv"), ("fig", "fig_s2_reliability_valid.png"), ("fig", "fig_s2_basket_conformal.png")]),
    ("GĐ3 – Giải thích và độ tin cậy (s3)",
     "Giải thích mô hình μ0 và sinh mã lý do top-3 cho Router; kiểm tra ổn định và faithfulness.",
     "PFI (cá nhân và theo nhóm), SHAP TreeExplainer (thang log-odds), ổn định hạng SHAP qua bootstrap, thử xóa top-k. SHAP/PFI không phải nhân quả.",
     "Chương 3 · mục IV.2",
     ["python src/s3_explain.py"],
     [("table", "s3_pfi_group.csv"), ("table", "s3_faithfulness_deletion.csv"), ("fig", "fig_s3_pfi.png"), ("fig", "fig_s3_shap_beeswarm.png")]),
    ("GĐ4 – Nhân quả: ATE, CATE, uplift (s4, s4b)",
     "Ước lượng tác động của từng voucher so với không làm gì, ở mức trung bình (ATE) và theo khách (CATE); dự báo power cho cổng G3.",
     "Propensity mô hình hóa vs ghi log; ATE naive/IPW/DML; T/X/DR-learner và CausalForestDML; Qini trên nhánh ngẫu nhiên; placebo. s4b: power của G3 trên locked (đăng ký trước).",
     "Chương 4 · mục IV.3, V.3",
     ["python src/s4_causal.py", "python src/s4b_power_g3.py"],
     [("table", "s4_acceptance.csv"), ("table", "s4_qini.csv", 8), ("table", "s4_power_g3_summary.csv"), ("fig", "fig_s4_ate_forest.png"), ("fig", "fig_s4_qini_curves.png"), ("fig", "fig_s4b_power_g3.png")]),
    ("GĐ5 – Predict-Then-Optimize và SPO+ (s5, s5b)",
     "So sánh PTO hai giai đoạn với SPO+ trên regret quyết định; thí nghiệm MSE vs regret; ước lượng DR/AIPW cho giá trị chính sách.",
     "Lô 100 khách, ràng buộc co theo tỷ lệ; vector chi phí DR ngoài fold; 5 seed; quy tắc mở rộng biên lưới đăng ký trước; kiểm chứng cài đặt trên instance 6 khách.",
     "Chương 5 · mục IV.4, V.4",
     ["python src/s5_spo.py", "S5_LUOI=mo_rong python src/s5_spo.py", "python src/s5b_dr_diagnostics.py"],
     [("table", "s5_regret_summary.csv", 9), ("table", "s5_spo_vs_pto_paired.csv"), ("table", "s5_verification_6_summary.csv"), ("fig", "fig_s5_regret_by_method.png"), ("fig", "fig_s5_mse_vs_regret.png"), ("fig", "fig_s5b_dr_z_scores.png")]),
    ("GĐ6 – MILP toàn quy mô (s6)",
     "Giải bài toán phân bổ ngân sách/ký hạn mức gọi/độ phủ cho toàn bộ khách của mỗi t0; đo thời gian và độ nhạy; tác động của hiệu chỉnh lên chi phí chiết khấu.",
     "PuLP + HiGHS (gap 0.1 %), so với Greedy và luật cũ; độ nhạy B, K, m; giá bóng từ LP relaxation; thời gian giải tới n = 50 000.",
     "Chương 6 · mục IV.4, V.2",
     ["python src/s6_milp.py"],
     [("table", "s6_main_summary.csv"), ("table", "s6_calibration_discount_cost_summary.csv"), ("table", "s6_scaling.csv", 14), ("fig", "fig_s6_main_comparison.png"), ("fig", "fig_s6_scaling.png"), ("fig", "fig_s6_calibration_discount_cost.png")]),
    ("GĐ7 – Action Router, guardrail, HITL (s7)",
     "Đưa đề xuất MILP qua guardrail cứng, HITL mô phỏng, nhóm đối chứng, bản ghi quyết định có khóa chống lặp. Router v2.1 (đăng ký trước).",
     "Hợp đồng đầu vào → guardrail (opt-out, cancel_rate, frequency cap, sàn biên) → HITL (quy tắc mô phỏng) → đối chứng 10 % → bù độ phủ → sửa trần a2.",
     "Chương 7 · mục IV.5",
     ["python src/s7_router.py", "python src/s7_router.py --v2", "python src/s7_router.py --v21"],
     [("table", "s7_policy_value_summary_v21.csv"), ("table", "s7_decision_flow_v21.csv"), ("table", "s7_latency_percentiles_v21.csv"), ("fig", "fig_s7_policy_value.png"), ("fig", "fig_s7_decision_flow.png")]),
]


def build_notebook() -> Path:
    C: list[dict] = []
    C.append(cell_md(f"""
# PTDLNC – Hệ thống ra quyết định giữ chân khách hàng đa kênh (Online Retail II)

Notebook chạy tuần tự **GĐ1 → GĐ8** và **s9 (kiểm định ngoài)**. Mặc định các ô code **ĐỌC kết quả có sẵn** trong `outputs/` (bảng, hình) nên chạy nhanh trên Colab;
phần chạy lại pipeline nằm dưới cờ `RUN_FULL = False`.

**Cảnh báo.** Trạng thái đã đóng băng: SHA-256 của `src/`, `configs/`, `models/`, `CLAUDE.md` và `data/processed/*.parquet` được ghi trong `outputs/manifests/s8_freeze_manifest.json`.
Đặt `RUN_FULL = True` sẽ **ghi đè** các file đó (⇒ hash không còn khớp) và **không thể mở lại Locked Test** (khóa `s8_locked_opened.lock`). Chỉ dùng trên **bản sao**.

**CLO liên quan:** {CLO} – repo chưa có danh mục CLO của học phần; điền theo đề cương.
"""))
    C.append(cell_code('''
import os, sys, json, hashlib, subprocess, platform
from pathlib import Path

RUN_FULL = False          # True: chạy lại pipeline (GHI ĐÈ data/processed, models, outputs – làm lệch SHA-256 đóng băng). Chỉ dùng trên BẢN SAO.
INSTALL_DEPS = "google.colab" in sys.modules   # tự cài thư viện trên Colab
REPO_URL = ""             # điền URL GitHub của repo (chỉ cần khi chạy trên Colab và chưa clone)

def find_root():
    p = Path.cwd()
    for q in [p, *p.parents]:
        if (q / "configs" / "config.yaml").exists():
            return q
    return None

ROOT = find_root()
if ROOT is None and REPO_URL:
    subprocess.run(["git", "clone", REPO_URL, "/content/repo"], check=True)
    ROOT = Path("/content/repo")
assert ROOT is not None, "Không tìm thấy thư mục dự án (cần configs/config.yaml). Clone repo hoặc đặt REPO_URL."
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
print("Thư mục dự án:", ROOT, "| Python", platform.python_version())
'''))
    C.append(cell_md("""
## 0. Cài đặt và kiểm SHA-256 so với `s8_freeze_manifest.json`

`requirements_lock.txt` được tạo bằng `pip freeze` trên **Python 3.14.7** (macOS Apple Silicon). Trên Colab (Python thấp hơn) các phiên bản ghim có thể không cài được;
khi đó ô dưới tự rơi về `requirements.txt` (ràng buộc `>=`). Phần chỉ-đọc của notebook chỉ cần `pandas` và `matplotlib`.
"""))
    C.append(cell_code('''
if INSTALL_DEPS:
    lock = ROOT / "outputs" / "manifests" / "requirements_lock.txt"
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(lock)])
    if r.returncode != 0:
        print("Không cài được bản ghim (khác phiên bản Python) → dùng requirements.txt")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")])
else:
    print("Bỏ qua cài đặt (không phải Colab). Tự cài: pip install -r outputs/manifests/requirements_lock.txt")

def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

man = json.loads((ROOT / "outputs/manifests/s8_freeze_manifest.json").read_text(encoding="utf-8"))
def check(entries, kind):
    bad, miss = [], []
    for e in entries:
        p = ROOT / e["file"]
        if not p.exists():
            miss.append(e["file"])
        elif sha256_file(p) != e["sha256"]:
            bad.append(e["file"])
    agg = hashlib.sha256("\\n".join(f"{e['file']}:{e['sha256']}" for e in entries).encode()).hexdigest()
    print(f"{kind}: {len(entries)} file | lệch {len(bad)} | thiếu {len(miss)} | hash tổng hợp manifest {agg[:16]}…")
    return bad, miss
b1, m1 = check(man["files"], "mã/mô hình/config/CLAUDE.md")
b2, m2 = check(man["data_files"], "dữ liệu data/processed")
print("ĐÓNG BĂNG KHỚP" if not (b1 or m1 or b2 or m2) else f"CÓ LỆCH: {b1 + b2} | THIẾU: {m1 + m2}")
print("Python khi đóng băng:", man["environment"]["python"], "| hiện tại:", platform.python_version())
'''))
    C.append(cell_code('''
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
try:
    from IPython.display import display
except ImportError:          # chạy ngoài Jupyter
    display = print
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 70)
TABLES, FIGS = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"

def show_table(name, n=12):
    df = pd.read_csv(TABLES / name)
    print(f"[{name}] {df.shape[0]} dòng × {df.shape[1]} cột")
    display(df.head(n).round(4))

def show_fig(name, width=11):
    img = mpimg.imread(FIGS / name)
    h = width * img.shape[0] / img.shape[1]
    plt.figure(figsize=(width, h)); plt.imshow(img); plt.axis("off"); plt.title(name, fontsize=9); plt.show()

def run(cmd):
    print("$", cmd)
    if RUN_FULL:
        subprocess.run(cmd, shell=True, check=True, cwd=ROOT)
    else:
        print("   (bỏ qua – RUN_FULL = False)")
'''))
    for title, goal, method, where, cmds, items in STAGES:
        C.append(cell_md(f"""
## {title}

**Mục tiêu.** {goal}

**Phương pháp.** {method}

**Liên quan:** {where} · **CLO:** {CLO} · **Lệnh tái lập:** {" ; ".join(f"`{c}`" for c in cmds)}
"""))
        code = [f'run("{c}")' for c in cmds]
        for it in items:
            code.append(f'show_table("{it[1]}", {it[2] if len(it) > 2 else 12})' if it[0] == "table" else f'show_fig("{it[1]}")')
        C.append(cell_code("\n".join(code)))
    # ---------------- GĐ8
    C.append(cell_md(f"""
## GĐ8 – Triển khai, đo lường thực chứng, giám sát MLOps (s8, s8b)

**Mục tiêu.** Đóng băng pipeline, mở Locked Test đúng một lần và chấm Release Gates G1–G7; phát lại OOT (mùa Q4, trôi thật) để đo độ trễ, PSI, Brier/ECE khi nhãn trưởng thành,
Action TVD, giá trị thực hiện bằng DR; so Current với Challenger; kịch bản trôi; khuyết nhãn có chọn lọc; playbook.

**Phương pháp.** Manifest SHA-256 + khóa mở một lần (O_EXCL); `read_rows` chặn locked/oot ngoài phiên được phép; Router v2.1 tuần tự từng t0; Challenger = Platt hiệu chỉnh lại theo kỳ chỉ dùng nhãn đã trưởng thành.

**Liên quan:** Chương 8 · mục IV.6–IV.7, V, VI · **CLO:** {CLO}

> **Locked đã mở và OOT đã chạy; khóa `outputs/manifests/s8_locked_opened.lock` đang tồn tại.** Các lệnh `--open-locked`, `--oot` **bị từ chối** nếu chạy lại. Phần dưới chỉ liệt kê lệnh diễn tập/tái lập được.
"""))
    C.append(cell_code('''
run("python src/s8_mlops.py --dry-run")                 # diễn tập: calib đóng vai locked (nhãn THỬ)
run("python src/s8b_oot.py --pilot")                    # thiết kế thí điểm (cỡ mẫu)
run("python src/s8b_oot.py --oot-rehearsal")            # diễn tập OOT: valid = tham chiếu, calib = OOT
print("Khóa locked tồn tại:", (ROOT / "outputs/manifests/s8_locked_opened.lock").exists())
print(open(ROOT / "outputs/manifests/s8_locked_opened.lock", encoding="utf-8").read())
'''))
    C.append(cell_md("""
### Release Gates trên Locked Test (mục IV.6)

Bảng đúng là `s8_locked_gate_summary_corrected.csv` (lập từ `s8_locked_gates.csv`); `s8_locked_gate_summary.csv` in toàn `n/a` do lỗi so sánh `is True` với `numpy.bool_`. Kết quả: **G4b trượt ⇒ không đạt đủ cổng**.
"""))
    C.append(cell_code('''
show_table("s8_locked_gate_summary_corrected.csv")
gates = pd.read_csv(TABLES / "s8_locked_gates.csv")
display(gates[gates.quan_the == "toan_bo"][["gate", "dat", "gia_tri", "nguong"]])
show_table("s8_locked_g3_qini.csv", 9)
show_fig("fig_s8_locked_gates.png")
'''))
    C.append(cell_md("""
### Phát lại OOT, độ trễ nhãn, Current vs Challenger, trôi, khuyết nhãn, playbook (mục IV.7, V, VI)
"""))
    C.append(cell_code('''
lat = pd.read_csv(TABLES / "s8_oot_latency.csv")
display(lat[lat.ky == "Gộp mọi kỳ"][["khau", "so_luot", "P50", "P95", "P99"]].round(3))
psi = pd.read_csv(TABLES / "s8_oot_psi.csv"); g = psi[psi.ky == "Gộp mọi kỳ"]
print("PSI (gộp):", g.trang_thai.value_counts().to_dict())
display(g.sort_values("psi", ascending=False).head(8)[["loai", "bien", "psi", "trang_thai"]].round(3))
show_table("s8_oot_calibration_matured.csv", 15)
show_table("s8_oot_action_tvd.csv", 5)
show_table("s8_oot_label_delay_by_date.csv", 6)
show_table("s8_oot_label_delay_by_metric.csv", 8)
show_table("s8_oot_dr_value.csv", 6)
show_table("s8_oot_drift_guardrail.csv", 6)
show_table("s8_oot_selective_effect.csv", 3)
show_table("s8_oot_selective_calibration.csv", 6)
show_table("s8_oot_playbook.csv", 9)
show_fig("fig_s8_oot.png")
'''))
    C.append(cell_code('''
show_table("s8_pilot_g3.csv", 6)
show_table("s8_pilot_dr.csv", 10)
show_fig("fig_s8_pilot_sample_size.png")
'''))
    C.append(cell_md("""
### Phụ lục SHA-256 và công khai sai lệch

`outputs/tables/s8_sha256_appendix.csv` liệt kê SHA-256 từng file đóng băng, hash tổng hợp, nội dung file khóa và ghi chú sửa sau băng.
`outputs/tables/summary_for_essay.md` ánh xạ kết quả → mục tiểu luận → file nguồn, kèm mục **Công khai và sai lệch so với đăng ký trước**.
"""))
    C.append(cell_code('''
show_table("s8_sha256_appendix.csv", 4)
print((TABLES / "summary_for_essay.md").read_text(encoding="utf-8")[:1500], "…")
'''))
    C.append(cell_md(f"""
## s9 – Kiểm định ngoài trên RCT thật (Hillstrom)

**Mục tiêu.** Kiểm định phương pháp (CATE/Qini, ước lượng giá trị chính sách, MILP) trên dữ liệu có ngẫu nhiên hóa thật, độc lập với Online Retail II.

**Phương pháp.** Chia 70/30 phân tầng; chọn learner bằng Qini CV; Qini và giá trị chính sách trên tập test với CI bootstrap; kết quả chính (conversion) khai báo trước, kết quả phụ (visit) tách riêng.
**Giới hạn.** Không phải kiểm định toàn pipeline. Hillstrom **không nêu giấy phép** nên không có trong repo; s9 tải bằng `sklift.datasets.fetch_hillstrom`.

**Liên quan:** mục IV.8, V · **CLO:** {CLO}
"""))
    C.append(cell_code('''
run("python src/s9_external_hillstrom.py")
show_table("s9_acceptance.csv", 8)
show_table("s9_ate_rct.csv", 6)
show_table("s9_qini_test.csv", 8)
show_fig("fig_s9_qini_curves.png")
'''))
    nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out = ROOT / "notebooks/PTDLNC_full_pipeline.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--appendix", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--notebook", action="store_true")
    a = ap.parse_args()
    every = not (a.appendix or a.summary or a.notebook)
    if every or a.appendix:
        print("→", build_appendix().relative_to(ROOT))
    if every or a.summary:
        print("→", build_summary().relative_to(ROOT))
    if every or a.notebook:
        print("→", build_notebook().relative_to(ROOT))


if __name__ == "__main__":
    main()
