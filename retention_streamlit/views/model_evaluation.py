"""Trang 4 – Dự báo & nhân quả (Model Evaluation)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import charts as C
from components.explanations import csv_button, explain, note
from components.kpi_cards import banner, html_table, kpi_card, kpi_row, page_header, vn
from services import data_service as D
from services import model_service as MS

page_header("Model Evaluation", "Dự báo & nhân quả",
            "Chất lượng mô hình xác suất mua nền, giải thích, tác động nhân quả và khả năng xếp hạng uplift.")

mode = st.segmented_control("Chế độ kiểm định", ["Benchmark bán tổng hợp (có oracle)", "Hillstrom RCT (can thiệp thật)"],
                            default="Benchmark bán tổng hợp (có oracle)", key="me_mode")

# =========================================================================================== Hillstrom
if mode and mode.startswith("Hillstrom"):
    banner("info", "Kiểm định ngoài · RCT thật", "Hillstrom (2008): 64 000 khách, 3 nhánh ngẫu nhiên cân bằng",
           ["Can thiệp là <b>ngẫu nhiên thật</b> (không gửi / email nam / email nữ) nên ATE ước lượng không chệch.",
            "<b>Không</b> quan sát được tác động cá nhân (CATE thật): chỉ đánh giá xếp hạng bằng Qini trên tập test.",
            "Kết quả chính đăng ký trước: chuyển đổi (conversion); kết quả phụ: chi tiêu (spend)."])
    ate = D.table("s9_ate_rct")
    q = D.table("s9_qini_test")
    pv = D.table("s9_policy_value_test")
    names = {"a1": "Email nam", "a2": "Email nữ"}
    t1, t2, t3 = st.tabs(["ATE (RCT)", "Qini trên test", "Giá trị chính sách"])
    with t1:
        oc = st.radio("Kết quả", ["conversion", "visit", "spend"], horizontal=True, key="hs_oc",
                      format_func={"conversion": "Chuyển đổi", "visit": "Lượt ghé thăm", "spend": "Chi tiêu ($)"}.get)
        d = ate[ate.outcome == oc]
        fig = go.Figure()
        for tap, col in [("train", C.BASE), ("test", C.MILP), ("toàn bộ", C.NAVY)]:
            x = d[d.tap == tap]
            C.ci_dot(fig, [f"{names[a]}" for a in x.hanh_dong], x.ATE, x.ci_lo, x.ci_hi, tap, col)
        fig.add_vline(x=0, line=dict(color=C.INK, width=1))
        fig.update_layout(title="ATE so với nhánh không gửi (CI 95 %)", xaxis_title="chênh lệch trung bình", yaxis=dict(autorange="reversed"))
        C.show(fig, 300)
        explain("Cả hai email đều có tác động trung bình dương có ý nghĩa: can thiệp thật sự có hiệu quả ở cấp quần thể.")
        csv_button(d, f"hillstrom_ate_{oc}.csv", key="dl_hs_ate")
    with t2:
        oc2 = st.radio("Kết quả", ["conversion", "spend"], horizontal=True, key="hs_oc2",
                       format_func={"conversion": "Chuyển đổi (chính)", "spend": "Chi tiêu (phụ)"}.get)
        d = q[q.outcome == oc2]
        fig = go.Figure()
        for a_, col in [("a1", C.MILP), ("a2", C.CHAL)]:
            x = d[d.hanh_dong == a_]
            C.ci_dot(fig, x.learner, x.qini, x.ci_lo, x.ci_hi, names[a_], col)
        fig.add_vline(x=0, line=dict(color=C.FAIL, dash="dash"))
        fig.update_layout(title="Hệ số Qini trên test ± CI bootstrap 95 %", xaxis_title="Qini", yaxis=dict(autorange="reversed"))
        C.show(fig, 320)
        n_ok = int(d[d.la_learner_thang].ci_khong_chua_0.sum())
        explain(f"Learner thắng (CausalForestDML): <b>{n_ok}/2</b> email có CI không chứa 0. Với tỷ lệ chuyển đổi nền ~0,6 %, "
                "tín hiệu CATE quá yếu so với nhiễu để xếp hạng tốt hơn ngẫu nhiên có ý nghĩa.")
        csv_button(d, f"hillstrom_qini_{oc2}.csv", key="dl_hs_q")
    with t3:
        fig = go.Figure()
        C.ci_dot(fig, pv.chinh_sach, pv.gia_tri_ipw_tb, pv.ci_lo, pv.ci_hi, "IPW", C.MILP)
        fig.update_layout(title="Giá trị chính sách ước lượng IPW trên test (CI 95 %)", yaxis=dict(autorange="reversed"),
                          xaxis_title="giá trị trung bình mỗi khách")
        C.show(fig, 380)
        explain("Với trần gửi 30 %, MILP trùng Greedy (chỉ một ràng buộc đếm) và không hơn gửi ngẫu nhiên 30 % có ý nghĩa.")
    st.stop()

# =========================================================================================== bán tổng hợp
note("Benchmark bán tổng hợp: kết quả dưới a0 là hành vi mua thật (Online Retail II); kết quả dưới a1–a3 sinh từ cơ chế đã biết, "
     "nên có <b>đáp án (oracle)</b> để kiểm tra phương pháp – nhưng không chứng minh hiệu quả thật của voucher.")
tb, tx, tc, tu = st.tabs(["Baseline Model", "Explainability", "Causal Inference", "Uplift"])

with tb:
    met = D.table("s2_metrics_valid")
    pl = met[(met.phan_khuc == "toan_bo") & (met.phuong_phap == "platt")].iloc[0]
    g2 = D.table("s8_locked_g2_calibration")
    g2t = g2[g2.quan_the == "toan_bo"].iloc[0]
    kpi_row([kpi_card("ROC-AUC (valid)", vn(pl.roc_auc, 3), f"CI [{vn(pl.roc_auc_ci_lo, 3)}; {vn(pl.roc_auc_ci_hi, 3)}]", "neutral"),
             kpi_card("PR-AUC (valid)", vn(pl.pr_auc, 3), f"tỷ lệ dương {vn(pl.ty_le_duong, 3)}", "neutral"),
             kpi_card("Brier (valid, Platt)", vn(pl.brier, 3), f"thô {vn(met.iloc[0].brier, 3)}", "pass"),
             kpi_card("ECE valid (Platt)", vn(pl.ece, 3), "ngưỡng 0,03 · không đạt trên valid", "warn",
                      "Trôi mùa: valid rơi vào tháng 1–3"),
             kpi_card("ECE Locked (G2)", vn(g2t.ece_hieu_chinh, 3),
                      f"CI [{vn(g2t.ece_hieu_chinh_ci_lo, 3)}; {vn(g2t.ece_hieu_chinh_ci_hi, 3)}]", "pass", f"n = {vn(g2t.n_a0)} dòng a0")])
    st.write("")
    a, b = st.columns(2, gap="large")
    with a:
        c1, c2 = st.columns(2)
        split = c1.selectbox("Tập", ["valid", "calib", "locked"], key="me_split",
                             format_func={"valid": "Valid (chọn mô hình)", "calib": "Calib (khớp Platt)",
                                          "locked": "Locked (chỉ đọc)"}.get)
        calib = c2.radio("Hiệu chỉnh", ["Platt", "Thô"], horizontal=True, key="me_cal")
        col = "mu0_cal" if calib == "Platt" else "mu0_raw"
        r, ece = MS.reliability(split, col)
        r2, ece2 = MS.reliability(split, "mu0_raw" if col == "mu0_cal" else "mu0_cal")
        fig = go.Figure()
        fig.add_scatter(x=[0, 0.8], y=[0, 0.8], mode="lines", line=dict(color="#C9D0DB", dash="dash"), name="Hoàn hảo")
        fig.add_scatter(x=r2.p_TB, y=r2.ty_le_thuc, mode="lines+markers", name=("Thô" if calib == "Platt" else "Platt"),
                        line=dict(color=C.BASE), marker=dict(size=6), opacity=0.6)
        fig.add_scatter(x=r.p_TB, y=r.ty_le_thuc, mode="lines+markers", name=calib, line=dict(color=C.MILP, width=3),
                        marker=dict(size=r.n / r.n.max() * 14 + 4))
        fig.update_layout(title=f"Reliability diagram – {split} · ECE {calib} = {vn(ece, 3)}", xaxis_title="xác suất dự báo (TB bin)",
                          yaxis_title="tỷ lệ mua lại thực tế", xaxis=dict(range=[0, 0.8]), yaxis=dict(range=[0, 0.8]))
        C.show(fig, 380)
        explain("15 bin cùng khối lượng trên các dòng a0. Điểm nằm trên đường chéo là hiệu chỉnh hoàn hảo; điểm thô nằm dưới "
                "đường chéo nghĩa là mô hình dự báo cao hơn thực tế. Platt kéo đường về gần đường chéo mà không đổi thứ hạng (AUC giữ nguyên).")
    with b:
        pp = MS.per_period(split)
        fig = go.Figure()
        fig.add_scatter(x=pp.t0, y=pp.thuc_te, name="Thực tế", mode="lines+markers", line=dict(color=C.INK, width=3))
        fig.add_scatter(x=pp.t0, y=pp.du_bao_platt, name="Dự báo Platt", mode="lines+markers", line=dict(color=C.MILP))
        fig.add_scatter(x=pp.t0, y=pp.du_bao_tho, name="Dự báo thô", mode="lines+markers", line=dict(color=C.BASE, dash="dot"))
        fig.update_layout(title="Dự báo so với thực tế theo kỳ t0", yaxis_title="tỷ lệ mua lại", yaxis_tickformat=".0%")
        C.show(fig, 380)
        explain("Khoảng cách giữa dự báo và thực tế ở từng kỳ cho thấy trôi theo mùa: bộ hiệu chỉnh khớp trên tháng 4–5 "
                "không hợp với tháng 1–3 (valid) nhưng hợp với tháng 7–8 (locked).")
    with st.expander("Bảng chỉ số đầy đủ theo phân khúc (valid)"):
        st.dataframe(met, hide_index=True, width="stretch")
        csv_button(met, "s2_metrics_valid.csv", key="dl_met")

with tx:
    sh = D.table("s3_shap_importance").sort_values("mean_abs_shap_logodds", ascending=False).head(15)
    pf = D.table("s3_pfi_group")
    a, b = st.columns([1.4, 1], gap="large")
    with a:
        grp_col = {"RFM": C.MILP, "Nhịp mua": C.CHAL, "Lịch": "#EA7A1A", "Quốc gia": C.BASE, "Hủy đơn": C.FAIL}
        fig = go.Figure(go.Bar(y=sh.nhan_nghiep_vu, x=sh.mean_abs_shap_logodds, orientation="h",
                               marker_color=[grp_col.get(g, C.NAVY) for g in sh.nhom],
                               customdata=sh.nhom, hovertemplate="%{y} (%{customdata}): %{x:.3f}<extra></extra>"))
        fig.update_layout(title="SHAP – |giá trị| trung bình (log-odds), 15 đặc trưng hàng đầu", yaxis=dict(autorange="reversed"))
        C.show(fig, 460)
        explain("SHAP đo mỗi đặc trưng đẩy xác suất mua nền lên/xuống bao nhiêu so với mức nền. Màu theo nhóm: "
                "xanh = RFM, tím = nhịp mua, cam = lịch. Số ngày đến Giáng sinh dẫn đầu vì nó dịch chuyển <b>mức</b> dự báo theo mùa.")
    with b:
        fig = go.Figure()
        C.ci_dot(fig, pf.nhom, pf.pfi_mean, pf.pfi_ci_lo, pf.pfi_ci_hi, "PFI", C.NAVY)
        fig.add_vline(x=0, line=dict(color=C.INK, width=1))
        fig.update_layout(title="PFI theo nhóm (tăng log loss khi hoán vị, CI 95 %)", yaxis=dict(autorange="reversed"), showlegend=False)
        C.show(fig, 300)
        explain("PFI hoán vị <b>trong</b> tập valid nên đo thông tin về thứ hạng: nhóm RFM gần như chiếm toàn bộ. "
                "Lịch ≈ 0 vì gần như không đổi trong 4 kỳ valid – khác SHAP, nhưng không mâu thuẫn.")
        note("SHAP và PFI giải thích mô hình <b>dự báo</b> μ̂₀, không phải quan hệ nhân quả của voucher.")

with tc:
    ate = D.table("s4_ate_vs_truth")
    c1, c2 = st.columns(2)
    tap = c1.selectbox("Tập", ["train", "valid+calib"], key="me_tap")
    kq = c2.radio("Kết quả", ["purchase", "profit"], horizontal=True, key="me_kq",
                  format_func={"purchase": "Mua lại", "profit": "Lợi nhuận (£)"}.get)
    d = ate[(ate.tap == tap) & (ate.ket_qua == kq)]
    fig = go.Figure()
    pal = [C.BASE, "#93B4F8", C.MILP, C.NAVY, C.CHAL, "#EA7A1A"]
    for k, (meth, x) in enumerate(d.groupby("phuong_phap", sort=False)):
        C.ci_dot(fig, [f"{a}" for a in x.hanh_dong], x.uoc_luong, x.ci_lo, x.ci_hi, meth, pal[k % len(pal)], size=9)
    tr = d.drop_duplicates("hanh_dong")
    fig.add_scatter(y=tr.hanh_dong, x=tr.ate_that, mode="markers", name="ATE thật (oracle)",
                    marker=dict(symbol="diamond", size=15, color=C.FAIL, line=dict(color="#FFFFFF", width=1)))
    fig.update_layout(title=f"Forest plot ATE – {tap} (CI 95 % bootstrap cụm theo khách)", scattermode="group",
                      yaxis=dict(autorange="reversed"), xaxis_title="ATE so với a0", height=440)
    C.show(fig, 440)
    cover = d.groupby("phuong_phap").ci_phu_ate_that.apply(lambda s: f"{int(s.sum())}/{len(s)}").to_dict()
    explain("Kim cương đỏ là ATE thật từ cơ chế sinh dữ liệu. Ước lượng naive chệch vì chính sách cũ gửi theo RFM; "
            "IPW và DML đưa ước lượng về sát sự thật. Số CI phủ ATE thật: " +
            " · ".join(f"{k}: <b>{v}</b>" for k, v in cover.items()) + ".")
    cate = D.table("s4_cate_vs_truth")
    st.markdown("##### CATE – tương quan hạng Spearman giữa τ̂ và τ thật (valid + calib)")
    pv = cate.pivot_table(index="learner", columns="hanh_dong", values="spearman_tau_hat_vs_tau_cond").reset_index()
    html_table(["Learner", "a1", "a2", "a3"], [[r.learner] + [vn(r[a], 3) for a in ("a1", "a2", "a3")] for _, r in pv.iterrows()],
               ["code", "num", "num", "num"])
    note("Spearman chỉ dùng để đánh giá (cần đáp án); learner được chọn bằng Qini trên nhánh ngẫu nhiên của valid.")
    csv_button(d, f"ate_{tap}_{kq}.csv", key="dl_ate")

with tu:
    q = D.table("s4_qini")
    lk = D.table("s8_locked_g3_qini")
    c1, c2, c3 = st.columns(3)
    act = c1.selectbox("Hành động", ["a1", "a2", "a3"], format_func=C.ACTION_NAMES.get, key="me_act")
    lrn = c2.multiselect("Learner", sorted(q.learner.unique()), default=sorted(q.learner.unique()), key="me_lrn")
    bonf = c3.toggle("Hiệu chỉnh nhiều phép thử (Bonferroni, 3 hành động)", value=False, key="me_bonf")
    d = q[(q.hanh_dong == act) & q.learner.isin(lrn) & (q.tap == "valid+calib")]
    fig = go.Figure()
    for k, r in enumerate(d.itertuples()):
        C.ci_dot(fig, [r.learner], [r.qini], [r.qini_ci_lo], [r.qini_ci_hi],
                 r.learner, C.MILP if r.duoc_chon else C.BASE, size=12 if r.duoc_chon else 9)
    l1 = lk[(lk.hanh_dong == act) & (lk.quan_the == "toan_bo")]
    if len(l1):
        r = l1.iloc[0]
        C.ci_dot(fig, ["CausalForestDML · LOCKED"], [r.qini], [r.qini_ci_lo], [r.qini_ci_hi], "Locked (G3)", C.NAVY,
                 symbol="square", size=12)
    fig.add_vline(x=0, line=dict(color=C.FAIL, dash="dash"))
    fig.update_layout(title=f"Hệ số Qini ± CI 95 % – {C.ACTION_NAMES[act]}", showlegend=False, xaxis_title="Qini",
                      yaxis=dict(autorange="reversed"))
    C.show(fig, 360)
    rows = []
    for r in d.itertuples():
        rows.append([r.learner + (" ✓ chọn" if r.duoc_chon else ""), "valid+calib", vn(r.n), vn(r.n_dieu_tri), vn(r.qini, 3),
                     f"[{vn(r.qini_ci_lo, 3)}; {vn(r.qini_ci_hi, 3)}]"])
    if len(l1):
        r = l1.iloc[0]
        rows.append(["CausalForestDML", "LOCKED", vn(r.n), vn(r.n_dieu_tri), vn(r.qini, 3),
                     f"[{vn(r.qini_ci_lo, 3)}; {vn(r.qini_ci_hi, 3)}]"])
    html_table(["Learner", "Tập", "n (nhánh ngẫu nhiên)", "n điều trị", "Qini", "CI 95 %"], rows,
               ["code", "", "num", "num", "num", "num"])
    if bonf:
        st.warning("Với hiệu chỉnh Bonferroni cho 3 phép thử (α = 0,05/3 ≈ 0,017), cần CI 98,3 % thay cho CI 95 %. Theo kiểm toán "
                   "độc lập, cận dưới CI của a3 trên Locked (0,010) sát 0 và **không còn dương** sau hiệu chỉnh – G3 khi đó trượt "
                   "(chỉ còn a2). Cổng G3 đăng ký trước dùng CI 95 % không hiệu chỉnh.", icon=":material/rule:")
    explain("Qini dương với CI không chứa 0 nghĩa là xếp hạng theo τ̂ tốt hơn ngẫu nhiên. Valid + calib chỉ có ~2 500 dòng "
            "ngẫu nhiên mỗi cặp nên CI rộng; Locked có ~1 300 dòng – vì vậy G3 được dự báo trước chỉ đạt với xác suất 4–7 %.")
    csv_button(d, f"qini_{act}.csv", key="dl_qini")
    with st.expander("Ảnh đường Qini gốc (pipeline offline)"):
        st.image(str(D.FIGS / "fig_s4_qini_curves.png"), width="stretch")
