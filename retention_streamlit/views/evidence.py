"""Trang 6 – Minh chứng & tái lập (Evidence & Reproducibility)."""
from __future__ import annotations

import platform
from importlib import metadata

import numpy as np
import pandas as pd
import streamlit as st

from components.explanations import csv_button, explain, note
from components.kpi_cards import badge, gbp, html_table, kpi_card, kpi_row, page_header, vn
from services import data_service as D
from services import hash_service as H
from services import milp_service as M

page_header("Evidence & Reproducibility", "Minh chứng & tái lập",
            "Nguồn dữ liệu, kiến trúc, kết quả kiểm định, môi trường chạy và kiểm tra toàn vẹn file bằng SHA-256.")

c1, c2, c3 = st.columns(3)
c1.link_button("GitHub repository", D.GITHUB_URL, icon=":material/code:", width="stretch")
c2.link_button("Google Colab notebook", D.COLAB_URL, icon=":material/play_circle:", width="stretch")
c3.link_button("Tài liệu kỹ thuật (PDF)", D.REPORT_URL, icon=":material/description:", width="stretch")
st.write("")

t1, t2, t3, t4 = st.tabs(["Kiến trúc hệ thống", "Dữ liệu & kết quả", "Môi trường & phụ thuộc", "Kiểm tra SHA-256"])

with t1:
    st.image(str(D.FIGS / "fig_graphic_abstract.png"), width="stretch",
             caption="Tổng quan hệ thống – dữ liệu, pipeline tám giai đoạn và kết quả cổng phát hành")
    steps = [("STAGE 1", "Dữ liệu point-in-time", "Embargo 42 ngày, entity holdout 15 %, 6 kiểm thử rò rỉ"),
             ("STAGE 2", "Dự báo & hiệu chỉnh", "LightGBM + Platt; Ŝ + conformal 90 %"),
             ("STAGE 3", "Giải thích", "PFI, SHAP, ổn định, faithfulness"),
             ("STAGE 4", "Nhân quả & uplift", "IPW, DML, CausalForestDML, Qini"),
             ("STAGE 5", "PTO vs SPO+", "Regret, quan hệ MSE ↔ regret"),
             ("STAGE 6", "MILP toàn quy mô", "HiGHS, giá bóng, độ nhạy, quy mô"),
             ("STAGE 7", "Action Router", "Guardrail, HITL, đối chứng 10 %, nhật ký"),
             ("STAGE 8", "Release Gates & MLOps", "Đóng băng, Locked 1 lần, OOT, playbook")]
    st.markdown('<div class="rds-flow">' + "".join(
        f'<div class="rds-step"><div class="n">{n}</div><div class="t">{t}</div><div class="d">{d}</div></div>' for n, t, d in steps)
        + "</div>", unsafe_allow_html=True)
    explain("Mỗi giai đoạn là một module <code>src/sX_*.py</code> độc lập, đọc tham số từ <code>configs/config.yaml</code> và ghi "
            "manifest SHA-256 cho mọi đầu ra. App này đọc đầu ra đó; riêng MILP được chạy lại bằng chính mã nguồn gốc.")

with t2:
    st.markdown("##### Nguồn dữ liệu")
    html_table(["Bộ dữ liệu", "Vai trò", "Quy mô", "Giấy phép"], [
        ["Online Retail II (UCI)", "Hành vi mua thật của nhà bán lẻ quà tặng trực tuyến UK, 12/2009–12/2011", "794 163 dòng giao dịch sau làm sạch; 151 013 cặp khách × t0", "CC BY 4.0"],
        ["Lớp can thiệp bán tổng hợp (Stage 1b)", "Y(0) thật; Y(a1..a3) sinh từ cơ chế đã biết; chính sách cũ 20 % ngẫu nhiên + 80 % luật RFM", "Có đáp án (oracle) để kiểm tra phương pháp", "—"],
        ["Hillstrom (2008)", "RCT thật 3 nhánh email – kiểm định ngoài", "64 000 khách", "Không kèm trong repo (tải tự động)"]],
        ["code", "", "", ""])
    st.markdown("##### Chia tập theo thời gian")
    sp = D.table("s1_split_summary")
    html_table(["Tập", "Số dòng", "Số khách", "Số kỳ", "Khoảng t0", "Tỷ lệ mua lại (a0)", "Dòng entity holdout"],
               [[r.split, vn(r.rows), vn(r.customers), str(r.n_t0), f"{r.t0_first} → {r.t0_last}", vn(r.base_rate, 4),
                 vn(r.entity_holdout_rows)] for r in sp.itertuples()],
               ["code", "num", "num", "num", "num", "num", "num"])
    st.markdown("##### Kết quả kiểm định chính thức (Locked Test)")
    g = D.table("s8_locked_gates")
    g = g[(g.quan_the == "toan_bo") & g.cham_chinh_thuc.astype(bool)]
    html_table(["Cổng", "Kết quả", "Ngưỡng", "Trạng thái"],
               [[r.gate, str(r.gia_tri), str(r.nguong), badge("pass", "ĐẠT") if r.dat else badge("fail", "TRƯỢT")] for r in g.itertuples()],
               ["code", "num", "num", ""])

with t3:
    cm = D.table("s8_compat_matrix")
    st.markdown("##### Ma trận tương thích của pipeline gốc")
    st.dataframe(cm, hide_index=True, width="stretch")
    st.markdown("##### Môi trường đang chạy app này")
    pk = ["streamlit", "pandas", "numpy", "pyarrow", "scipy", "highspy", "plotly", "PyYAML"]
    vers = []
    for p in pk:
        try:
            vers.append([p, metadata.version(p)])
        except metadata.PackageNotFoundError:
            vers.append([p, "không cài"])
    html_table(["Thành phần", "Phiên bản"], [["Python", platform.python_version()], ["Nền tảng", platform.platform()]] + vers,
               ["code", "num"])
    note("App không cần EconML/DoubleML/LightGBM: MILP, Greedy và LP relaxation được đọc nguyên văn từ "
         "<code>src/s5_spo.py</code>, <code>src/s6_milp.py</code>, <code>src/s5b_dr_diagnostics.py</code> và chạy với numpy + HiGHS.")
    with st.expander("Kiểm tra nhất quán: chạy lại MILP 7 kỳ valid + calib và so với kết quả đã đóng băng (≈ 10 giây)"):
        if st.button("Chạy kiểm tra nhất quán", icon=":material/fact_check:"):
            E = M.engine()
            df = D.sandbox_frame()
            res = {"MILP trên V̂": [], "Greedy (V̂/mệnh giá)": [], "Luật RFM cũ": []}
            prog = st.progress(0.0)
            per = D.sandbox_periods()
            for k, (s, t0) in enumerate(per):
                sub = df[(df.split == s) & (df.t0 == t0)].reset_index(drop=True)
                ins = M.build(E, sub, 8000, 60)
                V = M.value_matrix(E, sub, 0.30)
                _, Ue = E.evaluate_utilities(M.truth(sub), 0.30)
                w = M.solve_milp(E, ins, V, 0.25)["w"]
                M._set_a2_share(E, 0.25)
                wg = E.greedy(ins, V)
                wl = M.solve_milp(E, ins, M.legacy_matrix(E, sub), 0.25)["w"]
                for nm, ww in [("MILP trên V̂", w), ("Greedy (V̂/mệnh giá)", wg), ("Luật RFM cũ", wl)]:
                    res[nm].append(((ww[:, 1:] * V).sum(), (ww * Ue).sum()))
                prog.progress((k + 1) / len(per))
            ref = D.table("s6_main_summary").set_index("phuong_phap")
            rows = []
            for nm, v in res.items():
                plan, exp = np.mean([x[0] for x in v]), np.mean([x[1] for x in v])
                ok = abs(plan - ref.loc[nm, "V_ke_hoach_TB"]) < 0.5 and abs(exp - ref.loc[nm, "thuc_ky_vong_TB"]) < 0.5
                rows.append([nm, gbp(plan, 2), gbp(ref.loc[nm, "V_ke_hoach_TB"], 2), gbp(exp, 2), gbp(ref.loc[nm, "thuc_ky_vong_TB"], 2),
                             badge("pass", "KHỚP") if ok else badge("fail", "LỆCH")])
            html_table(["Phương pháp", "V̂ TB (app)", "V̂ TB (đóng băng)", "Kỳ vọng TB (app)", "Kỳ vọng TB (đóng băng)", "Kết quả"],
                       rows, ["code", "num", "num", "num", "num", ""])
            st.caption("Nguồn mã: " + " · ".join(f"src/{f} (sha256 {v['sha256_file'][:10]}…)" for f, v in E.origin.items()))

with t4:
    fz = D.freeze_info()
    kpi_row([kpi_card("File trong manifest đóng băng", vn(fz["n_files"]), "mã, mô hình, cấu hình, dữ liệu", "neutral"),
             kpi_card("Đóng băng lúc", fz["created_utc"][:16].replace("T", " "), "UTC", "neutral"),
             kpi_card("Locked Test mở lúc", fz["opened_utc"][:16].replace("T", " "), "UTC · đúng một lần", "neutral")])
    st.write("")
    st.markdown("**Quy trình:** ① đọc file thật trên server → ② tính SHA-256 → ③ lấy hash tham chiếu từ "
                "<code>outputs/manifests/s8_freeze_manifest.json</code> → ④ so sánh → ⑤ MATCH / MISMATCH.", unsafe_allow_html=True)
    if st.button("Kiểm tra toàn vẹn ngay", type="primary", icon=":material/verified_user:"):
        bar = st.progress(0.0, text="Đang băm…")
        tab, sec = H.verify(lambda f, name: bar.progress(f, text=f"Đang băm {name}"))
        bar.empty()
        n_ok = int((tab["Kết quả"] == "MATCH").sum())
        if n_ok == len(tab):
            st.success(f"MATCH {n_ok}/{len(tab)} file · {vn(sec, 1)} giây", icon=":material/check_circle:")
        else:
            st.error(f"{len(tab) - n_ok} file không khớp hoặc thiếu", icon=":material/error:")
        st.dataframe(tab, hide_index=True, width="stretch", height=420,
                     column_config={"Bytes": st.column_config.NumberColumn(format="%d")})
        csv_button(tab, "sha256_verify.csv", key="dl_hash")
    st.info("Hash chỉ xác nhận **tính toàn vẹn** của file (không bị sửa sau khi đóng băng). Nó **không** chứng minh nguồn gốc "
            "dữ liệu, cũng không tự chứng minh việc đăng ký trước – bằng chứng đăng ký trước là thời điểm sửa file và hash đóng "
            "băng, không có lịch sử git sớm (xem mục công khai trong tài liệu kỹ thuật).", icon=":material/info:")
