"""Trang 1 – Tổng quan điều hành (Executive Overview)."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from components import charts as C
from components.explanations import csv_button, explain, note
from components.gate_status import POP, gate_rows
from components.kpi_cards import banner, gbp, html_table, kpi_card, kpi_row, page_header, vn
from services import data_service as D

page_header("Executive Overview", "Tổng quan điều hành",
            "Kết quả kiểm định trên Locked Test (3 kỳ, 07–08/2011) – mở đúng một lần sau khi đóng băng mã, mô hình và cấu hình. "
            "Ngưỡng của mọi cổng được đăng ký trước.")

gates = D.table("s8_locked_gates")
g4 = D.table("s8_locked_g4a_comparison")
g7 = D.table("s8_locked_g7_segments")
main = g4[(g4.quan_the == "toan_bo") & g4.so_sanh.str.startswith("CHÍNH")].iloc[0]
sec = g4[(g4.quan_the == "toan_bo") & g4.so_sanh.str.startswith("PHỤ")].iloc[0]
official = gates[(gates.quan_the == "toan_bo") & gates.cham_chinh_thuc.astype(bool)]
n_pass, n_all = int(official.dat.sum()), len(official)
lo, hi = D.AUDIT_G4A_CI

banner("fail", "Quyết định phát hành · Release decision", "NO-GO – FULL ROLLOUT",
       [f"<b>{n_pass}/{n_all} tiêu chí đạt</b> trên 7 nhóm cổng (G4 gồm hai tiêu chí G4a và G4b).",
        "<b>Nguyên nhân trực tiếp:</b> G4b trượt – regret chuẩn hóa 0,894 &gt; ngưỡng 0,35. Ngưỡng này không thể đạt: "
        "ngay chính sách biết kỳ vọng thật cũng có regret 0,742.",
        f"<b>Nguyên nhân gốc:</b> thiếu sức mạnh thống kê – lợi thế lợi nhuận của MILP (+{gbp(main.chenh_lech_tuyet_doi)}) "
        f"có CI [{gbp(lo)}; {gbp(hi, sign=True)}] chứa 0; giá trị thực hiện trên OOT cũng có CI chứa 0."],
       rec="<b>Khuyến nghị:</b> chạy <b>thí điểm</b> với nhóm đối chứng 50 % trong 26–91 kỳ (dừng sớm khi CI giá trị "
           "chương trình loại trừ 0 ở hai kỳ liên tiếp); nới giới hạn cuộc gọi K theo giá bóng.")

# ---------------------------------------------------------------- KPI (mỗi chỉ số một benchmark phù hợp)
g2 = official[official.gate == "G2"].iloc[0]
g6 = official[official.gate == "G6"].iloc[0]
kpi_row([
    kpi_card("Lợi nhuận gia tăng (Locked)", gbp(main.chinh_sach),
             f"{gbp(main.chenh_lech_tuyet_doi, sign=True)} so với luật cũ qua Router", "pass",
             f"CI chênh lệch 95 %: [{gbp(lo)}; {gbp(hi, sign=True)}] – chứa 0"),
    kpi_card("Regret chuẩn hóa (G4b)", "0,894", "Ngưỡng ≤ 0,35 · TRƯỢT", "fail",
             "Mốc chính sách biết kỳ vọng thật: 0,742 · Regret B: 0,709 · CI: không tính (không đăng ký trước)"),
    kpi_card("ECE hiệu chỉnh (G2)", "0,020", "Ngưỡng ≤ 0,03 · ĐẠT", "pass", "CI 95 %: [0,017; 0,035] – ngưỡng nằm trong CI"),
    kpi_card("Độ trễ P95 (G6)", f"{vn(float(g6.gia_tri), 2)} s", "Ngưỡng ≤ 30 s · ĐẠT", "pass", "90 lượt đo trên 3 kỳ"),
    kpi_card("Vi phạm ràng buộc (G5)", "0", "Ngưỡng = 0 · ĐẠT", "pass", "Trên quyết định gửi thực tế"),
])
st.write("")

# ---------------------------------------------------------------- bảng cổng
st.subheader("Trạng thái từng cổng phát hành")
if True:
    c1, c2, _ = st.columns([2, 1.2, 2])
    pop = c1.selectbox("Quần thể", list(POP), format_func=POP.get, key="ov_pop")
    sec_on = c2.toggle("Hiện so sánh phụ", value=False, key="ov_sec")
    html_table(["Cổng", "Nhóm", "Tiêu chí", "Kết quả", "Ngưỡng", "Trạng thái", "Nguyên nhân đạt / trượt"],
               gate_rows(gates, pop, sec_on), ["code", "", "crit", "", "", "", "note wrap"])
    if pop != "toan_bo":
        note("Quần thể con chỉ mang tính mô tả: cổng được chấm chính thức trên toàn bộ Locked Test.")
    csv_button(gates, "release_gates_locked.csv", key="dl_gates")

st.write("")
left, right = st.columns([1.25, 1], gap="large")
with left:
    st.subheader("Giá trị chính sách trên Locked Test")
    fig = go.Figure()
    labels = ["MILP qua Router v2.1", "Luật cũ qua Router v2.1", "Luật cũ thuần"]
    real = [main.chinh_sach, main.doi_chung, sec.doi_chung]
    exp = [main.chinh_sach_ky_vong, main.doi_chung_ky_vong, sec.doi_chung_ky_vong]
    fig.add_bar(y=labels, x=real, orientation="h", name="Hiện thực", marker_color=[C.MILP, C.BASE, "#C3CAD6"],
                text=[gbp(v) for v in real], textposition="outside")
    fig.add_bar(y=labels, x=exp, orientation="h", name="Kỳ vọng (oracle)", marker_color="rgba(0,0,0,0)",
                marker_line=dict(color=[C.MILP, C.BASE, "#8A94A6"], width=2), text=[gbp(v) for v in exp],
                textposition="outside")
    fig.add_vline(x=main.doi_chung * 1.10, line=dict(color=C.WARN_FILL, dash="dot"),
                  annotation_text="ngưỡng G4a", annotation_position="bottom left",
                  annotation_font=dict(size=11, color=C.WARN))
    fig.update_layout(barmode="group", title="Lợi nhuận gia tăng so với a0 (£, cộng 3 kỳ)", yaxis=dict(autorange="reversed"),
                      xaxis=dict(range=[0, max(exp + real) * 1.25]))
    C.show(fig, 340)
    explain("Cột đặc là lợi nhuận <b>hiện thực</b>, khung rỗng là lợi nhuận <b>kỳ vọng</b> theo đáp án bán tổng hợp. "
            "MILP vượt ngưỡng G4a theo ước lượng điểm, nhưng hai luồng chỉ khác nhau ở 553/12 947 dòng nên chênh lệch "
            "vẫn nằm trong nhiễu.")

with right:
    st.subheader("Công bằng phân khúc (G7)")
    d = g7[(g7.quan_the == "toan_bo") & g7.so_sanh.str.startswith("CHÍNH")]
    fig = go.Figure()
    fig.add_bar(x=d.phan_khuc, y=d.chinh_sach, name="MILP qua Router", marker_color=C.MILP)
    fig.add_bar(x=d.phan_khuc, y=d.doi_chung, name="Luật cũ qua Router", marker_color=C.BASE)
    for r in d.itertuples():
        fig.add_annotation(x=r.phan_khuc, y=max(r.chinh_sach, r.doi_chung) * 1.08,
                           text=f"<b>{vn(r.chenh_tuong_doi * 100, 1, sign=True)} %</b>", showarrow=False,
                           font=dict(color=C.PASS))
    fig.update_layout(barmode="group", yaxis_title="£ (3 kỳ)", yaxis=dict(range=[0, d.chinh_sach.max() * 1.22]))
    C.show(fig, 340)
    explain("Không phân khúc nào kém luật cũ quá 5 %: UK +18,8 %, non-UK +4,7 %.")
