"""Trang 5 – Giám sát MLOps (Model Monitoring) trên 5 kỳ Out-of-Time phát lại."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from components import charts as C
from components.explanations import csv_button, explain, note
from components.kpi_cards import badge, gbp, html_table, kpi_card, kpi_row, page_header, vn
from services import data_service as D

page_header("Model Monitoring", "Giám sát MLOps",
            "Phát lại tuần tự 5 kỳ Out-of-Time (12/09 – 07/11/2011, mùa cao điểm quý IV) như vận hành thật với mô hình Current.")

mon = D.config()["monitoring"] if "monitoring" in D.config() else {}
PSI_W, PSI_A = mon.get("psi_warn", 0.10), mon.get("psi_alert", 0.25)
ECE_W, ECE_A = 0.03, mon.get("ece_alert", 0.05)
TVD_A = mon.get("action_tvd_alert", 0.20)
SEASONAL = {"days_to_xmas", "t0_month", "tenure_days"}

lat = D.table("s8_oot_latency")
cal = D.table("s8_oot_calibration_matured")
psi = D.table("s8_oot_psi")
tvd = D.table("s8_oot_action_tvd")
dr = D.table("s8_oot_dr_value")
pb = D.table("s8_oot_playbook")
ky = sorted(tvd.ky.unique())
kylab = {k: pd.Timestamp(k).strftime("%d/%m") for k in ky}

# ---------------------------------------------------------------- ngày phát lại: nhãn trưởng thành hay chưa
DATES = ky + ["2011-12-07"]
asof = st.select_slider("Ngày phát lại (as-of) – chỉ hiển thị chỉ số cần nhãn khi nhãn đã trưởng thành",
                        options=DATES, value=DATES[-1], format_func=lambda d: pd.Timestamp(d).strftime("%d/%m/%Y"), key="ml_asof")
asof_ts = pd.Timestamp(asof)
seen = [k for k in ky if pd.Timestamp(k) <= asof_ts]
cal["matured"] = pd.to_datetime(cal.ngay_tinh) <= asof_ts
cur = cal[(cal.phien_ban == "Current") & cal.ky.isin(seen)]
chal = cal[(cal.phien_ban.str.startswith("Challenger (")) & cal.ky.isin(seen)]

# ---------------------------------------------------------------- trạng thái tổng
last_ece = cur[cur.matured].ece.iloc[-1] if cur.matured.any() else None
p_feat = psi[(psi.ky.isin(seen)) & (psi.loai == "đặc trưng")]
alarm_feats = p_feat[(p_feat.psi >= PSI_A) & ~p_feat.bien.isin(SEASONAL)].bien.nunique()
p_score = psi[(psi.ky.isin(seen)) & (psi.loai == "điểm số")]
score_alarm = p_score[p_score.psi >= PSI_A].bien.nunique()
tvd_max = tvd[tvd.ky.isin(seen)].action_tvd.max() if seen else 0
p95 = lat[(lat.khau == "tong") & lat.ky.isin(seen)].P95.max() if seen else 0
kpi_row([
    kpi_card("ECE Current (kỳ trưởng thành gần nhất)", vn(last_ece, 3) if last_ece is not None else "Chưa đủ nhãn",
             f"cảnh báo {vn(ECE_W, 2)} · báo động {vn(ECE_A, 2)}",
             "neutral" if last_ece is None else ("fail" if last_ece >= ECE_A else "warn" if last_ece > ECE_W else "pass"),
             f"{int(cur.matured.sum())}/{len(seen)} kỳ đã có nhãn"),
    kpi_card("PSI đặc trưng báo động", f"{alarm_feats} biến", "không tính biến mùa vụ / cấu trúc", "fail" if alarm_feats >= 3 else
             "warn" if alarm_feats else "pass", f"Điểm số báo động: {score_alarm}"),
    kpi_card("Action TVD lớn nhất", vn(tvd_max, 3), f"ngưỡng báo động {vn(TVD_A, 2)}", "pass" if tvd_max < TVD_A else "fail"),
    kpi_card("Độ trễ P95 lớn nhất", f"{vn(p95, 2)} s", "ngưỡng 30 s", "pass" if p95 <= 30 else "fail"),
    kpi_card("Độ trễ nhãn", "30–58 ngày", "TB 43 ngày", "warn", "Brier/ECE luôn đến muộn hơn PSI/TVD"),
])
st.write("")

t1, t2, t3, t4, t5 = st.tabs(["Hiệu chỉnh: Current vs Challenger", "Độ trôi PSI", "Độ trễ & Action TVD",
                              "Giá trị thực hiện", "Playbook & cảnh báo"])

with t1:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ECE theo kỳ", "Brier theo kỳ"), horizontal_spacing=0.08)
    ymax = max(0.1, cal.ece.max() * 1.15)
    C.threshold_bands(fig, ECE_W, ECE_A, ymax, row=1, col=1)
    for nm, d, col in [("Current", cur, C.MILP), ("Challenger (Platt theo kỳ)", chal, C.CHAL)]:
        dm = d[d.matured]
        fig.add_scatter(x=[kylab[k] for k in dm.ky], y=dm.ece, name=nm, mode="lines+markers", line=dict(color=col, width=3),
                        row=1, col=1, legendgroup=nm)
        fig.add_scatter(x=[kylab[k] for k in dm.ky], y=dm.brier, name=nm, mode="lines+markers", line=dict(color=col, width=3),
                        row=1, col=2, legendgroup=nm, showlegend=False)
    pend = [k for k in seen if k not in set(cur[cur.matured].ky)]
    for k in pend:
        fig.add_annotation(x=kylab[k], y=ECE_W / 2, text="nhãn chưa<br>trưởng thành", showarrow=False, row=1, col=1,
                           font=dict(size=10, color=C.MUTED))
    fig.update_xaxes(categoryorder="array", categoryarray=[kylab[k] for k in ky])
    fig.update_yaxes(range=[0, ymax], row=1, col=1)
    C.show(fig, 380)
    explain("Dải xanh/vàng/đỏ là vùng an toàn / cảnh báo (&gt; 0,03) / báo động (≥ 0,05). Kỳ có nhãn chưa trưởng thành "
            "<b>không</b> hiển thị ECE – không kết luận mô hình tốt khi chưa có nhãn. Tỷ lệ mua lại tăng trong mùa cao điểm "
            "trong khi dự báo đứng yên nên ECE của Current tăng dần; Challenger hiệu chỉnh lại theo kỳ tốt hơn ở 4/5 kỳ.")
    show = cal[cal.ky.isin(seen) & ~cal.phien_ban.str.contains("PHỤ")][["ky", "phien_ban", "ngay_tinh", "tre_so_voi_quyet_dinh_ngay",
                                                                         "n_a0", "ty_le_mua", "p_TB", "brier", "ece", "matured"]]
    rows = [[kylab[r.ky], r.phien_ban, pd.Timestamp(r.ngay_tinh).strftime("%d/%m/%Y"), f"{r.tre_so_voi_quyet_dinh_ngay} ngày",
             vn(r.ty_le_mua, 3) if r.matured else "—", vn(r.p_TB, 3), vn(r.brier, 3) if r.matured else "—",
             vn(r.ece, 3) if r.matured else "—",
             badge("pass", "đã trưởng thành") if r.matured else badge("grey", "chưa trưởng thành")] for r in show.itertuples()]
    html_table(["Kỳ", "Phiên bản", "Ngày có nhãn", "Trễ", "Tỷ lệ mua", "p̄ dự báo", "Brier", "ECE", "Nhãn"], rows,
               ["code", "", "num", "num", "num", "num", "num", "num", ""])
    note("Challenger-2 (huấn luyện lại μ₀) là phương án phụ: bộ hiệu chỉnh khớp trên điểm trong mẫu nên không so sánh sạch, không hiển thị ở đây.")

with t2:
    loai = st.radio("Loại biến", ["đặc trưng", "điểm số"], horizontal=True, key="ml_loai")
    p = psi[(psi.loai == loai) & (psi.ky != "Gộp mọi kỳ")]
    agg = psi[(psi.loai == loai) & (psi.ky == "Gộp mọi kỳ")].sort_values("psi", ascending=False)
    top = agg.bien.head(18 if loai == "đặc trưng" else 10).tolist()
    mat = p[p.bien.isin(top)].pivot_table(index="bien", columns="ky", values="psi").reindex(top)
    mat = mat[[k for k in ky if k in mat.columns]]
    ylab = [f"{b} ⟳" if b in SEASONAL else b for b in mat.index]
    fig = go.Figure(go.Heatmap(z=mat.values, x=[kylab[k] for k in mat.columns], y=ylab, zmin=0, zmax=0.5,
                               colorscale=[[0, "#E8F5EC"], [PSI_W / 0.5, "#FFF4D6"], [PSI_A / 0.5, "#F9C784"], [1, "#B42318"]],
                               text=[[vn(v, 2) for v in row] for row in mat.values], texttemplate="%{text}",
                               colorbar=dict(title="PSI", tickvals=[0, PSI_W, PSI_A, 0.5], ticktext=["0", "0,10", "0,25", "≥0,5"])))
    fig.update_layout(title=f"PSI theo {loai} và kỳ (so với train)", yaxis=dict(autorange="reversed"))
    C.show(fig, 520 if loai == "đặc trưng" else 340)
    explain("Màu bão hòa ở 0,5 để các mức 0,10–0,25 vẫn phân biệt được (giá trị thật ghi trong ô). Biến có dấu ⟳ là "
            "<b>mùa vụ / cấu trúc thời gian</b> (số ngày đến Giáng sinh, tháng, thâm niên): PSI cao là tất yếu khi đi qua "
            "các tháng khác train, nên <b>không</b> tự động bị coi là trôi nghiêm trọng và không tính vào quy tắc báo động.")
    csv_button(p, f"psi_{loai}.csv", key="dl_psi")

with t3:
    a, b = st.columns(2, gap="large")
    with a:
        lt = lat[(lat.khau == "tong") & (lat.ky != "Gộp mọi kỳ")]
        fig = go.Figure()
        for q_, col, dash in [("P50", C.MILP, "solid"), ("P95", C.NAVY, "dash"), ("P99", C.CHAL, "dot")]:
            fig.add_scatter(x=[kylab[k] for k in lt.ky], y=lt[q_], name=q_, mode="lines+markers", line=dict(color=col, dash=dash, width=2.5))
        fig.update_layout(title="Độ trễ đầu–cuối mỗi kỳ (30 lượt/kỳ)", yaxis_title="giây / một t0",
                          annotations=[dict(text="Ngưỡng G6 = 30 s (ngoài trục)", x=0, y=1.08, xref="paper", yref="paper",
                                            showarrow=False, font=dict(size=11, color=C.MUTED))])
        C.show(fig, 340)
        st_ = lat[(lat.ky == "Gộp mọi kỳ") & (lat.khau != "tong")]
        names = {"t_dac_trung": "Đặc trưng", "t_ma_ly_do": "Mã lý do SHAP", "t_cham_diem": "Chấm điểm mô hình",
                 "t_hop_dong_va_mien": "Hợp đồng & miền", "t_milp": "MILP", "t_router": "Router"}
        fig = go.Figure(go.Bar(y=[names.get(k, k) for k in st_.khau], x=st_.P50, orientation="h", marker_color=C.NAVY,
                               text=[f"{vn(v, 3)} s" for v in st_.P50], textposition="outside"))
        fig.update_layout(title="P50 theo khâu (gộp 5 kỳ)", yaxis=dict(autorange="reversed"), xaxis_title="giây")
        C.show(fig, 280)
        explain("Khâu chấm điểm (μ̂₀, Ŝ, ba τ̂ kèm CI) chiếm phần lớn độ trễ; MILP chỉ khoảng 0,2 s. Nút thắt vận hành "
                "không nằm ở hạ tầng tính toán mà ở độ trễ nhãn.")
    with b:
        tv = tvd[tvd.ky.isin(seen)]
        fig = go.Figure()
        for a_ in ["a1", "a2", "a3"]:
            fig.add_bar(x=[kylab[k] for k in tv.ky], y=tv[f"p_{a_}"] * 100, name=C.ACTION_NAMES[a_], marker_color=C.ACTION_COLORS[a_])
        fig.update_layout(barmode="stack", title="Tỷ lệ khách theo hành động (%)", yaxis_title="% khách (a0 là phần còn lại)")
        C.show(fig, 300)
        fig = go.Figure()
        C.threshold_bands(fig, None, TVD_A, 0.25)
        fig.add_bar(x=[kylab[k] for k in tv.ky], y=tv.action_tvd, marker_color=C.MILP, text=[vn(v, 3) for v in tv.action_tvd],
                    textposition="outside", name="Action TVD")
        fig.update_layout(title="Action TVD so với kỳ tham chiếu", yaxis=dict(range=[0, 0.25]), showlegend=False)
        C.show(fig, 300)
        explain("TVD 0,011–0,015, xa dưới ngưỡng 0,20: phân phối hành động ổn định vì ràng buộc ngân sách và cuộc gọi cố định "
                "số lượng gửi. TVD hữu ích để phát hiện lỗi hệ thống hơn là trôi dữ liệu.")

with t4:
    d = dr[dr.ky != "Gộp mọi kỳ"]
    g = dr[dr.ky == "Gộp mọi kỳ"].iloc[0]
    kpi_row([kpi_card("Giá trị chương trình (DR, gộp 5 kỳ)", gbp(g.DR_uoc_luong), "CI chứa 0 – chưa phân định", "warn",
                      f"CI 95 %: [{gbp(g.DR_ci_lo)}; {gbp(g.DR_ci_hi)}]"),
             kpi_card("Đáp án (oracle)", gbp(g.that_dap_an_hien_thuc), "nằm trong CI", "pass"),
             kpi_card("Khách đối chứng", vn(g.n_doi_chung), f"trên {vn(g.n_tap_ngau_nhien)} dòng tập ngẫu nhiên", "fail",
                      "Quá ít để CI loại trừ 0")])
    fig = go.Figure()
    C.ci_dot(fig, [kylab[k] for k in d.ky], d.DR_uoc_luong, d.DR_ci_lo, d.DR_ci_hi, "DR", C.MILP)
    fig.add_scatter(y=[kylab[k] for k in d.ky], x=d.that_dap_an_hien_thuc, mode="markers", name="Thật (oracle)",
                    marker=dict(symbol="diamond", size=13, color=C.FAIL))
    fig.add_vline(x=0, line=dict(color=C.INK, width=1))
    fig.update_layout(title="Giá trị thực hiện theo kỳ – ước lượng DR trên nhóm đối chứng (£)", yaxis=dict(autorange="reversed"))
    C.show(fig, 340)
    explain("Ước lượng doubly robust dùng nhóm đối chứng 10 % để đo phần lợi nhuận tăng thêm do chương trình. Với chỉ 72 khách "
            "đối chứng, CI rất rộng – đây là lý do hệ thống đề xuất thí điểm với 50 % đối chứng.")

with t5:
    rows = []
    status = {"PSI đặc trưng": ("fail" if alarm_feats >= 3 else "warn" if alarm_feats else "pass"),
              "PSI điểm số": ("fail" if score_alarm else "pass"),
              "ECE": ("grey" if last_ece is None else "fail" if last_ece >= ECE_A else "warn" if last_ece > ECE_W else "pass"),
              "Action TVD": ("fail" if tvd_max >= TVD_A else "pass"), "Độ trễ": ("pass" if p95 <= 30 else "fail"),
              "Vi phạm": "pass", "Giá trị thực hiện": "warn"}
    lab = {"pass": "BÌNH THƯỜNG", "warn": "CẢNH BÁO", "fail": "BÁO ĐỘNG", "grey": "CHỜ NHÃN"}
    for r in pb.itertuples():
        key = next((k for k in status if r.canh_bao.startswith(k) or k.lower() in r.canh_bao.lower()), None)
        s_ = status.get(key, "info") if key else "info"
        rows.append([badge(s_, lab.get(s_, "THEO DÕI")), f"<b>{r.canh_bao}</b>", f"{r.nguong_canh_bao} / {r.nguong_bao_dong}",
                     r.do_tre_du_lieu, r.chu_so_huu, r.hanh_dong, r.dieu_kien_quay_lui])
    html_table(["Trạng thái", "Cảnh báo", "Cảnh báo / báo động", "Độ trễ dữ liệu", "Chủ sở hữu", "Hành động khắc phục",
                "Điều kiện rollback"], rows, ["", "", "num", "", "", "note", "note"])
    note(f"Trạng thái tính tại ngày phát lại {asof_ts.strftime('%d/%m/%Y')}. Ngưỡng lấy từ configs/config.yaml (mục monitoring).")
    csv_button(pb, "playbook.csv", key="dl_pb")
