"""Trang 3 – Customer 360 & Router."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import charts as C
from components.explanations import csv_button, explain, note
from components.kpi_cards import badge, gbp, html_table, kpi_card, kpi_row, page_header, vn
from services import data_service as D
from services import milp_service as M
from services import router_service as R

page_header("Customer 360 & Router", "Tra cứu khách hàng và luồng Router",
            "Mỗi quyết định đi qua 7 bước của Router v2.1. Trang này cho thấy luồng tổng thể, hồ sơ quyết định của "
            "từng khách và nhật ký kiểm toán.")

E = M.engine()
cfg = D.config()["business"]
SRC = {"sandbox": "Valid + calib (7 kỳ, Router v2.1)", "oot": "Out-of-Time phát lại (5 kỳ, 09–11/2011)"}
tab_flow, tab_cust, tab_log = st.tabs(["Luồng Router (Sankey)", "Hồ sơ quyết định khách hàng", "Nhật ký kiểm toán"])

# =========================================================================== A. Sankey
with tab_flow:
    src = st.radio("Nguồn nhật ký", list(SRC), format_func=SRC.get, horizontal=True, key="rt_src")
    dec = D.decisions(src)
    f = R.flow_counts(dec)
    scale = st.segmented_control("Độ dày luồng", ["Thang log – dễ đọc luồng nhỏ", "Tuyến tính – đúng tỷ lệ"],
                                 default="Thang log – dễ đọc luồng nhỏ", key="rt_scale")
    use_log = scale != "Tuyến tính – đúng tỷ lệ"
    fin_extra = f["backfill"] > 0 or f["dropped"] > 0
    L = dict(inp="①② Đầu vào", blk="Bị chặn cứng", milp="④ MILP phân bổ",
             m0="a0 – không làm gì", prop="MILP đề xuất voucher", hitl="⑤ HITL duyệt tay",
             rej="HITL từ chối → a0", ctl="③ Đối chứng 10 %", ctl0="Đối chứng → a0",
             bfin="Khách bù độ phủ", fin="⑥⑦ Bù phủ & khả thi", drop="Loại ở bước khả thi",
             sent="GỬI THỰC TẾ")
    out_ctl = f["pass_control"]
    links = [("inp", "blk", f["blocked"]), ("inp", "milp", f["milp_in"]),
             ("milp", "m0", f["milp_a0"]), ("milp", "prop", f["proposed"]),
             ("prop", "hitl", f["hitl"]), ("prop", "ctl", f["nohitl"]),
             ("hitl", "rej", f["rejected"]), ("hitl", "ctl", f["approved"]), ("ctl", "ctl0", f["control"])]
    if fin_extra:
        links += [("ctl", "fin", out_ctl), ("bfin", "fin", f["backfill"]), ("fin", "drop", f["dropped"]),
                  ("fin", "sent", f["sent"])]
    else:
        links += [("ctl", "sent", out_ctl)]
    links = [l for l in links if l[2] > 0]
    keys = [k for k in L if any(k in (s_, t_) for s_, t_, _ in links)]
    idx = {k: i for i, k in enumerate(keys)}
    tone = {"blk": C.FAIL, "rej": C.FAIL, "drop": C.FAIL, "m0": C.BASE, "ctl0": C.BASE, "hitl": C.WARN_FILL,
            "prop": C.MILP, "auto": C.MILP, "appr": C.MILP, "sent": C.PASS, "bfin": C.CHAL}
    node_col = [tone.get(k, C.NAVY) for k in keys]
    link_col = ["rgba(180,35,24,0.28)" if t in ("blk", "rej", "drop") else "rgba(138,148,166,0.30)" if t in ("m0", "ctl0")
                else "rgba(21,128,61,0.30)" if t == "sent" else "rgba(37,99,235,0.25)" for _, t, _ in links]
    cnt = {k: vn(sum(v for s_, t_, v in links if t_ == k) or sum(v for s_, t_, v in links if s_ == k)) for k in keys}
    fig = go.Figure(go.Sankey(
        arrangement="snap", textfont=dict(size=12.5, color="#172033"),
        node=dict(label=[f"{L[k]} · {cnt[k]}" for k in keys], color=node_col, pad=26, thickness=18, line=dict(width=0),
                  hovertemplate="%{label}<extra></extra>"),
        link=dict(source=[idx[s_] for s_, _, _ in links], target=[idx[t_] for _, t_, _ in links],
                  value=[np.log1p(v) if use_log else v for _, _, v in links],
                  customdata=[vn(v) for _, _, v in links], color=link_col,
                  hovertemplate="%{source.label} → %{target.label}: <b>%{customdata}</b><extra></extra>")))
    fig.update_layout(title=f"Luồng xử lý của Router – {SRC[src]}"
                      + (" (độ dày theo log, nhãn ghi số thật)" if use_log else ""), font=dict(size=12))
    C.show(fig, 580)
    note("①② kiểm tra dữ liệu và guardrail cứng (áp dưới dạng miền khả thi) · ④ MILP · ⑤ HITL · ③ nhóm đối chứng · "
         + ("⑥ bù độ phủ và ⑦ kiểm tra khả thi cuối." if fin_extra
            else "⑥ không cần bù độ phủ và ⑦ kiểm tra khả thi cuối không loại khách nào ở nguồn này."))
    kpi_row([kpi_card("Khách × kỳ", vn(f["n"]), "đầu vào Router", "neutral", f"{vn(f['blocked'])} bị chặn cứng sau MILP"),
             kpi_card("MILP đề xuất voucher", vn(f["proposed"]), f"{vn(f['proposed'] / f['n'] * 100, 1)} % khách", "neutral"),
             kpi_card("Chuyển HITL", vn(f["hitl"]), f"duyệt {vn(f['approved'])} · từ chối {vn(f['rejected'])}", "warn"),
             kpi_card("Nhóm đối chứng", vn(f["control"]), "ép về a0 để đo giá trị thực hiện", "neutral"),
             kpi_card("Gửi thực tế", vn(f["sent"]), f"{vn(f['backfill'])} khách bù độ phủ", "pass")])
    explain("Guardrail cứng (opt-out, giới hạn tần suất 28 ngày, tỷ lệ hủy đơn &gt; 0,3, sàn biên lợi nhuận) được áp "
            "<b>trước</b> MILP dưới dạng miền khả thi, nên số bị chặn sau MILP bằng 0. Nhóm đối chứng được băm theo khách × kỳ "
            "ngay từ đầu (③) nhưng chỉ có tác dụng trên khách đã được đề xuất và qua HITL – đây là chi phí đo lường có chủ đích. "
            "HITL từ chối các ca có V̂ âm khi tính tại nửa đường tới cận dưới CI của τ̂.")

# =========================================================================== B + C. hồ sơ khách
with tab_cust:
    df = D.sandbox_frame()
    dec = D.decisions("sandbox")
    periods = D.sandbox_periods()
    c1, c2, c3 = st.columns([1.2, 1.2, 2])
    pi = c1.selectbox("Kỳ t0 (sandbox)", range(len(periods)),
                      format_func=lambda i: f"{periods[i][0]} · {pd.Timestamp(periods[i][1]).strftime('%d/%m/%Y')}",
                      key="cu_per")
    s, t0 = periods[pi]
    dk = dec[(dec.split == s) & (dec.t0 == t0)]
    filt = c2.selectbox("Lọc khách", ["Được gửi voucher", "HITL từ chối", "Nhóm đối chứng", "Tất cả"], key="cu_f")
    pool = {"Được gửi voucher": dk[dk.final_action != "a0"], "HITL từ chối": dk[dk.hitl_outcome == "rejected"],
            "Nhóm đối chứng": dk[dk.control_group], "Tất cả": dk}[filt]
    ids = pool.sort_values("V_hat_proposed", ascending=False).customer_id.astype(str).tolist()
    if not ids:
        st.info("Không có khách nào thỏa bộ lọc ở kỳ này.")
        st.stop()
    cid = c3.selectbox("Customer ID (gõ để tìm)", ids, key="cu_id")
    row_d = dk[dk.customer_id.astype(str) == cid].iloc[0]
    row_f = df[(df.split == s) & (df.t0 == t0) & (df.customer_id.astype(str) == cid)].iloc[0]
    row = pd.concat([row_f, row_d.drop(labels=["customer_id", "t0", "split", "s_hat_mean", "s_lo", "s_hi"],
                                       errors="ignore")])
    m = cfg["gross_margin_rate"]
    V = np.array([row_f[f"V_hat_{a}"] for a in ("a1", "a2", "a3")], float)
    sh = D.table("s6_shadow_prices")
    shr = sh[(sh.split == s) & (pd.to_datetime(sh.t0) == pd.Timestamp(t0))].iloc[0]
    duals = {"pi_B": float(shr.gia_bong_ngan_sach_GBP_tren_GBP_menh_gia), "pi_K": float(shr.gia_bong_cuoc_goi_GBP_tren_cuoc)}
    ex = R.explain_customer(row, V, duals, m, E.CAP, E.DISC, E.CONTACT)

    kpi_row([kpi_card("Recency", f"{vn(row_f.recency_days)} ngày", "từ đơn gần nhất đến t0", "neutral"),
             kpi_card("Frequency", f"{vn(row_f.n_orders_365)} đơn", "trong 365 ngày", "neutral",
                      f"Toàn lịch sử: {vn(row_f.n_orders_all)} đơn"),
             kpi_card("Monetary", gbp(row_f.value_365), "doanh số 365 ngày", "neutral", f"Quốc gia: {row_f.country}"),
             kpi_card("Xác suất mua nền μ̂₀", vn(row_f.mu0_base, 3), "đã hiệu chỉnh Platt", "neutral",
                      f"Ŝ = {gbp(row_f.s_hat_mean)} [conformal {gbp(row_f.s_lo)}–{gbp(row_f.s_hi)}]"),
             kpi_card("Hành động cuối", C.ACTION_NAMES[row_d.final_action].split(" · ")[1],
                      f"MILP đề xuất: {row_d.proposed_action}", "pass" if row_d.final_action != "a0" else "neutral",
                      f"HITL: {row_d.hitl_outcome} · đối chứng: {'có' if row_d.control_group else 'không'}")])
    st.write("")
    a, b = st.columns([1.15, 1], gap="large")
    with a:
        P = ex["parts"]
        fig = go.Figure()
        for r_ in P.itertuples():
            fig.add_trace(go.Waterfall(
                x=[[C.ACTION_NAMES[r_.hanh_dong].split(" · ")[0]] * 4,
                   ["Lợi nhuận gia tăng", "− Chi phí voucher", "− Chi phí liên hệ", "= V̂"]],
                measure=["relative", "relative", "relative", "total"],
                y=[r_.loi_nhuan_gia_tang, -r_.chi_phi_voucher, -r_.chi_phi_lien_he, 0],
                text=[gbp(r_.loi_nhuan_gia_tang), gbp(-r_.chi_phi_voucher), gbp(-r_.chi_phi_lien_he), gbp(r_.V)],
                textposition="outside", showlegend=False,
                increasing=dict(marker=dict(color=C.PASS)), decreasing=dict(marker=dict(color=C.FAIL)),
                totals=dict(marker=dict(color=C.MILP if r_.V > 0 else C.BASE)), connector=dict(line=dict(color="#C9D0DB"))))
        fig.update_layout(title="Phân rã giá trị kinh tế từng hành động (£)", yaxis_title="£", waterfallgap=0.3)
        C.show(fig, 380)
        explain(f"V̂ = m·Ŝ·τ̂ − min(d·Ŝ, trần)·(μ̂₀ + τ̂) − chi phí liên hệ, với m = {vn(m, 2)}. Chi phí voucher tính trên "
                "<b>cả</b> khả năng khách đằng nào cũng mua (μ̂₀), nên khách có μ̂₀ cao thường có V̂ âm dù τ̂ dương.")
    with b:
        fig = go.Figure()
        C.ci_dot(fig, [C.ACTION_NAMES[a_] for a_ in P.hanh_dong], P.tau, P.tau_lo, P.tau_hi, "Uplift τ̂ (CI 95 %)", C.MILP)
        fig.add_vline(x=0, line=dict(color=C.INK, width=1))
        fig.update_layout(title="Uplift τ̂ từng hành động (CausalForestDML)", xaxis_title="thay đổi xác suất mua lại",
                          yaxis=dict(autorange="reversed"), showlegend=False)
        C.show(fig, 250)
        rows = [[C.ACTION_NAMES[p.hanh_dong], vn(p.tau, 4), f"[{vn(p.tau_lo, 3)}; {vn(p.tau_hi, 3)}]", gbp(p.V, 2),
                 badge("pass", "V̂ cao nhất") if p.hanh_dong == ex["best"] else ""] for p in P.itertuples()]
        html_table(["Hành động", "τ̂", "CI 95 %", "V̂", ""], rows, ["", "num", "num", "num", ""])

    st.markdown("##### Vì sao hệ thống chọn hành động này?")
    st.markdown("".join(f'<div class="rds-explain">{r_}</div>' for r_ in ex["reasons"]), unsafe_allow_html=True)
    note(f"Ràng buộc của kỳ: ngân sách mệnh giá £{vn(cfg['budget_face_value_per_t0'])} (giá bóng £{vn(duals['pi_B'], 3)}/£1), "
         f"{cfg['call_capacity_per_t0']} cuộc gọi (giá bóng £{vn(duals['pi_K'], 2)}/cuộc), trần voucher 20 % = 25 %. "
         "Lý do trên là <b>suy luận</b> từ nhật ký Router và giá bóng LP relaxation, không phải log nội bộ của solver.")
    st.markdown("##### Giải thích xác suất mua nền (SHAP) – không phải giải thích nhân quả")
    shap_rows = [[str(row_f[f"reason_{k}_code"]), str(row_f[f"reason_{k}_text"]), vn(row_f[f"reason_{k}_shap"], 3)]
                 for k in (1, 2, 3)]
    html_table(["Mã lý do", "Diễn giải", "SHAP (log-odds)"], shap_rows, ["code", "", "num"])
    note("SHAP cho biết vì sao μ̂₀ (khả năng khách tự mua lại) cao hay thấp. Tác động của voucher (τ̂) đến từ mô hình "
         "nhân quả riêng và được phân rã ở biểu đồ phía trên.")

# =========================================================================== D. audit log
with tab_log:
    src2 = st.radio("Nguồn nhật ký", list(SRC), format_func=SRC.get, horizontal=True, key="lg_src")
    d = D.decisions(src2)
    c1, c2, c3 = st.columns(3)
    only_sent = c1.toggle("Chỉ quyết định có hành động hoặc HITL", value=True)
    hitl_f = c2.multiselect("Trạng thái HITL", sorted(d.hitl_outcome.unique()), default=[])
    q = c3.text_input("Tìm Customer ID", "")
    if only_sent:
        d = d[(d.proposed_action != "a0") | d.control_group]
    if hitl_f:
        d = d[d.hitl_outcome.isin(hitl_f)]
    if q:
        d = d[d.customer_id.astype(str).str.contains(q.strip())]
    view = d[["decision_id", "customer_id", "t0", "model_version", "policy_version", "proposed_action", "final_action",
              "hitl_flag", "hitl_trigger", "hitl_outcome", "control_group", "router_reason_code", "reason_1", "reason_2",
              "reason_3", "decided_at", "idempotency_key"]].rename(columns={
        "decision_id": "Mã quyết định", "customer_id": "Customer ID", "model_version": "Model version",
        "policy_version": "Policy version", "proposed_action": "Đề xuất", "final_action": "Cuối cùng",
        "hitl_flag": "HITL", "hitl_trigger": "Lý do HITL", "hitl_outcome": "Kết quả HITL", "control_group": "Đối chứng",
        "router_reason_code": "Mã Router", "reason_1": "Mã lý do 1", "reason_2": "Mã lý do 2", "reason_3": "Mã lý do 3",
        "decided_at": "Thời điểm quyết định", "idempotency_key": "Khóa chống lặp"})
    st.caption(f"{vn(len(view))} bản ghi · khóa chống lặp duy nhất: "
               f"{'có' if view['Khóa chống lặp'].is_unique else 'KHÔNG'}")
    st.dataframe(view, hide_index=True, width="stretch", height=460)
    csv_button(view, f"decision_audit_log_{src2}.csv", key="dl_log")
