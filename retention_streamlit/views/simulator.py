"""Trang 2 – Mô phỏng ra quyết định (Decision Simulator). Sandbox: chỉ valid + calib."""
from __future__ import annotations

import datetime as dt
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import charts as C
from components.explanations import csv_button, explain, note
from components.kpi_cards import banner, gbp, html_table, kpi_card, kpi_row, page_header, vn
from services import data_service as D
from services import milp_service as M

page_header("Decision Simulator", "Mô phỏng ra quyết định",
            "Chạy lại bài toán phân bổ ngân sách bằng chính mã MILP của dự án với tham số do bạn chọn, "
            "so sánh với Greedy và luật RFM cũ.")
banner("info", "Sandbox · tính toàn vẹn thực nghiệm", "Chỉ chạy trên dữ liệu valid + calib (7 kỳ, 01–05/2011)",
       ["Locked Test và Out-of-Time <b>không</b> có trong mô phỏng – kết quả kiểm định chính thức không thể bị thay đổi.",
        "Kết quả thử tham số ở đây <b>không</b> phải bằng chứng kiểm định độc lập.",
        "<b>Kế hoạch (V̂)</b> là lợi nhuận mô hình dự kiến; <b>thật kỳ vọng / hiện thực</b> lấy từ đáp án bán tổng hợp, "
        "chỉ dùng để đánh giá."])

E = M.engine()
DEF = dict(per=0, pop="Toàn bộ", ver="v1", B=8000, K=60, m=0.30, a2=25)
for k, v in DEF.items():
    st.session_state.setdefault(f"sim_{k}", v)

df = D.sandbox_frame()
periods = D.sandbox_periods()
PER_LABEL = [f"{s} · {pd.Timestamp(t).strftime('%d/%m/%Y')}" for s, t in periods]
VERS = {"v1": "MILP trên V̂ – μ̂₀ đã hiệu chỉnh (Platt) · bản phát hành v2.1",
        "raw": "Biến thể: V̂ với μ̂₀ chưa hiệu chỉnh (thí nghiệm stage 6)"}


def reset() -> None:
    for k, v in DEF.items():
        st.session_state[f"sim_{k}"] = v
    st.session_state.pop("sim_result", None)


ctrl, res = st.columns([1, 2.6], gap="large")
with ctrl:
    with st.form("sim_form", border=True):
        st.markdown("**Thiết lập kịch bản**")
        st.selectbox("Kỳ quyết định t0", range(len(periods)), format_func=lambda i: PER_LABEL[i], key="sim_per")
        st.selectbox("Quần thể", ["Toàn bộ", "Chỉ UK", "Chỉ non-UK"], key="sim_pop")
        st.selectbox("Phiên bản chính sách", list(VERS), format_func=VERS.get, key="sim_ver")
        st.slider("Ngân sách mệnh giá B (£)", 1000, 32000, step=500, key="sim_B")
        st.slider("Số cuộc gọi tối đa K", 0, 200, step=5, key="sim_K")
        st.slider("Biên lợi nhuận gộp m", 0.20, 0.40, step=0.01, key="sim_m")
        st.slider("Trần voucher 20 % (a2) trên tổng voucher (%)", 0, 50, step=5, key="sim_a2")
        run = st.form_submit_button("Chạy mô phỏng", type="primary", icon=":material/play_arrow:", width="stretch")
    st.button("Khôi phục tham số ban đầu", on_click=reset, icon=":material/restart_alt:", width="stretch")
    note("Mặc định = cấu hình vận hành hiện tại: B = £8 000, K = 60, m = 0,30, trần a2 = 25 %.")


@st.cache_data(show_spinner=False, max_entries=64)
def simulate(per: int, pop: str, ver: str, B: float, K: float, m: float, a2: float) -> dict:
    s, t0 = periods[per]
    sub = df[(df.split == s) & (df.t0 == t0)]
    if pop != "Toàn bộ":
        sub = sub[sub.is_uk == (1 if pop == "Chỉ UK" else 0)]
    sub = sub.reset_index(drop=True)
    ins = M.build(E, sub, B, K)
    V = M.value_matrix(E, sub, m, calibrated=(ver == "v1"))
    V_eval = M.value_matrix(E, sub, m, calibrated=True)
    T = M.truth(sub)
    Ur, Ue = E.evaluate_utilities(T, m)
    out = {"n": len(sub), "ins": {k: ins[k] for k in ("B", "K", "req_uk", "req_non")}, "blocked": int(ins["blocked"].sum())}
    r = M.solve_milp(E, ins, V, a2)
    M._set_a2_share(E, a2)
    t = time.perf_counter()
    wg = E.greedy(ins, V)
    tg = time.perf_counter() - t
    L = M.legacy_matrix(E, sub)
    rl = M.solve_milp(E, ins, L, a2)
    duals = M.lp_duals(E, ins, V, a2)
    meths = {}
    for name, w, extra in [("MILP trên V̂", r["w"], r), ("Greedy (V̂/mệnh giá)", wg, {"t_solve": tg, "status": "heuristic",
                                                                                      "feasible": True}),
                           ("Luật RFM cũ", rl["w"], rl)]:
        if w is None:
            meths[name] = {"feasible": False, "status": extra["status"]}
            continue
        ck = E.check_feasible(ins, w)
        meths[name] = dict(feasible=bool(extra.get("feasible", True)), status=extra.get("status"),
                           t_solve=extra.get("t_solve"), t_build=extra.get("t_build"), gap=extra.get("gap"),
                           plan=float((w[:, 1:] * (V_eval if name == "Luật RFM cũ" else V)).sum()), real=float((w * Ur).sum()), exp=float((w * Ue).sum()),
                           budget=float(ck["ngan_sach_dung"]), vouchers=int(ck["so_voucher"]), calls=int(ck["so_cuoc_goi"]),
                           a2=int(ck["so_a2"]), violations=int(ck["n_vi_pham"]),
                           mix={a: int(w[:, k].sum()) for k, a in enumerate(["a0", "a1", "a2", "a3"])})
    sel = np.where(r["w"][:, 1:].sum(1) > 0)[0] if r["w"] is not None else np.array([], int)
    acts = np.array(["a0", "a1", "a2", "a3"])
    tab = pd.DataFrame({"customer_id": sub.customer_id.to_numpy()[sel],
                        "hanh_dong": acts[r["w"][sel].argmax(1)] if len(sel) else [],
                        "V_hat": (r["w"][sel, 1:] * V_eval[sel]).sum(1) if len(sel) else [],
                        "mu0": sub.mu0_base.to_numpy()[sel], "S_hat": sub.s_hat_mean.to_numpy()[sel],
                        "ly_do_1": sub.reason_1_text.to_numpy()[sel], "quoc_gia": sub.country.to_numpy()[sel]}
                       ).sort_values("V_hat", ascending=False)
    out.update(methods=meths, duals=duals, table=tab, period=PER_LABEL[per])
    return out


@st.cache_data(show_spinner=False, max_entries=32)
def budget_curve(per: int, pop: str, ver: str, K: float, m: float, a2: float) -> pd.DataFrame:
    s, t0 = periods[per]
    sub = df[(df.split == s) & (df.t0 == t0)]
    if pop != "Toàn bộ":
        sub = sub[sub.is_uk == (1 if pop == "Chỉ UK" else 0)]
    sub = sub.reset_index(drop=True)
    V = M.value_matrix(E, sub, m, calibrated=(ver == "v1"))
    V_eval = M.value_matrix(E, sub, m, calibrated=True)
    _, Ue = E.evaluate_utilities(M.truth(sub), m)
    rows = []
    for B in [2000, 4000, 6000, 8000, 12000, 16000, 24000, 32000]:
        r = M.solve_milp(E, M.build(E, sub, B, K), V, a2)
        if r["w"] is None:
            rows.append({"B": B, "plan": np.nan, "exp": np.nan, "kha_thi": False})
        else:
            rows.append({"B": B, "plan": float((r["w"][:, 1:] * V).sum()), "exp": float((r["w"] * Ue).sum()),
                         "kha_thi": True})
    return pd.DataFrame(rows)


ss = st.session_state
params = (ss.sim_per, ss.sim_pop, ss.sim_ver, float(ss.sim_B), float(ss.sim_K), float(ss.sim_m), ss.sim_a2 / 100)
if run or "sim_result" not in ss:
    with st.spinner("Đang giải MILP, Greedy và luật cũ bằng HiGHS…"):
        ss.sim_result = simulate(*params)
        ss.sim_params = params
R = ss.sim_result

with res:
    mi, gr, lg = R["methods"]["MILP trên V̂"], R["methods"]["Greedy (V̂/mệnh giá)"], R["methods"]["Luật RFM cũ"]
    if not mi.get("feasible"):
        banner("fail", "Solver", f"Không khả thi: {mi.get('status')}",
               ["Ngân sách hoặc số cuộc gọi quá thấp để đáp ứng độ phủ tối thiểu 3 % mỗi phân khúc. Hãy tăng B."])
        st.stop()
    st.markdown(
        f'<span class="rds-badge pass">Solver: {mi["status"]}</span> &nbsp; '
        f'<span class="rds-badge info">HiGHS · dựng {vn(mi["t_build"], 2)} s · giải {vn(mi["t_solve"], 2)} s · gap {vn(mi["gap"] * 100, 3)} %</span> &nbsp; '
        f'<span class="rds-badge {"pass" if mi["violations"] == 0 else "fail"}">{mi["violations"]} vi phạm ràng buộc</span> &nbsp; '
        f'<span class="rds-badge grey">{R["period"]} · {vn(R["n"])} khách · {R["blocked"]} bị guardrail chặn</span>',
        unsafe_allow_html=True)
    st.write("")
    kpi_row([
        kpi_card("Lợi nhuận kế hoạch V̂ (MILP)", gbp(mi["plan"]), f"{gbp(mi['plan'] - gr['plan'], sign=True)} so với Greedy",
                 "pass" if mi["plan"] >= gr["plan"] else "warn"),
        kpi_card("Thật kỳ vọng (oracle)", gbp(mi["exp"]), f"{gbp(mi['exp'] - lg['exp'], sign=True)} so với luật cũ",
                 "pass" if mi["exp"] >= lg["exp"] else "fail", f"Hiện thực: {gbp(mi['real'])}"),
        kpi_card("Ngân sách đã dùng", gbp(mi["budget"]), f"{vn(mi['budget'] / R['ins']['B'] * 100, 1)} % của {gbp(R['ins']['B'])}",
                 "neutral", f"{mi['vouchers']} voucher · {mi['calls']} cuộc gọi · {mi['a2']} voucher 20 %"),
    ])
    st.write("")
    kpi_row([
        kpi_card("Giá bóng ngân sách", f"£{vn(R['duals']['pi_B'], 3)}", "mỗi £1 mệnh giá thêm", "neutral", "Từ LP relaxation"),
        kpi_card("Giá bóng cuộc gọi", f"£{vn(R['duals']['pi_K'], 2)}", "mỗi cuộc gọi thêm (chi phí £3)",
                 "warn" if R["duals"]["pi_K"] > 3 else "neutral", "Từ LP relaxation"),
        kpi_card("MILP so với luật cũ (kỳ vọng)", gbp(mi["exp"] - lg["exp"], sign=True), "lợi nhuận thật kỳ vọng / kỳ",
                 "pass" if mi["exp"] > lg["exp"] else "fail", f"Hiện thực: {gbp(mi['real'] - lg['real'], sign=True)}"),
    ])
    note("Giá bóng lấy từ <b>LP relaxation</b> (bỏ ràng buộc nguyên), không phải giá trị đối ngẫu trực tiếp của MILP – "
         "MILP không có đối ngẫu chuẩn. Đây là xấp xỉ giá trị biên của thêm một đơn vị tài nguyên.")

    t1, t2, t3 = st.tabs(["So sánh phương pháp", "Đường cong theo ngân sách B", "Khách hàng được chọn"])
    with t1:
        a, b = st.columns(2)
        with a:
            names = list(R["methods"])
            fig = go.Figure()
            metrics = [("plan", "Kế hoạch V̂"), ("exp", "Thật kỳ vọng"), ("real", "Thật hiện thực")]
            for n_, col in zip(names, [C.MILP, C.CHAL, C.BASE]):
                vals = [R["methods"][n_].get(k) for k, _ in metrics]
                fig.add_bar(x=[l for _, l in metrics], y=vals, name=n_, marker_color=col,
                            text=[gbp(v) for v in vals], textposition="outside", textfont=dict(size=10))
            fig.update_layout(barmode="group", title="Lợi nhuận gia tăng một kỳ (£)", yaxis_title="£")
            C.show(fig, 340)
            explain("Kế hoạch là điều mô hình tin; kỳ vọng và hiện thực là điều thật sự xảy ra theo đáp án bán tổng hợp. "
                    "MILP luôn thắng Greedy về kế hoạch (vì Greedy là xấp xỉ), nhưng về hiện thực khoảng cách nhỏ hơn nhiễu.")
        with b:
            fig = go.Figure()
            for a_ in ["a1", "a2", "a3"]:
                fig.add_bar(y=names, x=[R["methods"][n]["mix"][a_] for n in names], name=C.ACTION_NAMES[a_],
                            orientation="h", marker_color=C.ACTION_COLORS[a_])
            fig.update_layout(barmode="stack", title="Cơ cấu phân bổ hành động (số khách)", yaxis=dict(autorange="reversed"))
            C.show(fig, 340)
            explain(f"Phần còn lại ({vn(mi['mix']['a0'])} khách với MILP) nhận a0 – không làm gì. "
                    "Luật cũ dồn nhiều voucher 20 % cho khách giá trị cao vốn sẽ mua.")
        MT = R["methods"]
        metrics = [("Kế hoạch V̂", lambda v: gbp(v["plan"])), ("Thật kỳ vọng (oracle)", lambda v: gbp(v["exp"])),
                   ("Thật hiện thực", lambda v: gbp(v["real"])), ("Ngân sách đã dùng", lambda v: gbp(v["budget"])),
                   ("Số voucher", lambda v: vn(v["vouchers"])), ("Số cuộc gọi", lambda v: vn(v["calls"])),
                   ("Voucher 20 % (a2)", lambda v: vn(v["a2"])), ("Vi phạm ràng buộc", lambda v: vn(v["violations"])),
                   ("Thời gian giải", lambda v: f"{vn(v['t_solve'], 3)} s" if v.get("t_solve") is not None else "—")]
        rows = [[lab] + [fn(v) for v in MT.values()] for lab, fn in metrics]
        html_table(["Chỉ số"] + list(MT), rows, [""] + ["num"] * len(MT))
    with t2:
        with st.spinner("Đang giải 8 mức ngân sách…"):
            p_ = ss.sim_params
            cv = budget_curve(p_[0], p_[1], p_[2], p_[4], p_[5], p_[6])
        fig = go.Figure()
        fig.add_scatter(x=cv.B, y=cv.plan, name="Kế hoạch V̂", mode="lines+markers", line=dict(color=C.MILP, dash="dot"))
        fig.add_scatter(x=cv.B, y=cv.exp, name="Thật kỳ vọng", mode="lines+markers", line=dict(color=C.MILP, width=3))
        fig.add_vline(x=R["ins"]["B"], line=dict(color=C.NAVY, dash="dash"), annotation_text="B hiện tại")
        fig.update_layout(title="Lợi nhuận MILP theo ngân sách B (giữ nguyên K, m, trần a2)", xaxis_title="B (£)",
                          yaxis_title="£ / kỳ")
        C.show(fig, 360)
        explain("Đường kế hoạch tăng chậm dần vì mỗi £ ngân sách thêm được dùng cho khách kém hiệu quả hơn (giá bóng giảm dần). "
                "Đường thật kỳ vọng thường thấp hơn kế hoạch ở ngân sách lớn: mô hình chọn thêm cả những khách có τ̂ cao do nhiễu.")
        csv_button(cv, "duong_cong_ngan_sach.csv", key="dl_curve")
    with t3:
        tab = R["table"].rename(columns={"customer_id": "Mã KH", "hanh_dong": "Hành động", "V_hat": "V̂ (£)", "mu0": "μ̂₀",
                                         "S_hat": "Ŝ (£)", "ly_do_1": "Lý do SHAP #1 (xác suất mua nền)",
                                         "quoc_gia": "Quốc gia"})
        st.dataframe(tab, hide_index=True, width="stretch", height=380,
                     column_config={"V̂ (£)": st.column_config.NumberColumn(format="%.2f"),
                                    "μ̂₀": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
                                    "Ŝ (£)": st.column_config.NumberColumn(format="%.0f")})
        csv_button(R["table"], "khach_duoc_chon.csv", key="dl_sel")
        note("Mã lý do SHAP giải thích <b>xác suất mua nền μ̂₀</b>, không phải lý do nhân quả của voucher. "
             "Xem phân rã kinh tế từng hành động ở trang Customer 360.")

    B_, K_, m_, a2_ = R["ins"]["B"], R["ins"]["K"], ss.sim_params[5], ss.sim_params[6]
    report = f"""# Scenario Report – Retention Decision System
Thời điểm xuất: {dt.datetime.now().strftime('%d/%m/%Y %H:%M')}
Phiên bản chính sách: {D.POLICY_VERSION} · {VERS[ss.sim_params[2]]}
Kỳ: {R['period']} · Quần thể: {ss.sim_params[1]} · {R['n']} khách

## Tham số
- Ngân sách mệnh giá B = £{B_:,.0f}; số cuộc gọi K = {K_:.0f}; biên m = {m_:.2f}; trần voucher 20 % = {a2_ * 100:.0f} %

## Kết quả
| Phương pháp | Kế hoạch V̂ | Thật kỳ vọng | Thật hiện thực | Voucher | Cuộc gọi | Vi phạm |
|---|---|---|---|---|---|---|
""" + "\n".join(f"| {n} | {v['plan']:.0f} | {v['exp']:.0f} | {v['real']:.0f} | {v['vouchers']} | {v['calls']} | {v['violations']} |"
                for n, v in R["methods"].items()) + f"""

Giá bóng (LP relaxation): ngân sách £{R['duals']['pi_B']:.3f}/£1 mệnh giá; cuộc gọi £{R['duals']['pi_K']:.2f}/cuộc.

> Sandbox valid + calib. Kết quả mô phỏng không phải bằng chứng kiểm định độc lập; lợi nhuận "thật" lấy từ đáp án bán tổng hợp.
"""
    st.download_button("Xuất Scenario Report", report.encode("utf-8"), file_name="scenario_report.md",
                       mime="text/markdown", icon=":material/description:")
