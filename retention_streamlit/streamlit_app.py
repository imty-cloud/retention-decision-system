"""Retention Decision System – CRM Intelligence & Prescriptive Analytics.

Chạy:  streamlit run retention_streamlit/streamlit_app.py   (từ thư mục gốc repo)
"""
from __future__ import annotations

import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Retention Decision System", page_icon=":material/insights:",
                   layout="wide", initial_sidebar_state="expanded")

from components import charts  # noqa: E402,F401  (đăng ký Plotly template "rds")
from services import data_service as D  # noqa: E402

st.markdown(f"<style>{(APP / 'assets' / 'styles.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

pages = {
    "Điều hành": [
        st.Page("views/overview.py", title="Tổng quan điều hành", icon=":material/dashboard:", default=True),
        st.Page("views/simulator.py", title="Mô phỏng ra quyết định", icon=":material/tune:"),
        st.Page("views/customer_router.py", title="Customer 360 & Router", icon=":material/route:"),
    ],
    "Mô hình & vận hành": [
        st.Page("views/model_evaluation.py", title="Dự báo & nhân quả", icon=":material/insights:"),
        st.Page("views/mlops_monitoring.py", title="Giám sát MLOps", icon=":material/monitor_heart:"),
        st.Page("views/evidence.py", title="Minh chứng & tái lập", icon=":material/verified:"),
    ],
}


nav = st.navigation(pages, position="sidebar")

fz = D.freeze_info()
with st.sidebar:
    st.markdown(
        f'<div class="rds-side-meta">Dữ liệu: Online Retail II (UCI) + lớp can thiệp bán tổng hợp<br>'
        f'Chính sách: <b>{D.POLICY_VERSION}</b><br>Đóng băng: {fz["created_utc"][:16].replace("T", " ")} UTC<br>'
        f'Locked Test mở 1 lần: {fz["opened_utc"][:16].replace("T", " ")} UTC<br>'
        f'<a href="{D.GITHUB_URL}" target="_blank">GitHub</a> · <a href="{D.COLAB_URL}" target="_blank">Colab</a></div>',
        unsafe_allow_html=True)

nav.run()

st.markdown(
    f'<div class="rds-footer"><span>Dữ liệu đóng băng: <code>{fz["created_utc"][:10]}</code> '
    f'(hash dữ liệu <code>{fz["data_aggregate"][:12]}…</code>)</span>'
    f'<span>Model version: <code>{D.model_version()}</code></span>'
    f'<span>Policy version: <code>{D.POLICY_VERSION}</code></span>'
    f'<span>Hash đóng băng mã/mô hình: <code>{fz["aggregate"][:12]}…</code></span>'
    f'<span>Retention Decision System · release {D.POLICY_VERSION} · © 2026</span></div>',
    unsafe_allow_html=True)
