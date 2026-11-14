"""Ô giải thích ngắn dưới mỗi biểu đồ và nút tải CSV."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def explain(text: str) -> None:
    st.markdown(f'<div class="rds-explain"><b>Đọc biểu đồ:</b> {text}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="rds-note">{text}</div>', unsafe_allow_html=True)


def csv_button(df: pd.DataFrame, filename: str, label: str = "Tải CSV", key: str | None = None) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8-sig"), file_name=filename, mime="text/csv",
                       key=key or f"dl_{filename}", icon=":material/download:")


def plain(text: str) -> None:
    """Đoạn 'Nói dễ hiểu' sau phần trình bày kỹ thuật, dành cho người không chuyên."""
    st.markdown(f'<div class="rds-plain"><div class="k">Nói dễ hiểu</div>{text}</div>', unsafe_allow_html=True)
