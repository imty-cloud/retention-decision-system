"""Số liệu đánh giá mô hình: đọc bảng gốc; reliability diagram tính trực tiếp từ dự báo đã lưu (pred_s2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from services import data_service as D


@st.cache_data(show_spinner=False)
def a0_frame(split: str) -> pd.DataFrame:
    """Dòng hành động a0 của một tập: xác suất thô / Platt và nhãn mua lại thật (Y(0))."""
    p2 = D.parquet("pred_s2", ("customer_id", "t0", "split", "A_logged", "mu0_raw", "mu0_cal"))
    p2 = p2[(p2.split == split) & (p2.A_logged == "a0")]
    y = D.parquet("snapshots", ("customer_id", "t0", "y"))
    return p2.merge(y, on=["customer_id", "t0"], how="left", validate="one_to_one")


@st.cache_data(show_spinner=False)
def reliability(split: str, col: str, n_bins: int = 15) -> tuple[pd.DataFrame, float]:
    """Bin cùng khối lượng (giống cách tính ECE của GĐ2)."""
    d = a0_frame(split)
    p, y = d[col].to_numpy(), d.y.to_numpy()
    order = np.argsort(p, kind="mergesort")
    bins = np.array_split(order, n_bins)
    rows = [{"p_TB": p[b].mean(), "ty_le_thuc": y[b].mean(), "n": len(b)} for b in bins if len(b)]
    r = pd.DataFrame(rows)
    ece = float((r.n * (r.p_TB - r.ty_le_thuc).abs()).sum() / r.n.sum())
    return r, ece


@st.cache_data(show_spinner=False)
def per_period(split: str) -> pd.DataFrame:
    d = a0_frame(split)
    return d.groupby("t0").agg(du_bao_tho=("mu0_raw", "mean"), du_bao_platt=("mu0_cal", "mean"),
                               thuc_te=("y", "mean"), n=("y", "size")).reset_index()
