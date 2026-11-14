"""Plotly template thống nhất và các biểu đồ dùng chung."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

NAVY, INK, MUTED, LINE = "#1F4E79", "#172033", "#5B6475", "#E3E8EF"
MILP, CHAL, BASE = "#2563EB", "#7C3AED", "#8A94A6"
PASS, WARN, FAIL, WARN_FILL = "#15803D", "#B45309", "#B42318", "#F59E0B"
ACTION_COLORS = {"a0": "#C3CAD6", "a1": "#2563EB", "a2": "#1F4E79", "a3": "#EA7A1A"}
ACTION_NAMES = {"a0": "a0 · Không làm gì", "a1": "a1 · Email voucher 10 %", "a2": "a2 · Email voucher 20 %",
                "a3": "a3 · Gọi điện + voucher 10 %"}

pio.templates["rds"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=13, color=INK),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        colorway=[MILP, CHAL, BASE, NAVY, "#EA7A1A", PASS, FAIL],
        title=dict(font=dict(size=15, color=INK, family="Be Vietnam Pro, Inter, Arial, sans-serif"), x=0, xanchor="left", y=0.97),
        margin=dict(l=10, r=10, t=52, b=10),
        xaxis=dict(gridcolor="#F0F2F6", linecolor=LINE, zerolinecolor=LINE, ticks="outside", tickcolor=LINE),
        yaxis=dict(gridcolor="#F0F2F6", linecolor=LINE, zerolinecolor="#C9D0DB"),
        legend=dict(orientation="h", yanchor="top", y=-0.30, xanchor="left", x=0, font=dict(size=12)),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=LINE, font=dict(color=INK)),
        separators=", ",
    ))
pio.templates.default = "rds"


def show(fig: go.Figure, height: int = 360) -> None:
    fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "displayModeBar": False,
                                                            "toImageButtonOptions": {"scale": 2}})


def threshold_bands(fig: go.Figure, warn: float | None, alarm: float | None, ymax: float, row=None, col=None) -> None:
    """Dải nền xanh / vàng / đỏ theo ngưỡng (giá trị càng cao càng xấu)."""
    kw = {} if row is None else dict(row=row, col=col)
    lo = 0
    if warn is not None:
        fig.add_hrect(y0=lo, y1=warn, fillcolor=PASS, opacity=0.06, line_width=0, **kw)
        fig.add_hline(y=warn, line=dict(color=WARN_FILL, dash="dot", width=1.5), **kw)
        lo = warn
    if alarm is not None:
        fig.add_hrect(y0=lo, y1=alarm, fillcolor=WARN_FILL, opacity=0.07, line_width=0, **kw)
        fig.add_hrect(y0=alarm, y1=max(ymax, alarm * 1.05), fillcolor=FAIL, opacity=0.06, line_width=0, **kw)
        fig.add_hline(y=alarm, line=dict(color=FAIL, dash="dash", width=1.5), **kw)


def ci_dot(fig: go.Figure, y, x, lo, hi, name: str, color: str, symbol="circle", size=11) -> None:
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=name, marker=dict(color=color, size=size, symbol=symbol),
                             error_x=dict(type="data", symmetric=False, array=[h - v for h, v in zip(hi, x)],
                                          arrayminus=[v - l for v, l in zip(x, lo)], color=color, thickness=2, width=6)))
