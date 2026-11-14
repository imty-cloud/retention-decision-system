"""Thẻ KPI, huy hiệu, băng trạng thái, tiêu đề trang – HTML riêng, không phụ thuộc st.metric."""
from __future__ import annotations

import html

import streamlit as st


def vn(x, d: int = 0, sign: bool = False) -> str:
    """Định dạng số kiểu Việt: phân cách nghìn bằng khoảng trắng, thập phân bằng dấu phẩy."""
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    s = f"{x:+,.{d}f}" if sign else f"{x:,.{d}f}"
    return s.replace(",", " ").replace(".", ",").replace("-", "−")


def gbp(x, d: int = 0, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    s = vn(abs(x), d)
    if x < 0:
        return f"−£{s}"
    return f"+£{s}" if sign else f"£{s}"


def page_header(eyebrow: str, title: str, sub: str) -> None:
    st.markdown(f'<div class="rds-title">{title}</div>'
                f'<p class="rds-sub">{sub}</p>', unsafe_allow_html=True)


def badge(kind: str, text: str) -> str:
    return f'<span class="rds-badge {kind}">{html.escape(text)}</span>'


def status_badge(ok: bool | None) -> str:
    if ok is None:
        return badge("grey", "CHƯA ĐỦ NHÃN")
    return badge("pass", "✓ ĐẠT") if ok else badge("fail", "✕ TRƯỢT")


def kpi_card(label: str, value: str, benchmark: str, tone: str = "neutral", ci: str = "") -> str:
    ci_html = f'<span class="ci">{ci}</span>' if ci else ""
    return (f'<div class="rds-card rds-kpi"><div class="bar bg-{tone}"></div><span class="lab">{label}</span>'
            f'<span class="val">{value}</span><span class="bm tone-{tone}">{benchmark}</span>{ci_html}</div>')


def kpi_row(cards: list[str]) -> None:
    cols = st.columns(len(cards))
    for c, h in zip(cols, cards):
        c.markdown(h, unsafe_allow_html=True)


def banner(kind: str, kicker: str, headline: str, bullets: list[str], rec: str = "") -> None:
    li = "".join(f"<li>{b}</li>" for b in bullets)
    rec_html = f'<div class="rec">{rec}</div>' if rec else ""
    color = {"fail": "var(--fail)", "warn": "var(--warn)", "info": "var(--navy)"}[kind]
    st.markdown(f'<div class="rds-banner {kind}"><div style="flex:1 1 420px"><div class="k" style="color:{color}">{kicker}</div>'
                f'<div class="h">{headline}</div><ul>{li}</ul></div>{rec_html}</div>', unsafe_allow_html=True)


def card(inner_html: str) -> None:
    st.markdown(f'<div class="rds-card">{inner_html}</div>', unsafe_allow_html=True)


def html_table(headers: list[str], rows: list[list[str]], classes: list[str] | None = None) -> None:
    classes = classes or [""] * len(headers)
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f'<td class="{c}">{v}</td>' for v, c in zip(r, classes)) + "</tr>" for r in rows)
    st.markdown(f'<div class="rds-table-wrap"><table class="rds-table"><thead><tr>{th}</tr></thead>'
                f'<tbody>{trs}</tbody></table></div>', unsafe_allow_html=True)
