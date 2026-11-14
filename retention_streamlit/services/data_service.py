"""Đọc dữ liệu và bảng kết quả GỐC của dự án (không tính lại mô hình).

Mọi con số hiển thị trong app đều đến từ `outputs/tables/*.csv` hoặc `data/processed/*.parquet`
do các module `src/s*.py` sinh ra, nên khớp 100 % với kết quả đã đóng băng.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[2]          # thư mục gốc repo
SRC = ROOT / "src"
TABLES = ROOT / "outputs" / "tables"
FIGS = ROOT / "outputs" / "figures"
MANIF = ROOT / "outputs" / "manifests"
PROC = ROOT / "data" / "processed"
ASSETS = Path(__file__).resolve().parents[1] / "assets"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

GITHUB_URL = "https://github.com/imty-cloud/retention-decision-system"
COLAB_URL = ("https://colab.research.google.com/github/imty-cloud/retention-decision-system/"
             "blob/main/notebooks/PTDLNC_full_pipeline.ipynb")
REPORT_URL = GITHUB_URL + "/blob/main/docs/technical_report.pdf"
POLICY_VERSION = "router-v2.1"

# Số liệu chỉ có trong báo cáo kiểm toán độc lập (outputs/manifests/audit_2026-11-13.md, mục A3):
# bootstrap cụm theo khách 5 000 lần cho chênh lệch G4a. Ghi tại một chỗ để trích nguồn rõ ràng.
AUDIT_G4A_CI = (-2273.0, 3980.0)
AUDIT_SRC = "outputs/manifests/audit_2026-11-13.md"


@st.cache_data(show_spinner=False)
def config() -> dict:
    with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data(show_spinner=False)
def table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv")


@st.cache_data(show_spinner=False)
def parquet(name: str, columns: tuple | None = None) -> pd.DataFrame:
    return pd.read_parquet(PROC / f"{name}.parquet", columns=list(columns) if columns else None)


@st.cache_data(show_spinner=False)
def manifest(name: str) -> dict:
    return json.loads((MANIF / name).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def freeze_info() -> dict:
    man = manifest("s8_freeze_manifest.json")
    lock = json.loads((MANIF / "s8_locked_opened.lock").read_text(encoding="utf-8"))
    return {"created_utc": man["created_utc"], "aggregate": man["aggregate_sha256"],
            "data_aggregate": man["data_aggregate_sha256"], "opened_utc": lock["opened_utc"],
            "n_files": len(man["files"]) + len(man["data_files"])}


@st.cache_data(show_spinner=False)
def model_version() -> str:
    d = parquet("decisions_s7_v21", ("model_version",))
    return str(d.model_version.iloc[0])


# ---------------------------------------------------------------- dữ liệu cho mô phỏng (sandbox = valid + calib)
SANDBOX = ("valid", "calib")
FEAT_COLS = ("customer_id", "t0", "split", "is_uk", "cancel_rate_365", "recency_days", "aov_365",
             "n_orders_365", "value_365", "n_orders_all", "tenure_days", "country", "is_one_time")


@st.cache_data(show_spinner="Đang nạp dữ liệu sandbox (valid + calib)…")
def sandbox_frame() -> pd.DataFrame:
    """Khung dữ liệu giống `s5b_dr_diagnostics.load_data` nhưng chỉ giữ valid + calib (không đụng locked/oot)."""
    lg = parquet("logged", FEAT_COLS)
    lg = lg[lg.split.isin(SANDBOX)].reset_index(drop=True)
    acts = ("a1", "a2", "a3")
    keep = ["customer_id", "t0"] + [f"{p}_{a}" for p in ("V_hat", "tau_hat", "tau_lo", "tau_hi") for a in acts] + \
        ["mu0_base", "s_hat_mean"]
    p4 = parquet("pred_s4")[keep]
    df = lg.merge(p4, on=["customer_id", "t0"], how="left", validate="one_to_one")
    p2 = parquet("pred_s2", ("customer_id", "t0", "mu0_raw", "mu0_cal", "s_lo", "s_hi"))
    df = df.merge(p2, on=["customer_id", "t0"], how="left", validate="one_to_one")
    rs = parquet("reasons_s3")
    rs = rs[rs.split.isin(SANDBOX)].drop(columns=["split"])
    df = df.merge(rs, on=["customer_id", "t0"], how="left", validate="one_to_one")
    return df


@st.cache_data(show_spinner="Đang nạp đáp án bán tổng hợp (chỉ để đánh giá)…")
def oracle_frame() -> pd.DataFrame:
    """Đáp án của lớp can thiệp bán tổng hợp – chỉ dùng để ĐÁNH GIÁ, không dùng để ra quyết định."""
    orc = parquet("oracle")
    return orc[orc.split.isin(SANDBOX)].reset_index(drop=True)


def sandbox_periods() -> list[tuple[str, pd.Timestamp]]:
    df = sandbox_frame()
    return [(s, t) for s in SANDBOX for t in sorted(df[df.split == s].t0.unique())]


@st.cache_data(show_spinner=False)
def decisions(kind: str) -> pd.DataFrame:
    return parquet("decisions_s7_v21" if kind == "sandbox" else "decisions_s8_oot")
