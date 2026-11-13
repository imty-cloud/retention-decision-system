"""
GĐ1 – Thiết kế bài toán, kỹ nghệ dữ liệu theo thời gian và phòng chống rò rỉ (Chương 1).

Đầu vào : data/raw/online_retail_II.xlsx (UCI Online Retail II, 2 sheet 2009-2010 và 2010-2011)
Đầu ra  : data/processed/{transactions_clean, orders, cancels, snapshots}.parquet
          outputs/tables/s1_*.csv, outputs/figures/fig_s1_*.png, outputs/manifests/s1_manifest.json

Nguyên tắc point-in-time:
  - Mỗi bản ghi = (customer_id, t0). Đặc trưng chỉ dùng sự kiện có thời điểm < cutoff = t0 - feature_latency.
  - Nhãn y = 1 nếu khách có >= 1 đơn hợp lệ trong [t0, t0 + H). Nhãn chỉ "trưởng thành" tại t0 + H.
  - Đơn hủy (hóa đơn 'C...') là sự kiện riêng, tính theo thời điểm hủy, không sửa lùi đơn gốc.

Chạy:  python src/s1_data.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_config, p, write_manifest  # noqa: E402

DAY = pd.Timedelta(days=1)
RAW_COLS = {"Invoice": "invoice", "StockCode": "stock_code", "Description": "description",
            "Quantity": "quantity", "InvoiceDate": "invoice_ts", "Price": "unit_price",
            "Customer ID": "customer_id", "Country": "country"}
SPLIT_ORDER = ["train", "valid", "calib", "locked", "oot"]
WINDOWS = (30, 90, 180, 365)


# ----------------------------------------------------------------------------- 1. Nạp dữ liệu
def load_raw(cfg: dict) -> pd.DataFrame:
    cache = p("data/processed/raw_combined.parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    sheets = pd.read_excel(ROOT / cfg["paths"]["raw_xlsx"], sheet_name=None,
                           dtype={"Invoice": str, "StockCode": str, "Description": str, "Country": str})
    raw = pd.concat([df.assign(source_sheet=name) for name, df in sheets.items()], ignore_index=True)
    raw.to_parquet(cache, index=False)
    return raw


# ----------------------------------------------------------------------------- 2. Làm sạch
def clean(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    log: list[dict] = []
    df = raw.rename(columns=RAW_COLS).copy()
    df["invoice"] = df["invoice"].astype(str).str.strip()
    df["stock_code"] = df["stock_code"].astype(str).str.strip()

    def record(step: str, rule: str, before: int, frame: pd.DataFrame) -> pd.DataFrame:
        log.append({"step": step, "rule": rule, "rows_before": before, "rows_after": len(frame),
                    "rows_removed": before - len(frame)})
        return frame

    n = len(df)
    record("0_raw", "Ghép 2 sheet", n, df)

    # (1) Hai sheet chồng lấn tháng 12/2010: hóa đơn đã có ở sheet đầu thì bỏ ở sheet sau.
    first = df["source_sheet"].iloc[0]
    inv_first = set(df.loc[df["source_sheet"] == first, "invoice"])
    n = len(df)
    df = record("1_sheet_overlap", "Bỏ hóa đơn trùng giữa 2 sheet",
                n, df[~((df["source_sheet"] != first) & df["invoice"].isin(inv_first))])

    n = len(df)
    df = record("2_exact_duplicates", "Bỏ dòng trùng tuyệt đối", n,
                df.drop_duplicates(subset=list(RAW_COLS.values())))

    n = len(df)
    df = record("3_missing_customer", "Bỏ dòng thiếu Customer ID (không gắn được đơn vị phân tích)",
                n, df[df["customer_id"].notna()])
    df = df.assign(customer_id=df["customer_id"].astype("int64").astype(str))

    n = len(df)
    df = record("4_invoice_type", "Chỉ giữ hóa đơn bán '\\d{6}' và hủy 'C\\d{6}' (bỏ 'A' điều chỉnh nợ)",
                n, df[df["invoice"].str.fullmatch(r"C?\d{6}")])
    df = df.assign(is_cancel=df["invoice"].str.startswith("C"))

    n = len(df)
    df = record("5_non_product", "Bỏ mã không phải hàng hóa (POST, M, D, BANK CHARGES...)",
                n, df[df["stock_code"].str.fullmatch(cfg["cleaning"]["product_code_regex"])])

    n = len(df)
    df = record("6_price", "Đơn giá > 0", n, df[df["unit_price"] > cfg["cleaning"]["min_price"]])

    n = len(df)
    sign_ok = (~df["is_cancel"] & (df["quantity"] > 0)) | (df["is_cancel"] & (df["quantity"] < 0))
    df = record("7_sign", "Dấu số lượng khớp loại hóa đơn", n, df[sign_ok])

    df = df.assign(invoice_ts=pd.to_datetime(df["invoice_ts"]),
                   line_value=df["quantity"] * df["unit_price"])
    df = df.drop(columns=["source_sheet"]).sort_values(["invoice_ts", "invoice"]).reset_index(drop=True)
    return df, pd.DataFrame(log)


def build_orders(tx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sales = tx[~tx["is_cancel"]]
    orders = (sales.groupby("invoice")
              .agg(customer_id=("customer_id", "first"), order_ts=("invoice_ts", "min"),
                   order_value=("line_value", "sum"), n_lines=("stock_code", "size"),
                   n_units=("quantity", "sum"), n_products=("stock_code", "nunique"),
                   country=("country", "first"))
              .reset_index().sort_values("order_ts").reset_index(drop=True))
    canc = tx[tx["is_cancel"]]
    cancels = (canc.groupby("invoice")
               .agg(customer_id=("customer_id", "first"), cancel_ts=("invoice_ts", "min"),
                    cancel_value=("line_value", lambda s: -s.sum()))
               .reset_index().sort_values("cancel_ts").reset_index(drop=True))
    return orders, cancels


# ----------------------------------------------------------------------------- 3. Snapshot tại t0
def t0_grid(cfg: dict) -> pd.DatetimeIndex:
    d = cfg["decision"]
    last_t0 = pd.Timestamp(d["data_end"]) - d["horizon_days"] * DAY   # cửa sổ nhãn phải nằm trọn trong dữ liệu
    return pd.date_range(d["t0_first"], last_t0, freq=f"{d['t0_freq_days']}D")


def features_at(t0: pd.Timestamp, orders: pd.DataFrame, cancels: pd.DataFrame,
                sales_lines: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Tính đặc trưng cho mọi khách đủ điều kiện tại t0, CHỈ từ sự kiện trước cutoff."""
    d = cfg["decision"]
    cutoff = t0 - d["feature_latency_days"] * DAY
    ho = orders[orders["order_ts"] < cutoff]
    last = ho.groupby("customer_id")["order_ts"].max()
    eligible = last.index[last >= t0 - d["eligibility_lookback_days"] * DAY]
    ho = ho[ho["customer_id"].isin(eligible)].sort_values(["customer_id", "order_ts"])
    hc = cancels[(cancels["cancel_ts"] < cutoff) & cancels["customer_id"].isin(eligible)]
    hl = sales_lines[(sales_lines["invoice_ts"] < cutoff)
                     & (sales_lines["invoice_ts"] >= t0 - 365 * DAY)
                     & sales_lines["customer_id"].isin(eligible)]

    g = ho.groupby("customer_id")
    f = pd.DataFrame(index=pd.Index(eligible, name="customer_id"))
    days = lambda td: td.dt.total_seconds() / 86400.0  # noqa: E731
    f["recency_days"] = days(t0 - g["order_ts"].max())
    f["tenure_days"] = days(t0 - g["order_ts"].min())
    f["n_orders_all"] = g.size()
    f["value_all"] = g["order_value"].sum()
    for w in WINDOWS:
        gw = ho[ho["order_ts"] >= t0 - w * DAY].groupby("customer_id")
        f[f"n_orders_{w}"] = gw.size().reindex(f.index, fill_value=0)
        f[f"value_{w}"] = gw["order_value"].sum().reindex(f.index, fill_value=0.0)
    g365 = ho[ho["order_ts"] >= t0 - 365 * DAY].groupby("customer_id")
    f["aov_365"] = f["value_365"] / f["n_orders_365"]
    f["aov_all"] = f["value_all"] / f["n_orders_all"]
    f["units_per_order_365"] = g365["n_units"].sum().reindex(f.index) / f["n_orders_365"]
    f["lines_per_order_365"] = g365["n_lines"].sum().reindex(f.index) / f["n_orders_365"]
    f["distinct_products_365"] = (hl.groupby("customer_id")["stock_code"].nunique()
                                  .reindex(f.index, fill_value=0))

    ipt = days(ho.groupby("customer_id")["order_ts"].diff()).groupby(ho["customer_id"])
    f["ipt_mean_days"] = ipt.mean()
    f["ipt_std_days"] = ipt.std()
    f["is_one_time"] = (f["n_orders_all"] == 1).astype("int8")
    # Nhiều hóa đơn cùng thời điểm => ipt = 0; chặn dưới 1 ngày để tránh chia cho 0
    f["overdue_ratio"] = f["recency_days"] / f["ipt_mean_days"].clip(lower=1.0)

    hc365 = hc[hc["cancel_ts"] >= t0 - 365 * DAY].groupby("customer_id")
    f["n_cancels_365"] = hc365.size().reindex(f.index, fill_value=0)
    f["cancel_value_365"] = hc365["cancel_value"].sum().reindex(f.index, fill_value=0.0)
    f["cancel_rate_365"] = f["cancel_value_365"] / (f["value_365"] + f["cancel_value_365"])
    last_cancel = hc.groupby("customer_id")["cancel_ts"].max()
    f["days_since_last_cancel"] = days(t0 - last_cancel).reindex(f.index)

    f["value_trend_90_365"] = f["value_90"] / (f["value_365"] * 90 / 365)
    f["country"] = g["country"].last()
    f["is_uk"] = (f["country"] == "United Kingdom").astype("int8")

    # Đặc trưng lịch – biết trước tại t0
    f["t0_month"] = t0.month
    xmas = pd.Timestamp(year=t0.year, month=12, day=25)
    f["days_to_xmas"] = (xmas - t0).days if t0 <= xmas else (xmas.replace(year=t0.year + 1) - t0).days

    # Siêu dữ liệu phục vụ kiểm toán rò rỉ
    f["max_feature_event_ts"] = pd.concat([g["order_ts"].max(), last_cancel.reindex(f.index)],
                                          axis=1).max(axis=1)
    f["cutoff_ts"] = cutoff
    f["t0"] = t0
    return f.reset_index()


def labels_at(t0: pd.Timestamp, orders: pd.DataFrame, H: int) -> pd.DataFrame:
    fut = orders[(orders["order_ts"] >= t0) & (orders["order_ts"] < t0 + H * DAY)]
    return (fut.groupby("customer_id")
            .agg(y_n_orders_h=("invoice", "size"), y_value_h=("order_value", "sum"))
            .reset_index())


def build_snapshots(tx: pd.DataFrame, orders: pd.DataFrame, cancels: pd.DataFrame,
                    cfg: dict) -> pd.DataFrame:
    H = cfg["decision"]["horizon_days"]
    sales_lines = tx.loc[~tx["is_cancel"], ["customer_id", "invoice_ts", "stock_code"]]
    parts = []
    for t0 in t0_grid(cfg):
        f = features_at(t0, orders, cancels, sales_lines, cfg)
        lab = labels_at(t0, orders, H)
        f = f.merge(lab, on="customer_id", how="left")
        f["y_n_orders_h"] = f["y_n_orders_h"].fillna(0).astype("int64")
        f["y_value_h"] = f["y_value_h"].fillna(0.0)
        f["y"] = (f["y_n_orders_h"] > 0).astype("int8")
        f["label_window_end"] = t0 + H * DAY
        f["label_available_ts"] = t0 + H * DAY      # độ trễ nhãn = H ngày
        parts.append(f)
    snap = pd.concat(parts, ignore_index=True)
    return assign_splits(snap, cfg)


# ----------------------------------------------------------------------------- 4. Chia tập
def is_entity_holdout(customer_id: str, salt: str, pct: int) -> bool:
    h = hashlib.sha256(f"{salt}|{customer_id}".encode()).hexdigest()
    return int(h, 16) % 100 < pct


def assign_splits(snap: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    s = cfg["splits"]
    split = pd.Series("embargo", index=snap.index, dtype="object")
    for name in SPLIT_ORDER:
        a, b = pd.Timestamp(s[name][0]), pd.Timestamp(s[name][1])
        split[(snap["t0"] >= a) & (snap["t0"] <= b)] = name
    ids = snap["customer_id"].unique()
    hold = {c: is_entity_holdout(c, s["entity_holdout_salt"], s["entity_holdout_pct"]) for c in ids}
    snap["is_entity_holdout"] = snap["customer_id"].map(hold).astype("int8")
    dev = split.isin(["train", "valid", "calib"]) & (snap["is_entity_holdout"] == 1)
    split[dev] = "excluded_entity_holdout"
    snap["split"] = split
    return snap


def check_embargo(snap: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    H = cfg["decision"]["horizon_days"]
    rng = snap[snap["split"].isin(SPLIT_ORDER)].groupby("split")["t0"].agg(["min", "max"])
    rows = []
    for a, b in zip(SPLIT_ORDER[:-1], SPLIT_ORDER[1:]):
        gap = (rng.loc[b, "min"] - rng.loc[a, "max"]).days
        rows.append({"from": a, "to": b, "last_t0_from": rng.loc[a, "max"].date(),
                     "first_t0_to": rng.loc[b, "min"].date(), "gap_days": gap,
                     "label_end_from": (rng.loc[a, "max"] + H * DAY).date(),
                     "pass": gap >= H})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 5. Hợp đồng dữ liệu
FEATURES_NUM = ["recency_days", "tenure_days", "n_orders_all", "value_all",
                *[f"n_orders_{w}" for w in WINDOWS], *[f"value_{w}" for w in WINDOWS],
                "aov_365", "aov_all", "units_per_order_365", "lines_per_order_365",
                "distinct_products_365", "ipt_mean_days", "ipt_std_days", "is_one_time",
                "overdue_ratio", "n_cancels_365", "cancel_value_365", "cancel_rate_365",
                "days_since_last_cancel", "value_trend_90_365", "is_uk", "t0_month", "days_to_xmas"]
FEATURES_CAT = ["country"]
EXCLUDE_FROM_MODEL = ["customer_id", "t0", "cutoff_ts", "max_feature_event_ts", "label_window_end",
                      "label_available_ts", "y", "y_n_orders_h", "y_value_h", "split",
                      "is_entity_holdout"]

DATA_DICTIONARY = {
    "customer_id": ("khóa", "Mã khách hàng (khóa kỹ thuật, không dùng làm đặc trưng)"),
    "t0": ("khóa", "Thời điểm quyết định (thứ Hai, 2 tuần/lần)"),
    "cutoff_ts": ("kiểm toán", "t0 - độ trễ dữ liệu; mọi sự kiện dùng cho đặc trưng phải < cutoff"),
    "max_feature_event_ts": ("kiểm toán", "Sự kiện muộn nhất đã dùng để tính đặc trưng"),
    "recency_days": ("đặc trưng", "Số ngày từ đơn gần nhất đến t0"),
    "tenure_days": ("đặc trưng", "Số ngày từ đơn đầu tiên đến t0"),
    "n_orders_all": ("đặc trưng", "Số đơn hợp lệ trong toàn bộ lịch sử trước cutoff"),
    "value_all": ("đặc trưng", "Tổng giá trị đơn (GBP) toàn lịch sử"),
    **{f"n_orders_{w}": ("đặc trưng", f"Số đơn trong {w} ngày trước t0") for w in WINDOWS},
    **{f"value_{w}": ("đặc trưng", f"Tổng giá trị đơn trong {w} ngày trước t0") for w in WINDOWS},
    "aov_365": ("đặc trưng", "Giá trị đơn trung bình 365 ngày"),
    "aov_all": ("đặc trưng", "Giá trị đơn trung bình toàn lịch sử"),
    "units_per_order_365": ("đặc trưng", "Số đơn vị hàng / đơn (đại diện khách sỉ)"),
    "lines_per_order_365": ("đặc trưng", "Số dòng hàng / đơn"),
    "distinct_products_365": ("đặc trưng", "Số mã hàng khác nhau đã mua trong 365 ngày"),
    "ipt_mean_days": ("đặc trưng", "Khoảng cách trung bình giữa hai đơn (NaN nếu mới mua 1 lần)"),
    "ipt_std_days": ("đặc trưng", "Độ lệch chuẩn khoảng cách giữa hai đơn"),
    "is_one_time": ("đặc trưng", "1 nếu mới mua đúng 1 đơn"),
    "overdue_ratio": ("đặc trưng", "recency / ipt_mean: >1 nghĩa là đã quá chu kỳ mua thường lệ"),
    "n_cancels_365": ("đặc trưng", "Số hóa đơn hủy/trả trong 365 ngày (theo ngày hủy)"),
    "cancel_value_365": ("đặc trưng", "Giá trị hủy/trả trong 365 ngày"),
    "cancel_rate_365": ("đặc trưng", "Tỷ lệ giá trị hủy / (bán + hủy) trong 365 ngày"),
    "days_since_last_cancel": ("đặc trưng", "Số ngày từ lần hủy gần nhất (NaN nếu chưa từng hủy)"),
    "value_trend_90_365": ("đặc trưng", "Giá trị 90 ngày so với mức trung bình 90 ngày của năm"),
    "country": ("đặc trưng", "Quốc gia của đơn gần nhất"),
    "is_uk": ("đặc trưng", "1 nếu khách ở Anh"),
    "t0_month": ("đặc trưng", "Tháng của t0 (biết trước)"),
    "days_to_xmas": ("đặc trưng", "Số ngày từ t0 đến Giáng sinh kế tiếp (biết trước)"),
    "y": ("nhãn", "1 nếu có >= 1 đơn hợp lệ trong [t0, t0 + H)"),
    "y_n_orders_h": ("nhãn", "Số đơn trong cửa sổ nhãn"),
    "y_value_h": ("nhãn", "Tổng giá trị đơn trong cửa sổ nhãn (dùng cho AOV/giá trị ở GĐ4–6)"),
    "label_window_end": ("kiểm toán", "t0 + H"),
    "label_available_ts": ("kiểm toán", "Thời điểm nhãn trưởng thành (độ trễ nhãn)"),
    "split": ("kiểm toán", "train/valid/calib/locked/oot/embargo/excluded_entity_holdout"),
    "is_entity_holdout": ("kiểm toán", "1 nếu mã khách thuộc nhóm giữ riêng"),
}


def snapshot_schema(cfg: dict):
    try:
        import pandera.pandas as pa
    except ImportError:  # pandera < 0.24
        import pandera as pa
    lat = cfg["decision"]["feature_latency_days"]
    nonneg = pa.Check.ge(0)
    cols = {
        "customer_id": pa.Column(str, nullable=False),
        "t0": pa.Column("datetime64[ns]", nullable=False, coerce=True),
        "recency_days": pa.Column(float, pa.Check.ge(lat)),
        "tenure_days": pa.Column(float, pa.Check.ge(lat)),
        "n_orders_all": pa.Column(int, pa.Check.ge(1)),
        "cancel_rate_365": pa.Column(float, pa.Check.in_range(0, 1)),
        "y": pa.Column("int8", pa.Check.isin([0, 1])),
        "y_value_h": pa.Column(float, nonneg),
        "is_entity_holdout": pa.Column("int8", pa.Check.isin([0, 1])),
        "split": pa.Column(str, pa.Check.isin(SPLIT_ORDER + ["embargo", "excluded_entity_holdout"])),
        **{f"n_orders_{w}": pa.Column(int, nonneg) for w in WINDOWS},
        **{f"value_{w}": pa.Column(float, nonneg) for w in WINDOWS},
    }
    return pa.DataFrameSchema(
        cols, strict=False, unique=["customer_id", "t0"],
        checks=[
            pa.Check(lambda d: d["tenure_days"] >= d["recency_days"], name="tenure>=recency"),
            pa.Check(lambda d: (d["n_orders_30"] <= d["n_orders_90"]) & (d["n_orders_90"] <= d["n_orders_180"])
                     & (d["n_orders_180"] <= d["n_orders_365"]) & (d["n_orders_365"] <= d["n_orders_all"]),
                     name="window_monotone"),
            pa.Check(lambda d: d["max_feature_event_ts"] < d["cutoff_ts"], name="no_event_after_cutoff"),
            pa.Check(lambda d: d["label_available_ts"]
                     == d["t0"] + pd.Timedelta(days=cfg["decision"]["horizon_days"]), name="label_maturity"),
            pa.Check(lambda d: (d["y"] == 1) == (d["y_value_h"] > 0), name="y_consistent_with_value"),
            pa.Check(lambda d: np.isfinite(d[FEATURES_NUM].astype(float).fillna(0)).all(axis=None),
                     name="no_infinite_features"),
        ])


# ----------------------------------------------------------------------------- 6. Kiểm thử rò rỉ
def leakage_tests(tx: pd.DataFrame, orders: pd.DataFrame, cancels: pd.DataFrame,
                  snap: pd.DataFrame, cfg: dict, n_t0: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["project"]["seed"])
    H = cfg["decision"]["horizon_days"]
    lat = cfg["decision"]["feature_latency_days"]
    results = []
    t0s = rng.choice(snap["t0"].unique(), size=n_t0, replace=False)
    feat_cols = FEATURES_NUM + FEATURES_CAT

    for t0 in pd.to_datetime(t0s):
        stored = snap[snap["t0"] == t0].set_index("customer_id")[feat_cols].sort_index()

        # T1: Dựng lại từ kho dữ liệu đã cắt tại cutoff (mô phỏng "chỉ có những gì tồn tại tại t0")
        cutoff = t0 - lat * DAY
        tx_cut = tx[tx["invoice_ts"] < cutoff]
        o_cut, c_cut = build_orders(tx_cut)
        rebuilt = features_at(t0, o_cut, c_cut, tx_cut.loc[~tx_cut["is_cancel"],
                              ["customer_id", "invoice_ts", "stock_code"]], cfg)
        rebuilt = rebuilt.set_index("customer_id")[feat_cols].sort_index()
        same = stored.index.equals(rebuilt.index) and np.allclose(
            stored[FEATURES_NUM].to_numpy(float), rebuilt[FEATURES_NUM].to_numpy(float), equal_nan=True) \
            and (stored["country"] == rebuilt["country"]).all()
        results.append({"test": "T1_point_in_time_rebuild", "t0": t0.date(), "pass": bool(same),
                        "detail": f"{len(stored)} khách; đặc trưng dựng lại từ dữ liệu cắt tại cutoff"})

        # T2: Nhiễu tương lai – nhân 10 mọi giá trị sau cutoff, đặc trưng phải không đổi
        tx_pert = tx.copy()
        fut = tx_pert["invoice_ts"] >= cutoff
        tx_pert.loc[fut, "line_value"] *= 10
        tx_pert.loc[fut, "quantity"] *= 10
        o_p, c_p = build_orders(tx_pert)
        pert = features_at(t0, o_p, c_p, tx_pert.loc[~tx_pert["is_cancel"],
                           ["customer_id", "invoice_ts", "stock_code"]], cfg)
        pert = pert.set_index("customer_id")[FEATURES_NUM].sort_index()
        same2 = np.allclose(stored[FEATURES_NUM].to_numpy(float), pert.to_numpy(float), equal_nan=True)
        results.append({"test": "T2_future_perturbation_invariance", "t0": t0.date(), "pass": bool(same2),
                        "detail": "Giá trị/số lượng sau cutoff x10 không làm đổi đặc trưng"})

        # T3: Nhãn tính lại trực tiếp từ dòng giao dịch
        win = tx[(~tx["is_cancel"]) & (tx["invoice_ts"] >= t0) & (tx["invoice_ts"] < t0 + H * DAY)]
        buyers = set(win["customer_id"])
        s = snap[snap["t0"] == t0]
        same3 = ((s["customer_id"].isin(buyers)).astype(int) == s["y"]).all()
        results.append({"test": "T3_label_recompute", "t0": t0.date(), "pass": bool(same3),
                        "detail": f"base rate={s['y'].mean():.3f}"})

    emb = check_embargo(snap, cfg)
    results.append({"test": "T4_embargo_gap>=H", "t0": None, "pass": bool(emb["pass"].all()),
                    "detail": "; ".join(f"{r['from']}->{r['to']}: {r['gap_days']}d" for _, r in emb.iterrows())})

    dev = snap[snap["split"].isin(["train", "valid", "calib"])]
    results.append({"test": "T5_entity_holdout_isolation", "t0": None,
                    "pass": bool(dev["is_entity_holdout"].sum() == 0),
                    "detail": f"{snap.loc[snap['is_entity_holdout'] == 1, 'customer_id'].nunique()} khách giữ riêng"})

    leaked = [c for c in EXCLUDE_FROM_MODEL if c in FEATURES_NUM + FEATURES_CAT]
    results.append({"test": "T6_exclusion_list", "t0": None, "pass": not leaked,
                    "detail": "Nhãn/khóa/cột kiểm toán không nằm trong danh sách đặc trưng"})
    return pd.DataFrame(results)


def single_feature_auc(snap: pd.DataFrame) -> pd.DataFrame:
    """AUC đơn biến trên train: AUC > 0.90 là tín hiệu cần điều tra rò rỉ (Chương 3.2)."""
    from sklearn.metrics import roc_auc_score
    tr = snap[snap["split"] == "train"]
    rows = []
    for c in FEATURES_NUM:
        x = tr[c].astype(float)
        m = x.notna()
        if m.sum() < 100 or x[m].nunique() < 2:
            continue
        auc = roc_auc_score(tr.loc[m, "y"], x[m])
        rows.append({"feature": c, "auc": round(max(auc, 1 - auc), 4), "direction": "+" if auc >= .5 else "-",
                     "missing_rate": round(1 - m.mean(), 4), "flag_leakage_check": max(auc, 1 - auc) > 0.90})
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


# ----------------------------------------------------------------------------- 7. Báo cáo & hình
def split_summary(snap: pd.DataFrame) -> pd.DataFrame:
    g = snap.groupby("split")
    out = g.agg(rows=("y", "size"), customers=("customer_id", "nunique"), n_t0=("t0", "nunique"),
                t0_first=("t0", "min"), t0_last=("t0", "max"), base_rate=("y", "mean"),
                mean_value_if_buy=("y_value_h", lambda s: s[s > 0].mean()),
                entity_holdout_rows=("is_entity_holdout", "sum"))
    order = SPLIT_ORDER + ["embargo", "excluded_entity_holdout"]
    out = out.reindex([o for o in order if o in out.index])
    out["t0_first"] = out["t0_first"].dt.date
    out["t0_last"] = out["t0_last"].dt.date
    return out.round(4).reset_index()


def make_figures(snap: pd.DataFrame, cfg: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    H = cfg["decision"]["horizon_days"]
    colors = {"train": "#1f77b4", "valid": "#2ca02c", "calib": "#9467bd", "locked": "#d62728",
              "oot": "#ff7f0e", "embargo": "#bbbbbb"}
    per_t0 = (snap[snap["split"] != "excluded_entity_holdout"]
              .assign(split=lambda d: d["split"])
              .groupby(["t0", "split"]).agg(n=("y", "size"), rate=("y", "mean")).reset_index())
    outs = []

    # Hình 1: thiết kế trục thời gian (t0 + cửa sổ nhãn)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    for i, (name, a) in enumerate([(s, None) for s in SPLIT_ORDER + ["embargo"]]):
        t = per_t0.loc[per_t0["split"] == name, "t0"].drop_duplicates()
        for x in t:
            ax.plot([x, x + pd.Timedelta(days=H)], [i, i], color=colors[name], lw=6, alpha=.35,
                    solid_capstyle="butt")
            ax.plot(x, i, "o", color=colors[name], ms=5)
    ax.set_yticks(range(6), ["Train", "Validation", "Calibration", "Locked Test", "Out-of-Time", "Embargo"])
    ax.invert_yaxis()
    ax.set_title(f"Thiết kế trục thời gian: điểm = t0 (2 tuần/lần), vạch mờ = cửa sổ nhãn {H} ngày")
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    outs.append(p("outputs/figures/fig_s1_split_timeline.png"))
    fig.savefig(outs[-1], dpi=160)
    plt.close(fig)

    # Hình 2: tỷ lệ mua lại và số khách đủ điều kiện theo t0
    tot = (snap[snap["split"] != "excluded_entity_holdout"].groupby("t0")
           .agg(n=("y", "size"), rate=("y", "mean"), split=("split", "first")).reset_index())
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.bar(tot["t0"], tot["n"], width=10, color=[colors[s] for s in tot["split"]], alpha=.35)
    ax1.set_ylabel("Số khách đủ điều kiện")
    ax2 = ax1.twinx()
    ax2.plot(tot["t0"], tot["rate"], "-o", color="black", ms=4)
    ax2.set_ylabel(f"Tỷ lệ mua lại trong {H} ngày")
    ax1.set_title("Quy mô quần thể và tỷ lệ nhãn theo t0 (màu cột = tập dữ liệu) — mùa Q4 tạo độ trôi tự nhiên")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s], alpha=.5) for s in colors]
    ax1.legend(handles, list(colors), loc="upper left", ncol=6, fontsize=8)
    fig.tight_layout()
    outs.append(p("outputs/figures/fig_s1_base_rate_by_t0.png"))
    fig.savefig(outs[-1], dpi=160)
    plt.close(fig)
    return outs


# ----------------------------------------------------------------------------- main
def main() -> None:
    cfg = load_config()
    paths = cfg["paths"]
    print("[1/7] Nạp dữ liệu thô...")
    raw = load_raw(cfg)
    print(f"      {len(raw):,} dòng")

    print("[2/7] Làm sạch...")
    tx, clog = clean(raw, cfg)
    orders, cancels = build_orders(tx)
    tx.to_parquet(p(paths["transactions"]), index=False)
    orders.to_parquet(p(paths["orders"]), index=False)
    cancels.to_parquet(p("data/processed/cancels.parquet"), index=False)
    clog.to_csv(p("outputs/tables/s1_cleaning_log.csv"), index=False)
    print(f"      {len(tx):,} dòng sạch | {len(orders):,} đơn | {len(cancels):,} hủy | "
          f"{orders['customer_id'].nunique():,} khách")

    print("[3/7] Dựng snapshot point-in-time...")
    snap = build_snapshots(tx, orders, cancels, cfg)
    snap.to_parquet(p(paths["snapshots"]), index=False)
    print(f"      {len(snap):,} dòng (khách x t0), {snap['t0'].nunique()} kỳ t0")

    print("[4/7] Kiểm tra hợp đồng dữ liệu (pandera)...")
    snapshot_schema(cfg).validate(snap, lazy=True)
    print("      PASS")
    pd.DataFrame([{"column": k, "role": v[0], "description": v[1],
                   "dtype": str(snap[k].dtype) if k in snap else ""}
                  for k, v in DATA_DICTIONARY.items()]).to_csv(
        p("outputs/tables/s1_data_dictionary.csv"), index=False)

    print("[5/7] Kiểm thử rò rỉ...")
    lt = leakage_tests(tx, orders, cancels, snap, cfg)
    lt.to_csv(p("outputs/tables/s1_leakage_tests.csv"), index=False)
    print(lt[["test", "t0", "pass"]].to_string(index=False))
    assert lt["pass"].all(), "Có kiểm thử rò rỉ thất bại – dừng pipeline."
    sfa = single_feature_auc(snap)
    sfa.to_csv(p("outputs/tables/s1_single_feature_auc.csv"), index=False)
    check_embargo(snap, cfg).to_csv(p("outputs/tables/s1_embargo_check.csv"), index=False)

    print("[6/7] Bảng tóm tắt & hình...")
    ss = split_summary(snap)
    ss.to_csv(p("outputs/tables/s1_split_summary.csv"), index=False)
    print(ss.to_string(index=False))
    figs = make_figures(snap, cfg)

    print("[7/7] Manifest SHA-256...")
    files = [paths["raw_zip"], paths["raw_xlsx"], paths["transactions"], paths["orders"],
             "data/processed/cancels.parquet", paths["snapshots"],
             *[f"outputs/tables/{f}" for f in ["s1_cleaning_log.csv", "s1_data_dictionary.csv",
                                               "s1_leakage_tests.csv", "s1_single_feature_auc.csv",
                                               "s1_embargo_check.csv", "s1_split_summary.csv"]],
             *[f.relative_to(ROOT) for f in figs]]
    files = [f for f in files if (ROOT / f).exists()]
    m = write_manifest("s1", files, extra={"config_decision": cfg["decision"], "config_splits": cfg["splits"],
                                           "features_num": FEATURES_NUM, "features_cat": FEATURES_CAT})
    print(f"      {m.relative_to(ROOT)}")
    print("GĐ1 hoàn tất.")


if __name__ == "__main__":
    main()
