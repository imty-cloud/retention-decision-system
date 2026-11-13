"""GĐ7 – Action Router, guardrails, HITL (Chương 7). Chỉ valid + calib (7 kỳ); không mở locked/oot.

ĐĂNG KÝ TRƯỚC (ghi trước khi chạy):
 1. Luồng mỗi t0: kiểm tra hợp đồng đầu vào (pandera) → MILP (miền khả thi của GĐ5/GĐ6, gồm chặn cancel_rate_365 > 0.3 trong miền)
    → Router áp guardrail theo THỨ TỰ CỐ ĐỊNH lên đề xuất của MILP → HITL → nhóm đối chứng → bản ghi quyết định.
    Router nhận chính đầu ra của MILP nên mọi chặn xảy ra SAU khi giải.
 2. Thứ tự guardrail cứng (chặn thẳng về a0, ghi lý do đầu tiên kích hoạt; mọi điều kiện đúng đều ghi vào `guardrails_all`):
      G1 opt-out giả lập 8 % (hash SHA-256 của mã khách, cố định theo khách) · G2 cancel_rate_365 > 0.3 ·
      G3 frequency cap 28 ngày: chặn nếu khách đã được GỬI voucher bởi chính router cách t0 hiện tại < 28 ngày
         (lịch sử = hành động cuối đã gửi ở các kỳ trước của router; t0 cách nhau 14 ngày ⇒ chặn kỳ liền trước, cho phép lại từ kỳ thứ hai) ·
      G4 sàn biên LN theo hành động: min(d·Ŝ, cap) ≤ m·Ŝ (với m = 0.30 và d ≤ 0.20 luôn thỏa ⇒ không chặn; vẫn ghi nhận).
    Kích hoạt HITL (không chặn, chuyển người duyệt; điều kiện đúng đầu tiên theo thứ tự):
      H1 khoảng tin cậy 95 % của τ̂ (CausalForestDML) của hành động đề xuất chứa 0 · H2 khoảng conformal của Ŝ rộng: s_hi − s_lo > phân vị 95 %
      của train (tính từ pred_s2, chỉ train) · H3 khách VIP (top 2 % value_365 trong t0) được đề xuất a2/a3.
 3. KHÔNG TÁI PHÂN BỔ: ngân sách, công suất gọi và voucher bị giải phóng do chặn/HITL/đối chứng KHÔNG được giao lại cho khách khác trong
    kỳ (đơn giản, kiểm toán được). Phương án tái phân bổ chỉ được tính như ĐỐI CHỨNG phản thực (một kỳ, không dùng để chấm).
 4. Người duyệt HITL mô phỏng (chỉ dùng đầu ra mô hình, KHÔNG dùng đáp án): tính V̂ tại τ_mid = τ̂ − 0.5·(τ̂ − τ_lo) (nửa đường tới cận dưới CI);
    duyệt nếu V̂(τ_mid) ≥ 0; nếu không, với đề xuất a2/a3 thử hạ xuống a1 (cùng quy tắc, dùng CI của a1); còn không thì từ chối (a0).
    Ghi nhận mọi ghi đè (final ≠ proposed).
 5. Nhóm đối chứng: trong số khách mà sau guardrail + HITL vẫn được gửi voucher, 10 % bị giữ lại a0 theo hash SHA-256(salt|khách|t0)
    (seed cố định). Propensity ghi lại: khách thuộc tập ngẫu nhiên (`in_randomization_set`) có P(hành động cuối) = 0.9, P(a0) = 0.1;
    khách ngoài tập có P(a0) = 1 (chính sách tất định ⇒ GĐ8 chỉ nhận dạng được hiệu ứng trong tập ngẫu nhiên).
 6. Bản ghi quyết định theo schema pandera cố định (`DECISION_SCHEMA`); khóa chống lặp `customer_id|t0|policy_version`;
    decision_id = uuid5; thời điểm quyết định mô phỏng = t0 06:00 UTC (xác định, tái lập được).
 7. Đo độ trễ đầu-cuối MỘT t0 (đặc trưng → chấm điểm → MILP → router) ≥ 30 lần mỗi kỳ → P50/P95/P99. Chấm điểm là CHẤM THẬT bằng mô hình đã lưu
    (μ0 + Platt, Ŝ + conformal, CausalForestDML), và được đối chiếu với pred_s2/pred_s4.
 8. Đánh giá bằng đáp án chỉ trong evaluate_* (dùng lại evaluate_truth/evaluate_utilities của GĐ6).

AMENDMENT v2 (router-v2.0) — 2026-11-12
Bối cảnh: GĐ7 v1 trên valid/calib cho thấy bản ghi cuối vi phạm độ phủ tối
thiểu 4 lần trong 3/7 kỳ, do quy tắc "không tái phân bổ" mâu thuẫn với ràng
buộc độ phủ. Sửa đổi này được quyết định SAU KHI đã thấy kết quả v1 trên
valid/calib. Chưa mở locked/oot. Không sửa ngưỡng cổng nào. Kết quả v1 được
giữ nguyên làm hồ sơ.

(1) Kiểm tra G5 trên BẢN GHI CUỐI.
    Số tối thiểu mỗi phân khúc = min(ceil(3% × n_phân_khúc), số khách đủ điều
    kiện), trong đó n_phân_khúc = MỌI khách của phân khúc trong kỳ (giữ nguyên
    như s5 build_instance). "Đủ điều kiện" tại v2 loại: cancel_rate > 0.3,
    opt-out, khách đang trong frequency cap.
    Tử số = số khách thực sự được gửi voucher, không tính nhóm đối chứng.
    Mẫu số vẫn gồm cả khách đối chứng.
(2) Frequency cap 28 ngày và opt-out (hash_unit cố định theo customer_id)
    được đưa vào miền khả thi của MILP. Cap được tính từ bản ghi cuối v2 của
    các kỳ trước, nên các kỳ phải chạy tuần tự. Router vẫn giữ hai guardrail
    này làm lớp kiểm tra thứ hai.
(3) Bù độ phủ, chạy sau guardrail → HITL → chọn đối chứng, theo từng phân khúc
    còn dưới mức tối thiểu:
    - Danh sách dự phòng: khách cùng phân khúc, chưa được gửi trong kỳ, chưa
      bị HITL từ chối trong kỳ, qua mọi guardrail cứng, có V̂(a1) ≥ 0. Xếp theo
      V̂(a1) giảm dần. Chỉ bù bằng a1.
    - Từng khách bù đi qua guardrail và HITL như thường. HITL từ chối, hoặc
      thêm khách này sẽ vượt ngân sách / công suất gọi / trần a2, thì bỏ qua
      và lấy người kế tiếp.
    - Dừng khi đạt mức tối thiểu hoặc hết danh sách. Hết danh sách mà vẫn
      thiếu thì ghi nhận là vi phạm, không nới thêm điều kiện.
    - Khách bù không được chọn vào đối chứng (propensity đối chứng = 0, ghi
      vào log) và có mã lý do COVERAGE_BACKFILL.
    - Đây là ngoại lệ duy nhất của quy tắc "không tái phân bổ".
(4) So sánh chính sách: "MILP qua Router" và "Luật cũ qua Router" đều dùng
    router-v2.0. Với luật cũ (không có MILP), opt-out và cap được lọc ở bước
    sinh đề xuất. Bước bù vẫn dùng V̂ của mô hình, nên số khách bù và giá trị
    của họ được báo cáo riêng cho từng chính sách. MILP thuần và Luật cũ thuần
    giữ nguyên.
Giữ nguyên: mọi ngưỡng cổng, thứ tự guardrail, luật HITL, ngưỡng Ŝ (£1 833),
tỷ lệ đối chứng 10%, người duyệt mô phỏng.
"""
from __future__ import annotations

import hashlib
import json
import sys
import uuid
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import time

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pandera.pandas as pa
except ImportError:  # pandera < 0.24
    import pandera as pa

import common
import s2_predict  # noqa: F401  (lớp Platt/Isotonic phải import được để nạp calibrator)
import s5_spo as S5
import s6_milp as S6
from s1_data import FEATURES_CAT, FEATURES_NUM
from s4_causal import FEATS_C, predict_tau, predict_tau_interval
from s5b_dr_diagnostics import legacy_scores, load_data

CFG = common.load_config()
SEED = CFG["project"]["seed"]
TABLES, FIGS = Path(CFG["paths"]["tables"]), Path(CFG["paths"]["figures"])
BIZ = CFG["business"]
M0 = BIZ["gross_margin_rate"]
TREATED = S5.TREATED
ACTIONS = S5.ACTIONS
CHANNEL = {"a0": "none", "a1": "email", "a2": "email", "a3": "call"}
POLICY_VERSION = "router-v1.0"
NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "ptdlnc-2026/retention-router")
SALT = "ptdlnc-router-2026"
OPT_OUT_PCT, FREQ_CAP_DAYS, CONTROL_PCT, VIP_PCT = 8, BIZ["frequency_cap_days"], BIZ["control_holdout_pct"], 2
CANCEL_MAX = BIZ["guardrail_cancel_rate_max"]
HARD = ["G1_opt_out", "G2_cancel_rate", "G3_frequency_cap", "G4_margin_floor"]
SOFT = ["H1_CI_tau_chua_0", "H2_S_khoang_rong", "H3_VIP_a2_a3"]
N_LAT = 30


# ============================================================================ schema
def _schemas():
    inp = pa.DataFrameSchema({
        "customer_id": pa.Column(int, nullable=False),
        "t0": pa.Column(pa.DateTime, nullable=False),
        "cancel_rate_365": pa.Column(float, pa.Check.in_range(0, 1), nullable=True),
        "value_365": pa.Column(float, pa.Check.ge(0), nullable=False),
        "s_hat_mean": pa.Column(float, pa.Check.gt(0), nullable=False),
        "s_lo": pa.Column(float, pa.Check.gt(0), nullable=False),
        "s_hi": pa.Column(float, pa.Check.gt(0), nullable=False),
        "mu0_cal": pa.Column(float, pa.Check.in_range(0, 1), nullable=False),
        **{f"{p}_{a}": pa.Column(float, nullable=False) for p in ("V_hat", "tau_hat", "tau_lo", "tau_hi") for a in TREATED},
    }, checks=[pa.Check(lambda d: (d.s_hi >= d.s_lo).all(), error="s_hi < s_lo"),
               pa.Check(lambda d: ~d.duplicated(["customer_id", "t0"]).any(), error="trùng khóa khách × t0")],
        coerce=True, strict=False)
    dec = pa.DataFrameSchema({
        "decision_id": pa.Column(str, unique=True),
        "idempotency_key": pa.Column(str, unique=True),
        "customer_id": pa.Column(int), "t0": pa.Column(pa.DateTime), "split": pa.Column(str),
        "policy_version": pa.Column(str), "model_version": pa.Column(str),
        "proposed_action": pa.Column(str, pa.Check.isin(ACTIONS)), "final_action": pa.Column(str, pa.Check.isin(ACTIONS)),
        "channel": pa.Column(str, pa.Check.isin(["none", "email", "call"])),
        "guardrail_triggered": pa.Column(str), "guardrails_all": pa.Column(str),
        "hitl_flag": pa.Column(bool), "hitl_trigger": pa.Column(str), "hitl_outcome": pa.Column(str),
        "control_group": pa.Column(bool), "in_randomization_set": pa.Column(bool),
        "propensity_logged": pa.Column(float, pa.Check.in_range(0, 1)), "propensity_a0": pa.Column(float, pa.Check.in_range(0, 1)),
        "V_hat_proposed": pa.Column(float), "tau_hat": pa.Column(float, nullable=True), "tau_lo": pa.Column(float, nullable=True),
        "tau_hi": pa.Column(float, nullable=True), "s_hat_mean": pa.Column(float), "s_lo": pa.Column(float), "s_hi": pa.Column(float),
        "reason_1": pa.Column(str, nullable=True), "reason_2": pa.Column(str, nullable=True), "reason_3": pa.Column(str, nullable=True),
        "decided_at": pa.Column(pa.DateTime),
    }, checks=[
        pa.Check(lambda d: ~(d.control_group & (d.final_action != "a0")), error="đối chứng phải có final_action = a0"),
        pa.Check(lambda d: ~(d.control_group & ~d.in_randomization_set), error="đối chứng phải thuộc tập ngẫu nhiên"),
        pa.Check(lambda d: ~((d.guardrail_triggered != "none") & (d.final_action != "a0")), error="bị chặn cứng phải về a0"),
        pa.Check(lambda d: ~((d.proposed_action == "a0") & (d.final_action != "a0")), error="Router không được tự thêm voucher"),
        pa.Check(lambda d: (d.channel == d.final_action.map(CHANNEL)).all(), error="kênh không khớp hành động cuối"),
        pa.Check(lambda d: ~(d.hitl_flag & (d.hitl_outcome == "NA")), error="HITL phải có kết quả duyệt"),
    ], coerce=True, strict=True)
    return inp, dec


INPUT_SCHEMA, DECISION_SCHEMA = _schemas()


def hash_unit(*parts) -> float:
    """Số xác định trong [0,1) từ SHA-256 của các thành phần (dùng cho opt-out và đối chứng)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


# ============================================================================ người duyệt mô phỏng (không dùng đáp án)
def reviewer_decision(tau_hat, tau_lo, mu0, S, action: int) -> np.ndarray:
    """V̂ tại nửa đường từ τ̂ tới cận dưới CI ≥ 0 ? (vector hóa; action ∈ {1,2,3})."""
    tau_mid = tau_hat - 0.5 * (tau_hat - tau_lo)
    v = M0 * S * tau_mid - np.minimum(S5.DISC[action] * S, S5.CAP[action]) * np.clip(mu0 + tau_mid, 0, 1) - S5.CONTACT[action]
    return v >= 0


# ============================================================================ Router
class ActionRouter:
    def __init__(self, s_wide_threshold: float, model_version: str, policy_version: str = POLICY_VERSION):
        self.s_wide, self.model_version, self.policy_version = float(s_wide_threshold), model_version, policy_version
        self.history: dict[int, np.datetime64] = {}          # khách → t0 gần nhất đã được GỬI voucher
        self.log: dict[str, dict] = {}                      # khóa chống lặp → bản ghi
        self.frames: list[pd.DataFrame] = []

    def validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        return INPUT_SCHEMA.validate(df, lazy=True)

    def route(self, df: pd.DataFrame, w: np.ndarray, split: str, dry_run: bool = False) -> pd.DataFrame:
        """df: đầu vào đã chấm điểm của MỘT t0 (mọi khách); w: n×4 đề xuất của MILP. Trả bản ghi quyết định của kỳ."""
        t0 = pd.Timestamp(df.t0.iloc[0])
        keys = (df.customer_id.astype(str) + "|" + str(t0.date()) + "|" + self.policy_version).to_numpy()
        if not dry_run and all(k in self.log for k in keys):                       # chống lặp: kỳ đã xử lý ⇒ trả lại bản ghi cũ
            return pd.DataFrame([self.log[k] for k in keys])
        n = len(df)
        prop = np.asarray(w).argmax(1)                                              # 0..3
        has = prop > 0
        S, mu0 = df.s_hat_mean.to_numpy(), df.mu0_cal.to_numpy()
        tau = df[[f"tau_hat_{a}" for a in TREATED]].to_numpy()
        lo, hi = df[[f"tau_lo_{a}" for a in TREATED]].to_numpy(), df[[f"tau_hi_{a}" for a in TREATED]].to_numpy()
        V = df[[f"V_hat_{a}" for a in TREATED]].to_numpy()
        idx = np.arange(n)
        pj = np.maximum(prop - 1, 0)
        # ---- guardrail cứng (điều kiện đánh giá cho MỌI khách được đề xuất voucher để kiểm toán)
        cid = df.customer_id.to_numpy()
        opt = np.array([hash_unit("optout", SALT, c) < OPT_OUT_PCT / 100 for c in cid])
        canc = df.cancel_rate_365.fillna(0).to_numpy() > CANCEL_MAX
        last = np.array([self.history.get(int(c), np.datetime64("NaT", "ns")) for c in cid], dtype="datetime64[ns]")
        days = (np.datetime64(t0, "ns") - last) / np.timedelta64(1, "D")
        freq = ~np.isnat(last) & (days < FREQ_CAP_DAYS)
        marg = np.array([min(S5.DISC[a] * s, S5.CAP[a]) > M0 * s + 1e-9 for a, s in zip(prop, S)])
        conds = {"G1_opt_out": opt, "G2_cancel_rate": canc, "G3_frequency_cap": freq, "G4_margin_floor": marg}
        # ---- điều kiện HITL
        ci0 = (lo[idx, pj] <= 0) & (hi[idx, pj] >= 0)
        wide = (df.s_hi.to_numpy() - df.s_lo.to_numpy()) > self.s_wide
        vip = (df.groupby("t0").value_365.rank(pct=True).to_numpy() >= 1 - VIP_PCT / 100) & (prop >= 2)
        soft = {"H1_CI_tau_chua_0": ci0, "H2_S_khoang_rong": wide, "H3_VIP_a2_a3": vip}
        first_hard = np.full(n, "none", dtype=object)
        for g in reversed(HARD):
            first_hard[has & conds[g]] = g
        hard_block = first_hard != "none"
        first_soft = np.full(n, "none", dtype=object)
        for g in reversed(SOFT):
            first_soft[has & ~hard_block & soft[g]] = g
        hitl = first_soft != "none"
        # ---- người duyệt mô phỏng
        final = prop.copy()
        final[hard_block] = 0
        outcome = np.full(n, "NA", dtype=object)
        for i in np.where(hitl)[0]:
            a = prop[i]
            if reviewer_decision(tau[i, a - 1], lo[i, a - 1], mu0[i], S[i], a):
                outcome[i] = "approved"
            elif a >= 2 and reviewer_decision(tau[i, 0], lo[i, 0], mu0[i], S[i], 1):
                final[i], outcome[i] = 1, "downgraded"
            else:
                final[i], outcome[i] = 0, "rejected"
        # ---- nhóm đối chứng (trong số khách vẫn được gửi)
        sendable = final > 0
        ctrl = np.array([sendable[i] and hash_unit("control", SALT, cid[i], t0.date()) < CONTROL_PCT / 100 for i in range(n)])
        final_c = final.copy()
        final_c[ctrl] = 0
        p_ctrl = CONTROL_PCT / 100
        prop_logged = np.where(sendable, np.where(ctrl, p_ctrl, 1 - p_ctrl), 1.0)
        prop_a0 = np.where(sendable, p_ctrl, 1.0)
        # ---- cập nhật lịch sử: chỉ voucher THỰC SỰ gửi
        if not dry_run:
            for i in np.where(final_c > 0)[0]:
                self.history[int(cid[i])] = np.datetime64(t0, "ns")
        # ---- bản ghi
        act = np.array(ACTIONS)
        rc = [df.get(f"reason_{k}_code", pd.Series([None] * n, index=df.index)).to_numpy() for k in (1, 2, 3)]
        out = pd.DataFrame({
            "decision_id": [str(uuid.uuid5(NAMESPACE, k)) for k in keys], "idempotency_key": keys,
            "customer_id": cid.astype(int), "t0": t0, "split": split, "policy_version": self.policy_version,
            "model_version": self.model_version, "proposed_action": act[prop], "final_action": act[final_c],
            "channel": [CHANNEL[a] for a in act[final_c]], "guardrail_triggered": first_hard,
            "guardrails_all": ["+".join([g for g in HARD if conds[g][i]] + [g for g in SOFT if soft[g][i]]) if has[i] else ""
                               for i in range(n)],
            "hitl_flag": hitl, "hitl_trigger": first_soft, "hitl_outcome": outcome, "control_group": ctrl,
            "in_randomization_set": sendable, "propensity_logged": prop_logged, "propensity_a0": prop_a0,
            "V_hat_proposed": np.where(has, V[idx, pj], 0.0),
            "tau_hat": np.where(has, tau[idx, pj], np.nan), "tau_lo": np.where(has, lo[idx, pj], np.nan),
            "tau_hi": np.where(has, hi[idx, pj], np.nan), "s_hat_mean": S, "s_lo": df.s_lo.to_numpy(), "s_hi": df.s_hi.to_numpy(),
            "reason_1": rc[0], "reason_2": rc[1], "reason_3": rc[2],
            "decided_at": t0 + pd.Timedelta(hours=6)})
        out["guardrails_all"] = out.guardrails_all.replace("", "none")
        out = DECISION_SCHEMA.validate(out, lazy=True)
        if not dry_run:
            for r in out.to_dict("records"):
                self.log[r["idempotency_key"]] = r
            self.frames.append(out)
        self.last_conds = {**conds, **soft}      # điều kiện từng guardrail của kỳ vừa xử lý (phục vụ kiểm toán)
        return out


# ============================================================================ chấm điểm thật (đặc trưng → mô hình)
class Scorer:
    """Nạp mô hình một lần; score(df) trả cột μ̂0, Ŝ, khoảng conformal, τ̂ ± CI, V̂ cho MỘT t0."""

    def __init__(self):
        self.mu0 = joblib.load(common.p("models/s2_mu0_model.joblib"))
        self.cal = joblib.load(common.p("models/s2_calibrator.joblib"))["calibrator"]
        self.basket = joblib.load(common.p("models/s2_basket_model.joblib"))
        self.bundle = joblib.load(common.p("models/s4_cate_bundle.joblib"))
        ex = json.loads((common.ROOT / "outputs/manifests/s2_manifest.json").read_text(encoding="utf-8"))["extra"]
        self.q, self.smear = float(ex["conformal_q_log"]), float(ex["smearing_duan"])
        self.feats = FEATURES_NUM + FEATURES_CAT
        h = {k: hashlib.sha256(Path(common.p(f"models/{k}.joblib")).read_bytes()).hexdigest()[:8]
             for k in ("s2_mu0_model", "s2_calibrator", "s2_basket_model", "s4_cate_bundle")}
        self.version = "mu0:" + h["s2_mu0_model"] + "|cal:" + h["s2_calibrator"] + "|S:" + h["s2_basket_model"] + "|cate:" + h["s4_cate_bundle"]

    def features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        return df[self.feats], self.bundle["pre"].transform(df[FEATS_C])

    def score(self, X: pd.DataFrame, Xc: np.ndarray) -> dict:
        raw = np.clip(self.mu0.predict_proba(X)[:, 1], s2_predict.EPS, 1 - s2_predict.EPS)
        mu_cal = self.cal.predict(raw)
        lp = self.basket.predict(X)
        s_hat = np.exp(lp)
        tau = predict_tau(self.bundle["learner"], Xc)
        lo, hi = predict_tau_interval(self.bundle["cf"], Xc)
        s_mean = s_hat * self.smear
        V = S6.value_hat_m(tau, mu_cal, s_mean, M0)
        return dict(mu0_raw=raw, mu0_cal=mu_cal, s_hat=s_hat, s_hat_mean=s_mean, s_lo=np.exp(lp - self.q), s_hi=np.exp(lp + self.q),
                    tau=tau, tau_lo=lo, tau_hi=hi, V=V)


def scored_frame(base: pd.DataFrame, sc: dict) -> pd.DataFrame:
    d = base[["customer_id", "t0", "cancel_rate_365", "value_365"]].copy()
    for k in ("s_hat_mean", "s_lo", "s_hi", "mu0_cal"):
        d[k] = sc[k]
    for j, a in enumerate(TREATED):
        d[f"V_hat_{a}"], d[f"tau_hat_{a}"], d[f"tau_lo_{a}"], d[f"tau_hi_{a}"] = sc["V"][:, j], sc["tau"][:, j], sc["tau_lo"][:, j], sc["tau_hi"][:, j]
    return d


# ============================================================================ đánh giá (được đọc đáp án)
def evaluate_period(T: dict, idx: np.ndarray, w: np.ndarray, m: float = M0) -> dict:
    return S6.evaluate_policy(T, idx, w, m)


def evaluate_unit_values(T: dict, idx: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Giá trị thật (hiện thực, kỳ vọng) CỦA TỪNG KHÁCH dưới hành động w (n×4 one-hot)."""
    Ur, Ue = S6.evaluate_utilities({k: v[idx] for k, v in T.items()}, M0)
    return (w * Ur).sum(1), (w * Ue).sum(1)


# ============================================================================ chính
def pipeline_once(scorer: Scorer, base: pd.DataFrame, ins: dict, router: ActionRouter, split: str, dry_run=True) -> dict:
    """Một lượt đầu-cuối cho một t0: đặc trưng → chấm điểm → MILP → router. Trả thời gian từng khâu (giây)."""
    t = time.perf_counter()
    X, Xc = scorer.features(base)
    t_feat = time.perf_counter() - t
    t = time.perf_counter()
    sc = scorer.score(X, Xc)
    frame = scored_frame(base, sc)
    t_score = time.perf_counter() - t
    t = time.perf_counter()
    frame = router.validate_input(frame)
    t_contract = time.perf_counter() - t
    t = time.perf_counter()
    r = S6.solve_hs(ins, sc["V"])
    t_milp = time.perf_counter() - t
    t = time.perf_counter()
    dec = router.route(frame, r["w"], split, dry_run=dry_run)
    t_router = time.perf_counter() - t
    return dict(t_dac_trung=t_feat, t_cham_diem=t_score, t_hop_dong=t_contract, t_milp=t_milp, t_router=t_router,
                tong=t_feat + t_score + t_contract + t_milp + t_router, frame=frame, w=r["w"], dec=dec, sc=sc)


def main() -> None:
    np.random.seed(SEED)
    t_start = time.time()
    files: list[str] = []
    pd.set_option("display.width", 250)
    df_all = load_data()
    lg = pd.read_parquet(common.p("data/processed/logged.parquet"))
    lg = lg[lg.split.isin(["train", "valid", "calib"])].reset_index(drop=True)
    assert (lg.customer_id.to_numpy() == df_all.customer_id.to_numpy()).all()
    df = df_all.copy()
    for c in lg.columns:                                                  # đặc trưng thô cho chấm điểm thật
        if c not in df.columns:
            df[c] = lg[c].to_numpy()
    rs = pd.read_parquet(common.p("data/processed/reasons_s3.parquet"))
    df = df.merge(rs[["customer_id", "t0", "reason_1_code", "reason_2_code", "reason_3_code"]], on=["customer_id", "t0"], how="left",
                  validate="one_to_one")
    p2 = pd.read_parquet(common.p("data/processed/pred_s2.parquet"), columns=["customer_id", "t0", "split", "s_lo", "s_hi", "mu0_cal"])
    trn = p2[p2.split == "train"]
    s_wide = float(np.percentile(trn.s_hi - trn.s_lo, 95))               # chỉ train
    p4 = pd.read_parquet(common.p("data/processed/pred_s4.parquet"))
    df = df.merge(p2[["customer_id", "t0", "s_lo", "s_hi"]], on=["customer_id", "t0"], how="left", validate="one_to_one")
    df = df.merge(p4[["customer_id", "t0", "tau_lo_a1", "tau_lo_a2", "tau_lo_a3", "tau_hi_a1", "tau_hi_a2", "tau_hi_a3"]],
                  on=["customer_id", "t0"], how="left", validate="one_to_one")
    meta = df[["split", "t0", S5.SEG_COL, "cancel_rate_365"]].copy()
    scorer = Scorer()
    legacy = legacy_scores(df)
    per = S6.periods(df)
    vc = df.split.isin(["valid", "calib"]).to_numpy()
    T = S6.evaluate_truth(df)
    print(f"s_wide (p95 train) = {s_wide:.1f} · {len(per)} kỳ · {scorer.version} ({time.time() - t_start:.0f}s)")

    # ---------------------------------------------------------------- chạy tuần tự 7 kỳ
    router, router_lg = ActionRouter(s_wide, scorer.version), ActionRouter(s_wide, scorer.version, POLICY_VERSION + "-legacy")
    frames_rows, val_rows, audit_rows, cond_rows, check_rows = [], [], [], [], []
    for split, t0 in per:
        ins = S6.make_ins(df, meta, split, t0)
        idx = ins["idx"]
        base = df.iloc[idx].reset_index(drop=True)
        out = pipeline_once(scorer, base, ins, router, split, dry_run=False)
        dec, w_prop, fr = out["dec"], out["w"], out["frame"]
        # đối chiếu chấm điểm thật với pred_s2/pred_s4 đã lưu
        stored_V = df.iloc[idx][[f"V_hat_{a}" for a in TREATED]].to_numpy()
        check_rows.append({"split": split, "t0": t0, "max_lech_V_hat": float(np.abs(out["sc"]["V"] - stored_V).max()),
                           "max_lech_tau": float(np.abs(out["sc"]["tau"] - df.iloc[idx][[f"tau_hat_{a}" for a in TREATED]].to_numpy()).max()),
                           "max_lech_tau_lo": float(np.abs(out["sc"]["tau_lo"] - df.iloc[idx][[f"tau_lo_{a}" for a in TREATED]].to_numpy()).max()),
                           "max_lech_mu0_cal": float(np.abs(out["sc"]["mu0_cal"] - df.iloc[idx].mu0_base.to_numpy()).max()),
                           "max_lech_s_hat_mean": float(np.abs(out["sc"]["s_hat_mean"] - df.iloc[idx].s_hat_mean.to_numpy()).max())})
        w_fin = np.eye(4)[pd.Categorical(dec.final_action, categories=ACTIONS).codes]
        w_prop4 = np.asarray(w_prop)
        # chính sách so sánh
        w_leg = S6.solve_hs(ins, legacy[idx])["w"]
        legr = router_lg.route(fr, w_leg, split, dry_run=False)
        w_leg_fin = np.eye(4)[pd.Categorical(legr.final_action, categories=ACTIONS).codes]
        # đối chứng phản thực: tái phân bổ (chặn cứng đưa vào MILP) – KHÔNG dùng để chấm
        gr_idx = np.where(dec.guardrail_triggered.to_numpy() != "none")[0]
        blocked_ids = set(base.customer_id.to_numpy()[gr_idx])
        ins_re = dict(ins)
        ins_re["blocked"] = ins["blocked"] | np.isin(base.customer_id.to_numpy(), list(blocked_ids))
        ins_re["req_uk"] = min(ins["req_uk"], int(((ins["is_uk"]) & ~ins_re["blocked"]).sum()))
        ins_re["req_non"] = min(ins["req_non"], int(((~ins["is_uk"]) & ~ins_re["blocked"]).sum()))
        w_re = S6.solve_hs(ins_re, out["sc"]["V"])["w"]
        orc = S6.evaluate_oracle_policy(T, ins)
        for nm, w in [("MILP thuần (đề xuất)", w_prop4), ("Sau Router (gồm đối chứng)", w_fin), ("Luật cũ (MILP trên điểm luật)", w_leg),
                      ("Luật cũ sau Router", w_leg_fin), ("Đối chứng phản thực: tái phân bổ (không dùng chấm)", w_re)]:
            ev = evaluate_period(T, idx, w)
            ck = S5.check_feasible(ins, w)
            val_rows.append({"split": split, "t0": t0, "chinh_sach": nm, "so_voucher": int(w[:, 1:].sum()), "so_cuoc_goi": int(w[:, 3].sum()),
                             "so_a2": int(w[:, 2].sum()), "ngan_sach_dung": float((w @ S5.CAP).sum()),
                             "V_hat_ke_hoach": float((w[:, 1:] * out["sc"]["V"]).sum()), **ev, **orc,
                             "regret_cn_hien_thuc": (orc["z_sao_hien_thuc"] - ev["gia_tri_hien_thuc"]) / abs(orc["z_sao_hien_thuc"]),
                             "regret_cn_ky_vong": (orc["z_sao_ky_vong"] - ev["gia_tri_ky_vong"]) / abs(orc["z_sao_ky_vong"]),
                             **{k: ck[k] for k in ("n_vi_pham", "vp_ngan_sach", "vp_cuoc_goi", "vp_ty_le_a2", "vp_do_phu_UK", "vp_do_phu_nonUK",
                                                    "vp_guardrail")}})
        # chi phí từng guardrail / HITL / đối chứng (giá trị thật của phần bị bỏ)
        vr_p, ve_p = evaluate_unit_values(T, idx, w_prop4)
        vr_f, ve_f = evaluate_unit_values(T, idx, w_fin)
        grp = np.where(dec.guardrail_triggered.to_numpy() != "none", dec.guardrail_triggered.to_numpy(),
                       np.where(dec.control_group.to_numpy(), "Đối chứng 10%", np.where(dec.hitl_flag.to_numpy(), "HITL:" + dec.hitl_outcome.to_numpy().astype(str), "tu_dong")))
        grp = np.where(dec.proposed_action.to_numpy() == "a0", "khong_de_xuat", grp)
        for g in np.unique(grp):
            m_ = grp == g
            audit_rows.append({"split": split, "t0": t0, "nhom": g, "n": int(m_.sum()),
                               "V_hat_de_xuat": float(dec.V_hat_proposed.to_numpy()[m_].sum()),
                               "gia_tri_that_hien_thuc_de_xuat": float(vr_p[m_].sum()), "gia_tri_that_ky_vong_de_xuat": float(ve_p[m_].sum()),
                               "gia_tri_that_hien_thuc_cuoi": float(vr_f[m_].sum()), "gia_tri_that_ky_vong_cuoi": float(ve_f[m_].sum())})
        for g, v in router.last_conds.items():
            vv = np.asarray(v) & (dec.proposed_action.to_numpy() != "a0")
            cond_rows.append({"split": split, "t0": t0, "guardrail": g, "n_dieu_kien_dung": int(vv.sum()),
                              "n_de_xuat_voucher": int((dec.proposed_action.to_numpy() != "a0").sum()),
                              "n_kich_hoat_dau_tien": int((dec.guardrail_triggered.to_numpy() == g).sum() + (dec.hitl_trigger.to_numpy() == g).sum())})
        frames_rows.append(dec.assign(V_hat_de_xuat=dec.V_hat_proposed, vr=vr_p, ve=ve_p, vr_cuoi=vr_f, ve_cuoi=ve_f))
        print(f"  {split} {pd.Timestamp(t0).date()}: đề xuất {int((w_prop4[:, 1:].sum()))} · cuối {int(w_fin[:, 1:].sum())} · "
              f"HITL {int(dec.hitl_flag.sum())} · đối chứng {int(dec.control_group.sum())} ({time.time() - t_start:.0f}s)")
    log = pd.concat(frames_rows, ignore_index=True)
    dlog = DECISION_SCHEMA.validate(log[list(DECISION_SCHEMA.columns)], lazy=True)
    dlog.to_parquet(common.p("data/processed/decisions_s7.parquet"), index=False)
    files.append("data/processed/decisions_s7.parquet")
    pd.DataFrame(check_rows).to_csv(common.p(TABLES / "s7_scoring_consistency.csv"), index=False)
    vt = pd.DataFrame(val_rows)
    vt.to_csv(common.p(TABLES / "s7_policy_value_by_period.csv"), index=False)
    ct = pd.DataFrame(cond_rows)
    ct.to_csv(common.p(TABLES / "s7_guardrail_rates_by_period.csv"), index=False)
    au = pd.DataFrame(audit_rows)
    au.to_csv(common.p(TABLES / "s7_guardrail_cost_by_period.csv"), index=False)
    files += [str(TABLES / f) for f in ("s7_scoring_consistency.csv", "s7_policy_value_by_period.csv", "s7_guardrail_rates_by_period.csv",
                                         "s7_guardrail_cost_by_period.csv")]
    print(pd.DataFrame(check_rows).round(8).to_string())

    # ---- tổng hợp
    summ = vt.groupby("chinh_sach").apply(lambda d: pd.Series({
        "n_ky": len(d), "so_voucher_TB": d.so_voucher.mean(), "ngan_sach_dung_TB": d.ngan_sach_dung.mean(),
        "V_hat_ke_hoach_TB": d.V_hat_ke_hoach.mean(), "thuc_hien_thuc_TB": d.gia_tri_hien_thuc.mean(), "thuc_ky_vong_TB": d.gia_tri_ky_vong.mean(),
        "thuc_hien_thuc_cong_don": d.gia_tri_hien_thuc.sum(), "thuc_ky_vong_cong_don": d.gia_tri_ky_vong.sum(),
        "regret_A_cong_don": (d.z_sao_hien_thuc.sum() - d.gia_tri_hien_thuc.sum()) / abs(d.z_sao_hien_thuc.sum()),
        "regret_B_cong_don": (d.z_sao_ky_vong.sum() - d.gia_tri_ky_vong.sum()) / abs(d.z_sao_ky_vong.sum()),
        "tong_vi_pham_rang_buoc": int(d.n_vi_pham.sum()), "ky_vi_pham_do_phu_UK": int(d.vp_do_phu_UK.sum()),
        "ky_vi_pham_do_phu_nonUK": int(d.vp_do_phu_nonUK.sum()), "ky_vi_pham_ty_le_a2": int(d.vp_ty_le_a2.sum())}),
        include_groups=False).reset_index()
    summ.to_csv(common.p(TABLES / "s7_policy_value_summary.csv"), index=False)
    files.append(str(TABLES / "s7_policy_value_summary.csv"))
    print(summ.round(3).T.to_string())

    # tỷ lệ chặn theo guardrail (tổng 7 kỳ)
    rates = ct.groupby("guardrail")[["n_dieu_kien_dung", "n_de_xuat_voucher", "n_kich_hoat_dau_tien"]].sum().reset_index()
    rates["ty_le_tren_de_xuat_dieu_kien_dung"] = rates.n_dieu_kien_dung / rates.n_de_xuat_voucher
    rates["ty_le_tren_de_xuat_kich_hoat_dau_tien"] = rates.n_kich_hoat_dau_tien / rates.n_de_xuat_voucher
    order = {g: i for i, g in enumerate(HARD + SOFT)}
    rates = rates.sort_values("guardrail", key=lambda s: s.map(order))
    rates.to_csv(common.p(TABLES / "s7_guardrail_rates.csv"), index=False)
    files.append(str(TABLES / "s7_guardrail_rates.csv"))
    print(rates.round(3).to_string())
    # luồng quyết định
    d = dlog
    prop_v = d.proposed_action != "a0"
    flow = pd.DataFrame([{
        "de_xuat_voucher": int(prop_v.sum()), "bi_chan_cung": int((d.guardrail_triggered != "none").sum()),
        "chuyen_HITL": int(d.hitl_flag.sum()), "HITL_duyet": int((d.hitl_outcome == "approved").sum()),
        "HITL_ha_a1": int((d.hitl_outcome == "downgraded").sum()), "HITL_tu_choi": int((d.hitl_outcome == "rejected").sum()),
        "doi_chung": int(d.control_group.sum()), "gui_tu_dong_khong_HITL": int((prop_v & (d.guardrail_triggered == "none") & ~d.hitl_flag & ~d.control_group).sum()),
        "gui_thuc_te": int((d.final_action != "a0").sum()), "so_khach_x_ky": len(d)}])
    for c in ("bi_chan_cung", "chuyen_HITL", "doi_chung", "gui_tu_dong_khong_HITL", "gui_thuc_te"):
        flow[f"ty_le_{c}_tren_de_xuat"] = flow[c] / flow.de_xuat_voucher
    flow["ty_le_HITL_ghi_de"] = (flow.HITL_ha_a1 + flow.HITL_tu_choi) / flow.chuyen_HITL
    flow.to_csv(common.p(TABLES / "s7_decision_flow.csv"), index=False)
    files.append(str(TABLES / "s7_decision_flow.csv"))
    print(flow.T.to_string())
    hb = d[d.hitl_flag].groupby(["hitl_trigger", "hitl_outcome"]).size().rename("n").reset_index()
    hb.to_csv(common.p(TABLES / "s7_hitl_breakdown.csv"), index=False)
    files.append(str(TABLES / "s7_hitl_breakdown.csv"))
    # hàng đợi HITL mẫu cho người duyệt (kèm mã lý do, V̂, τ̂ ± CI, Ŝ)
    q = d[d.hitl_flag].sort_values(["t0", "V_hat_proposed"], ascending=[True, False])[[
        "decision_id", "customer_id", "t0", "proposed_action", "final_action", "hitl_trigger", "hitl_outcome", "V_hat_proposed", "tau_hat", "tau_lo",
        "tau_hi", "s_hat_mean", "s_lo", "s_hi", "reason_1", "reason_2", "reason_3"]]
    q.head(300).to_csv(common.p(TABLES / "s7_hitl_queue_sample.csv"), index=False)
    files.append(str(TABLES / "s7_hitl_queue_sample.csv"))

    # chi phí guardrail tổng
    cost = au.groupby("nhom")[["n", "V_hat_de_xuat", "gia_tri_that_hien_thuc_de_xuat", "gia_tri_that_ky_vong_de_xuat",
                                "gia_tri_that_hien_thuc_cuoi", "gia_tri_that_ky_vong_cuoi"]].sum().reset_index()
    cost["anh_huong_hien_thuc_len_tong"] = cost.gia_tri_that_hien_thuc_cuoi - cost.gia_tri_that_hien_thuc_de_xuat
    cost["anh_huong_ky_vong_len_tong"] = cost.gia_tri_that_ky_vong_cuoi - cost.gia_tri_that_ky_vong_de_xuat
    cost.to_csv(common.p(TABLES / "s7_guardrail_cost.csv"), index=False)
    files.append(str(TABLES / "s7_guardrail_cost.csv"))
    print(cost.round(1).to_string())
    # người duyệt có phân biệt được không (đáp án chỉ để đánh giá)
    hq = frames_rows and pd.concat(frames_rows)[lambda x: x.hitl_flag]
    hqt = hq.groupby("hitl_outcome").agg(n=("customer_id", "size"), V_hat_TB=("V_hat_de_xuat", "mean"), gia_tri_that_hien_thuc_TB=("vr", "mean"),
                                         gia_tri_that_ky_vong_TB=("ve", "mean")).reset_index()
    hqt.to_csv(common.p(TABLES / "s7_hitl_reviewer_value.csv"), index=False)
    files.append(str(TABLES / "s7_hitl_reviewer_value.csv"))
    print(hqt.round(2).to_string())

    # ---------------------------------------------------------------- độ trễ đầu-cuối (≥ 30 lượt mỗi kỳ)
    lat = []
    for split, t0 in per:
        ins = S6.make_ins(df, meta, split, t0)
        base = df.iloc[ins["idx"]].reset_index(drop=True)
        r_dry = ActionRouter(s_wide, scorer.version)
        pipeline_once(scorer, base, ins, r_dry, split)                                       # khởi động (không tính)
        for rep in range(N_LAT):
            o = pipeline_once(scorer, base, ins, r_dry, split)
            lat.append({"split": split, "t0": t0, "lan": rep, "n_khach": len(base), **{k: v for k, v in o.items() if k.startswith("t_") or k == "tong"}})
        print(f"  độ trễ {pd.Timestamp(t0).date()} xong ({time.time() - t_start:.0f}s)")
    lt = pd.DataFrame(lat)
    lt.to_csv(common.p(TABLES / "s7_latency_runs.csv"), index=False)
    stages = ["t_dac_trung", "t_cham_diem", "t_hop_dong", "t_milp", "t_router", "tong"]
    pct = pd.DataFrame([{"khau": s, "P50": lt[s].quantile(.5), "P95": lt[s].quantile(.95), "P99": lt[s].quantile(.99), "max": lt[s].max(),
                         "TB": lt[s].mean()} for s in stages])
    pct.to_csv(common.p(TABLES / "s7_latency_percentiles.csv"), index=False)
    per_t0 = lt.groupby("t0").tong.agg(["mean", lambda s: s.quantile(.95), "max"]).reset_index()
    per_t0.columns = ["t0", "TB", "P95", "max"]
    per_t0.to_csv(common.p(TABLES / "s7_latency_by_period.csv"), index=False)
    files += [str(TABLES / f) for f in ("s7_latency_runs.csv", "s7_latency_percentiles.csv", "s7_latency_by_period.csv")]
    print(pct.round(3).to_string())
    print(f"G6 (P95 ≤ {CFG['release_gates']['G6_latency_p95_seconds_max']} s): P95 = {pct.set_index('khau').loc['tong', 'P95']:.2f} s")

    # ---------------------------------------------------------------- kiểm tra chống lặp
    n_before = len(router.log)
    ps0 = per[0]
    ins0 = S6.make_ins(df, meta, ps0[0], ps0[1])
    again = router.route(scored_frame(df.iloc[ins0["idx"]].reset_index(drop=True), scorer.score(*scorer.features(df.iloc[ins0["idx"]].reset_index(drop=True)))),
                         S6.solve_hs(ins0, df.iloc[ins0["idx"]][[f"V_hat_{a}" for a in TREATED]].to_numpy())["w"], ps0[0])
    idem = pd.DataFrame([{"so_ban_ghi_truoc": n_before, "so_ban_ghi_sau_chay_lai_ky_dau": len(router.log), "chay_lai_tra_ban_ghi_cu":
                          bool(again.decision_id.isin(dlog.decision_id).all()), "khoa_chong_lap_duy_nhat": bool(dlog.idempotency_key.is_unique)}])
    idem.to_csv(common.p(TABLES / "s7_idempotency.csv"), index=False)
    files.append(str(TABLES / "s7_idempotency.csv"))
    print(idem.to_string())

    files += plots(vt, au, ct, lt, flow)
    common.write_manifest("s7", files, extra={"policy_version": POLICY_VERSION, "model_version": scorer.version, "s_wide_threshold": s_wide,
                                              "control_pct": CONTROL_PCT, "opt_out_pct": OPT_OUT_PCT, "freq_cap_days": FREQ_CAP_DAYS,
                                              "n_latency_runs": len(lt), "runtime_s": round(time.time() - t_start)},
                          packages=["pandas", "numpy", "pandera", "highspy", "scikit-learn", "lightgbm", "econml", "matplotlib"])
    print(f"XONG GĐ7 ({time.time() - t_start:.0f}s)")


def plots(vt, au, ct, lt, flow) -> list[str]:
    out = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    lab = [f"{pd.Timestamp(t).date()}\n({s})" for s, t in vt[vt.chinh_sach == "MILP thuần (đề xuất)"][["split", "t0"]].to_numpy()]
    x = np.arange(len(lab))
    pols = ["MILP thuần (đề xuất)", "Sau Router (gồm đối chứng)", "Luật cũ (MILP trên điểm luật)", "Đối chứng phản thực: tái phân bổ (không dùng chấm)"]
    for ax, col, ttl in [(axes[0], "gia_tri_hien_thuc", "Giá trị thật HIỆN THỰC (A)"), (axes[1], "gia_tri_ky_vong", "Giá trị thật KỲ VỌNG (B)")]:
        for i, p in enumerate(pols):
            d = vt[vt.chinh_sach == p]
            ax.bar(x + (i - 1.5) * 0.2, d[col], 0.2, label=p.replace(" (không dùng chấm)", "*"))
        ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7); ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("GBP / kỳ"); ax.set_title(ttl); ax.legend(fontsize=7)
    fig.suptitle("Giá trị thật của chính sách trước và sau Router (* = đối chứng phản thực, không dùng để chấm)")
    fig.tight_layout(); fig.savefig(common.p(FIGS / "fig_s7_policy_value.png"), dpi=160); plt.close(fig)
    out.append(str(FIGS / "fig_s7_policy_value.png"))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    a = au[au.nhom != "khong_de_xuat"].groupby("nhom").n.sum().sort_values(ascending=False)
    ax.bar(a.index, a.values, color="tab:blue")
    for i, v in enumerate(a.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=8)
    ax.set_xticklabels(a.index, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Số khách × kỳ được MILP đề xuất voucher (tổng 7 kỳ)")
    ax.set_title("Số phận của đề xuất MILP qua Router (mỗi khách xếp vào một nhóm)")
    fig.tight_layout(); fig.savefig(common.p(FIGS / "fig_s7_decision_flow.png"), dpi=160); plt.close(fig)
    out.append(str(FIGS / "fig_s7_decision_flow.png"))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    stages = ["t_dac_trung", "t_cham_diem", "t_hop_dong", "t_milp", "t_router"]
    ax.boxplot([lt[s] for s in stages], tick_labels=["đặc trưng", "chấm điểm", "hợp đồng", "MILP", "router"], showfliers=False)
    ax.set_ylabel("Giây / một t0 (≈ 3.6k khách)"); ax.set_title(f"Độ trễ từng khâu ({len(lt)} lượt) – P95 tổng = {lt.tong.quantile(.95):.2f} s")
    fig.tight_layout(); fig.savefig(common.p(FIGS / "fig_s7_latency.png"), dpi=160); plt.close(fig)
    out.append(str(FIGS / "fig_s7_latency.png"))
    return out


# ============================================================================ ROUTER v2 (AMENDMENT v2 – router-v2.0)
POLICY_VERSION_V2 = "router-v2.0"
BACKFILL_CODE = "COVERAGE_BACKFILL"


def _schema_v2():
    cols = dict(DECISION_SCHEMA.columns)
    cols.update({
        "is_backfill": pa.Column(bool), "p_control": pa.Column(float, pa.Check.isin([0.0, CONTROL_PCT / 100])),
        "router_reason_code": pa.Column(str), "backfill_status": pa.Column(str)})
    return pa.DataFrameSchema(cols, checks=[
        pa.Check(lambda d: ~(d.control_group & (d.final_action != "a0")), error="đối chứng phải có final_action = a0"),
        pa.Check(lambda d: ~(d.control_group & ~d.in_randomization_set), error="đối chứng phải thuộc tập ngẫu nhiên"),
        pa.Check(lambda d: ~((d.guardrail_triggered != "none") & (d.final_action != "a0")), error="bị chặn cứng phải về a0"),
        pa.Check(lambda d: ~((d.proposed_action == "a0") & (d.final_action != "a0") & ~d.is_backfill),
                 error="Router chỉ được thêm voucher qua bước bù độ phủ"),
        pa.Check(lambda d: ~(d.is_backfill & (d.final_action != "a1")), error="khách bù chỉ nhận a1"),
        pa.Check(lambda d: ~(d.is_backfill & ((d.p_control != 0) | d.control_group | d.in_randomization_set)),
                 error="khách bù không thuộc đối chứng, propensity đối chứng = 0"),
        pa.Check(lambda d: ~(d.is_backfill & (d.router_reason_code != BACKFILL_CODE)), error="khách bù phải có mã COVERAGE_BACKFILL"),
        pa.Check(lambda d: (d.p_control == np.where(d.in_randomization_set, CONTROL_PCT / 100, 0.0)).all(),
                 error="p_control không khớp tập ngẫu nhiên"),
        pa.Check(lambda d: (d.channel == d.final_action.map(CHANNEL)).all(), error="kênh không khớp hành động cuối"),
        pa.Check(lambda d: ~(d.hitl_flag & (d.hitl_outcome == "NA")), error="HITL phải có kết quả duyệt"),
    ], coerce=True, strict=True)


DECISION_SCHEMA_V2 = _schema_v2()
POLICY_VERSION_V21 = "router-v2.1"
REPAIR_CODES = {1: "A2_CAP_REPAIR_A1", 2: "A2_CAP_REPAIR_A0"}


def _schema_v21():
    cols = dict(DECISION_SCHEMA_V2.columns)
    cols["repair_status"] = pa.Column(str, pa.Check.isin(["none", "downgrade_a1", "drop_a0"]))
    extra = [
        pa.Check(lambda d: ~((d.repair_status == "downgrade_a1") & ((d.proposed_action == "a0") | (d.final_action != "a1")) & ~d.is_backfill),
                 error="hạ a2→a1: hành động cuối phải là a1"),
        pa.Check(lambda d: ~((d.repair_status == "drop_a0") & ((d.final_action != "a0") | d.control_group | d.in_randomization_set)),
                 error="hạ về a0 ở bước sửa: final a0, không đối chứng, ngoài tập ngẫu nhiên (propensity a0 = 1)"),
        pa.Check(lambda d: ~((d.repair_status != "none") & (d.router_reason_code != d.repair_status.map(
            {"downgrade_a1": REPAIR_CODES[1], "drop_a0": REPAIR_CODES[2]}))), error="mã lý do bước sửa"),
    ]
    return pa.DataFrameSchema(cols, checks=list(DECISION_SCHEMA_V2.checks) + extra, coerce=True, strict=True)


DECISION_SCHEMA_V21 = _schema_v21()


def coverage_requirement(is_uk: np.ndarray, eligible: np.ndarray) -> tuple[int, int]:
    """AMENDMENT (1): min(ceil(3 % × MỌI khách của phân khúc), số khách đủ điều kiện) – giống s5 build_instance."""
    out = []
    for seg in (True, False):
        ns, el = int((is_uk == seg).sum()), int(((is_uk == seg) & eligible).sum())
        out.append(min(int(np.ceil(S5.COV * ns)), el) if ns else 0)
    return out[0], out[1]


class ActionRouterV2(ActionRouter):
    """Router v2: opt-out + cap nằm trong miền MILP (router chỉ kiểm tra lại); bù độ phủ sau guardrail → HITL → đối chứng."""

    def __init__(self, s_wide_threshold: float, model_version: str, policy_version: str = POLICY_VERSION_V2):
        super().__init__(s_wide_threshold, model_version, policy_version)
        self.repair = policy_version.startswith("router-v2.1")      # v2.1 = v2 + bước sửa khả thi cuối (trần a2)
        self.last_repair: dict = {}

    def domain_flags(self, df: pd.DataFrame) -> dict:
        """Cờ biết tại t0 (đưa vào miền MILP): opt-out, cancel_rate > 0.3, đang trong frequency cap theo bản ghi cuối v2 của kỳ trước."""
        t0 = np.datetime64(pd.Timestamp(df.t0.iloc[0]), "ns")
        cid = df.customer_id.to_numpy()
        opt = np.array([hash_unit("optout", SALT, c) < OPT_OUT_PCT / 100 for c in cid])
        canc = df.cancel_rate_365.fillna(0).to_numpy() > CANCEL_MAX
        last = np.array([self.history.get(int(c), np.datetime64("NaT", "ns")) for c in cid], dtype="datetime64[ns]")
        incap = ~np.isnat(last) & ((t0 - last) / np.timedelta64(1, "D") < FREQ_CAP_DAYS)
        return dict(opt=opt, canc=canc, incap=incap)

    def route(self, df: pd.DataFrame, w: np.ndarray, split: str, ins: dict | None = None, dry_run: bool = False) -> pd.DataFrame:
        t0 = pd.Timestamp(df.t0.iloc[0])
        keys = (df.customer_id.astype(str) + "|" + str(t0.date()) + "|" + self.policy_version).to_numpy()
        if not dry_run and all(k in self.log for k in keys):
            return pd.DataFrame([self.log[k] for k in keys])
        n = len(df)
        prop = np.asarray(w).argmax(1)
        has = prop > 0
        S, mu0 = df.s_hat_mean.to_numpy(), df.mu0_cal.to_numpy()
        tau = df[[f"tau_hat_{a}" for a in TREATED]].to_numpy()
        lo, hi = df[[f"tau_lo_{a}" for a in TREATED]].to_numpy(), df[[f"tau_hi_{a}" for a in TREATED]].to_numpy()
        V = df[[f"V_hat_{a}" for a in TREATED]].to_numpy()
        idx = np.arange(n)
        pj = np.maximum(prop - 1, 0)
        cid = df.customer_id.to_numpy()
        fl = self.domain_flags(df)
        marg_a = lambda a: np.array([min(S5.DISC[a] * s, S5.CAP[a]) > M0 * s + 1e-9 for s in S])  # noqa: E731
        marg = np.array([min(S5.DISC[a] * s, S5.CAP[a]) > M0 * s + 1e-9 for a, s in zip(prop, S)])
        conds = {"G1_opt_out": fl["opt"], "G2_cancel_rate": fl["canc"], "G3_frequency_cap": fl["incap"], "G4_margin_floor": marg}
        wide = (df.s_hi.to_numpy() - df.s_lo.to_numpy()) > self.s_wide
        vip_rank = df.value_365.rank(pct=True).to_numpy() >= 1 - VIP_PCT / 100
        ci0 = (lo[idx, pj] <= 0) & (hi[idx, pj] >= 0)
        soft = {"H1_CI_tau_chua_0": ci0, "H2_S_khoang_rong": wide, "H3_VIP_a2_a3": vip_rank & (prop >= 2)}
        first_hard = np.full(n, "none", dtype=object)
        for g in reversed(HARD):
            first_hard[has & conds[g]] = g
        hard_block = first_hard != "none"
        first_soft = np.full(n, "none", dtype=object)
        for g in reversed(SOFT):
            first_soft[has & ~hard_block & soft[g]] = g
        hitl = first_soft != "none"
        final = prop.copy()
        final[hard_block] = 0
        outcome = np.full(n, "NA", dtype=object)
        for i in np.where(hitl)[0]:
            a = prop[i]
            if reviewer_decision(tau[i, a - 1], lo[i, a - 1], mu0[i], S[i], a):
                outcome[i] = "approved"
            elif a >= 2 and reviewer_decision(tau[i, 0], lo[i, 0], mu0[i], S[i], 1):
                final[i], outcome[i] = 1, "downgraded"
            else:
                final[i], outcome[i] = 0, "rejected"
        # ---- đối chứng (giữ nguyên v1)
        sendable = final > 0
        ctrl = np.array([sendable[i] and hash_unit("control", SALT, cid[i], t0.date()) < CONTROL_PCT / 100 for i in range(n)])
        final_c = final.copy()
        final_c[ctrl] = 0
        # ---- bù độ phủ (AMENDMENT (3)) – theo từng phân khúc còn thiếu
        is_uk = np.asarray(ins["is_uk"]) if ins is not None else (df.get("is_uk", pd.Series(1, index=df.index)).to_numpy() == 1)
        eligible = ~(fl["opt"] | fl["canc"] | fl["incap"])
        req = coverage_requirement(is_uk, eligible)
        B = ins["B"] if ins is not None else np.inf
        K = ins["K"] if ins is not None else np.inf
        backfill = np.zeros(n, bool)
        bstatus = np.full(n, "", dtype=object)
        repaired = np.zeros(n, np.int8)                      # v2.1: 0 = không sửa, 1 = hạ a2 → a1, 2 = hạ a2 → a0
        rep_out = np.zeros(n, bool)                          # khách bị đưa về a0 ở bước sửa (không được bù lại)
        m1 = marg_a(1)

        def totals():
            return float(S5.CAP[final_c].sum()), int((final_c == 3).sum()), int((final_c > 0).sum()), int((final_c == 2).sum())

        def run_backfill():
            rejected_today = outcome == "rejected"
            for seg, r_ in ((True, req[0]), (False, req[1])):
                face, calls, nv, na2 = totals()
                got = int(((final_c > 0) & (is_uk == seg)).sum())
                if got >= r_:
                    continue
                cand = np.where((is_uk == seg) & (final_c == 0) & ~ctrl & ~rejected_today & ~rep_out & eligible & ~m1 &
                                (V[:, 0] >= 0))[0]
                cand = cand[np.argsort(-V[cand, 0], kind="stable")]
                for i in cand:
                    if got >= r_:
                        break
                    trig = "H1_CI_tau_chua_0" if (lo[i, 0] <= 0 <= hi[i, 0]) else ("H2_S_khoang_rong" if wide[i] else "none")
                    if trig != "none":                                     # khách bù đi qua HITL như thường
                        hitl[i], first_soft[i] = True, trig
                        if not reviewer_decision(tau[i, 0], lo[i, 0], mu0[i], S[i], 1):
                            outcome[i], bstatus[i] = "rejected", "ung_vien_bu_bi_HITL_tu_choi"
                            continue
                        outcome[i] = "approved"
                    # chỉ kiểm ràng buộc mà VIỆC THÊM a1 có thể làm vượt: ngân sách mệnh giá (a1 không dùng cuộc gọi; thêm a1 chỉ
                    # làm GIẢM tỷ lệ a2).
                    if face + S5.CAP[1] > B + 1e-9:
                        bstatus[i] = "ung_vien_bu_vuot_nguon_luc"
                        if outcome[i] == "approved" and trig != "none":
                            outcome[i] = "rejected"                        # không gửi được ⇒ ghi là không duyệt gửi
                        continue
                    final_c[i], backfill[i], bstatus[i] = 1, True, "bu"
                    face += S5.CAP[1]; nv += 1; got += 1

        run_backfill()
        n_rep_iter = 0
        if self.repair:
            # ---- v2.1: bước sửa khả thi cuối – trần a2 ≤ 25 % số voucher phát (G5 chấm trên quyết định GỬI THỰC TẾ)
            for _ in range(n):                                              # dừng chắc chắn: mỗi vòng giảm đúng một a2
                _, _, nv, na2 = totals()
                if na2 <= S5.A2_SHARE * nv + 1e-9:
                    break
                cand2 = np.where(final_c == 2)[0]
                i = cand2[np.argmin(V[cand2, 1])]                           # a2 có V̂(a2) thấp nhất
                if V[i, 0] > 0:
                    final_c[i], repaired[i] = 1, 1
                else:
                    final_c[i], repaired[i], rep_out[i] = 0, 2, True
                n_rep_iter += 1
            if n_rep_iter:
                run_backfill()                                              # kiểm lại độ phủ (ngân sách/K chỉ giảm sau sửa)
        face, calls, nv, na2 = totals()
        # ngân sách / K: bước sửa và bù không thể làm tăng (chỉ hạ a2→a1/a0 hoặc thêm a1 trong ngân sách còn lại) – kiểm toán ở bảng audit
        if self.repair:
            assert na2 <= S5.A2_SHARE * nv + 1e-9, "bước sửa không đạt trần a2"
        self.last_repair = {"sua_a2_so_vong": n_rep_iter, "sua_a2_ha_a1": int((repaired == 1).sum()), "sua_a2_ve_a0": int((repaired == 2).sum()),
                            "bu_sau_sua": int(backfill.sum())}
        gui_uk, gui_non = int(((final_c > 0) & is_uk).sum()), int(((final_c > 0) & ~is_uk).sum())
        self.last_coverage = {"req_UK": req[0], "req_nonUK": req[1], "gui_UK": gui_uk, "gui_nonUK": gui_non,
                              "bu_UK": int((backfill & is_uk).sum()), "bu_nonUK": int((backfill & ~is_uk).sum())}
        p_ctrl = CONTROL_PCT / 100
        in_rand = sendable & ~backfill & ~rep_out
        prop_logged = np.where(in_rand, np.where(ctrl, p_ctrl, 1 - p_ctrl), 1.0)
        prop_a0 = np.where(in_rand, p_ctrl, np.where(final_c > 0, 0.0, 1.0))
        if not dry_run:
            for i in np.where(final_c > 0)[0]:
                self.history[int(cid[i])] = np.datetime64(t0, "ns")
        act = np.array(ACTIONS)
        rc = [df.get(f"reason_{k}_code", pd.Series([None] * n, index=df.index)).to_numpy() for k in (1, 2, 3)]
        rcode = np.where(backfill, BACKFILL_CODE, np.where(first_hard != "none", first_hard, np.where(
            ctrl, "CONTROL_HOLDOUT", np.where(hitl & has, "HITL_" + outcome.astype(str), np.where(has, "MILP_AUTO", "NO_ACTION")))))
        rcode = np.where(repaired == 1, REPAIR_CODES[1], np.where(repaired == 2, REPAIR_CODES[2], rcode))
        out = pd.DataFrame({
            "decision_id": [str(uuid.uuid5(NAMESPACE, k)) for k in keys], "idempotency_key": keys,
            "customer_id": cid.astype(int), "t0": t0, "split": split, "policy_version": self.policy_version,
            "model_version": self.model_version, "proposed_action": act[prop], "final_action": act[final_c],
            "channel": [CHANNEL[a] for a in act[final_c]], "guardrail_triggered": first_hard,
            "guardrails_all": ["+".join([g for g in HARD if conds[g][i]] + [g for g in SOFT if soft[g][i]]) if has[i] else ""
                               for i in range(n)],
            "hitl_flag": hitl, "hitl_trigger": first_soft, "hitl_outcome": outcome, "control_group": ctrl,
            "in_randomization_set": in_rand, "propensity_logged": prop_logged, "propensity_a0": prop_a0,
            "V_hat_proposed": np.where(has, V[idx, pj], np.where(backfill, V[:, 0], 0.0)),
            "tau_hat": np.where(has, tau[idx, pj], np.where(backfill, tau[:, 0], np.nan)),
            "tau_lo": np.where(has, lo[idx, pj], np.where(backfill, lo[:, 0], np.nan)),
            "tau_hi": np.where(has, hi[idx, pj], np.where(backfill, hi[:, 0], np.nan)),
            "s_hat_mean": S, "s_lo": df.s_lo.to_numpy(), "s_hi": df.s_hi.to_numpy(),
            "reason_1": rc[0], "reason_2": rc[1], "reason_3": rc[2], "decided_at": t0 + pd.Timedelta(hours=6),
            "is_backfill": backfill, "p_control": np.where(in_rand, p_ctrl, 0.0), "router_reason_code": rcode,
            "backfill_status": bstatus})
        if self.repair:
            out["repair_status"] = np.array(["none", "downgrade_a1", "drop_a0"], dtype=object)[repaired]
        out["guardrails_all"] = out.guardrails_all.replace("", "none")
        out = (DECISION_SCHEMA_V21 if self.repair else DECISION_SCHEMA_V2).validate(out, lazy=True)
        if not dry_run:
            for r in out.to_dict("records"):
                self.log[r["idempotency_key"]] = r
            self.frames.append(out)
        self.last_conds = {**conds, **soft}
        return out


def make_ins_v2(df: pd.DataFrame, meta: pd.DataFrame, split: str, t0, fl: dict) -> dict:
    """Miền MILP v2 = miền s5/s6 + chặn opt-out và frequency cap; mức tối thiểu theo AMENDMENT (1)."""
    ins = S6.make_ins(df, meta, split, t0)
    blocked = ins["blocked"] | fl["opt"] | fl["canc"] | fl["incap"]
    ins = dict(ins, blocked=blocked)
    ins["req_uk"], ins["req_non"] = coverage_requirement(ins["is_uk"], ~blocked)
    return ins


def scored_frame_v2(base: pd.DataFrame, sc: dict) -> pd.DataFrame:
    """Như scored_frame nhưng MANG THEO mã lý do top-3 (sửa lỗi v1: bản ghi v1 có mã lý do trống)."""
    d = scored_frame(base, sc)
    for k in (1, 2, 3):
        d[f"reason_{k}_code"] = base[f"reason_{k}_code"].to_numpy()
    return d


def audit_final(ins: dict, dec: pd.DataFrame, fl: dict) -> dict:
    """Kiểm ràng buộc trên BẢN GHI CUỐI (AMENDMENT (1)): tử số không gồm đối chứng; mức tối thiểu tính với 'đủ điều kiện' v2."""
    w = np.eye(4)[pd.Categorical(dec.final_action, categories=ACTIONS).codes]
    is_uk = ins["is_uk"]
    req = coverage_requirement(is_uk, ~(fl["opt"] | fl["canc"] | fl["incap"]))
    nv, na2 = w[:, 1:].sum(), w[:, 2].sum()
    sent = w[:, 1:].sum(1) > 0
    return {"so_voucher": int(nv), "so_a2": int(na2), "so_cuoc_goi": int(w[:, 3].sum()), "ngan_sach_dung": float((w @ S5.CAP).sum()),
            "B": ins["B"], "K": ins["K"], "req_UK": req[0], "req_nonUK": req[1],
            "gui_UK": int((sent & is_uk).sum()), "gui_nonUK": int((sent & ~is_uk).sum()),
            "vp_ngan_sach": bool((w @ S5.CAP).sum() > ins["B"] + 1e-6), "vp_cuoc_goi": bool(w[:, 3].sum() > ins["K"] + 1e-6),
            "vp_ty_le_a2": bool(na2 > S5.A2_SHARE * nv + 1e-9), "vp_do_phu_UK": bool((sent & is_uk).sum() < req[0]),
            "vp_do_phu_nonUK": bool((sent & ~is_uk).sum() < req[1]),
            "vp_gui_khach_bi_chan": bool((sent & (fl["opt"] | fl["canc"] | fl["incap"])).sum() > 0)}


def evaluate_v1_coverage_under_amendment(df: pd.DataFrame, meta: pd.DataFrame, per: list) -> pd.DataFrame:
    """Đo lại bản ghi cuối v1 theo định nghĩa độ phủ của AMENDMENT (1) (để so sánh công bằng v1 – v2)."""
    d1 = pd.read_parquet(common.p("data/processed/decisions_s7.parquet"))
    hist, rows = {}, []
    for split, t0 in per:
        ins = S6.make_ins(df, meta, split, t0)
        cid = df.customer_id.to_numpy()[ins["idx"]].astype(int)
        g = d1[d1.t0 == pd.Timestamp(t0)].set_index("customer_id").loc[cid]
        opt = np.array([hash_unit("optout", SALT, c) < OPT_OUT_PCT / 100 for c in cid])
        canc = df.cancel_rate_365.fillna(0).to_numpy()[ins["idx"]] > CANCEL_MAX
        last = np.array([hist.get(int(c), np.datetime64("NaT", "ns")) for c in cid], dtype="datetime64[ns]")
        incap = ~np.isnat(last) & ((np.datetime64(pd.Timestamp(t0), "ns") - last) / np.timedelta64(1, "D") < FREQ_CAP_DAYS)
        a = audit_final(ins, g.reset_index(), dict(opt=opt, canc=canc, incap=incap))
        rows.append({"phien_ban": "v1", "split": split, "t0": t0, **a})
        for c in g.index[g.final_action != "a0"]:
            hist[int(c)] = np.datetime64(pd.Timestamp(t0), "ns")
    return pd.DataFrame(rows)


def main_v2() -> None:
    np.random.seed(SEED)
    t_start = time.time()
    files: list[str] = []
    pd.set_option("display.width", 250)
    df_all = load_data()
    lg = pd.read_parquet(common.p("data/processed/logged.parquet"))
    lg = lg[lg.split.isin(["train", "valid", "calib"])].reset_index(drop=True)
    assert (lg.customer_id.to_numpy() == df_all.customer_id.to_numpy()).all()
    df = df_all.copy()
    for c in lg.columns:
        if c not in df.columns:
            df[c] = lg[c].to_numpy()
    rs = pd.read_parquet(common.p("data/processed/reasons_s3.parquet"))
    df = df.merge(rs[["customer_id", "t0", "reason_1_code", "reason_2_code", "reason_3_code"]], on=["customer_id", "t0"], how="left",
                  validate="one_to_one")
    p2 = pd.read_parquet(common.p("data/processed/pred_s2.parquet"), columns=["customer_id", "t0", "split", "s_lo", "s_hi"])
    trn = p2[p2.split == "train"]
    s_wide = float(np.percentile(trn.s_hi - trn.s_lo, 95))
    assert abs(s_wide - 1833.28) < 0.01                                   # AMENDMENT: giữ nguyên ngưỡng Ŝ (£1 833)
    meta = df[["split", "t0", S5.SEG_COL, "cancel_rate_365"]].copy()
    scorer = Scorer()
    legacy = legacy_scores(df)
    per = S6.periods(df)
    T = S6.evaluate_truth(df)
    print(f"v2 · s_wide = {s_wide:.1f} · {len(per)} kỳ ({time.time() - t_start:.0f}s)")

    rm, rl = ActionRouterV2(s_wide, scorer.version), ActionRouterV2(s_wide, scorer.version, POLICY_VERSION_V2 + "-legacy")
    snaps, val_rows, aud_rows, cond_rows, bf_rows, frames = {}, [], [], [], [], {"MILP": [], "Luật cũ": []}
    for split, t0 in per:
        ins0 = S6.make_ins(df, meta, split, t0)                           # miền v1 (cho MILP thuần / luật cũ thuần)
        idx = ins0["idx"]
        base = df.iloc[idx].reset_index(drop=True)
        X, Xc = scorer.features(base)
        sc = scorer.score(X, Xc)
        frame = rm.validate_input(scored_frame_v2(base, sc))
        Vs = sc["V"]
        orc = S6.evaluate_oracle_policy(T, ins0)
        w_pure = S6.solve_hs(ins0, Vs)["w"]
        w_leg_pure = S6.solve_hs(ins0, legacy[idx])["w"]
        results = {"MILP thuần (đề xuất, miền v1)": (w_pure, None), "Luật cũ thuần (miền v1)": (w_leg_pure, None)}
        for name, router, score in (("MILP", rm, Vs), ("Luật cũ", rl, legacy[idx])):
            snaps[(name, split, t0)] = dict(router.history)
            fl = router.domain_flags(frame)
            ins = make_ins_v2(df, meta, split, t0, fl)
            w_prop = S6.solve_hs(ins, score)["w"]
            assert not (w_prop[:, 1:].sum(1) > 0)[fl["opt"] | fl["incap"] | fl["canc"]].any()
            dec = router.route(frame, w_prop, split, ins=ins)
            w_fin = np.eye(4)[pd.Categorical(dec.final_action, categories=ACTIONS).codes]
            aud = audit_final(ins, dec, fl)
            aud_rows.append({"phien_ban": "v2", "chinh_sach": f"{name} qua Router v2", "split": split, "t0": t0, **aud})
            results[f"{name} v2: đề xuất (miền v2)"] = (w_prop, None)
            results[f"{name} qua Router v2"] = (w_fin, dec)
            for g, v in router.last_conds.items():
                vv = np.asarray(v) & (dec.proposed_action.to_numpy() != "a0")
                cond_rows.append({"chinh_sach": name, "split": split, "t0": t0, "guardrail": g, "n_dieu_kien_dung": int(vv.sum()),
                                  "n_de_xuat_voucher": int((dec.proposed_action != "a0").sum()),
                                  "n_kich_hoat_dau_tien": int((dec.guardrail_triggered == g).sum() + ((dec.hitl_trigger == g) &
                                                                                                     (dec.proposed_action != "a0")).sum())})
            vr, ve = evaluate_unit_values(T, idx, w_fin)
            bfm = dec.is_backfill.to_numpy()
            bf_rows.append({"chinh_sach": name, "split": split, "t0": t0, **router.last_coverage, "so_khach_bu": int(bfm.sum()),
                            "ung_vien_bi_HITL_tu_choi": int((dec.backfill_status == "ung_vien_bu_bi_HITL_tu_choi").sum()),
                            "ung_vien_vuot_nguon_luc": int((dec.backfill_status == "ung_vien_bu_vuot_nguon_luc").sum()),
                            "V_hat_khach_bu": float(dec.V_hat_proposed[bfm].sum()),
                            "gia_tri_that_hien_thuc_khach_bu": float(vr[bfm].sum()), "gia_tri_that_ky_vong_khach_bu": float(ve[bfm].sum()),
                            "het_danh_sach_van_thieu_UK": aud["vp_do_phu_UK"], "het_danh_sach_van_thieu_nonUK": aud["vp_do_phu_nonUK"]})
            vr_p, ve_p = evaluate_unit_values(T, idx, w_prop)
            frames[name].append(dec.assign(vr=vr_p, ve=ve_p, vr_cuoi=vr, ve_cuoi=ve))
        for nm, (w, _) in results.items():
            ev = evaluate_period(T, idx, w)
            val_rows.append({"split": split, "t0": t0, "chinh_sach": nm, "so_voucher": int(w[:, 1:].sum()),
                             "ngan_sach_dung": float((w @ S5.CAP).sum()), "V_hat_ke_hoach": float((w[:, 1:] * Vs).sum()), **ev, **orc,
                             "regret_cn_hien_thuc": (orc["z_sao_hien_thuc"] - ev["gia_tri_hien_thuc"]) / abs(orc["z_sao_hien_thuc"]),
                             "regret_cn_ky_vong": (orc["z_sao_ky_vong"] - ev["gia_tri_ky_vong"]) / abs(orc["z_sao_ky_vong"])})
        print(f"  {split} {pd.Timestamp(t0).date()}: MILP đề xuất {int(results['MILP v2: đề xuất (miền v2)'][0][:, 1:].sum())} · "
              f"gửi {int((results['MILP qua Router v2'][1].final_action != 'a0').sum())} · bù {bf_rows[-2]['so_khach_bu']} | "
              f"luật cũ gửi {int((results['Luật cũ qua Router v2'][1].final_action != 'a0').sum())} · bù {bf_rows[-1]['so_khach_bu']} "
              f"({time.time() - t_start:.0f}s)")

    # ---- bản ghi
    for name, router, fn in (("MILP", rm, "decisions_s7_v2.parquet"), ("Luật cũ", rl, "decisions_s7_v2_legacy.parquet")):
        lg_ = DECISION_SCHEMA_V2.validate(pd.concat(router.frames, ignore_index=True), lazy=True)
        lg_.to_parquet(common.p(f"data/processed/{fn}"), index=False)
        files.append(f"data/processed/{fn}")
    dlog = pd.read_parquet(common.p("data/processed/decisions_s7_v2.parquet"))

    vt = pd.DataFrame(val_rows)
    vt.to_csv(common.p(TABLES / "s7_policy_value_by_period_v2.csv"), index=False)
    au = pd.concat([pd.DataFrame(aud_rows), evaluate_v1_coverage_under_amendment(df, meta, per).assign(chinh_sach="MILP qua Router v1")],
                   ignore_index=True)
    au.to_csv(common.p(TABLES / "s7_constraint_audit_v2.csv"), index=False)
    bf = pd.DataFrame(bf_rows)
    bf.to_csv(common.p(TABLES / "s7_backfill_v2.csv"), index=False)
    ct = pd.DataFrame(cond_rows)
    ct.to_csv(common.p(TABLES / "s7_guardrail_rates_by_period_v2.csv"), index=False)
    files += [str(TABLES / f) for f in ("s7_policy_value_by_period_v2.csv", "s7_constraint_audit_v2.csv", "s7_backfill_v2.csv",
                                         "s7_guardrail_rates_by_period_v2.csv")]
    summ = vt.groupby("chinh_sach", sort=False).apply(lambda d: pd.Series({
        "n_ky": len(d), "so_voucher_TB": d.so_voucher.mean(), "ngan_sach_dung_TB": d.ngan_sach_dung.mean(),
        "V_hat_ke_hoach_TB": d.V_hat_ke_hoach.mean(), "thuc_hien_thuc_cong_don": d.gia_tri_hien_thuc.sum(),
        "thuc_ky_vong_cong_don": d.gia_tri_ky_vong.sum(),
        "regret_A_cong_don": (d.z_sao_hien_thuc.sum() - d.gia_tri_hien_thuc.sum()) / abs(d.z_sao_hien_thuc.sum()),
        "regret_B_cong_don": (d.z_sao_ky_vong.sum() - d.gia_tri_ky_vong.sum()) / abs(d.z_sao_ky_vong.sum())}), include_groups=False).reset_index()
    a2 = au.groupby("chinh_sach").agg(vp_ngan_sach=("vp_ngan_sach", "sum"), vp_cuoc_goi=("vp_cuoc_goi", "sum"),
                                      vp_ty_le_a2=("vp_ty_le_a2", "sum"), vp_do_phu_UK=("vp_do_phu_UK", "sum"),
                                      vp_do_phu_nonUK=("vp_do_phu_nonUK", "sum"), vp_gui_khach_bi_chan=("vp_gui_khach_bi_chan", "sum")).reset_index()
    summ = summ.merge(a2, on="chinh_sach", how="left")
    summ.to_csv(common.p(TABLES / "s7_policy_value_summary_v2.csv"), index=False)
    files.append(str(TABLES / "s7_policy_value_summary_v2.csv"))
    print(summ.round(3).T.to_string())

    # ---- luồng quyết định, chi phí guardrail, tỷ lệ chặn (MILP qua Router v2 và Luật cũ qua Router v2)
    flows, costs, rates = [], [], []
    for name in ("MILP", "Luật cũ"):
        d = pd.concat(frames[name], ignore_index=True)
        pv = d.proposed_action != "a0"
        flows.append({"chinh_sach": f"{name} qua Router v2", "de_xuat_voucher": int(pv.sum()),
                      "bi_chan_cung": int((d.guardrail_triggered != "none").sum()), "chuyen_HITL_de_xuat": int((d.hitl_flag & pv).sum()),
                      "HITL_duyet": int(((d.hitl_outcome == "approved") & pv).sum()), "HITL_ha_a1": int((d.hitl_outcome == "downgraded").sum()),
                      "HITL_tu_choi": int(((d.hitl_outcome == "rejected") & pv).sum()), "doi_chung": int(d.control_group.sum()),
                      "gui_tu_dong_khong_HITL": int((pv & (d.guardrail_triggered == "none") & ~d.hitl_flag & ~d.control_group).sum()),
                      "khach_bu": int(d.is_backfill.sum()), "ung_vien_bu_qua_HITL": int((~pv & d.hitl_flag).sum()),
                      "ung_vien_bu_bi_HITL_tu_choi": int((d.backfill_status == "ung_vien_bu_bi_HITL_tu_choi").sum()),
                      "gui_thuc_te": int((d.final_action != "a0").sum()), "so_khach_x_ky": len(d),
                      "so_khach_propensity_a0_bang_0": int((d.propensity_a0 == 0).sum()),
                      "so_khach_tap_ngau_nhien": int(d.in_randomization_set.sum())})
        grp = np.where(d.is_backfill, "Bù độ phủ (COVERAGE_BACKFILL)", np.where(
            d.guardrail_triggered != "none", d.guardrail_triggered, np.where(d.control_group, "Đối chứng 10%", np.where(
                d.hitl_flag & pv, "HITL:" + d.hitl_outcome.astype(str), np.where(pv, "tu_dong", "khong_de_xuat")))))
        for g in np.unique(grp):
            m_ = grp == g
            costs.append({"chinh_sach": f"{name} qua Router v2", "nhom": g, "n": int(m_.sum()),
                          "gia_tri_that_hien_thuc_de_xuat": float(d.vr[m_].sum()), "gia_tri_that_ky_vong_de_xuat": float(d.ve[m_].sum()),
                          "gia_tri_that_hien_thuc_cuoi": float(d.vr_cuoi[m_].sum()), "gia_tri_that_ky_vong_cuoi": float(d.ve_cuoi[m_].sum())})
        c_ = ct[ct.chinh_sach == name].groupby("guardrail")[["n_dieu_kien_dung", "n_de_xuat_voucher", "n_kich_hoat_dau_tien"]].sum()
        c_["ty_le_kich_hoat_dau_tien"] = c_.n_kich_hoat_dau_tien / c_.n_de_xuat_voucher
        rates.append(c_.reset_index().assign(chinh_sach=f"{name} qua Router v2"))
    flow = pd.DataFrame(flows)
    for c in ("bi_chan_cung", "chuyen_HITL_de_xuat", "doi_chung", "gui_thuc_te"):
        flow[f"ty_le_{c}_tren_de_xuat"] = flow[c] / flow.de_xuat_voucher
    flow.to_csv(common.p(TABLES / "s7_decision_flow_v2.csv"), index=False)
    cost = pd.DataFrame(costs)
    cost["anh_huong_hien_thuc_len_tong"] = cost.gia_tri_that_hien_thuc_cuoi - cost.gia_tri_that_hien_thuc_de_xuat
    cost["anh_huong_ky_vong_len_tong"] = cost.gia_tri_that_ky_vong_cuoi - cost.gia_tri_that_ky_vong_de_xuat
    cost.to_csv(common.p(TABLES / "s7_guardrail_cost_v2.csv"), index=False)
    rates = pd.concat(rates)
    rates.to_csv(common.p(TABLES / "s7_guardrail_rates_v2.csv"), index=False)
    files += [str(TABLES / f) for f in ("s7_decision_flow_v2.csv", "s7_guardrail_cost_v2.csv", "s7_guardrail_rates_v2.csv")]
    print(flow.T.to_string()); print(cost.round(0).to_string()); print(bf.to_string())

    # ---- bảng so sánh v1 – v2
    s1 = pd.read_csv(TABLES / "s7_policy_value_summary.csv").set_index("chinh_sach")
    s2 = summ.set_index("chinh_sach")
    cmp_rows = []
    for lab, k1, k2 in [("MILP thuần", "MILP thuần (đề xuất)", "MILP thuần (đề xuất, miền v1)"),
                        ("MILP qua Router", "Sau Router (gồm đối chứng)", "MILP qua Router v2"),
                        ("Luật cũ thuần", "Luật cũ (MILP trên điểm luật)", "Luật cũ thuần (miền v1)"),
                        ("Luật cũ qua Router", "Luật cũ sau Router", "Luật cũ qua Router v2")]:
        for ver, s_, k in (("v1", s1, k1), ("v2", s2, k2)):
            r = s_.loc[k]
            cmp_rows.append({"chinh_sach": lab, "phien_ban": ver, "thuc_hien_thuc_cong_don": r.thuc_hien_thuc_cong_don,
                             "thuc_ky_vong_cong_don": r.thuc_ky_vong_cong_don, "regret_A_cong_don": r.regret_A_cong_don,
                             "regret_B_cong_don": r.regret_B_cong_don, "so_voucher_TB": r.so_voucher_TB})
    cmp_ = pd.DataFrame(cmp_rows)
    cov = au.groupby("chinh_sach")[["vp_do_phu_UK", "vp_do_phu_nonUK", "vp_ngan_sach", "vp_cuoc_goi", "vp_ty_le_a2"]].sum()
    cmp_.to_csv(common.p(TABLES / "s7_v1_vs_v2.csv"), index=False)
    cov.reset_index().to_csv(common.p(TABLES / "s7_v1_vs_v2_constraints.csv"), index=False)
    files += [str(TABLES / "s7_v1_vs_v2.csv"), str(TABLES / "s7_v1_vs_v2_constraints.csv")]
    print(cmp_.round(3).to_string()); print(cov.to_string())

    # ---- chống lặp v2
    n_before = len(rm.log)
    s0, t00 = per[0]
    base0 = df.iloc[S6.make_ins(df, meta, s0, t00)["idx"]].reset_index(drop=True)
    fr0 = scored_frame_v2(base0, scorer.score(*scorer.features(base0)))
    r_snap = ActionRouterV2(s_wide, scorer.version)
    r_snap.history = dict(snaps[("MILP", s0, t00)])
    ins_a = make_ins_v2(df, meta, s0, t00, r_snap.domain_flags(fr0))
    again = rm.route(fr0, S6.solve_hs(ins_a, fr0[[f"V_hat_{a}" for a in TREATED]].to_numpy())["w"], s0, ins=ins_a)
    idem = pd.DataFrame([{"so_ban_ghi_truoc": n_before, "so_ban_ghi_sau": len(rm.log),
                          "chay_lai_tra_ban_ghi_cu": bool(again.decision_id.isin(dlog.decision_id).all()),
                          "khoa_chong_lap_duy_nhat": bool(dlog.idempotency_key.is_unique)}])
    idem.to_csv(common.p(TABLES / "s7_idempotency_v2.csv"), index=False)
    files.append(str(TABLES / "s7_idempotency_v2.csv"))

    # ---- độ trễ v2 (dùng ảnh chụp lịch sử trước mỗi kỳ, chạy khô)
    lat = []
    for split, t0 in per:
        ins0 = S6.make_ins(df, meta, split, t0)
        base = df.iloc[ins0["idx"]].reset_index(drop=True)
        for rep in range(N_LAT + 1):
            r = ActionRouterV2(s_wide, scorer.version)
            r.history = dict(snaps[("MILP", split, t0)])
            tt = time.perf_counter(); X, Xc = scorer.features(base); t_f = time.perf_counter() - tt
            tt = time.perf_counter(); sc = scorer.score(X, Xc); fr = scored_frame_v2(base, sc); t_s = time.perf_counter() - tt
            tt = time.perf_counter(); fr = r.validate_input(fr); fl = r.domain_flags(fr); ins = make_ins_v2(df, meta, split, t0, fl)
            t_c = time.perf_counter() - tt
            tt = time.perf_counter(); w = S6.solve_hs(ins, sc["V"])["w"]; t_m = time.perf_counter() - tt
            tt = time.perf_counter(); r.route(fr, w, split, ins=ins, dry_run=True); t_r = time.perf_counter() - tt
            if rep:
                lat.append({"t0": t0, "lan": rep, "t_dac_trung": t_f, "t_cham_diem": t_s, "t_hop_dong_va_mien": t_c, "t_milp": t_m,
                            "t_router": t_r, "tong": t_f + t_s + t_c + t_m + t_r})
    lt = pd.DataFrame(lat)
    pct = pd.DataFrame([{"khau": s, "P50": lt[s].quantile(.5), "P95": lt[s].quantile(.95), "P99": lt[s].quantile(.99), "max": lt[s].max()}
                        for s in ["t_dac_trung", "t_cham_diem", "t_hop_dong_va_mien", "t_milp", "t_router", "tong"]])
    pct.to_csv(common.p(TABLES / "s7_latency_percentiles_v2.csv"), index=False)
    files.append(str(TABLES / "s7_latency_percentiles_v2.csv"))
    print(pct.round(3).to_string())
    common.write_manifest("s7_v2", files, extra={"policy_version": POLICY_VERSION_V2, "model_version": scorer.version,
                                                 "s_wide_threshold": s_wide, "runtime_s": round(time.time() - t_start)},
                          packages=["pandas", "numpy", "pandera", "highspy", "scikit-learn", "lightgbm", "econml"])
    print(f"XONG GĐ7 v2 ({time.time() - t_start:.0f}s)")




def main_v21() -> None:
    np.random.seed(SEED)
    t_start = time.time()
    files: list[str] = []
    pd.set_option("display.width", 250)
    df_all = load_data()
    lg = pd.read_parquet(common.p("data/processed/logged.parquet"))
    lg = lg[lg.split.isin(["train", "valid", "calib"])].reset_index(drop=True)
    assert (lg.customer_id.to_numpy() == df_all.customer_id.to_numpy()).all()
    df = df_all.copy()
    for c in lg.columns:
        if c not in df.columns:
            df[c] = lg[c].to_numpy()
    rs = pd.read_parquet(common.p("data/processed/reasons_s3.parquet"))
    df = df.merge(rs[["customer_id", "t0", "reason_1_code", "reason_2_code", "reason_3_code"]], on=["customer_id", "t0"], how="left",
                  validate="one_to_one")
    p2 = pd.read_parquet(common.p("data/processed/pred_s2.parquet"), columns=["customer_id", "t0", "split", "s_lo", "s_hi"])
    trn = p2[p2.split == "train"]
    s_wide = float(np.percentile(trn.s_hi - trn.s_lo, 95))
    assert abs(s_wide - 1833.28) < 0.01                                   # AMENDMENT: giữ nguyên ngưỡng Ŝ (£1 833)
    meta = df[["split", "t0", S5.SEG_COL, "cancel_rate_365"]].copy()
    scorer = Scorer()
    legacy = legacy_scores(df)
    per = S6.periods(df)
    T = S6.evaluate_truth(df)
    print(f"v2.1 · s_wide = {s_wide:.1f} · {len(per)} kỳ ({time.time() - t_start:.0f}s)")

    rm, rl = ActionRouterV2(s_wide, scorer.version, POLICY_VERSION_V21), ActionRouterV2(s_wide, scorer.version, POLICY_VERSION_V21 + "-legacy")
    seg_rows = []
    snaps, val_rows, aud_rows, cond_rows, bf_rows, frames = {}, [], [], [], [], {"MILP": [], "Luật cũ": []}
    for split, t0 in per:
        ins0 = S6.make_ins(df, meta, split, t0)                           # miền v1 (cho MILP thuần / luật cũ thuần)
        idx = ins0["idx"]
        base = df.iloc[idx].reset_index(drop=True)
        X, Xc = scorer.features(base)
        sc = scorer.score(X, Xc)
        frame = rm.validate_input(scored_frame_v2(base, sc))
        Vs = sc["V"]
        orc = S6.evaluate_oracle_policy(T, ins0)
        w_pure = S6.solve_hs(ins0, Vs)["w"]
        w_leg_pure = S6.solve_hs(ins0, legacy[idx])["w"]
        results = {"MILP thuần (đề xuất, miền v1)": (w_pure, None), "Luật cũ thuần (miền v1)": (w_leg_pure, None)}
        for name, router, score in (("MILP", rm, Vs), ("Luật cũ", rl, legacy[idx])):
            snaps[(name, split, t0)] = dict(router.history)
            fl = router.domain_flags(frame)
            ins = make_ins_v2(df, meta, split, t0, fl)
            w_prop = S6.solve_hs(ins, score)["w"]
            assert not (w_prop[:, 1:].sum(1) > 0)[fl["opt"] | fl["incap"] | fl["canc"]].any()
            dec = router.route(frame, w_prop, split, ins=ins)
            w_fin = np.eye(4)[pd.Categorical(dec.final_action, categories=ACTIONS).codes]
            aud = audit_final(ins, dec, fl)
            aud_rows.append({"phien_ban": "v2.1", "chinh_sach": f"{name} qua Router v2.1", "split": split, "t0": t0, **aud})
            results[f"{name} v2: đề xuất (miền v2.1)"] = (w_prop, None)
            results[f"{name} qua Router v2.1"] = (w_fin, dec)
            for g, v in router.last_conds.items():
                vv = np.asarray(v) & (dec.proposed_action.to_numpy() != "a0")
                cond_rows.append({"chinh_sach": name, "split": split, "t0": t0, "guardrail": g, "n_dieu_kien_dung": int(vv.sum()),
                                  "n_de_xuat_voucher": int((dec.proposed_action != "a0").sum()),
                                  "n_kich_hoat_dau_tien": int((dec.guardrail_triggered == g).sum() + ((dec.hitl_trigger == g) &
                                                                                                     (dec.proposed_action != "a0")).sum())})
            vr, ve = evaluate_unit_values(T, idx, w_fin)
            bfm = dec.is_backfill.to_numpy()
            bf_rows.append({"chinh_sach": name, "split": split, "t0": t0, **router.last_coverage, "so_khach_bu": int(bfm.sum()),
                            "ung_vien_bi_HITL_tu_choi": int((dec.backfill_status == "ung_vien_bu_bi_HITL_tu_choi").sum()),
                            "ung_vien_vuot_nguon_luc": int((dec.backfill_status == "ung_vien_bu_vuot_nguon_luc").sum()),
                            "V_hat_khach_bu": float(dec.V_hat_proposed[bfm].sum()),
                            "gia_tri_that_hien_thuc_khach_bu": float(vr[bfm].sum()), "gia_tri_that_ky_vong_khach_bu": float(ve[bfm].sum()),
                            "het_danh_sach_van_thieu_UK": aud["vp_do_phu_UK"], "het_danh_sach_van_thieu_nonUK": aud["vp_do_phu_nonUK"], **router.last_repair})
            vr_p, ve_p = evaluate_unit_values(T, idx, w_prop)
            frames[name].append(dec.assign(vr=vr_p, ve=ve_p, vr_cuoi=vr, ve_cuoi=ve))
        for nm, (w, _) in results.items():
            vr_s, ve_s = evaluate_unit_values(T, idx, w)
            for seg_name, sm in (("UK", ins0["is_uk"]), ("non-UK", ~ins0["is_uk"])):
                seg_rows.append({"split": split, "t0": t0, "chinh_sach": nm, "phan_khuc": seg_name, "gia_tri_hien_thuc": float(vr_s[sm].sum()),
                                 "gia_tri_ky_vong": float(ve_s[sm].sum())})
            ev = evaluate_period(T, idx, w)
            val_rows.append({"split": split, "t0": t0, "chinh_sach": nm, "so_voucher": int(w[:, 1:].sum()),
                             "ngan_sach_dung": float((w @ S5.CAP).sum()), "V_hat_ke_hoach": float((w[:, 1:] * Vs).sum()), **ev, **orc,
                             "regret_cn_hien_thuc": (orc["z_sao_hien_thuc"] - ev["gia_tri_hien_thuc"]) / abs(orc["z_sao_hien_thuc"]),
                             "regret_cn_ky_vong": (orc["z_sao_ky_vong"] - ev["gia_tri_ky_vong"]) / abs(orc["z_sao_ky_vong"])})
        print(f"  {split} {pd.Timestamp(t0).date()}: MILP đề xuất {int(results['MILP v2: đề xuất (miền v2.1)'][0][:, 1:].sum())} · "
              f"gửi {int((results['MILP qua Router v2.1'][1].final_action != 'a0').sum())} · bù {bf_rows[-2]['so_khach_bu']} | "
              f"luật cũ gửi {int((results['Luật cũ qua Router v2.1'][1].final_action != 'a0').sum())} · bù {bf_rows[-1]['so_khach_bu']} "
              f"({time.time() - t_start:.0f}s)")

    # ---- bản ghi
    for name, router, fn in (("MILP", rm, "decisions_s7_v21.parquet"), ("Luật cũ", rl, "decisions_s7_v21_legacy.parquet")):
        lg_ = DECISION_SCHEMA_V21.validate(pd.concat(router.frames, ignore_index=True), lazy=True)
        lg_.to_parquet(common.p(f"data/processed/{fn}"), index=False)
        files.append(f"data/processed/{fn}")
    dlog = pd.read_parquet(common.p("data/processed/decisions_s7_v21.parquet"))

    vt = pd.DataFrame(val_rows)
    vt.to_csv(common.p(TABLES / "s7_policy_value_by_period_v21.csv"), index=False)
    au = pd.concat([pd.DataFrame(aud_rows), evaluate_v1_coverage_under_amendment(df, meta, per).assign(chinh_sach="MILP qua Router v1")],
                   ignore_index=True)
    au.to_csv(common.p(TABLES / "s7_constraint_audit_v21.csv"), index=False)
    bf = pd.DataFrame(bf_rows)
    bf.to_csv(common.p(TABLES / "s7_backfill_v21.csv"), index=False)
    ct = pd.DataFrame(cond_rows)
    ct.to_csv(common.p(TABLES / "s7_guardrail_rates_by_period_v21.csv"), index=False)
    files += [str(TABLES / f) for f in ("s7_policy_value_by_period_v21.csv", "s7_constraint_audit_v21.csv", "s7_backfill_v21.csv",
                                         "s7_guardrail_rates_by_period_v21.csv")]
    summ = vt.groupby("chinh_sach", sort=False).apply(lambda d: pd.Series({
        "n_ky": len(d), "so_voucher_TB": d.so_voucher.mean(), "ngan_sach_dung_TB": d.ngan_sach_dung.mean(),
        "V_hat_ke_hoach_TB": d.V_hat_ke_hoach.mean(), "thuc_hien_thuc_cong_don": d.gia_tri_hien_thuc.sum(),
        "thuc_ky_vong_cong_don": d.gia_tri_ky_vong.sum(),
        "regret_A_cong_don": (d.z_sao_hien_thuc.sum() - d.gia_tri_hien_thuc.sum()) / abs(d.z_sao_hien_thuc.sum()),
        "regret_B_cong_don": (d.z_sao_ky_vong.sum() - d.gia_tri_ky_vong.sum()) / abs(d.z_sao_ky_vong.sum())}), include_groups=False).reset_index()
    a2 = au.groupby("chinh_sach").agg(vp_ngan_sach=("vp_ngan_sach", "sum"), vp_cuoc_goi=("vp_cuoc_goi", "sum"),
                                      vp_ty_le_a2=("vp_ty_le_a2", "sum"), vp_do_phu_UK=("vp_do_phu_UK", "sum"),
                                      vp_do_phu_nonUK=("vp_do_phu_nonUK", "sum"), vp_gui_khach_bi_chan=("vp_gui_khach_bi_chan", "sum")).reset_index()
    summ = summ.merge(a2, on="chinh_sach", how="left")
    summ.to_csv(common.p(TABLES / "s7_policy_value_summary_v21.csv"), index=False)
    files.append(str(TABLES / "s7_policy_value_summary_v21.csv"))
    print(summ.round(3).T.to_string())

    # ---- luồng quyết định, chi phí guardrail, tỷ lệ chặn (MILP qua Router v2.1 và Luật cũ qua Router v2.1)
    flows, costs, rates = [], [], []
    for name in ("MILP", "Luật cũ"):
        d = pd.concat(frames[name], ignore_index=True)
        pv = d.proposed_action != "a0"
        flows.append({"chinh_sach": f"{name} qua Router v2.1", "de_xuat_voucher": int(pv.sum()),
                      "bi_chan_cung": int((d.guardrail_triggered != "none").sum()), "chuyen_HITL_de_xuat": int((d.hitl_flag & pv).sum()),
                      "HITL_duyet": int(((d.hitl_outcome == "approved") & pv).sum()), "HITL_ha_a1": int((d.hitl_outcome == "downgraded").sum()),
                      "HITL_tu_choi": int(((d.hitl_outcome == "rejected") & pv).sum()), "doi_chung": int(d.control_group.sum()),
                      "gui_tu_dong_khong_HITL": int((pv & (d.guardrail_triggered == "none") & ~d.hitl_flag & ~d.control_group).sum()),
                      "khach_bu": int(d.is_backfill.sum()), "ung_vien_bu_qua_HITL": int((~pv & d.hitl_flag).sum()),
                      "ung_vien_bu_bi_HITL_tu_choi": int((d.backfill_status == "ung_vien_bu_bi_HITL_tu_choi").sum()),
                      "gui_thuc_te": int((d.final_action != "a0").sum()), "so_khach_x_ky": len(d),
                      "so_khach_propensity_a0_bang_0": int((d.propensity_a0 == 0).sum()),
                      "so_khach_tap_ngau_nhien": int(d.in_randomization_set.sum())})
        grp = np.where(d.is_backfill, "Bù độ phủ (COVERAGE_BACKFILL)", np.where(
            d.guardrail_triggered != "none", d.guardrail_triggered, np.where(d.control_group, "Đối chứng 10%", np.where(
                d.hitl_flag & pv, "HITL:" + d.hitl_outcome.astype(str), np.where(pv, "tu_dong", "khong_de_xuat")))))
        for g in np.unique(grp):
            m_ = grp == g
            costs.append({"chinh_sach": f"{name} qua Router v2.1", "nhom": g, "n": int(m_.sum()),
                          "gia_tri_that_hien_thuc_de_xuat": float(d.vr[m_].sum()), "gia_tri_that_ky_vong_de_xuat": float(d.ve[m_].sum()),
                          "gia_tri_that_hien_thuc_cuoi": float(d.vr_cuoi[m_].sum()), "gia_tri_that_ky_vong_cuoi": float(d.ve_cuoi[m_].sum())})
        c_ = ct[ct.chinh_sach == name].groupby("guardrail")[["n_dieu_kien_dung", "n_de_xuat_voucher", "n_kich_hoat_dau_tien"]].sum()
        c_["ty_le_kich_hoat_dau_tien"] = c_.n_kich_hoat_dau_tien / c_.n_de_xuat_voucher
        rates.append(c_.reset_index().assign(chinh_sach=f"{name} qua Router v2.1"))
    flow = pd.DataFrame(flows)
    for c in ("bi_chan_cung", "chuyen_HITL_de_xuat", "doi_chung", "gui_thuc_te"):
        flow[f"ty_le_{c}_tren_de_xuat"] = flow[c] / flow.de_xuat_voucher
    flow.to_csv(common.p(TABLES / "s7_decision_flow_v21.csv"), index=False)
    cost = pd.DataFrame(costs)
    cost["anh_huong_hien_thuc_len_tong"] = cost.gia_tri_that_hien_thuc_cuoi - cost.gia_tri_that_hien_thuc_de_xuat
    cost["anh_huong_ky_vong_len_tong"] = cost.gia_tri_that_ky_vong_cuoi - cost.gia_tri_that_ky_vong_de_xuat
    cost.to_csv(common.p(TABLES / "s7_guardrail_cost_v21.csv"), index=False)
    rates = pd.concat(rates)
    rates.to_csv(common.p(TABLES / "s7_guardrail_rates_v21.csv"), index=False)
    files += [str(TABLES / f) for f in ("s7_decision_flow_v21.csv", "s7_guardrail_cost_v21.csv", "s7_guardrail_rates_v21.csv")]
    print(flow.T.to_string()); print(cost.round(0).to_string()); print(bf.to_string())

    # ---- bảng so sánh v1 – v2
    s1 = pd.read_csv(TABLES / "s7_policy_value_summary.csv").set_index("chinh_sach")
    s2 = summ.set_index("chinh_sach")
    cmp_rows = []
    for lab, k1, k2 in [("MILP thuần", "MILP thuần (đề xuất)", "MILP thuần (đề xuất, miền v1)"),
                        ("MILP qua Router", "Sau Router (gồm đối chứng)", "MILP qua Router v2.1"),
                        ("Luật cũ thuần", "Luật cũ (MILP trên điểm luật)", "Luật cũ thuần (miền v1)"),
                        ("Luật cũ qua Router", "Luật cũ sau Router", "Luật cũ qua Router v2.1")]:
        for ver, s_, k in (("v1", s1, k1), ("v2.1", s2, k2)):
            r = s_.loc[k]
            cmp_rows.append({"chinh_sach": lab, "phien_ban": ver, "thuc_hien_thuc_cong_don": r.thuc_hien_thuc_cong_don,
                             "thuc_ky_vong_cong_don": r.thuc_ky_vong_cong_don, "regret_A_cong_don": r.regret_A_cong_don,
                             "regret_B_cong_don": r.regret_B_cong_don, "so_voucher_TB": r.so_voucher_TB})
    cmp_ = pd.DataFrame(cmp_rows)
    cov = au.groupby("chinh_sach")[["vp_do_phu_UK", "vp_do_phu_nonUK", "vp_ngan_sach", "vp_cuoc_goi", "vp_ty_le_a2"]].sum()
    cmp_.to_csv(common.p(TABLES / "s7_v1_vs_v21.csv"), index=False)
    cov.reset_index().to_csv(common.p(TABLES / "s7_v1_vs_v21_constraints.csv"), index=False)
    files += [str(TABLES / "s7_v1_vs_v21.csv"), str(TABLES / "s7_v1_vs_v21_constraints.csv")]
    print(cmp_.round(3).to_string()); print(cov.to_string())


    # ---- v2 → v2.1: bước sửa kích hoạt, thay đổi giá trị
    rp = bf.groupby("chinh_sach")[["sua_a2_so_vong", "sua_a2_ha_a1", "sua_a2_ve_a0", "bu_sau_sua"]].sum()
    rp["so_ky_kich_hoat"] = bf.assign(k=bf.sua_a2_so_vong > 0).groupby("chinh_sach").k.sum()
    rp.reset_index().to_csv(common.p(TABLES / "s7_repair_v21.csv"), index=False)
    bf.to_csv(common.p(TABLES / "s7_repair_v21_by_period.csv"), index=False)
    files += [str(TABLES / "s7_repair_v21.csv"), str(TABLES / "s7_repair_v21_by_period.csv")]
    v2s = pd.read_csv(TABLES / "s7_policy_value_summary_v2.csv").set_index("chinh_sach")
    chg = []
    for nm in ("MILP qua Router", "Luật cũ qua Router"):
        a_, b_ = v2s.loc[f"{nm} v2"], s2.loc[f"{nm} v2.1"]
        chg.append({"luong": nm, "thuc_hien_thuc_v2": a_.thuc_hien_thuc_cong_don, "thuc_hien_thuc_v21": b_.thuc_hien_thuc_cong_don,
                    "thay_doi_hien_thuc": b_.thuc_hien_thuc_cong_don - a_.thuc_hien_thuc_cong_don,
                    "thuc_ky_vong_v2": a_.thuc_ky_vong_cong_don, "thuc_ky_vong_v21": b_.thuc_ky_vong_cong_don,
                    "thay_doi_ky_vong": b_.thuc_ky_vong_cong_don - a_.thuc_ky_vong_cong_don,
                    "so_voucher_TB_v2": a_.so_voucher_TB, "so_voucher_TB_v21": b_.so_voucher_TB,
                    "vi_pham_con_lai_v21": int(b_[["vp_ngan_sach", "vp_cuoc_goi", "vp_ty_le_a2", "vp_do_phu_UK", "vp_do_phu_nonUK",
                                                    "vp_gui_khach_bi_chan"]].sum()),
                    "vi_pham_ty_le_a2_v2": a_.vp_ty_le_a2 if "vp_ty_le_a2" in a_ else np.nan})
    pd.DataFrame(chg).to_csv(common.p(TABLES / "s7_v2_vs_v21.csv"), index=False)
    files.append(str(TABLES / "s7_v2_vs_v21.csv"))
    print(pd.DataFrame(chg).round(2).T.to_string()); print(rp.to_string())
    # ---- G4a và G7 (so sánh CHÍNH: MILP qua Router vs Luật cũ qua Router; PHỤ: vs luật cũ thuần) – giá trị hiện thực, cộng dồn
    sg = pd.DataFrame(seg_rows)
    cum = sg.groupby(["chinh_sach", "phan_khuc"])[["gia_tri_hien_thuc", "gia_tri_ky_vong"]].sum()
    tot = vt.groupby("chinh_sach")[["gia_tri_hien_thuc", "gia_tri_ky_vong"]].sum()
    gate = []
    for so_sanh, ref in (("CHÍNH: vs Luật cũ qua Router v2.1", "Luật cũ qua Router v2.1"), ("PHỤ: vs Luật cũ thuần", "Luật cũ thuần (miền v1)")):
        for col in ("gia_tri_hien_thuc", "gia_tri_ky_vong"):
            pol, leg = tot.loc["MILP qua Router v2.1", col], tot.loc[ref, col]
            row = {"so_sanh": so_sanh, "thuoc_do": col, "MILP_qua_Router": pol, "doi_chung": leg, "chenh_lech": pol - leg,
                   "G4a_nguong_0.10_abs_legacy": 0.10 * abs(leg), "G4a_dat": bool(pol - leg >= 0.10 * abs(leg))}
            for seg in ("UK", "non-UK"):
                p_, l_ = cum.loc[("MILP qua Router v2.1", seg), col], cum.loc[(ref, seg), col]
                row[f"G7_{seg}_chenh_tuong_doi"] = (p_ - l_) / abs(l_) if abs(l_) > 1e-9 else np.nan
                row[f"G7_{seg}_dat"] = bool((p_ - l_) / abs(l_) >= -0.05) if abs(l_) > 1e-9 else np.nan
            gate.append(row)
    pd.DataFrame(gate).to_csv(common.p(TABLES / "s7_g4a_g7_v21.csv"), index=False)
    sg.to_csv(common.p(TABLES / "s7_value_by_segment_v21.csv"), index=False)
    files += [str(TABLES / "s7_g4a_g7_v21.csv"), str(TABLES / "s7_value_by_segment_v21.csv")]
    print(pd.DataFrame(gate).round(3).T.to_string())

    # ---- chống lặp v2.1
    n_before = len(rm.log)
    s0, t00 = per[0]
    base0 = df.iloc[S6.make_ins(df, meta, s0, t00)["idx"]].reset_index(drop=True)
    fr0 = scored_frame_v2(base0, scorer.score(*scorer.features(base0)))
    r_snap = ActionRouterV2(s_wide, scorer.version, POLICY_VERSION_V21)
    r_snap.history = dict(snaps[("MILP", s0, t00)])
    ins_a = make_ins_v2(df, meta, s0, t00, r_snap.domain_flags(fr0))
    again = rm.route(fr0, S6.solve_hs(ins_a, fr0[[f"V_hat_{a}" for a in TREATED]].to_numpy())["w"], s0, ins=ins_a)
    idem = pd.DataFrame([{"so_ban_ghi_truoc": n_before, "so_ban_ghi_sau": len(rm.log),
                          "chay_lai_tra_ban_ghi_cu": bool(again.decision_id.isin(dlog.decision_id).all()),
                          "khoa_chong_lap_duy_nhat": bool(dlog.idempotency_key.is_unique)}])
    idem.to_csv(common.p(TABLES / "s7_idempotency_v21.csv"), index=False)
    files.append(str(TABLES / "s7_idempotency_v21.csv"))

    # ---- độ trễ v2 (dùng ảnh chụp lịch sử trước mỗi kỳ, chạy khô)
    lat = []
    for split, t0 in per:
        ins0 = S6.make_ins(df, meta, split, t0)
        base = df.iloc[ins0["idx"]].reset_index(drop=True)
        for rep in range(N_LAT + 1):
            r = ActionRouterV2(s_wide, scorer.version, POLICY_VERSION_V21)
            r.history = dict(snaps[("MILP", split, t0)])
            tt = time.perf_counter(); X, Xc = scorer.features(base); t_f = time.perf_counter() - tt
            tt = time.perf_counter(); sc = scorer.score(X, Xc); fr = scored_frame_v2(base, sc); t_s = time.perf_counter() - tt
            tt = time.perf_counter(); fr = r.validate_input(fr); fl = r.domain_flags(fr); ins = make_ins_v2(df, meta, split, t0, fl)
            t_c = time.perf_counter() - tt
            tt = time.perf_counter(); w = S6.solve_hs(ins, sc["V"])["w"]; t_m = time.perf_counter() - tt
            tt = time.perf_counter(); r.route(fr, w, split, ins=ins, dry_run=True); t_r = time.perf_counter() - tt
            if rep:
                lat.append({"t0": t0, "lan": rep, "t_dac_trung": t_f, "t_cham_diem": t_s, "t_hop_dong_va_mien": t_c, "t_milp": t_m,
                            "t_router": t_r, "tong": t_f + t_s + t_c + t_m + t_r})
    lt = pd.DataFrame(lat)
    pct = pd.DataFrame([{"khau": s, "P50": lt[s].quantile(.5), "P95": lt[s].quantile(.95), "P99": lt[s].quantile(.99), "max": lt[s].max()}
                        for s in ["t_dac_trung", "t_cham_diem", "t_hop_dong_va_mien", "t_milp", "t_router", "tong"]])
    pct.to_csv(common.p(TABLES / "s7_latency_percentiles_v21.csv"), index=False)
    files.append(str(TABLES / "s7_latency_percentiles_v21.csv"))
    print(pct.round(3).to_string())
    common.write_manifest("s7_v21", files, extra={"policy_version": POLICY_VERSION_V21, "model_version": scorer.version,
                                                 "s_wide_threshold": s_wide, "runtime_s": round(time.time() - t_start)},
                          packages=["pandas", "numpy", "pandera", "highspy", "scikit-learn", "lightgbm", "econml"])
    print(f"XONG GĐ7 v2.1 ({time.time() - t_start:.0f}s)")



if __name__ == "__main__":
    if "--v21" in sys.argv:                      # v2.1 (bước sửa trần a2): python src/s7_router.py --v21
        from s7_router import main_v21 as _main
    elif "--v2" in sys.argv:                     # AMENDMENT v2: python src/s7_router.py --v2
        from s7_router import main_v2 as _main
    else:                                        # v1 (giữ nguyên làm hồ sơ)
        from s7_router import main as _main
    _main()
