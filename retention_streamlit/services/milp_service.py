"""MILP, Greedy, LP relaxation và đánh giá – dùng NGUYÊN VĂN mã gốc trong `src/`.

Không viết lại thuật toán: các hàm/lớp dưới đây được đọc trực tiếp từ `src/s5_spo.py`, `src/s6_milp.py`
và `src/s5b_dr_diagnostics.py` (cắt đúng đoạn mã nguồn bằng `ast`), rồi chạy trong một namespace chỉ
chứa numpy, scipy, highspy và các hằng số đọc từ `configs/config.yaml` – giống hệt cách module gốc khởi tạo.
Nhờ vậy app không phải cài EconML, DoubleML, LightGBM… (các module gốc import chúng ở đầu file),
nhưng kết quả vẫn là kết quả của chính mã nguồn đã đóng băng SHA-256.
"""
from __future__ import annotations

import ast
import hashlib
import math
import time
import types

import highspy
import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st

from services import data_service as D

PICK = {
    "s5_spo.py": ["build_instance", "check_feasible", "MILP", "cmat", "greedy"],
    "s6_milp.py": ["value_hat_m", "solve_hs", "lp_relaxation", "evaluate_utilities", "evaluate_policy",
                   "evaluate_oracle_policy"],
    "s5b_dr_diagnostics.py": ["legacy_scores"],
}


def _extract(fname: str, names: list[str]) -> tuple[str, str]:
    src = (D.SRC / fname).read_text(encoding="utf-8")
    tree = ast.parse(src)
    parts = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            parts[node.name] = ast.get_source_segment(src, node)
    missing = set(names) - set(parts)
    if missing:
        raise RuntimeError(f"Không tìm thấy {missing} trong src/{fname}")
    return "\n\n\n".join(parts[n] for n in names), hashlib.sha256(src.encode("utf-8")).hexdigest()


@st.cache_resource(show_spinner=False)
def engine() -> types.SimpleNamespace:
    """Namespace chứa các hàm gốc, đã gắn hằng số đúng như phần đầu của s5_spo.py / s6_milp.py."""
    cfg = D.config()
    biz = cfg["business"]
    acts = ["a0", "a1", "a2", "a3"]
    ns: dict = {"np": np, "pd": pd, "sp": sp, "math": math, "time": time, "highspy": highspy}
    ns.update(
        CFG=cfg, BIZ=biz, ACTIONS=acts, TREATED=acts[1:],
        CAP=np.array([biz["actions"][a]["face_value_cap"] for a in acts], float),
        DISC=np.array([biz["actions"][a]["discount"] for a in acts], float),
        CONTACT=np.array([biz["actions"][a]["contact_cost"] for a in acts], float),
        MARGIN=biz["gross_margin_rate"], M0=biz["gross_margin_rate"], COV=biz["min_coverage_per_segment"],
        SEG_COL=biz["coverage_segment_col"], GR_CANCEL=biz["guardrail_cancel_rate_max"],
        A2_SHARE=biz["max_share_deep_voucher"], GAP=1e-3, TIME_LIMIT=60.0, HI=highspy.HighsModelStatus,
    )
    origin = {}
    s5 = types.SimpleNamespace()
    ns["S5"] = s5
    for fname, names in PICK.items():
        code, digest = _extract(fname, names)
        exec(compile(code, f"src/{fname}", "exec"), ns)
        origin[fname] = {"ham": names, "sha256_file": digest}
    for k in ["MILP", "cmat", "check_feasible", "greedy", "build_instance", "SEG_COL", "CAP"]:
        setattr(s5, k, ns[k])
    out = types.SimpleNamespace(**{k: ns[k] for k in ns if not k.startswith("__")})
    out.ns, out.origin = ns, origin
    return out


# ---------------------------------------------------------------------------------------------- wrapper
def _set_a2_share(E, share: float) -> None:
    E.ns["A2_SHARE"] = share                    # greedy() và check_feasible() đọc biến toàn cục này


def _apply_a2_share(m, share: float) -> None:
    """MILP gốc mã hóa trần a2 = 25 % thành 3·x2 − x1 − x3 ≤ 0. Với trần s khác: (1−s)·x2 − s·x1 − s·x3 ≤ 0
    (chỉ đổi hệ số của đúng hàng ràng buộc đó trên mô hình gốc đã dựng)."""
    if abs(share - 0.25) < 1e-12:
        return
    row = m.n + 2
    for i in range(m.n):
        for a in (1, 2, 3):
            v = (1 - share) if a == 2 else -share
            m.h.changeCoeff(row, 4 * i + a, float(v))


def build(E, sub: pd.DataFrame, B: float, K: float):
    meta = sub[["split", "t0", "is_uk", "cancel_rate_365"]].reset_index(drop=True)
    return E.build_instance(np.arange(len(sub)), meta, len(sub), str(sub.split.iloc[0]), sub.t0.iloc[0], B=B, K=K)


def solve_milp(E, ins: dict, c: np.ndarray, share: float, limit: float = 30.0) -> dict:
    t = time.perf_counter()
    m = E.MILP(ins)
    _apply_a2_share(m, share)
    t_build = time.perf_counter() - t
    m.h.setOptionValue("mip_rel_gap", 1e-4)
    m.h.setOptionValue("time_limit", limit)
    cc = E.cmat(np.asarray(c, float))
    m.h.changeColsCost(m.N, m.cols, -cc.ravel())
    t = time.perf_counter()
    m.h.run()
    t_solve = time.perf_counter() - t
    st_ = m.h.getModelStatus()
    info = m.h.getInfo()
    ok = info.primal_solution_status == 2
    w = np.rint(np.array(m.h.getSolution().col_value)).reshape(ins["n"], 4) if ok else None
    return dict(w=w, status=m.h.modelStatusToString(st_), feasible=ok, gap=float(info.mip_gap) if ok else np.nan,
                obj=float((cc * w).sum()) if ok else np.nan, t_build=t_build, t_solve=t_solve)


def lp_duals(E, ins: dict, c: np.ndarray, share: float) -> dict:
    """Giá bóng từ LP RELAXATION (bỏ tính nguyên) – không phải đối ngẫu của MILP."""
    m = E.MILP(ins)
    _apply_a2_share(m, share)
    m.h.changeColsIntegrality(m.N, m.cols, np.array([highspy.HighsVarType.kContinuous] * m.N))
    m.h.changeColsCost(m.N, m.cols, -E.cmat(np.asarray(c, float)).ravel())
    m.h.run()
    n = ins["n"]
    if m.h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return dict(lp=np.nan, pi_B=np.nan, pi_K=np.nan)
    d = np.array(m.h.getSolution().row_dual)
    return dict(lp=-float(m.h.getInfo().objective_function_value), pi_B=-d[n], pi_K=-d[n + 1], pi_a2=-d[n + 2])


def truth(sub: pd.DataFrame) -> dict:
    """Kết quả tiềm năng thật (đáp án bán tổng hợp) – giống `s6_milp.evaluate_truth`, chỉ để đánh giá."""
    orc = D.oracle_frame()
    t = sub[["customer_id", "t0"]].merge(orc.drop(columns=["split"]), on=["customer_id", "t0"], how="left",
                                         validate="one_to_one")
    acts = ["a0", "a1", "a2", "a3"]
    return dict(S=t.basket_value_S.to_numpy(), y0=t.y_a0.to_numpy().astype(float),
                Y=np.column_stack([t[f"y_{a}"] for a in acts]).astype(float),
                tau=np.column_stack([t[f"tau_cond_{a}"] for a in acts[1:]]),
                dc_real=np.column_stack([t[f"discount_cost_{a}"] for a in acts]),
                profit=np.column_stack([t[f"profit_{a}"] for a in acts]))


def value_matrix(E, sub: pd.DataFrame, m: float, calibrated: bool = True) -> np.ndarray:
    tau = sub[["tau_hat_a1", "tau_hat_a2", "tau_hat_a3"]].to_numpy()
    mu0 = (sub.mu0_base if calibrated else sub.mu0_raw).to_numpy()
    return E.value_hat_m(tau, mu0, sub.s_hat_mean.to_numpy(), m)


def legacy_matrix(E, sub: pd.DataFrame) -> np.ndarray:
    return E.legacy_scores(sub.reset_index(drop=True))
