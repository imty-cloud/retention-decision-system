"""Đọc nhật ký quyết định do `src/s7_router.py` (Router v2.1) và `src/s8b_oot.py` sinh ra; dựng luồng Sankey
và giải thích quyết định của từng khách. Không chạy lại Router – nhật ký chính là đầu ra của module gốc."""
from __future__ import annotations

import numpy as np
import pandas as pd

from services import data_service as D

ACTIONS = ["a0", "a1", "a2", "a3"]


def flow_counts(dec: pd.DataFrame) -> dict:
    """Đếm theo đúng trình tự xử lý ghi trong nhật ký Router v2.1: guardrail là miền khả thi trước MILP;
    nhóm đối chứng (băm khách × kỳ) chỉ có hiệu lực trên khách đã được đề xuất và qua HITL."""
    n = len(dec)
    blocked = int((dec.guardrail_triggered != "none").sum())
    proposed = int((dec.proposed_action != "a0").sum())
    hitl = int(dec.hitl_flag.sum())
    appr = int((dec.hitl_outcome == "approved").sum())
    rej = int((dec.hitl_outcome == "rejected").sum())
    nohitl = int(((dec.proposed_action != "a0") & ~dec.hitl_flag).sum())
    control = int(dec.control_group.sum())
    backfill = int(dec.is_backfill.sum())
    sent = int((dec.final_action != "a0").sum())
    pass_control = nohitl + appr - control
    dropped = max(pass_control + backfill - sent, 0)
    return dict(n=n, blocked=blocked, milp_in=n - blocked, proposed=proposed, milp_a0=n - blocked - proposed, hitl=hitl,
                approved=appr, rejected=rej, nohitl=nohitl, control=control, pass_control=pass_control,
                backfill=backfill, dropped=dropped, sent=sent, repaired=int((dec.repair_status != "none").sum()))


def explain_customer(row: pd.Series, V: np.ndarray, duals: dict, m: float, cap, disc, contact) -> dict:
    """Phân rã giá trị kinh tế từng hành động và lý do hành động cuối khác hành động có V cao nhất.

    Lý do là SUY LUẬN từ nhật ký Router + giá bóng LP relaxation của kỳ (không phải log nội bộ của solver)."""
    S, mu0 = float(row.s_hat_mean), float(row.mu0_base)
    parts = []
    for j, a in enumerate(ACTIONS[1:], start=1):
        tau = float(row[f"tau_hat_{a}"])
        gain = m * S * tau
        dcost = min(disc[j] * S, cap[j]) * min(max(mu0 + tau, 0), 1)
        parts.append(dict(hanh_dong=a, loi_nhuan_gia_tang=gain, chi_phi_voucher=dcost, chi_phi_lien_he=float(contact[j]),
                          V=float(V[j - 1]), tau=tau, tau_lo=float(row[f"tau_lo_{a}"]), tau_hi=float(row[f"tau_hi_{a}"]),
                          menh_gia=float(cap[j]),
                          V_tru_gia_bong=float(V[j - 1] - duals.get("pi_B", 0) * cap[j] - (duals.get("pi_K", 0) if j == 3 else 0))))
    best = max(parts, key=lambda p: p["V"])
    final, prop = row.final_action, row.proposed_action
    reasons = []
    if best["V"] <= 0:
        reasons.append("Mọi hành động đều có V̂ ≤ 0: gửi ưu đãi sẽ lỗ kỳ vọng, nên không làm gì (a0) là tối ưu.")
    elif prop == "a0":
        if best["V_tru_gia_bong"] <= 0:
            reasons.append(f"{best['hanh_dong']} có V̂ dương ({best['V']:.2f} £) nhưng không vượt được giá bóng ngân sách "
                           f"(£{duals.get('pi_B', 0):.3f} mỗi £1 mệnh giá × £{best['menh_gia']:.0f})"
                           + (" và giá bóng cuộc gọi" if best["hanh_dong"] == "a3" else "")
                           + ": ngân sách được ưu tiên cho khách hiệu quả hơn.")
        else:
            reasons.append("V̂ dương nhưng MILP không chọn do các ràng buộc tổ hợp (trần a2 25 %, độ phủ phân khúc) "
                           "ở mức nghiệm nguyên.")
    elif prop != best["hanh_dong"]:
        reasons.append(f"MILP đề xuất {prop} thay vì {best['hanh_dong']} (V̂ cao nhất): hành động có V̂ cao nhất tốn nhiều "
                       "ngân sách/cuộc gọi hơn so với phần giá trị tăng thêm, hoặc vướng trần a2.")
    if bool(row.control_group):
        reasons.append("Khách rơi vào nhóm đối chứng ngẫu nhiên 10 % (băm theo khách × kỳ) nên bị ép về a0 để đo giá trị thực hiện.")
    if row.hitl_outcome == "rejected":
        reasons.append(f"Chuyển người duyệt ({row.hitl_trigger}) và bị TỪ CHỐI: V̂ tính tại nửa đường tới cận dưới CI của τ̂ âm.")
    elif row.hitl_outcome == "approved":
        reasons.append(f"Chuyển người duyệt ({row.hitl_trigger}) và được DUYỆT.")
    if final == prop and not reasons:
        reasons.append("MILP đề xuất và Router gửi tự động, không chạm guardrail hay HITL.")
    return dict(parts=pd.DataFrame(parts), best=best["hanh_dong"], reasons=reasons)
