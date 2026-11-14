"""Bảng trạng thái cổng phát hành từ `s8_locked_gates.csv`."""
from __future__ import annotations

import html
import re

import pandas as pd

from components.kpi_cards import status_badge

POP = {"toan_bo": "Toàn bộ khách hàng", "khach_thuong": "Khách thường (bỏ entity holdout)",
       "entity_holdout": "Entity holdout (khách mô hình chưa từng thấy)"}
GROUP = {"G1": "Dữ liệu", "G2": "Hiệu chỉnh", "G3": "Uplift", "G4a": "Giá trị", "G4b": "Giá trị",
         "G5": "Ràng buộc", "G6": "Độ trễ", "G7": "Công bằng"}
SHORT = {"G1": "Mọi kiểm tra hợp đồng dữ liệu đạt", "G2": "ECE ≤ 0,03 và Brier không tệ hơn điểm thô",
         "G3": "Qini có CI 95 % > 0 cho ≥ 2 hành động", "G4a": "Lợi nhuận ≥ luật cũ + 10 % (so sánh chính)",
         "G4a-phụ": "Lợi nhuận ≥ luật cũ thuần + 10 % (phụ)", "G4b": "Regret chuẩn hóa A ≤ 0,35",
         "G4b-phụ": "Regret A – MILP thuần (phụ)", "G5": "0 vi phạm ràng buộc trên quyết định gửi",
         "G6": "P95 độ trễ đầu–cuối ≤ 30 s", "G7": "Mỗi phân khúc ≥ −5 % so với luật cũ", "G7-phụ": "Phân khúc so với luật cũ thuần (phụ)"}


def vnum(text: str) -> str:
    """Đổi số trong chuỗi gốc sang kiểu Việt: 8,785 → 8 785; 0.894 → 0,894."""
    text = re.sub(r"(\d),(\d{3})", r"\1 \2", str(text))
    return re.sub(r"(\d)\.(\d)", r"\1,\2", text)


REASON = {
    "G1": "Hợp đồng dữ liệu và kiểm thử rò rỉ đều đạt.",
    "G2": "ECE dưới ngưỡng, nhưng ngưỡng 0,03 nằm trong CI percentile (bằng chứng mỏng).",
    "G3": "Đạt 2/3 hành động – trái dự báo đăng ký trước (4–7 %); cận dưới a3 sát 0, hiệu chỉnh Bonferroni thì trượt.",
    "G4a": "Chênh £961 vượt ngưỡng £782 theo ước lượng điểm; CI bootstrap [−£2 273; +£3 980] chứa 0.",
    "G4b": "Mốc so sánh là chính sách biết trước kết quả hiện thực – không thể đạt: chính sách biết kỳ vọng thật cũng 0,742.",
    "G5": "Bước sửa khả thi cuối của Router đảm bảo 0 vi phạm.",
    "G6": "Đã gồm khâu mã lý do SHAP; xa dưới ngưỡng 30 s.",
    "G7": "Không phân khúc nào kém luật cũ quá 5 %.",
}


def gate_rows(g: pd.DataFrame, pop: str, include_secondary: bool) -> list[list[str]]:
    d = g[g.quan_the == pop]
    if not include_secondary:
        d = d[~d.gate.str.contains("phụ")]
    rows = []
    for r in d.itertuples():
        base = r.gate.replace("-phụ", "")
        official = bool(r.cham_chinh_thuc)
        tag = "" if official else ' <span class="rds-badge grey">mô tả</span>'
        reason = REASON.get(base, "") if (pop == "toan_bo" and "phụ" not in r.gate) else ""
        rows.append([f"{html.escape(r.gate)}", GROUP.get(base, ""), SHORT.get(r.gate, html.escape(str(r.chi_so))),
                     html.escape(vnum(r.gia_tri)), html.escape(vnum(r.nguong)), status_badge(bool(r.dat)) + tag,
                     reason])
    return rows
