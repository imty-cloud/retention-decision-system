# Phụ lục VIII – Hợp đồng dữ liệu (trích tự động từ schema pandera)

> Sinh bởi `tools/build_appendix_extras.py` từ các đối tượng schema có sẵn trong `src/`; `mo_ta` lấy từ `outputs/tables/s1_data_dictionary.csv` (chỉ có cho cột của snapshot/logged; cột của Router để trống). Kiểm tra dạng hàm (lambda) hiển thị bằng thông điệp lỗi khai báo trong mã.


## Hợp đồng snapshot (GĐ1: đặc trưng point-in-time + nhãn)

*Nguồn:* `src/s1_data.py:snapshot_schema` · 18 cột, 8 ràng buộc cấp bảng

| cot | kieu | nullable | unique | rang_buoc | mo_ta |
|---|---|---|---|---|---|
| customer_id | string[pyarrow] | False | False |  | Mã khách hàng (khóa kỹ thuật, không dùng làm đặc trưng) |
| t0 | datetime64[ns] | False | False |  | Thời điểm quyết định (thứ Hai, 2 tuần/lần) |
| recency_days | float64 | False | False | greater_than_or_equal_to(min_value=1) | Số ngày từ đơn gần nhất đến t0 |
| tenure_days | float64 | False | False | greater_than_or_equal_to(min_value=1) | Số ngày từ đơn đầu tiên đến t0 |
| n_orders_all | int64 | False | False | greater_than_or_equal_to(min_value=1) | Số đơn hợp lệ trong toàn bộ lịch sử trước cutoff |
| cancel_rate_365 | float64 | False | False | in_range(min_value=0, max_value=1, include_min=True, include_max=True) | Tỷ lệ giá trị hủy / (bán + hủy) trong 365 ngày |
| y | int8 | False | False | isin(allowed_values=[0, 1]) | 1 nếu có >= 1 đơn hợp lệ trong [t0, t0 + H) |
| y_value_h | float64 | False | False | greater_than_or_equal_to(min_value=0) | Tổng giá trị đơn trong cửa sổ nhãn (dùng cho AOV/giá trị ở GĐ4–6) |
| is_entity_holdout | int8 | False | False | isin(allowed_values=[0, 1]) | 1 nếu mã khách thuộc nhóm giữ riêng |
| split | string[pyarrow] | False | False | isin(allowed_values=['train', 'valid', 'calib', 'locked', 'oot', 'embargo', 'excluded_entity_holdout']) | train/valid/calib/locked/oot/embargo/excluded_entity_holdout |
| n_orders_30 | int64 | False | False | greater_than_or_equal_to(min_value=0) | Số đơn trong 30 ngày trước t0 |
| n_orders_90 | int64 | False | False | greater_than_or_equal_to(min_value=0) | Số đơn trong 90 ngày trước t0 |
| n_orders_180 | int64 | False | False | greater_than_or_equal_to(min_value=0) | Số đơn trong 180 ngày trước t0 |
| n_orders_365 | int64 | False | False | greater_than_or_equal_to(min_value=0) | Số đơn trong 365 ngày trước t0 |
| value_30 | float64 | False | False | greater_than_or_equal_to(min_value=0) | Tổng giá trị đơn trong 30 ngày trước t0 |
| value_90 | float64 | False | False | greater_than_or_equal_to(min_value=0) | Tổng giá trị đơn trong 90 ngày trước t0 |
| value_180 | float64 | False | False | greater_than_or_equal_to(min_value=0) | Tổng giá trị đơn trong 180 ngày trước t0 |
| value_365 | float64 | False | False | greater_than_or_equal_to(min_value=0) | Tổng giá trị đơn trong 365 ngày trước t0 |
| (cấp bảng) |  |  |  | duy nhất theo (customer_id, t0) |  |
| (cấp bảng) |  |  |  | tenure>=recency |  |
| (cấp bảng) |  |  |  | window_monotone |  |
| (cấp bảng) |  |  |  | no_event_after_cutoff |  |
| (cấp bảng) |  |  |  | label_maturity |  |
| (cấp bảng) |  |  |  | y_consistent_with_value |  |
| (cấp bảng) |  |  |  | no_infinite_features |  |
| (cấp bảng) |  |  |  | strict=False; coerce=False | cột thừa: cho phép |

## Hợp đồng dữ liệu ghi log (GĐ1b/GĐ8: propensity, nhãn quan sát, cutoff, độ trễ nhãn)

*Nguồn:* `src/s8_mlops.py:LOGGED_SCHEMA` · 42 cột, 8 ràng buộc cấp bảng

| cot | kieu | nullable | unique | rang_buoc | mo_ta |
|---|---|---|---|---|---|
| customer_id | (không ràng buộc kiểu) | False | False |  | Mã khách hàng (khóa kỹ thuật, không dùng làm đặc trưng) |
| t0 | (không ràng buộc kiểu) | False | False |  | Thời điểm quyết định (thứ Hai, 2 tuần/lần) |
| A_logged | (không ràng buộc kiểu) | False | False | isin(allowed_values=['a0', 'a1', 'a2', 'a3']) |  |
| is_random_arm | (không ràng buộc kiểu) | False | False | isin(allowed_values=[0, 1]) |  |
| propensity_a0 | float64 | False | False | in_range(min_value=0, max_value=1, include_min=False, include_max=True) |  |
| propensity_a1 | float64 | False | False | in_range(min_value=0, max_value=1, include_min=False, include_max=True) |  |
| propensity_a2 | float64 | False | False | in_range(min_value=0, max_value=1, include_min=False, include_max=True) |  |
| propensity_a3 | float64 | False | False | in_range(min_value=0, max_value=1, include_min=False, include_max=True) |  |
| propensity_logged | float64 | False | False | in_range(min_value=0, max_value=1, include_min=False, include_max=True) |  |
| y_obs | (không ràng buộc kiểu) | False | False | isin(allowed_values=[0, 1]) |  |
| value_obs | float64 | False | False | greater_than_or_equal_to(min_value=0) |  |
| profit_obs | float64 | False | False |  |  |
| recency_days | (không ràng buộc kiểu) | True | False |  | Số ngày từ đơn gần nhất đến t0 |
| tenure_days | (không ràng buộc kiểu) | True | False |  | Số ngày từ đơn đầu tiên đến t0 |
| n_orders_all | (không ràng buộc kiểu) | True | False |  | Số đơn hợp lệ trong toàn bộ lịch sử trước cutoff |
| value_all | (không ràng buộc kiểu) | True | False |  | Tổng giá trị đơn (GBP) toàn lịch sử |
| n_orders_30 | (không ràng buộc kiểu) | True | False |  | Số đơn trong 30 ngày trước t0 |
| n_orders_90 | (không ràng buộc kiểu) | True | False |  | Số đơn trong 90 ngày trước t0 |
| n_orders_180 | (không ràng buộc kiểu) | True | False |  | Số đơn trong 180 ngày trước t0 |
| n_orders_365 | (không ràng buộc kiểu) | True | False |  | Số đơn trong 365 ngày trước t0 |
| value_30 | (không ràng buộc kiểu) | True | False |  | Tổng giá trị đơn trong 30 ngày trước t0 |
| value_90 | (không ràng buộc kiểu) | True | False |  | Tổng giá trị đơn trong 90 ngày trước t0 |
| value_180 | (không ràng buộc kiểu) | True | False |  | Tổng giá trị đơn trong 180 ngày trước t0 |
| value_365 | (không ràng buộc kiểu) | True | False |  | Tổng giá trị đơn trong 365 ngày trước t0 |
| aov_365 | (không ràng buộc kiểu) | True | False |  | Giá trị đơn trung bình 365 ngày |
| aov_all | (không ràng buộc kiểu) | True | False |  | Giá trị đơn trung bình toàn lịch sử |
| units_per_order_365 | (không ràng buộc kiểu) | True | False |  | Số đơn vị hàng / đơn (đại diện khách sỉ) |
| lines_per_order_365 | (không ràng buộc kiểu) | True | False |  | Số dòng hàng / đơn |
| distinct_products_365 | (không ràng buộc kiểu) | True | False |  | Số mã hàng khác nhau đã mua trong 365 ngày |
| ipt_mean_days | (không ràng buộc kiểu) | True | False |  | Khoảng cách trung bình giữa hai đơn (NaN nếu mới mua 1 lần) |
| ipt_std_days | (không ràng buộc kiểu) | True | False |  | Độ lệch chuẩn khoảng cách giữa hai đơn |
| is_one_time | (không ràng buộc kiểu) | True | False |  | 1 nếu mới mua đúng 1 đơn |
| overdue_ratio | (không ràng buộc kiểu) | True | False |  | recency / ipt_mean: >1 nghĩa là đã quá chu kỳ mua thường lệ |
| n_cancels_365 | (không ràng buộc kiểu) | True | False |  | Số hóa đơn hủy/trả trong 365 ngày (theo ngày hủy) |
| cancel_value_365 | (không ràng buộc kiểu) | True | False |  | Giá trị hủy/trả trong 365 ngày |
| cancel_rate_365 | (không ràng buộc kiểu) | True | False |  | Tỷ lệ giá trị hủy / (bán + hủy) trong 365 ngày |
| days_since_last_cancel | (không ràng buộc kiểu) | True | False |  | Số ngày từ lần hủy gần nhất (NaN nếu chưa từng hủy) |
| value_trend_90_365 | (không ràng buộc kiểu) | True | False |  | Giá trị 90 ngày so với mức trung bình 90 ngày của năm |
| is_uk | (không ràng buộc kiểu) | True | False |  | 1 nếu khách ở Anh |
| t0_month | (không ràng buộc kiểu) | True | False |  | Tháng của t0 (biết trước) |
| days_to_xmas | (không ràng buộc kiểu) | True | False |  | Số ngày từ t0 đến Giáng sinh kế tiếp (biết trước) |
| country | (không ràng buộc kiểu) | True | False |  | Quốc gia của đơn gần nhất |
| (cấp bảng) |  |  |  | duy nhất theo (customer_id, t0) |  |
| (cấp bảng) |  |  |  | propensity không cộng bằng 1 |  |
| (cấp bảng) |  |  |  | propensity_logged ≠ propensity của A_logged |  |
| (cấp bảng) |  |  |  | value_obs > 0 ⇔ y_obs = 1 |  |
| (cấp bảng) |  |  |  | cutoff = t0 − 1 ngày |  |
| (cấp bảng) |  |  |  | sự kiện đặc trưng sau cutoff |  |
| (cấp bảng) |  |  |  | nhãn trưởng thành = t0 + 30 ngày |  |
| (cấp bảng) |  |  |  | strict=False; coerce=False | cột thừa: cho phép |

## Hợp đồng đầu vào của Router (GĐ7)

*Nguồn:* `src/s7_router.py:_schemas → INPUT_SCHEMA` · 20 cột, 3 ràng buộc cấp bảng

| cot | kieu | nullable | unique | rang_buoc | mo_ta |
|---|---|---|---|---|---|
| customer_id | int64 | False | False |  | Mã khách hàng (khóa kỹ thuật, không dùng làm đặc trưng) |
| t0 | datetime64[ns] | False | False |  | Thời điểm quyết định (thứ Hai, 2 tuần/lần) |
| cancel_rate_365 | float64 | True | False | in_range(min_value=0, max_value=1, include_min=True, include_max=True) | Tỷ lệ giá trị hủy / (bán + hủy) trong 365 ngày |
| value_365 | float64 | False | False | greater_than_or_equal_to(min_value=0) | Tổng giá trị đơn trong 365 ngày trước t0 |
| s_hat_mean | float64 | False | False | greater_than(min_value=0) |  |
| s_lo | float64 | False | False | greater_than(min_value=0) |  |
| s_hi | float64 | False | False | greater_than(min_value=0) |  |
| mu0_cal | float64 | False | False | in_range(min_value=0, max_value=1, include_min=True, include_max=True) |  |
| V_hat_a1 | float64 | False | False |  |  |
| V_hat_a2 | float64 | False | False |  |  |
| V_hat_a3 | float64 | False | False |  |  |
| tau_hat_a1 | float64 | False | False |  |  |
| tau_hat_a2 | float64 | False | False |  |  |
| tau_hat_a3 | float64 | False | False |  |  |
| tau_lo_a1 | float64 | False | False |  |  |
| tau_lo_a2 | float64 | False | False |  |  |
| tau_lo_a3 | float64 | False | False |  |  |
| tau_hi_a1 | float64 | False | False |  |  |
| tau_hi_a2 | float64 | False | False |  |  |
| tau_hi_a3 | float64 | False | False |  |  |
| (cấp bảng) |  |  |  | s_hi < s_lo |  |
| (cấp bảng) |  |  |  | trùng khóa khách × t0 |  |
| (cấp bảng) |  |  |  | strict=False; coerce=True | cột thừa: cho phép |

## Hợp đồng bản ghi quyết định – Router v1 (GĐ7, DECISION_SCHEMA)

*Nguồn:* `src/s7_router.py:_schemas → DECISION_SCHEMA` · 30 cột, 7 ràng buộc cấp bảng

| cot | kieu | nullable | unique | rang_buoc | mo_ta |
|---|---|---|---|---|---|
| decision_id | string[pyarrow] | False | True |  |  |
| idempotency_key | string[pyarrow] | False | True |  |  |
| customer_id | int64 | False | False |  | Mã khách hàng (khóa kỹ thuật, không dùng làm đặc trưng) |
| t0 | datetime64[ns] | False | False |  | Thời điểm quyết định (thứ Hai, 2 tuần/lần) |
| split | string[pyarrow] | False | False |  | train/valid/calib/locked/oot/embargo/excluded_entity_holdout |
| policy_version | string[pyarrow] | False | False |  |  |
| model_version | string[pyarrow] | False | False |  |  |
| proposed_action | string[pyarrow] | False | False | isin(allowed_values=['a0', 'a1', 'a2', 'a3']) |  |
| final_action | string[pyarrow] | False | False | isin(allowed_values=['a0', 'a1', 'a2', 'a3']) |  |
| channel | string[pyarrow] | False | False | isin(allowed_values=['none', 'email', 'call']) |  |
| guardrail_triggered | string[pyarrow] | False | False |  |  |
| guardrails_all | string[pyarrow] | False | False |  |  |
| hitl_flag | bool | False | False |  |  |
| hitl_trigger | string[pyarrow] | False | False |  |  |
| hitl_outcome | string[pyarrow] | False | False |  |  |
| control_group | bool | False | False |  |  |
| in_randomization_set | bool | False | False |  |  |
| propensity_logged | float64 | False | False | in_range(min_value=0, max_value=1, include_min=True, include_max=True) |  |
| propensity_a0 | float64 | False | False | in_range(min_value=0, max_value=1, include_min=True, include_max=True) |  |
| V_hat_proposed | float64 | False | False |  |  |
| tau_hat | float64 | True | False |  |  |
| tau_lo | float64 | True | False |  |  |
| tau_hi | float64 | True | False |  |  |
| s_hat_mean | float64 | False | False |  |  |
| s_lo | float64 | False | False |  |  |
| s_hi | float64 | False | False |  |  |
| reason_1 | string[pyarrow] | True | False |  |  |
| reason_2 | string[pyarrow] | True | False |  |  |
| reason_3 | string[pyarrow] | True | False |  |  |
| decided_at | datetime64[ns] | False | False |  |  |
| (cấp bảng) |  |  |  | đối chứng phải có final_action = a0 |  |
| (cấp bảng) |  |  |  | đối chứng phải thuộc tập ngẫu nhiên |  |
| (cấp bảng) |  |  |  | bị chặn cứng phải về a0 |  |
| (cấp bảng) |  |  |  | Router không được tự thêm voucher |  |
| (cấp bảng) |  |  |  | kênh không khớp hành động cuối |  |
| (cấp bảng) |  |  |  | HITL phải có kết quả duyệt |  |
| (cấp bảng) |  |  |  | strict=True; coerce=True | cột thừa: cấm |

## Hợp đồng bản ghi quyết định – Router v2.1 (đang triển khai; mở rộng từ DECISION_SCHEMA)

*Nguồn:* `src/s7_router.py:_schema_v21 → DECISION_SCHEMA_V21` · 35 cột, 14 ràng buộc cấp bảng

| cot | kieu | nullable | unique | rang_buoc | mo_ta |
|---|---|---|---|---|---|
| decision_id | string[pyarrow] | False | True |  |  |
| idempotency_key | string[pyarrow] | False | True |  |  |
| customer_id | int64 | False | False |  | Mã khách hàng (khóa kỹ thuật, không dùng làm đặc trưng) |
| t0 | datetime64[ns] | False | False |  | Thời điểm quyết định (thứ Hai, 2 tuần/lần) |
| split | string[pyarrow] | False | False |  | train/valid/calib/locked/oot/embargo/excluded_entity_holdout |
| policy_version | string[pyarrow] | False | False |  |  |
| model_version | string[pyarrow] | False | False |  |  |
| proposed_action | string[pyarrow] | False | False | isin(allowed_values=['a0', 'a1', 'a2', 'a3']) |  |
| final_action | string[pyarrow] | False | False | isin(allowed_values=['a0', 'a1', 'a2', 'a3']) |  |
| channel | string[pyarrow] | False | False | isin(allowed_values=['none', 'email', 'call']) |  |
| guardrail_triggered | string[pyarrow] | False | False |  |  |
| guardrails_all | string[pyarrow] | False | False |  |  |
| hitl_flag | bool | False | False |  |  |
| hitl_trigger | string[pyarrow] | False | False |  |  |
| hitl_outcome | string[pyarrow] | False | False |  |  |
| control_group | bool | False | False |  |  |
| in_randomization_set | bool | False | False |  |  |
| propensity_logged | float64 | False | False | in_range(min_value=0, max_value=1, include_min=True, include_max=True) |  |
| propensity_a0 | float64 | False | False | in_range(min_value=0, max_value=1, include_min=True, include_max=True) |  |
| V_hat_proposed | float64 | False | False |  |  |
| tau_hat | float64 | True | False |  |  |
| tau_lo | float64 | True | False |  |  |
| tau_hi | float64 | True | False |  |  |
| s_hat_mean | float64 | False | False |  |  |
| s_lo | float64 | False | False |  |  |
| s_hi | float64 | False | False |  |  |
| reason_1 | string[pyarrow] | True | False |  |  |
| reason_2 | string[pyarrow] | True | False |  |  |
| reason_3 | string[pyarrow] | True | False |  |  |
| decided_at | datetime64[ns] | False | False |  |  |
| is_backfill | bool | False | False |  |  |
| p_control | float64 | False | False | isin(allowed_values=[0.0, 0.1]) |  |
| router_reason_code | string[pyarrow] | False | False |  |  |
| backfill_status | string[pyarrow] | False | False |  |  |
| repair_status | string[pyarrow] | False | False | isin(allowed_values=['none', 'downgrade_a1', 'drop_a0']) |  |
| (cấp bảng) |  |  |  | đối chứng phải có final_action = a0 |  |
| (cấp bảng) |  |  |  | đối chứng phải thuộc tập ngẫu nhiên |  |
| (cấp bảng) |  |  |  | bị chặn cứng phải về a0 |  |
| (cấp bảng) |  |  |  | Router chỉ được thêm voucher qua bước bù độ phủ |  |
| (cấp bảng) |  |  |  | khách bù chỉ nhận a1 |  |
| (cấp bảng) |  |  |  | khách bù không thuộc đối chứng, propensity đối chứng = 0 |  |
| (cấp bảng) |  |  |  | khách bù phải có mã COVERAGE_BACKFILL |  |
| (cấp bảng) |  |  |  | p_control không khớp tập ngẫu nhiên |  |
| (cấp bảng) |  |  |  | kênh không khớp hành động cuối |  |
| (cấp bảng) |  |  |  | HITL phải có kết quả duyệt |  |
| (cấp bảng) |  |  |  | hạ a2→a1: hành động cuối phải là a1 |  |
| (cấp bảng) |  |  |  | hạ về a0 ở bước sửa: final a0, không đối chứng, ngoài tập ngẫu nhiên (propensity a0 = 1) |  |
| (cấp bảng) |  |  |  | mã lý do bước sửa |  |
| (cấp bảng) |  |  |  | strict=True; coerce=True | cột thừa: cấm |
