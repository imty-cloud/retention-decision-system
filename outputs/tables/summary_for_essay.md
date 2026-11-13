# Ánh xạ kết quả → tiểu luận (PTDLNC – hệ thống ra quyết định giữ chân khách hàng đa kênh)

> Sinh tự động ngày 2026-11-12 bởi `tools/build_packaging.py` từ các CSV trong `outputs/tables/`; **mọi con số đều được đọc trực tiếp từ file nguồn ghi ở cột cuối** (không gõ tay). Bản này chỉ tổng hợp kết quả đã có; không chạy lại pipeline.

**Cần người dùng điền / xác nhận:**

- Cột CLO theo bảng chuẩn đầu ra trong đề: CLO1 → phần I–III; CLO2 → I, IV; CLO3 → III, IV; CLO4 → II, IV; CLO5 → IV, V. Gán theo nội dung từng kết quả.
- Đánh số mục I–VI là suy từ yêu cầu đóng gói (IV = kết quả, V = thảo luận, VI = playbook) và CLAUDE.md §5 (III = thiết kế dữ liệu, VIII = phụ lục). Mục I, II, VII chưa có nội dung kết quả cụ thể nên không liệt kê.

**Kết luận tổng quát:** hệ thống **KHÔNG đạt đủ cổng phát hành** trên Locked Test vì G4b (regret chuẩn hóa) trượt; bảy tiêu chí còn lại đạt ở mức toàn bộ (xem IV.6). Entity-holdout (chỉ mô tả) trượt G2, G3, G4a, G7. Xem thêm mục "Công khai và sai lệch" ở cuối.


## III. Thiết kế bài toán và dữ liệu

| Kết quả | Mục | CLO | Nguồn |
|---|---|---|---|
| Quy mô các tập (dòng khách × t0 / số kỳ): train 43,921 / 15; valid 14,346 / 4; calib 10,981 / 3; locked 12,947 / 3; oot 21,693 / 5 | III | CLO1, CLO2 | `s1_split_summary.csv` |
| Kiểm thử rò rỉ: 15/15 phép thử đạt (point-in-time, bất biến khi nhiễu tương lai, tính lại nhãn) | III | CLO1, CLO2 | `s1_leakage_tests.csv` |
| Embargo giữa các khối ≥ độ trễ nhãn 30 ngày: khoảng cách nhỏ nhất 42 ngày, 4/4 đạt | III | CLO1, CLO2 | `s1_embargo_check.csv` |
| Tỷ lệ mua lại gốc trôi theo mùa (mọi dòng): train 0.258 → valid 0.175 → calib 0.203 → locked 0.187 → oot 0.282 | III/V | CLO1, CLO2 | `s1_split_summary.csv; fig_s1_base_rate_by_t0.png` |
| Sơ đồ chia tập theo thời gian | III (Graphic Abstract) | CLO1, CLO2 | `fig_s1_split_timeline.png` |
| Lớp can thiệp bán tổng hợp: so thô lệch khỏi ATE thật (train, mua lại) a1 -0.0176, a2 -0.0075, a3 0.0276 | III/V | CLO1, CLO2 | `s1b_confounding_check.csv` |

## IV. Kết quả

### IV.1 Dự báo nền và hiệu chỉnh xác suất (GĐ2)

| Kết quả | Mục | CLO | Nguồn |
|---|---|---|---|
| Valid: ROC-AUC 0.786, PR-AUC 0.532; Brier 0.1371 → 0.1173 sau Platt; ECE 0.1387 → 0.0474 | IV | CLO2 | `s2_metrics_valid.csv` |
| Tiêu chí chấp nhận GĐ2: ECE ≤ 0.03 trên valid **không đạt** (0.0474); Brier không tệ hơn: đạt; coverage conformal Ŝ 0.8822 (mục tiêu 0.90, trong [0.85, 0.95]) | IV | CLO2 | `s2_acceptance.csv; s2_basket_conformal_valid.csv` |
| Locked (G2, dòng a0, n = 3,422): ECE 0.0906 (thô) → 0.0201 (Platt); Brier 0.1213 → 0.1127 | IV | CLO2 | `s8_locked_g2_calibration.csv` |
| Biểu đồ tin cậy trước/sau hiệu chỉnh; khoảng conformal của giá trị giỏ | IV | CLO2 | `fig_s2_reliability_valid.png; fig_s2_basket_conformal.png` |

### IV.2 Giải thích và độ tin cậy (GĐ3)

| Kết quả | Mục | CLO | Nguồn |
|---|---|---|---|
| PFI theo nhóm (valid, neg log-loss): RFM 0.1408, Nhịp mua 0.0068, Hủy đơn -0.0017, Lịch 0.0000, Quốc gia -0.0000 | IV | CLO2 | `s3_pfi_group.csv; fig_s3_pfi.png` |
| Ổn định SHAP qua 5 lần bootstrap: Spearman trung bình 0.889 (nhỏ nhất 0.798) | IV | CLO2 | `s3_shap_stability.csv` |
| Faithfulness: xóa top-1 theo SHAP làm lệch logit 0.356 so với 0.054 khi xóa ngẫu nhiên | IV | CLO2 | `s3_faithfulness_deletion.csv` |
| SHAP/PFI **không phải nhân quả** (nhắc trong báo cáo); mã lý do top-3 cho Router | IV/V | CLO2 | `fig_s3_shap_beeswarm.png; reasons_s3.parquet` |

### IV.3 Nhân quả và uplift (GĐ4)

| Kết quả | Mục | CLO | Nguồn |
|---|---|---|---|
| So thô lệch ATE thật; IPW và DML khử nhiễu: ATE mua lại a1 thật 0.0385, thô 0.0209, IPW 0.0368 | IV | CLO3 | `s4_ate_vs_truth.csv; fig_s4_ate_forest.png` |
| CI DML phủ ATE thật: 6/6 (APOS), 6/6 (IRM); placebo (xáo A) CI chứa 0: 6/6 | IV | CLO3 | `s4_acceptance.csv; s4_placebo.csv` |
| Qini CausalForestDML trên nhánh ngẫu nhiên valid+calib: a1 0.065 [0.008; 0.116], a2 0.030 [-0.026; 0.089], a3 0.035 [-0.019; 0.096]; Qini > 0 cho 3/3, CI > 0 cho 1/3 | IV | CLO3 | `s4_qini.csv; fig_s4_qini_curves.png` |
| Dự báo power cho G3 (đăng ký trước) | IV/V | CLO3 | `s4_power_g3_summary.csv; fig_s4b_power_g3.png` |

### IV.4 Tối ưu hóa: SPO+, MILP (GĐ5–GĐ6)

| Kết quả | Mục | CLO | Nguồn |
|---|---|---|---|
| Regret chuẩn hóa hiện thực (lô 100 khách, calib, 5 seed): Luật RFM cũ 1.008 ± 0.033; Greedy (V̂/mệnh giá) 0.926 ± 0.031; MILP trên V̂ (GĐ4) 0.925 ± 0.025; PTO tuyến tính (MSE) 0.999 ± 0.024; PTO LightGBM (MSE) 1.110 ± 0.031; SPO+ tuyến tính 1.036 ± 0.064 | IV | CLO4 | `s5_regret_summary.csv; fig_s5_regret_by_method.png` |
| SPO+ − PTO tuyến tính (regret hiện thực): +0.0373 ± 0.0657; SPO+ tốt hơn ở 1/5 seed – **SPO+ không thắng PTO** (giữ nguyên kết quả) | IV/V | CLO4 | `s5_spo_vs_pto_paired.csv; s5_regret_summary_hai_luoi.csv (lưới gốc vs mở rộng)` |
| Quy mô toàn t0 (7 kỳ valid+calib, cộng dồn): regret A MILP 0.8321, Greedy 0.8310, luật cũ 1.0158; MILP ≥ Greedy trên V̂ ở 7/7 kỳ | IV | CLO4 | `s6_main_summary.csv; fig_s6_main_comparison.png` |
| Hiệu chỉnh xác suất ↔ chi phí chiết khấu: tỷ lệ chi thực/kế hoạch (kỳ vọng) 0.648 khi μ̂0 thô so với 0.902 khi đã hiệu chỉnh | IV/V | CLO4 | `s6_calibration_discount_cost_summary.csv; fig_s6_calibration_discount_cost.png` |
| Giá bóng ngân sách / cuộc gọi (LP relaxation); độ nhạy B, K, biên lợi nhuận m | IV | CLO4 | `s6_shadow_prices.csv; s6_sensitivity_BK.csv; s6_sensitivity_margin.csv` |

### IV.5 Action Router, guardrail, HITL (GĐ7, Router v2.1, valid+calib 7 kỳ)

| Kết quả | Mục | CLO | Nguồn |
|---|---|---|---|
| Giá trị hiện thực cộng dồn: MILP qua Router £21,483, luật cũ qua Router £15,039, luật cũ thuần −£2,650 | IV | CLO4, CLO5 | `s7_policy_value_summary_v21.csv; fig_s7_policy_value.png` (v2.1; £15 006 ở G4a là v2, xem C8) |
| Luồng quyết định MILP: đề xuất 1,788 → HITL 819 → đối chứng 137 → gửi thực tế 1,240; luật cũ: gửi 809, bù độ phủ 288 | IV | CLO4, CLO5 | `s7_decision_flow_v21.csv; fig_s7_decision_flow.png` |
| Người duyệt HITL là **quy tắc mô phỏng**, không phải người thật (6b) | V | CLO4, CLO5 | `CLAUDE.md §6b; s7_hitl_reviewer_value.csv` |

### IV.6 Bảng Release Gates (Locked Test, mở 1 lần)

| Gate | Kết quả | Giá trị | Ngưỡng | Dự báo đăng ký trước (6b) | Dự báo so với thực tế |
|---|---|---|---|---|---|
| G1 | ĐẠT | 7/7 | tất cả | 6b không có dự báo riêng. Tham chiếu GĐ1: data contract + 6 kiểm thử rò rỉ đều PASS; embargo calib→locked 42 ngày PASS. | — |
| G2 | ĐẠT | ECE 0.0201 [CI 0.0169–0.0351] · Brier 0.1127 vs thô 0.1213 | ECE ≤ 0.03; Brier không tệ hơn | 6b: Platt khớp trên calib (tháng 4–5) CÓ THỂ KHÔNG hợp locked (tháng 7–8). Tham chiếu §6: ECE valid 0.047 > 0.03 (trượt), ECE ngoài fold trong calib 0.028. Không có xác suất dự báo bằng số. | dự báo chỉ nêu khả năng 'không hợp'; không xảy ra |
| G3 | ĐẠT (phụ thuộc hạt giống, xem (iii)/B2) | 2/3 (a1 0.035 [-0.037, 0.098] · a2 0.098 [0.015, 0.161] · a3 0.075 [0.010, 0.144]) | ≥ 2 | 6b: P(G3 đạt) ≈ 4–7 % (kịch bản chính), 10–13 % (lạc quan), 17–21 % (train ngoài fold) ⇒ nhiều khả năng KHÔNG đạt (do cỡ mẫu nhánh ngẫu nhiên); entity-holdout < 3 %, chỉ mô tả. | **SAI** (dự báo ≈ 4–7 %, thực tế đạt) |
| G4a | ĐẠT | £8,785 vs £7,824 (chênh £961; CI 95 % [−£2 273; +£3 980], xem B1) | chênh ≥ £782 | 6b: KHẢ THI. valid+calib (Router v2): MILP qua Router £21 483 vs Luật cũ qua Router £15 006 vs luật cũ thuần −£2 650 ⇒ đạt cả hai cách so. | đúng |
| G4a-phụ | ĐẠT | £8,785 vs £3,739 (chênh £5,046) | chênh ≥ £374 | 6b: KHẢ THI. valid+calib (Router v2): MILP qua Router £21 483 vs Luật cũ qua Router £15 006 vs luật cũ thuần −£2 650 ⇒ đạt cả hai cách so. | đúng |
| G4b | TRƯỢT | 0.894 | ≤ 0.35 | 6b: gần như chắc TRƯỢT – ngay chính sách biết kỳ vọng thật (đáp án B) có regret A = 0.742 ± 0.048 (lô 100, calib). | đúng |
| G4b-phụ | TRƯỢT | 0.898 | ≤ 0.35 | 6b: gần như chắc TRƯỢT – ngay chính sách biết kỳ vọng thật (đáp án B) có regret A = 0.742 ± 0.048 (lô 100, calib). | đúng |
| G5 | ĐẠT | 0 | ≤ 0 | 6b (Router v2.1): bước sửa khả thi cuối nhằm 0 vi phạm; valid+calib v2.1 = 0 vi phạm. Không có xác suất dự báo. | đúng |
| G6 | ĐẠT | 1.052 | ≤ 30 | 6b không có dự báo riêng. Tham chiếu GĐ7 v2.1: P95 đầu–cuối = 0.75 s ≪ 30 s. | đúng |
| G7 | ĐẠT | UK +18.8% · non-UK +4.7% | ≥ −5% | 6b: so sánh chính như G4a. Tham chiếu valid+calib v2.1: UK +58 %, non-UK +21 % vs Luật cũ qua Router ⇒ đạt. | đúng |
| G7-phụ | ĐẠT | UK +5283.9% (mẫu số £92.83; chênh tuyệt đối +£4 905, xem C2) · non-UK +3.9% | ≥ −5% | 6b: so sánh chính như G4a. Tham chiếu valid+calib v2.1: UK +58 %, non-UK +21 % vs Luật cũ qua Router ⇒ đạt. | — |

*Nguồn:* `s8_locked_gates.csv` (toàn bộ locked: 12 947 dòng, 3 kỳ). **G4 = G4a ∧ G4b → TRƯỢT ⇒ không đạt đủ cổng.** Hình: `fig_s8_locked_gates.png`.

Theo quần thể (bảng tổng hợp đã sửa, lập từ `s8_locked_gates.csv`):

| quan_the | G1 | G2 | G3 | G4a | G4b | G5 | G6 | G7 | G4 |
|---|---|---|---|---|---|---|---|---|---|
| Toàn bộ (chấm chính thức) | ĐẠT | ĐẠT | ĐẠT | ĐẠT | TRƯỢT | ĐẠT | ĐẠT | ĐẠT | TRƯỢT |
| Khách thường (is_entity_holdout=0) | n/a | ĐẠT | TRƯỢT | ĐẠT | TRƯỢT | n/a | n/a | ĐẠT | TRƯỢT |
| Entity-holdout (is_entity_holdout=1, mô tả) | n/a | TRƯỢT | TRƯỢT | TRƯỢT | TRƯỢT | n/a | n/a | TRƯỢT | TRƯỢT |

*Nguồn:* `s8_locked_gate_summary_corrected.csv`. Entity-holdout (2 005 dòng) chỉ mang tính mô tả.

Qini trên nhánh ngẫu nhiên của locked (G3):

| quan_the | hanh_dong | n | qini | qini_ci_lo | qini_ci_hi | ci_duong |
|---|---|---|---|---|---|---|
| toan_bo | a1 | 1343 | 0.035 | -0.037 | 0.098 | False |
| toan_bo | a2 | 1323 | 0.098 | 0.015 | 0.161 | True |
| toan_bo | a3 | 1301 | 0.075 | 0.01 | 0.144 | True |
| khach_thuong | a1 | 1153 | 0.026 | -0.053 | 0.1 | False |
| khach_thuong | a2 | 1107 | 0.111 | 0.037 | 0.188 | True |
| khach_thuong | a3 | 1096 | 0.067 | -0.015 | 0.138 | False |
| entity_holdout | a1 | 190 | 0.1 | -0.069 | 0.291 | False |
| entity_holdout | a2 | 216 | -0.005 | -0.205 | 0.178 | False |
| entity_holdout | a3 | 205 | 0.095 | -0.124 | 0.288 | False |

*Nguồn:* `s8_locked_g3_qini.csv`.


### IV.6b Ghi chú sau kiểm toán độc lập 13/11 (mức B và C; số mục theo `outputs/manifests/audit_2026-11-13.md`)

Kiểm toán độc lập (chỉ đọc) tính lại 7/7 gate khớp tới chữ số cuối, **không phát hiện lỗi mức A**; 443 con số đáng kể trong file này khớp CSV nguồn. Các điểm sau không đổi kết luận "không đạt đủ cổng vì G4b" nhưng phải nêu trong bài:

- **B1 – G4a đạt nhưng nằm trong nhiễu.** Chênh £961 so với ngưỡng £782 (dư £179). Bootstrap cụm khách 5 000 lần: chênh có **CI 95 % [−£2 273; +£3 980]**, SE £1 591, chỉ **55.8 %** lần lấy lại mẫu đạt ngưỡng; hai luồng chỉ khác nhau ở 553/12 947 dòng; theo kỳ +£397, +£368, +£196. Theo giá trị kỳ vọng (B): £8 211 vs £5 129. Được viết "đạt theo ước lượng điểm đã đăng ký"; **không** viết "MILP hơn luật cũ có ý nghĩa thống kê". *Nguồn:* `s8_locked_gates.csv`, `s8_locked_g4a_comparison.csv`, audit B1 (CI do kiểm toán tính, `s8_mlops.py` không tính CI cho gate này).
- **B4 – G2: ngưỡng nằm trong CI; CI percentile chệch.** ECE 0.0201, CI [0.0169; 0.0351] chứa 0.03; bootstrap của ECE chệch lên (trung bình 0.0256 so với 0.0201); ở khách thường điểm 0.0182 nằm ngoài CI của chính nó [0.0193; 0.0392]. Không dùng CI này để nói "ECE khác 0.03 có ý nghĩa". Điểm đỡ: ECE trên **toàn bộ** 12 947 dòng với Y(0) thật = **0.0123**; độ nhạy theo bin: 10 bin 0.0171, 20 bin 0.0212, bin đều 0.0192 (đều < 0.03). *Nguồn:* `s8_locked_gates.csv` (G2), audit B4.
- **B5 – G4b không thể đạt; MILP ≈ Greedy theo hiện thực.** Regret A lấy mốc là chính sách biết trước kết quả hiện thực; ngay chính sách biết **kỳ vọng thật** cũng có regret A 0.742 (`s5_regret_summary.csv`); mọi phương pháp ở quy mô toàn t0 ≥ 0.83. Trên valid+calib: MILP **0.8321** vs Greedy **0.8310**, MILP hơn Greedy theo hiện thực ở 4/7 kỳ (dù hơn trên V̂ ở 7/7); SPO+ thua PTO tuyến tính (1.036 vs 0.999). Không viết MILP hay SPO+ "giảm regret" so với Greedy/PTO theo giá trị hiện thực; tách rõ regret A (gồm may rủi) và regret B. *Nguồn:* `s6_main_summary.csv`, `s5_regret_summary.csv`.
- **B6 – Challenger-2 khớp trong mẫu.** μ0 học lại trên train+valid+calib+locked rồi chấm điểm chính kho calib+locked; bộ Platt khớp trên kho đó (ba kỳ OOT đầu có cửa sổ toàn locked = dữ liệu mô hình đã học). ECE/Brier của Challenger-2 (0.026 / 0.039 / 0.042 / 0.043 / 0.060) **không so sánh sạch** với Current và Challenger; không ảnh hưởng gate (PHỤ). Nếu trích thì kèm cảnh báo này. Challenger (Platt theo kỳ) đúng. *Nguồn:* `s8_oot_*` bảng Brier/ECE, audit B6.
- **B8 – Đăng ký trước chỉ có mtime + hash.** Cả ba commit đều sau khi mọi việc xong; GĐ1→GĐ8 diễn ra trong một ngày. Thứ tự "đăng ký trước → đóng băng → mở locked" được chứng minh bằng mtime `CLAUDE.md` 21:30:21 < freeze 21:31:07 < khóa 21:40:19 (giờ +07, 12/11) và `freeze_aggregate_sha256` trong file khóa trùng manifest. Đủ để tin nội dung không đổi từ lúc đóng băng; **không chứng minh được với bên thứ ba thời điểm viết từng đoạn trước đó**. *Nguồn:* `s8_locked_opened.lock`, `s8_freeze_manifest.json`, audit B8.
- **B9 – Giới hạn bán tổng hợp.** Cơ chế sinh tác động là hàm của hạng recency, n_orders_365, aov_365, units_per_order_365 (đều là đặc trưng mô hình thấy) và chính sách ghi log chỉ phụ thuộc các đặc trưng đó ⇒ không nhiễu chưa quan sát, chồng lấp đều đúng **theo cấu trúc**. "CI DML phủ ATE thật 6/6", placebo 6/6, Qini > 0 chỉ kiểm chứng **cài đặt**, không phải hiệu quả trên dữ liệu thật; trên RCT thật (Hillstrom, IV.8) không có CATE nào hơn ngẫu nhiên có ý nghĩa ở kết quả chính. Không khái quát sang dữ liệu thật. *Nguồn:* `s1b_intervention.py:37-64`, `s9_*`, audit B9.
- **B7 – Môi trường chạy (đính chính).** GĐ1/GĐ1b chạy trên **Linux x86_64, Python 3.13.18** (pandas 3.0.5, numpy 2.5.3); các GĐ còn lại chạy trên **macOS arm64, Python 3.14.7**. Dựng lại GĐ1+GĐ1b trên macOS (3.14.7) cho nội dung trùng 100 % ⇒ bằng chứng tái lập chéo nền tảng. Đã sửa ở README và `s8_compat_matrix.csv`. *Nguồn:* `s1_manifest.json`, `s1b_manifest.json`, audit B7.

**Ghi chú mức C:**

- **C2 – G7-phụ.** "UK +5283.9 %" đúng số học nhưng vô nghĩa về độ lớn: mẫu số là luật cũ thuần ở UK chỉ £92.83 (khách thường còn âm, −£91.58 ⇒ +5 645 %). Báo **chênh tuyệt đối +£4 905** (MILP qua Router £4 997.92 vs £92.83) bên cạnh phần trăm; G7 chính (+18.8 % / +4.7 %) không bị ảnh hưởng. *Nguồn:* `s8_locked_gates.csv` (G7-phụ).
- **C5 – Hai đầu vào ngoài phạm vi băm.** `Scorer` đọc hệ số smearing (1.0492446593) và q conformal (0.9028838202) từ `outputs/manifests/s2_manifest.json`; G1 đọc `s1_embargo_check.csv`. Kiểm toán đã đối chiếu: hai hệ số trùng tỷ số trong `pred_s2.parquet` (có băm), `s_hat_mean` trong bản ghi quyết định trùng `pred_s2` (lệch 0).
- **C6 – G6 không gồm khâu dựng đặc trưng từ giao dịch** (`features_at`). Đo riêng: ≈ 0.06 s/t0 ⇒ không đáng kể so với P95 1.05 s.
- **C7 – PSI phóng đại trôi.** Điểm tham chiếu train là điểm **trong mẫu**; `tenure_days`, `days_to_xmas`, `t0_month` trôi theo cấu trúc thời gian (PSI tenure 1.86, μ̂0 0.88). Diễn giải "8 báo động" cần lưu ý; một cảnh báo PSI đơn lẻ không tự kích hoạt tái huấn luyện.
- **C8 – Hai con số cho cùng khái niệm.** Luật cũ qua Router **£15 039** là phiên bản Router v2.1 (IV.5, `s7_policy_value_summary_v21.csv`); **£15 006** là phiên bản v2 (cột dự báo đăng ký trước của G4a, `s7_policy_value_summary_v2.csv`). Cả hai đúng, khác phiên bản.
- **C9 – Hai nguồn regret V̂.** Bảng V.4 dùng instance của **seed 0** (regret V̂ 0.9602, `s5_mse_vs_regret.csv`); bảng IV.4 dùng **trung bình 5 seed** (0.925, `s5_regret_summary.csv`). Không so trực tiếp hai con số.
- **C10 – "Luật cũ" là gì.** Là MILP trên điểm logit của luật RFM, **không phải** chính sách ghi log ngẫu nhiên; "Luật cũ qua Router" dùng V̂ và CI τ̂ của mô hình ở bước bù độ phủ và HITL. So sánh chính vì thế **bất lợi cho MILP**. *Nguồn:* `s5b_dr_diagnostics.py:110-114`, `s7_router.py:741-769`.
- **C11 – Kiểm thử rò rỉ GĐ1.** T1–T3 chỉ chạy trên 4 t0 ngẫu nhiên (không có locked/oot). Kiểm toán bổ sung: T1 trên 3 kỳ locked đạt; tự tính lại cả 38 kỳ đạt, 0 kỳ lệch. Có thể trích.
- **C12 – Ghi chú tái lập.** `CLAUDE.md` §2 còn ghi "tiếp theo: `--open-locked`" (lỗi thời; không sửa được vì đã băm). Chạy lại GĐ1 tạo `data/processed/raw_combined.parquet` ⇒ `verify_freeze` báo "file mới" dù nội dung không đổi. `s1_manifest.json` không còn khớp SHA của `fig_s1_split_timeline.png` và `fig_s1_base_rate_by_t0.png` (hình ghi sau manifest). `REPO_URL` của notebook đã điền ở commit 6570461.
- **C1/C3/C4** (test oracle chỉ kiểm theo tên, `s8_locked_gate_summary.csv` toàn n/a, hai file test sửa sau băng): đã nêu ở mục (ii), (v) bên dưới; kiểm toán soát tay không thấy vi phạm cô lập đáp án, nên không cần thêm ghi chú ở đây.


### IV.7 Phát lại OOT (5 kỳ, 21 693 dòng, mô hình Current đóng băng)

**Độ trễ từng khâu (giây / một t0, ≥ 30 lượt/kỳ; gộp 5 kỳ)** – P50 / P95 / P99:

| khau | so_luot | P50 | P95 | P99 |
|---|---|---|---|---|
| t_dac_trung | 150 | 0.004 | 0.004 | 0.005 |
| t_ma_ly_do | 150 | 0.067 | 0.076 | 0.091 |
| t_cham_diem | 150 | 0.816 | 0.93 | 1.02 |
| t_hop_dong_va_mien | 150 | 0.011 | 0.013 | 0.017 |
| t_milp | 150 | 0.205 | 0.236 | 0.267 |
| t_router | 150 | 0.038 | 0.045 | 0.07 |
| tong | 150 | 1.14 | 1.316 | 1.406 |

Tổng đầu–cuối theo kỳ:

| ky | so_luot | P50 | P95 | P99 |
|---|---|---|---|---|
| 2011-09-12 | 30 | 1.023 | 1.133 | 1.201 |
| 2011-09-26 | 30 | 1.13 | 1.253 | 1.35 |
| 2011-10-10 | 30 | 1.178 | 1.392 | 1.407 |
| 2011-10-24 | 30 | 1.164 | 1.33 | 1.464 |
| 2011-11-07 | 30 | 1.133 | 1.168 | 1.179 |

*Nguồn:* `s8_oot_latency.csv` (locked: `s8_locked_g6_latency_percentiles.csv`).

**PSI so với train (gộp 5 kỳ):** 18 ổn định, 12 cảnh báo (≥ 0.10), 8 báo động (≥ 0.25) trên 38 biến/điểm. 8 giá trị lớn nhất (đặc trưng lịch `t0_month`, `days_to_xmas` trôi theo thiết kế mùa vụ):

| loai | bien | psi | trang_thai |
|---|---|---|---|
| đặc trưng | days_to_xmas | 5.127 | báo động |
| đặc trưng | t0_month | 4.722 | báo động |
| đặc trưng | tenure_days | 1.858 | báo động |
| điểm số | mu0_cal | 0.879 | báo động |
| đặc trưng | ipt_std_days | 0.421 | báo động |
| đặc trưng | days_since_last_cancel | 0.342 | báo động |
| điểm số | V_hat_a2 | 0.287 | báo động |
| đặc trưng | ipt_mean_days | 0.278 | báo động |

*Nguồn:* `s8_oot_psi.csv`.

**Brier / ECE chỉ trên kỳ có nhãn đã trưởng thành** (t0 + 30 ngày ≤ ngày tính), dòng a0; Current vs Challenger (Platt theo kỳ) vs Challenger-2 (μ0 học lại, PHỤ):

| ky | ngày tính | trễ (ngày) | ece – Challenger (Platt theo kỳ) | ece – Challenger-2 (μ0 học lại, PHỤ) | ece – Current | brier – Challenger (Platt theo kỳ) | brier – Challenger-2 (μ0 học lại, PHỤ) | brier – Current |
|---|---|---|---|---|---|---|---|---|
| 2011-09-12 | 2011-10-24 | 42 | 0.0302 | 0.0264 | 0.0319 | 0.1486 | 0.1464 | 0.1486 |
| 2011-09-26 | 2011-11-07 | 42 | 0.0375 | 0.0389 | 0.0364 | 0.1642 | 0.1615 | 0.1638 |
| 2011-10-10 | 2011-12-07 | 58 | 0.0539 | 0.0416 | 0.0573 | 0.1643 | 0.1595 | 0.1649 |
| 2011-10-24 | 2011-12-07 | 44 | 0.067 | 0.0429 | 0.0755 | 0.1754 | 0.1718 | 0.1771 |
| 2011-11-07 | 2011-12-07 | 30 | 0.0656 | 0.0603 | 0.0863 | 0.1812 | 0.1789 | 0.1849 |

Ngưỡng báo động ECE = 0.05 (`monitoring.ece_alert`). *Nguồn:* `s8_oot_calibration_matured.csv`. Ghi chú: tỷ lệ mua thực tế tăng từ 0.245 lên 0.327 trong khi p̂ trung bình chỉ ≈ 0.24 ⇒ Current bị hụt (mùa Q4).

**Action TVD** so với phân phối hành động cuối của Current trên valid+calib: 0.0112 – 0.0146 (ngưỡng 0.2; cảnh báo: 0/5 kỳ). *Nguồn:* `s8_oot_action_tvd.csv`.

**Độ trễ nhãn – tại mỗi ngày phát lại, bao nhiêu kỳ/dòng đã có nhãn trưởng thành:**

| ngay_phat_lai | so_ky_da_quyet_dinh | so_ky_nhan_truong_thanh | so_dong_da_quyet_dinh | so_dong_nhan_truong_thanh | so_dong_a0_co_nhan | ty_le_dong_co_nhan |
|---|---|---|---|---|---|---|
| 2011-09-12 | 1 | 0 | 4302 | 0 | 0 | 0.0 |
| 2011-09-26 | 2 | 0 | 8648 | 0 | 0 | 0.0 |
| 2011-10-10 | 3 | 0 | 13008 | 0 | 0 | 0.0 |
| 2011-10-24 | 4 | 1 | 17358 | 4302 | 1133 | 0.248 |
| 2011-11-07 | 5 | 2 | 21693 | 8648 | 2238 | 0.399 |
| 2011-12-07 | 5 | 5 | 21693 | 21693 | 5637 | 1.0 |

Mỗi chỉ số giám sát bị trễ bao lâu so với lúc ra quyết định:

| chi_so | can_nhan | tre_toi_thieu_ngay | tre_thuc_te_TB_ngay | tre_thuc_te_max_ngay | ghi_chu |
|---|---|---|---|---|---|
| Độ trễ xử lý P95 (G6) | False | 0 | 0.0 | 0 | đo ngay tại t0 |
| PSI đặc trưng / điểm số | False | 0 | 0.0 | 0 | đặc trưng có tại cutoff = t0 − 1 ngày |
| Action TVD | False | 0 | 0.0 | 0 | có ngay sau khi Router ra quyết định |
| Vi phạm ràng buộc (G5) | False | 0 | 0.0 | 0 | kiểm toán ngay trên bản ghi quyết định |
| Brier / ECE (Current, Challenger) | True | 30 | 43.2 | 58 | chờ hết cửa sổ [t0, t0+30) rồi tới ngày phát lại kế tiếp (lịch 2 tuần) |
| Khớp lại Challenger (Platt) | True | 30 | 43.2 | 58 | chỉ dùng kỳ đã trưởng thành ⇒ áp cho kỳ sau với độ trễ ≥ 30 ngày |
| Giá trị thực hiện DR / đối chứng | True | 30 | 43.2 | 58 | cần lợi nhuận trong cửa sổ kết quả |
| Khuyết nhãn có chọn lọc (độ chệch) | True | 30 | 43.2 | 58 | cần Y(0) của nhóm đối chứng |

*Nguồn:* `s8_oot_label_delay_by_date.csv`, `s8_oot_label_delay_by_metric.csv`.

**Giá trị thực hiện (DR/AIPW trên nhóm đối chứng 10 %, thước đo giám sát – KHÔNG chấm gate):** gộp 5 kỳ £21,703 [CI 95 % −£10,524; £48,043], đáp án (hiện thực) £7,143; CI phủ đáp án và chứa 0; chỉ 72 khách đối chứng. Theo kỳ, CI không phủ đáp án ở 1/5 kỳ. *Nguồn:* `s8_oot_dr_value.csv`; hình `fig_s8_oot.png`.


### IV.8 Kiểm định ngoài trên RCT thật – Hillstrom (s9)

| Kết quả | Mục | CLO | Nguồn |
|---|---|---|---|
| Learner thắng (Qini CV, train): CausalForestDML | IV/V | CLO3 | `s9_acceptance.csv; s9_qini_test.csv; s9_policy_value_test.csv; fig_s9_*.png` |
| Qini conversion/test: số email có CI hoàn toàn > 0: 0/2 | IV/V | CLO3 | `s9_acceptance.csv; s9_qini_test.csv; s9_policy_value_test.csv; fig_s9_*.png` |
| Qini chi tiêu/test: số email có CI hoàn toàn > 0: 0/2 | IV/V | CLO3 | `s9_acceptance.csv; s9_qini_test.csv; s9_policy_value_test.csv; fig_s9_*.png` |
| MILP hơn gửi ngẫu nhiên 30% (CI 95% hoàn toàn > 0)?: Không (CI chứa 0 hoặc < 0) | IV/V | CLO3 | `s9_acceptance.csv; s9_qini_test.csv; s9_policy_value_test.csv; fig_s9_*.png` |
| MILP hơn không gửi (CI 95% hoàn toàn > 0)?: Không (CI chứa 0 hoặc < 0) | IV/V | CLO3 | `s9_acceptance.csv; s9_qini_test.csv; s9_policy_value_test.csv; fig_s9_*.png` |
| Giới hạn: kiểm định phương pháp trên RCT thật, **không** phải toàn pipeline Online Retail | V | CLO3 | `s9_acceptance.csv` |

## V. Thảo luận

### V.1 Khuyết nhãn có chọn lọc (OOT)

Router chỉ quan sát kết quả của khách nó chọn gửi. Ước lượng tác động chương trình trên nhóm được chọn (thật = 8.61 £/khách):

| phuong_phap | uoc_luong_GBP_moi_khach_duoc_chon | do_chech | n_doi_chung |
|---|---|---|---|
| So thô: được gửi − không được chọn (KHÔNG đối chứng) | 33.5 | 24.89 | 72 |
| Đối chứng ngẫu nhiên: được gửi − đối chứng | 32.13 | 23.52 | 72 |
| DR/AIPW (có đối chứng + IPW) | 25.92 | 17.31 | 72 |

Hiệu chỉnh μ̂0 học từ dữ liệu do Router sinh ra (nhóm được chọn): có vs không có nhóm đối chứng + IPW:

| phuong_phap | p_du_bao_TB | Y0_that_TB | do_chech | ece_vs_Y0_that |
|---|---|---|---|---|
| Current (Platt GĐ2, không học lại) | 0.1694 | 0.2234 | -0.054 | 0.0629 |
| Challenger KHÔNG đối chứng (chỉ a0 không được chọn) | 0.2129 | 0.2234 | -0.0105 | 0.0438 |
| Challenger CÓ đối chứng + IPW (đối chứng × 1/0.1) | 0.2138 | 0.2234 | -0.0096 | 0.044 |

*Nguồn:* `s8_oot_selective_effect.csv`, `s8_oot_selective_calibration.csv`. Cả ba ước lượng tác động đều **chệch dương** so với đáp án (đáp án chỉ dùng để đo độ chệch). Học lại hiệu chỉnh từ dữ liệu Router giảm độ chệch của p̂ trên nhóm được chọn từ -0.054 (Current) xuống ≈ -0.011; thêm nhóm đối chứng + IPW chỉ cải thiện thêm ≈ 0.001 – **không đáng kể** với chỉ 72 khách đối chứng (phương sai lớn).

### V.2 Độ trễ hạ tầng và chi phí bộ giải

| Kết quả | Mục | CLO | Nguồn |
|---|---|---|---|
| Locked (G6): P95 đầu–cuối 1.052 s (ngưỡng 30 s); khâu chấm điểm mô hình chiếm P50 0.594 s / 0.843 s; MILP P95 0.179 s | V | CLO4, CLO5 | `s8_locked_g6_latency_percentiles.csv` |
| Độ trễ nhãn (hạ tầng dữ liệu) chi phối giám sát: Brier/ECE chỉ có sau tối thiểu 30 ngày, thực tế 43.2 ngày (trung bình) | V | CLO4, CLO5 | `s8_oot_label_delay_by_metric.csv` |
| Chi phí bộ giải theo quy mô: n = 50,000 khách → HiGHS 14.31 s (gap 0), Greedy 0.085 s; Greedy kém MILP trên V̂ 3.8%–17.1% (các cỡ n) | V | CLO4, CLO5 | `s6_scaling.csv; fig_s6_scaling.png` |
| MILP một t0 (~3.6k khách): 0.150 s trung bình | V | CLO4, CLO5 | `s6_main_summary.csv` |
| Hillstrom: chỉ có 1 ràng buộc đếm ⇒ MILP = Greedy (ma trận unimodular) | V | CLO4, CLO5 | `s9_milp_vs_greedy.csv` |

### V.3 Giả định nhận dạng nhân quả

| Kết quả | Mục | CLO | Nguồn |
|---|---|---|---|
| Không nhiễu chưa quan sát (ignorability): chính sách lịch sử chỉ phụ thuộc đặc trưng đã quan sát (20 % ngẫu nhiên + 80 % luật RFM) – đúng **theo thiết kế bán tổng hợp**, không kiểm định được trên dữ liệu thật | V | CLO3 | `CLAUDE.md §2, §3.2; s1b_intervention.py` |
| Chồng lấp (overlap): propensity ghi log nhỏ nhất 0.139 trên train; tỷ lệ propensity ước lượng < 0.05 bằng 0 | V | CLO3 | `s4_propensity_compare.csv; fig_s4_propensity_overlap.png` |
| Kiểm tra mô hình hóa propensity: ước lượng vs ghi log sai lệch tuyệt đối trung bình ≤ 0.026 | V | CLO3 | `s4_propensity_compare.csv` |
| Placebo (xáo A): ATE ≈ 0 ở 6/6 phép thử | V | CLO3 | `s4_placebo.csv` |
| SUTVA / không lan truyền: không có hiệu ứng mang sang giữa các t0, mỗi (khách, t0) độc lập – **giới hạn đã biết**; giỏ của người mới chuyển đổi là mô phỏng; biên lợi nhuận 30 % là giả định | V | CLO3 | `CLAUDE.md §6` |
| Ở vận hành, Router gần tất định: chỉ nhóm đối chứng 10 % trong tập ngẫu nhiên nhận dạng được tác động (propensity a0 = 0 cho khách bù độ phủ) | V | CLO3 | `CLAUDE.md §6b; s8_oot_dr_value.csv` |

### V.4 Quan hệ phi tuyến giữa sai số dự báo và regret

Bơm nhiễu cộng vào V̂ (σ × sd V̂): MSE tăng gấp ≈ 4.7 lần nhưng regret chỉ tăng chậm (và không đơn điệu ở mức nhiễu nhỏ) – quan hệ **phi tuyến**:

| σ | MSE_hien_thuc | regret_cn_hien_thuc |
|---|---|---|
| 0.0 | 2144.7 | 0.9602 |
| 0.25 | 2153.4 | 0.9576 |
| 0.5 | 2175.0 | 0.9736 |
| 1.0 | 2267.4 | 0.9861 |
| 2.0 | 2649.1 | 0.9912 |
| 4.0 | 4128.0 | 1.0374 |
| 8.0 | 10113.9 | 1.0429 |

Co giãn V̂ × k đổi MSE từ 2213 đến 3262 nhưng regret **không đổi** (0.9602) vì thứ hạng không đổi. Giữa các phương pháp: SPO+ có MSE cao nhất (3220) nhưng regret 0.985 thấp hơn PTO LightGBM (MSE 2613, regret 1.148). *Nguồn:* `s5_mse_vs_regret.csv`; hình `fig_s5_mse_vs_regret.png`. (MSE thấp ≠ regret thấp; regret phụ thuộc thứ hạng, không phụ thuộc thang đo.)

### V.5 Hiệu quả rào chắn nghiệp vụ (guardrail) khi dữ liệu bị trôi (τ thật × 0.6 ở OOT)

| kich_ban | chinh_sach | gia_tri_hien_thuc | gia_tri_ky_vong | vi_pham_quy_dinh | HITL_tu_choi_ha_cap | ton_that_tranh_duoc_hien_thuc |
|---|---|---|---|---|---|---|
| không trôi | KHÔNG guardrail: MILP thuần (miền v1) | 14587.0 | 16343.0 | 5 | 0 | 0.0 |
| không trôi | Đề xuất MILP (miền v2.1, trước Router) | 5479.0 | 9921.0 | 0 | 0 | -9108.0 |
| không trôi | CÓ guardrail: MILP qua Router v2.1 | 6611.0 | 11190.0 | 0 | 448 | -7976.0 |
| trôi (τ × 0.6) | KHÔNG guardrail: MILP thuần (miền v1) | 7642.0 | 3988.0 | 5 | 0 | 0.0 |
| trôi (τ × 0.6) | Đề xuất MILP (miền v2.1, trước Router) | -730.0 | 337.0 | 0 | 0 | -8371.0 |
| trôi (τ × 0.6) | CÓ guardrail: MILP qua Router v2.1 | 1776.0 | 3808.0 | 0 | 448 | -5866.0 |

Có guardrail (MILP qua Router) £6,611 / £1,776 (không trôi / trôi) so với không guardrail £14,587 / £7,642: guardrail **làm mất giá trị ở cả hai kịch bản** (−£7,976 / −£5,866), đổi lại không vi phạm quy định bắt buộc (opt-out, frequency cap, hủy đơn): không guardrail vi phạm 5 lần, có guardrail 0 lần. Ở router v2.1, opt-out/cap/hủy đơn nằm trong miền MILP nên không còn "lượt bị chặn cứng" sau khi giải; tác động nằm ở HITL (từ chối/hạ cấp). *Nguồn:* `s8_oot_drift_guardrail.csv`; đáp án trôi: `oracle_oot_drift.parquet` (chỉ đọc trong bước OOT).


## VI. Playbook vận hành

| canh_bao | nguong_canh_bao | nguong_bao_dong | chu_so_huu | hanh_dong | dieu_kien_quay_lui |
|---|---|---|---|---|---|
| PSI đặc trưng (không tính đặc trưng lịch) | ≥ 0.1 | ≥ 0.25 | Data Scientist (MLOps) | Cảnh báo: soát pipeline dữ liệu, so phân phối; Báo động ≥ 3 đặc trưng: chuyển Challenger-2 sang đánh giá | Báo động PSI điểm μ̂0 hoặc V̂ ≥ 0.25 hai kỳ liên tiếp ⇒ quay lui về Luật cũ qua Router |
| PSI điểm số (μ̂0, Ŝ, τ̂, V̂) | ≥ 0.1 | ≥ 0.25 | Data Scientist (MLOps) | Kiểm tra trôi đầu vào; tạm giảm ngân sách B nếu V̂ trôi lên | như trên |
| ECE μ̂0 trên kỳ đã trưởng thành | > 0.03 | ≥ 0.05 | Data Scientist | Báo động: dùng Challenger (Platt theo kỳ) cho kế hoạch ngân sách; giữ Current để ra quyết định cho tới khi duyệt | Chi phí chiết khấu thực tế vượt kế hoạch > 20 % hai kỳ liên tiếp ⇒ chuyển hẳn sang Challenger |
| Action TVD so với tham chiếu | — | ≥ 0.2 | Trưởng nhóm CRM | Rà soát hành động bất thường (vd a2 tăng vọt); duyệt thủ công đợt gửi | TVD ≥ 0.20 kèm PSI báo động ⇒ quay lui Luật cũ qua Router |
| Độ trễ P95 đầu–cuối | > 15 s | > 30 s | MLOps | Solver quá thời gian ⇒ Greedy (dự phòng Router) | Quá ngưỡng 2 kỳ ⇒ chạy Luật cũ qua Router |
| Vi phạm ràng buộc trên quyết định gửi | — | > 0 | Tài chính + MLOps | CHẶN gửi đợt đó; sửa và chạy lại | Bất kỳ vi phạm nào ⇒ dừng gửi tự động (HITL 100 %) |
| Giá trị thực hiện DR (đối chứng 10 %) | CI chứa 0 | CI < 0 | Trưởng nhóm CRM + Tài chính | CI < 0: thu hẹp ngân sách, tăng tỷ lệ đối chứng | CI < 0 hai kỳ trưởng thành liên tiếp ⇒ quay lui Luật cũ qua Router |
| Tải HITL | > 150 ca/kỳ | > 250 ca/kỳ | CSKH | Bổ sung người duyệt / nới hàng đợi | Không duyệt kịp ⇒ ca chưa duyệt về a0 |
| Tỷ lệ bị chặn cứng | tăng > 2× tham chiếu | — | Trưởng nhóm CRM | Soát danh sách opt-out / cap / hủy đơn | — |

*Nguồn:* `s8_oot_playbook.csv` (ngưỡng PSI/ECE/TVD lấy từ `configs/config.yaml` mục `monitoring`; ba ngưỡng cảnh báo P95 > 15 s, tải HITL > 150/250 ca/kỳ, tỷ lệ chặn tăng > 2× là **ngưỡng tự đặt**, không thuộc `monitoring`).


## Phụ: thiết kế thí điểm (cỡ mẫu)

G3 – cỡ mẫu nhánh ngẫu nhiên để đạt power 80 % (Qini giả định = ước lượng calib):

| hanh_dong | qini_gia_dinh | n_cap_can_power80 | boi_so_locked |
|---|---|---|---|
| a1 | 0.068 | 3276.707 | 2.531 |
| a2 | 0.004 | 987357.156 | 762.615 |
| a3 | 0.021 | 40422.181 | 31.221 |

DR – số kỳ cần để CI giá trị chương trình loại trừ 0 (thiết kế Router):

| ty_le_doi_chung | kich_ban_hieu_ung | se_mot_ky | so_ky_can_power80 |
|---|---|---|---|
| 0.1 | Δ = V̂ kế hoạch (£2,332/kỳ, lạc quan) | 12800.2 | 236.5 |
| 0.1 | Δ = ước lượng DR valid+calib (£1,255/kỳ) | 12800.2 | 816.3 |
| 0.5 | Δ = V̂ kế hoạch (£2,332/kỳ, lạc quan) | 4266.7 | 26.3 |
| 0.5 | Δ = ước lượng DR valid+calib (£1,255/kỳ) | 4266.7 | 90.7 |

*Nguồn:* `s8_pilot_g3.csv`, `s8_pilot_dr.csv`, `fig_s8_pilot_sample_size.png`.


## Công khai và sai lệch so với đăng ký trước

**(i) Thông tin tổng hợp của locked/oot đã có trước khi mở locked (đính chính theo kiểm toán 13/11, mục B3).** Locked không "mù hoàn toàn". Có bốn nguồn, ba nguồn do chính code của repo sinh ra từ GĐ1/GĐ1b/GĐ2, một nguồn là phép tính tay: (1) `s1_data.py` ghi `s1_split_summary.csv` với tỷ lệ mua lại và giá trị giỏ trung bình của **locked (0.1873) và oot (0.2817)**, và vẽ `fig_s1_base_rate_by_t0.png` cho mọi t0; (2) `s1b_intervention.py` ghi `s1b_true_ate_by_split.csv`: **ATE thật và lợi nhuận gia tăng thật trung bình của locked và oot, lấy từ đáp án** (locked: τ a1 0.0438, a2 0.0906, a3 0.0626); (3) `s2_predict.py` chấm μ̂0, Ŝ cho **mọi** dòng kể cả locked/oot (`pred_s2.parquet`; chỉ dự báo, không dùng nhãn, đúng đặc tả "lưu cho MỌI dòng"); (4) tỷ lệ mua lại a0 của locked 0.187 và oot 0.289 trong CLAUDE.md §6 ("phát hiện ở GĐ2") **không do script nào trong repo sinh ra**: kiểm toán tính lại được đúng 0.1870 / 0.2893 từ `logged.parquet`, nên đó là một phép tính tay ở GĐ2, và theo lời người dùng Claude (trong chat) đã xem số liệu tổng hợp trên locked ở GĐ2. Repo không có nhật ký phiên chat nên *không xác minh được chính xác số liệu nào*. Hệ quả: (a) đăng ký trước ở 6b ("Platt có thể không hợp locked/oot") được viết khi đã biết xu hướng mức nền; (b) kết quả G2 trên locked không phải kiểm định *mù hoàn toàn* về mức nền. **Không thấy dấu hiệu dùng để chỉnh:** `release_gates` trùng giữa config hiện tại, bản nháp đóng băng và bản chính thức; `configs/config.yaml` khớp hash đóng băng; mô hình được băm; không code huấn luyện/chọn mô hình nào đọc các bảng trên. Nhãn cấp dòng, Qini và giá trị chính sách của locked chỉ được tính ở bước 2 (theo các handoff 1a/1b; không kiểm chứng được cho các phiên trước đó). Theo cách đọc nghiêm ngặt, điều này lệch khỏi luật cứng #4. *Nguồn:* `outputs/manifests/audit_2026-11-13.md` (B3), `s1_split_summary.csv`, `s1b_true_ate_by_split.csv`.

**(ii) Lỗi bảng tổng hợp gate.** `s8_locked_gate_summary.csv` in toàn `n/a` vì mã so sánh `v is True` với `numpy.bool_` (ở dry-run cột `dat` kiểu object nên hiển thị đúng; ở locked cột kiểu bool). Chỉ ảnh hưởng bảng tổng hợp; `s8_locked_gates.csv` (cột `dat`) đúng. Không sửa được vì `src/` đã đóng băng và locked đã mở. Bảng đúng: `s8_locked_gate_summary_corrected.csv`, lập từ `s8_locked_gates.csv`. Diễn tập đường locked không bắt được lỗi này vì không kiểm nội dung bảng tổng hợp.

**(iii) G3 dự báo ≈ 4–7 % nhưng đạt (đính chính theo kiểm toán 13/11, mục B2).** Toàn bộ locked: a2 Qini 0.098 [CI 0.015; 0.161], a3 0.075 [0.010; 0.144] ⇒ 2/3 hành động có CI > 0 (cận dưới chỉ 0.015 và 0.010, sát 0). Cần đọc thận trọng: (1) **không hiệu chỉnh đa kiểm định** (3 hành động, mỗi hành động một CI 95 %); với hiệu chỉnh Bonferroni cho 3 hành động, **a3 trượt** (cận dưới −0.012), chỉ còn a2; (2) **khách thường chỉ đạt 1/3** (a2 0.111; a3 CI [-0.015; 0.138] chứa 0) – a3 chỉ qua ngưỡng nhờ 2 005 dòng entity-holdout; entity-holdout riêng 0/3; (3) **phụ thuộc hạt giống bootstrap**: với seed đăng ký cận dưới a3 = 0.0104 (300 lần lấy lại mẫu); chạy lại 20 seed khác, cận dưới a3 nằm trong [−0.006; 0.013], **dương ở 15/20 seed**; với 5 000 lần lấy lại mẫu a3 [0.0035; 0.144], a2 [0.024; 0.173] ⇒ vẫn 2/3; (4) **mâu thuẫn bề ngoài với bảng thí điểm**: bảng thí điểm lấy Qini a2 = 0.004 (ước lượng calib) và cần ≈ 763× cỡ locked để đạt power 80 %, trong khi locked cho 0.098. Câu cũ "dữ liệu hiện có không phân biệt được hai khả năng" được thay bằng kết quả hậu kiểm sau, **dùng đáp án (gán lại nhánh ngẫu nhiên 1 000 lần bằng kết quả tiềm năng), không ảnh hưởng gate**:

| Qini kỳ vọng của bộ xếp hạng | a1 | a2 | a3 |
|---|---|---|---|
| valid | 0.031 | 0.068 | 0.070 |
| calib | 0.021 | 0.062 | 0.064 |
| locked | 0.021 | 0.059 | 0.065 |

Độ lệch chuẩn do gán ngẫu nhiên ≈ 0.038–0.040 ở cả ba tập ⇒ chất lượng xếp hạng **ổn định qua các mùa**; không cần giả thuyết hiệu ứng khác nhau theo mùa. Qini a2 = 0.004 trên calib là một lần rút thấp (phân vị 7 %); 0.098 trên locked là một lần rút cao (phân vị 86 %). Dự báo "P(G3) ≈ 4–7 %" thấp vì coi ước lượng điểm của calib là hiệu ứng thật; với giá trị kỳ vọng trên và SE ≈ 0.038, xác suất đạt G3 theo xấp xỉ chuẩn độc lập là **≈ 16 %**. G3 vẫn ĐẠT theo định nghĩa đã đăng ký (toàn bộ locked) nhưng là kết cục thiểu số, **không dùng G3 làm bằng chứng cho luận điểm chính**, và **không diễn giải 0.098 như độ lớn hiệu ứng thật**. *Nguồn:* `audit_2026-11-13.md` (B2; số liệu hậu kiểm từ `/tmp/audit_qini_expected.csv` của kiểm toán, không nằm trong repo), `s8_locked_g3_qini.csv`, `s4_power_g3*.csv`.

**(iv) Các chỗ tự quyết định** (chi tiết: `outputs/manifests/s8_handoff_1a.md`, `s8_handoff_1b.md`). *Bước 1a:* G4b chấm trên MILP qua Router v2.1, MILP thuần là G4b-phụ; G6 gồm khâu mã lý do SHAP top-3; quần thể chấm chính thức = toàn bộ; G1 là bộ kiểm tra lắp ghép (từng trượt ở dry-run đầu do lỗi mã kiểm tra propensity, đã sửa) – bốn điểm này đã được người dùng đồng ý và ghi vào 6b ("Làm rõ trước khi mở locked"); G6 dùng P95 gộp mọi lượt; frequency cap bắt đầu rỗng ở tập chấm; G2 trên calib ở dry-run là trong mẫu; không làm phân rã đáp án B/C; phạm vi băm dữ liệu loại `decisions_s8_*`; lệch phiên bản thư viện bị coi là lệch đóng băng. *Bước 1b:* cửa sổ Challenger = 3 kỳ gần nhất đã trưởng thành (chỉ ghi trong docstring `s8b_oot.py`, không vào 6b); "phân định MILP với Current" hiểu theo hai nghĩa (MILP − luật cũ dưới log 20 % ngẫu nhiên; Current so với không làm gì dưới đối chứng của Router); MILP qua Router vs luật cũ qua Router không nhận dạng được từ log Router (chưa tính); phản hồi sau triển khai lấy từ đáp án trong hàm `oracle_*`; "không guardrail" = MILP thuần miền v1; diễn tập trôi dùng cặp đáp án sinh trong bộ nhớ; ba ngưỡng playbook tự đặt; π của khách đối chứng = a1 nếu HITL hạ cấp (bỏ qua ảnh hưởng của bước sửa trần a2). *Bước đóng gói:* cột CLO được điền sau đó theo bảng chuẩn đầu ra trong đề (gán theo nội dung từng kết quả, không do script sinh); repo không đưa bản sao Hillstrom (xem README).

**(v) Test sửa sau đóng băng** (thư mục `tests/` nằm ngoài phạm vi băm nên `verify_freeze` không bao quát): `tests/test_s8b.py` sửa lúc 2026-11-12 14:39:52 UTC (freeze `s8_freeze` lúc 14:31:07 UTC; khóa locked lúc 14:40:19 UTC) – thêm `test_real_oot_path_preflight`, diễn tập nhánh OOT "real" bằng dữ liệu thay thế *trước* khi mở locked. `tests/test_s8.py` sửa lúc 2026-11-12 14:54:40 UTC (sau khi mở locked) – hai test (`test_dry_run_does_not_touch_locked_oot`, `test_locked_path_rehearsal`) từng khẳng định "khóa chưa tồn tại", đổi thành "khóa và `s8_locked_gates.csv` thật không bị thay đổi" (ý định kiểm tra không đổi). Phạm vi băm không thay đổi (`verify_freeze` = 0 lệch sau các lần sửa test); chỉ phát sinh thêm các file đầu ra `decisions_s8_*` và bảng/hình mới. Lưu ý nhỏ: `tests/test_s8b.py` dòng `assert not S8.LOCK_PATH.exists() or True` là khẳng định vô hiệu (luôn đúng), không ảnh hưởng kết quả. Hash hiện tại của hai file nằm trong `s8_sha256_appendix.csv`.


## Phụ lục tái lập

- `outputs/tables/s8_sha256_appendix.csv` – SHA-256 từng file trong phạm vi băm, hash tổng hợp, nội dung file khóa, ghi chú sửa sau băng.
- `outputs/manifests/s8_freeze_manifest.json`, `requirements_lock.txt` (Python 3.14.7), `outputs/tables/s8_compat_matrix.csv` (GĐ1/GĐ1b chạy Linux, Python 3.13.18; xem B7).
- `outputs/tables/s8_checklist_1a.csv`, `outputs/manifests/s8_handoff_1a.md`, `s8_handoff_1b.md`.

