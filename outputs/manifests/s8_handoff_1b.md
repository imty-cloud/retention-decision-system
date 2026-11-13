# Handoff – GĐ8 bước 1b (phát lại OOT, giám sát, thí điểm, đóng băng) · 2026-11-12

## Trạng thái: bước 1b XONG – ĐÃ ĐÓNG BĂNG CHÍNH THỨC
- `outputs/manifests/s8_freeze_manifest.json` (stage `s8_freeze`): 28 file mã/mô hình/config + 15 file dữ liệu (băm theo byte).
  - aggregate_sha256 (mã/mô hình/config/CLAUDE.md) = `1d078744042c80b978eaae0e49b8e63c81bae18809ae090ad6734f13312ebd1a`
  - data_aggregate_sha256 = `ccdbd1c36585d5cd134cd0458261019d62f4870cdcba3a331c289b6ec6627ac5`
  - Phạm vi gồm `src/s8b_oot.py` và `data/processed/oracle_oot_drift.parquet`. `verify_freeze()` = **0 lệch** ngay sau khi đóng băng.
- File khóa `s8_locked_opened.lock`: **CHƯA tồn tại**. Chưa chạy `--open-locked`, chưa chạy `--oot`. Chưa đọc locked/oot thật.
- **Từ nay:** không sửa src/ configs/ models/ CLAUDE.md; không ghi parquet vào data/processed (trừ `decisions_s8_*`); không pip install.
  Nếu không, `--open-locked` / `--oot` sẽ từ chối.

## Đã làm
| # | Việc | Bằng chứng |
|---|---|---|
| 1 | `--oot` có rào: `read_rows` chỉ cho đọc oot khi `oot_open` bật VÀ file khóa tồn tại; `open_oot` bật cờ chỉ khi khóa tồn tại VÀ `verify_freeze` = 0 lệch (từ chối trước khi đọc). `--oot-rehearsal`: valid = tham chiếu, calib đóng vai OOT, nhãn THỬ | `src/s8b_oot.py::open_oot`, `s8_mlops.read_rows`; tests `test_oot_refused_without_lock`, `test_cli_oot_refuses_now`, `test_oot_refused_when_freeze_mismatch`, `test_oot_opens_only_with_lock_and_matching_freeze`, `test_oot_rehearsal_does_not_touch_locked_oot` |
| 2a | Độ trễ từng khâu, 30 lượt/kỳ, P50/P95/P99 theo kỳ và gộp | `s8_oot_rehearsal_latency.csv` (P95 gộp 1.70 s) |
| 2b | PSI đặc trưng + điểm số so với train; Brier/ECE chỉ khi nhãn trưởng thành; Action TVD so với tham chiếu | `..._psi.csv`, `..._calibration_matured.csv`, `..._action_tvd.csv` |
| 2c | Bảng độ trễ nhãn theo ngày phát lại + theo chỉ số | `..._label_delay_by_date.csv`, `..._label_delay_by_metric.csv` |
| 2d | DR/AIPW trên nhóm đối chứng của Router, CI bootstrap cụm khách, đặt cạnh đáp án | `..._dr_value.csv` |
| 2e | Current vs Challenger (Platt, 3 kỳ trưởng thành gần nhất) + Challenger-2 (ở chế độ thật chỉ chạy khi đã có `s8_locked_gates.csv`) | `..._calibration_matured.csv` |
| 2f | `oracle_oot_drift.parquet` sinh bằng s1b (scale 0.6, cùng số ngẫu nhiên; ngoài oot hai lần sinh giống hệt), chỉ chứa dòng oot, chưa đọc lại. 16/16 file hiện có không đổi hash. Code so sánh CÓ / KHÔNG guardrail | `s8_drift_generation_hashes.csv`, `s8_drift_manifest.json`, `..._drift_guardrail.csv`, test `test_drift_file_generated_without_changing_existing` |
| 2g | Khuyết nhãn có chọn lọc: ước lượng tác động (so thô / đối chứng / DR) và Challenger hiệu chỉnh (không đối chứng vs đối chứng + IPW) – độ chệch so với đáp án | `..._selective_effect.csv`, `..._selective_calibration.csv` |
| 2h | Playbook | `..._playbook.csv` |
| 3 | Thiết kế thí điểm | `s8_pilot_g3.csv`, `s8_pilot_dr.csv`, `s8_pilot_dr_router_valid_calib.csv`, `fig_s8_pilot_sample_size.png` |
| 4 | pytest toàn bộ **76 passed, 0 failed** (thêm `tests/test_s8b.py`, 12 test) | |
| 5 | `--freeze` → verify 0 lệch, khóa chưa tồn tại | trên |

## Kết quả diễn tập chính (THỬ)
- Độ trễ P95 gộp 1.70 s (theo kỳ 1.52–2.22 s), dưới xa 30 s.
- PSI (gộp): 20 ổn định, 12 cảnh báo, 6 báo động. Báo động gồm t0_month, days_to_xmas (trôi theo thiết kế), tenure_days 1.17, value_trend_90_365 0.35,
  recency_days 0.29 và **điểm μ̂0 hiệu chỉnh 0.39**.
- ECE (dòng a0, khi nhãn trưởng thành): Current 0.039 / 0.052 / 0.026; Challenger 0.049 / 0.063 / 0.031 ⇒ **Challenger không tốt hơn Current trong diễn tập**
  (cửa sổ 3 kỳ của valid, tháng 2–3, không hợp hơn calib).
- Độ trễ nhãn: chỉ số cần nhãn trễ tối thiểu 30 ngày, thực tế 44 ngày (trung bình) / 58 ngày (tối đa) theo lịch 2 tuần.
- DR chương trình Router (gộp 3 kỳ): £6 229 [−41 880; 41 903] so với đáp án £13 540. CI phủ đáp án, nhưng rất rộng (đúng dự báo 6b); kỳ 2011-05-09 CI không phủ.
- Trôi 0.6: CÓ guardrail £5 068 vs KHÔNG guardrail £7 837 (hiện thực); không trôi £9 600 vs £12 845. Guardrail làm GIẢM giá trị ở cả hai kịch bản,
  nhưng mức giảm thu hẹp khi trôi (kỳ vọng: −£3 686 → −£1 286). Bù lại, KHÔNG guardrail vi phạm quy định ở 3/3 kỳ (gửi cho khách opt-out / đang cap / hủy đơn cao);
  CÓ guardrail 0 vi phạm. Lượt bị chặn cứng = 0, vì ở v2.1 opt-out / cap / hủy đơn đã nằm trong miền MILP; HITL từ chối hoặc hạ cấp 184 lượt.
- Khuyết nhãn có chọn lọc: cả ba ước lượng tác động đều chệch âm so với đáp án (£23.2 / khách được chọn): so thô −13.6, đối chứng −27.3 (chỉ 56 đối chứng),
  DR −12.5. Hiệu chỉnh trên nhóm được chọn: Current chệch +0.027, Challenger không đối chứng +0.017, Challenger có đối chứng + IPW +0.015.

## Thiết kế thí điểm
- G3 (S1 – calib riêng): a1 cần 3 277 dòng cặp ≈ 2.53× locked (nhánh ngẫu nhiên ≈ 51 % của locked, hoặc ≈ 7.6 kỳ ở mức 20 %); a2 cần 762×; a3 cần 31× ⇒
  ở cỡ locked chỉ a1 khả thi; G3 (≥ 2 hành động) không khả thi. Khớp `s4_power_g3.csv` (lệch 2.5e-5, chỉ do làm tròn z).
- DR, thiết kế ghi log 20 % (s5b, MILP − luật cũ): SE một kỳ £14.7k (valid) / £26.0k (calib) ⇒ với Δ = £4 500/kỳ cần 84–263 kỳ.
- DR, thiết kế Router (Current so với không làm gì): SE một kỳ £12.8k khi đối chứng 10 % ⇒ 237 kỳ (Δ = V̂ kế hoạch £2 332) / 816 kỳ (Δ = DR £1 255);
  đối chứng 50 % ⇒ 26 / 91 kỳ (xấp xỉ theo (1−p)/p).

## Quyết định đã tự đưa ra
1. **Cửa sổ Challenger = 3 kỳ gần nhất đã trưởng thành** (bằng cỡ calib của Current). Kho nhãn thật = calib + locked + oot. Định nghĩa nằm trong docstring
   `s8b_oot.py` (đã đóng băng), KHÔNG ghi vào 6b vì chưa được yêu cầu.
2. **"Phân định MILP với Current" được hiểu theo hai nghĩa:** (A) MILP − luật cũ dưới thiết kế ghi log 20 % (s5b); (B) chương trình Current so với không làm gì,
   dưới thiết kế đối chứng của Router. MILP qua Router vs Luật cũ qua Router KHÔNG nhận dạng được từ log Router (không chồng lấp), cần thí nghiệm chia luồng;
   phần này chưa tính.
3. **Phản hồi sau triển khai** (kết quả dưới hành động Router đã làm) lấy từ đáp án trong hàm `oracle_feedback`. Đó là mô phỏng thế giới; Challenger khuyết nhãn
   học từ phản hồi này, nhưng Current không bị chỉnh.
4. **"KHÔNG guardrail" = MILP thuần trên miền v1.** Thêm "Đề xuất MILP miền v2.1" làm mốc trung gian.
5. **Diễn tập trôi** dùng cặp đáp án sinh TRONG BỘ NHỚ trên snapshot train+valid+calib (calib gắn nhãn oot). Số ngẫu nhiên khác đáp án chính thức, chỉ để chạy thử code.
6. **Playbook có 3 ngưỡng tự đặt, chỉ ở mức cảnh báo** (P95 > 15 s, HITL > 150 / 250 ca/kỳ, tỷ lệ bị chặn tăng > 2×). Ngưỡng config không đổi.
7. Ở chế độ thật, Challenger-2 tự bỏ qua nếu chưa có `outputs/tables/s8_locked_gates.csv`.
8. π của khách đối chứng = a1 nếu HITL đã hạ xuống a1, ngược lại là hành động đề xuất. Bỏ qua khả năng bước sửa trần a2 sẽ đổi π (xấp xỉ).

## Việc tiếp theo (bước 2)
1. `python src/s8_mlops.py --open-locked` (một lần) → `s8_locked_*`.
2. `python src/s8b_oot.py --oot` → `s8_oot_*` (Challenger-2 chạy vì đã có gate).
3. Notebook + `summary_for_essay.md` (ghi ra outputs/notebooks; không sửa src/).

---
# BƯỚC 2 – ĐÃ CHẠY (2026-11-12): `--open-locked` rồi `--oot`
- Trước khi chạy: `verify_freeze` = 0 lệch, khóa chưa tồn tại; thêm test `test_real_oot_path_preflight` (trong tests/, ngoài phạm vi đóng băng)
  diễn tập nhánh vai "real" của phát lại OOT trên dữ liệu thay thế → qua.
- `--open-locked`: chạy trọn vẹn (89 s), khóa `s8_locked_opened.lock` ĐÃ GHI, không có `s8_locked_error.log`. **Locked không mở lại được nữa.**
- `--oot`: chạy trọn vẹn (192 s; 21 693 dòng, 5 kỳ), Challenger-2 có chạy. Sau cả hai lệnh `verify_freeze` vẫn = 0 lệch.

## Gate CHÍNH THỨC trên locked (toàn bộ: 12 947 dòng, 3 kỳ)
| Gate | Kết quả | Giá trị | Dự báo 6b |
|---|---|---|---|
| G1 | ĐẠT | 7/7 | — |
| G2 | ĐẠT | ECE 0.0201 ≤ 0.03; Brier 0.1127 ≤ 0.1213 | "có thể không hợp" – thực tế hợp |
| G3 | ĐẠT | 2/3: a2 0.098 [0.015, 0.161], a3 0.075 [0.010, 0.144]; a1 0.035 [−0.037, 0.098] | P(đạt) ≈ 4–7 % ⇒ **dự báo SAI** |
| G4a | ĐẠT (sát ngưỡng) | £8 785 vs £7 824, chênh £961 ≥ £782; phụ: vs £3 739 | khả thi – đúng |
| G4b | TRƯỢT | regret A 0.894 > 0.35 (MILP thuần 0.898) | gần như chắc trượt – đúng |
| G5 | ĐẠT | 0 vi phạm | đúng |
| G6 | ĐẠT | P95 1.05 s | đúng |
| G7 | ĐẠT | UK +18.8 %, non-UK +4.7 % | đúng |
⇒ **G4 TRƯỢT (do G4b) ⇒ không đạt đủ cổng phát hành.** Khách thường: G3 trượt (1/3). Entity-holdout (mô tả): G2 trượt (ECE 0.049), G3 0/3,
G4a trượt (−£167 vs £384), G7 trượt (non-UK).

## LỖI ĐÃ BIẾT (không sửa được vì code đã đóng băng và locked đã mở)
`outputs/tables/s8_locked_gate_summary.csv` in toàn "n/a": `gate_table` so sánh `v is True` với numpy.bool_ (ở dry-run cột `dat` có None nên là kiểu object và
hiện đúng; ở locked cột thành kiểu bool). Chỉ ảnh hưởng bảng tổng hợp; `s8_locked_gates.csv` (cột `dat`) đúng. Bảng tổng hợp lập lại từ bảng chi tiết:
**`outputs/tables/s8_locked_gate_summary_corrected.csv`**. Diễn tập đường locked không bắt được lỗi này vì không kiểm nội dung bảng tổng hợp.

## Phát lại OOT (5 kỳ 2011-09-12 → 2011-11-07)
- Độ trễ P95 gộp 1.32 s (150 lượt). Action TVD 0.011–0.015 (< 0.20).
- PSI gộp: 18 ổn định / 12 cảnh báo / 8 báo động; điểm μ̂0 hiệu chỉnh 0.88, V̂_a2 0.29, tenure_days 1.86.
- ECE Current tăng dần theo mùa: 0.032 → 0.036 → 0.057 → 0.076 → 0.086 (vượt ngưỡng báo động 0.05 từ kỳ 10-10; tỷ lệ mua 0.245 → 0.327, p̂ ≈ 0.23–0.24).
  Challenger: 0.030 / 0.038 / 0.054 / 0.067 / 0.066. Challenger-2: 0.026 / 0.039 / 0.042 / 0.043 / 0.060. Nhãn trễ trung bình 43 ngày (tối đa 58).
- DR giá trị chương trình (gộp): £21 703 [−10 524; 48 043] vs đáp án £7 143 – CI phủ đáp án nhưng chứa 0 (72 khách đối chứng); kỳ 10-24 CI không phủ.
- Trôi: CÓ guardrail £6 611 (không trôi) / £1 776 (trôi) vs KHÔNG guardrail £14 587 / £7 642 ⇒ guardrail làm mất £7 976 / £5 866;
  KHÔNG guardrail vi phạm quy định ở 5/5 kỳ, CÓ guardrail 0; HITL từ chối/hạ cấp 448 lượt.
- Khuyết nhãn: cả ba ước lượng tác động chệch DƯƠNG (thật £8.61/khách; so thô +24.9, đối chứng +23.5, DR +17.3). Hiệu chỉnh nhóm được chọn:
  Current −0.054, Challenger không đối chứng −0.011, có đối chứng + IPW −0.010.

## Còn lại
Notebook `PTDLNC_full_pipeline.ipynb`, `outputs/tables/summary_for_essay.md`. Không chỉnh mô hình/ngưỡng sau khi đã xem locked (luật cứng #4).
