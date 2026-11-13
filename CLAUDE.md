# CLAUDE.md – Tiểu luận PTDLNC: Hệ thống ra quyết định giữ chân khách hàng đa kênh

> Đọc toàn bộ file này trước khi làm bất cứ việc gì. Mọi tham số nằm trong `configs/config.yaml`.
> Viết code, chú thích và nhãn biểu đồ bằng **tiếng Việt**. Trả lời người dùng bằng tiếng Việt.

## 0. Đề tài và đề bài

**Tên đề tài:** Hệ thống hỗ trợ ra quyết định khép kín trong phân bổ ngân sách khuyến mãi và điều phối chiến dịch
giữ chân khách hàng đa kênh trên nền tảng Thương mại điện tử.

**Học phần:** Phân tích dữ liệu nâng cao (UEL, TS. Nguyễn Thôn Dã). Hạn: 29/11/2026.
Đề yêu cầu chu trình 8 giai đoạn ≙ 8 chương: (1) thiết kế bài toán & dữ liệu, chống rò rỉ → (2) dự báo + hiệu chỉnh
xác suất → (3) giải thích & độ tin cậy → (4) nhân quả, ATE/CATE/uplift → (5) Predict-Then-Optimize & SPO+ →
(6) MILP → (7) Action Router, guardrails, HITL → (8) triển khai, đo lường thực chứng, giám sát MLOps.
Tiêu chí chấm: minh bạch, kiểm chứng thực chứng, giảm **Regret**, kiểm soát rủi ro, tối ưu nguồn lực hữu hạn.

## 1. Bài toán quyết định (đã chốt – không tự ý đổi)

| Thành phần | Thiết kế |
|---|---|
| Chủ thể quyết định | Trưởng nhóm CRM; Tài chính duyệt ngân sách; CSKH xử lý HITL |
| Đơn vị phân tích | khách hàng × t0 (thứ Hai, 2 tuần/lần, 38 kỳ: 2010-06-07 → 2011-11-07) |
| Chặn dữ liệu | đặc trưng chỉ dùng sự kiện < cutoff = t0 − 1 ngày |
| Kết quả | mua lại trong [t0, t0+30 ngày); độ trễ nhãn 30 ngày |
| Hành động | a0 không làm gì · a1 email voucher 10% (trần £30) · a2 email voucher 20% (trần £60) · a3 account manager gọi + voucher 10% (£3/cuộc) |
| Hàm giá trị | `V[i,a] = m·Ŝ_i·τ̂_a(x_i) − min(d_a·Ŝ_i, cap_a)·(μ̂0(x_i)+τ̂_a(x_i)) − contact_a` (gia tăng so với a0; V[i,a0]=0) |
| Ràng buộc | mỗi khách đúng 1 hành động; Σ cap_a·x ≤ B (mệnh giá phát hành, **cố định** ⇒ SPO+ hợp lệ); Σ x_a3 ≤ K; a2 ≤ 25% số voucher; độ phủ tối thiểu theo phân khúc; x=0 nếu guardrail chặn |

Ý quan trọng phải giữ trong toàn bài:
- Chi phí chiết khấu trả cho **cả người đằng nào cũng mua** (số hạng μ̂0+τ̂) ⇒ hiệu chỉnh xác suất ảnh hưởng trực tiếp đến chi phí và ngân sách.
- Ngân sách tính theo **mệnh giá phát hành** (không phụ thuộc dự báo) ⇒ miền khả thi cố định ⇒ đúng giả định SPO+.

## 2. Dữ liệu và trạng thái hiện tại

**ĐÃ XONG (không chạy lại trừ khi được yêu cầu):**
- `src/s1_data.py` – GĐ1: làm sạch Online Retail II, snapshot point-in-time, nhãn, chia tập + embargo + entity holdout,
  data contract (pandera), 6 kiểm thử rò rỉ (đều PASS), SHA-256 manifest.
- `src/s1b_intervention.py` – GĐ1b: lớp can thiệp bán tổng hợp. Y(0) = hành vi THẬT; Y(a) sinh từ cơ chế đã biết;
  chính sách lịch sử = 20% ngẫu nhiên + 80% luật RFM (gây nhiễu); propensity được ghi lại.
- `src/s9_external_hillstrom.py` (+ `tests/test_s9.py`) – s9: kiểm định phương pháp trên RCT thật (Hillstrom), độc lập với
  Online Retail. Kết quả chính (conversion): CATE không hơn ngẫu nhiên có ý nghĩa; kết quả phụ khai báo trước (visit): Qini
  dương có ý nghĩa ở nhánh email nữ; MILP = Greedy vì chỉ có 1 ràng buộc đếm. Không chạy lại.

**GĐ8 bước 1a (chuẩn bị gate) – XONG:** `src/s8_mlops.py` + `tests/test_s8.py` (gồm diễn tập đường locked với calib đổi nhãn, băm byte dữ liệu
trong phạm vi đóng băng); `--dry-run` (calib đóng vai locked, nhãn THỬ) khớp GĐ4/GĐ7; `s8_freeze_draft` ghi lại theo code cuối; các quyết định
đã ghi vào 6b ("Làm rõ trước khi mở locked"). CHƯA chạm locked/oot, CHƯA có file khóa, CHƯA đóng băng chính thức. Tiếp theo: bước 1b →
`--freeze` → `--open-locked`. Chi tiết: **`outputs/manifests/s8_handoff_1a.md`**, checklist `outputs/tables/s8_checklist_1a.csv`.

**GĐ8 bước 1b – XONG, ĐÃ ĐÓNG BĂNG CHÍNH THỨC (`s8_freeze`):** `src/s8b_oot.py` (phát lại OOT có rào: chỉ chạy khi đã có khóa locked VÀ
verify_freeze khớp; `--oot-rehearsal`, `--pilot`, `--make-drift`), `tests/test_s8b.py`, `data/processed/oracle_oot_drift.parquet` (trôi 0.6,
sinh trước khi đóng băng, chưa đọc). Từ đây KHÔNG sửa src/ configs/ models/ CLAUDE.md, không ghi parquet vào data/processed (trừ decisions_s8_*),
không pip install. Mọi ghi chú trạng thái sau đóng băng: **`outputs/manifests/s8_handoff_1b.md`**. Tiếp theo (bước 2): `--open-locked` → `--oot`.

**Tập dữ liệu (`split` trong `logged.parquet`):**

| split | t0 | Dùng cho | Quy tắc |
|---|---|---|---|
| train | 2010-06-07 → 2010-12-20 (15 kỳ) | huấn luyện | GroupKFold theo `customer_id` khi CV |
| valid | 2011-01-31 → 2011-03-14 (4 kỳ) | chọn mô hình/siêu tham số | |
| calib | 2011-04-25 → 2011-05-23 (3 kỳ) | CHỈ khớp bộ hiệu chỉnh / conformal | |
| locked | 2011-07-04 → 2011-08-01 (3 kỳ) | **mở đúng 1 lần ở GĐ8** sau khi đóng băng | báo cáo riêng khách thường và `is_entity_holdout==1` |
| oot | 2011-09-12 → 2011-11-07 (5 kỳ) | phát lại vận hành ở GĐ8 | mùa Q4, trôi thật |
| embargo / excluded_entity_holdout | – | **không dùng** | |

**File dữ liệu:**
- `data/processed/logged.parquet` – DỮ LIỆU MÔ HÌNH ĐƯỢC THẤY: đặc trưng + `A_logged`, `propensity_a0..a3`,
  `propensity_logged`, `is_random_arm`, `y_obs`, `value_obs`, `discount_cost_obs`, `profit_obs`.
- `data/processed/oracle.parquet` – ĐÁP ÁN: `y_a*`, `profit_a*`, `uplift_profit_a*`, `tau_cond_a*`, `basket_value_S`.
- `data/processed/snapshots.parquet`, `orders.parquet`, `transactions_clean.parquet`, `cancels.parquet`.
- Danh sách đặc trưng: `FEATURES_NUM`, `FEATURES_CAT` trong `src/s1_data.py` (import, không tự liệt kê lại).

## 3. LUẬT CỨNG (vi phạm = hỏng bài)

1. **Không bao giờ đọc `oracle.parquet` trong code huấn luyện, chọn mô hình, hiệu chỉnh hay ra quyết định.**
   Chỉ được đọc trong hàm đánh giá có tên bắt đầu bằng `evaluate_` / `oracle_`. Viết một test kiểm tra điều này.
2. Nhãn quan sát là `y_obs` (kết quả dưới hành động đã ghi log). μ0 = P(Y=1 | A=a0, x) học trên các dòng `A_logged=="a0"`
   (hợp lệ vì chính sách lịch sử chỉ phụ thuộc x đã quan sát). Không dùng `y_obs` của dòng a≠a0 làm Y(0).
3. Mọi bộ tiền xử lý / lọc biến / bộ hiệu chỉnh chỉ `fit` trên tập được phép (bảng trên), trong `sklearn.Pipeline`.
4. `locked` chỉ mở ở GĐ8, một lần, sau khi ghi manifest đóng băng. Không quay lại chỉnh mô hình sau khi xem locked.
5. `release_gates` trong config đã công bố trước – **không sửa ngưỡng**.
6. Mỗi script: đọc config, đặt seed, ghi bảng vào `outputs/tables/sX_*.csv`, hình vào `outputs/figures/fig_sX_*.png`
   (dpi 160, nhãn tiếng Việt), mô hình vào `models/`, cuối cùng gọi `common.write_manifest("sX", files)`.
7. Không xóa/ghi đè dữ liệu thô trong `data/raw/`.
8. Báo cáo trung thực: nếu kết quả xấu (ví dụ SPO+ không thắng PTO), giữ nguyên và giải thích – đề chấm tính thực chứng.

## 4. Kế hoạch từng giai đoạn

Mỗi giai đoạn = 1 file `src/sX_*.py` có `main()`. Làm xong một giai đoạn → chạy → kiểm tra tiêu chí chấp nhận →
báo người dùng kết quả chính → mới sang giai đoạn sau.

### GĐ2 – Dự báo nền & hiệu chỉnh xác suất (Chương 2) · `src/s2_predict.py` · Sonnet · effort medium
- μ0(x): Logistic (impute + scale + one-hot `country` top-10) và LightGBM; huấn luyện trên `train` ∧ `A_logged=="a0"`,
  tinh chỉnh bằng GroupKFold(5) theo khách (lưới nhỏ), chọn mô hình trên `valid` (log loss, PR-AUC).
- Hiệu chỉnh: Platt và Isotonic, khớp trên `calib` ∧ a0; chọn phương pháp bằng CV 5-fold bên trong `calib`; báo cáo trên `valid`.
- Chỉ số: ROC-AUC, PR-AUC, log loss, Brier, ECE (15 bin equal-mass), reliability diagram trước/sau hiệu chỉnh;
  theo phân khúc UK/non-UK và one-time/repeat; CI bootstrap 95%.
- Giá trị giỏ Ŝ: hồi quy (LightGBM trên log) trên các dòng `value_obs>0` của train; khoảng **split-conformal** 90% khớp trên calib;
  báo cáo coverage trên valid.
- Lưu `data/processed/pred_s2.parquet` cho MỌI dòng: `mu0_raw, mu0_cal, s_hat, s_lo, s_hi`.
- Chấp nhận: ECE sau hiệu chỉnh ≤ 0.03 trên valid; Brier sau ≤ trước.

### GĐ3 – Giải thích & độ tin cậy (Chương 3) · `src/s3_explain.py` · Sonnet · medium
- PFI trên valid (neg_log_loss, 10 lần lặp) + PFI theo nhóm (RFM, hủy đơn, lịch, quốc gia).
- SHAP TreeExplainer (thang log-odds, ghi rõ): beeswarm, waterfall 3 khách tiêu biểu.
- Từ điển đặc trưng → nhãn nghiệp vụ; **mã lý do** top-3 cho mỗi khách (Router dùng ở GĐ7).
- Stability: tương quan hạng SHAP qua 5 lần bootstrap huấn luyện lại; faithfulness: kiểm tra xóa top-k.
- Nhắc trong báo cáo: SHAP/PFI không phải nhân quả.

### GĐ4 – Nhân quả & uplift (Chương 4) · `src/s4_causal.py` · **Opus · effort high**
- Ước lượng propensity bằng mô hình (giả vờ không biết) và so với propensity ghi log.
- ATE từng hành động so với a0 trên **purchase** và **profit**: naive, IPW, DML (DoubleML `DoubleMLIRM` hoặc EconML,
  cross-fitting 5 fold) → so với ATE thật (đọc oracle trong `evaluate_`). Bảng + forest plot có CI.
- CATE đa nhánh: T-learner (baseline), X-learner, DR-learner (EconML), CausalForestDML (cho CI). Mục tiêu τ̂_a(x) trên purchase.
- Đánh giá ngoài mẫu: Qini/AUUC (scikit-uplift) trên **nhánh ngẫu nhiên** của valid+calib; tương quan hạng τ̂ với `tau_cond` (oracle).
- Chẩn đoán: histogram propensity/overlap, placebo (xáo A → ATE≈0), ổn định theo tập con.
- Lưu `data/processed/pred_s4.parquet`: `tau_hat_a1..a3`, `tau_lo_a*`, `tau_hi_a*`, `V_hat_a1..a3` (công thức mục 1, dùng μ̂0 cal, Ŝ).
- Tùy chọn: chạy lại DR-learner trên Hillstrom (`sklift.datasets.fetch_hillstrom`) → Qini trên RCT thật.
- Chấp nhận: CI DML phủ ATE thật (hoặc giải thích); Qini > ngẫu nhiên cho ≥ 2 hành động.

### GĐ5 – Predict-Then-Optimize & SPO+ (Chương 5) · `src/s5_spo.py` · **Opus · effort high**
- Instance = lô 100 khách lấy mẫu trong một t0; ràng buộc như mục 1, B và K co theo tỷ lệ 100/n_t0. Hàm mục tiêu tuyến tính, **max**.
- Vector chi phí huấn luyện: **pseudo-outcome DR/AIPW ngoài fold** của lợi nhuận gia tăng (dùng propensity ghi log,
  μ̂_a từ cross-fitting). KHÔNG dùng oracle.
- So sánh: (i) luật RFM cũ, (ii) nhắm theo rủi ro (μ̂0 thấp nhất), (iii) Greedy theo V̂/mệnh giá, (iv) PTO hai giai đoạn
  (hồi quy MSE → MILP), (v) SPO+ (PyEPO hoặc tự cài subgradient `2(w*(c) − w*(2ĉ − c))` theo đúng quy ước dấu cho max), (vi) oracle.
- Regret chuẩn hóa = (giá trị oracle − giá trị chính sách)/|giá trị oracle|, giá trị tính bằng `profit_a*` của oracle,
  cùng miền khả thi và cùng solver cho mọi phương pháp. Nhiều seed, báo mean ± sd.
- **Thí nghiệm đinh:** bơm nhiễu vào V̂ ở nhiều mức → đồ thị MSE vs Regret (phi tuyến).
- Kiểm chứng cài đặt: instance 6 khách → liệt kê toàn bộ nghiệm, so SPO/SPO+ loss với solver (Chương 5.3).
- **Quy tắc đăng ký trước "mở rộng biên lưới"** (ghi TRƯỚC khi xem regret calib của lưới gốc; áp cho MỌI phương pháp có học
  ở GĐ5 – PTO tuyến tính, PTO LightGBM, SPO+):
  1. Tiêu chí chọn giữ nguyên, CHỈ trên valid (PTO: MSE với Γ; SPO+: SPO+ loss với Γ). Báo cáo trên calib.
  2. Một siêu tham số "ở biên" nếu giá trị được chọn trên valid nằm ở đầu/cuối lưới gốc ở ≥ 1 seed. Khi đó mở rộng
     **MỘT lần** theo hướng đó, cho mọi seed:
     - tham số thang đo (lr của SPO+, alpha của ridge): thêm ×3 và ×10 (biên trên) hoặc ÷3 và ÷10 (biên dưới);
     - tham số nguyên của LightGBM (num_leaves, n_estimators): thêm ÷3 và ÷10 (hoặc ×3 và ×10) làm tròn, num_leaves ≥ 2,
       ghép với giá trị được chọn của chiều còn lại;
     - epoch của SPO+: số epoch tối đa ×3 (15 → 45) kèm early stopping theo SPO+ loss valid, patience 3, cho mọi lr.
  3. Không mở rộng lần hai, kể cả khi tối ưu mới vẫn nằm ở biên – ghi nhận và báo cáo nguyên trạng.
  4. Cùng 5 seed, cùng instance (cùng thứ tự sinh số ngẫu nhiên), cùng solver. Báo cáo song song "lưới gốc" và
     "lưới mở rộng" (mean ± sd) kèm đường SPO+ loss valid theo epoch. Giữ nguyên kết quả dù SPO+ thắng hay thua PTO.

### GĐ6 – MILP toàn quy mô (Chương 6) · `src/s6_milp.py` · Sonnet · medium
- Mỗi t0: biến nhị phân x[i,a], PuLP + HiGHS (dự phòng CBC), time limit 60s, gap 0.1%.
- So sánh MILP vs Greedy vs luật cũ: giá trị (oracle), khoảng cách tối ưu, nguồn lực dùng, thời gian giải.
- Độ nhạy: B ∈ {2k,4k,8k,16k,32k}, K ∈ {30,60,120}; giá bóng ngân sách (LP relaxation dual); thời gian giải theo n (tới 50k).
- **Thí nghiệm đinh:** μ̂0 chưa hiệu chỉnh vs đã hiệu chỉnh → chi phí chiết khấu thực tế so với kế hoạch.

### GĐ7 – Action Router, guardrails, HITL (Chương 7) · `src/s7_router.py` · Sonnet · medium
- Lớp `ActionRouter`: kiểm tra hợp đồng đầu vào → guardrail cứng (opt-out giả lập 8% theo hash; frequency cap 28 ngày
  theo nhật ký quyết định của chính router; sàn biên LN: chiết khấu ≤ m·Ŝ; `cancel_rate_365`>0.3 → chặn voucher)
  → nhóm đối chứng 10% (hash khách×t0 → a0) → MILP → hậu kiểm (CI τ̂ vắt qua 0 mà V̂ lớn → HITL; VIP top 2% `value_365`
  với a2/a3 → HITL; solver timeout → Greedy; thiếu mô hình → luật cũ).
- Bản ghi quyết định: `decision_id` (uuid5), khóa chống lặp `customer_id|t0|policy_version`, phiên bản mô hình/quy tắc,
  mã lý do (từ GĐ3 + guardrail), hành động, kênh. Mô phỏng người duyệt HITL và ghi nhận ghi đè.
- Thống kê: tỷ lệ tự động / bị chặn (theo lý do) / HITL / đối chứng.

### GĐ8 – Triển khai, đo lường thực chứng, MLOps (Chương 8) · `src/s8_mlops.py` · Sonnet (Opus cho phần chọn lọc nhãn) · medium
- Đóng băng: manifest SHA-256 của mô hình + config + mã; **ma trận tương thích** (python + thư viện chính).
- **Locked Test (1 lần):** chỉ số dự báo, hiệu chỉnh, Qini (nhánh ngẫu nhiên), giá trị chính sách & regret (oracle) vs luật cũ
  → bảng **Release Gates** pass/fail theo `release_gates`.
- **Phát lại OOT** tuần tự từng t0 với mô hình Current: độ trễ từng khâu (perf_counter, lặp ≥30 lần/kỳ) → P50/P95/P99;
  PSI từng đặc trưng & điểm so với train; Brier/ECE **chỉ khi nhãn đã trưởng thành** (t0+30 ≤ ngày phát lại);
  Action TVD so với kỳ tham chiếu; giá trị thực hiện ước lượng bằng DR trên nhóm đối chứng vs oracle.
- Kịch bản trôi: chạy lại s1b với `oot_persuasion_scale: 0.6` (ghi file riêng, không ghi đè) → có/không guardrail.
- Current vs Challenger (định nghĩa thống nhất với mục 6b): **Challenger** = Current + hiệu chỉnh lại Platt theo kỳ, chỉ dùng nhãn
  đã trưởng thành (t0 + 30 ngày ≤ thời điểm phát lại) → so với Current trên OOT. **Challenger-2 (PHỤ)** = huấn luyện lại μ0 trên
  train+valid+calib+locked (nhãn đã trưởng thành trước OOT) → chạy SAU khi đã chấm gate, chỉ báo cáo, không ảnh hưởng gate.
- Khuyết nhãn có chọn lọc: Challenger học từ dữ liệu do router sinh ra, có vs không có nhóm đối chứng/IPW.
- Playbook: bảng cảnh báo → ngưỡng (`monitoring`) → chủ sở hữu → hành động.

### Cuối cùng
- `notebooks/PTDLNC_full_pipeline.ipynb`: chạy tuần tự s1 → s8 (để nộp Colab), có markdown giải thích từng bước.
- `outputs/tables/summary_for_essay.md`: bảng ánh xạ kết quả → mục IV/V của tiểu luận.

## 5. Ánh xạ kết quả → tiểu luận

| Mục tiểu luận | Lấy từ |
|---|---|
| III. Thiết kế dữ liệu, Graphic Abstract | `fig_s1_split_timeline`, `s1_*` bảng, mục 1–2 file này |
| IV. Dự báo & hiệu chỉnh | `s2_*` |
| IV. Nhân quả | `s4_*`, `s1b_confounding_check` |
| IV. Tối ưu | `s5_*`, `s6_*` |
| IV. Router & MLOps | `s7_*`, `s8_*` |
| V. Thảo luận | MSE vs Regret, hiệu chỉnh vs vượt ngân sách, guardrail khi trôi, khuyết nhãn chọn lọc, giả định bán tổng hợp |
| VIII. Phụ lục | `outputs/manifests/*.json`, ma trận tương thích, `s1_data_dictionary.csv` |

## 6. Hạn chế đã biết (ghi vào mục V)
- Lớp can thiệp là bán tổng hợp: không có hiệu ứng mang sang giữa các t0; giá trị giỏ người mới chuyển đổi là mô phỏng.
- Biên lợi nhuận 30% là giả định (cần phân tích độ nhạy 20–40%).
- Nhiều khách là khách sỉ; quần thể ~2.3–4.8k khách/kỳ.
- **Trôi tỷ lệ nền theo mùa (phát hiện ở GĐ2):** tỷ lệ mua lại a0 là 0.255 (train) → 0.179 (valid) → 0.212 (calib) → 0.187 (locked) → 0.289 (oot). Bộ hiệu chỉnh khớp trên calib (tháng 4–5) nên ECE trên valid = 0.047 > 0.03 (ECE out-of-fold trong calib = 0.028). Bỏ đặc trưng lịch không giải quyết được (xem `s2_ablation_calendar.csv`). **Hiệu chỉnh lại theo kỳ (recalibration) sẽ xử lý ở GĐ8**; ngưỡng G2 không đổi. Ŝ kỳ vọng (`s_hat_mean`) cũng lệch +13% trên valid vì cùng nguyên nhân.

## 6b. Đăng ký trước cho GĐ8 (ghi TRƯỚC khi mở locked – không sửa sau khi xem kết quả)
- Bộ hiệu chỉnh Platt được khớp trên calib (t0 tháng 4–5) trong khi μ0 dùng đặc trưng lịch ngoại suy (valid = tháng 1–3, ngoài
  khoảng train 6–12). Locked (tháng 7–8) và oot (tháng 9–11) nằm TRONG khoảng tháng của train, nên bộ hiệu chỉnh này có thể không hợp.
- **Current** = pipeline đóng băng hiện tại (μ0 + Platt từ GĐ2, không chỉnh lại) → là phương án DUY NHẤT dùng để chấm `release_gates`.
- **Challenger** = Current + hiệu chỉnh lại **Platt** theo kỳ (recalibration; μ0 giữ nguyên), chỉ dùng nhãn đã trưởng thành
  (t0 + 30 ngày ≤ thời điểm phát lại) → báo cáo song song như đề xuất vận hành, KHÔNG thay thế Current khi chấm gate.
- **Challenger-2 (PHỤ)** = huấn luyện lại μ0 trên train+valid+calib+locked (nhãn đã trưởng thành trước OOT) → chỉ chạy SAU khi đã chấm
  gate trên locked, báo cáo trên OOT, không ảnh hưởng gate. (Thay cho mô tả "Challenger huấn luyện lại…" cũ ở mục GĐ8.)
- Không đổi bất kỳ ngưỡng nào trong `release_gates`.
- **Dự báo power cho G3 (đăng ký trước, `src/s4b_power_g3.py` → `s4_power_g3*.csv`, `fig_s4b_power_g3.png`):**
  learner = CausalForestDML (đã chốt ở GĐ4, không chọn lại). Locked có 12 947 dòng (10 942 khách thường, 2 005 entity-holdout)
  ⇒ theo thiết kế nhánh ngẫu nhiên 20%, mỗi cặp Qini (a0 ∪ a) có ≈ 1 295 dòng, bằng ~½ valid+calib (2 541) ⇒ SE Qini ×1.4.
  P(G3 đạt, ≥ 2/3 hành động có CI 95% > 0), toàn bộ locked: **≈ 4–7%** với kịch bản chính (Qini calib riêng – không thiên lệch
  do chọn mô hình: a1 0.068, a2 0.004, a3 0.021), ≈ 10–13% (valid+calib gộp, lạc quan), ≈ 17–21% (train ngoài fold).
  Khoảng là ρ = 0 và ρ = 0.3 giữa ba phép so sánh. ⇒ **Dự báo trước: G3 nhiều khả năng KHÔNG đạt**, chủ yếu do cỡ mẫu
  nhánh ngẫu nhiên (a1 cần ~2.5× cỡ mẫu locked để đạt power 80%; a2, a3 cần > 10×), không phải do mô hình sai hướng.
  Entity-holdout riêng (≈ 200 dòng/cặp): P(G3) < 3%, chỉ mang tính mô tả.
- **G4 – định nghĩa (GIỮ đặc tả gốc) và dự báo đăng ký trước:**
  - G4a = `G4_policy_value_lift_vs_legacy_min` (giá trị chính sách ≥ luật cũ × 1.10); G4b = `G4_normalized_regret_max`
    (regret chuẩn hóa ≤ 0.35). Regret tính theo lợi nhuận **HIỆN THỰC** `profit_a*` (oracle A) – đây là định nghĩa DUY NHẤT để chấm.
  - Y(0) và giỏ S thuộc cửa sổ kết quả [t0, t0+30) nên **chưa biết tại t0**. Oracle A (biết hiện thực) và oracle B (kỳ vọng theo U,
    điều kiện trên Y(0), S hiện thực) đều dùng thông tin tương lai. Trong thiết kế bán tổng hợp **không có oracle khả thi thật**,
    vì P(Y0|x) và E[S|x] thật không biết.
  - B và C (Oracle-τ: c_a, d_a thật của cơ chế + μ̂0, Ŝ của mô hình) chỉ là công cụ **PHÂN RÃ sai số** (may rủi / sai số μ̂0, Ŝ /
    sai số τ̂), báo kèm, không dùng để chấm gate.
  - **Dự báo:** G4b gần như chắc **trượt** – ngay chính sách biết kỳ vọng thật (oracle B) cũng có regret A = 0.742 ± 0.048 ở lô
    100 khách (GĐ5, calib). G4a là tiêu chí **khả thi** (GĐ5 calib, lô 100: MILP trên V̂ £122.7 so với luật cũ £40.3 lợi nhuận
    gia tăng hiện thực / lô). So sánh G4a theo đúng công thức config (V_chính_sách ≥ 1.10 × V_luật_cũ); báo kèm chênh lệch tuyệt đối.
  - GĐ8 tính G4 ở **quy mô toàn bộ một t0** (MILP trên mọi khách của kỳ, B và K không co), **cộng dồn qua các kỳ** locked
    (cộng tử số và mẫu số qua các kỳ rồi mới lấy tỷ số).
- **G4a – cách đọc công thức (đăng ký trước GĐ8):** G4a ⇔ (giá trị chính sách − legacy) ≥ 0.10 × |legacy|, giá trị tính theo lợi nhuận
  gia tăng **HIỆN THỰC (oracle A)**, **cộng dồn qua các kỳ** locked. Lý do: công thức `legacy × 1.10` vô nghĩa khi legacy âm (legacy × 1.10 < legacy).
  Ngưỡng 0.10 giữ nguyên. Trên valid+calib (7 kỳ, `s6_main_comparison.csv`): MILP trên V̂ = £28 134, luật cũ = −£2 650 ⇒ MILP đạt G4a ở **cả hai**
  cách đọc (legacy × 1.10 và legacy + 0.10 × |legacy|) ⇒ việc chọn cách đọc KHÔNG làm gate dễ hơn.
- **Router v2.1 – đăng ký trước cho GĐ8 (quyết định SAU khi đã thấy kết quả v2 trên valid/calib; chưa mở locked/oot; không sửa ngưỡng):**
  1. **Bước sửa khả thi cuối** (sau HITL và bù độ phủ; áp cho MỌI luồng): nếu số a2 > 25 % số voucher phát, hạ lần lượt các a2 có V̂ (a2) thấp nhất
     xuống a1 nếu V̂(a1) > 0, ngược lại về a0, lặp tới khi thỏa; sau đó kiểm lại ngân sách / K / độ phủ (bù nếu cần theo đúng quy tắc v2, loại các
     khách vừa bị đưa về a0 khỏi danh sách bù). **G5 chấm trên quyết định GỬI THỰC TẾ của luồng Current (MILP qua Router)**, tử số độ phủ không gồm đối chứng.
  2. **G4a và G7 – so sánh CHÍNH** = MILP qua Router v2.x vs **Luật cũ qua Router v2.x** (cùng cap / opt-out / HITL / đối chứng, vì cap và opt-out là quy định
     bắt buộc, không phải lựa chọn của chính sách); **PHỤ** = so với luật cũ thuần. Công thức G4a (đọc theo mục trên) và G7 (phân khúc UK / non-UK) áp cho cả hai.
     Trên valid+calib (kết quả v2): MILP qua Router v2 = £21 483 vs luật cũ qua Router v2 = £15 006 vs luật cũ thuần = −£2 650 (hiện thực, 7 kỳ cộng dồn) ⇒
     đạt G4a ở **cả hai** cách so ⇒ việc chọn so sánh chính KHÔNG làm gate dễ hơn (số liệu v2.1 xem `s7_*_v21.csv`).
  3. **Giả định HITL:** người duyệt là QUY TẮC MÔ PHỎNG (V̂ tại nửa đường từ τ̂ tới cận dưới CI ≥ 0), không phải người thật; tải HITL ≈ 117 ca/kỳ ở luồng
     MILP (819/7), giả định đủ nhân sự duyệt hết trong kỳ (không mô hình hóa hàng đợi hay độ trễ duyệt).
  4. Quyết định v2 đã xác nhận: khách đối chứng KHÔNG nằm trong danh sách bù; khi bù chỉ kiểm ngân sách mệnh giá (thêm một a1 không dùng cuộc gọi và chỉ làm
     GIẢM tỷ lệ a2); ngân sách do đối chứng giải phóng được dùng để bù.
- **Bộ ước lượng giá trị chính sách CHÍNH cho GĐ8** (đăng ký trước, chọn theo nguyên tắc không chệch – KHÔNG theo độ gần đáp án):
  AIPW **không winsorize** Γ_ia = ψ_a − ψ_0, ψ_k = μ̂_k + 1{A=k}(Y − μ̂_k)/e_k, Y = `profit_obs`, e = propensity ghi log,
  μ̂_k như `s5_spo.py` (LightGBM; ngoài fold trên train, học trên toàn train khi chấm tập khác); giá trị chính sách = Σ w·Γ;
  CI 95% bootstrap khối theo `customer_id`. Các biến thể (winsorize Γ, winsorize phần dư Y − μ̂_k, self-normalized, cắt trọng số)
  CHỈ là phân tích độ nhạy (`src/s5b_dr_diagnostics.py`).
- **Dự báo đăng ký trước về DR ở GĐ8 (OOT):** ở s5b, DR dùng chính sách ghi log (20% ngẫu nhiên) nên mọi dòng đều cho thông tin.
  Ở OOT, Router vận hành gần tất định: chỉ 10% đối chứng ngẫu nhiên trong số khách được chọn nhận voucher ⇒ mẫu để ước lượng DR cho
  phần tăng thêm chỉ còn ≈ 25 khách/kỳ (thay vì toàn bộ dòng). **Kỳ vọng:** CI của DR ở GĐ8 rộng hơn NHIỀU so với
  `s5b_full_t0_milp.csv`/`s5b_estimates.csv` (ở s5b SE chênh lệch MILP − luật cũ đã ≈ £7.5–14.7k/kỳ so với chênh lệch thật ≈ £4–5k/kỳ),
  gần như chắc **không phân định được MILP với Current**. G4a vẫn chấm bằng oracle theo đặc tả; DR ở GĐ8 chỉ là **thước đo giám sát
  kèm CI**, KHÔNG dùng để quyết định gate.
- **Làm rõ trước khi mở locked** (ghi 2026-11-12 21:05 +07, người dùng đã đồng ý; tại thời điểm ghi **locked và oot CHƯA được mở**,
  chưa có file khóa `s8_locked_opened.lock`; chỉ mới diễn tập `--dry-run` với calib đóng vai locked; không đổi ngưỡng nào):
  (a) **G4b** chấm trên **MILP qua Router v2.1** (chính sách triển khai). MILP thuần chỉ báo kèm, ghi là **G4b-phụ**.
  (b) **G6** tính độ trễ đầu–cuối **gồm cả khâu mã lý do SHAP top-3** (GĐ7 không tính khâu này), vì locked không có `reasons_s3` nên phải tính trực tiếp.
  (c) **Quần thể chấm chính thức = toàn bộ** locked. Khách thường và entity-holdout báo riêng (entity-holdout chỉ mô tả). G1/G5/G6 chỉ tính trên toàn bộ.
  (d) **G1** là bộ kiểm tra lắp ghép ở GĐ8 (không có đăng ký trước riêng): schema snapshot GĐ1, hợp đồng logged, khóa logged = snapshot,
  y_obs dưới a0 = Y(0), entity holdout, embargo, hợp đồng đầu vào Router. Ở lần dry-run đầu G1 từng trượt do **lỗi code** kiểm tra propensity
  (không phải lỗi dữ liệu). Lỗi đã được sửa.

## 7. Môi trường (macOS Apple Silicon)
```bash
python3.11 -m venv .venv && source .venv/bin/activate
brew install libomp            # LightGBM cần trên macOS
pip install -r requirements.txt
python src/s1_data.py && python src/s1b_intervention.py   # tái lập GĐ1 (~2 phút)
```
Sau khi cài xong, ghi `pip freeze` vào `outputs/manifests/requirements_lock.txt` (dùng cho ma trận tương thích).

## 8. Hướng phát triển (NGOÀI phạm vi bài – không áp dụng, không chạy)
- **Winner's curse theo biên m (phát hiện ở GĐ6):** MILP trên V̂ với m̂ thấp hơn m thật cho giá trị thật cao hơn m̂ đúng (ví dụ m thật 0.30:
  m̂ = 0.20 cho £5 024 so với £4 538 khi m̂ = 0.30) ⇒ V̂ chọn khách có τ̂ cao một phần do nhiễu. Hướng xử lý: **shrinkage τ̂** (co về trung bình
  nhóm, hoặc dùng cận dưới CI trong V̂). **KHÔNG áp dụng trong bài** vì phát hiện này đến từ đối chiếu với đáp án (luật cứng #1: đáp án chỉ để
  đánh giá, không để chỉnh mô hình/tham số).
