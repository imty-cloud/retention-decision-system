# PTDLNC – Hệ thống ra quyết định giữ chân khách hàng đa kênh (Online Retail II)

Tiểu luận cuối kỳ *Phân tích dữ liệu nâng cao*. Đặc tả đầy đủ, luật cứng và kế hoạch từng giai đoạn: **xem [`CLAUDE.md`](CLAUDE.md)**.

Chu trình 8 giai đoạn: thiết kế bài toán và dữ liệu (chống rò rỉ) → dự báo + hiệu chỉnh xác suất → giải thích → nhân quả (ATE/CATE/uplift) →
Predict-Then-Optimize và SPO+ → MILP → Action Router (guardrail, HITL) → triển khai, đo lường thực chứng, giám sát MLOps. Kèm kiểm định ngoài trên RCT thật (Hillstrom, `s9`).

## Trạng thái và kết quả chính

Pipeline GĐ1–GĐ8 và s9 đã chạy xong; **Locked Test đã được mở đúng một lần** (khóa: `outputs/manifests/s8_locked_opened.lock`) và OOT đã phát lại.
**Hệ thống KHÔNG đạt đủ cổng phát hành**: G4b (regret chuẩn hóa 0.894 > 0.35) trượt; G1, G2, G3, G4a, G5, G6, G7 đạt ở mức toàn bộ locked.
Kết quả theo từng mục tiểu luận, các số liệu kèm file nguồn, và mục **"Công khai và sai lệch so với đăng ký trước"** (đọc kỹ):
[`outputs/tables/summary_for_essay.md`](outputs/tables/summary_for_essay.md).

## Cấu trúc

```
CLAUDE.md                      đặc tả, luật cứng, đăng ký trước (6b) – nằm trong phạm vi đóng băng
configs/config.yaml            mọi tham số thiết kế (t0, H, chia tập, hành động, DGP, release gates, monitoring)
src/s1_data.py … s8b_oot.py    GĐ1 … GĐ8 (mỗi giai đoạn một script); s9_external_hillstrom.py: kiểm định ngoài
tools/build_packaging.py       sinh phụ lục SHA-256, summary_for_essay.md, notebook (chỉ đọc kết quả có sẵn)
tests/                         pytest -q
notebooks/                     PTDLNC_full_pipeline.ipynb (Colab / Jupyter)
models/                        mô hình đã huấn luyện (.joblib) – đã đóng băng
data/raw/                      dữ liệu thô (chỉ có online_retail_ii.zip, xem "Dữ liệu và giấy phép")
data/processed/                snapshot, logged/oracle, dự báo, bản ghi quyết định (.parquet) – đã đóng băng
outputs/tables|figures         bảng, hình cho tiểu luận
outputs/manifests/             SHA-256, phiên bản thư viện, handoff, khóa, requirements_lock.txt
```

## Tái lập

Môi trường kiểm chứng: **GĐ1 và GĐ1b chạy trên Linux x86_64, Python 3.13.18** (pandas 3.0.5, numpy 2.5.3); **các GĐ còn lại (GĐ2–GĐ8, s9) chạy trên macOS Apple Silicon (arm64), Python 3.14.7**.
Dựng lại GĐ1 + GĐ1b trên macOS (3.14.7) cho nội dung `transactions_clean`, `snapshots`, `logged`, `oracle` trùng 100 % với parquet đã đóng băng (kiểm toán 13/11). `requirements.txt` khuyến nghị Python 3.11 nhưng **chưa kiểm trên 3.11**.
`outputs/manifests/requirements_lock.txt` là `pip freeze` của môi trường 3.14.7 (nên trên Python khác có thể không cài được bản ghim);
ma trận tương thích: `outputs/tables/s8_compat_matrix.csv`.

```bash
python3 -m venv .venv && source .venv/bin/activate
brew install libomp                                   # LightGBM trên macOS
pip install -r outputs/manifests/requirements_lock.txt   # hoặc: pip install -r requirements.txt
```

### 1. Kiểm trạng thái đóng băng (không ghi gì)

SHA-256 của `src/`, `configs/`, `models/`, `CLAUDE.md` và `data/processed/*.parquet` (băm theo byte) nằm trong `outputs/manifests/s8_freeze_manifest.json`; bảng đầy đủ: `outputs/tables/s8_sha256_appendix.csv`.

```bash
python -c "import sys; sys.path.insert(0,'src'); import s8_mlops as S8; print(S8.verify_freeze() or 'KHỚP')"
python -m pytest -q          # 76 passed, 1 skipped (kiểm thử --oot bị từ chối được bỏ qua khi khóa đã tồn tại)
```

`.gitattributes` đặt `* -text` để Git không đổi ký tự xuống dòng (làm lệch hash trên Windows).

### 2. Xem kết quả có sẵn

Mở `notebooks/PTDLNC_full_pipeline.ipynb` (mặc định `RUN_FULL = False`: chỉ đọc bảng/hình trong `outputs/`, kiểm SHA-256 ở ô đầu).

#### Chạy trên Colab

[![Mở trong Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imty-cloud/retention-decision-system/blob/main/notebooks/PTDLNC_full_pipeline.ipynb)

Notebook đã đặt sẵn `REPO_URL = "https://github.com/imty-cloud/retention-decision-system.git"` ở ô thứ hai: trên Colab nó tự clone repo
(về `/content/repo`) rồi chạy tuần tự. Cần repo ở chế độ public và đã push nhánh `main`.

### 3. Chạy lại pipeline từ đầu (GHI ĐÈ, chỉ làm trên bản sao)

Chạy lại sẽ ghi đè `data/processed`, `models`, `outputs` và làm lệch hash đóng băng. **Không thể mở lại Locked Test** (`--open-locked` và `--oot` từ chối khi đã có khóa).

```bash
unzip -o data/raw/online_retail_ii.zip -d data/raw/        # giải ra data/raw/online_retail_II.xlsx
python src/s1_data.py && python src/s1b_intervention.py    # GĐ1, GĐ1b (~2 phút)
python src/s2_predict.py && python src/s3_explain.py && python src/s4_causal.py && python src/s4b_power_g3.py
python src/s5_spo.py && S5_LUOI=mo_rong python src/s5_spo.py && python src/s5b_dr_diagnostics.py
python src/s6_milp.py && python src/s7_router.py --v21 && python src/s9_external_hillstrom.py
python src/s8_mlops.py --dry-run && python src/s8b_oot.py --pilot && python src/s8b_oot.py --oot-rehearsal   # diễn tập
```

Thứ tự đóng băng ban đầu: `s8b_oot.py --make-drift` → `s8_mlops.py --freeze` → `--open-locked` → `s8b_oot.py --oot`.
Kết quả không nhất thiết trùng từng byte (phụ thuộc phiên bản thư viện và đa luồng); hash đóng băng chỉ đảm bảo cho các file trong repo này.

## Dữ liệu và giấy phép

| Dữ liệu | Giấy phép (kiểm tra ngày đóng gói, trên trang gốc) | Trong repo? |
|---|---|---|
| **Online Retail II** – Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5CG6D> | **CC BY 4.0** ("cho phép chia sẻ và chuyển thể cho mọi mục đích, với điều kiện ghi nguồn phù hợp"). Trang: <https://archive.ics.uci.edu/dataset/502/online+retail+ii> | Có `data/raw/online_retail_ii.zip` (43.5 MB). Bản giải nén `online_retail_II.xlsx` (trùng nội dung) **không** đưa vào, giải bằng lệnh `unzip` ở trên |
| **Hillstrom** – *MineThatData E-Mail Analytics And Data Mining Challenge* (K. Hillstrom, 2008). Trang: <https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html> | **Không nêu giấy phép** trên trang blog hoặc trên trang tài liệu `scikit-uplift`. Quyền phân phối lại chưa rõ | **Không.** `s9_external_hillstrom.py` tự tải qua `sklift.datasets.fetch_hillstrom` (file `hillstorm_no_indices.csv.gz`, 64 000 dòng) vào `data/raw/hillstrom/` (đã `.gitignore`) |

SHA-256 để đối chiếu (ghi lúc chạy, `outputs/tables/s8_sha256_appendix.csv`):
`online_retail_ii.zip` = `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb`;
`online_retail_II.xlsx` = `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`;
`hillstorm_no_indices.csv.gz` = `bab6578f60db5d792f1c2372c502f029152a5249cf5ea84390f3b7f885d7234f`.

Lớp can thiệp (voucher) là **bán tổng hợp**: Y(0) là hành vi mua lại thật, Y(a) sinh từ cơ chế đã biết; các hạn chế xem `CLAUDE.md` mục 6.
Không file nào trong repo vượt 50 MB (lớn nhất: `models/s4_cate_bundle.joblib` ≈ 41 MB). `.venv/`, `__pycache__/`, `.pytest_cache/` không được theo dõi.

**Giấy phép mã nguồn:** chưa chọn. Thêm file `LICENSE` trước khi công khai nếu muốn người khác tái sử dụng mã.

## Ghi chú sau kiểm toán độc lập (13/11/2026)

Chi tiết: `outputs/manifests/audit_2026-11-13.md`; ghi chú tương ứng trong bài: `outputs/tables/summary_for_essay.md` (mục IV.6b và phần "Công khai").
- Chạy lại GĐ1 sẽ tạo `data/processed/raw_combined.parquet`, nằm trong phạm vi băm dữ liệu ⇒ `verify_freeze` báo "file mới" dù nội dung không đổi. Hai hình `fig_s1_split_timeline.png`, `fig_s1_base_rate_by_t0.png` ghi sau `s1_manifest.json` nên SHA trong manifest đó không còn khớp.
- `CLAUDE.md` §2 còn ghi "tiếp theo: `--open-locked`" (lỗi thời, không sửa được vì nằm trong phạm vi băm). Trạng thái thật: locked đã mở lúc 2026-11-12 21:40:19 (+07), xem `outputs/manifests/s8_locked_opened.lock`.
- `Scorer` đọc hệ số smearing và q conformal từ `outputs/manifests/s2_manifest.json`, và G1 đọc `s1_embargo_check.csv`: hai đầu vào này nằm ngoài phạm vi băm (đã đối chiếu khớp với `pred_s2.parquet`).
