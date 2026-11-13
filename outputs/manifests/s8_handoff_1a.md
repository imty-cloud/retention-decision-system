# Handoff – GĐ8 bước 1a (chuẩn bị phần gate) · cập nhật 2026-11-12 21:10 +07

## Trạng thái: bước 1a XONG
| # | Việc | Trạng thái |
|---|---|---|
| 1 | Thống nhất Challenger trong CLAUDE.md, mục GĐ8 và 6b (Challenger = Current + Platt hiệu chỉnh lại theo kỳ, chỉ dùng nhãn đã trưởng thành; Challenger-2 PHỤ = huấn luyện lại μ0, chạy sau khi chấm gate) | XONG |
| 2 | `src/s8_mlops.py`: `--dry-run`, `--freeze-draft`, `--freeze`, `--open-locked` (từ chối khi đã có khóa; từ chối khi lệch hash mã/mô hình/config/CLAUDE.md, **lệch byte dữ liệu `data/processed/*.parquet`** hoặc lệch phiên bản thư viện; ghi khóa O_EXCL TRƯỚC khi đọc locked; nếu lỗi sau khi đã có khóa thì ghi `s8_locked_error.log`). Mọi lần đọc parquet đi qua `read_rows` | XONG |
| 3 | Chấm G1–G7, tách toàn bộ / khách thường / entity-holdout, có cột dự báo đăng ký trước | XONG |
| 4 | `s8_freeze_draft_manifest.json` (27 file mã/mô hình/config + 14 file dữ liệu, băm theo byte), `requirements_lock.txt`, `s8_compat_matrix.csv` (Python 3.14.7). **Khớp code cuối** (đã kiểm: 0 lệch) | XONG |
| 5 | `tests/test_s8.py` (15 test), gồm diễn tập đường locked và băm dữ liệu | XONG |
| 6 | `--dry-run` chạy lại theo code cuối → `outputs/tables/s8_dryrun_*.csv`, `fig_s8_dryrun_gates.png`, `s8_dryrun_manifest.json` | XONG |
| 7 | Đoạn "Làm rõ trước khi mở locked" trong 6b (người dùng đã đồng ý), điểm (a)–(d) | XONG |
| 8 | `outputs/tables/s8_checklist_1a.csv` (18 hạng mục, đều ĐẠT / KHỚP) | XONG |

## Kết quả dry-run (THỬ, calib đóng vai locked) – giống lần trước, trừ độ trễ
G1 ĐẠT 7/7 · G2 TRƯỢT (ECE 0.0311 > 0.03; Brier 0.1304 ≤ 0.1405) · G3 TRƯỢT 0/3 · G4a ĐẠT (£12 296 vs £9 358) · G4b TRƯỢT (0.834) ·
G5 ĐẠT (0) · G6 ĐẠT (P95 0.88 s; lần trước 0.73 s, chênh do dao động thời gian máy) · G7 ĐẠT (UK +55 %, non-UK +8.7 %).
Đối chiếu GĐ4/GĐ7: giá trị chính sách lệch ≤ 2e-12, Qini lệch 0, kiểm toán ràng buộc lệch 0, mã lý do khác 0 %.

## Diễn tập đường locked (`test_locked_path_rehearsal`)
`pd.read_parquet` được bọc lại: bộ lọc split "locked" đổi thành "calib" TRƯỚC khi đọc, rồi gán split = "locked"; giả lập khoảng 15 % khách là entity-holdout (theo hash).
`run_gates("locked")` chạy trọn vẹn, ghi `s8_locked_*` vào tmp: nhãn CHÍNH THỨC, 3 kỳ, nhánh entity-holdout tính được (G2/G3/G4); toàn bộ = khách thường + entity-holdout;
G1 bắt đúng khách "holdout" giả lập có mặt trong tập mô hình; không đọc reasons_s3. **Không lộ lỗi riêng nào của đường locked.**
Giới hạn của buổi diễn tập: dữ liệu vẫn là calib, nên không kiểm được phân phối thật của locked (khách entity-holdout chưa từng thấy, quốc gia lạ). Các chỗ này do mô hình
và các schema đã có xử lý; lỗi phát sinh, nếu có, sẽ nằm trong `s8_locked_error.log`.

## pytest gần nhất
`python -m pytest -q` → **64 passed, 0 failed**. test_s9 qua khi chạy cả bộ (nhờ reset `highspy.Highs.resetGlobalScheduler(True)` trong test_s8), không cần conftest.py.

## Xác nhận an toàn
- **CHƯA** đọc dữ liệu locked/oot. Phần băm dữ liệu chỉ đọc byte file, không parse dòng.
- `outputs/manifests/s8_locked_opened.lock`: **CHƯA tồn tại**. Chưa có `s8_freeze_manifest.json` chính thức. Chưa chạy `--open-locked` hay `--freeze`.
- Không sửa ngưỡng `release_gates`.

## Việc tiếp theo (theo thứ tự)
1. Bước 1b (theo kế hoạch của người dùng).
2. `python src/s8_mlops.py --freeze` (đóng băng chính thức). Sau đó KHÔNG sửa src/ configs/ models/ CLAUDE.md và data/processed/*.parquet (trừ decisions_s8_*),
   và không đổi phiên bản thư viện, nếu không `--open-locked` sẽ từ chối. Lưu ý: chạy lại GĐ nào ghi parquet vào data/processed cũng làm lệch hash.
3. `python src/s8_mlops.py --open-locked` (một lần).
4. Phát lại OOT, PSI, Action TVD, DR giám sát, kịch bản trôi 0.6, Challenger, Challenger-2, khuyết nhãn có chọn lọc, playbook (`read_rows` hiện chặn oot – phải mở rộng ở bước đó).

## Quyết định đã tự đưa ra (ngoài những điểm đã ghi vào 6b)
1. Phạm vi băm dữ liệu = `data/processed/*.parquet` **trừ `decisions_s8_*`** (đầu ra của chính s8; nếu không loại, chạy lại dry-run sẽ chặn việc mở locked). Không băm `data/raw/`.
2. Lệch phiên bản thư viện chính cũng bị coi là lệch đóng băng (chặt hơn yêu cầu).
3. G6 dùng P95 gộp mọi lượt (giống GĐ7), có báo kèm P95 lớn nhất theo kỳ. Router của tập chấm bắt đầu với lịch sử frequency cap rỗng (calib→locked cách 42 ngày, lớn hơn 28).
4. G2 THỬ trên calib là trong mẫu (calibrator khớp trên calib), dù vậy ECE vẫn 0.0311 > 0.03.
5. Không làm trong 1a: phân rã đáp án B/C (Oracle-τ) và DR giám sát; regret B có báo kèm.
