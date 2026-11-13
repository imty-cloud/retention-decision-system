"""Kiểm thử GĐ8 bước 1a: file khóa chặn lần mở thứ hai; hash lệch thì từ chối; khóa ghi TRƯỚC khi đọc locked;
dry-run không chạm locked/oot (kể cả đáp án)."""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import s8_mlops as S8  # noqa: E402


def fake_root(tmp_path: Path) -> Path:
    r = tmp_path / "proj"
    for d in ("src", "configs", "models"):
        (r / d).mkdir(parents=True)
    (r / "src/a.py").write_text("x = 1\n")
    (r / "configs/config.yaml").write_text("seed: 42\n")
    (r / "models/m.joblib").write_bytes(b"\x00model")
    (r / "CLAUDE.md").write_text("# đặc tả\n")
    (r / "data/processed").mkdir(parents=True)
    (r / "data/processed/logged.parquet").write_bytes(b"PAR1" + bytes(range(64)) + b"PAR1")   # byte giả, không phải parquet thật
    (r / "data/processed/decisions_s8_dryrun.parquet").write_bytes(b"PAR1-dau-ra-s8")       # đầu ra của s8: ngoài phạm vi
    return r


def official_freeze(root: Path, path: Path) -> Path:
    path.write_text(json.dumps(S8.build_freeze("s8_freeze", root=root)), encoding="utf-8")
    return path


def _real_state():
    """Dấu vân tay của khóa và bảng gate locked THẬT (sau khi locked đã mở các file này tồn tại; test chỉ được phép KHÔNG làm đổi chúng)."""
    f = [S8.LOCK_PATH, S8.ROOT / "outputs/tables/s8_locked_gates.csv"]
    return [(x.exists(), S8.common.sha256_file(x) if x.exists() else None) for x in f]


@pytest.fixture(autouse=True)
def reset_state():
    S8._STATE.update(locked_open=False, oot_open=False, lock_path=S8.LOCK_PATH)
    yield
    S8._STATE.update(locked_open=False, oot_open=False, lock_path=S8.LOCK_PATH)


def test_lock_blocks_second_run(tmp_path):
    root = fake_root(tmp_path)
    fz, lock = official_freeze(root, tmp_path / "s8_freeze_manifest.json"), tmp_path / "s8_locked_opened.lock"
    calls = []
    S8.open_locked(lock, fz, root, runner=lambda: calls.append(lock.exists()))
    assert calls == [True] and lock.exists()                       # khóa đã có khi runner (đọc locked) chạy
    with pytest.raises(S8.LockedAlreadyOpened):
        S8.open_locked(lock, fz, root, runner=lambda: calls.append("lần 2"))
    assert calls == [True]                                          # lần hai không chạy
    with pytest.raises(S8.LockedAlreadyOpened):                     # tạo khóa nguyên tử: không ghi đè
        S8.acquire_lock(lock, {})


def test_lock_survives_failed_run(tmp_path):
    root = fake_root(tmp_path)
    fz, lock = official_freeze(root, tmp_path / "f.json"), tmp_path / "l.lock"

    def boom():
        raise RuntimeError("lỗi giữa chừng")
    with pytest.raises(RuntimeError):
        S8.open_locked(lock, fz, root, runner=boom)
    assert lock.exists() and not S8._STATE["locked_open"]
    assert "lỗi giữa chừng" in (tmp_path / "s8_locked_error.log").read_text(encoding="utf-8")
    with pytest.raises(S8.LockedAlreadyOpened):
        S8.open_locked(lock, fz, root, runner=lambda: None)


@pytest.mark.parametrize("change", ["sua_model", "sua_code", "them_file", "xoa_file", "sua_claude"])
def test_hash_mismatch_refuses(tmp_path, change):
    root = fake_root(tmp_path)
    fz, lock = official_freeze(root, tmp_path / "f.json"), tmp_path / "l.lock"
    {"sua_model": lambda: (root / "models/m.joblib").write_bytes(b"\x01model"),
     "sua_code": lambda: (root / "src/a.py").write_text("x = 2\n"),
     "them_file": lambda: (root / "src/b.py").write_text("y = 1\n"),
     "xoa_file": lambda: (root / "configs/config.yaml").unlink(),
     "sua_claude": lambda: (root / "CLAUDE.md").write_text("# sửa sau đóng băng\n")}[change]()
    ran = []
    with pytest.raises(S8.FreezeMismatch):
        S8.open_locked(lock, fz, root, runner=lambda: ran.append(1))
    assert not lock.exists() and not ran                            # từ chối TRƯỚC khi ghi khóa / đọc dữ liệu


def test_data_byte_change_refuses_before_lock(tmp_path):
    root = fake_root(tmp_path)
    fz, lock = official_freeze(root, tmp_path / "f.json"), tmp_path / "l.lock"
    man = json.loads(fz.read_text(encoding="utf-8"))
    assert [e["file"] for e in man["data_files"]] == ["data/processed/logged.parquet"]       # đầu ra s8 bị loại khỏi phạm vi
    f = root / "data/processed/logged.parquet"
    b = bytearray(f.read_bytes()); b[10] ^= 0x01; f.write_bytes(bytes(b))                   # sửa đúng 1 byte
    ran = []
    with pytest.raises(S8.FreezeMismatch, match="dữ liệu"):
        S8.open_locked(lock, fz, root, runner=lambda: ran.append(1))
    assert not lock.exists() and not ran


def test_data_new_file_refuses_and_s8_output_ignored(tmp_path):
    root = fake_root(tmp_path)
    fz = official_freeze(root, tmp_path / "f.json")
    (root / "data/processed/decisions_s8_dryrun.parquet").write_bytes(b"khac")              # đổi đầu ra s8: không ảnh hưởng
    assert S8.verify_freeze(fz, root) == []
    (root / "data/processed/them.parquet").write_bytes(b"PAR1moi")
    assert any("dữ liệu data/processed/them.parquet" in m for m in S8.verify_freeze(fz, root))


def test_missing_or_draft_freeze_refuses(tmp_path):
    root = fake_root(tmp_path)
    lock = tmp_path / "l.lock"
    with pytest.raises(S8.FreezeMismatch):
        S8.open_locked(lock, tmp_path / "khong_co.json", root, runner=lambda: None)
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps(S8.build_freeze("s8_freeze_draft", root=root)), encoding="utf-8")
    with pytest.raises(S8.FreezeMismatch):                          # bản nháp không đủ để mở locked
        S8.open_locked(lock, draft, root, runner=lambda: None)
    assert not lock.exists()


def test_matching_freeze_passes(tmp_path):
    root = fake_root(tmp_path)
    assert S8.verify_freeze(official_freeze(root, tmp_path / "f.json"), root) == []


def test_guard_blocks_locked_and_oot_without_lock(monkeypatch):
    seen = []
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: seen.append(a) or pd.DataFrame({"split": []}))
    for s in (["locked"], ["oot"], ["calib", "oot"], ["embargo"]):
        with pytest.raises(PermissionError):
            S8.read_rows("logged", s)
    S8._STATE.update(locked_open=True, lock_path=Path("/khong/ton/tai.lock"))   # cờ bật nhưng thiếu file khóa ⇒ vẫn chặn
    with pytest.raises(PermissionError):
        S8.read_rows("logged", ["locked"])
    assert seen == []                                               # bị chặn trước khi chạm file


def test_s8_reads_parquet_only_through_guard():
    tree = ast.parse((ROOT / "src/s8_mlops.py").read_text(encoding="utf-8"))
    bad = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for c in ast.walk(fn):
            if isinstance(c, ast.Attribute) and c.attr in ("read_parquet", "read_table", "ParquetFile") and fn.name != "read_rows":
                bad.append((fn.name, c.attr))
    assert not bad, bad


def test_dry_run_does_not_touch_locked_oot(monkeypatch, tmp_path):
    calls = []
    orig = pd.read_parquet

    def spy(path, *a, **k):
        calls.append((str(path), k.get("filters")))
        return orig(path, *a, **k)
    monkeypatch.setattr(pd, "read_parquet", spy)
    monkeypatch.setattr(S8.S4, "N_BOOT", 20)
    S8.ACCESS_LOG.clear()
    real_before = _real_state()
    try:
        out = S8.run_gates("dry-run", out_root=tmp_path, n_lat=1, n_boot_g2=10, manifest=False)
    finally:   # highspy khởi tạo bộ lập lịch luồng toàn cục; reset để PuLP+HiGHS (threads=1) ở test khác không bị NotSolved
        import highspy
        highspy.Highs.resetGlobalScheduler(True)
    data_reads = [c for c in calls if "data/processed" in c[0]]
    assert data_reads, "phải có đọc dữ liệu"
    for path, filt in data_reads:                                   # mọi lần đọc đều có bộ lọc split, chỉ split được phép
        assert filt and filt[0][0] == "split", path
        assert set(filt[0][2]) <= {"train", "valid", "calib"}, (path, filt)
    assert all(not ({"locked", "oot"} & set(r["splits"])) for r in S8.ACCESS_LOG)
    assert any(r["file"] == "oracle" and r["splits"] == ("calib",) for r in S8.ACCESS_LOG)
    assert _real_state() == real_before                              # không tạo/đổi khóa hay bảng gate locked thật
    g = out["gates"]
    assert (g.nhan.str.startswith("THỬ")).all()
    assert set(out["gate_summary"].columns) >= {"G1", "G2", "G3", "G4a", "G4b", "G5", "G6", "G7", "G4"}
    c = out["consistency"].set_index("doi_chieu").lech_lon_nhat
    assert c.iloc[0] < 1e-6 and c.iloc[3] < 1e-9                    # tái lập đúng GĐ7 v2.1 trên calib


def test_locked_path_rehearsal(monkeypatch, tmp_path):
    """Diễn tập ĐƯỜNG locked end-to-end mà KHÔNG đọc locked thật: mọi bộ lọc split 'locked' được đổi thành 'calib' TRƯỚC khi đọc,
    rồi gán lại split = 'locked'. Giả lập ~15 % khách là entity-holdout (hash khách) để chạy nhánh mô tả entity-holdout."""
    orig = pd.read_parquet
    swapped, others = [], []

    def swap(path, *a, **k):
        f = k.get("filters")
        if f and "locked" in f[0][2]:
            assert list(f[0][2]) == ["locked"], f                   # s8 chỉ xin locked riêng, không trộn split
            df = orig(path, *a, **dict(k, filters=[("split", "in", ["calib"])]))
            df["split"] = "locked"
            if "is_entity_holdout" in df.columns:
                eh = df.customer_id.map(lambda c: S8.S7.hash_unit("dien-tap-locked", c) < 0.15).to_numpy()
                df["is_entity_holdout"] = eh.astype(df["is_entity_holdout"].dtype)
            swapped.append(Path(path).name)
            return df
        assert not f or not ({"locked", "oot"} & set(f[0][2])), (path, f)
        others.append(Path(path).name)
        return orig(path, *a, **k)

    monkeypatch.setattr(pd, "read_parquet", swap)
    monkeypatch.setattr(S8.S4, "N_BOOT", 20)
    lock = tmp_path / "s8_locked_opened.lock"
    lock.write_text("{}")
    real_before = _real_state()
    S8._STATE.update(locked_open=True, lock_path=lock)
    try:
        out = S8.run_gates("locked", out_root=tmp_path, n_lat=1, n_boot_g2=10, manifest=False)
    finally:
        import highspy
        highspy.Highs.resetGlobalScheduler(True)
    assert sorted(set(swapped)) == ["logged.parquet", "oracle.parquet", "snapshots.parquet"]
    assert "reasons_s3.parquet" not in swapped + others             # đường locked không dựa vào reasons_s3 (tính lại mã lý do)
    T = tmp_path / "outputs/tables"
    assert (T / "s8_locked_gates.csv").exists() and not (T / "s8_locked_consistency.csv").exists()
    assert _real_state() == real_before                              # không ghi vào khóa / bảng gate locked THẬT
    g = out["gates"]
    assert g.nhan.str.startswith("CHÍNH THỨC").all()
    assert set(g.quan_the) == {"toan_bo", "khach_thuong", "entity_holdout"}
    # nhánh entity-holdout thực sự được tính
    e2 = out["g2_calibration"].set_index("quan_the").loc["entity_holdout"]
    assert e2.n_a0 > 100 and np.isfinite(e2.ece_hieu_chinh)
    e4 = out["g4_policy_value"].query("quan_the == 'entity_holdout'")
    assert e4.n_dong.gt(0).all() and np.isfinite(e4.regret_A).all()
    assert out["g3_qini"].query("quan_the == 'entity_holdout'").n.gt(0).all()
    # G1: kiểm tra entity holdout phải BẮT được khách 'holdout' giả lập vốn có trong tập mô hình; mọi kiểm tra khác đạt
    g1 = out["g1_contract"].set_index("kiem_tra").dat
    assert not g1["Entity holdout không xuất hiện ở train/valid/calib"]
    assert g1.drop("Entity holdout không xuất hiện ở train/valid/calib").all()
    # phân tách quần thể cộng đúng
    v = out["g4_policy_value"].pivot(index="chinh_sach", columns="quan_the", values="gia_tri_hien_thuc")
    assert np.allclose(v.toan_bo, v.khach_thuong + v.entity_holdout)
    assert (out["policy_value_by_period"].t0.nunique() == 3) and out["g5_constraint_audit"].n_vi_pham.sum() == 0
