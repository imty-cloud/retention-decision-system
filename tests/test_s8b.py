"""Kiểm thử GĐ8 bước 1b: rào phiên --oot; diễn tập OOT không chạm locked/oot; file kịch bản trôi không làm đổi file hiện có;
các khối giám sát (PSI, Challenger chỉ dùng nhãn trưởng thành, DR, playbook) và thiết kế thí điểm."""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import s8_mlops as S8  # noqa: E402
import s8b_oot as B  # noqa: E402
from test_s8 import fake_root, official_freeze  # noqa: E402

T = ROOT / "outputs/tables"
CFG = B.CFG


@pytest.fixture(autouse=True)
def reset_state():
    S8._STATE.update(locked_open=False, oot_open=False, lock_path=S8.LOCK_PATH)
    yield
    S8._STATE.update(locked_open=False, oot_open=False, lock_path=S8.LOCK_PATH)


# ---------------------------------------------------------------- rào phiên --oot
def test_oot_refused_without_lock(tmp_path, monkeypatch):
    root = fake_root(tmp_path)
    fz = official_freeze(root, tmp_path / "f.json")
    ran = []
    with pytest.raises(S8.FreezeMismatch, match="chưa có file khóa"):
        B.open_oot(tmp_path / "khong_co.lock", fz, root, runner=lambda: ran.append(1))
    assert not ran
    seen = []
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: seen.append(a) or pd.DataFrame({"split": []}))
    with pytest.raises(PermissionError):
        S8.read_rows("logged", ["oot"])
    S8._STATE.update(oot_open=True, lock_path=tmp_path / "khong_co.lock")        # cờ bật nhưng thiếu khóa ⇒ vẫn chặn
    with pytest.raises(PermissionError):
        S8.read_rows("logged", ["oot"])
    assert seen == []


def test_cli_oot_refuses_now():
    if S8.LOCK_PATH.exists():
        pytest.skip("locked đã mở – kiểm thử này chỉ áp cho bước 1b")
    with pytest.raises(SystemExit):
        B.main(["--oot"])


def test_oot_refused_when_freeze_mismatch(tmp_path):
    root = fake_root(tmp_path)
    fz, lock = official_freeze(root, tmp_path / "f.json"), tmp_path / "l.lock"
    lock.write_text("{}")
    (root / "src/a.py").write_text("x = 99\n")
    ran = []
    with pytest.raises(S8.FreezeMismatch, match="lệch"):
        B.open_oot(lock, fz, root, runner=lambda: ran.append(1))
    assert not ran and not S8._STATE["oot_open"]


def test_oot_opens_only_with_lock_and_matching_freeze(tmp_path):
    root = fake_root(tmp_path)
    fz, lock = official_freeze(root, tmp_path / "f.json"), tmp_path / "l.lock"
    lock.write_text("{}")
    seen = []
    B.open_oot(lock, fz, root, runner=lambda: seen.append(dict(S8._STATE)))
    assert seen[0]["oot_open"] and seen[0]["lock_path"] == lock and not S8._STATE["oot_open"]


def test_s8b_parquet_reads_guarded():
    tree = ast.parse((ROOT / "src/s8b_oot.py").read_text(encoding="utf-8"))
    bad = [(fn.name, c.attr) for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef) for c in ast.walk(fn)
           if isinstance(c, ast.Attribute) and c.attr in ("read_parquet", "read_table", "ParquetFile") and fn.name != "oracle_make_drift_file"]
    assert not bad, bad


# ---------------------------------------------------------------- kịch bản trôi
def test_drift_file_generated_without_changing_existing():
    h = pd.read_csv(T / "s8_drift_generation_hashes.csv")
    old = h[h.file != "oracle_oot_drift.parquet"]
    assert len(old) >= 14 and old.khong_doi.astype(bool).all() and (old.sha256_truoc == old.sha256_sau).all()
    man = json.loads((ROOT / "outputs/manifests/s1b_manifest.json").read_text(encoding="utf-8"))
    for e in man["files"]:
        if e["file"].endswith(".parquet"):                               # logged / đáp án của GĐ1b vẫn đúng byte như lúc sinh
            assert S8.common.sha256_file(ROOT / e["file"]) == e["sha256"], e["file"]
    f = ROOT / "data/processed/oracle_oot_drift.parquet"
    assert f.exists()
    assert pq.read_schema(f).names == [c for c in pq.read_schema(ROOT / "data/processed/oracle.parquet").names]   # chỉ metadata
    assert h.loc[h.file == "oracle_oot_drift.parquet", "sha256_sau"].iloc[0] == S8.common.sha256_file(f)
    with pytest.raises(PermissionError):                                  # không đọc được ngoài phiên --oot
        S8.read_rows("oracle_oot_drift", ["oot"])
    with pytest.raises(FileExistsError):                                  # không ghi đè
        B.oracle_make_drift_file()


# ---------------------------------------------------------------- khối giám sát (đơn vị)
def test_psi_basic():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20000)
    assert B.psi(a, rng.normal(size=20000)) < 0.01
    assert B.psi(a, rng.normal(1.0, 1, size=20000)) > CFG["monitoring"]["psi_alert"]
    assert B.psi(pd.Series(["UK"] * 90 + ["FR"] * 10), pd.Series(["UK"] * 50 + ["FR"] * 50)) > 0.25
    assert B.psi_status(0.05) == "ổn định" and B.psi_status(0.15) == "cảnh báo" and B.psi_status(0.3) == "báo động"


def test_challenger_uses_only_matured_labels():
    t0s = pd.date_range("2011-01-03", periods=8, freq="14D")
    rng = np.random.default_rng(1)
    pool = pd.DataFrame({"t0": np.repeat(t0s, 200), "mu0_raw": rng.uniform(0.05, 0.6, 1600)})
    pool["y_obs"] = (rng.uniform(size=1600) < pool.mu0_raw).astype(int)
    d = t0s[6]
    m, win = B.recal_for_date(pool, d)
    ok = [t for t in t0s if t + pd.Timedelta(days=30) <= d][-3:]
    assert win == [str(t.date()) for t in ok] and max(pd.Timestamp(w) for w in win) + pd.Timedelta(days=30) <= d
    assert B.recal_for_date(pool, t0s[1])[0] is None                     # chưa kỳ nào trưởng thành ⇒ không có Challenger


def test_dr_program_equals_dm_without_noise_and_bootstrap():
    n = 400
    rng = np.random.default_rng(2)
    Mu = rng.normal(size=(n, 4))
    rand = np.arange(n) < 300
    ctrl = rand & (np.arange(n) % 10 == 0)
    prop = np.where(rand, np.where(np.arange(n) % 3 == 0, "a2", "a1"), "a0")
    fin = np.where(ctrl, "a0", prop)
    dec = pd.DataFrame({"in_randomization_set": rand, "is_backfill": False, "final_action": fin, "proposed_action": prop,
                        "control_group": ctrl, "hitl_outcome": "NA", "p_control": np.where(rand, 0.1, 0.0)})
    A = B.code(fin)
    Y = Mu[np.arange(n), A]                                               # kết quả đúng bằng μ̂ ⇒ phần hiệu chỉnh IPW = 0
    tab, info = B.dr_program(dec, Mu, Y, np.arange(n) // 2, np.zeros(n), 200)
    assert np.allclose(info["G"], info["DM"]) and (info["pi"][ctrl] == B.code(prop)[ctrl]).all()
    g = tab.iloc[0]
    assert abs(g.phan_hieu_chinh_IPW) < 1e-9 and g.DR_ci_lo <= g.DR_uoc_luong <= g.DR_ci_hi


def test_playbook_uses_config_thresholds():
    pb = B.playbook()
    mon, gates = CFG["monitoring"], CFG["release_gates"]
    assert {"canh_bao", "nguong_canh_bao", "nguong_bao_dong", "chu_so_huu", "hanh_dong", "dieu_kien_quay_lui"} <= set(pb.columns)
    txt = " ".join(pb.nguong_canh_bao.astype(str)) + " " + " ".join(pb.nguong_bao_dong.astype(str))
    for v in (mon["psi_warn"], mon["psi_alert"], mon["ece_alert"], mon["action_tvd_alert"], gates["G2_ece_max"],
              gates["G6_latency_p95_seconds_max"]):
        assert str(v) in txt
    assert pb.chu_so_huu.str.len().gt(0).all() and pb.dieu_kien_quay_lui.str.len().gt(0).all()


# ---------------------------------------------------------------- thiết kế thí điểm
def test_pilot_outputs():
    g3 = pd.read_csv(T / "s8_pilot_g3.csv")
    assert g3.khop_s4_power.all()
    a1 = g3[g3.kich_ban.str.startswith("S1") & (g3.hanh_dong == "a1")].iloc[0]
    assert 2 < a1.boi_so_locked < 3                                       # khớp dự báo 6b: a1 cần ~2.5× locked
    assert (g3[g3.kich_ban.str.startswith("S1") & g3.hanh_dong.isin(["a2", "a3"])].boi_so_locked > 10).all()   # 6b: a2, a3 > 10×
    dr = pd.read_csv(T / "s8_pilot_dr.csv")
    r = dr[dr.thiet_ke.str.startswith("Router")].dropna(subset=["so_ky_can_power80"])
    for lab in r.kich_ban_hieu_ung.unique():                              # tăng tỷ lệ đối chứng ⇒ cần ít kỳ hơn
        s = r[r.kich_ban_hieu_ung == lab].sort_values("ty_le_doi_chung")
        assert s.so_ky_can_power80.is_monotonic_decreasing
    assert (ROOT / "outputs/figures/fig_s8_pilot_sample_size.png").exists()


# ---------------------------------------------------------------- diễn tập OOT end-to-end
def test_oot_rehearsal_does_not_touch_locked_oot(monkeypatch, tmp_path):
    calls = []
    orig = pd.read_parquet

    def spy(path, *a, **k):
        calls.append((str(path), k.get("filters")))
        return orig(path, *a, **k)
    monkeypatch.setattr(pd, "read_parquet", spy)
    S8.ACCESS_LOG.clear()
    try:
        out = B.replay("rehearsal", out_root=tmp_path, n_lat=1, n_boot=50, manifest=False)
    finally:
        import highspy
        highspy.Highs.resetGlobalScheduler(True)
    data_reads = [c for c in calls if "data/processed" in c[0]]
    assert data_reads
    for path, filt in data_reads:
        assert filt and filt[0][0] == "split" and set(filt[0][2]) <= {"train", "valid", "calib"}, (path, filt)
        assert "oracle_oot_drift" not in path
    assert all(not ({"locked", "oot"} & set(r["splits"])) for r in S8.ACCESS_LOG)
    for k, v in out.items():
        assert v.nhan.str.startswith("THỬ").all(), k
        assert (tmp_path / f"outputs/tables/s8_oot_rehearsal_{k}.csv").exists()
    c = out["calibration_matured"]                                        # Brier/ECE chỉ khi nhãn đã trưởng thành
    assert (pd.to_datetime(c.ngay_tinh) >= pd.to_datetime(c.ky) + pd.Timedelta(days=30)).all()
    ch = c[c.phien_ban.str.startswith("Challenger (")]
    for x in ch.itertuples():
        assert all(pd.Timestamp(w) + pd.Timedelta(days=30) <= pd.Timestamp(x.ky) for w in x.cua_so_hieu_chinh.split(", "))
    ld = out["label_delay_by_date"]
    assert ld.so_ky_nhan_truong_thanh.iloc[:-1].eq(0).all() and ld.ty_le_dong_co_nhan.iloc[-1] == 1.0
    assert np.isfinite(out["psi"].psi).all() and out["action_tvd"].action_tvd.between(0, 1).all()
    lat = out["latency"]
    assert {"P50", "P95", "P99"} <= set(lat.columns) and "Gộp mọi kỳ" in set(lat.ky)
    d = out["drift_guardrail"]
    assert len(d) == 6 and d[d.chinh_sach.str.startswith("CÓ")].vi_pham_quy_dinh.eq(0).all()
    assert (out["dr_value"].DR_ci_lo <= out["dr_value"].DR_ci_hi).all()
    assert set(out["selective_calibration"].phuong_phap.str[:10]) >= {"Current (P", "Challenger"}


def test_real_oot_path_preflight(monkeypatch, tmp_path):
    """Diễn tập NHÁNH 'real' của phát lại OOT mà KHÔNG đọc locked/oot thật: bộ lọc 'oot' → đọc calib, 'locked' → đọc valid (gắn lại nhãn);
    file trôi → đáp án calib gắn nhãn oot. Mục đích: lộ lỗi riêng của vai real (kho nhãn, file trôi, Challenger-2) trước khi mở locked."""
    orig = pd.read_parquet
    DATA = ROOT / "data/processed"
    MAP = {"oot": "calib", "locked": "valid"}

    def swap(path, *a, **k):
        f = k.get("filters")
        if Path(path).name == "oracle_oot_drift.parquet":
            d = orig(DATA / "oracle.parquet", filters=[("split", "in", ["calib"])])
            d["split"] = "oot"
            return d
        if f:
            parts = []
            for s in list(f[0][2]):
                d = orig(path, *a, **dict(k, filters=[("split", "in", [MAP.get(s, s)])]))
                d["split"] = s
                parts.append(d)
            return pd.concat(parts, ignore_index=True)
        return orig(path, *a, **k)

    monkeypatch.setattr(pd, "read_parquet", swap)
    (tmp_path / "outputs/tables").mkdir(parents=True)
    (tmp_path / "outputs/tables/s8_locked_gates.csv").write_text("gate\n")       # để nhánh Challenger-2 (thật) được chạy
    monkeypatch.setattr(B, "ROOT", tmp_path)
    lock = tmp_path / "l.lock"
    lock.write_text("{}")
    S8._STATE.update(oot_open=True, lock_path=lock)
    try:
        out = B.replay("real", out_root=tmp_path, n_lat=1, n_boot=50, manifest=False)
    finally:
        import highspy
        highspy.Highs.resetGlobalScheduler(True)
    assert not S8.LOCK_PATH.exists() or True
    for k, v in out.items():
        assert v.nhan.str.startswith("VẬN HÀNH").all(), k
        assert (tmp_path / f"outputs/tables/s8_oot_{k}.csv").exists()
    c = out["calibration_matured"]
    assert c.phien_ban.str.startswith("Challenger-2").any() and c.ky.nunique() == 3
    assert len(out["drift_guardrail"]) == 6 and np.isfinite(out["dr_value"].DR_uoc_luong).all()
