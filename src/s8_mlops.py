"""GĐ8 – Triển khai, đo lường thực chứng, MLOps (Chương 8) · BƯỚC 1a: CHUẨN BỊ PHẦN GATE.

Chế độ chạy
  python src/s8_mlops.py --dry-run       Diễn tập: CALIB đóng vai "locked" → bảng gate gắn nhãn THỬ. Không đọc locked/oot.
  python src/s8_mlops.py --freeze-draft  Manifest đóng băng BẢN NHÁP (s8_freeze_draft) + requirements_lock.txt + ma trận tương thích.
  python src/s8_mlops.py --freeze        Manifest đóng băng CHÍNH THỨC (s8_freeze) – CHỈ chạy sau bước 1b.
  python src/s8_mlops.py --open-locked   Mở locked ĐÚNG MỘT LẦN. Từ chối nếu đã có outputs/manifests/s8_locked_opened.lock,
                                         hoặc nếu hash trong s8_freeze không khớp src/ configs/ models/ CLAUDE.md (và phiên bản
                                         thư viện chính) hiện tại. File khóa được ghi (tạo nguyên tử, O_EXCL) TRƯỚC khi đọc
                                         bất kỳ dòng locked nào và KHÔNG BAO GIỜ bị xóa – kể cả khi lần chạy lỗi giữa chừng.

Định nghĩa gate (CLAUDE.md mục 6b – đăng ký trước; ngưỡng lấy nguyên từ `release_gates`, không sửa):
  G1 data contract: schema snapshot của GĐ1 + hợp đồng logged + không rò rỉ thời gian + entity holdout + embargo
     + hợp đồng đầu vào Router ở mọi kỳ.
  G2 ECE (15 bin equal-mass) ≤ G2_ece_max và Brier(đã hiệu chỉnh) ≤ Brier(thô), trên dòng A_logged = a0, calibrator Current
     (Platt GĐ2, không khớp lại).
  G3 số hành động có Qini > 0 với CI bootstrap 95 % (theo cụm khách) > 0 ≥ G3_qini_positive_actions_min – nhánh ngẫu nhiên của
     dữ liệu ghi log, τ̂ = CausalForestDML (đã chốt ở GĐ4).
  G4 quy mô toàn bộ một t0 (B, K không co), cộng dồn tử/mẫu qua các kỳ, lợi nhuận gia tăng HIỆN THỰC (đáp án A):
     G4a (chính sách − đối chứng) ≥ 0.10 × |đối chứng|; CHÍNH = MILP qua Router v2.1 vs Luật cũ qua Router v2.1, PHỤ = vs luật cũ thuần.
     G4b regret chuẩn hóa A ≤ G4_normalized_regret_max – chấm trên chính sách TRIỂN KHAI (MILP qua Router v2.1); MILP thuần báo kèm.
  G5 số vi phạm ràng buộc trên quyết định GỬI THỰC TẾ của Current (bảng kiểm toán ràng buộc của Router v2.1) ≤ G5_constraint_violations_max.
  G6 P95 độ trễ đầu–cuối một t0 (đặc trưng → mã lý do → chấm điểm → hợp đồng & miền → MILP → router), ≥ 30 lượt mỗi kỳ.
  G7 mỗi phân khúc UK / non-UK: (chính sách − đối chứng) / |đối chứng| ≥ −G7_segment_value_drop_max; so sánh chính/phụ như G4a.
Quần thể: toàn bộ (CHẤM CHÍNH THỨC) · khách thường (is_entity_holdout = 0) · entity-holdout (= 1, chỉ mô tả).
Mỗi gate đặt cạnh dự báo đăng ký trước tương ứng trong 6b (cột du_bao_dang_ky_truoc).

Rào dữ liệu: mọi đọc parquet đi qua `read_rows` (lọc split ở tầng pyarrow). locked chỉ đọc được sau khi file khóa tồn tại
trong lần chạy --open-locked; oot bị chặn hoàn toàn ở bước 1a. Đáp án chỉ đọc trong hàm evaluate_* (luật cứng #1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pandera.pandas as pa
except ImportError:  # pandera < 0.24
    import pandera as pa

import common
import s2_predict as S2
import s3_explain as S3
import s4_causal as S4
import s5_spo as S5
import s6_milp as S6
import s7_router as S7
from s1_data import FEATURES_CAT, FEATURES_NUM, snapshot_schema
from s5b_dr_diagnostics import legacy_scores

CFG = common.load_config()
SEED = CFG["project"]["seed"]
GATES = CFG["release_gates"]
ROOT = common.ROOT
TABLES, FIGS = Path(CFG["paths"]["tables"]), Path(CFG["paths"]["figures"])
MANIFESTS = ROOT / CFG["paths"]["manifests"]
LOCK_PATH = MANIFESTS / "s8_locked_opened.lock"
FREEZE_PATH = MANIFESTS / "s8_freeze_manifest.json"
DRAFT_PATH = MANIFESTS / "s8_freeze_draft_manifest.json"
REQ_LOCK = MANIFESTS / "requirements_lock.txt"
DATA_DIR = ROOT / "data/processed"
ACTIONS, TREATED = S5.ACTIONS, S5.TREATED
M0 = CFG["business"]["gross_margin_rate"]
H_DAYS = CFG["decision"]["horizon_days"]
MODEL_SPLITS = ("train", "valid", "calib")
N_LAT = 30
N_BOOT_G2 = 300
S_WIDE_EXPECTED = 1833.28                       # ngưỡng Ŝ của GĐ7 (giữ nguyên)
KEY_PKGS = ["pandas", "numpy", "pyarrow", "pandera", "scikit-learn", "lightgbm", "econml", "doubleml", "scikit-uplift",
            "shap", "highspy", "pulp", "torch", "pyepo", "matplotlib", "joblib", "pyyaml"]
POL_MILP_PURE, POL_LEG_PURE = "MILP thuần (đề xuất, miền v1)", "Luật cũ thuần (miền v1)"
POL_MILP_R, POL_LEG_R = "MILP qua Router v2.1", "Luật cũ qua Router v2.1"
POLICIES = [POL_MILP_PURE, POL_LEG_PURE, POL_MILP_R, POL_LEG_R]
POPS = {"toan_bo": "Toàn bộ (chấm chính thức)", "khach_thuong": "Khách thường (is_entity_holdout=0)",
        "entity_holdout": "Entity-holdout (is_entity_holdout=1, mô tả)"}
COMPARISONS = (("CHÍNH: vs Luật cũ qua Router v2.1", POL_LEG_R), ("PHỤ: vs Luật cũ thuần", POL_LEG_PURE))

# Dự báo / tham chiếu đã ĐĂNG KÝ TRƯỚC (trích CLAUDE.md 6b và §6) – đặt cạnh từng gate.
PRED = {
    "G1": "6b không có dự báo riêng. Tham chiếu GĐ1: data contract + 6 kiểm thử rò rỉ đều PASS; embargo calib→locked 42 ngày PASS.",
    "G2": "6b: Platt khớp trên calib (tháng 4–5) CÓ THỂ KHÔNG hợp locked (tháng 7–8). Tham chiếu §6: ECE valid 0.047 > 0.03 (trượt), "
          "ECE ngoài fold trong calib 0.028. Không có xác suất dự báo bằng số.",
    "G3": "6b: P(G3 đạt) ≈ 4–7 % (kịch bản chính), 10–13 % (lạc quan), 17–21 % (train ngoài fold) ⇒ nhiều khả năng KHÔNG đạt "
          "(do cỡ mẫu nhánh ngẫu nhiên); entity-holdout < 3 %, chỉ mô tả.",
    "G4a": "6b: KHẢ THI. valid+calib (Router v2): MILP qua Router £21 483 vs Luật cũ qua Router £15 006 vs luật cũ thuần −£2 650 "
           "⇒ đạt cả hai cách so.",
    "G4b": "6b: gần như chắc TRƯỢT – ngay chính sách biết kỳ vọng thật (đáp án B) có regret A = 0.742 ± 0.048 (lô 100, calib).",
    "G5": "6b (Router v2.1): bước sửa khả thi cuối nhằm 0 vi phạm; valid+calib v2.1 = 0 vi phạm. Không có xác suất dự báo.",
    "G6": "6b không có dự báo riêng. Tham chiếu GĐ7 v2.1: P95 đầu–cuối = 0.75 s ≪ 30 s.",
    "G7": "6b: so sánh chính như G4a. Tham chiếu valid+calib v2.1: UK +58 %, non-UK +21 % vs Luật cũ qua Router ⇒ đạt.",
}


# ============================================================================ rào dữ liệu
_STATE: dict = {"locked_open": False, "oot_open": False, "lock_path": LOCK_PATH}
ACCESS_LOG: list[dict] = []


def read_rows(name: str, splits, columns: list[str] | None = None) -> pd.DataFrame:
    """CỬA DUY NHẤT để s8 đọc parquet trong data/processed: chỉ các split được phép, lọc ở tầng pyarrow (filters)."""
    splits = sorted(set(splits))
    lock_ok = Path(_STATE["lock_path"]).exists()
    for s in splits:
        if s == "oot" and not (_STATE.get("oot_open") and lock_ok):
            raise PermissionError("split oot chỉ đọc được trong phiên --oot (file khóa locked ĐÃ tồn tại và verify_freeze khớp).")
        if s == "locked" and not ((_STATE["locked_open"] or _STATE.get("oot_open")) and lock_ok):
            raise PermissionError("split locked chỉ đọc được trong lần chạy --open-locked (sau khi ghi khóa) hoặc phiên --oot.")
        if s not in MODEL_SPLITS + ("locked", "oot"):
            raise PermissionError(f"split '{s}' không được dùng (embargo / excluded_entity_holdout).")
    cols = None if columns is None else list(dict.fromkeys(list(columns) + ["split"]))
    ACCESS_LOG.append({"file": name, "splits": tuple(splits)})
    df = pd.read_parquet(DATA_DIR / f"{name}.parquet", columns=cols, filters=[("split", "in", splits)])
    assert set(df["split"].astype(str).unique()) <= set(splits), "bộ lọc split không có hiệu lực"
    return df.reset_index(drop=True)


# ============================================================================ đóng băng & khóa
class LockedAlreadyOpened(RuntimeError):
    pass


class FreezeMismatch(RuntimeError):
    pass


def freeze_scope(root: Path = ROOT) -> list[Path]:
    """Phạm vi đóng băng: src/*.py, mọi file trong configs/ và models/, CLAUDE.md."""
    root = Path(root)
    files = sorted((root / "src").glob("*.py"))
    for d in ("configs", "models"):
        files += sorted(f for f in (root / d).rglob("*") if f.is_file())
    files.append(root / "CLAUDE.md")
    return [f for f in files if f.exists() and "__pycache__" not in f.parts and f.name != ".DS_Store"]


def hash_scope(root: Path = ROOT) -> list[dict]:
    root = Path(root)
    return [{"file": f.relative_to(root).as_posix(), "bytes": f.stat().st_size, "sha256": common.sha256_file(f)}
            for f in freeze_scope(root)]


DATA_SCOPE_EXCLUDE = "decisions_s8_"          # đầu ra của chính s8 (dry-run / locked) – không phải đầu vào cần đóng băng


def data_scope(root: Path = ROOT) -> list[Path]:
    """Dữ liệu đóng băng: data/processed/*.parquet (trừ đầu ra của s8). Băm theo BYTE – KHÔNG parse parquet,
    nên không đọc nội dung dòng nào (kể cả locked/oot)."""
    d = Path(root) / "data/processed"
    return sorted(f for f in d.glob("*.parquet") if f.is_file() and not f.name.startswith(DATA_SCOPE_EXCLUDE)) if d.exists() else []


def hash_data(root: Path = ROOT) -> list[dict]:
    root = Path(root)
    return [{"file": f.relative_to(root).as_posix(), "bytes": f.stat().st_size, "sha256": common.sha256_file(f)}
            for f in data_scope(root)]


def env_versions() -> dict:
    return common.package_versions(KEY_PKGS)


def aggregate_hash(entries: list[dict]) -> str:
    return hashlib.sha256("\n".join(f"{e['file']}:{e['sha256']}" for e in entries).encode()).hexdigest()


def build_freeze(stage: str, root: Path = ROOT, extra: dict | None = None) -> dict:
    entries, data = hash_scope(root), hash_data(root)
    return {"stage": stage, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scope": ["src/*.py", "configs/**", "models/**", "CLAUDE.md"], "files": entries,
            "aggregate_sha256": aggregate_hash(entries),
            "data_scope": f"data/processed/*.parquet (trừ {DATA_SCOPE_EXCLUDE}*; băm theo byte, không parse)", "data_files": data,
            "data_aggregate_sha256": aggregate_hash(data), "environment": env_versions(), "extra": extra or {}}


def _diff_entries(old_list: list[dict], cur_list: list[dict], kind: str) -> list[str]:
    old = {e["file"]: e["sha256"] for e in old_list}
    cur = {e["file"]: e["sha256"] for e in cur_list}
    out = []
    for f in sorted(set(old) | set(cur)):
        if f not in cur:
            out.append(f"{kind} {f}: có trong manifest nhưng không còn")
        elif f not in old:
            out.append(f"{kind} {f}: file mới, không có trong manifest")
        elif old[f] != cur[f]:
            out.append(f"{kind} {f}: SHA-256 lệch")
    return out


def verify_freeze(freeze_path: Path = FREEZE_PATH, root: Path = ROOT, check_env: bool = True) -> list[str]:
    """Danh sách điểm lệch giữa manifest đóng băng CHÍNH THỨC và trạng thái hiện tại (rỗng = khớp)."""
    freeze_path = Path(freeze_path)
    if not freeze_path.exists():
        return [f"thiếu manifest đóng băng chính thức {freeze_path.name} (chạy --freeze sau bước 1b)"]
    man = json.loads(freeze_path.read_text(encoding="utf-8"))
    out = [] if man.get("stage") == "s8_freeze" else [f"manifest có stage='{man.get('stage')}', không phải 's8_freeze'"]
    out += _diff_entries(man.get("files", []), hash_scope(root), "mã/mô hình/config")
    out += _diff_entries(man.get("data_files", []), hash_data(root), "dữ liệu")
    if check_env:
        now = env_versions()
        for k in ["python"] + KEY_PKGS:
            if man.get("environment", {}).get(k) != now.get(k):
                out.append(f"môi trường {k}: manifest {man.get('environment', {}).get(k)} ≠ hiện tại {now.get(k)}")
    return out


def acquire_lock(lock_path: Path, info: dict) -> None:
    """Tạo file khóa NGUYÊN TỬ (O_EXCL): nếu đã có thì thất bại, không bao giờ ghi đè."""
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError as e:
        raise LockedAlreadyOpened(f"locked đã được mở trước đó ({lock_path}).") from e
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def open_locked(lock_path: Path = LOCK_PATH, freeze_path: Path = FREEZE_PATH, root: Path = ROOT, runner=None):
    """Mở locked đúng MỘT lần: kiểm khóa → kiểm hash đóng băng → GHI KHÓA → mới chạy (đọc locked)."""
    lock_path, freeze_path = Path(lock_path), Path(freeze_path)
    if lock_path.exists():
        raise LockedAlreadyOpened(f"Từ chối: đã có {lock_path.name} – locked chỉ được mở một lần (luật cứng #4).")
    mism = verify_freeze(freeze_path, root)
    if mism:
        raise FreezeMismatch("Từ chối: trạng thái hiện tại không khớp manifest đóng băng:\n  " + "\n  ".join(mism[:30]))
    man = json.loads(freeze_path.read_text(encoding="utf-8"))
    acquire_lock(lock_path, {"opened_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "freeze_aggregate_sha256": man["aggregate_sha256"], "pid": os.getpid(),
                             "ghi_chu": "Ghi TRƯỚC khi đọc locked; không xóa. Không chỉnh mô hình sau khi xem kết quả locked."})
    _STATE.update(locked_open=True, lock_path=lock_path)
    try:
        return (runner or (lambda: run_gates("locked")))()
    except Exception:                                   # khóa đã ghi ⇒ không chạy lại được: lưu dấu vết lỗi để báo cáo trung thực
        import traceback
        (lock_path.parent / "s8_locked_error.log").write_text(
            f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\n{traceback.format_exc()}", encoding="utf-8")
        raise
    finally:
        _STATE["locked_open"] = False


# ============================================================================ requirements & ma trận tương thích
def pip_freeze() -> str:
    try:
        return subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
    except Exception:  # dự phòng khi không có pip
        return "\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())) + "\n"


def compat_matrix() -> pd.DataFrame:
    from packaging.version import Version
    req = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if ">=" in line:
            n, v = line.split(">=")
            req[n.strip().lower()] = v.strip()
    env = env_versions()
    rows = [{"thanh_phan": "python", "yeu_cau": "3.11 (khuyến nghị, CLAUDE.md §7)", "phien_ban_chay_thuc_te": env["python"],
             "dat_yeu_cau_toi_thieu": None,
             "ghi_chu": f"Toàn bộ pipeline GĐ1–GĐ8 chạy thực tế trên Python {env['python']} (KHÔNG phải 3.11); chưa kiểm trên 3.11."},
            {"thanh_phan": "nen_tang", "yeu_cau": "macOS Apple Silicon (CLAUDE.md §7)", "phien_ban_chay_thuc_te": env["platform"],
             "dat_yeu_cau_toi_thieu": True, "ghi_chu": "HiGHS qua highspy; LightGBM cần libomp (brew)."}]
    for n in sorted(set(req) | set(KEY_PKGS)):
        v = env.get(n) or (metadata.version(n) if n in {d.metadata["Name"].lower() for d in metadata.distributions()} else "not-installed")
        ok = None
        if n in req and v != "not-installed":
            ok = bool(Version(v) >= Version(req[n]))
        rows.append({"thanh_phan": n, "yeu_cau": f">={req[n]}" if n in req else "(không ghi trong requirements.txt)",
                     "phien_ban_chay_thuc_te": v, "dat_yeu_cau_toi_thieu": ok, "ghi_chu": ""})
    return pd.DataFrame(rows)


def write_freeze(stage: str) -> Path:
    """stage = 's8_freeze_draft' (bản nháp) hoặc 's8_freeze' (chính thức, sau bước 1b)."""
    assert stage in ("s8_freeze_draft", "s8_freeze")
    if stage == "s8_freeze" and LOCK_PATH.exists():
        raise LockedAlreadyOpened("Không đóng băng lại sau khi locked đã mở.")
    REQ_LOCK.write_text(pip_freeze(), encoding="utf-8")
    cm = compat_matrix()
    cm.to_csv(common.p(TABLES / "s8_compat_matrix.csv"), index=False)
    man = build_freeze(stage, extra={
        "requirements_lock": REQ_LOCK.relative_to(ROOT).as_posix(), "requirements_lock_sha256": common.sha256_file(REQ_LOCK),
        "compat_matrix": cm.to_dict("records"), "seed": SEED, "release_gates": GATES,
        "ghi_chu": ("BẢN NHÁP – chưa phải đóng băng chính thức; --open-locked chỉ chấp nhận s8_freeze."
                    if stage == "s8_freeze_draft" else "ĐÓNG BĂNG CHÍNH THỨC – --open-locked đối chiếu với file này.")})
    out = DRAFT_PATH if stage == "s8_freeze_draft" else FREEZE_PATH
    out.write_text(json.dumps(man, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"{stage}: {len(man['files'])} file mã/mô hình/config (SHA-256 tổng {man['aggregate_sha256'][:16]}…) · "
          f"{len(man['data_files'])} file dữ liệu (SHA-256 tổng {man['data_aggregate_sha256'][:16]}…) → {out.relative_to(ROOT)}")
    print(cm.to_string(index=False))
    return out


# ============================================================================ G1 – hợp đồng dữ liệu
LOGGED_SCHEMA = pa.DataFrameSchema({
    "customer_id": pa.Column(nullable=False), "t0": pa.Column(nullable=False),
    "A_logged": pa.Column(nullable=False, checks=pa.Check.isin(ACTIONS)),
    "is_random_arm": pa.Column(nullable=False, checks=pa.Check.isin([0, 1])),
    **{f"propensity_{a}": pa.Column(float, pa.Check.in_range(0, 1, include_min=False)) for a in ACTIONS},
    "propensity_logged": pa.Column(float, pa.Check.in_range(0, 1, include_min=False)),
    "y_obs": pa.Column(nullable=False, checks=pa.Check.isin([0, 1])),
    "value_obs": pa.Column(float, pa.Check.ge(0)), "profit_obs": pa.Column(float, nullable=False),
    **{f: pa.Column(nullable=True) for f in FEATURES_NUM + FEATURES_CAT},
}, unique=["customer_id", "t0"], strict=False, checks=[
    pa.Check(lambda d: (d[[f"propensity_{a}" for a in ACTIONS]].sum(axis=1) - 1).abs() < 1e-6, error="propensity không cộng bằng 1"),
    pa.Check(lambda d: pd.Series(np.isclose(d.propensity_logged.to_numpy(), d[[f"propensity_{a}" for a in ACTIONS]].to_numpy()[
        np.arange(len(d)), pd.Categorical(d.A_logged, categories=ACTIONS).codes]), index=d.index),
             error="propensity_logged ≠ propensity của A_logged"),
    pa.Check(lambda d: (d.value_obs > 0) == (d.y_obs == 1), error="value_obs > 0 ⇔ y_obs = 1"),
    pa.Check(lambda d: pd.to_datetime(d.cutoff_ts) == pd.to_datetime(d.t0) - pd.Timedelta(days=CFG["decision"]["feature_latency_days"]),
             error="cutoff = t0 − 1 ngày"),
    pa.Check(lambda d: pd.to_datetime(d.max_feature_event_ts) < pd.to_datetime(d.cutoff_ts), error="sự kiện đặc trưng sau cutoff"),
    pa.Check(lambda d: pd.to_datetime(d.label_available_ts) == pd.to_datetime(d.t0) + pd.Timedelta(days=H_DAYS),
             error="nhãn trưởng thành = t0 + 30 ngày"),
])


def _schema_check(name: str, schema, df: pd.DataFrame) -> dict:
    try:
        schema.validate(df, lazy=True)
        return {"kiem_tra": name, "dat": True, "chi_tiet": f"{len(df)} dòng"}
    except pa.errors.SchemaErrors as e:
        fc = e.failure_cases
        return {"kiem_tra": name, "dat": False, "chi_tiet": f"{len(fc)} lỗi: " + "; ".join(fc["check"].astype(str).unique()[:5])}


def g1_contract(lg: pd.DataFrame, snap: pd.DataFrame, mod: pd.DataFrame, input_ok: list[bool]) -> pd.DataFrame:
    rows = [_schema_check("Schema snapshot GĐ1 (s1_data.snapshot_schema)", snapshot_schema(CFG), snap),
            _schema_check("Hợp đồng logged (propensity, nhãn quan sát, cutoff, độ trễ nhãn)", LOGGED_SCHEMA, lg)]
    k1 = set(zip(lg.customer_id.astype(str), lg.t0)); k2 = set(zip(snap.customer_id.astype(str), snap.t0))
    rows.append({"kiem_tra": "Khóa (khách, t0) logged = snapshot", "dat": k1 == k2 and len(k1) == len(lg) == len(snap),
                 "chi_tiet": f"logged {len(lg)} · snapshot {len(snap)} · chung {len(k1 & k2)}"})
    m = lg[["customer_id", "t0", "A_logged", "y_obs"]].merge(snap[["customer_id", "t0", "y"]], on=["customer_id", "t0"], how="left")
    a0 = m.A_logged == "a0"
    rows.append({"kiem_tra": "y_obs dưới a0 = hành vi thật Y(0) của snapshot", "dat": bool((m.y_obs[a0] == m.y[a0]).all()),
                 "chi_tiet": f"{int(a0.sum())} dòng a0"})
    eh_in_model = int((mod.is_entity_holdout == 1).sum())
    eh_ids = set(lg.customer_id[lg.is_entity_holdout == 1].astype(str))
    leak = len(eh_ids & set(mod.customer_id.astype(str)))
    rows.append({"kiem_tra": "Entity holdout không xuất hiện ở train/valid/calib", "dat": eh_in_model == 0 and leak == 0,
                 "chi_tiet": f"dòng holdout trong tập mô hình = {eh_in_model}; khách holdout của tập chấm trùng tập mô hình = {leak}; "
                             f"khách holdout trong tập chấm = {len(eh_ids)}"})
    emb = pd.read_csv(ROOT / TABLES / "s1_embargo_check.csv")
    rows.append({"kiem_tra": "Embargo giữa các khối ≥ độ trễ nhãn (s1_embargo_check)",
                 "dat": bool(emb["pass"].all() and (emb.gap_days >= H_DAYS).all()), "chi_tiet": f"gap tối thiểu {emb.gap_days.min()} ngày"})
    rows.append({"kiem_tra": "Hợp đồng đầu vào Router (INPUT_SCHEMA) mọi kỳ", "dat": bool(all(input_ok)) and len(input_ok) > 0,
                 "chi_tiet": f"{sum(input_ok)}/{len(input_ok)} kỳ"})
    return pd.DataFrame(rows)


# ============================================================================ G2 – hiệu chỉnh
def g2_calibration(lg, mu0_raw, mu0_cal, pops: dict, n_boot: int) -> pd.DataFrame:
    rows = []
    for pk, pm in pops.items():
        m = pm & (lg.A_logged.to_numpy() == "a0")
        r = {"quan_the": pk, "n_a0": int(m.sum())}
        if m.sum() < 30:
            rows.append(r); continue
        y, pr, pc = lg.y_obs.to_numpy()[m].astype(float), mu0_raw[m], mu0_cal[m]
        r.update(ty_le_mua_a0=y.mean(), p_cal_TB=pc.mean(), ece_tho=S2.ece_equal_mass(y, pr), ece_hieu_chinh=S2.ece_equal_mass(y, pc),
                 brier_tho=float(np.mean((pr - y) ** 2)), brier_hieu_chinh=float(np.mean((pc - y) ** 2)))
        draw = S4.cluster_resampler(lg.customer_id.to_numpy()[m], seed=SEED)
        e_b, d_b = [], []
        for _ in range(n_boot):
            i = draw()
            e_b.append(S2.ece_equal_mass(y[i], pc[i]))
            d_b.append(np.mean((pc[i] - y[i]) ** 2) - np.mean((pr[i] - y[i]) ** 2))
        r.update(ece_hieu_chinh_ci_lo=np.percentile(e_b, 2.5), ece_hieu_chinh_ci_hi=np.percentile(e_b, 97.5),
                 brier_chenh_ci_lo=np.percentile(d_b, 2.5), brier_chenh_ci_hi=np.percentile(d_b, 97.5))
        rows.append(r)
    return pd.DataFrame(rows)


# ============================================================================ G3 – Qini
def g3_qini(lg, tau, pops: dict) -> pd.DataFrame:
    out = []
    for pk, pm in pops.items():
        if pm.sum() == 0 or (lg.is_random_arm.to_numpy()[pm] == 1).sum() < 20:
            out.append(pd.DataFrame([{"quan_the": pk, "hanh_dong": a, "n": 0} for a in TREATED])); continue
        d = lg[pm].reset_index(drop=True)
        out.append(pd.DataFrame(S4.qini_table(d, tau[pm], pk, "CausalForestDML", boot=True)).assign(quan_the=pk))
    q = pd.concat(out, ignore_index=True)
    q["ci_duong"] = q.get("qini_ci_lo", pd.Series(np.nan, index=q.index)) > 0
    return q


# ============================================================================ đánh giá bằng đáp án (luật cứng #1)
def evaluate_truth_rows(keys: pd.DataFrame, splits) -> dict:
    """Đáp án của đúng các dòng `keys`, đọc qua read_rows (chỉ các split được phép)."""
    orc = read_rows("oracle", splits)
    t = keys[["customer_id", "t0"]].merge(orc.drop(columns=["split"]), on=["customer_id", "t0"], how="left", validate="one_to_one")
    assert t.y_a0.notna().all()
    return dict(S=t.basket_value_S.to_numpy(), y0=t.y_a0.to_numpy().astype(float),
                Y=np.column_stack([t[f"y_{a}"] for a in ACTIONS]).astype(float),
                tau=np.column_stack([t[f"tau_cond_{a}"] for a in TREATED]),
                dc_real=np.column_stack([t[f"discount_cost_{a}"] for a in ACTIONS]),
                profit=np.column_stack([t[f"profit_{a}"] for a in ACTIONS]))


def evaluate_best_w(T: dict, ins: dict) -> tuple[np.ndarray, np.ndarray]:
    """Chính sách biết trước trên CÙNG miền khả thi (miền v1 toàn t0): theo hiện thực (A) và theo kỳ vọng (B)."""
    Ur, Ue = S6.evaluate_utilities({k: v[ins["idx"]] for k, v in T.items()}, M0)
    return S6.solve_hs(ins, Ur[:, 1:])["w"], S6.solve_hs(ins, Ue[:, 1:])["w"]


def evaluate_policy_tables(T, W: dict, WA, WB, pops: dict, is_uk, t0_arr, t0s) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    Ur, Ue = S6.evaluate_utilities(T, M0)
    vA = {p: (w * Ur).sum(1) for p, w in W.items()}
    vB = {p: (w * Ue).sum(1) for p, w in W.items()}
    zA, zB = (WA * Ur).sum(1), (WB * Ue).sum(1)
    per = []
    for t in t0s:
        mt = t0_arr == t
        for p in W:
            per.append({"t0": pd.Timestamp(t), "chinh_sach": p, "gia_tri_hien_thuc": vA[p][mt].sum(), "gia_tri_ky_vong": vB[p][mt].sum(),
                        "z_sao_hien_thuc": zA[mt].sum(), "z_sao_ky_vong": zB[mt].sum(), "so_voucher": int(W[p][mt, 1:].sum())})
    g4, g4a, g7 = [], [], []
    for pk, pm in pops.items():
        for p in W:
            a, b, za, zb = vA[p][pm].sum(), vB[p][pm].sum(), zA[pm].sum(), zB[pm].sum()
            g4.append({"quan_the": pk, "chinh_sach": p, "n_dong": int(pm.sum()), "gia_tri_hien_thuc": a, "gia_tri_ky_vong": b,
                       "z_sao_hien_thuc": za, "z_sao_ky_vong": zb,
                       "regret_A": (za - a) / abs(za) if abs(za) > 1e-9 else np.nan,
                       "regret_B": (zb - b) / abs(zb) if abs(zb) > 1e-9 else np.nan})
        for lab, ref in COMPARISONS:
            pol, leg = vA[POL_MILP_R][pm].sum(), vA[ref][pm].sum()
            ok = bool(pol - leg >= GATES["G4_policy_value_lift_vs_legacy_min"] * abs(leg)) if pm.sum() else np.nan
            g4a.append({"quan_the": pk, "so_sanh": lab, "chinh_sach": pol, "doi_chung": leg, "chenh_lech_tuyet_doi": pol - leg,
                        "nguong_010_abs_doi_chung": GATES["G4_policy_value_lift_vs_legacy_min"] * abs(leg), "dat": ok,
                        "dat_cach_doc_legacy_x110": bool(pol >= (1 + GATES["G4_policy_value_lift_vs_legacy_min"]) * leg) if pm.sum() else np.nan,
                        "chinh_sach_ky_vong": vB[POL_MILP_R][pm].sum(), "doi_chung_ky_vong": vB[ref][pm].sum()})
            for seg, sm in (("UK", is_uk), ("non-UK", ~is_uk)):
                mm = pm & sm
                p_, l_ = vA[POL_MILP_R][mm].sum(), vA[ref][mm].sum()
                rel = (p_ - l_) / abs(l_) if abs(l_) > 1e-9 else np.nan
                g7.append({"quan_the": pk, "so_sanh": lab, "phan_khuc": seg, "n_dong": int(mm.sum()), "chinh_sach": p_, "doi_chung": l_,
                           "chenh_tuong_doi": rel, "dat": bool(rel >= -GATES["G7_segment_value_drop_max"]) if np.isfinite(rel) else np.nan})
    return pd.DataFrame(per), pd.DataFrame(g4), pd.DataFrame(g4a), pd.DataFrame(g7)


# ============================================================================ chạy gate
def _onehot(actions) -> np.ndarray:
    return np.eye(4)[pd.Categorical(actions, categories=ACTIONS).codes]


def run_gates(mode: str, out_root: Path = ROOT, n_lat: int = N_LAT, n_boot_g2: int | None = None, manifest: bool = True) -> dict:
    assert mode in ("dry-run", "locked")
    dry = mode == "dry-run"
    split, tag = ("calib", "dryrun") if dry else ("locked", "locked")
    nhan = "THỬ (calib đóng vai locked)" if dry else "CHÍNH THỨC (locked, mở 1 lần)"
    out_root = Path(out_root)
    tdir, fdir, ddir = out_root / TABLES, out_root / FIGS, out_root / "data/processed"
    for d in (tdir, fdir, ddir):
        d.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    t_start = time.time()
    files: list[Path] = []

    # ---------------------------------------------------------------- dữ liệu (chỉ qua read_rows)
    lg = read_rows("logged", [split])
    snap = read_rows("snapshots", [split])
    mod = read_rows("logged", MODEL_SPLITS, columns=["customer_id", "is_entity_holdout"])
    p2tr = read_rows("pred_s2", ["train"], columns=["s_lo", "s_hi"])
    s_wide = float(np.percentile(p2tr.s_hi - p2tr.s_lo, 95))
    assert abs(s_wide - S_WIDE_EXPECTED) < 0.01, "ngưỡng Ŝ khác GĐ7"
    meta = lg[["split", "t0", S5.SEG_COL, "cancel_rate_365"]].copy()
    legacy = legacy_scores(lg)
    T = evaluate_truth_rows(lg, [split])
    scorer = S7.Scorer()
    t0s = np.sort(lg.t0.unique())
    n = len(lg)
    eh = lg.is_entity_holdout.to_numpy() == 1
    pops = {"toan_bo": np.ones(n, bool), "khach_thuong": ~eh, "entity_holdout": eh}
    is_uk = lg[S5.SEG_COL].to_numpy() == 1
    print(f"[{nhan}] split={split} · {n} dòng · {len(t0s)} kỳ · entity-holdout {int(eh.sum())} dòng · {scorer.version}")

    # ---------------------------------------------------------------- tuần tự từng t0: chấm điểm → MILP → Router v2.1
    rm = S7.ActionRouterV2(s_wide, scorer.version, S7.POLICY_VERSION_V21)
    rl = S7.ActionRouterV2(s_wide, scorer.version, S7.POLICY_VERSION_V21 + "-legacy")
    W = {p: np.zeros((n, 4)) for p in POLICIES}
    WA, WB = np.zeros((n, 4)), np.zeros((n, 4))
    mu0_raw, mu0_cal, tau = np.full(n, np.nan), np.full(n, np.nan), np.full((n, 3), np.nan)
    rc_all = pd.DataFrame(index=range(n), columns=["reason_1_code", "reason_2_code", "reason_3_code"], dtype=object)
    snaps, aud, input_ok = {}, [], []
    for t0 in t0s:
        ins0 = S6.make_ins(lg, meta, split, t0)
        idx = ins0["idx"]
        base = lg.iloc[idx].reset_index(drop=True)
        rc = S3.reason_codes(scorer.mu0, base[S3.FEATS])
        for k in (1, 2, 3):
            base[f"reason_{k}_code"] = rc[f"reason_{k}_code"].to_numpy()
            rc_all.loc[idx, f"reason_{k}_code"] = rc[f"reason_{k}_code"].to_numpy()
        X, Xc = scorer.features(base)
        sc = scorer.score(X, Xc)
        try:
            frame = rm.validate_input(S7.scored_frame_v2(base, sc))
            input_ok.append(True)
        except pa.errors.SchemaErrors:
            input_ok.append(False)
            frame = S7.scored_frame_v2(base, sc)
            frame["customer_id"] = frame.customer_id.astype(int)
        mu0_raw[idx], mu0_cal[idx], tau[idx] = sc["mu0_raw"], sc["mu0_cal"], sc["tau"]
        V = sc["V"]
        W[POL_MILP_PURE][idx] = S6.solve_hs(ins0, V)["w"]
        W[POL_LEG_PURE][idx] = S6.solve_hs(ins0, legacy[idx])["w"]
        for name, router, score, pol in (("MILP", rm, V, POL_MILP_R), ("Luật cũ", rl, legacy[idx], POL_LEG_R)):
            snaps[(name, t0)] = dict(router.history)
            fl = router.domain_flags(frame)
            ins = S7.make_ins_v2(lg, meta, split, t0, fl)
            w_prop = S6.solve_hs(ins, score)["w"]
            assert not (w_prop[:, 1:].sum(1) > 0)[fl["opt"] | fl["incap"] | fl["canc"]].any()
            dec = router.route(frame, w_prop, split, ins=ins)
            W[pol][idx] = _onehot(dec.final_action)
            a = S7.audit_final(ins, dec, fl)
            a["n_vi_pham"] = int(sum(bool(v) for k, v in a.items() if k.startswith("vp_")))
            aud.append({"chinh_sach": pol, "t0": pd.Timestamp(t0), **a})
        WA[idx], WB[idx] = evaluate_best_w(T, ins0)
        print(f"  {pd.Timestamp(t0).date()}: MILP qua Router gửi {int(W[POL_MILP_R][idx, 1:].sum())} · luật cũ qua Router gửi "
              f"{int(W[POL_LEG_R][idx, 1:].sum())} ({time.time() - t_start:.0f}s)")
    for name, router, fn in (("MILP", rm, f"decisions_s8_{tag}.parquet"), ("Luật cũ", rl, f"decisions_s8_{tag}_legacy.parquet")):
        dl = S7.DECISION_SCHEMA_V21.validate(pd.concat(router.frames, ignore_index=True), lazy=True)
        dl.to_parquet(ddir / fn, index=False)
        files.append(ddir / fn)

    # ---------------------------------------------------------------- G1 … G7
    g1 = g1_contract(lg, snap, mod, input_ok)
    g2 = g2_calibration(lg, mu0_raw, mu0_cal, pops, n_boot_g2 or N_BOOT_G2)
    g3 = g3_qini(lg, tau, pops)
    per, g4, g4a, g7 = evaluate_policy_tables(T, W, WA, WB, pops, is_uk, lg.t0.to_numpy(), t0s)
    au = pd.DataFrame(aud)
    lt = latency(lg, meta, split, t0s, scorer, snaps, s_wide, n_lat)
    pct = pd.DataFrame([{"khau": s, "P50": lt[s].quantile(.5), "P95": lt[s].quantile(.95), "P99": lt[s].quantile(.99), "max": lt[s].max()}
                        for s in ["t_dac_trung", "t_ma_ly_do", "t_cham_diem", "t_hop_dong_va_mien", "t_milp", "t_router", "tong"]])
    gates, summ = gate_table(g1, g2, g3, g4, g4a, au, lt, g7, pops, nhan)

    # ---------------------------------------------------------------- ghi bảng
    out = {"gates": gates, "gate_summary": summ, "g1_contract": g1, "g2_calibration": g2, "g3_qini": g3, "g4_policy_value": g4,
           "g4a_comparison": g4a, "g5_constraint_audit": au, "g6_latency_runs": lt, "g6_latency_percentiles": pct, "g7_segments": g7,
           "policy_value_by_period": per}
    if dry:
        out["consistency"] = dryrun_consistency(per, g3, au, rc_all, lg, pct)
    for k, v in out.items():
        f = tdir / f"s8_{tag}_{k}.csv"
        v.to_csv(f, index=False)
        files.append(f)
    files.append(plot_gates(fdir / f"fig_s8_{tag}_gates.png", lg, mu0_raw, mu0_cal, g3, g4, lt, nhan))
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    print(summ.to_string(index=False))
    print(gates[gates.quan_the == "toan_bo"][["gate", "chi_so", "gia_tri", "nguong", "dat"]].to_string(index=False))
    if dry:
        print(out["consistency"].to_string(index=False))
    if manifest:
        rel = [f.relative_to(ROOT) if f.is_relative_to(ROOT) else f for f in files]
        common.write_manifest(f"s8_{tag}", rel, extra={"mode": mode, "split_dong_vai_locked": split, "nhan": nhan,
                                                       "model_version": scorer.version, "policy_version": S7.POLICY_VERSION_V21,
                                                       "n_lat_moi_ky": n_lat, "runtime_s": round(time.time() - t_start)},
                              packages=KEY_PKGS)
    print(f"XONG GĐ8 bước 1a – {nhan} ({time.time() - t_start:.0f}s)")
    return out


def latency(lg, meta, split, t0s, scorer, snaps, s_wide, n_lat: int) -> pd.DataFrame:
    """G6: ≥ n_lat lượt đầu–cuối mỗi kỳ (lượt 0 khởi động, không tính), router chạy khô từ ảnh chụp lịch sử trước kỳ."""
    rows = []
    for t0 in t0s:
        base0 = lg.iloc[S6.make_ins(lg, meta, split, t0)["idx"]].reset_index(drop=True)
        for rep in range(n_lat + 1):
            r = S7.ActionRouterV2(s_wide, scorer.version, S7.POLICY_VERSION_V21)
            r.history = dict(snaps[("MILP", t0)])
            base = base0.copy()
            tt = time.perf_counter(); X, Xc = scorer.features(base); t_f = time.perf_counter() - tt
            tt = time.perf_counter()
            rc = S3.reason_codes(scorer.mu0, base[S3.FEATS])
            for k in (1, 2, 3):
                base[f"reason_{k}_code"] = rc[f"reason_{k}_code"].to_numpy()
            t_r0 = time.perf_counter() - tt
            tt = time.perf_counter(); sc = scorer.score(X, Xc); fr = S7.scored_frame_v2(base, sc); t_s = time.perf_counter() - tt
            tt = time.perf_counter(); fr = r.validate_input(fr); fl = r.domain_flags(fr); ins = S7.make_ins_v2(lg, meta, split, t0, fl)
            t_c = time.perf_counter() - tt
            tt = time.perf_counter(); w = S6.solve_hs(ins, sc["V"])["w"]; t_m = time.perf_counter() - tt
            tt = time.perf_counter(); r.route(fr, w, split, ins=ins, dry_run=True); t_ro = time.perf_counter() - tt
            if rep:
                rows.append({"t0": pd.Timestamp(t0), "lan": rep, "n_khach": len(base), "t_dac_trung": t_f, "t_ma_ly_do": t_r0,
                             "t_cham_diem": t_s, "t_hop_dong_va_mien": t_c, "t_milp": t_m, "t_router": t_ro,
                             "tong": t_f + t_r0 + t_s + t_c + t_m + t_ro})
    return pd.DataFrame(rows)


# ============================================================================ bảng gate
def _row(gate, chi_so, pk, val, thr, ok, pred, note, nhan):
    return {"gate": gate, "chi_so": chi_so, "quan_the": pk, "gia_tri": val, "nguong": thr,
            "dat": bool(ok) if isinstance(ok, (bool, np.bool_)) else None, "nhan": nhan, "du_bao_dang_ky_truoc": pred, "ghi_chu": note}


def gate_table(g1, g2, g3, g4, g4a, au, lt, g7, pops, nhan) -> tuple[pd.DataFrame, pd.DataFrame]:
    R = []
    R.append(_row("G1", "mọi kiểm tra hợp đồng dữ liệu đạt", "toan_bo", f"{int(g1.dat.sum())}/{len(g1)}", "tất cả",
                  bool(g1.dat.all()) == GATES["G1_data_contract_pass"], PRED["G1"],
                  "; ".join(g1.loc[~g1.dat, "kiem_tra"]) or "không có kiểm tra trượt", nhan))
    for pk in pops:
        r = g2[g2.quan_the == pk].iloc[0]
        if "ece_hieu_chinh" not in r or pd.isna(r.get("ece_hieu_chinh")):
            R.append(_row("G2", "ECE & Brier (dòng a0)", pk, np.nan, "", None, PRED["G2"], f"không đủ dòng a0 (n={int(r.n_a0)})", nhan))
            continue
        ok = bool(r.ece_hieu_chinh <= GATES["G2_ece_max"]) and (bool(r.brier_hieu_chinh <= r.brier_tho)
                                                                 if GATES["G2_brier_not_worse_than_raw"] else True)
        R.append(_row("G2", "ECE hiệu chỉnh ≤ ngưỡng và Brier hiệu chỉnh ≤ Brier thô", pk,
                      f"ECE {r.ece_hieu_chinh:.4f} [CI {r.ece_hieu_chinh_ci_lo:.4f}–{r.ece_hieu_chinh_ci_hi:.4f}] · Brier {r.brier_hieu_chinh:.4f} vs thô {r.brier_tho:.4f}",
                      f"ECE ≤ {GATES['G2_ece_max']}; Brier không tệ hơn", ok, PRED["G2"],
                      f"n_a0 = {int(r.n_a0)}; tỷ lệ mua {r.ty_le_mua_a0:.3f} vs p̂ TB {r.p_cal_TB:.3f}; ECE thô {r.ece_tho:.4f}", nhan))
    for pk in pops:
        q = g3[g3.quan_the == pk]
        if q.n.max() == 0:
            R.append(_row("G3", "số hành động Qini CI95 > 0", pk, np.nan, "", None, PRED["G3"], "không có dòng nhánh ngẫu nhiên", nhan))
            continue
        k = int(q.ci_duong.sum())
        txt = " · ".join(f"{x.hanh_dong} {x.qini:.3f} [{x.qini_ci_lo:.3f}, {x.qini_ci_hi:.3f}]" for x in q.itertuples())
        R.append(_row("G3", "số hành động Qini > 0 với CI 95 % > 0", pk, f"{k}/3 ({txt})",
                      f"≥ {GATES['G3_qini_positive_actions_min']}", k >= GATES["G3_qini_positive_actions_min"], PRED["G3"],
                      f"n cặp (a0 ∪ a) = {', '.join(str(int(v)) for v in q.n)}; bootstrap cụm khách {S4.N_BOOT} lần", nhan))
    for pk in pops:
        for x in g4a[g4a.quan_the == pk].itertuples():
            chinh = x.so_sanh.startswith("CHÍNH")
            R.append(_row("G4a" if chinh else "G4a-phụ", f"(MILP qua Router − đối chứng) ≥ 0.10 × |đối chứng| [{x.so_sanh}]", pk,
                          f"£{x.chinh_sach:,.0f} vs £{x.doi_chung:,.0f} (chênh £{x.chenh_lech_tuyet_doi:,.0f})",
                          f"chênh ≥ £{x.nguong_010_abs_doi_chung:,.0f}",
                          x.dat if isinstance(x.dat, (bool, np.bool_)) else None, PRED["G4a"],
                          f"cách đọc legacy × 1.10: {'đạt' if x.dat_cach_doc_legacy_x110 else 'trượt'}; kỳ vọng (B): £{x.chinh_sach_ky_vong:,.0f} vs £{x.doi_chung_ky_vong:,.0f}",
                          nhan))
        for p, gname in ((POL_MILP_R, "G4b"), (POL_MILP_PURE, "G4b-phụ")):
            x = g4[(g4.quan_the == pk) & (g4.chinh_sach == p)].iloc[0]
            ok = bool(x.regret_A <= GATES["G4_normalized_regret_max"]) if np.isfinite(x.regret_A) else None
            R.append(_row(gname, f"regret chuẩn hóa A (hiện thực) – {p}", pk, f"{x.regret_A:.3f}" if np.isfinite(x.regret_A) else np.nan,
                          f"≤ {GATES['G4_normalized_regret_max']}", ok, PRED["G4b"],
                          f"giá trị £{x.gia_tri_hien_thuc:,.0f} vs tốt nhất biết trước £{x.z_sao_hien_thuc:,.0f}; regret B (kỳ vọng, báo kèm) "
                          f"{x.regret_B:.3f}", nhan))
    m = au[au.chinh_sach == POL_MILP_R]
    vcols = [c for c in m.columns if c.startswith("vp_")]
    nvp = int(m.n_vi_pham.sum())
    R.append(_row("G5", "số vi phạm ràng buộc trên quyết định GỬI THỰC TẾ (MILP qua Router v2.1)", "toan_bo", nvp,
                  f"≤ {GATES['G5_constraint_violations_max']}", nvp <= GATES["G5_constraint_violations_max"], PRED["G5"],
                  "vi phạm theo loại: " + ", ".join(f"{c}={int(m[c].sum())}" for c in vcols) + f"; {len(m)} kỳ", nhan))
    p95 = float(lt.tong.quantile(.95))
    R.append(_row("G6", "P95 độ trễ đầu–cuối một t0 (giây)", "toan_bo", round(p95, 3), f"≤ {GATES['G6_latency_p95_seconds_max']}",
                  p95 <= GATES["G6_latency_p95_seconds_max"], PRED["G6"],
                  f"{len(lt)} lượt ({lt.groupby('t0').size().min()} lượt/kỳ); P95 lớn nhất theo kỳ {lt.groupby('t0').tong.quantile(.95).max():.3f} s; "
                  "đã gồm khâu mã lý do (GĐ7 không tính)", nhan))
    for pk in pops:
        for lab, _ in COMPARISONS:
            s = g7[(g7.quan_the == pk) & (g7.so_sanh == lab)]
            vals = s.dat.tolist()
            ok = all(bool(v) for v in vals) if all(isinstance(v, (bool, np.bool_)) for v in vals) else None
            R.append(_row("G7" if lab.startswith("CHÍNH") else "G7-phụ", f"chênh tương đối mỗi phân khúc ≥ −{GATES['G7_segment_value_drop_max']} [{lab}]",
                          pk, " · ".join(f"{x.phan_khuc} {x.chenh_tuong_doi:+.1%}" if np.isfinite(x.chenh_tuong_doi) else f"{x.phan_khuc} n/a"
                                         for x in s.itertuples()),
                          f"≥ −{GATES['G7_segment_value_drop_max']:.0%}", ok, PRED["G7"],
                          " · ".join(f"{x.phan_khuc}: £{x.chinh_sach:,.0f} vs £{x.doi_chung:,.0f}" for x in s.itertuples()), nhan))
    gates = pd.DataFrame(R)
    gates["cham_chinh_thuc"] = (gates.quan_the == "toan_bo") & ~gates.gate.str.endswith("phụ")
    # tổng hợp: G4 = G4a ∧ G4b
    summ = []
    for pk in pops:
        g = gates[(gates.quan_the == pk) & ~gates.gate.str.endswith("phụ")].set_index("gate").dat
        rec = {"quan_the": POPS[pk], "nhan": nhan}
        for k in ["G1", "G2", "G3", "G4a", "G4b", "G5", "G6", "G7"]:
            v = g.get(k, None)
            rec[k] = "ĐẠT" if v is True else ("TRƯỢT" if v is False else "n/a")
        rec["G4"] = "ĐẠT" if rec["G4a"] == rec["G4b"] == "ĐẠT" else ("n/a" if "n/a" in (rec["G4a"], rec["G4b"]) else "TRƯỢT")
        summ.append(rec)
    return gates, pd.DataFrame(summ)


# ============================================================================ đối chiếu diễn tập với GĐ4 / GĐ7 (chỉ dry-run)
def dryrun_consistency(per, g3, au, rc_all, lg, pct) -> pd.DataFrame:
    rows = []
    s7 = pd.read_csv(ROOT / TABLES / "s7_policy_value_by_period_v21.csv", parse_dates=["t0"])
    s7 = s7[s7.split == "calib"]
    mg = per.merge(s7[["t0", "chinh_sach", "gia_tri_hien_thuc", "gia_tri_ky_vong"]], on=["t0", "chinh_sach"], suffixes=("", "_s7"))
    rows.append({"doi_chieu": "Giá trị hiện thực theo kỳ × 4 chính sách vs s7_policy_value_by_period_v21 (calib)", "so_cap": len(mg),
                 "lech_lon_nhat": float((mg.gia_tri_hien_thuc - mg.gia_tri_hien_thuc_s7).abs().max()) if len(mg) else np.nan})
    rows.append({"doi_chieu": "Giá trị kỳ vọng theo kỳ × 4 chính sách vs s7 v2.1 (calib)", "so_cap": len(mg),
                 "lech_lon_nhat": float((mg.gia_tri_ky_vong - mg.gia_tri_ky_vong_s7).abs().max()) if len(mg) else np.nan})
    q4 = pd.read_csv(ROOT / TABLES / "s4_qini.csv")
    q4 = q4[(q4.tap == "calib") & (q4.learner == "CausalForestDML")].set_index("hanh_dong").qini
    q8 = g3[g3.quan_the == "toan_bo"].set_index("hanh_dong").qini
    rows.append({"doi_chieu": "Qini calib (CausalForestDML) vs s4_qini.csv", "so_cap": len(q4),
                 "lech_lon_nhat": float((q8.reindex(q4.index) - q4).abs().max())})
    a7 = pd.read_csv(ROOT / TABLES / "s7_constraint_audit_v21.csv", parse_dates=["t0"])
    a7 = a7[(a7.split == "calib") & a7.chinh_sach.isin([POL_MILP_R, POL_LEG_R])]
    ma = au.merge(a7, on=["chinh_sach", "t0"], suffixes=("", "_s7"))
    rows.append({"doi_chieu": "Kiểm toán ràng buộc (số voucher, a2, cuộc gọi, ngân sách) vs s7_constraint_audit_v21 (calib)",
                 "so_cap": len(ma), "lech_lon_nhat": float(max((ma[c] - ma[f"{c}_s7"]).abs().max()
                                                               for c in ("so_voucher", "so_a2", "so_cuoc_goi", "ngan_sach_dung")))})
    r3 = read_rows("reasons_s3", ["calib"], columns=["customer_id", "t0", "reason_1_code", "reason_2_code", "reason_3_code"])
    k = lg[["customer_id", "t0"]].join(rc_all).merge(r3, on=["customer_id", "t0"], suffixes=("", "_s3"))
    eq = np.mean([(k[f"reason_{i}_code"] == k[f"reason_{i}_code_s3"]).mean() for i in (1, 2, 3)])
    rows.append({"doi_chieu": "Mã lý do top-3 tính lại vs reasons_s3 (calib) – tỷ lệ KHÁC", "so_cap": len(k), "lech_lon_nhat": float(1 - eq)})
    l7 = pd.read_csv(ROOT / TABLES / "s7_latency_percentiles_v21.csv").set_index("khau")
    rows.append({"doi_chieu": "P95 độ trễ (s8, gồm mã lý do) − P95 GĐ7 v2.1 (không gồm mã lý do), giây", "so_cap": 1,
                 "lech_lon_nhat": float(pct.set_index("khau").loc["tong", "P95"] - l7.loc["tong", "P95"])})
    return pd.DataFrame(rows)


# ============================================================================ hình
def plot_gates(path: Path, lg, mu0_raw, mu0_cal, g3, g4, lt, nhan) -> Path:
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    a0 = lg.A_logged.to_numpy() == "a0"
    y = lg.y_obs.to_numpy()[a0].astype(float)
    for p, lab, c in ((mu0_raw[a0], "μ̂0 thô", "tab:gray"), (mu0_cal[a0], "μ̂0 hiệu chỉnh (Current)", "tab:blue")):
        o = np.argsort(p)
        b = np.array_split(o, 15)
        ax[0, 0].plot([p[i].mean() for i in b], [y[i].mean() for i in b], "o-", color=c,
                      label=f"{lab} – ECE {S2.ece_equal_mass(y, p):.3f}")
    lim = max(0.6, float(np.nanmax(mu0_cal[a0])))
    ax[0, 0].plot([0, lim], [0, lim], "k--", lw=0.8)
    ax[0, 0].set_xlabel("Xác suất dự báo (TB bin)"); ax[0, 0].set_ylabel("Tỷ lệ mua lại thực tế (dòng a0)")
    ax[0, 0].set_title(f"G2 – Biểu đồ tin cậy (15 bin cùng khối lượng), ngưỡng ECE ≤ {GATES['G2_ece_max']}"); ax[0, 0].legend(fontsize=8)
    q = g3[(g3.quan_the == "toan_bo")]
    xs = np.arange(len(q))
    ax[0, 1].bar(xs, q.qini, color=["tab:green" if v else "tab:orange" for v in q.ci_duong])
    ax[0, 1].errorbar(xs, q.qini, yerr=[q.qini - q.qini_ci_lo, q.qini_ci_hi - q.qini], fmt="none", ecolor="k", capsize=4)
    ax[0, 1].axhline(0, color="k", lw=0.8); ax[0, 1].set_xticks(xs); ax[0, 1].set_xticklabels(q.hanh_dong)
    ax[0, 1].set_ylabel("Hệ số Qini (CI 95 % bootstrap cụm)"); ax[0, 1].set_title("G3 – Qini trên nhánh ngẫu nhiên (xanh = CI > 0)")
    g = g4[g4.quan_the == "toan_bo"]
    ax[1, 0].barh(g.chinh_sach, g.gia_tri_hien_thuc, color="tab:blue")
    ax[1, 0].axvline(g.z_sao_hien_thuc.iloc[0], color="tab:red", ls="--", label="tốt nhất biết trước (hiện thực)")
    for i, x in enumerate(g.itertuples()):
        ax[1, 0].text(x.gia_tri_hien_thuc, i, f" £{x.gia_tri_hien_thuc:,.0f} (regret A {x.regret_A:.2f})", va="center", fontsize=8)
    ax[1, 0].axvline(0, color="k", lw=0.8); ax[1, 0].set_xlabel("Lợi nhuận gia tăng HIỆN THỰC, cộng dồn các kỳ (GBP)")
    ax[1, 0].set_title("G4 – Giá trị chính sách toàn t0"); ax[1, 0].legend(fontsize=8); ax[1, 0].tick_params(axis="y", labelsize=8)
    st = ["t_dac_trung", "t_ma_ly_do", "t_cham_diem", "t_hop_dong_va_mien", "t_milp", "t_router"]
    ax[1, 1].boxplot([lt[s] for s in st] + [lt.tong], tick_labels=["đặc trưng", "mã lý do", "chấm điểm", "hợp đồng", "MILP", "router", "tổng"],
                     showfliers=False)
    ax[1, 1].set_ylabel("Giây / một t0"); ax[1, 1].tick_params(axis="x", labelsize=8)
    ax[1, 1].set_title(f"G6 – Độ trễ ({len(lt)} lượt), P95 tổng = {lt.tong.quantile(.95):.2f} s (ngưỡng {GATES['G6_latency_p95_seconds_max']} s)")
    fig.suptitle(f"Cổng phát hành GĐ8 – {nhan}")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return path


# ============================================================================ CLI
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="GĐ8 bước 1a – chuẩn bị phần gate")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="calib đóng vai locked, bảng gate nhãn THỬ")
    g.add_argument("--freeze-draft", action="store_true", help="manifest đóng băng BẢN NHÁP")
    g.add_argument("--freeze", action="store_true", help="manifest đóng băng CHÍNH THỨC (chỉ sau bước 1b)")
    g.add_argument("--open-locked", action="store_true", help="mở locked đúng MỘT lần")
    a = ap.parse_args(argv)
    if a.dry_run:
        run_gates("dry-run")
    elif a.freeze_draft:
        write_freeze("s8_freeze_draft")
    elif a.freeze:
        write_freeze("s8_freeze")
    else:
        try:
            open_locked()
        except (LockedAlreadyOpened, FreezeMismatch) as e:
            print(e)
            sys.exit(2)


if __name__ == "__main__":
    main()
