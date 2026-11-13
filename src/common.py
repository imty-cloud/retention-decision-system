"""Tiện ích dùng chung cho mọi giai đoạn: đọc config, đường dẫn, băm SHA-256, ghi manifest."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "configs/config.yaml") -> dict:
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def p(rel: str | Path) -> Path:
    """Đường dẫn tuyệt đối tính từ thư mục gốc dự án; tự tạo thư mục cha."""
    out = ROOT / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def package_versions(names: list[str]) -> dict[str, str]:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = "not-installed"
    return out


def write_manifest(stage: str, files: list[str | Path], extra: dict | None = None,
                   packages: list[str] | None = None) -> Path:
    """Ghi manifest JSON: SHA-256 từng file + phiên bản thư viện + tham số. Phục vụ phụ lục tái lập."""
    entries = []
    for f in files:
        fp = ROOT / f if not Path(f).is_absolute() else Path(f)
        entries.append({"file": str(fp.relative_to(ROOT)), "bytes": fp.stat().st_size,
                        "sha256": sha256_file(fp)})
    manifest = {
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": entries,
        "environment": package_versions(packages or ["pandas", "numpy", "pyarrow", "pandera",
                                                     "scikit-learn", "matplotlib", "pyyaml"]),
        "extra": extra or {},
    }
    out = p(f"outputs/manifests/{stage}_manifest.json")
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out
