"""Kiểm tra SHA-256: đọc file thật trên server → băm → so với manifest đóng băng → MATCH / MISMATCH."""
from __future__ import annotations

import hashlib
import time

import pandas as pd

from services import data_service as D


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(progress=None) -> tuple[pd.DataFrame, float]:
    man = D.manifest("s8_freeze_manifest.json")
    entries = [("Mã / mô hình / cấu hình", e) for e in man["files"]] + [("Dữ liệu đã xử lý", e) for e in man["data_files"]]
    rows, t = [], time.perf_counter()
    for k, (grp, e) in enumerate(entries):
        fp = D.ROOT / e["file"]
        if fp.exists():
            got = sha256_file(fp)
            status = "MATCH" if got == e["sha256"] else "MISMATCH"
        else:
            got, status = "", "MISSING"
        rows.append({"Nhóm": grp, "File": e["file"], "Bytes": e["bytes"], "SHA-256 tham chiếu": e["sha256"],
                     "SHA-256 tính lại": got, "Kết quả": status})
        if progress:
            progress((k + 1) / len(entries), e["file"])
    return pd.DataFrame(rows), time.perf_counter() - t
