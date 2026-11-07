import json
import os
import tempfile
from pathlib import Path

import numpy as np


def atomic_write(path, writer, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if binary else "w"
    options = {} if binary else {"encoding": "utf-8"}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode=mode, dir=path.parent, delete=False, **options) as handle:
            temporary = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, values):
    atomic_write(path, lambda handle: handle.write(json.dumps(values, indent=2, allow_nan=False) + "\n"))


def write_npz(path, **values):
    atomic_write(path, lambda handle: np.savez_compressed(handle, **values), binary=True)


def recover_records(path):
    """Recover committed JSONL rows after an interrupted final write."""
    path = Path(path)
    if not path.exists():
        return []
    rows, offset = [], 0
    with path.open("r+b") as handle:
        for raw in handle:
            try:
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError("a result record must be a JSON object")
            except (ValueError, UnicodeDecodeError):
                # A complete malformed row is an error; an unfinished final row can be retried.
                if raw.endswith(b"\n"):
                    raise ValueError("invalid committed result record") from None
                handle.truncate(offset)
                break
            rows.append(row)
            offset += len(raw)
            if not raw.endswith(b"\n"):
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
                break
        handle.flush()
        os.fsync(handle.fileno())
    return rows


def append_record(path, row):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
