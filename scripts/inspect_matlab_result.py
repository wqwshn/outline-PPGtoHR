"""Dump structure of MATLAB Best_Params_Result_*.mat into JSON-friendly form."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def to_py(obj):
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [to_py(x) for x in obj.tolist()]
        return obj.tolist()
    if hasattr(obj, "_fieldnames"):
        return {f: to_py(getattr(obj, f)) for f in obj._fieldnames}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return obj


def main(path: str) -> None:
    d = loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in d if not k.startswith("__")]
    for k in keys:
        v = d[k]
        if hasattr(v, "_fieldnames"):
            print(f"\n=== {k} (struct) ===")
            print(json.dumps(to_py(v), ensure_ascii=False, indent=2, default=str))
        elif isinstance(v, np.ndarray):
            print(f"\n=== {k} (array shape={v.shape} dtype={v.dtype}) ===")
            if v.size <= 20:
                print(v.tolist())
            else:
                print("min=", float(np.nanmin(v)), "max=", float(np.nanmax(v)),
                      "mean=", float(np.nanmean(v)))
        else:
            print(f"\n=== {k} === {v!r}")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else r"20260418test_python\Best_Params_Result_multi_tiaosheng1_processed.mat"
    main(str(Path(p)))
