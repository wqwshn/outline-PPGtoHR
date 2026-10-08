"""Standalone process adapter: never import the current ppg_hr in this process.

Return-event observation is necessary for the immutable paper release, which predates
an array diagnostics API. It does not replace calls, arguments, results or state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SCHEMA = 1


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    for field in ("source", "trace", "trace-sha", "data", "ref", "cache"):
        parser.add_argument("--" + field, required=True)
    args = parser.parse_args()
    source, cache = Path(args.source), Path(args.cache)
    archive = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    identity = archive["identity"]
    if digest(args.trace) != args.trace_sha:
        raise ValueError("Frozen trace hash mismatch")
    hashes = {p.relative_to(source).as_posix(): digest(p)
              for p in sorted((source / "python/src").rglob("*.py"))}
    source_sha = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if source_sha != identity["source_sha256"]:
        raise ValueError("Frozen source identity mismatch; replay refused")
    for field in ("data", "ref"):
        if digest(getattr(args, field)) != identity[field + "_sha256"]:
            raise ValueError(f"Frozen {field} hash mismatch")
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{SCHEMA}:{args.trace_sha}:{source_sha}".encode()).hexdigest()
    dest = cache / (key + ".json")
    vectors = dest.with_suffix(".npz")
    if dest.is_file() and vectors.is_file():
        old = json.loads(dest.read_text(encoding="utf-8"))
        if old.get("arrays_sha256") == digest(vectors) and old.get("schema_version") == SCHEMA:
            print(dest, flush=True)
            return
    sys.path.insert(0, str(source / "python/src"))
    from ppg_hr.v2.solver import solve_v2
    from ppg_hr.v2.types import V2RunConfig

    config = dict(identity["config"])
    config["data_path"], config["ref_path"] = Path(args.data), Path(args.ref)
    config["reference_groups_order"] = tuple(config["reference_groups_order"])
    arrays, metadata = {}, []
    stage_counts = {}

    def capture(frame, event, result):
        if event != "return":
            return
        name = frame.f_code.co_name
        if name not in {"_process_spectrum_with_trace_impl", "_run_v1_style_reference_cascade",
                        "apply_adaptive_cascade"}:
            return
        parent = frame.f_back
        while parent is not None and parent.f_code.co_name != "_unified_solve":
            parent = parent.f_back
        if parent is None or "center" not in parent.f_locals:
            return
        loc = frame.f_locals
        center = float(parent.f_locals["center"])
        prefix = f"c{center:.6f}"
        if name == "apply_adaptive_cascade":
            cascade = frame.f_back
            if cascade.f_code.co_name != "_run_v1_style_reference_cascade":
                return
            stage_counts[prefix] = stage_counts.get(prefix, 0) + 1
            key = f"{prefix}_stage{stage_counts[prefix]}"
            for field in ("u", "d"):
                arrays[key + "_" + field] = np.asarray(loc[field]).copy()
            arrays[key + "_output"] = np.asarray(result).copy()
            metadata.append(dict(key=key, center_s=center, kind="stage",
                                 fs=int(cascade.f_locals["fs"])))
        elif name == "_run_v1_style_reference_cascade":
            key = prefix + "_cascade"
            for field, values in (("input", loc["sig_p"]), ("output", result[0]),
                                  ("reference", result[1])):
                arrays[key + "_" + field] = np.asarray(values).copy()
            metadata.append(dict(key=key, center_s=center, kind="cascade",
                                 fs=int(loc["fs"]), stages=result[2]))
        else:
            key = prefix + "_" + loc["path"]
            for field in ("freqs", "raw_amps", "scored_amps", "sig_in", "sig_penalty_ref",
                          "ref_freqs", "ref_amps"):
                if field in loc:
                    arrays[key + "_" + field] = np.asarray(loc[field]).copy()
            metadata.append(dict(key=key, center_s=center, kind="spectrum",
                                 fs=int(loc["fs"]), path=loc["path"], trace=result[1].to_dict()))

    sys.setprofile(capture)
    try:
        result = solve_v2(V2RunConfig(**config))
    finally:
        sys.setprofile(None)
    expected = np.asarray(archive["hr"], float)
    np.testing.assert_allclose(result.HR, expected, atol=1e-9, rtol=0, equal_nan=True)
    differences = np.abs(result.HR - expected)
    finite = differences[np.isfinite(differences)]
    # Write data first, then publish the manifest as the completed-cache marker.
    temporary_vectors = vectors.with_suffix(".tmp.npz")
    np.savez_compressed(temporary_vectors, **arrays)
    temporary_vectors.replace(vectors)
    payload = dict(schema_version=SCHEMA, source_sha256=source_sha,
                   trace_sha256=args.trace_sha, arrays_sha256=digest(vectors),
                   config=identity["config"], max_abs_hr_difference=float(finite.max()) if finite.size else 0,
                   exact_frozen_replay=True, hr=result.HR.tolist(), metadata=metadata,
                   window_table=result.window_table)
    temporary = dest.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, default=json_default), encoding="utf-8")
    temporary.replace(dest)
    print(dest, flush=True)


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
