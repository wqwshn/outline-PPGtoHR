"""Read-only paper cohort browsing and isolated, verified frozen replay."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import numpy as np
import pandas as pd

SCENES = {
    "bobi": ("BUR", "Burpees"), "jianpan": ("TYP", "Typing"),
    "kaihe": ("JJ", "Jumping Jacks"), "quanji": ("PCH", "Punching"),
    "run": ("RUN", "Running"), "tiaosheng": ("RS", "Rope Skipping"),
    "woli": ("HG", "Handgrip"), "xiezi": ("HW", "Handwriting"),
}
VERSION = "paper_release_20260906 / paper_fft_acc_audit_20260907"
RAW_COLUMNS = ["PPG_Green", "Ut1(mV)", "Ut2(mV)", "AccX(g)", "AccY(g)", "AccZ(g)"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_data_root() -> Path:
    """Prefer explicit/local resources; support Git worktrees without copying data."""
    if os.environ.get("PPG_HR_DATA_ROOT"):
        return Path(os.environ["PPG_HR_DATA_ROOT"])
    local = Path.cwd() / "data"
    if (local / "experiments/paper_release_20260906").is_dir():
        return local
    return Path("D:/data/PPG_HeartRate/Algorithm/Algorithm/outline-PPGtoHR/data")


@dataclass(frozen=True)
class PaperRecord:
    record_id: str
    label: str
    subject: str
    scene: str
    ordinal: int
    data_path: Path
    ref_path: Path
    routes: dict


class PaperReplayIndex:
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.audit = self.data_root / "experiments/paper_fft_acc_audit_20260907"
        self.release = self.data_root / "experiments/paper_release_20260906"
        metrics = pd.read_csv(self.audit / "record_route_metrics.csv")
        metrics = metrics[metrics.cohort == "cross_subject119"]
        if metrics.duplicated(["record_id", "route_id"]).any():
            raise ValueError("Duplicate record/route in paper index")
        raw = json.loads((self.release / "raw/record_index.json").read_text(encoding="utf-8"))
        raw_index = {row["record_id"]: row for row in raw}
        hf = metrics[metrics.route_id == "HF"].sort_values(["scene", "subject_id", "record_id"])
        if len(hf) != 119 or hf.record_id.nunique() != 119:
            raise ValueError("Paper cohort must contain exactly 119 unique records")
        if hf.subject_id.nunique() != 6 or set(hf.scene) != set(SCENES):
            raise ValueError("Paper cohort subject/scene identity mismatch")
        counts = {}
        self.records = []
        for row in hf.itertuples():
            key = row.subject_id, row.scene
            counts[key] = counts.get(key, 0) + 1
            routes = metrics[metrics.record_id == row.record_id]
            if set(routes.route_id) != {"HF", "ACC", "FFT"}:
                raise ValueError(f"Missing route: {row.record_id}")
            if set(routes.subject_id) != {row.subject_id} or set(routes.scene) != {row.scene}:
                raise ValueError("Inconsistent route identity")
            item = raw_index[row.record_id]
            label = f"{SCENES[row.scene][0]}-{row.subject_id.split('-')[-1]}-{counts[key]:02d}"
            self.records.append(PaperRecord(
                row.record_id, label, row.subject_id, row.scene, counts[key],
                self.release / item["data_path"], self.release / item["ref_path"],
                {r["route_id"]: r for r in routes.to_dict("records")},
            ))
        self.by_label = {r.label: r for r in self.records}
        self.windows = pd.read_csv(self.audit / "window_route_metrics.csv")
        self.windows = self.windows[self.windows.cohort == "cross_subject119"]
        if self.windows.duplicated(["record_id", "route_id", "window_idx"]).any():
            raise ValueError("Duplicate paper windows")
        for record in self.records:
            for route, meta in record.routes.items():
                rows = self.route_windows(record, route)
                if len(rows) != int(meta["total_window_count"]):
                    raise ValueError(f"Missing windows: {record.label}/{route}")

    def select(self, subject: str, scene: str) -> list[PaperRecord]:
        return [r for r in self.records if r.subject == subject and r.scene == scene]

    def route_windows(self, record: PaperRecord, route: str) -> pd.DataFrame:
        return self.windows[(self.windows.record_id == record.record_id)
                            & (self.windows.route_id == route)].sort_values("center_s")

    def load_record(self, record: PaperRecord, route: str) -> dict:
        meta = record.routes[route]
        trace_path = self.audit / meta["trace_path"]
        if sha256(trace_path) != meta["trace_sha256"]:
            raise ValueError("Archived trace hash mismatch")
        archive = json.loads(trace_path.read_text(encoding="utf-8"))
        if sha256(record.data_path) != archive["identity"]["data_sha256"]:
            raise ValueError("Raw data hash mismatch")
        raw = pd.read_csv(record.data_path)
        t = raw["Time(s)"].to_numpy(float)
        y = raw[RAW_COLUMNS].to_numpy(float)
        valid = np.isfinite(y).all(axis=1) & (y[:, 0] >= 0)
        if "ValidFlag" in raw:
            valid &= raw.ValidFlag.fillna(0).to_numpy() > 0
        interpolated = (raw.InterpFlag.fillna(1).to_numpy() != 0
                        if "InterpFlag" in raw else np.zeros(len(t), dtype=bool))
        rows = self.route_windows(record, route).copy()
        hr = np.asarray(archive["hr"], float)
        column = 2 if route == "FFT" else 3
        np.testing.assert_allclose(rows.center_s, hr[:, 0], atol=1e-9, rtol=0)
        np.testing.assert_allclose(rows.prediction_bpm, hr[:, column],
                                   atol=1e-9, rtol=0, equal_nan=True)
        return dict(record=record, route=route, archive=archive, rows=rows,
                    time=t, raw=y, valid=valid, interpolated=interpolated,
                    trace_path=trace_path)


def nearest_window(rows: pd.DataFrame, seconds: float, *, start=False, duration=8.0) -> int:
    target = float(seconds) + (duration / 2 if start else 0)
    return int(np.argmin(np.abs(rows.center_s.to_numpy(float) - target)))


def run_frozen_replay(index: PaperReplayIndex, record: PaperRecord, route: str,
                      cache_root: Path, cancelled: Event) -> dict:
    """Run an unmodified frozen solver in its own interpreter; cancellation kills it."""
    meta = record.routes[route]
    cache_root.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-I", "-B", str(Path(__file__).with_name("paper_replay_runner.py")),
               "--source", str(index.release / "source_code/main"),
               "--trace", str(index.audit / meta["trace_path"]),
               "--trace-sha", meta["trace_sha256"], "--data", str(record.data_path),
               "--ref", str(record.ref_path), "--cache", str(cache_root)]
    env = dict(os.environ, NUMBA_CACHE_DIR=str(cache_root / "numba"), PYTHONDONTWRITEBYTECODE="1")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", env=env,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        while True:
            if cancelled.is_set():
                raise InterruptedError("Replay cancelled")
            try:
                stdout, stderr = process.communicate(timeout=0.15)
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode:
            raise RuntimeError(stderr[-5000:] or stdout[-5000:])
        path = Path(stdout.strip().splitlines()[-1])
        payload = json.loads(path.read_text(encoding="utf-8"))
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
            payload["arrays"] = {key: arrays[key] for key in arrays.files}
        return payload
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def selection_payload(loaded: dict, position: int, context_s: float,
                      rest_start_s: float, rest_duration_s: float, notes: str) -> dict:
    record = loaded["record"]
    center = float(loaded["rows"].iloc[position].center_s)
    cfg = loaded["archive"]["identity"]["config"]
    duration = float(cfg["window_seconds"])
    return dict(schema_version=1, record_label=record.label, subject=record.subject,
                scene=record.scene, ordinal=record.ordinal, route=loaded["route"],
                version=VERSION, source_sha256=loaded["archive"]["identity"]["source_sha256"],
                trace_sha256=record.routes[loaded["route"]]["trace_sha256"],
                start_s=center-duration/2, center_s=center, end_s=center+duration/2,
                context_s=context_s, rest_start_s=rest_start_s,
                rest_duration_s=rest_duration_s, notes=notes)


def validate_selection(index: PaperReplayIndex, payload: dict) -> PaperRecord:
    if payload.get("schema_version") != 1 or payload.get("version") != VERSION:
        raise ValueError("Selection version is not the paper release")
    record = index.by_label[payload["record_label"]]
    if record.routes[payload["route"]]["trace_sha256"] != payload["trace_sha256"]:
        raise ValueError("Selection trace identity mismatch")
    return record
