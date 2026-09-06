"""Performance-blind Handgrip signal fingerprints and training composition."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

FEATURE_CLASS_IDS = (
    "hf_interface_baseline",
    "acc_tremor",
    "hf_relative_acc_ppg_artifact_response",
    "post_motion_hf_recovery",
    "artifact_persistence_bandwidth",
    "dual_hf_consistency",
)
FEATURE_COMPONENT_COUNTS = {
    "hf_interface_baseline": 2,
    "acc_tremor": 1,
    "hf_relative_acc_ppg_artifact_response": 2,
    "post_motion_hf_recovery": 2,
    "artifact_persistence_bandwidth": 2,
    "dual_hf_consistency": 2,
}


@dataclass(frozen=True)
class HandgripSignal:
    """The no-reference raw channels required by the frozen fingerprint contract."""

    record_id: str
    subject_id: str
    time_s: np.ndarray
    valid: np.ndarray
    hf1: np.ndarray
    hf2: np.ndarray
    accx: np.ndarray
    accy: np.ndarray
    accz: np.ndarray
    ppg: np.ndarray
    sampling_rate_hz: int = 100


def build_frozen_baseline_snapshot(
    probe: dict[str, Any],
    *,
    challenge_record_ids: tuple[str, ...],
    source_sha256: str,
) -> dict[str, Any]:
    """Extract only the predeclared D24 acceptance baselines from a local probe."""

    if len(source_sha256) != 64:
        raise ValueError("handgrip_baseline_source_sha256")
    summaries = dict(probe.get("variant_summaries") or {})
    hf = dict(summaries.get("hf_original_d24_reselected") or {})
    acc = dict(summaries.get("acc_minimax_d24") or {})
    reachability = {str(row["record_id"]): row for row in list(probe.get("reachability") or [])}
    try:
        result = {
            "schema_id": "d24_handgrip_blind_composition_baseline_snapshot_v1",
            "source_sha256": source_sha256,
            "hf_mean_mae_bpm": float(hf["mean"]),
            "acc_mean_mae_bpm": float(acc["mean"]),
            "hf_sample_sd_bpm": float(hf["sample_sd"]),
            "challenge_hf_mae_bpm": {
                record_id: float(reachability[record_id]["selected_mae_bpm"])
                for record_id in challenge_record_ids
            },
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("handgrip_baseline_probe_contract") from error
    values = [
        result["hf_mean_mae_bpm"],
        result["acc_mean_mae_bpm"],
        result["hf_sample_sd_bpm"],
        *result["challenge_hf_mae_bpm"].values(),
    ]
    if not all(np.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("handgrip_baseline_nonfinite")
    return result


def load_handgrip_signal_csv(path: Path, *, subject_id: str) -> HandgripSignal:
    """Load only the no-reference columns used by the Handgrip fingerprint."""

    source = Path(path).resolve()
    columns = {
        "time_s": "Time(s)",
        "valid": "ValidFlag",
        "hf1": "Ut1(mV)",
        "hf2": "Ut2(mV)",
        "accx": "AccX(g)",
        "accy": "AccY(g)",
        "accz": "AccZ(g)",
        "ppg": "PPG_Green",
    }
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        available = set(reader.fieldnames or ())
        missing = sorted(set(columns.values()) - available)
        if missing:
            raise ValueError(f"handgrip_csv_columns:{source.stem}:{','.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"handgrip_csv_empty:{source.stem}")

    def floats(column: str) -> np.ndarray:
        return np.asarray([float(row[column]) for row in rows], dtype=float)

    return HandgripSignal(
        record_id=source.stem,
        subject_id=str(subject_id),
        time_s=floats(columns["time_s"]),
        valid=np.asarray([int(float(row[columns["valid"]])) == 1 for row in rows], dtype=bool),
        hf1=floats(columns["hf1"]),
        hf2=floats(columns["hf2"]),
        accx=floats(columns["accx"]),
        accy=floats(columns["accy"]),
        accz=floats(columns["accz"]),
        ppg=floats(columns["ppg"]),
        sampling_rate_hz=100,
    )


def extract_signal_fingerprint(signal: HandgripSignal) -> dict[str, Any]:
    """Return the full, four-block and fixed-30-second fingerprint versions."""

    prepared, audit = _prepare_signal(signal)
    recovery, recovery_audit = _post_motion_recovery(prepared)
    audit.update(recovery_audit)

    primary_blocks = tuple(np.array_split(np.arange(prepared["time_s"].size), 4))
    primary_rows = tuple(_block_features(prepared, indices) for indices in primary_blocks)
    full = _aggregate_block_features(primary_rows, recovery)
    leave_one_out = [
        _aggregate_block_features(
            tuple(row for index, row in enumerate(primary_rows) if index != omitted),
            recovery,
        )
        for omitted in range(4)
    ]

    samples_per_30s = 30 * int(signal.sampling_rate_hz)
    required_samples = 6 * samples_per_30s
    if prepared["time_s"].size < required_samples:
        raise ValueError(f"handgrip_signal_shorter_than_180s:{signal.record_id}")
    sensitivity_blocks = tuple(
        np.arange(index * samples_per_30s, (index + 1) * samples_per_30s) for index in range(6)
    )
    sensitivity_rows = tuple(_block_features(prepared, indices) for indices in sensitivity_blocks)
    sensitivity_full = _aggregate_block_features(sensitivity_rows, recovery)
    sensitivity_leave_one_out = [
        _aggregate_block_features(
            tuple(row for index, row in enumerate(sensitivity_rows) if index != omitted),
            recovery,
        )
        for omitted in range(6)
    ]

    return {
        "record_id": signal.record_id,
        "physical_subject_id": signal.subject_id,
        "feature_contract_id": "handgrip_no_hr_signal_fingerprint_v1",
        "full": full,
        "leave_one_block_out": leave_one_out,
        "sensitivity_30s": {
            "full": sensitivity_full,
            "leave_one_block_out": sensitivity_leave_one_out,
        },
        "audit": audit,
    }


def compose_fold_training_core(fingerprints: list[dict[str, Any]]) -> dict[str, Any]:
    """Freeze one fold's stable common modes and performance-blind training core."""

    ordered = sorted(fingerprints, key=lambda row: str(row["record_id"]))
    if len(ordered) < 3:
        raise ValueError("handgrip_training_record_count")
    record_ids = [str(row["record_id"]) for row in ordered]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("duplicate_handgrip_training_record")
    subject_by_record = {str(row["record_id"]): str(row["physical_subject_id"]) for row in ordered}
    if len(set(subject_by_record.values())) < 1:
        raise ValueError("handgrip_training_subject_count")

    primary_versions = [
        [row["full"] for row in ordered],
        *[[row["leave_one_block_out"][index] for row in ordered] for index in range(4)],
    ]
    primary = _group_feature_versions(record_ids, primary_versions)
    core = _select_training_core(
        record_ids,
        subject_by_record,
        primary["selected_cluster_count"],
        primary["full_labels"],
        np.asarray(primary["full_distance_matrix"], dtype=float),
    )

    sensitivity_versions = [
        [row["sensitivity_30s"]["full"] for row in ordered],
        *[
            [row["sensitivity_30s"]["leave_one_block_out"][index] for row in ordered]
            for index in range(6)
        ],
    ]
    sensitivity = _group_feature_versions(record_ids, sensitivity_versions)
    sensitivity_core = _select_training_core(
        record_ids,
        subject_by_record,
        sensitivity["selected_cluster_count"],
        sensitivity["full_labels"],
        np.asarray(sensitivity["full_distance_matrix"], dtype=float),
    )
    return {
        "grouping_rule_id": "fold_local_stable_agglomerative_representative_v1",
        **primary,
        **core,
        "sensitivity_30s": {**sensitivity, **sensitivity_core},
    }


def _group_feature_versions(
    record_ids: list[str],
    versions: list[list[dict[str, list[float]]]],
) -> dict[str, Any]:
    distance_versions: list[np.ndarray] = []
    scaling_versions: list[dict[str, Any]] = []
    for rows in versions:
        distance, scaling = _equal_class_distance(rows)
        distance_versions.append(distance)
        scaling_versions.append(scaling)

    candidates: dict[str, dict[str, Any]] = {}
    for cluster_count in (2, 3):
        labels_versions: list[np.ndarray] = []
        silhouettes: list[float] = []
        if cluster_count >= len(record_ids):
            continue
        for distance in distance_versions:
            labels = AgglomerativeClustering(
                n_clusters=cluster_count,
                metric="precomputed",
                linkage="average",
            ).fit_predict(distance)
            labels_versions.append(labels)
            if len(set(int(value) for value in labels)) < 2:
                silhouettes.append(0.0)
            else:
                silhouettes.append(float(silhouette_score(distance, labels, metric="precomputed")))
        pairwise_ari = [
            float(adjusted_rand_score(labels_versions[left], labels_versions[right]))
            for left in range(len(labels_versions))
            for right in range(left + 1, len(labels_versions))
        ]
        median_silhouette = float(np.median(silhouettes))
        median_ari = float(np.median(pairwise_ari))
        minimum_ari = min(pairwise_ari)
        passed = bool(
            all(value > 0.0 for value in silhouettes)
            and median_silhouette >= 0.25
            and median_ari >= 0.80
            and minimum_ari >= 0.50
        )
        candidates[str(cluster_count)] = {
            "cluster_count": cluster_count,
            "silhouettes": silhouettes,
            "median_silhouette": median_silhouette,
            "pairwise_ari": pairwise_ari,
            "median_pairwise_ari": median_ari,
            "minimum_pairwise_ari": minimum_ari,
            "passed": passed,
            "labels_by_version": [labels.tolist() for labels in labels_versions],
        }

    passing = [value for value in candidates.values() if value["passed"]]
    if passing:
        selected = min(
            passing,
            key=lambda row: (-float(row["median_silhouette"]), int(row["cluster_count"])),
        )
        selected_cluster_count = int(selected["cluster_count"])
        full_labels = list(selected["labels_by_version"][0])
        fallback_policy = None
    else:
        selected_cluster_count = 1
        full_labels = [0] * len(record_ids)
        fallback_policy = "retain_all_training_records"
    return {
        "record_ids": record_ids,
        "selected_cluster_count": selected_cluster_count,
        "fallback_policy": fallback_policy,
        "candidates": candidates,
        "full_labels": full_labels,
        "full_distance_matrix": distance_versions[0].tolist(),
        "scaling_by_version": scaling_versions,
    }


def _equal_class_distance(
    feature_rows: list[dict[str, list[float]]],
) -> tuple[np.ndarray, dict[str, Any]]:
    record_count = len(feature_rows)
    scaled_by_class: dict[str, np.ndarray] = {}
    scaling: dict[str, Any] = {}
    for class_id in FEATURE_CLASS_IDS:
        expected_count = FEATURE_COMPONENT_COUNTS[class_id]
        matrix = np.asarray([row[class_id] for row in feature_rows], dtype=float)
        if matrix.shape != (record_count, expected_count) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"handgrip_feature_contract:{class_id}")
        if class_id == "post_motion_hf_recovery":
            if np.any((matrix < 0.0) | (matrix > 1.0)):
                raise ValueError("handgrip_recovery_feature_bounds")
            scaled = matrix
            centers = np.zeros(expected_count, dtype=float)
            scales = np.ones(expected_count, dtype=float)
            scale_sources = ["bounded_identity"] * expected_count
        else:
            centers = np.median(matrix, axis=0)
            q75 = np.quantile(matrix, 0.75, axis=0)
            q25 = np.quantile(matrix, 0.25, axis=0)
            scales = q75 - q25
            scale_sources = []
            for component in range(expected_count):
                if scales[component] > 0.0:
                    scale_sources.append("iqr")
                    continue
                mad = float(np.median(np.abs(matrix[:, component] - centers[component])))
                fallback = 1.4826 * mad
                if fallback > 0.0:
                    scales[component] = fallback
                    scale_sources.append("mad")
                else:
                    scales[component] = 1.0
                    scale_sources.append("zero_contribution")
            scaled = (matrix - centers) / scales
            for component, source in enumerate(scale_sources):
                if source == "zero_contribution":
                    scaled[:, component] = 0.0
        scaled_by_class[class_id] = scaled
        scaling[class_id] = {
            "center": [float(value) for value in centers],
            "scale": [float(value) for value in scales],
            "scale_source": scale_sources,
        }

    total_squared = np.zeros((record_count, record_count), dtype=float)
    for class_id in FEATURE_CLASS_IDS:
        matrix = scaled_by_class[class_id]
        differences = matrix[:, None, :] - matrix[None, :, :]
        class_distance = np.sqrt(np.mean(differences**2, axis=2))
        total_squared += class_distance**2
    distance = np.sqrt(total_squared / len(FEATURE_CLASS_IDS))
    np.fill_diagonal(distance, 0.0)
    return distance, scaling


def _select_training_core(
    record_ids: list[str],
    subject_by_record: dict[str, str],
    cluster_count: int,
    labels: list[int],
    distance: np.ndarray,
) -> dict[str, Any]:
    if cluster_count == 1:
        return {
            "training_core_record_ids": list(record_ids),
            "common_patterns": [],
            "rare_patterns": [],
        }

    retained: set[str] = set()
    common_patterns: list[dict[str, Any]] = []
    rare_patterns: list[dict[str, Any]] = []
    labels_array = np.asarray(labels, dtype=int)
    for label in sorted(set(int(value) for value in labels)):
        members = np.flatnonzero(labels_array == label)
        member_ids = [record_ids[index] for index in members]
        subjects = sorted({subject_by_record[record_id] for record_id in member_ids})
        if len(subjects) == 1:
            retained.update(member_ids)
            rare_patterns.append(
                {
                    "label": label,
                    "subject_ids": subjects,
                    "record_ids": member_ids,
                    "retained_record_ids": member_ids,
                }
            )
            continue

        totals = np.sum(distance[np.ix_(members, members)], axis=1)
        minimum = float(np.min(totals))
        medoid_candidates = [
            record_ids[members[index]]
            for index, value in enumerate(totals)
            if np.isclose(float(value), minimum, atol=1e-12, rtol=0.0)
        ]
        medoid_id = min(medoid_candidates)
        medoid_index = record_ids.index(medoid_id)
        representatives: list[str] = []
        for subject in subjects:
            subject_members = [
                index for index in members if subject_by_record[record_ids[index]] == subject
            ]
            minimum_distance = min(
                float(distance[index, medoid_index]) for index in subject_members
            )
            candidates = [
                record_ids[index]
                for index in subject_members
                if np.isclose(
                    float(distance[index, medoid_index]),
                    minimum_distance,
                    atol=1e-12,
                    rtol=0.0,
                )
            ]
            representative = min(candidates)
            representatives.append(representative)
            retained.add(representative)
        common_patterns.append(
            {
                "label": label,
                "subject_ids": subjects,
                "record_ids": member_ids,
                "medoid_record_id": medoid_id,
                "retained_record_ids": representatives,
            }
        )

    expected_subjects = set(subject_by_record.values())
    retained_subjects = {subject_by_record[record_id] for record_id in retained}
    if retained_subjects != expected_subjects:
        raise ValueError("handgrip_training_subject_lost")
    return {
        "training_core_record_ids": sorted(retained),
        "common_patterns": common_patterns,
        "rare_patterns": rare_patterns,
    }


def _prepare_signal(signal: HandgripSignal) -> tuple[dict[str, Any], dict[str, Any]]:
    fs = int(signal.sampling_rate_hz)
    if fs != 100:
        raise ValueError(f"handgrip_sampling_rate:{signal.record_id}:{fs}")
    channels = {
        "hf1": np.asarray(signal.hf1, dtype=float),
        "hf2": np.asarray(signal.hf2, dtype=float),
        "accx": np.asarray(signal.accx, dtype=float),
        "accy": np.asarray(signal.accy, dtype=float),
        "accz": np.asarray(signal.accz, dtype=float),
        "ppg": np.asarray(signal.ppg, dtype=float),
    }
    time_s = np.asarray(signal.time_s, dtype=float)
    valid = np.asarray(signal.valid, dtype=bool)
    sizes = {time_s.size, valid.size, *(values.size for values in channels.values())}
    if len(sizes) != 1 or not time_s.size:
        raise ValueError(f"handgrip_signal_shape:{signal.record_id}")
    if not np.all(np.isfinite(time_s)) or np.any(np.diff(time_s) <= 0.0):
        raise ValueError(f"handgrip_time_axis:{signal.record_id}")
    expected_step = 1.0 / fs
    if not np.allclose(np.diff(time_s), expected_step, atol=1e-6, rtol=0.0):
        raise ValueError(f"handgrip_time_step:{signal.record_id}")

    invalid = ~valid
    filled: dict[str, np.ndarray] = {}
    for name, values in channels.items():
        usable = valid & np.isfinite(values)
        if np.count_nonzero(usable) < 2:
            raise ValueError(f"handgrip_channel_insufficient:{signal.record_id}:{name}")
        if np.any(valid & ~np.isfinite(values)):
            raise ValueError(f"handgrip_nonfinite_valid_sample:{signal.record_id}:{name}")
        filled[name] = np.interp(time_s, time_s[usable], values[usable])

    slow = {name: _bandpass(values, fs, 0.05, 0.5) for name, values in filled.items()}
    algorithm = {name: _bandpass(values, fs, 0.5, 5.0) for name, values in filled.items()}
    acc_tremor = {name: _bandpass(filled[name], fs, 4.0, 12.0) for name in ("accx", "accy", "accz")}
    acc_dynamic = {
        name: _bandpass(filled[name], fs, 0.5, 12.0) for name in ("accx", "accy", "accz")
    }
    return (
        {
            "record_id": signal.record_id,
            "fs": fs,
            "time_s": time_s,
            "valid": valid,
            "raw": filled,
            "slow": slow,
            "algorithm": algorithm,
            "acc_tremor": acc_tremor,
            "acc_dynamic": acc_dynamic,
        },
        {
            "sample_count": int(time_s.size),
            "valid_sample_count": int(np.count_nonzero(valid)),
            "invalid_sample_count": int(np.count_nonzero(invalid)),
            "invalid_policy": "interpolate_for_filter_then_exclude_from_aggregation",
        },
    )


def _bandpass(values: np.ndarray, fs: int, low_hz: float, high_hz: float) -> np.ndarray:
    sos = butter(4, (low_hz, high_hz), btype="bandpass", fs=fs, output="sos")
    return np.asarray(sosfiltfilt(sos, values), dtype=float)


def _block_features(prepared: dict[str, Any], indices: np.ndarray) -> dict[str, list[float]]:
    valid = prepared["valid"][indices]
    if np.count_nonzero(valid) < 100:
        raise ValueError(f"handgrip_block_insufficient:{prepared['record_id']}")
    raw = prepared["raw"]
    slow = prepared["slow"]
    tremor = prepared["acc_tremor"]
    dynamic = prepared["acc_dynamic"]

    interface = [
        _quantile_span(slow[name][indices], valid) / _iqr(raw[name][indices], valid)
        for name in ("hf1", "hf2")
    ]
    tremor_rms = _xyz_rms(tuple(tremor[name][indices] for name in ("accx", "accy", "accz")), valid)
    dynamic_rms = _xyz_rms(
        tuple(dynamic[name][indices] for name in ("accx", "accy", "accz")), valid
    )
    if dynamic_rms <= 0.0:
        raise ValueError(f"handgrip_acc_dynamic_zero:{prepared['record_id']}")

    algorithm_advantage = _windowed_correlation_advantage(prepared, indices)
    slow_hf = max(
        abs(_correlation(slow["ppg"][indices], slow[name][indices], valid))
        for name in ("hf1", "hf2")
    )
    slow_acc = max(
        abs(_correlation(slow["ppg"][indices], slow[name][indices], valid))
        for name in ("accx", "accy", "accz")
    )

    persistence = float(
        np.median(
            [
                _lag_one_envelope_autocorrelation(slow[name][indices], valid, prepared["fs"])
                for name in ("hf1", "hf2")
            ]
        )
    )
    bandwidth = float(
        np.median(
            [
                _power_bandwidth_90(slow[name][indices], valid, prepared["fs"]) / 0.45
                for name in ("hf1", "hf2")
            ]
        )
    )
    hf_pair_corr = _max_lagged_abs_correlation(
        slow["hf1"][indices],
        slow["hf2"][indices],
        valid,
        max_lag_samples=prepared["fs"],
    )
    hf_rms = [_rms(slow[name][indices], valid) for name in ("hf1", "hf2")]
    hf_ratio = min(hf_rms) / max(hf_rms) if max(hf_rms) > 0.0 else 0.0

    result = {
        "hf_interface_baseline": interface,
        "acc_tremor": [tremor_rms / dynamic_rms],
        "hf_relative_acc_ppg_artifact_response": [
            algorithm_advantage,
            slow_hf - slow_acc,
        ],
        "artifact_persistence_bandwidth": [persistence, bandwidth],
        "dual_hf_consistency": [hf_pair_corr, hf_ratio],
    }
    for values in result.values():
        if not all(np.isfinite(values)):
            raise ValueError(f"handgrip_nonfinite_feature:{prepared['record_id']}")
    return result


def _aggregate_block_features(
    rows: tuple[dict[str, list[float]], ...],
    recovery: list[float],
) -> dict[str, list[float]]:
    if not rows:
        raise ValueError("empty_handgrip_feature_blocks")
    result: dict[str, list[float]] = {}
    for class_id in FEATURE_CLASS_IDS:
        if class_id == "post_motion_hf_recovery":
            result[class_id] = [float(value) for value in recovery]
            continue
        matrix = np.asarray([row[class_id] for row in rows], dtype=float)
        result[class_id] = [float(value) for value in np.median(matrix, axis=0)]
    return result


def _post_motion_recovery(
    prepared: dict[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    fs = int(prepared["fs"])
    valid = prepared["valid"]
    acc = prepared["algorithm"]
    acc_mag = np.sqrt(acc["accx"] ** 2 + acc["accy"] ** 2 + acc["accz"] ** 2)
    calibration = acc_mag[: min(acc_mag.size, 30 * fs)]
    calibration_valid = valid[: calibration.size]
    calibration_values = calibration[calibration_valid]
    threshold = 2.5 * (
        float(np.std(calibration_values, ddof=1)) if calibration_values.size > 1 else 0.0
    )
    flags: list[bool] = []
    centers: list[float] = []
    window = 8 * fs
    step = fs
    for start in range(0, acc_mag.size - window + 1, step):
        stop = start + window
        mask = valid[start:stop]
        values = acc_mag[start:stop][mask]
        score = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        flags.append(score > threshold)
        centers.append((start + window / 2.0) / fs)
    event_ends = [
        centers[index] for index in range(len(flags) - 1) if flags[index] and not flags[index + 1]
    ]

    second_count = prepared["time_s"].size // fs
    envelope_times: list[float] = []
    envelope_values: list[float] = []
    hf1 = prepared["slow"]["hf1"]
    hf2 = prepared["slow"]["hf2"]
    for second in range(second_count):
        start = second * fs
        stop = start + fs
        mask = valid[start:stop]
        if np.count_nonzero(mask) < fs // 2:
            continue
        combined = np.sqrt((hf1[start:stop] ** 2 + hf2[start:stop] ** 2) / 2.0)
        envelope_times.append(second + 0.5)
        envelope_values.append(float(np.median(combined[mask])))
    times = np.asarray(envelope_times, dtype=float)
    envelope = np.asarray(envelope_values, dtype=float)

    recovery_times: list[float] = []
    censored = 0
    for event_end in event_ends:
        pre = envelope[(times >= event_end - 10.0) & (times <= event_end)]
        if pre.size < 3:
            recovery_times.append(30.0)
            censored += 1
            continue
        center = float(np.median(pre))
        mad = float(np.median(np.abs(pre - center)))
        limit = center + 1.5 * mad
        post_mask = (times > event_end) & (times <= event_end + 30.0)
        post_times = times[post_mask]
        recovered = envelope[post_mask] <= limit
        found: float | None = None
        for index in range(max(0, recovered.size - 2)):
            if bool(np.all(recovered[index : index + 3])):
                found = max(0.0, float(post_times[index] - event_end))
                break
        if found is None:
            recovery_times.append(30.0)
            censored += 1
        else:
            recovery_times.append(min(30.0, found))

    if not recovery_times:
        values = [0.0, 0.0]
    else:
        values = [
            float(np.median(recovery_times)) / 30.0,
            float(censored) / len(recovery_times),
        ]
    return values, {
        "motion_detector": "existing_acc_8s_1s_default_v1",
        "motion_threshold": float(threshold),
        "motion_end_event_count": len(event_ends),
        "right_censored_recovery_count": int(censored),
    }


def _windowed_correlation_advantage(prepared: dict[str, Any], indices: np.ndarray) -> float:
    fs = int(prepared["fs"])
    window = 8 * fs
    step = fs
    start_bound = int(indices[0])
    stop_bound = int(indices[-1]) + 1
    hf_values: list[float] = []
    acc_values: list[float] = []
    algorithm = prepared["algorithm"]
    valid = prepared["valid"]
    for start in range(start_bound, stop_bound - window + 1, step):
        stop = start + window
        mask = valid[start:stop]
        ppg = algorithm["ppg"][start:stop]
        hf_values.append(
            max(
                abs(_correlation(ppg, algorithm[name][start:stop], mask)) for name in ("hf1", "hf2")
            )
        )
        acc_values.append(
            max(
                abs(_correlation(ppg, algorithm[name][start:stop], mask))
                for name in ("accx", "accy", "accz")
            )
        )
    if not hf_values:
        raise ValueError(f"handgrip_correlation_window_empty:{prepared['record_id']}")
    return float(np.median(hf_values) - np.median(acc_values))


def _correlation(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> float:
    mask = np.asarray(valid, dtype=bool) & np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(mask) < 3:
        return 0.0
    x = np.asarray(left[mask], dtype=float)
    y = np.asarray(right[mask], dtype=float)
    x -= float(np.mean(x))
    y -= float(np.mean(y))
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return 0.0 if denominator <= 0.0 else float(np.dot(x, y) / denominator)


def _max_lagged_abs_correlation(
    left: np.ndarray,
    right: np.ndarray,
    valid: np.ndarray,
    *,
    max_lag_samples: int,
) -> float:
    best = 0.0
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        if lag < 0:
            value = _correlation(left[-lag:], right[:lag], valid[-lag:] & valid[:lag])
        elif lag > 0:
            value = _correlation(left[:-lag], right[lag:], valid[:-lag] & valid[lag:])
        else:
            value = _correlation(left, right, valid)
        best = max(best, abs(value))
    return best


def _lag_one_envelope_autocorrelation(
    values: np.ndarray,
    valid: np.ndarray,
    fs: int,
) -> float:
    bins = []
    for start in range(0, values.size - fs + 1, fs):
        mask = valid[start : start + fs]
        if np.count_nonzero(mask) >= fs // 2:
            bins.append(_rms(values[start : start + fs], mask))
    if len(bins) < 3:
        return 0.0
    array = np.asarray(bins, dtype=float)
    return _correlation(array[:-1], array[1:], np.ones(array.size - 1, dtype=bool))


def _power_bandwidth_90(values: np.ndarray, valid: np.ndarray, fs: int) -> float:
    usable = np.asarray(values[valid], dtype=float)
    if usable.size < fs * 4:
        return 0.0
    frequencies, power = welch(usable, fs=fs, nperseg=min(4096, usable.size))
    band = (frequencies >= 0.05) & (frequencies <= 0.5)
    frequencies = frequencies[band]
    power = power[band]
    total = float(np.sum(power))
    if frequencies.size < 2 or total <= 0.0:
        return 0.0
    cumulative = np.cumsum(power) / total
    low = float(frequencies[min(int(np.searchsorted(cumulative, 0.05)), frequencies.size - 1)])
    high = float(frequencies[min(int(np.searchsorted(cumulative, 0.95)), frequencies.size - 1)])
    return max(0.0, high - low)


def _quantile_span(values: np.ndarray, valid: np.ndarray) -> float:
    selected = values[valid]
    return float(np.quantile(selected, 0.95) - np.quantile(selected, 0.05))


def _iqr(values: np.ndarray, valid: np.ndarray) -> float:
    selected = values[valid]
    value = float(np.quantile(selected, 0.75) - np.quantile(selected, 0.25))
    if value <= 0.0:
        raise ValueError("handgrip_raw_iqr_zero")
    return value


def _rms(values: np.ndarray, valid: np.ndarray) -> float:
    selected = values[valid]
    return float(np.sqrt(np.mean(selected**2))) if selected.size else 0.0


def _xyz_rms(axes: tuple[np.ndarray, np.ndarray, np.ndarray], valid: np.ndarray) -> float:
    selected = sum(axis[valid] ** 2 for axis in axes)
    return float(np.sqrt(np.mean(selected))) if selected.size else 0.0
