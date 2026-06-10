from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

from .data import interpolate_nonfinite, zero_phase_lowpass


@dataclass
class PressureFeatures:
    names: tuple[str, ...]
    values: np.ndarray


class PressureEffectModel(Protocol):
    name: str

    def fit(
        self,
        features: PressureFeatures,
        target: np.ndarray,
        state: np.ndarray,
    ) -> "PressureEffectModel": ...

    def predict(self, features: PressureFeatures, state: np.ndarray) -> np.ndarray: ...

    def parameters(self) -> dict[str, Any]: ...


def _derivative(values: np.ndarray, fs_hz: float) -> np.ndarray:
    gradient = np.gradient(interpolate_nonfinite(values)) * float(fs_hz)
    return zero_phase_lowpass(
        gradient,
        fs_hz=fs_hz,
        cutoff_hz=min(2.0, 0.45 * fs_hz),
        order=2,
    )


def build_pressure_features(
    ut1_mv: np.ndarray,
    ut2_mv: np.ndarray,
    *,
    fs_hz: float,
    group: str,
) -> PressureFeatures:
    ut1 = interpolate_nonfinite(ut1_mv)
    ut2 = interpolate_nonfinite(ut2_mv)
    n = min(ut1.size, ut2.size)
    ut1 = ut1[:n]
    ut2 = ut2[:n]
    baseline_count = max(1, min(n, int(round(0.20 * n))))
    ut1 = ut1 - float(np.median(ut1[:baseline_count]))
    ut2 = ut2 - float(np.median(ut2[:baseline_count]))
    common = 0.5 * (ut1 + ut2)
    difference = 0.5 * (ut1 - ut2)
    mapping = {
        "ut1": (("ut1", "ut1_d1"), (ut1, _derivative(ut1, fs_hz))),
        "ut2": (("ut2", "ut2_d1"), (ut2, _derivative(ut2, fs_hz))),
        "common": (
            ("common", "common_d1"),
            (common, _derivative(common, fs_hz)),
        ),
        "common_difference": (
            ("common", "common_d1", "difference", "difference_d1"),
            (
                common,
                _derivative(common, fs_hz),
                difference,
                _derivative(difference, fs_hz),
            ),
        ),
    }
    if group not in mapping:
        raise ValueError(f"Unsupported pressure feature group: {group}")
    names, columns = mapping[group]
    return PressureFeatures(names=names, values=np.column_stack(columns))


def _lag_matrix(values: np.ndarray, taps: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    n, features = matrix.shape
    width = max(1, int(taps))
    out = np.zeros((n, features * width), dtype=float)
    for lag in range(width):
        if lag == 0:
            out[:, lag * features : (lag + 1) * features] = matrix
        else:
            out[lag:, lag * features : (lag + 1) * features] = matrix[:-lag]
            out[:lag, lag * features : (lag + 1) * features] = matrix[0]
    return out


class RidgeFIRModel:
    name = "ridge_fir"

    def __init__(self, *, taps: int = 11, alpha: float = 1e-3) -> None:
        self.taps = max(1, int(taps))
        self.alpha = float(alpha)
        self._model = Ridge(alpha=self.alpha, fit_intercept=True)
        self._feature_names: tuple[str, ...] = ()

    def fit(
        self,
        features: PressureFeatures,
        target: np.ndarray,
        state: np.ndarray,
    ) -> "RidgeFIRModel":
        del state
        design = _lag_matrix(features.values, self.taps)
        self._model.fit(design, interpolate_nonfinite(target))
        self._feature_names = features.names
        return self

    def predict(self, features: PressureFeatures, state: np.ndarray) -> np.ndarray:
        del state
        return np.asarray(
            self._model.predict(_lag_matrix(features.values, self.taps)),
            dtype=float,
        )

    def parameters(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "taps": self.taps,
            "alpha": self.alpha,
            "feature_names": list(self._feature_names),
            "intercept": float(self._model.intercept_),
            "coefficients": np.asarray(self._model.coef_, dtype=float).tolist(),
        }


class HysteresisSplineModel:
    name = "hysteresis_spline"

    def __init__(self, *, n_knots: int = 4, alpha: float = 1e-3) -> None:
        self.n_knots = max(3, int(n_knots))
        self.alpha = float(alpha)
        self._branches: dict[str, tuple[SplineTransformer, Ridge]] = {}
        self._feature_names: tuple[str, ...] = ()

    def fit(
        self,
        features: PressureFeatures,
        target: np.ndarray,
        state: np.ndarray,
    ) -> "HysteresisSplineModel":
        values = np.asarray(features.values, dtype=float)
        output = interpolate_nonfinite(target)
        branch_state = np.asarray(state, dtype=float)
        self._feature_names = features.names
        for name, mask in (
            ("loading", branch_state >= 0.0),
            ("release", branch_state < 0.0),
        ):
            if np.count_nonzero(mask) < self.n_knots + 2:
                mask = np.ones(values.shape[0], dtype=bool)
            spline = SplineTransformer(
                n_knots=self.n_knots,
                degree=2,
                include_bias=False,
                extrapolation="linear",
            )
            transformed = spline.fit_transform(values[mask])
            model = Ridge(alpha=self.alpha, fit_intercept=True)
            model.fit(transformed, output[mask])
            self._branches[name] = (spline, model)
        return self

    def predict(self, features: PressureFeatures, state: np.ndarray) -> np.ndarray:
        values = np.asarray(features.values, dtype=float)
        branch_state = np.asarray(state, dtype=float)
        out = np.empty(values.shape[0], dtype=float)
        for name, mask in (
            ("loading", branch_state >= 0.0),
            ("release", branch_state < 0.0),
        ):
            spline, model = self._branches[name]
            if np.any(mask):
                out[mask] = model.predict(spline.transform(values[mask]))
        return out

    def parameters(self) -> dict[str, Any]:
        branches: dict[str, Any] = {}
        for name, (_, model) in self._branches.items():
            branches[name] = {
                "intercept": float(model.intercept_),
                "coefficients": np.asarray(model.coef_, dtype=float).tolist(),
            }
        return {
            "name": self.name,
            "n_knots": self.n_knots,
            "alpha": self.alpha,
            "feature_names": list(self._feature_names),
            "branches": branches,
        }


class HammersteinFIRModel:
    name = "hammerstein_fir"

    def __init__(
        self,
        *,
        n_knots: int = 4,
        taps: int = 11,
        alpha: float = 1e-3,
    ) -> None:
        self.n_knots = max(3, int(n_knots))
        self.taps = max(1, int(taps))
        self.alpha = float(alpha)
        self._spline = SplineTransformer(
            n_knots=self.n_knots,
            degree=2,
            include_bias=False,
            extrapolation="linear",
        )
        self._model = Ridge(alpha=self.alpha, fit_intercept=True)
        self._feature_names: tuple[str, ...] = ()

    def fit(
        self,
        features: PressureFeatures,
        target: np.ndarray,
        state: np.ndarray,
    ) -> "HammersteinFIRModel":
        del state
        static = self._spline.fit_transform(features.values)
        self._model.fit(_lag_matrix(static, self.taps), interpolate_nonfinite(target))
        self._feature_names = features.names
        return self

    def predict(self, features: PressureFeatures, state: np.ndarray) -> np.ndarray:
        del state
        static = self._spline.transform(features.values)
        return np.asarray(
            self._model.predict(_lag_matrix(static, self.taps)),
            dtype=float,
        )

    def parameters(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_knots": self.n_knots,
            "taps": self.taps,
            "alpha": self.alpha,
            "feature_names": list(self._feature_names),
            "intercept": float(self._model.intercept_),
            "coefficients": np.asarray(self._model.coef_, dtype=float).tolist(),
        }
