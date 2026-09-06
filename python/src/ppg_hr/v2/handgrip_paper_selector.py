"""Frozen two-coordinate Handgrip selector used by the paper's 119-record result."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .handgrip_blind_composition import FEATURE_CLASS_IDS

COORDINATE_PAIR = (12, 101)


def fingerprint_vector(payload: Mapping[str, Any]) -> np.ndarray:
    """Keep the original signal-feature ordering used during training."""
    return np.asarray(
        [float(value) for name in FEATURE_CLASS_IDS for value in payload["full"][name]],
        dtype=float,
    )


def fit_coordinate_selector(
    training_features: np.ndarray, training_pair_mae: np.ndarray
) -> DecisionTreeClassifier:
    """Fit one stump from training records; columns follow ``COORDINATE_PAIR``."""
    pair_mae = np.asarray(training_pair_mae, dtype=float)
    labels = (pair_mae[:, 0] > pair_mae[:, 1]).astype(int)
    tree = DecisionTreeClassifier(max_depth=1, min_samples_leaf=2, random_state=0)
    tree.fit(training_features, labels)
    return tree


def predict_coordinates(tree: DecisionTreeClassifier, features: np.ndarray) -> np.ndarray:
    """Select coordinates using signal features alone."""
    labels = tree.predict(features).astype(int)
    return np.asarray(COORDINATE_PAIR, dtype=int)[labels]
