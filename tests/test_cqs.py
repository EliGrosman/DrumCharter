from __future__ import annotations

import numpy as np
import pytest

from drumcharter.training.cqs import compute_cqs


def test_cqs_exact_match_scores_one() -> None:
    labels = np.zeros((32, 8), dtype=np.float32)
    labels[10, 1] = 1.0

    report = compute_cqs({1: np.asarray([10], dtype=np.int64)}, labels)

    assert report.coverage == 1.0
    assert report.identity == 1.0
    assert report.restraint == 1.0
    assert report.playability == 1.0
    assert report.cqs == pytest.approx(1.0)


def test_cqs_gives_half_identity_for_same_family_wrong_class() -> None:
    labels = np.zeros((32, 8), dtype=np.float32)
    labels[10, 3] = 1.0

    report = compute_cqs({5: np.asarray([10], dtype=np.int64)}, labels)

    assert report.coverage == 1.0
    assert report.identity == 0.5
    assert report.restraint == 1.0
    assert report.cqs == pytest.approx(0.5 ** 0.25)


def test_cqs_penalizes_spurious_predictions() -> None:
    labels = np.zeros((32, 8), dtype=np.float32)
    labels[10, 1] = 1.0

    report = compute_cqs(
        {
            1: np.asarray([10], dtype=np.int64),
            2: np.asarray([20], dtype=np.int64),
        },
        labels,
    )

    assert report.n_spurious == 1
    assert report.restraint == 0.0
    assert report.cqs < 0.05


def test_cqs_flags_impossible_hand_triples() -> None:
    labels = np.zeros((32, 8), dtype=np.float32)

    report = compute_cqs(
        {
            1: np.asarray([10], dtype=np.int64),
            2: np.asarray([12], dtype=np.int64),
            4: np.asarray([14], dtype=np.int64),
        },
        labels,
    )

    assert report.playability == 0.0
