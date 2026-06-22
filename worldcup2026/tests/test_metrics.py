"""Unit tests for the MS1 scoring + de-vig machinery."""

from __future__ import annotations

import numpy as np

from worldcup2026.eval import metrics
from worldcup2026.models.baseline import devig_normalize, devig_shin, overround


def test_rps_perfect_forecast_is_zero():
    p = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert metrics.rps(p, np.array([0, 2])) == 0.0


def test_rps_known_value():
    # Constantinou & Fenton worked example: forecast [.8,.1,.1], home occurs.
    # cum=[.8,.9,1], target cum=[1,1,1]; ((.8-1)^2+(.9-1)^2)/2 = (0.04+0.01)/2
    p = np.array([[0.8, 0.1, 0.1]])
    assert np.isclose(metrics.rps(p, np.array([0])), 0.025)


def test_rps_orders_matter():
    # a near-miss (predict draw, away happens) should beat a far-miss
    p = np.array([[0.1, 0.8, 0.1]])
    near = metrics.rps(p, np.array([2]))  # away
    far = metrics.rps(np.array([[0.8, 0.1, 0.1]]), np.array([2]))  # home predicted, away happens
    assert near < far


def test_brier_and_logloss_sane():
    p = np.array([[0.7, 0.2, 0.1]])
    assert metrics.brier(p, np.array([0])) > 0
    assert metrics.log_loss(p, np.array([0])) > 0


def test_devig_sums_to_one_and_reduces_margin():
    oh, od, oa = np.array([2.0]), np.array([3.5]), np.array([4.0])
    ov = overround(oh, od, oa)[0]
    assert ov > 1.0  # bookmaker margin present
    p = devig_normalize(oh, od, oa)
    assert np.isclose(p.sum(), 1.0)
    # de-vigged favourite prob below the raw implied (margin removed)
    assert p[0, 0] < 1.0 / 2.0


def test_devig_shin_valid_distribution():
    oh = np.array([1.5, 2.0])
    od = np.array([4.0, 3.3])
    oa = np.array([7.0, 3.6])
    p = devig_shin(oh, od, oa)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all() and (p <= 1).all()
