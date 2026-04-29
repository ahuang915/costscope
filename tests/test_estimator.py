import math

import pytest

from costscope import (
    CostEstimator,
    EstimationCancelled,
    SyntheticConfig,
    compute_estimate,
)


def test_compute_estimate_basic():
    costs = [0.10, 0.12, 0.08, 0.11, 0.09, 0.13, 0.10, 0.11, 0.09, 0.12]
    est = compute_estimate(costs, total_calls=1000, confidence=0.95)
    assert est.sample_size == 10
    assert math.isclose(est.mean_per_call, 0.105, abs_tol=1e-9)
    assert est.total_estimate == pytest.approx(105.0)
    assert est.lower < est.total_estimate < est.upper
    assert est.margin > 0


def test_compute_estimate_zero_variance():
    est = compute_estimate([0.5, 0.5, 0.5, 0.5], total_calls=100)
    assert est.stdev_per_call == 0.0
    assert est.margin == 0.0
    assert est.lower == est.upper == est.total_estimate == 50.0


def test_synthetic_run_proceeds():
    cfg = SyntheticConfig(seed=42)
    with CostEstimator(
        model="o1-mini",
        total_calls=200,
        sample_size=20,
        synthetic=True,
        synthetic_config=cfg,
        auto_confirm=True,
    ) as ce:
        for _ in range(200):
            ce.completion(messages=[{"role": "user", "content": "hi"}])

    assert ce.calls_made == 200
    assert ce.estimate is not None
    assert ce.estimate.sample_size == 20
    assert ce.actual_total_cost > 0


def test_synthetic_run_cancels():
    cfg = SyntheticConfig(seed=7)
    estimator = CostEstimator(
        model="o1-mini",
        total_calls=100,
        sample_size=10,
        synthetic=True,
        synthetic_config=cfg,
        auto_confirm=False,
    )
    with pytest.raises(EstimationCancelled):
        with estimator as ce:
            for _ in range(100):
                ce.completion(messages=[{"role": "user", "content": "hi"}])

    assert estimator.calls_made == 10
    assert estimator.estimate is not None


def test_threshold_auto_proceed():
    cfg = SyntheticConfig(
        seed=1,
        input_median=10, input_sigma=0.1,
        output_median=10, output_sigma=0.1,
        reasoning_median=0,
    )
    with CostEstimator(
        model="o1-mini",
        total_calls=50,
        sample_size=5,
        synthetic=True,
        synthetic_config=cfg,
        threshold_usd=1.00,
        auto_confirm=False,
    ) as ce:
        for _ in range(50):
            ce.completion(messages=[{"role": "user", "content": "hi"}])

    assert ce.calls_made == 50


def test_threshold_blocks_when_estimate_too_high():
    cfg = SyntheticConfig(
        seed=1,
        input_median=2000, input_sigma=0.3,
        output_median=500, output_sigma=0.3,
        reasoning_median=8000, reasoning_sigma=0.5,
    )
    estimator = CostEstimator(
        model="o1",
        total_calls=10000,
        sample_size=10,
        synthetic=True,
        synthetic_config=cfg,
        threshold_usd=1.00,
        auto_confirm=False,
    )
    with pytest.raises(EstimationCancelled):
        with estimator as ce:
            for _ in range(10000):
                ce.completion(messages=[{"role": "user", "content": "hi"}])

    assert estimator.calls_made == 10
    assert estimator.estimate.upper > 1.00


def test_estimate_margin_shrinks_with_larger_sample():
    costs = [0.10 + 0.01 * (i % 3) for i in range(100)]
    small = compute_estimate(costs[:5], total_calls=1000)
    large = compute_estimate(costs[:50], total_calls=1000)
    assert large.margin < small.margin


def test_confidence_levels_ordered():
    costs = [0.10, 0.15, 0.08, 0.12, 0.11, 0.09, 0.13, 0.10, 0.14, 0.07]
    e90 = compute_estimate(costs, 1000, confidence=0.90)
    e95 = compute_estimate(costs, 1000, confidence=0.95)
    e99 = compute_estimate(costs, 1000, confidence=0.99)
    assert e90.margin < e95.margin < e99.margin
