import math

import pytest

from costscope import (
    CostEstimator,
    EstimationCancelled,
    SyntheticConfig,
    compute_estimate,
)
from costscope.pricing import builtin_cost


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
        total_iterations=200,
        sample_iterations=20,
        synthetic=True,
        synthetic_config=cfg,
        auto_confirm=True,
    ) as ce:
        for _ in range(200):
            ce.completion(messages=[{"role": "user", "content": "hi"}])

    assert ce.iterations_done == 200
    assert ce.estimate is not None
    assert ce.estimate.sample_size == 20
    assert ce.actual_total_cost > 0


def test_synthetic_run_cancels():
    cfg = SyntheticConfig(seed=7)
    estimator = CostEstimator(
        model="o1-mini",
        total_iterations=100,
        sample_iterations=10,
        synthetic=True,
        synthetic_config=cfg,
        auto_confirm=False,
    )
    with pytest.raises(EstimationCancelled):
        with estimator as ce:
            for _ in range(100):
                ce.completion(messages=[{"role": "user", "content": "hi"}])

    assert estimator.iterations_done == 10
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
        total_iterations=50,
        sample_iterations=5,
        synthetic=True,
        synthetic_config=cfg,
        threshold_usd=1.00,
        auto_confirm=False,
    ) as ce:
        for _ in range(50):
            ce.completion(messages=[{"role": "user", "content": "hi"}])

    assert ce.iterations_done == 50


def test_threshold_blocks_when_estimate_too_high():
    cfg = SyntheticConfig(
        seed=1,
        input_median=2000, input_sigma=0.3,
        output_median=500, output_sigma=0.3,
        reasoning_median=8000, reasoning_sigma=0.5,
    )
    estimator = CostEstimator(
        model="o1",
        total_iterations=10000,
        sample_iterations=10,
        synthetic=True,
        synthetic_config=cfg,
        threshold_usd=1.00,
        auto_confirm=False,
    )
    with pytest.raises(EstimationCancelled):
        with estimator as ce:
            for _ in range(10000):
                ce.completion(messages=[{"role": "user", "content": "hi"}])

    assert estimator.iterations_done == 10
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


def test_iteration_groups_multiple_calls():
    """3 calls per iteration → per-iteration cost is ~3× per-call cost."""
    cfg = SyntheticConfig(seed=11)
    with CostEstimator(
        model="o1-mini",
        total_iterations=30,
        sample_iterations=10,
        synthetic=True,
        synthetic_config=cfg,
        auto_confirm=True,
    ) as ce:
        for _ in range(30):
            with ce.iteration():
                ce.completion(messages=[{"role": "user", "content": "a"}])
                ce.completion(messages=[{"role": "user", "content": "b"}])
                ce.completion(messages=[{"role": "user", "content": "c"}])

    assert ce.iterations_done == 30
    assert ce.calls_made == 90  # 30 iterations × 3 calls
    assert ce.avg_calls_per_iteration == 3.0
    # Sample average should reflect 3 calls' worth of cost per iteration
    sample_costs = ce._sample_costs
    assert len(sample_costs) == 10
    assert all(c > 0 for c in sample_costs)


def test_iteration_back_compat_without_context():
    """Without iteration(), each call is its own iteration (0.1.x behavior)."""
    cfg = SyntheticConfig(seed=3)
    with CostEstimator(
        model="o1-mini",
        total_calls=50,  # back-compat alias
        sample_size=10,  # back-compat alias
        synthetic=True,
        synthetic_config=cfg,
        auto_confirm=True,
    ) as ce:
        for _ in range(50):
            ce.completion(messages=[{"role": "user", "content": "x"}])

    assert ce.iterations_done == 50
    assert ce.calls_made == 50  # alias still works


def test_time_estimate_present_with_latency():
    cfg = SyntheticConfig(seed=42, latency_median=0.5, latency_sigma=0.2)
    with CostEstimator(
        model="o1-mini",
        total_iterations=100,
        sample_iterations=10,
        synthetic=True,
        synthetic_config=cfg,
        auto_confirm=True,
    ) as ce:
        for _ in range(100):
            ce.completion(messages=[{"role": "user", "content": "hi"}])

    assert ce.time_estimate is not None
    assert ce.time_estimate.total_estimate > 0


def test_drift_warns_when_post_sample_costs_jump(capsys):
    """If post-sample iterations are much pricier than the sample, drift fires."""
    with CostEstimator(
        model="o1-mini",
        total_iterations=60,
        sample_iterations=10,
        synthetic=True,
        synthetic_config=SyntheticConfig(seed=1),
        auto_confirm=True,
        drift_check_every=10,
    ) as ce:
        for _ in range(10):
            ce.record(cost=0.01)  # tight sample → narrow CI
        for _ in range(20):
            ce.record(cost=0.10)  # 10x jump in post-sample

    err = capsys.readouterr().err
    assert "drift" in err.lower()
    assert "above" in err
    assert ce.drift_detected


def test_drift_quiet_when_costs_stay_within_band(capsys):
    """No warning when post-sample iterations stay near the sample mean."""
    with CostEstimator(
        model="o1-mini",
        total_iterations=60,
        sample_iterations=10,
        synthetic=True,
        synthetic_config=SyntheticConfig(seed=2),
        auto_confirm=True,
        drift_check_every=10,
    ) as ce:
        for _ in range(10):
            ce.record(cost=0.01)
        for _ in range(20):
            ce.record(cost=0.01)  # same as sample

    err = capsys.readouterr().err
    assert "drift" not in err.lower()
    assert not ce.drift_detected


def test_drift_disabled_with_zero(capsys):
    """drift_check_every=0 disables the check entirely."""
    with CostEstimator(
        model="o1-mini",
        total_iterations=60,
        sample_iterations=10,
        synthetic=True,
        synthetic_config=SyntheticConfig(seed=3),
        auto_confirm=True,
        drift_check_every=0,
    ) as ce:
        for _ in range(10):
            ce.record(cost=0.01)
        for _ in range(20):
            ce.record(cost=0.10)

    err = capsys.readouterr().err
    assert "drift" not in err.lower()
    assert not ce.drift_detected


def test_record_escape_hatch():
    """ce.record(cost, time) lets the caller drive the SDK themselves."""
    with CostEstimator(
        model="o1-mini",
        total_iterations=10,
        sample_iterations=5,
        synthetic=True,  # synthetic so we don't need an SDK
        synthetic_config=SyntheticConfig(seed=0),
        auto_confirm=True,
    ) as ce:
        for _ in range(10):
            ce.record(cost=0.01, elapsed=0.5)

    assert ce.iterations_done == 10
    assert ce.actual_total_cost == pytest.approx(0.10)
    assert ce.actual_total_time == pytest.approx(5.0)


def test_record_inside_iteration_accumulates():
    with CostEstimator(
        model="o1-mini",
        total_iterations=4,
        sample_iterations=2,
        synthetic=True,
        synthetic_config=SyntheticConfig(seed=0),
        auto_confirm=True,
    ) as ce:
        for _ in range(4):
            with ce.iteration():
                ce.record(cost=0.01, elapsed=0.1)
                ce.record(cost=0.02, elapsed=0.2)

    assert ce.iterations_done == 4
    # 4 iterations × 0.03 cost = 0.12; × 0.3s = 1.2s
    assert ce.actual_total_cost == pytest.approx(0.12)
    assert ce.actual_total_time == pytest.approx(1.2)


def test_image_token_pricing():
    """gpt-image-1: 1M image-output tokens at $40 → $40."""
    cost = builtin_cost("gpt-image-1", prompt_tokens=0, completion_tokens=0, image_output_tokens=1_000_000)
    assert cost == pytest.approx(40.00)
    # Text-only model ignores image-output tokens
    cost = builtin_cost("o1-mini", prompt_tokens=0, completion_tokens=0, image_output_tokens=1_000_000)
    assert cost == 0.0


def test_api_auto_routes_gpt_image():
    ce = CostEstimator(
        model="gpt-image-1",
        total_iterations=5,
        sample_iterations=2,
        synthetic=True,
        synthetic_config=SyntheticConfig(seed=0),
        auto_confirm=True,
    )
    assert ce.api == "responses"


def test_api_auto_keeps_chat_for_classic_models():
    ce = CostEstimator(
        model="o1-mini",
        total_iterations=5,
        sample_iterations=2,
        synthetic=True,
        synthetic_config=SyntheticConfig(seed=0),
        auto_confirm=True,
    )
    assert ce.api == "chat"


def test_on_cancel_invoked_when_user_saves(monkeypatch):
    """User declines run, then says yes to save: callback fires with the estimator."""
    captured = {}

    def save(ce):
        captured["estimate_total"] = ce.estimate.total_estimate
        captured["actual_spent"] = ce.actual_total_cost
        captured["iterations"] = ce.iterations_done

    # confirm_fn returns False → declined; input() returns "y" → save
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")

    estimator = CostEstimator(
        model="o1-mini",
        total_iterations=100,
        sample_iterations=10,
        synthetic=True,
        synthetic_config=SyntheticConfig(seed=4),
        confirm_fn=lambda est, model: False,
        on_cancel=save,
    )
    with pytest.raises(EstimationCancelled):
        with estimator as ce:
            for _ in range(100):
                ce.completion(messages=[{"role": "user", "content": "hi"}])

    assert captured["iterations"] == 10
    assert captured["estimate_total"] > 0
    assert captured["actual_spent"] > 0


def test_on_cancel_skipped_when_user_declines_save(monkeypatch):
    """User declines run, then says no to save: callback does NOT fire."""
    calls = []

    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")

    estimator = CostEstimator(
        model="o1-mini",
        total_iterations=100,
        sample_iterations=10,
        synthetic=True,
        synthetic_config=SyntheticConfig(seed=5),
        confirm_fn=lambda est, model: False,
        on_cancel=lambda ce: calls.append(ce),
    )
    with pytest.raises(EstimationCancelled):
        with estimator as ce:
            for _ in range(100):
                ce.completion(messages=[{"role": "user", "content": "hi"}])

    assert calls == []


def test_on_cancel_not_called_without_callback(monkeypatch):
    """No on_cancel registered: no save prompt is shown."""
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return ""

    monkeypatch.setattr("builtins.input", fake_input)

    estimator = CostEstimator(
        model="o1-mini",
        total_iterations=100,
        sample_iterations=10,
        synthetic=True,
        synthetic_config=SyntheticConfig(seed=6),
        confirm_fn=lambda est, model: False,  # no input() call from confirm
    )
    with pytest.raises(EstimationCancelled):
        with estimator as ce:
            for _ in range(100):
                ce.completion(messages=[{"role": "user", "content": "hi"}])

    # confirm_fn was provided, so the only input() would have been the save prompt
    assert prompts == []


def test_rejects_both_total_aliases():
    with pytest.raises(ValueError, match="only one"):
        CostEstimator(
            model="o1-mini",
            total_iterations=10,
            total_calls=10,
            synthetic=True,
            auto_confirm=True,
        )
