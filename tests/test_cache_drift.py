"""Tests for the cache-aware drift detection added to CostEstimator.

We feed the estimator a controlled stream of CallTelemetry by monkeypatching
`_dispatch`, so each test asserts deterministic drift behavior independent of
the synthetic backend's distributions.
"""
import time

import pytest

from costscope import CostEstimator
from costscope import litellm_prices
from costscope.estimator import CallTelemetry


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Force the offline LiteLLM path so tests don't touch the network."""
    monkeypatch.setenv("COSTSCOPE_OFFLINE", "1")
    monkeypatch.setattr(litellm_prices, "_memo", None)


def _make_dispatcher(stream: list[CallTelemetry]):
    """Returns a _dispatch replacement that pops successive telemetries from `stream`.

    The shape mirrors the real _dispatch: takes `self, kwargs`, returns the
    pre-filled CallTelemetry (cost / elapsed already populated).
    """
    iterator = iter(stream)

    def _dispatch(self, kwargs):
        return next(iterator)

    return _dispatch


def _tel(*, cost: float, in_uncached: int = 100, out: int = 100,
         read: int = 0, write: int = 0) -> CallTelemetry:
    return CallTelemetry(
        response=None, cost=cost, elapsed=0.0,
        input_uncached_tokens=in_uncached, output_tokens=out,
        cache_read_tokens=read, cache_write_tokens=write,
    )


def _run_with_stream(stream: list[CallTelemetry], *, model="claude-opus-4-7",
                     sample=4, total=8, **kw) -> CostEstimator:
    """Drive an estimator through a controlled telemetry stream."""
    ce = CostEstimator(
        model=model,
        total_iterations=total,
        sample_iterations=sample,
        synthetic=False,           # we're patching _dispatch directly
        auto_confirm=True,
        drift_check_every=kw.pop("drift_check_every", 2),
        **kw,
    )
    ce._dispatch = _make_dispatcher(stream).__get__(ce, CostEstimator)
    with ce:
        for _ in range(total):
            ce.completion(messages=[{"role": "user", "content": "x"}])
    return ce


def test_stable_cache_does_not_drift():
    """Same cache mix in sample and exec → cache_drift_detected stays False."""
    # 0.5 ratio reads/writes in both windows.
    stream = [_tel(cost=0.10, read=50, write=100)] * 8
    ce = _run_with_stream(stream)
    assert ce.cache_drift_detected is False
    assert ce.drift_detected is False  # cost is also flat


def test_cache_drift_fires_when_hit_rate_collapses():
    """Sample is hit-heavy; exec is miss-heavy. Cache drift fires."""
    # Sample: lots of reads, tiny writes → ratio ~10
    sample = [_tel(cost=0.05, read=1000, write=100)] * 4
    # Exec: no reads, all writes → ratio 0
    exec_ = [_tel(cost=0.20, read=0, write=1000)] * 4
    ce = _run_with_stream(sample + exec_)
    assert ce.cache_drift_detected is True


def test_cache_drift_respects_threshold():
    """Same regression, but a permissive threshold suppresses the warning."""
    sample = [_tel(cost=0.05, read=1000, write=100)] * 4
    exec_ = [_tel(cost=0.20, read=0, write=1000)] * 4
    # Threshold 10× the savings swing → silenced.
    ce = _run_with_stream(sample + exec_, cache_drift_threshold=100.0)
    assert ce.cache_drift_detected is False


def test_no_cache_activity_means_no_cache_drift():
    """If neither sample nor exec used the cache, the detector stays silent."""
    stream = [_tel(cost=0.10, in_uncached=500, out=200, read=0, write=0)] * 8
    ce = _run_with_stream(stream)
    assert ce.cache_drift_detected is False


def test_cost_drift_message_includes_cache_note(monkeypatch, capsys):
    """Existing cost-drift warning gains a 'cache: ratio X → Y' diagnostic line."""
    # Cheap sample with healthy ratio, then pricey exec with collapsed ratio.
    sample = [_tel(cost=0.01, read=1000, write=100)] * 4
    exec_ = [_tel(cost=0.20, read=0, write=1000)] * 4
    ce = _run_with_stream(sample + exec_)
    captured = capsys.readouterr()
    # Both warnings should have appeared on stderr.
    assert "drift at iter" in captured.err
    assert "cache: read/write ratio" in captured.err


def test_openai_hit_rate_drift():
    """OpenAI path: hit rate falls; cache-drift uses hit-rate framing."""
    # Set up gpt-4o pricing via LiteLLM mock so the cost math works.
    sample = [_tel(cost=0.005, in_uncached=200, read=800, write=0)] * 4
    exec_ = [_tel(cost=0.020, in_uncached=1000, read=0, write=0)] * 4
    ce = _run_with_stream(sample + exec_, model="gpt-4o")
    assert ce.cache_drift_detected is True


def test_cache_drift_re_arms_when_recovery_happens():
    """Detector re-arms once cumulative running savings returns within threshold.

    The exec window averages cumulatively (matches cost-drift behavior), so a
    short bad burst needs a proportionally longer good run to wash it out.
    """
    sample = [_tel(cost=0.05, read=1000, write=100)] * 4
    bad = [_tel(cost=0.20, read=0, write=1000)] * 4         # fires
    good = [_tel(cost=0.05, read=1000, write=100)] * 12     # dominates the average
    ce = _run_with_stream(sample + bad + good, total=20, drift_check_every=2)
    # After enough good iters, cumulative savings swings back inside the threshold band.
    assert ce.cache_drift_detected is False


def test_cache_drift_threshold_validation():
    """Negative thresholds are rejected at construction time."""
    with pytest.raises(ValueError, match="cache_drift_threshold"):
        CostEstimator(model="claude-opus-4-7", total_iterations=10,
                      sample_iterations=2, cache_drift_threshold=-0.1)
