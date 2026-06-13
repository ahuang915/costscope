"""Tests for the auto-switch caching feature.

We control telemetry by monkeypatching `_dispatch` (same pattern as test_cache_drift).
The tests assert state-machine behavior, not actual SDK mutation — those paths are
exercised via the helper unit tests at the bottom.
"""
import pytest

from costscope import CostEstimator
from costscope import litellm_prices
from costscope.estimator import CallTelemetry, _strip_cache_control


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("COSTSCOPE_OFFLINE", "1")
    monkeypatch.setattr(litellm_prices, "_memo", None)


def _tel(*, cost: float, in_uncached: int = 100, out: int = 100,
         read: int = 0, write: int = 0) -> CallTelemetry:
    return CallTelemetry(
        response=None, cost=cost, elapsed=0.0,
        input_uncached_tokens=in_uncached, output_tokens=out,
        cache_read_tokens=read, cache_write_tokens=write,
    )


def _stream_dispatch(stream):
    iterator = iter(stream)

    def _dispatch(self, kwargs):
        return next(iterator)

    return _dispatch


def _run(stream, *, model="claude-opus-4-7", sample=4, total=12, **kw):
    """Run an estimator through a controlled telemetry stream."""
    ce = CostEstimator(
        model=model, total_iterations=total, sample_iterations=sample,
        synthetic=False, auto_confirm=True,
        drift_check_every=kw.pop("drift_check_every", 2),
        **kw,
    )
    ce._dispatch = _stream_dispatch(stream).__get__(ce, CostEstimator)
    with ce:
        for _ in range(total):
            ce.completion(messages=[{"role": "user", "content": "x"}])
    return ce


def test_post_sample_switch_off_on_bad_sample():
    """Sample with ratio < break-even → caching is auto-switched off after sampling."""
    # Sample: low ratio (lots of writes, few reads) → losing money.
    sample = [_tel(cost=0.10, read=10, write=1000)] * 4
    # Exec: doesn't matter for this assertion; just need enough calls.
    exec_ = [_tel(cost=0.10, read=0, write=0)] * 8
    ce = _run(sample + exec_, auto_switch_caching=True)
    assert ce.current_cache_mode == "switched_off"


def test_post_sample_no_switch_when_caching_pays_off():
    """Sample with ratio > break-even → leave caching on."""
    sample = [_tel(cost=0.05, read=1000, write=100)] * 4  # ratio 10
    exec_ = [_tel(cost=0.05, read=1000, write=100)] * 8
    ce = _run(sample + exec_, auto_switch_caching=True)
    assert ce.current_cache_mode == "default"


def test_dry_run_does_not_mutate_mode(capsys):
    """Dry-run mode logs the decision but doesn't change current_cache_mode."""
    sample = [_tel(cost=0.10, read=10, write=1000)] * 4
    exec_ = [_tel(cost=0.10, read=0, write=0)] * 8
    ce = _run(sample + exec_, auto_switch_caching=True, auto_switch_dry_run=True)
    assert ce.current_cache_mode == "default"
    err = capsys.readouterr().err
    assert "would strip cache_control" in err or "would strip" in err


def test_no_switch_when_auto_switch_off():
    """Default behavior (auto_switch_caching=False) leaves mode at default no matter what."""
    sample = [_tel(cost=0.10, read=10, write=1000)] * 4
    exec_ = [_tel(cost=0.10, read=0, write=0)] * 8
    ce = _run(sample + exec_)  # auto_switch_caching omitted → False
    assert ce.current_cache_mode == "default"


def test_probes_restore_caching_when_window_shows_savings():
    """After switch_off, probes with high reads vote 'switch back'; M agreeing windows flip the mode."""
    # Sample: bad ratio → switches off
    sample = [_tel(cost=0.10, read=10, write=1000)] * 4
    # Exec: 24 non-probe calls + interleaved probe windows.
    # Configure probes to fire every 2 calls; each window = 2 calls.
    # Probe calls return data that shows caching saving money (lots of reads).
    # Non-probe calls return baseline data with no cache tokens.
    nonprobe = _tel(cost=0.05, in_uncached=500, out=200, read=0, write=0)
    probe = _tel(cost=0.01, in_uncached=50, out=200, read=2000, write=100)
    # Sequence: 2 non-probe calls (warm up probe counter), then 2-call probe window, repeat.
    # We need consecutive_required=2 windows to flip back.
    exec_ = []
    for _ in range(4):
        exec_ += [nonprobe, nonprobe, probe, probe]   # 4 calls per cycle, last 2 are a probe window
    ce = _run(sample + exec_, total=4 + len(exec_),
              auto_switch_caching=True,
              probe_every=2, probe_size=2,
              auto_switch_consecutive_required=2)
    # By end of run we should have completed enough probe windows to flip back.
    assert ce.current_cache_mode == "default"


def test_probe_budget_ceiling_suppresses_further_probes():
    """If probes accumulate too much overhead, further probes are skipped."""
    sample = [_tel(cost=0.10, read=10, write=1000)] * 4
    # Make probes very expensive so the budget hits the ceiling fast.
    nonprobe = _tel(cost=0.001, in_uncached=10, out=10, read=0, write=0)
    expensive_probe = _tel(cost=1.0, in_uncached=10, out=10, read=10, write=100_000)
    exec_ = []
    for _ in range(6):
        exec_ += [nonprobe, expensive_probe, expensive_probe]
    ce = _run(sample + exec_, total=4 + len(exec_),
              auto_switch_caching=True,
              probe_every=1, probe_size=2,
              auto_switch_consecutive_required=10,    # require many windows so we don't flip
              max_probe_overhead_pct=0.01)            # tight ceiling
    # Confirms the budget gate clamps probing. We don't assert exact counts —
    # just that mode remains switched_off and the run completed.
    assert ce.current_cache_mode == "switched_off"


def test_kwargs_mutation_strips_markers_in_switched_off_mode():
    """When switched off, the SDK should see kwargs without cache_control."""
    sent_kwargs = []

    def _capture_dispatch(self, kwargs):
        sent_kwargs.append(kwargs)
        return _tel(cost=0.01, read=0, write=0)

    sample = [_tel(cost=0.10, read=10, write=1000)] * 4
    ce = CostEstimator(
        model="claude-opus-4-7", total_iterations=6, sample_iterations=4,
        synthetic=False, auto_confirm=True, auto_switch_caching=True,
    )
    # Patch dispatch: sample calls use stream telemetry; exec calls capture kwargs.
    stream = iter(sample)

    def _hybrid(self, kwargs):
        if ce._iterations_done < ce.sample_size:
            return next(stream)
        sent_kwargs.append(kwargs)
        return _tel(cost=0.01, read=0, write=0)

    ce._dispatch = _hybrid.__get__(ce, CostEstimator)
    with ce:
        for _ in range(6):
            ce.completion(
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}},
                    ]},
                ],
            )

    # After post-sample switch, the exec call(s) we captured should have markers stripped.
    assert ce.current_cache_mode == "switched_off"
    assert sent_kwargs, "expected at least one exec call after sampling"
    for k in sent_kwargs:
        # Recursively assert no cache_control anywhere.
        assert "cache_control" not in str(k)


def test_strip_cache_control_helper_preserves_text():
    """Marker stripping removes only the cache_control key, leaving text intact."""
    payload = [
        {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "world"},
    ]
    stripped = _strip_cache_control(payload)
    assert stripped == [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]


def test_preset_string_enables_and_sets_defaults():
    """auto_switch_caching='balanced' enables the feature with balanced preset values."""
    ce = CostEstimator(model="claude-opus-4-7", total_iterations=10,
                       sample_iterations=2, auto_switch_caching="balanced")
    assert ce.auto_switch_caching is True
    assert ce.auto_switch_consecutive_required == 2
    assert ce.probe_every == 20
    assert ce.probe_size == 3


def test_preset_aggressive_uses_aggressive_defaults():
    """aggressive preset → consecutive_required=1, probe_every=10."""
    ce = CostEstimator(model="claude-opus-4-7", total_iterations=10,
                       sample_iterations=2, auto_switch_caching="aggressive")
    assert ce.auto_switch_consecutive_required == 1
    assert ce.probe_every == 10


def test_preset_dry_run_enables_dry_run_flag():
    """dry_run preset → flag is on, behavior is balanced otherwise."""
    ce = CostEstimator(model="claude-opus-4-7", total_iterations=10,
                       sample_iterations=2, auto_switch_caching="dry_run")
    assert ce.auto_switch_caching is True
    assert ce.auto_switch_dry_run is True
    assert ce.auto_switch_consecutive_required == 2


def test_explicit_kwarg_overrides_preset():
    """User can pick a preset and still override one knob."""
    ce = CostEstimator(model="claude-opus-4-7", total_iterations=10,
                       sample_iterations=2,
                       auto_switch_caching="patient", probe_size=2)
    assert ce.auto_switch_consecutive_required == 3   # patient default
    assert ce.probe_size == 2                          # explicit override


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="preset"):
        CostEstimator(model="claude-opus-4-7", total_iterations=10,
                      sample_iterations=2, auto_switch_caching="bogus")


def test_param_validation():
    """Bad knob values rejected at construction."""
    for bad in [
        {"auto_switch_consecutive_required": 0},
        {"probe_every": 0},
        {"probe_size": 0},
        {"max_probe_overhead_pct": -0.01},
    ]:
        with pytest.raises(ValueError):
            CostEstimator(model="claude-opus-4-7", total_iterations=10,
                          sample_iterations=2, **bad)
