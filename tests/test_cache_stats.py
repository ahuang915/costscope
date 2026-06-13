"""Tests for compute_cache_verdict and the cache-aware pricing layer.

No network: monkeypatch litellm_prices so the tests run offline and deterministic.
"""
import pytest

from costscope import litellm_prices
from costscope.cache_stats import BREAKEVEN_5M, BREAKEVEN_1H, compute_cache_verdict
from costscope.pricing import cost_with_cache, lookup_cache_prices


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    # Force builtin pricing for these tests so behavior doesn't drift if LiteLLM updates.
    monkeypatch.setenv("COSTSCOPE_OFFLINE", "1")
    monkeypatch.setattr(litellm_prices, "_memo", None)


def test_anthropic_paying_off_5min():
    """A sample with reads ≥ 0.28× writes is flagged as paying off at 5min TTL."""
    v = compute_cache_verdict(
        "claude-opus-4-7",
        sample_cache_read_tokens=400,
        sample_cache_write_tokens=1000,   # ratio 0.40 > 0.278
        sample_input_uncached_tokens=200,
        sample_output_tokens=100,
        sample_size=10,
        total_iterations=100,
    )
    assert v is not None
    assert v.provider == "anthropic"
    assert v.read_write_ratio == pytest.approx(0.40)
    assert v.pays_off_5m is True
    assert v.pays_off_1h is False  # 0.40 < 1.11


def test_anthropic_losing_money():
    """Ratio below 0.28 → 5min cache is net negative."""
    v = compute_cache_verdict(
        "claude-opus-4-7",
        sample_cache_read_tokens=100,
        sample_cache_write_tokens=1000,   # ratio 0.10
        sample_input_uncached_tokens=200,
        sample_output_tokens=100,
        sample_size=10,
        total_iterations=100,
    )
    assert v.pays_off_5m is False
    assert v.pays_off_1h is False
    # No-cache counterfactual should be cheaper than actual when ratio is this bad.
    assert v.projected_no_cache < v.projected_actual


def test_anthropic_no_cache_activity():
    """When no markers are set, returns a verdict with has_cache_activity=False."""
    v = compute_cache_verdict(
        "claude-opus-4-7",
        sample_cache_read_tokens=0,
        sample_cache_write_tokens=0,
        sample_input_uncached_tokens=5000,
        sample_output_tokens=1000,
        sample_size=10,
        total_iterations=100,
    )
    assert v is not None
    assert v.has_cache_activity is False
    assert v.read_write_ratio is None


def test_returns_none_with_zero_input():
    """Empty sample → nothing meaningful to report."""
    v = compute_cache_verdict(
        "claude-opus-4-7",
        sample_cache_read_tokens=0,
        sample_cache_write_tokens=0,
        sample_input_uncached_tokens=0,
        sample_output_tokens=0,
        sample_size=10,
        total_iterations=100,
    )
    assert v is None


def test_openai_hit_rate():
    """OpenAI verdicts surface hit rate, not a break-even ratio."""
    v = compute_cache_verdict(
        "gpt-4o",
        sample_cache_read_tokens=600,
        sample_cache_write_tokens=0,      # OpenAI writes are free / always 0
        sample_input_uncached_tokens=400,
        sample_output_tokens=200,
        sample_size=10,
        total_iterations=100,
    )
    assert v is not None
    assert v.provider == "openai"
    assert v.hit_rate == pytest.approx(0.60)
    assert v.read_write_ratio is None  # Anthropic-only field


def test_breakeven_constants_match_documented_multipliers():
    """The break-even ratios drop out of the published multipliers."""
    # 5-min: write premium 0.25, read savings 0.90  → 0.25/0.90 ≈ 0.278
    assert BREAKEVEN_5M == pytest.approx(0.25 / 0.90, abs=1e-4)
    # 1-hour: write premium 1.00, read savings 0.90 → 1.00/0.90 ≈ 1.111
    assert BREAKEVEN_1H == pytest.approx(1.00 / 0.90, abs=1e-4)


def test_anthropic_cache_pricing_via_lookup(monkeypatch):
    """Anthropic LiteLLM cache fields drive lookup_cache_prices; otherwise multiplier fallback."""
    monkeypatch.delenv("COSTSCOPE_OFFLINE", raising=False)
    monkeypatch.setattr(litellm_prices, "_memo", None)
    monkeypatch.setattr(litellm_prices, "lookup", lambda model: (5.0, 25.0, 0.0))
    monkeypatch.setattr(
        litellm_prices,
        "lookup_cache_prices",
        lambda model: (0.5, 6.25, 10.0),  # explicit read / w5m / w1h per 1M
    )
    read, w5m, w1h = lookup_cache_prices("claude-opus-4-7")
    assert (read, w5m, w1h) == (0.5, 6.25, 10.0)


def test_openai_write_default_zero(monkeypatch):
    """When LiteLLM omits write fields (OpenAI case), fallback gives write_5m = 0."""
    monkeypatch.delenv("COSTSCOPE_OFFLINE", raising=False)
    monkeypatch.setattr(litellm_prices, "_memo", None)
    monkeypatch.setattr(litellm_prices, "lookup", lambda model: (2.50, 10.00, 0.0))
    monkeypatch.setattr(litellm_prices, "lookup_cache_prices", lambda model: (1.25, None, None))
    read, w5m, w1h = lookup_cache_prices("gpt-4o")
    assert read == 1.25
    assert w5m == 0.0
    assert w1h == 0.0


def test_openai_suggests_prompt_cache_key_when_routing_looks_at_fault():
    """Low hit rate + no key passed + stable prompt size → suggestion fires."""
    v = compute_cache_verdict(
        "gpt-4o",
        sample_cache_read_tokens=200,
        sample_cache_write_tokens=0,
        sample_input_uncached_tokens=800,      # hit rate = 200/1000 = 20%
        sample_output_tokens=200,
        sample_size=10,
        total_iterations=100,
        sample_prompt_token_counts=[1000, 1020, 980, 1010, 1005, 990, 1015, 995, 1000, 985],
        sample_openai_calls_without_key=10,
        sample_openai_calls=10,
    )
    assert v.suggest_prompt_cache_key is True
    assert v.prompt_token_cv is not None and v.prompt_token_cv < 0.05


def test_no_suggestion_when_user_already_passed_key():
    """If the user passed prompt_cache_key on any sample call, we don't second-guess."""
    v = compute_cache_verdict(
        "gpt-4o",
        sample_cache_read_tokens=200,
        sample_cache_write_tokens=0,
        sample_input_uncached_tokens=800,
        sample_output_tokens=200,
        sample_size=10,
        total_iterations=100,
        sample_prompt_token_counts=[1000] * 10,
        sample_openai_calls_without_key=3,    # 7 of 10 passed the key
        sample_openai_calls=10,
    )
    assert v.suggest_prompt_cache_key is False


def test_no_suggestion_when_prompt_sizes_vary_widely():
    """High prompt-token CV → low hit rate is likely content variability, not routing."""
    v = compute_cache_verdict(
        "gpt-4o",
        sample_cache_read_tokens=200,
        sample_cache_write_tokens=0,
        sample_input_uncached_tokens=800,
        sample_output_tokens=200,
        sample_size=10,
        total_iterations=100,
        sample_prompt_token_counts=[500, 1500, 800, 200, 1200, 400, 1800, 600, 1100, 900],
        sample_openai_calls_without_key=10,
        sample_openai_calls=10,
    )
    assert v.suggest_prompt_cache_key is False
    assert v.prompt_token_cv is not None and v.prompt_token_cv > 0.2


def test_no_suggestion_when_hit_rate_is_already_healthy():
    """If caching already works (≥50% hits), we don't suggest the key."""
    v = compute_cache_verdict(
        "gpt-4o",
        sample_cache_read_tokens=700,
        sample_cache_write_tokens=0,
        sample_input_uncached_tokens=300,    # hit rate = 70%
        sample_output_tokens=200,
        sample_size=10,
        total_iterations=100,
        sample_prompt_token_counts=[1000] * 10,
        sample_openai_calls_without_key=10,
        sample_openai_calls=10,
    )
    assert v.suggest_prompt_cache_key is False


def test_suggestion_off_for_anthropic_models():
    """Anthropic doesn't have prompt_cache_key; the field stays False even with low ratio."""
    v = compute_cache_verdict(
        "claude-opus-4-7",
        sample_cache_read_tokens=100,
        sample_cache_write_tokens=1000,
        sample_input_uncached_tokens=500,
        sample_output_tokens=200,
        sample_size=10,
        total_iterations=100,
        sample_prompt_token_counts=[1000] * 10,
        sample_openai_calls_without_key=10,
        sample_openai_calls=10,
    )
    assert v.suggest_prompt_cache_key is False


def test_cost_with_cache_bills_all_categories(monkeypatch):
    """cost_with_cache adds up input_uncached + output + cache_read + cache_write."""
    monkeypatch.delenv("COSTSCOPE_OFFLINE", raising=False)
    monkeypatch.setattr(litellm_prices, "_memo", None)
    monkeypatch.setattr(litellm_prices, "lookup", lambda model: (5.0, 25.0, 0.0))
    monkeypatch.setattr(litellm_prices, "lookup_cache_prices", lambda model: (0.5, 6.25, 10.0))
    cost = cost_with_cache(
        "claude-opus-4-7",
        input_uncached_tokens=1_000_000,    # $5
        output_tokens=1_000_000,            # $25
        cache_read_tokens=1_000_000,        # $0.50
        cache_write_tokens=1_000_000,       # $6.25
    )
    assert cost == pytest.approx(5.0 + 25.0 + 0.5 + 6.25)
