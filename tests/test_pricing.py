"""Tests for the LiteLLM-first / built-in-fallback price lookup.

These tests never hit the network — they monkeypatch the LiteLLM module to
inject controlled behavior.
"""

import pytest

from costscope import lookup_prices
from costscope import litellm_prices
from costscope.pricing import _BUILTIN_PRICES, builtin_cost


def test_offline_env_forces_builtin(monkeypatch):
    """COSTSCOPE_OFFLINE=1 must skip LiteLLM and return the built-in price."""
    monkeypatch.setenv("COSTSCOPE_OFFLINE", "1")
    monkeypatch.setattr(litellm_prices, "_memo", None)
    prices = lookup_prices("claude-opus-4-7")
    assert prices == _BUILTIN_PRICES["claude-opus-4-7"]


def test_litellm_overrides_builtin_when_present(monkeypatch):
    """When LiteLLM returns a price, it wins over the built-in table."""
    monkeypatch.delenv("COSTSCOPE_OFFLINE", raising=False)
    monkeypatch.setattr(litellm_prices, "lookup", lambda model: (7.0, 35.0, 0.0))
    prices = lookup_prices("claude-opus-4-7")
    assert prices == (7.0, 35.0, 0.0)


def test_litellm_preserves_builtin_image_price(monkeypatch):
    """LiteLLM does not expose image-output price; we keep the built-in one."""
    monkeypatch.delenv("COSTSCOPE_OFFLINE", raising=False)
    monkeypatch.setattr(litellm_prices, "lookup", lambda model: (4.0, 0.0, 0.0))
    prices = lookup_prices("gpt-image-1")
    assert prices is not None
    assert prices[2] == _BUILTIN_PRICES["gpt-image-1"][2]


def test_litellm_unknown_falls_back_to_builtin(monkeypatch):
    """If LiteLLM returns None, the built-in table answers."""
    monkeypatch.delenv("COSTSCOPE_OFFLINE", raising=False)
    monkeypatch.setattr(litellm_prices, "lookup", lambda model: None)
    prices = lookup_prices("o1")
    assert prices == _BUILTIN_PRICES["o1"]


def test_unknown_model_raises_through_builtin_cost(monkeypatch):
    monkeypatch.setenv("COSTSCOPE_OFFLINE", "1")
    monkeypatch.setattr(litellm_prices, "_memo", None)
    with pytest.raises(KeyError):
        builtin_cost("definitely-not-a-real-model", 100, 100)


def test_builtin_cost_uses_litellm_rate(monkeypatch):
    """End-to-end: the cost calc respects the LiteLLM-supplied rate."""
    monkeypatch.delenv("COSTSCOPE_OFFLINE", raising=False)
    monkeypatch.setattr(litellm_prices, "lookup", lambda model: (10.0, 50.0, 0.0))
    # 1M input @ $10 + 1M output @ $50 = $60
    assert builtin_cost("claude-opus-4-7", 1_000_000, 1_000_000) == pytest.approx(60.0)
