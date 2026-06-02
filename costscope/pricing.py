"""Per-1M-token pricing, with LiteLLM auto-refresh and a hand-maintained fallback.

Lookups consult LiteLLM's hosted price database first (cached locally for ~7
days) and fall back to `_BUILTIN_PRICES` when LiteLLM does not know the model
or the network is unavailable. Set COSTSCOPE_OFFLINE=1 to skip LiteLLM entirely.

Reasoning tokens are billed as output tokens for both OpenAI o-series and
Anthropic extended-thinking models. The third field, when non-zero, prices
image-output tokens separately (e.g. gpt-image-1).
"""

from typing import Optional

from . import litellm_prices

# (input_per_1m, output_per_1m, image_output_per_1m)
_BUILTIN_PRICES: dict[str, tuple[float, float, float]] = {
    "o1": (15.00, 60.00, 0.0),
    "o1-mini": (3.00, 12.00, 0.0),
    "o1-preview": (15.00, 60.00, 0.0),
    "o3": (10.00, 40.00, 0.0),
    "o3-mini": (1.10, 4.40, 0.0),
    "claude-opus-4-7": (5.00, 25.00, 0.0),
    "claude-opus-4-5": (5.00, 25.00, 0.0),
    "claude-sonnet-4-6": (3.00, 15.00, 0.0),
    "claude-sonnet-4-5": (3.00, 15.00, 0.0),
    "claude-haiku-4-5": (1.00, 5.00, 0.0),
    "gpt-4o": (2.50, 10.00, 0.0),
    "gpt-4o-mini": (0.15, 0.60, 0.0),
    "gpt-5": (5.00, 15.00, 0.0),
    "gpt-image-1": (5.00, 0.0, 40.00),
}


def lookup_prices(model: str) -> Optional[tuple[float, float, float]]:
    """Per-1M-token (input, output, image_output) prices for `model`, or None.

    Tries LiteLLM's hosted price database first (cached for ~7 days), then
    falls back to the built-in table. Use the env var COSTSCOPE_OFFLINE=1 to
    skip the LiteLLM lookup entirely. Image-output pricing comes from the
    built-in table — LiteLLM does not currently expose it in a uniform field.
    """
    remote = litellm_prices.lookup(model)
    key = _resolve(model)
    builtin = _BUILTIN_PRICES[key] if key else None
    if remote is not None:
        # Keep the local image-output price if we have one — LiteLLM lacks it.
        img = builtin[2] if builtin else 0.0
        return (remote[0], remote[1], img)
    return builtin


def builtin_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    image_output_tokens: int = 0,
) -> float:
    prices = lookup_prices(model)
    if prices is None:
        raise KeyError(
            f"No price found for model '{model}'. "
            f"Known built-ins: {sorted(_BUILTIN_PRICES)}. "
            f"Pass synthetic_config with explicit prices, extend pricing.py, "
            f"or check connectivity to LiteLLM's price database."
        )
    in_price, out_price, img_price = prices
    return (
        (prompt_tokens / 1_000_000) * in_price
        + (completion_tokens / 1_000_000) * out_price
        + (image_output_tokens / 1_000_000) * img_price
    )


def _resolve(model: str) -> str | None:
    if model in _BUILTIN_PRICES:
        return model
    lowered = model.lower()
    for key in _BUILTIN_PRICES:
        if key in lowered:
            return key
    return None
