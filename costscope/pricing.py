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


# Default multipliers (vs. plain input rate) when LiteLLM doesn't have the field.
# Anthropic's documented numbers: 5-min write = 1.25×, 1-hour write = 2.0×, read = 0.10×.
# OpenAI varies by model; default to 0.5× read (gpt-4o family) when nothing is in LiteLLM.
DEFAULT_ANTHROPIC_CACHE_READ_MULT = 0.10
DEFAULT_ANTHROPIC_CACHE_WRITE_5M_MULT = 1.25
DEFAULT_ANTHROPIC_CACHE_WRITE_1H_MULT = 2.00
DEFAULT_OPENAI_CACHE_READ_MULT = 0.50


def _is_anthropic_model(model: str) -> bool:
    return model.lower().startswith("claude")


def lookup_cache_prices(model: str) -> tuple[float, float, float]:
    """Per-1M (cache_read, cache_write_5min, cache_write_1h) prices for `model`.

    Tries LiteLLM first (it carries `cache_read_input_token_cost`,
    `cache_creation_input_token_cost`, and `cache_creation_input_token_cost_above_1hr`
    for many models). For any field LiteLLM omits, falls back to:
      - Anthropic: 0.10× / 1.25× / 2.0× of the plain input rate (documented multipliers)
      - OpenAI:    0.50× / 0× / 0×    (writes are free; default read discount for gpt-4o)

    Raises KeyError if the model has no plain input price at all (same behavior as
    `builtin_cost`), since cache prices are expressed relative to it.
    """
    base = lookup_prices(model)
    if base is None:
        raise KeyError(
            f"No price found for model '{model}'. Cache prices need the plain input "
            f"rate as a fallback. Known built-ins: {sorted(_BUILTIN_PRICES)}."
        )
    input_per_1m = base[0]

    is_anth = _is_anthropic_model(model)
    if is_anth:
        fb_read = input_per_1m * DEFAULT_ANTHROPIC_CACHE_READ_MULT
        fb_w5m = input_per_1m * DEFAULT_ANTHROPIC_CACHE_WRITE_5M_MULT
        fb_w1h = input_per_1m * DEFAULT_ANTHROPIC_CACHE_WRITE_1H_MULT
    else:
        fb_read = input_per_1m * DEFAULT_OPENAI_CACHE_READ_MULT
        fb_w5m = 0.0
        fb_w1h = 0.0

    cache = litellm_prices.lookup_cache_prices(model) if not litellm_prices.offline() else None
    if cache is None:
        return (fb_read, fb_w5m, fb_w1h)
    read, w5m, w1h = cache
    return (
        read if read is not None else fb_read,
        w5m if w5m is not None else fb_w5m,
        w1h if w1h is not None else fb_w1h,
    )


def cost_with_cache(
    model: str,
    *,
    input_uncached_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    image_output_tokens: int = 0,
) -> float:
    """Cost of one call, billing each token category at its correct rate.

    `input_uncached_tokens` is the portion of the prompt that wasn't served from
    cache (and isn't being written to cache). For Anthropic, this matches the
    `input_tokens` field on the usage object directly. For OpenAI, it's
    `prompt_tokens - cached_tokens`.

    Cache reads and writes are billed at the rates from `lookup_cache_prices`.
    """
    base = lookup_prices(model)
    if base is None:
        raise KeyError(f"No price found for model '{model}'.")
    in_price, out_price, img_price = base
    cache_read_price, cache_write_price, _w1h = lookup_cache_prices(model)
    return (
        (input_uncached_tokens / 1_000_000) * in_price
        + (output_tokens / 1_000_000) * out_price
        + (image_output_tokens / 1_000_000) * img_price
        + (cache_read_tokens / 1_000_000) * cache_read_price
        + (cache_write_tokens / 1_000_000) * cache_write_price
    )


def _resolve(model: str) -> str | None:
    if model in _BUILTIN_PRICES:
        return model
    lowered = model.lower()
    for key in _BUILTIN_PRICES:
        if key in lowered:
            return key
    return None
