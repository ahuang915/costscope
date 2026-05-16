"""Built-in pricing fallback for synthetic mode and when no external pricing source is available.

Prices are USD per 1M tokens. Reasoning tokens are billed as output tokens
for both OpenAI o-series and Anthropic extended-thinking models. The third
field, when non-zero, prices image-output tokens separately (e.g. gpt-image-1).
"""

# (input_per_1m, output_per_1m, image_output_per_1m)
_BUILTIN_PRICES: dict[str, tuple[float, float, float]] = {
    "o1": (15.00, 60.00, 0.0),
    "o1-mini": (3.00, 12.00, 0.0),
    "o1-preview": (15.00, 60.00, 0.0),
    "o3": (10.00, 40.00, 0.0),
    "o3-mini": (1.10, 4.40, 0.0),
    "claude-opus-4-7": (15.00, 75.00, 0.0),
    "claude-opus-4-5": (15.00, 75.00, 0.0),
    "claude-sonnet-4-6": (3.00, 15.00, 0.0),
    "claude-sonnet-4-5": (3.00, 15.00, 0.0),
    "claude-haiku-4-5": (1.00, 5.00, 0.0),
    "gpt-4o": (2.50, 10.00, 0.0),
    "gpt-4o-mini": (0.15, 0.60, 0.0),
    "gpt-5": (5.00, 15.00, 0.0),
    "gpt-image-1": (5.00, 0.0, 40.00),
}


def builtin_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    image_output_tokens: int = 0,
) -> float:
    key = _resolve(model)
    if key is None:
        raise KeyError(
            f"No built-in price for model '{model}'. "
            f"Known: {sorted(_BUILTIN_PRICES)}. "
            f"Pass synthetic_config with explicit prices."
        )
    in_price, out_price, img_price = _BUILTIN_PRICES[key]
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
