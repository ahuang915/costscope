"""Built-in pricing fallback for synthetic mode and when litellm is unavailable.

Prices are USD per 1M tokens. Reasoning tokens are billed as output tokens
for both OpenAI o-series and Anthropic extended-thinking models.
"""

_BUILTIN_PRICES = {
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o1-preview": (15.00, 60.00),
    "o3": (10.00, 40.00),
    "o3-mini": (1.10, 4.40),
    "claude-opus-4-7": (15.00, 75.00),
    "claude-opus-4-5": (15.00, 75.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}


def builtin_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    key = _resolve(model)
    if key is None:
        raise KeyError(
            f"No built-in price for model '{model}'. "
            f"Known: {sorted(_BUILTIN_PRICES)}. "
            f"Pass synthetic_config with explicit prices, or use litellm-backed mode."
        )
    in_price, out_price = _BUILTIN_PRICES[key]
    return (prompt_tokens / 1_000_000) * in_price + (completion_tokens / 1_000_000) * out_price


def _resolve(model: str) -> str | None:
    if model in _BUILTIN_PRICES:
        return model
    lowered = model.lower()
    for key in _BUILTIN_PRICES:
        if key in lowered:
            return key
    return None
