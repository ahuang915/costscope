"""Prompt-cache effectiveness verdict for a sampled batch.

`compute_cache_verdict` rolls up the sample's cache token counts into a single
struct that the confirmation box can render.

Anthropic and OpenAI need different framings:
- Anthropic charges a write premium (1.25× / 2.0×) and offers ~90% read discount,
  so the meaningful metric is `cache_read / cache_write` and a break-even threshold.
- OpenAI writes are free; the meaningful metric is `cached / prompt` hit rate. Any
  hit rate above 0 is a win.

For OpenAI we also detect when `prompt_cache_key` would likely help. The signal:
a low hit rate combined with a *stable* per-call prompt size — that points at
backend routing (calls landing on different machines, each with its own cache)
rather than prefix variability, which is what the cache_key fixes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .pricing import (
    DEFAULT_ANTHROPIC_CACHE_READ_MULT,
    DEFAULT_ANTHROPIC_CACHE_WRITE_1H_MULT,
    DEFAULT_ANTHROPIC_CACHE_WRITE_5M_MULT,
    cost_with_cache,
    lookup_cache_prices,
    lookup_prices,
)


# Break-even reads-per-write derived from the documented Anthropic multipliers.
# write_premium / read_savings = (W - 1) / (1 - R)
BREAKEVEN_5M = (DEFAULT_ANTHROPIC_CACHE_WRITE_5M_MULT - 1.0) / (1.0 - DEFAULT_ANTHROPIC_CACHE_READ_MULT)
BREAKEVEN_1H = (DEFAULT_ANTHROPIC_CACHE_WRITE_1H_MULT - 1.0) / (1.0 - DEFAULT_ANTHROPIC_CACHE_READ_MULT)

# Heuristic thresholds for the prompt_cache_key suggestion (OpenAI).
# Hit rate below this AND stable prompt-token CV → likely routing-related miss.
PROMPT_CACHE_KEY_HIT_RATE_MAX = 0.50
PROMPT_CACHE_KEY_TOKEN_CV_MAX = 0.20  # ≤ 20% relative spread in prompt tokens


@dataclass
class CacheVerdict:
    """Sample-derived cache effectiveness summary.

    Always populated:
      provider, cache_read_tokens, cache_write_tokens, input_uncached_tokens,
      output_tokens.

    Anthropic-only:
      read_write_ratio, pays_off_5m, pays_off_1h,
      projected_actual / no_cache / 1h_ttl  (full-job dollars).

    OpenAI-only:
      hit_rate (cache_read / (cache_read + input_uncached)).
    """
    provider: str
    cache_read_tokens: int
    cache_write_tokens: int
    input_uncached_tokens: int
    output_tokens: int

    # Anthropic
    read_write_ratio: Optional[float] = None
    breakeven_5m: float = BREAKEVEN_5M
    breakeven_1h: float = BREAKEVEN_1H
    pays_off_5m: Optional[bool] = None
    pays_off_1h: Optional[bool] = None
    projected_actual: Optional[float] = None
    projected_no_cache: Optional[float] = None
    projected_1h_ttl: Optional[float] = None

    # OpenAI
    hit_rate: Optional[float] = None
    suggest_prompt_cache_key: bool = False
    prompt_token_cv: Optional[float] = None  # coefficient of variation across sample

    @property
    def has_cache_activity(self) -> bool:
        return self.cache_read_tokens > 0 or self.cache_write_tokens > 0


def _provider_for(model: str) -> str:
    return "anthropic" if model.lower().startswith("claude") else "openai"


def compute_cache_verdict(
    model: str,
    *,
    sample_cache_read_tokens: int,
    sample_cache_write_tokens: int,
    sample_input_uncached_tokens: int,
    sample_output_tokens: int,
    sample_size: int,
    total_iterations: int,
    # Optional per-call data used by the OpenAI prompt_cache_key heuristic.
    sample_prompt_token_counts: Optional[list[int]] = None,
    sample_openai_calls_without_key: int = 0,
    sample_openai_calls: int = 0,
) -> Optional[CacheVerdict]:
    """Build a verdict from the sample's accumulated token counts.

    Returns None when the sample saw zero cache activity AND no prompt input —
    nothing meaningful to say. For Anthropic with prompts-but-no-cache, returns
    a verdict with `has_cache_activity=False` so the UI can suggest enabling it.
    """
    if (sample_input_uncached_tokens + sample_cache_read_tokens + sample_cache_write_tokens) == 0:
        return None

    provider = _provider_for(model)
    verdict = CacheVerdict(
        provider=provider,
        cache_read_tokens=sample_cache_read_tokens,
        cache_write_tokens=sample_cache_write_tokens,
        input_uncached_tokens=sample_input_uncached_tokens,
        output_tokens=sample_output_tokens,
    )

    if provider == "anthropic":
        _fill_anthropic(verdict, model, sample_size, total_iterations)
    else:
        _fill_openai(
            verdict,
            sample_prompt_token_counts=sample_prompt_token_counts,
            sample_openai_calls_without_key=sample_openai_calls_without_key,
            sample_openai_calls=sample_openai_calls,
        )
    return verdict


def _fill_anthropic(v: CacheVerdict, model: str, sample_size: int, total_iterations: int) -> None:
    if v.cache_write_tokens > 0:
        v.read_write_ratio = v.cache_read_tokens / v.cache_write_tokens
        v.pays_off_5m = v.read_write_ratio >= v.breakeven_5m
        v.pays_off_1h = v.read_write_ratio >= v.breakeven_1h

    if sample_size <= 0:
        return
    scale = total_iterations / sample_size

    # Actual = what the sample actually paid, scaled to the full job.
    actual_sample = cost_with_cache(
        model,
        input_uncached_tokens=v.input_uncached_tokens,
        output_tokens=v.output_tokens,
        cache_read_tokens=v.cache_read_tokens,
        cache_write_tokens=v.cache_write_tokens,
    )
    v.projected_actual = actual_sample * scale

    # No-cache counterfactual: every cached byte gets re-billed at the plain input rate.
    # (The bytes existed and had to be tokenized either way; we just remove the discount/penalty.)
    base = lookup_prices(model)
    in_price = base[0] if base else 0.0
    out_price = base[1] if base else 0.0
    all_prompt = v.input_uncached_tokens + v.cache_read_tokens + v.cache_write_tokens
    no_cache_sample = (all_prompt / 1_000_000) * in_price + (v.output_tokens / 1_000_000) * out_price
    v.projected_no_cache = no_cache_sample * scale

    # 1h-TTL counterfactual: same token counts, but every write is billed at the 1h rate.
    # Reads stay at the 5min read rate (LiteLLM exposes only one read rate per model).
    # Note: this is a pricing swap, not a hit-rate boost. To estimate hit-rate boosts you
    # need timing data, which the sample doesn't carry — we surface the conservative
    # "what would I pay if I flipped the flag and behavior didn't change" number.
    read_per_1m, _w5m_per_1m, w1h_per_1m = lookup_cache_prices(model)
    one_h_sample = (
        (v.input_uncached_tokens / 1_000_000) * in_price
        + (v.output_tokens / 1_000_000) * out_price
        + (v.cache_read_tokens / 1_000_000) * read_per_1m
        + (v.cache_write_tokens / 1_000_000) * w1h_per_1m
    )
    v.projected_1h_ttl = one_h_sample * scale


def _fill_openai(
    v: CacheVerdict,
    *,
    sample_prompt_token_counts: Optional[list[int]] = None,
    sample_openai_calls_without_key: int = 0,
    sample_openai_calls: int = 0,
) -> None:
    prompt_total = v.input_uncached_tokens + v.cache_read_tokens
    if prompt_total > 0:
        v.hit_rate = v.cache_read_tokens / prompt_total

    # prompt_cache_key suggestion heuristic — only fires when ALL these hold:
    #  1. Hit rate is below threshold (otherwise caching already works)
    #  2. No call in the sample passed prompt_cache_key (else the user already tried it)
    #  3. Prompt-token CV is low (stable prefix → low hit rate is likely routing,
    #     not content variability that the key can't fix)
    if (sample_prompt_token_counts and len(sample_prompt_token_counts) >= 2
            and sample_openai_calls > 0):
        v.prompt_token_cv = _coefficient_of_variation(sample_prompt_token_counts)
        all_without_key = sample_openai_calls_without_key == sample_openai_calls
        if (v.hit_rate is not None
                and v.hit_rate < PROMPT_CACHE_KEY_HIT_RATE_MAX
                and all_without_key
                and v.prompt_token_cv is not None
                and v.prompt_token_cv <= PROMPT_CACHE_KEY_TOKEN_CV_MAX):
            v.suggest_prompt_cache_key = True


def _coefficient_of_variation(xs: list[int]) -> Optional[float]:
    """Sample CV (stdev / mean). None for empty/zero-mean lists."""
    if not xs:
        return None
    mean = sum(xs) / len(xs)
    if mean <= 0:
        return None
    if len(xs) < 2:
        return 0.0
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var) / mean
