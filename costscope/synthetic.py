import math
import random
from dataclasses import dataclass, field
from typing import Any

from .pricing import builtin_cost, cost_with_cache


@dataclass
class SyntheticConfig:
    """Distribution parameters for fake responses.

    Token counts are drawn log-normal: median sets center, sigma sets spread.
    sigma ~0.3 gives tight clusters; ~0.8 gives heavy-tailed, highly variable
    counts characteristic of reasoning models on heterogeneous tasks.
    Set reasoning_median=0 to disable reasoning tokens (non-thinking models).
    Set image_output_median>0 to simulate image-generation models (gpt-image-1).
    Set latency_median>0 to simulate per-call wall time for time estimates.
    """

    input_median: int = 500
    input_sigma: float = 0.3
    output_median: int = 200
    output_sigma: float = 0.4
    reasoning_median: int = 1500
    reasoning_sigma: float = 0.7
    image_output_median: int = 0
    image_output_sigma: float = 0.4
    latency_median: float = 0.0  # seconds; 0 disables simulated latency
    latency_sigma: float = 0.3
    # Cache-token simulation. Default 0 → no cache activity. Set median>0 to drive
    # cache_stats branches in tests / demos. The split between read and write per call
    # is controlled by `cache_hit_probability` (read on a hit, write on a miss).
    cache_prefix_median: int = 0
    cache_prefix_sigma: float = 0.3
    cache_hit_probability: float = 0.5
    seed: int | None = None
    custom_prices: dict[str, tuple[float, float] | tuple[float, float, float]] = field(default_factory=dict)


@dataclass
class SyntheticResponse:
    model: str
    usage: "SyntheticUsage"
    latency_s: float


@dataclass
class SyntheticUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int
    image_output_tokens: int = 0
    # Cache fields — populated when SyntheticConfig.cache_prefix_median > 0. Otherwise
    # left at 0 and treated as "no cache activity" by the estimator.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    input_uncached_tokens: int = 0


class SyntheticBackend:
    def __init__(self, config: SyntheticConfig):
        self.config = config
        self._rng = random.Random(config.seed)

    def completion(self, model: str, **_kwargs: Any) -> SyntheticResponse:
        cfg = self.config
        prompt = self._lognorm(cfg.input_median, cfg.input_sigma)
        output = self._lognorm(cfg.output_median, cfg.output_sigma)
        reasoning = (
            self._lognorm(cfg.reasoning_median, cfg.reasoning_sigma)
            if cfg.reasoning_median > 0 else 0
        )
        image_out = (
            self._lognorm(cfg.image_output_median, cfg.image_output_sigma)
            if cfg.image_output_median > 0 else 0
        )
        latency = (
            self._lognorm_float(cfg.latency_median, cfg.latency_sigma)
            if cfg.latency_median > 0 else 0.0
        )

        cache_read = 0
        cache_write = 0
        if cfg.cache_prefix_median > 0:
            prefix = self._lognorm(cfg.cache_prefix_median, cfg.cache_prefix_sigma)
            if self._rng.random() < cfg.cache_hit_probability:
                cache_read = prefix
            else:
                cache_write = prefix

        return SyntheticResponse(
            model=model,
            usage=SyntheticUsage(
                prompt_tokens=prompt,
                completion_tokens=output + reasoning,
                total_tokens=prompt + output + reasoning + image_out,
                reasoning_tokens=reasoning,
                image_output_tokens=image_out,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                input_uncached_tokens=prompt,
            ),
            latency_s=latency,
        )

    def cost(self, response: SyntheticResponse) -> float:
        usage = response.usage
        custom = self.config.custom_prices.get(response.model)
        if custom is not None:
            in_price = custom[0]
            out_price = custom[1]
            img_price = custom[2] if len(custom) > 2 else 0.0
            return (
                usage.prompt_tokens / 1_000_000 * in_price
                + usage.completion_tokens / 1_000_000 * out_price
                + usage.image_output_tokens / 1_000_000 * img_price
            )
        # No cache activity → keep the cheap pricing path. With cache tokens, defer
        # to cost_with_cache so the synthetic backend matches real-API pricing rules.
        if usage.cache_read_tokens or usage.cache_write_tokens:
            return cost_with_cache(
                response.model,
                input_uncached_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                cache_write_tokens=usage.cache_write_tokens,
                image_output_tokens=usage.image_output_tokens,
            )
        return builtin_cost(
            response.model,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.image_output_tokens,
        )

    def _lognorm(self, median: float, sigma: float) -> int:
        if median <= 0:
            return 0
        mu = math.log(median)
        return max(1, int(self._rng.lognormvariate(mu, sigma)))

    def _lognorm_float(self, median: float, sigma: float) -> float:
        if median <= 0:
            return 0.0
        mu = math.log(median)
        return max(0.0, self._rng.lognormvariate(mu, sigma))
