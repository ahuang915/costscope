import math
import random
from dataclasses import dataclass, field
from typing import Any

from .pricing import builtin_cost


@dataclass
class SyntheticConfig:
    """Distribution parameters for fake responses.

    Token counts are drawn log-normal: median sets center, sigma sets spread.
    sigma ~0.3 gives tight clusters; ~0.8 gives heavy-tailed, highly variable
    counts characteristic of reasoning models on heterogeneous tasks.
    Set reasoning_median=0 to disable reasoning tokens (non-thinking models).
    """

    input_median: int = 500
    input_sigma: float = 0.3
    output_median: int = 200
    output_sigma: float = 0.4
    reasoning_median: int = 1500
    reasoning_sigma: float = 0.7
    seed: int | None = None
    custom_prices: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class SyntheticResponse:
    """Litellm-shaped response stub. Exposes .usage with the fields we need."""

    model: str
    usage: "SyntheticUsage"


@dataclass
class SyntheticUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int


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
        return SyntheticResponse(
            model=model,
            usage=SyntheticUsage(
                prompt_tokens=prompt,
                completion_tokens=output + reasoning,
                total_tokens=prompt + output + reasoning,
                reasoning_tokens=reasoning,
            ),
        )

    def cost(self, response: SyntheticResponse) -> float:
        if response.model in self.config.custom_prices:
            in_price, out_price = self.config.custom_prices[response.model]
            return (
                response.usage.prompt_tokens / 1_000_000 * in_price
                + response.usage.completion_tokens / 1_000_000 * out_price
            )
        return builtin_cost(
            response.model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

    def _lognorm(self, median: float, sigma: float) -> int:
        if median <= 0:
            return 0
        mu = math.log(median)
        return max(1, int(self._rng.lognormvariate(mu, sigma)))
