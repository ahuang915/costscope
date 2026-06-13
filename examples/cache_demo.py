"""Synthetic demo of the prompt-cache verdict in the estimate box.

Runs three back-to-back batches against the synthetic backend so each provider
branch in `cache_stats.compute_cache_verdict` renders:

  1. Anthropic with a strong hit rate     → "✓ paying off" verdict + counterfactuals
  2. Anthropic with a weak hit rate       → "✗ losing money" verdict
  3. OpenAI with cached_tokens reported   → hit-rate framing, no break-even

No real API calls; nothing to spend.

    python examples/cache_demo.py
"""

from costscope import CostEstimator, SyntheticConfig


def _run(label: str, *, model: str, cfg: SyntheticConfig, iterations: int = 100,
         sample: int = 20) -> None:
    print(f"\n=== {label} ===")
    with CostEstimator(
        model=model,
        total_iterations=iterations,
        sample_iterations=sample,
        synthetic=True,
        synthetic_config=cfg,
        auto_confirm=True,
    ) as ce:
        for _ in range(iterations):
            ce.completion(messages=[{"role": "user", "content": "demo"}])


def main():
    # 1. Anthropic, healthy cache: big prefix + high hit probability. The verdict
    # reads "✓ paying off" and the counterfactual row shows no-cache costing more.
    _run(
        "Anthropic — cache paying off",
        model="claude-opus-4-7",
        cfg=SyntheticConfig(
            input_median=400, output_median=200, reasoning_median=0,
            cache_prefix_median=20_000,   # ~20k-token cached prefix per call
            cache_hit_probability=0.85,   # 85% of calls land on a hot cache
            seed=42,
        ),
    )

    # 2. Same shape, but the prefix is small and hits are rare → ratio falls below
    # the 0.28 break-even, verdict reads "✗ losing money".
    _run(
        "Anthropic — cache losing money",
        model="claude-opus-4-7",
        cfg=SyntheticConfig(
            input_median=400, output_median=200, reasoning_median=0,
            cache_prefix_median=8_000,
            cache_hit_probability=0.10,   # most calls miss and pay the write premium
            seed=7,
        ),
    )

    # 3. OpenAI: writes are free, so the headline shifts to a hit-rate percentage.
    # Any hit rate > 0 is a strict win; no break-even threshold to clear.
    _run(
        "OpenAI — hit rate framing",
        model="gpt-4o",
        cfg=SyntheticConfig(
            input_median=2_000, output_median=200, reasoning_median=0,
            cache_prefix_median=1_500,
            cache_hit_probability=0.70,
            seed=2026,
        ),
    )


if __name__ == "__main__":
    main()
