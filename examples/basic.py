"""Synthetic demo — runs end-to-end without spending a cent.

    python examples/basic.py
"""

from costscope import CostEstimator, SyntheticConfig


def main():
    prompts = [f"prompt {i}" for i in range(500)]

    cfg = SyntheticConfig(
        input_median=800,
        output_median=300,
        reasoning_median=2000,
        reasoning_sigma=0.8,
        seed=2026,
    )

    with CostEstimator(
        model="o1",
        total_calls=len(prompts),
        sample_size=20,
        confidence=0.95,
        synthetic=True,
        synthetic_config=cfg,
    ) as ce:
        for prompt in prompts:
            ce.completion(messages=[{"role": "user", "content": prompt}])

    print(f"\nDone. Actual total: ${ce.actual_total_cost:.2f}")
    print(f"Projected was:   ${ce.estimate.total_estimate:.2f} "
          f"({int(ce.estimate.confidence*100)}% CI ±${ce.estimate.margin:.2f})")


if __name__ == "__main__":
    main()
