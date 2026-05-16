"""Synthetic demo — runs end-to-end without spending a cent.

Each iteration makes 2 LLM calls. We sample 20 iterations, then a 95% CI is
projected over 500 iterations of cost and (sequential) wall time.

    python examples/basic.py
"""

from costscope import CostEstimator, SyntheticConfig


def main():
    items = [f"item {i}" for i in range(500)]

    cfg = SyntheticConfig(
        input_median=800,
        output_median=300,
        reasoning_median=2000,
        reasoning_sigma=0.8,
        latency_median=1.2,       # simulate ~1.2s per call
        latency_sigma=0.4,
        seed=2026,
    )

    with CostEstimator(
        model="o1",
        total_iterations=len(items),
        sample_iterations=20,
        confidence=0.95,
        synthetic=True,
        synthetic_config=cfg,
        concurrency=4,            # we plan to run 4 iterations in parallel
        auto_confirm=True,        # for the demo; remove to get the interactive prompt
    ) as ce:
        for item in items:
            with ce.iteration():
                ce.completion(messages=[{"role": "user", "content": f"extract: {item}"}])
                ce.completion(messages=[{"role": "user", "content": f"summarize: {item}"}])

    print(f"\nDone. Actual total: ${ce.actual_total_cost:.2f}")
    print(
        f"Projected cost:  ${ce.estimate.total_estimate:.2f} "
        f"({int(ce.estimate.confidence*100)}% CI ±${ce.estimate.margin:.2f})"
    )
    print(f"Projected wall time: {ce.wall_time_estimate:.1f}s @ concurrency={ce.concurrency}")


if __name__ == "__main__":
    main()
