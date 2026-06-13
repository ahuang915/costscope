"""Synthetic demo: a mid-run cache regression triggers both drift detectors.

Simulates a 200-iteration batch where the sample window (first 20) sees a
healthy 85% cache hit rate. At iter 20, imagine an upstream change introduced
a per-row variable header that invalidated the cache prefix — hit rate collapses
to 5%. Two things happen:

  1. Cost drift fires because cache misses pay the 1.25× write premium instead
     of the 0.10× read rate.
  2. Cache drift fires *in addition*, with a diagnostic message that points at
     the cache as the cause — the unique value-add over plain cost drift.

In a real workload where the cost shift might be masked by another factor
(shorter outputs, model switch), the cache-drift signal is what surfaces the
regime change.

    python examples/cache_drift_demo.py
"""

from costscope import CostEstimator, SyntheticConfig


def main():
    cfg = SyntheticConfig(
        input_median=500,
        output_median=200,
        reasoning_median=0,
        cache_prefix_median=8_000,     # ~8k-token cached prefix per call
        cache_hit_probability=0.85,    # sample window: mostly hits
        seed=11,
    )

    with CostEstimator(
        model="claude-opus-4-7",
        total_iterations=200,
        sample_iterations=20,
        synthetic=True,
        synthetic_config=cfg,
        auto_confirm=True,
        drift_check_every=20,
        cache_drift_threshold=0.05,   # default; lower → more sensitive
    ) as ce:
        for i in range(200):
            if i == 20:
                # Simulate the upstream change that broke the cache prefix.
                # The synthetic backend keeps emitting cache tokens, but now most
                # calls miss → cache_write tokens dominate.
                ce._backend.config.cache_hit_probability = 0.05
            ce.completion(messages=[{"role": "user", "content": "demo row"}])

    print(f"\nActual total:        ${ce.actual_total_cost:.2f}")
    print(f"Sample projection:   ${ce.estimate.total_estimate:.2f}")
    print(f"Cost drift detected:  {ce.drift_detected}")
    print(f"Cache drift detected: {ce.cache_drift_detected}")


if __name__ == "__main__":
    main()
