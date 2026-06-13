"""Synthetic demo: mid-run the cache *starts* paying off — both detectors fire.

Mirror of cache_drift_demo.py. In that demo the sample looks healthy and the
cache regresses mid-run. Here the sample looks bad (mostly misses, paying the
1.25× write premium for nothing) and mid-run the cache turns into a win.

What you'll see:
  - Sample box: ratio well below 0.28 break-even, verdict "✗ losing money",
    counterfactual row shows no-cache would have been cheaper.
  - At iter 40, cost-drift fires *below* the CI band because the per-iter cost
    dropped (reads are 1/12.5 the price of writes).
  - Cache-drift fires in the "more savings" direction: caching has flipped
    from net-negative value to net-positive value.

Real-world analogue: you were burning the write premium because some variable
header at the top of every prompt was breaking the cache prefix; mid-run you
fixed the prompt builder and the prefix started hitting cleanly.

    python examples/cache_recovery_demo.py
"""

from costscope import CostEstimator, SyntheticConfig


def main():
    cfg = SyntheticConfig(
        input_median=500,
        output_median=200,
        reasoning_median=0,
        cache_prefix_median=8_000,
        cache_hit_probability=0.10,    # sample window: mostly misses, paying writes
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
        cache_drift_threshold=0.05,
    ) as ce:
        for i in range(200):
            if i == 20:
                # Simulate the upstream fix that stabilizes the cache prefix.
                # Same prefix size, just hits instead of misses now.
                ce._backend.config.cache_hit_probability = 0.85
            ce.completion(messages=[{"role": "user", "content": "demo row"}])

    print(f"\nActual total:        ${ce.actual_total_cost:.2f}")
    print(f"Sample projection:   ${ce.estimate.total_estimate:.2f}  "
          f"(overshoots — the cache got cheaper post-sample)")
    print(f"Cost drift detected:  {ce.drift_detected}")
    print(f"Cache drift detected: {ce.cache_drift_detected}")


if __name__ == "__main__":
    main()
