"""Synthetic demo of auto_switch_caching — costscope actually flips caching on/off.

Shows the full lifecycle with the simplest possible invocation:

  1. Sample window has a poor read/write ratio (caching is wasting money).
  2. At end of sample, costscope auto-strips cache_control markers and logs
     the decision.
  3. During exec, periodic probe windows briefly re-enable caching to gather
     fresh evidence on whether the workload has shifted.
  4. At iter 20 the synthetic workload changes — caching would now help.
     After two consecutive probe windows agree, costscope restores the markers.

The whole thing is driven by one flag:

    CostEstimator(..., auto_switch_caching=True)

Other useful values:
    auto_switch_caching="aggressive"   # flip on first probe verdict
    auto_switch_caching="patient"      # require 3 windows of evidence
    auto_switch_caching="dry_run"      # log decisions, don't mutate kwargs

    python examples/auto_switch_demo.py
"""

from costscope import CostEstimator, SyntheticConfig


def main():
    # Sample phase: low hit probability → ratio below break-even → auto-switch off.
    cfg = SyntheticConfig(
        input_median=500, output_median=200, reasoning_median=0,
        cache_prefix_median=4_000,
        cache_hit_probability=0.05,    # sample: cache barely hits → losing money
        seed=11,
    )

    with CostEstimator(
        model="claude-opus-4-7",
        total_iterations=80,
        sample_iterations=20,
        synthetic=True,
        synthetic_config=cfg,
        auto_confirm=True,
        auto_switch_caching=True,      # one flag enables the whole feature
        # Demo: probe more often than default so the lifecycle is visible
        # in a small run. Real workloads typically use the preset defaults.
        probe_every=10, probe_size=3,
    ) as ce:
        for i in range(80):
            if i == 30:
                # Workload shifts mid-run — now the cache prefix is stable and
                # mostly hits. Probe windows will see this and vote 'restore'.
                ce._backend.config.cache_hit_probability = 0.90
            ce.completion(messages=[{"role": "user", "content": "demo row"}])

    print(f"\nFinal cache mode:    {ce.current_cache_mode}")
    print(f"Actual total:        ${ce.actual_total_cost:.2f}")
    print(f"Cost drift detected:  {ce.drift_detected}")
    print(f"Cache drift detected: {ce.cache_drift_detected}")


if __name__ == "__main__":
    main()
