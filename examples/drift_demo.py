"""Synthetic demo showing the drift warning fire mid-run.

Simulates a job where the first 20 rows are cheap (~$0.012/iter) but the
remaining 180 rows are ~4x pricier, so the sample's projection underestimates
the real total. costscope's drift check catches this at iter 40.

    python examples/drift_demo.py
"""

import random

from costscope import CostEstimator


def main():
    random.seed(7)
    costs = [random.gauss(0.012, 0.002) for _ in range(20)]
    costs += [random.gauss(0.048, 0.008) for _ in range(180)]

    with CostEstimator(
        model="claude-opus-4-7",
        total_iterations=200,
        sample_iterations=20,
        auto_confirm=True,
        drift_check_every=20,
    ) as ce:
        for c in costs:
            ce.record(cost=max(c, 0.0001), elapsed=1.0)

    print(f"\nActual total:        ${ce.actual_total_cost:.2f}")
    print(f"Sample projection:   ${ce.estimate.total_estimate:.2f}")
    print(f"Drift detected:      {ce.drift_detected}")


if __name__ == "__main__":
    main()
