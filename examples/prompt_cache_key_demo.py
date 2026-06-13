"""Synthetic demo of the prompt_cache_key suggestion heuristic.

OpenAI prompt caching needs your requests to land on the same backend machine
for a cached prefix to be reused — each machine has its own in-memory cache.
With many parallel callers, default prefix-hash routing can spread one user's
traffic across many backends, each tokenizing the prefix from scratch.
`prompt_cache_key` is a routing hint that pins together calls sharing the same
key, dramatically raising the hit rate.

This demo runs two scenarios back to back:

  1. Naive: no `prompt_cache_key`. Hit rate is low; prompt sizes are stable.
     The cache-verdict heuristic fires its "try prompt_cache_key" suggestion.
  2. Same workload with `prompt_cache_key="user-42"` passed on every call.
     Hit rate jumps; the suggestion no longer appears.

We simulate the routing-fix effect by raising the synthetic backend's hit
probability in scenario 2 — in real life OpenAI's load balancer would do
the analogous thing.

    python examples/prompt_cache_key_demo.py
"""

from costscope import CostEstimator, SyntheticConfig


def _run(label: str, *, hit_probability: float, pass_key: bool) -> None:
    print(f"\n=== {label} ===")
    cfg = SyntheticConfig(
        input_median=2_000,
        input_sigma=0.05,                  # tight CV → suggests routing is the cause
        output_median=200,
        reasoning_median=0,
        cache_prefix_median=600,
        cache_hit_probability=hit_probability,
        seed=42,
    )
    with CostEstimator(
        model="gpt-4o",
        total_iterations=100,
        sample_iterations=20,
        synthetic=True,
        synthetic_config=cfg,
        auto_confirm=True,
    ) as ce:
        for _ in range(100):
            kwargs = {"messages": [{"role": "user", "content": "demo row"}]}
            if pass_key:
                kwargs["prompt_cache_key"] = "user-42"
            ce.completion(**kwargs)


def main():
    # Scenario 1: naive caller. Suggestion should fire.
    _run("Without prompt_cache_key", hit_probability=0.20, pass_key=False)
    # Scenario 2: same workload, key passed. Routing pins requests to one backend;
    # the simulated hit rate jumps from 20% to 80% and the suggestion is suppressed.
    _run("With prompt_cache_key='user-42'", hit_probability=0.80, pass_key=True)


if __name__ == "__main__":
    main()
