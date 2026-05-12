---
name: costscope
description: Wrap batched LLM jobs (loops calling OpenAI/Anthropic over many prompts) with `CostEstimator` from the `costscope` package so the user sees a confidence-interval estimate after a small sample of calls and can confirm before spending the rest. TRIGGER when about to write or modify code that loops over prompts/rows/items calling chat completions (`for x in items: client.chat.completions.create(...)` or similar with Anthropic), or when the user says things like "run this over N rows", "batch process", "process these prompts", "estimate cost before running". SKIP for one-shot calls, agentic loops with unpredictable branching, or jobs already gated by another budget mechanism.
---

# costscope

`costscope` samples a handful of LLM calls, projects total cost with a confidence interval, and prompts the user to confirm before running the rest. It's a context manager that wraps an existing batched job.

## When to apply

Suggest `CostEstimator` when the planned/edited code makes ≥ ~50 LLM calls in a loop with non-trivial per-call cost (reasoning models, large prompts, or just unknown cost shape). Don't push it for:

- Single calls or small fixed counts (< ~20).
- Agentic loops where call count is unpredictable.
- Jobs already wrapped in another budget gate.

## Install

```bash
pip install costscope
pip install openai anthropic   # only the SDK(s) the job uses
```

## Standard pattern

`CostEstimator.completion(**kwargs)` dispatches to OpenAI or Anthropic based on `model` and returns the SDK's native response. The user keeps using their response handling code unchanged.

```python
from costscope import CostEstimator

with CostEstimator(
    model="claude-opus-4-7",
    total_calls=len(prompts),
    sample_size=20,
    confidence=0.95,
) as ce:
    for prompt in prompts:
        response = ce.completion(messages=[{"role": "user", "content": prompt}])
        # ...handle response as usual...
```

After `sample_size` calls, the user sees a formatted CI block and a `Proceed? [y/N]` prompt. On decline, further `.completion()` calls raise `EstimationCancelled`.

## Knobs to surface when relevant

- `synthetic=True, synthetic_config=SyntheticConfig(...)` — dry-run with fake usage drawn from log-normal distributions. No API spend. Use to validate wiring before a real run.
- `auto_confirm=True` — skip the prompt, always proceed. For CI / non-interactive scripts.
- `auto_confirm=False` — skip the prompt, always decline. Rarely what's wanted; do not confuse with `None`.
- `auto_confirm=None` (default) — actually prompt the user.
- `threshold_usd=N` — auto-proceed if the CI's upper bound is ≤ N. Cleaner than `auto_confirm=True` when a budget is known.
- `confirm_fn=callable` — custom confirm. Required when running on infra where `input()` doesn't see local stdin (e.g. Modal — call `modal.interact()` first, then `input()`).

## Gotchas to flag

- `sample_size` must be ≥ 2 (CI needs variance). Defaults to 20.
- Instantiate the context manager *outside* the loop, not inside.
- `model` must be one of the built-in priced models (o-series, GPT-4o, Claude 4.x). For others, pass explicit prices via `SyntheticConfig.custom_prices` (synthetic mode) or extend `pricing.py`.
- The estimator only knows what it has seen — early sample skew (e.g. first 20 prompts shorter than the rest) widens the CI but won't catch a bimodal distribution. Suggest a larger `sample_size` if the prompt set is heterogeneous.
