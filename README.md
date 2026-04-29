# costscope

Real-time reasoning-token cost estimation for batched LLM jobs.

Sample a handful of calls, project the total cost with a confidence interval, confirm before spending the rest. Works with [litellm](https://github.com/BerriAI/litellm) for real API calls or a built-in synthetic backend for tests and demos.

## Install

```bash
pip install -e .             # core
pip install -e '.[litellm]'  # with litellm backend
pip install -e '.[dev]'      # with pytest
```

Requires Python 3.10+.

## Usage

```python
from costscope import CostEstimator

with CostEstimator(model="o1", total_calls=500, sample_size=20) as ce:
    for prompt in prompts:
        response = ce.completion(messages=[{"role": "user", "content": prompt}])
        ...
```

The first 20 calls are billed normally and used to build a per-call cost distribution. After that, you'll see a 95% confidence interval over the projected total and a `Proceed? [y/N]` prompt — decline and subsequent `.completion()` calls raise `EstimationCancelled`.

### Skip the prompt

- `auto_confirm=True` — always proceed
- `threshold_usd=10.0` — auto-proceed when the upper bound is under the threshold
- `confirm_fn=...` — supply your own confirmation callback

### Synthetic mode

For tests, demos, and dev loops where real API calls would cost money:

```python
from costscope import CostEstimator, SyntheticConfig

cfg = SyntheticConfig(input_median=800, output_median=300, reasoning_median=2000, seed=42)

with CostEstimator(model="o1", total_calls=500, synthetic=True, synthetic_config=cfg) as ce:
    ...
```

See `examples/basic.py` for a full runnable example.

## Supported models (built-in pricing)

OpenAI o-series (`o1`, `o3`, `o3-mini`, ...), GPT-4o, Claude 4.x (Opus, Sonnet, Haiku). For other models, supply prices via `SyntheticConfig` or use the litellm-backed mode.

## Tests

```bash
pytest
```
