# costscope

Cost + time estimation for batched LLM jobs.

Sample a handful of iterations, project the total cost **and** wall time with a confidence interval, confirm before spending the rest. Each "iteration" can be a single call or a multi-call pipeline. Works with OpenAI (chat completions + Responses API, including `gpt-image-1`), Anthropic, or a built-in synthetic backend for tests and demos.

## Install

```bash
pip install -e .             # core
pip install -e '.[openai]'   # for OpenAI models
pip install -e '.[anthropic]' # for Claude models
pip install -e '.[dev]'      # with pytest
```

Requires Python 3.10+.

## Usage

A single call per iteration (the classic case):

```python
from costscope import CostEstimator

with CostEstimator(model="o1", total_iterations=500, sample_iterations=20) as ce:
    for prompt in prompts:
        response = ce.completion(messages=[{"role": "user", "content": prompt}])
        ...
```

Multiple calls per iteration — sample reflects the full pipeline cost:

```python
with CostEstimator(model="claude-opus-4-7", total_iterations=500) as ce:
    for row in rows:
        with ce.iteration():
            facts = ce.completion(messages=[{"role": "user", "content": extract(row)}])
            summary = ce.completion(messages=[{"role": "user", "content": summarize(facts)}])
```

The first 20 iterations are billed normally and used to build a per-iteration cost and time distribution. After that you'll see something like:

```
┌────────────────────────────────────────────────────────────┐
│ Cost & Time Estimate                                       │
├────────────────────────────────────────────────────────────┤
│  Model:        claude-opus-4-7                             │
│  Sample:       20 of 500 iter  (actual $0.4321)            │
│  Per iter:     $0.0216  (σ $0.0042)                        │
│  Projected:    $10.81                                      │
│  95% CI cost:  $10.05 – $11.57  (±7.0%)                    │
│  Per iter time:  3.4s                                      │
│  Wall time:    28min                                       │
│  95% CI time:  26min – 30min                               │
└────────────────────────────────────────────────────────────┘
  → Proceed? [y/N]:
```

Decline and subsequent `.completion()` calls raise `EstimationCancelled`.

### Skip the prompt

- `auto_confirm=True` — always proceed
- `threshold_usd=10.0` — auto-proceed when the upper bound is under the threshold
- `confirm_fn=...` — supply your own confirmation callback

### OpenAI Responses API

`api="auto"` (default) routes `gpt-image-*` and `gpt-5*` to the Responses API, leaving chat-style models on chat completions. Force one explicitly:

```python
CostEstimator(model="gpt-5", api="responses", ...)
```

The adapter translates `messages=` → `input=` and reads tokens from `response.usage.input_tokens` / `output_tokens` (and `output_tokens_details.image_tokens` for image generation).

### Image generation (gpt-image-1)

```python
with CostEstimator(model="gpt-image-1", total_iterations=200) as ce:
    for prompt in prompts:
        ce.completion(input=prompt, tools=[{"type": "image_generation"}])
```

Image-output tokens are priced separately ($40/1M for gpt-image-1). See `examples/image_generation.py`.

### Driving the SDK yourself

If you can't use `ce.completion()` (e.g. you call `client.images.generate()` directly, or stream), use the escape hatch:

```python
with ce.iteration():
    resp = my_custom_call(...)
    ce.record(cost=compute_cost(resp), elapsed=measured_seconds)
```

### Synthetic mode

For tests, demos, and dev loops where real API calls would cost money:

```python
from costscope import CostEstimator, SyntheticConfig

cfg = SyntheticConfig(
    input_median=800, output_median=300, reasoning_median=2000,
    latency_median=1.2,            # simulate ~1.2s/call for time estimates
    image_output_median=4000,      # for image-gen models
    seed=42,
)

with CostEstimator(model="o1", total_iterations=500, synthetic=True, synthetic_config=cfg) as ce:
    ...
```

See `examples/basic.py` for a full runnable example.

## Supported models (built-in pricing)

OpenAI o-series (`o1`, `o3`, `o3-mini`, ...), GPT-4o, GPT-5, gpt-image-1, Claude 4.x (Opus, Sonnet, Haiku). For other models, supply prices via `SyntheticConfig.custom_prices` or extend `pricing.py`.

## Use it as a Claude Code skill

I've also packaged costscope as a Claude Code plugin so the wrapping happens automatically. When Claude is writing or editing code that loops over LLM calls — the canonical `for x in items: client.chat.completions.create(...)` shape, or its Anthropic equivalent — the skill nudges it to reach for `CostEstimator` before the diff ever lands. It skips one-shot calls, agentic loops with unpredictable branching, and jobs already gated by another budget mechanism, so it stays out of the way when sampling-based estimation isn't the right tool.

To install in a Claude Code session, register this repo as a plugin marketplace and then install the `costscope` plugin from it:

```text
/plugin marketplace add ahuang915/costscope
/plugin install costscope@costscope-marketplace
```

The first command points Claude Code at `.claude-plugin/marketplace.json` in this repo; the second installs the plugin, which loads `skills/costscope/SKILL.md` into your session. Run `/plugin list` afterwards to confirm it's enabled — the skill will then surface automatically whenever you're about to write a batched LLM loop. To pin to a local checkout instead (useful while editing the skill yourself), pass an absolute path to `/plugin marketplace add` in place of the GitHub shorthand:

```text
/plugin marketplace add /absolute/path/to/costscope
```

## Tests

```bash
pytest
```
