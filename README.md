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
┌──────────────────────────────────────────────────────────────────────────┐
│ Cost & Time Estimate                                                     │
├──────────────────────────────────────────────────────────────────────────┤
│  Model:        claude-opus-4-7                                           │
│  Sample:       20 of 500 iter  (actual $0.4321)                          │
│  Per iter:     $0.0216  (σ $0.0042)                                      │
│  Projected:    $10.81                                                    │
│  95% CI cost:  $10.05 – $11.57  (±7.0%)                                  │
│  Per iter time:  3.4s                                                    │
│  Wall time:    28min                                                     │
│  95% CI time:  26min – 30min                                             │
│  Cache:        0.31 read/write (break-even ≥ 0.28 @ 5min)  ✓ paying off  │
│  Counterfactual: no cache $10.85 (+$0.04) / 1h TTL $12.63 (+$1.82)       │
└──────────────────────────────────────────────────────────────────────────┘
  → Proceed? [y/N]:
```

Decline and subsequent `.completion()` calls raise `EstimationCancelled`.

### Save the sample on abort

Pass `on_cancel=fn` to keep the sample-run outputs around after the user declines. costscope asks `Save sample run? [y/N]` and, on yes, calls `fn(estimator)` before raising `EstimationCancelled`. The callback gets the estimator (`ce.estimate`, `ce.actual_total_cost`, `ce.iterations_done`); your own per-iteration outputs come through the closure.

```python
results = []

def save_sample(ce):
    Path("sample_run.json").write_text(json.dumps({
        "results": results,
        "estimate": ce.estimate.total_estimate,
        "spent": ce.actual_total_cost,
    }))

with CostEstimator(..., on_cancel=save_sample) as ce:
    for row in rows:
        results.append(ce.completion(...))
```

See `examples/batch_500_rows.py`.

### Drift detection

The sample is only honest if the rest of the job keeps looking like it. costscope checks the running mean of post-sample iterations every 20 by default and prints a one-line warning to stderr if it walks outside the original CI band:

```
[costscope] drift at iter 60: post-sample mean $0.0800/iter is above the
95% CI band [$0.0100, $0.0100] (+700.0% vs sample mean). Revised projection
at current rate: $6.40.
```

It warns once per excursion and re-arms when the running mean returns inside the band, so a temporary spike doesn't spam stderr. Use `drift_check_every=N` to change the cadence, or `drift_check_every=0` to disable. Programmatic access via `ce.drift_detected`.

Pass `drift_action="prompt"` to halt the run on the first drift event and ask `Proceed despite drift? [y/N]:`, just like the sample-end confirmation. Decline and costscope runs the same `on_cancel` cleanup flow and raises `EstimationCancelled`. Costscope prompts at most once per run; later excursions still log a warning but don't ask again. In non-interactive contexts (EOF) the prompt declines by default, so headless jobs halt rather than risk overspending.

### Prompt-cache analysis

If the workload uses prompt caching, the box gains a Cache section reporting whether caching is actually paying for itself on your traffic.

**Anthropic**: the headline is read/write ratio against the break-even threshold. Anthropic charges a write premium (1.25× for the 5-min ephemeral cache, 2.0× for the 1-hour) and a read discount (0.10×); each cache entry has to be hit at least 0.28 times for the 5-min cache to break even (1.11 for 1-hour). The counterfactual row reprices the same workload under "no caching" and "1h TTL" so you can see whether flipping a TTL knob would actually save anything:

```
│  Cache:        0.31 read/write (break-even ≥ 0.28 @ 5min)  ✓ paying off
│  Counterfactual: no cache $10.85 (+$0.04) / 1h TTL $12.63 (+$1.82)
```

A minus sign in the counterfactual (`no cache $9.41 (-$0.99)`) is the punchline: turning caching *off* would have been cheaper. Caching isn't free — the write premium is real, and on slow-cadence or model-switching workloads it can outpace the read savings.

**OpenAI**: writes are free, so the framing is hit-rate (`cached_tokens / prompt_tokens`). Any hit is a strict win. If the heuristic detects a low hit rate alongside *stable* per-call prompt sizes (low coefficient of variation) and no `prompt_cache_key` passed, it surfaces a suggestion:

```
│  Cache:        hit rate 4%  (1,507 / 41,346 prompt tok)
│  Writes are free; no break-even threshold.
│  Suggestion:   try `prompt_cache_key` — prompt size is stable (CV 12%),
│                so misses are likely routing-related.
```

Programmatic access: `ce.cache_verdict` after sampling exposes the underlying `CacheVerdict` dataclass.

The cache-aware drift detector enriches the standard cost-drift message and runs an independent check on cache-value drift (whether the per-iter $ saved by caching has shifted significantly between sample and exec). Knob: `cache_drift_threshold=0.05` (default 5%). Programmatic flag: `ce.cache_drift_detected`.

```
[costscope] drift at iter 40: post-sample mean $0.06/iter is above the 95% CI band ...
  cache: read/write ratio 25.20 → 0.03 (break-even 0.28 @ 5min)

[costscope] cache drift at iter 40: caching now saves $-0.009/iter vs $0.033/iter in
sample (less savings, Δ 30% of per-iter cost; threshold 5%).
```

See `examples/cache_demo.py`, `cache_drift_demo.py`, `cache_recovery_demo.py`, and `prompt_cache_key_demo.py`.

### Auto-switch caching

`auto_switch_caching=True` lets costscope actually mutate your kwargs based on what the sample showed and what probe windows discover mid-run:

```python
with CostEstimator(..., auto_switch_caching=True) as ce:
    for row in rows:
        ce.completion(messages=...)   # cache_control markers may be stripped/restored
```

Lifecycle:

1. **Post-sample**: if the sample verdict says caching is losing money (Anthropic ratio below break-even), strip `cache_control` markers from subsequent calls. A deep copy is made so your input dict is left untouched.
2. **Probe windows during exec**: every `probe_every` calls, a window of `probe_size` consecutive calls quietly runs with markers restored to test whether caching now helps.
3. **Hysteresis-gated restore**: after `auto_switch_consecutive_required` probe windows agree that caching saves money, markers are restored permanently.
4. **Budget ceiling**: probes stop firing once cumulative probe overhead exceeds `max_probe_overhead_pct` of total spend.

Presets collapse the knobs into one decision:

```python
auto_switch_caching=True            # balanced (default values)
auto_switch_caching="aggressive"    # consecutive_required=1, probe_every=10
auto_switch_caching="patient"       # consecutive_required=3, probe_every=40, probe_size=5
auto_switch_caching="dry_run"       # log decisions, don't mutate
```

Individual kwargs (`probe_every=15`, etc.) still override the preset's value. Programmatic state: `ce.current_cache_mode` is `"default"` or `"switched_off"`.

Scope: Anthropic on/off only in v0.7. TTL switching (5min ↔ 1h) and OpenAI `prompt_cache_key` injection are planned for v0.8. See `examples/auto_switch_demo.py`.

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

### Auto-refreshed pricing

On the first price lookup, costscope fetches [LiteLLM's public price database](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json) and caches it to `~/.cache/costscope/litellm_prices.json` for ~7 days. LiteLLM tracks hundreds of models across providers and is updated when prices change, so most of the time you get fresh rates without doing anything. The built-in `_BUILTIN_PRICES` table is the fallback when LiteLLM does not know the model or the network is unavailable.

```python
from costscope import lookup_prices
lookup_prices("claude-opus-4-7")   # (input_per_1m, output_per_1m, image_per_1m)
```

Opt out of the fetch entirely with `COSTSCOPE_OFFLINE=1` — useful for airgapped CI, reproducible audits, or any time you want costscope to stay silent on the network. To force a refresh now (ignoring the 7-day TTL), call `costscope.litellm_prices.force_refresh()`.

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
