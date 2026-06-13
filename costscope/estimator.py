import copy
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

from tqdm.auto import tqdm

from .cache_stats import CacheVerdict, compute_cache_verdict
from .exceptions import EstimationCancelled
from .pricing import builtin_cost, cost_with_cache
from .stats import CostEstimate, compute_estimate
from .synthetic import SyntheticBackend, SyntheticConfig
from .ui import format_estimate


@dataclass
class CallTelemetry:
    """What `_dispatch` returns: cost + timing + cache-relevant token counts.

    `input_uncached_tokens` is the prefix portion that paid the plain input rate.
    For Anthropic this is `usage.input_tokens` directly; for OpenAI it's
    `prompt_tokens - cached_tokens`.

    `has_prompt_cache_key` is None for non-OpenAI calls; True/False for OpenAI
    calls based on whether the user passed `prompt_cache_key` in the request.
    Used by the cache verdict to decide whether to suggest enabling routing.
    """
    response: Any
    cost: float
    elapsed: float
    input_uncached_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    image_output_tokens: int = 0
    has_prompt_cache_key: Optional[bool] = None


ConfirmFn = Callable[[CostEstimate, str], bool]
CancelCleanupFn = Callable[["CostEstimator"], None]

_API_AUTO = "auto"
_API_CHAT = "chat"
_API_RESPONSES = "responses"

# Auto-switch presets. Pass any key on `auto_switch_caching=` (also True == "balanced").
# Individual kwargs (probe_every, probe_size, etc.) still override the preset value.
_AUTO_SWITCH_PRESETS = {
    "balanced": {
        "auto_switch_dry_run": False,
        "auto_switch_consecutive_required": 2,
        "probe_every": 20,
        "probe_size": 3,
        "max_probe_overhead_pct": 0.05,
    },
    "aggressive": {
        # React on the first probe verdict; probe more often; tolerate higher probe cost.
        "auto_switch_dry_run": False,
        "auto_switch_consecutive_required": 1,
        "probe_every": 10,
        "probe_size": 3,
        "max_probe_overhead_pct": 0.10,
    },
    "patient": {
        # Want strong evidence before flipping; probe less often; tight cost ceiling.
        "auto_switch_dry_run": False,
        "auto_switch_consecutive_required": 3,
        "probe_every": 40,
        "probe_size": 5,
        "max_probe_overhead_pct": 0.03,
    },
    "dry_run": {
        # Balanced cadence, but no mutation — logs the decisions only.
        "auto_switch_dry_run": True,
        "auto_switch_consecutive_required": 2,
        "probe_every": 20,
        "probe_size": 3,
        "max_probe_overhead_pct": 0.05,
    },
}
_UNSET = object()


class CostEstimator:
    """Sample → estimate → confirm → run, around a batched LLM job.

    Sampling unit is an *iteration*. By default each `.completion()` call is its
    own iteration (back-compat). Wrap a multi-call iteration in `with ce.iteration():`
    to roll several calls into one sample; the CI is then projected over
    `total_iterations`, not raw call count.

    Per-iteration wall time is recorded alongside cost and projected sequentially.

    `api="auto"` routes models to the right OpenAI endpoint (chat completions
    or responses). Force one explicitly with `api="chat"` or `api="responses"`.
    """

    def __init__(
        self,
        model: str,
        total_iterations: Optional[int] = None,
        total_calls: Optional[int] = None,  # back-compat alias
        sample_iterations: Optional[int] = None,
        sample_size: int = 20,
        confidence: float = 0.95,
        synthetic: bool = False,
        synthetic_config: Optional[SyntheticConfig] = None,
        auto_confirm: Optional[bool] = None,
        threshold_usd: Optional[float] = None,
        confirm_fn: Optional[ConfirmFn] = None,
        on_cancel: Optional[CancelCleanupFn] = None,
        api: str = _API_AUTO,
        drift_check_every: int = 20,
        drift_action: str = "warn",
        cache_drift_threshold: float = 0.05,
        auto_switch_caching: Any = False,
        auto_switch_dry_run: Any = _UNSET,
        auto_switch_consecutive_required: Any = _UNSET,
        probe_every: Any = _UNSET,
        probe_size: Any = _UNSET,
        max_probe_overhead_pct: Any = _UNSET,
    ):
        total = self._pick_one(total_iterations, total_calls, "total_iterations", "total_calls")
        if total is None or total < 1:
            raise ValueError("total_iterations must be >= 1")
        sample = sample_iterations if sample_iterations is not None else sample_size
        if sample < 2:
            raise ValueError("sample size must be >= 2 to compute a CI")
        if confidence not in (0.90, 0.95, 0.99):
            raise ValueError("confidence must be one of 0.90, 0.95, 0.99")
        if api not in (_API_AUTO, _API_CHAT, _API_RESPONSES):
            raise ValueError("api must be 'auto', 'chat', or 'responses'")
        if drift_action not in ("warn", "prompt"):
            raise ValueError("drift_action must be 'warn' or 'prompt'")
        if cache_drift_threshold < 0:
            raise ValueError("cache_drift_threshold must be >= 0")

        # Resolve `auto_switch_caching` (bool or preset string) into the on/off flag plus
        # the defaults that fill in any kwargs the caller didn't explicitly pass. The
        # explicit-kwarg-wins layering lets users say `auto_switch_caching="aggressive"`
        # and then bump `probe_size=5` without losing the rest of the preset.
        if isinstance(auto_switch_caching, str):
            preset_name = auto_switch_caching
            if preset_name not in _AUTO_SWITCH_PRESETS:
                raise ValueError(
                    f"auto_switch_caching preset {preset_name!r} unknown; "
                    f"choices: {sorted(_AUTO_SWITCH_PRESETS)}"
                )
            auto_switch_enabled = True
            preset = _AUTO_SWITCH_PRESETS[preset_name]
        elif auto_switch_caching is True:
            auto_switch_enabled = True
            preset = _AUTO_SWITCH_PRESETS["balanced"]
        else:
            auto_switch_enabled = False
            # Even when off, keep the balanced numbers so the runtime state is well-defined.
            preset = _AUTO_SWITCH_PRESETS["balanced"]

        auto_switch_dry_run = (
            preset["auto_switch_dry_run"] if auto_switch_dry_run is _UNSET
            else auto_switch_dry_run
        )
        auto_switch_consecutive_required = (
            preset["auto_switch_consecutive_required"] if auto_switch_consecutive_required is _UNSET
            else auto_switch_consecutive_required
        )
        probe_every = preset["probe_every"] if probe_every is _UNSET else probe_every
        probe_size = preset["probe_size"] if probe_size is _UNSET else probe_size
        max_probe_overhead_pct = (
            preset["max_probe_overhead_pct"] if max_probe_overhead_pct is _UNSET
            else max_probe_overhead_pct
        )

        if auto_switch_consecutive_required < 1:
            raise ValueError("auto_switch_consecutive_required must be >= 1")
        if probe_every < 1:
            raise ValueError("probe_every must be >= 1")
        if probe_size < 1:
            raise ValueError("probe_size must be >= 1")
        if max_probe_overhead_pct < 0:
            raise ValueError("max_probe_overhead_pct must be >= 0")

        self.model = model
        self.total_iterations = total
        self.sample_size = min(sample, total)
        self.confidence = confidence
        self.threshold_usd = threshold_usd
        self.auto_confirm = auto_confirm
        self.confirm_fn = confirm_fn or _default_confirm
        self.on_cancel = on_cancel
        self.api = self._resolve_api(model, api)
        self._drift_check_every = max(drift_check_every, 0)
        self.drift_action = drift_action
        self.cache_drift_threshold = cache_drift_threshold
        self._drift_detected = False
        self._drift_prompted = False
        self._cache_drift_detected = False
        self._cache_drift_prompted = False
        # Auto-switch configuration. The state machine is dormant unless
        # auto_switch_caching is enabled; that keeps default behavior unchanged.
        self.auto_switch_caching = auto_switch_enabled
        self.auto_switch_dry_run = auto_switch_dry_run
        self.auto_switch_consecutive_required = auto_switch_consecutive_required
        self.probe_every = probe_every
        self.probe_size = probe_size
        self.max_probe_overhead_pct = max_probe_overhead_pct
        # "default" = user's original cache_control markers in place.
        # "switched_off" = markers stripped on every call; probes restore them in a window.
        self._current_cache_mode = "default"
        self._pending_probe_count = 0
        self._calls_since_last_probe = 0
        self._probe_verdict_history: list[bool] = []
        # Per-window accumulators (reset at start of each probe window).
        self._active_probe_input_uncached = 0
        self._active_probe_output = 0
        self._active_probe_read = 0
        self._active_probe_write = 0
        self._active_probe_cost = 0.0
        # Budget tracking for the overhead ceiling.
        self._cumulative_probe_overhead = 0.0
        self._cumulative_exec_cost = 0.0

        self._synthetic = synthetic
        self._backend = SyntheticBackend(synthetic_config or SyntheticConfig()) if synthetic else None

        self._in_iteration = False
        self._iter_cost = 0.0
        self._iter_time = 0.0
        self._iter_calls = 0

        self._sample_costs: list[float] = []
        self._sample_times: list[float] = []
        self._sample_call_counts: list[int] = []
        self._exec_costs: list[float] = []
        self._exec_times: list[float] = []
        self._exec_call_counts: list[int] = []
        # Cache-token totals across the sample only — the cache verdict is computed once,
        # at the end of sampling, alongside the cost CI.
        self._sample_cache_read_tokens = 0
        self._sample_cache_write_tokens = 0
        self._sample_input_uncached_tokens = 0
        self._sample_output_tokens = 0
        # Per-call sample data used by the prompt_cache_key suggestion heuristic (OpenAI).
        # Stable prompt-token counts + low hit rate + no key passed = routing-fix signal.
        self._sample_prompt_token_counts: list[int] = []
        self._sample_openai_calls = 0
        self._sample_openai_calls_without_key = 0
        # Exec-phase cache totals are kept separately so the drift detector can compare
        # the post-sample window against the sample's cache behavior.
        self._exec_cache_read_tokens = 0
        self._exec_cache_write_tokens = 0
        self._exec_input_uncached_tokens = 0
        self._exec_output_tokens = 0
        self._iterations_done = 0
        self._total_calls_made = 0
        self._cancelled = False
        self._sample_bar: Optional[tqdm] = None
        self._exec_bar: Optional[tqdm] = None
        self._final_estimate: Optional[CostEstimate] = None
        self._final_time_estimate: Optional[CostEstimate] = None
        self._final_cache_verdict: Optional[CacheVerdict] = None

    @staticmethod
    def _pick_one(a, b, a_name, b_name):
        if a is not None and b is not None:
            raise ValueError(f"Pass only one of {a_name} / {b_name}")
        return a if a is not None else b

    @staticmethod
    def _resolve_api(model: str, api: str) -> str:
        if api != _API_AUTO:
            return api
        lowered = model.lower()
        if lowered.startswith(("gpt-image", "gpt-5")):
            return _API_RESPONSES
        return _API_CHAT

    def __enter__(self) -> "CostEstimator":
        self._sample_bar = tqdm(
            total=self.sample_size,
            desc="Sampling",
            unit="iter",
            leave=True,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        for bar in (self._sample_bar, self._exec_bar):
            if bar is not None:
                bar.close()
        self._sample_bar = None
        self._exec_bar = None
        return False

    @contextmanager
    def iteration(self) -> Iterator["CostEstimator"]:
        """Group calls into a single sample unit.

        Cost and wall time accumulated inside the block become one iteration's
        worth. Calls made outside any `iteration()` block each count as their
        own iteration.
        """
        if self._in_iteration:
            raise RuntimeError("iteration() cannot be nested")
        if self._cancelled:
            raise EstimationCancelled("Estimator was cancelled; refusing further calls")
        self._in_iteration = True
        self._iter_cost = 0.0
        self._iter_time = 0.0
        self._iter_calls = 0
        try:
            yield self
        finally:
            cost = self._iter_cost
            elapsed = self._iter_time
            calls = self._iter_calls
            self._in_iteration = False
            self._iter_cost = 0.0
            self._iter_time = 0.0
            self._iter_calls = 0
            self._record_iteration(cost, elapsed, calls)

    def completion(self, **kwargs: Any) -> Any:
        """Dispatch one LLM call. Routes to OpenAI/Anthropic SDK or synthetic backend.

        Outside an `iteration()` block, the call is its own iteration. Inside one,
        cost and time accumulate into the active iteration; the iteration sample
        is recorded when the block exits.
        """
        if self._cancelled:
            raise EstimationCancelled("Estimator was cancelled; refusing further calls")

        # Auto-switch can rewrite the call's cache_control markers and/or run the
        # call as part of a probe window. Both decisions are made up-front so the
        # kwargs handed to the SDK reflect the chosen config.
        is_probe = self._decide_probe()
        effective_kwargs = self._kwargs_for_mode(kwargs, is_probe=is_probe)

        tel = self._dispatch(effective_kwargs)
        self._total_calls_made += 1

        # Sample window feeds the cache verdict at finalize; exec window feeds the
        # cache-drift detector. Both run cheaply — just four integer adds per call.
        if self._iterations_done < self.sample_size:
            self._sample_cache_read_tokens += tel.cache_read_tokens
            self._sample_cache_write_tokens += tel.cache_write_tokens
            self._sample_input_uncached_tokens += tel.input_uncached_tokens
            self._sample_output_tokens += tel.output_tokens
            # Per-call counts for the prompt_cache_key suggestion (OpenAI only).
            if tel.has_prompt_cache_key is not None:
                self._sample_openai_calls += 1
                if not tel.has_prompt_cache_key:
                    self._sample_openai_calls_without_key += 1
                self._sample_prompt_token_counts.append(
                    tel.input_uncached_tokens + tel.cache_read_tokens
                )
        else:
            self._exec_cache_read_tokens += tel.cache_read_tokens
            self._exec_cache_write_tokens += tel.cache_write_tokens
            self._exec_input_uncached_tokens += tel.input_uncached_tokens
            self._exec_output_tokens += tel.output_tokens
            # Probe bookkeeping happens in exec only — sample never probes.
            self._cumulative_exec_cost += tel.cost
            if is_probe:
                self._record_probe_call(tel)
            else:
                self._calls_since_last_probe += 1

        if self._in_iteration:
            self._iter_cost += tel.cost
            self._iter_time += tel.elapsed
            self._iter_calls += 1
        else:
            self._record_iteration(tel.cost, tel.elapsed, 1)

        return tel.response

    def record(self, cost: float, elapsed: float = 0.0) -> None:
        """Manually record a cost/time pair as a call.

        Escape hatch for callers that drive the LLM SDK themselves (e.g.
        `images.generate`, streaming responses) but still want CI-based gating.
        Aggregation rules match `completion()`: inside `iteration()`, accumulates
        into the active iteration; outside, becomes its own iteration.
        """
        if self._cancelled:
            raise EstimationCancelled("Estimator was cancelled; refusing further calls")
        self._total_calls_made += 1
        if self._in_iteration:
            self._iter_cost += cost
            self._iter_time += elapsed
            self._iter_calls += 1
        else:
            self._record_iteration(cost, elapsed, 1)

    def _record_iteration(self, cost: float, elapsed: float, calls: int) -> None:
        self._iterations_done += 1
        if self._iterations_done <= self.sample_size:
            self._sample_costs.append(cost)
            self._sample_times.append(elapsed)
            self._sample_call_counts.append(calls)
            self._update_sample_bar()
            if self._iterations_done == self.sample_size:
                self._finalize_sampling()
        else:
            self._exec_costs.append(cost)
            self._exec_times.append(elapsed)
            self._exec_call_counts.append(calls)
            if self._exec_bar is not None:
                self._exec_bar.update(1)
            self._maybe_check_drift()

    def _prompt_proceed_despite_drift(self) -> bool:
        """Ask the user whether to continue after a drift warning.

        EOF (non-interactive) is treated as a decline so headless jobs halt
        rather than risk overspending after a drift event.
        """
        try:
            ans = input("  → Proceed despite drift? [y/N]: ").strip().lower()
        except EOFError:
            return False
        return ans in ("y", "yes")

    def _maybe_run_cleanup(self) -> None:
        """Prompt the user; if they confirm, invoke on_cancel with self.

        Skipped silently when no callback is registered or when the prompt EOFs
        (non-interactive caller). Exceptions from the callback propagate so the
        caller sees what went wrong saving.
        """
        if self.on_cancel is None:
            return
        try:
            ans = input("  → Save sample run? [y/N]: ").strip().lower()
        except EOFError:
            return
        if ans not in ("y", "yes"):
            return
        self.on_cancel(self)

    def _maybe_check_drift(self) -> None:
        """Warn once if the running post-sample mean walks outside the CI band.

        Re-arms if the running mean returns inside the band, so a recovered job
        can warn again on a later excursion.
        """
        if self._drift_check_every <= 0 or self._final_estimate is None:
            return
        n = len(self._exec_costs)
        if n == 0 or n % self._drift_check_every != 0:
            return
        est = self._final_estimate
        if est.total_calls <= 0:
            return
        per_iter_margin = est.margin / est.total_calls
        lo = est.mean_per_call - per_iter_margin
        hi = est.mean_per_call + per_iter_margin
        running_mean = sum(self._exec_costs) / n
        outside = running_mean < lo or running_mean > hi
        if outside and not self._drift_detected:
            direction = "above" if running_mean > hi else "below"
            revised_total = running_mean * est.total_calls
            delta_pct = (
                (running_mean - est.mean_per_call) / est.mean_per_call * 100
                if est.mean_per_call else 0.0
            )
            msg = (
                f"\n[costscope] drift at iter {self._iterations_done}: "
                f"post-sample mean ${running_mean:.4f}/iter is {direction} the "
                f"{int(est.confidence*100)}% CI band "
                f"[${lo:.4f}, ${hi:.4f}] ({delta_pct:+.1f}% vs sample mean). "
                f"Revised projection at current rate: ${revised_total:,.2f}."
            )
            cache_line = self._cache_drift_note()
            if cache_line:
                msg = f"{msg}\n  {cache_line}"
            print(msg, file=sys.stderr)
            self._drift_detected = True
            if self.drift_action == "prompt" and not self._drift_prompted:
                self._drift_prompted = True
                if not self._prompt_proceed_despite_drift():
                    self._cancelled = True
                    self._maybe_run_cleanup()
                    raise EstimationCancelled(
                        f"User declined after drift at iter {self._iterations_done}."
                    )
        elif not outside and self._drift_detected:
            self._drift_detected = False

        # Independent regime-flip check — fires even when cost is stable.
        self._maybe_check_cache_drift()

    def _cache_drift_note(self) -> str:
        """One-line diagnostic explaining how caching shifted between sample and exec.

        Returns "" when there's nothing useful to say (no cache in sample, no exec
        iters yet, etc.). Otherwise reports the meaningful provider-specific delta:
          - Anthropic: read/write ratio change
          - OpenAI:    cached / prompt hit-rate change
        """
        is_anthropic = self.model.lower().startswith("claude")
        s_read = self._sample_cache_read_tokens
        s_write = self._sample_cache_write_tokens
        s_uncached = self._sample_input_uncached_tokens
        e_read = self._exec_cache_read_tokens
        e_write = self._exec_cache_write_tokens
        e_uncached = self._exec_input_uncached_tokens
        if is_anthropic:
            if s_write == 0 and e_write == 0:
                return ""
            sample_r = s_read / s_write if s_write else float("inf")
            exec_r = e_read / e_write if e_write else float("inf")
            return (
                f"cache: read/write ratio {self._fmt_ratio(sample_r)} → "
                f"{self._fmt_ratio(exec_r)} (break-even 0.28 @ 5min)"
            )
        sample_prompt = s_read + s_uncached
        exec_prompt = e_read + e_uncached
        if sample_prompt == 0 and exec_prompt == 0:
            return ""
        sample_hit = (s_read / sample_prompt) if sample_prompt else 0.0
        exec_hit = (e_read / exec_prompt) if exec_prompt else 0.0
        return f"cache: hit rate {sample_hit:.0%} → {exec_hit:.0%}"

    @staticmethod
    def _fmt_ratio(r: float) -> str:
        if r == float("inf"):
            return "∞"
        return f"{r:.2f}"

    def _maybe_check_cache_drift(self) -> None:
        """Fire when the per-iter cache savings drifts by > cache_drift_threshold.

        "Cache savings" is the delta between actual cost and the no-cache
        counterfactual: positive when caching helps, negative when it hurts. We
        compute it per iteration in the sample and in the exec window; if the
        absolute swing exceeds `cache_drift_threshold * sample_actual_per_iter`,
        we warn.

        This catches the case the cost-drift detector misses: caching has flipped
        from "paying off" to "losing money" (or vice versa) while total cost
        stayed inside the CI band because some other factor offset the shift.
        """
        if self._drift_check_every <= 0 or self._final_estimate is None:
            return
        n_exec = len(self._exec_costs)
        if n_exec == 0 or n_exec % self._drift_check_every != 0:
            return
        if self.sample_size <= 0:
            return
        # No cache activity anywhere → nothing meaningful to drift from.
        if not (self._sample_cache_read_tokens or self._sample_cache_write_tokens
                or self._exec_cache_read_tokens or self._exec_cache_write_tokens):
            return

        sample_per_iter = self._cache_value_per_iter(
            self._sample_input_uncached_tokens,
            self._sample_output_tokens,
            self._sample_cache_read_tokens,
            self._sample_cache_write_tokens,
            n=self.sample_size,
        )
        exec_per_iter = self._cache_value_per_iter(
            self._exec_input_uncached_tokens,
            self._exec_output_tokens,
            self._exec_cache_read_tokens,
            self._exec_cache_write_tokens,
            n=n_exec,
        )
        delta = exec_per_iter - sample_per_iter

        # Compare against per-iter sample *actual* cost so the threshold scales with
        # the workload: 5% drift on a 10¢-per-call job is materially different
        # from 5% drift on a $10-per-call job, but always the right order of magnitude.
        sample_actual_per_iter = (
            sum(self._sample_costs) / self.sample_size if self._sample_costs else 0.0
        )
        if sample_actual_per_iter <= 0:
            return
        rel_swing = abs(delta) / sample_actual_per_iter
        outside = rel_swing > self.cache_drift_threshold

        if outside and not self._cache_drift_detected:
            direction = "more" if delta > 0 else "less"
            print(
                f"\n[costscope] cache drift at iter {self._iterations_done}: "
                f"caching now saves ${exec_per_iter:.4f}/iter "
                f"vs ${sample_per_iter:.4f}/iter in sample "
                f"({direction} savings, Δ {rel_swing*100:.1f}% of per-iter cost; "
                f"threshold {self.cache_drift_threshold*100:.1f}%).",
                file=sys.stderr,
            )
            self._cache_drift_detected = True
            if self.drift_action == "prompt" and not self._cache_drift_prompted:
                self._cache_drift_prompted = True
                if not self._prompt_proceed_despite_drift():
                    self._cancelled = True
                    self._maybe_run_cleanup()
                    raise EstimationCancelled(
                        f"User declined after cache drift at iter {self._iterations_done}."
                    )
        elif not outside and self._cache_drift_detected:
            self._cache_drift_detected = False

    def _decide_probe(self) -> bool:
        """Return True if this call should run in the probe (original) config.

        Three cases produce a True:
          1. We're mid-window (still finishing a started probe window).
          2. It's time to start a new window: enough calls have passed AND the
             cumulative probe-overhead budget hasn't been exceeded.
          3. Auto-switch is on AND we're in a switched-off mode (so probing
             back to the original config makes sense — nothing to probe otherwise).
        """
        if not self.auto_switch_caching:
            return False
        if self._iterations_done < self.sample_size:
            return False  # Sample phase doesn't probe.
        if self._current_cache_mode == "default":
            return False  # No alternate config to probe.

        # Mid-window: finish what we started.
        if self._pending_probe_count > 0:
            return True

        # Start a new probe window?
        if self._calls_since_last_probe < self.probe_every:
            return False
        if not self._probe_budget_available():
            return False

        # Open a fresh window. `pending` counts *all* probe calls including this one;
        # _record_probe_call decrements after each, so finalize fires after probe_size calls.
        self._pending_probe_count = self.probe_size
        self._calls_since_last_probe = 0
        self._reset_active_probe_window()
        return True

    def _probe_budget_available(self) -> bool:
        """Stop probing once cumulative probe overhead exceeds max_probe_overhead_pct
        of total exec spend. Negative overhead (probes turned out cheaper than current
        config) doesn't count against the budget."""
        if self._cumulative_exec_cost <= 0:
            return True
        if self._cumulative_probe_overhead <= 0:
            return True
        return (self._cumulative_probe_overhead / self._cumulative_exec_cost) < self.max_probe_overhead_pct

    def _kwargs_for_mode(self, kwargs: dict, *, is_probe: bool) -> dict:
        """Return kwargs to send to the SDK based on current_cache_mode and whether this
        call is a probe.

        - Probe call: always uses the user's original kwargs (probing the alternate config).
        - Non-probe in switched_off mode: strip all cache_control markers (deep copy first
          so the caller's dict is untouched).
        - Non-probe in default mode: pass through unchanged.
        """
        if is_probe or self._current_cache_mode == "default":
            return kwargs
        # switched_off — strip markers from system + messages.
        out = copy.deepcopy(kwargs)
        if "system" in out:
            out["system"] = _strip_cache_control(out["system"])
        if "messages" in out:
            out["messages"] = _strip_cache_control(out["messages"])
        return out

    def _reset_active_probe_window(self) -> None:
        self._active_probe_input_uncached = 0
        self._active_probe_output = 0
        self._active_probe_read = 0
        self._active_probe_write = 0
        self._active_probe_cost = 0.0

    def _record_probe_call(self, tel: CallTelemetry) -> None:
        """Accumulate a probe call into the current window and update overhead budget.

        Per-call overhead = (probe cost) − (what this call would have cost in switched_off
        mode, i.e. with every prefix byte billed at the plain input rate). Positive means
        the probe was more expensive than the current mode; negative means it was cheaper.
        """
        # Per-call overhead is computed against the current (switched-off) mode's pricing.
        probe_cost = tel.cost
        no_cache_cost = cost_with_cache(
            self.model,
            input_uncached_tokens=(
                tel.input_uncached_tokens + tel.cache_read_tokens + tel.cache_write_tokens
            ),
            output_tokens=tel.output_tokens,
            cache_read_tokens=0,
            cache_write_tokens=0,
            image_output_tokens=tel.image_output_tokens,
        )
        self._cumulative_probe_overhead += probe_cost - no_cache_cost

        self._active_probe_input_uncached += tel.input_uncached_tokens
        self._active_probe_output += tel.output_tokens
        self._active_probe_read += tel.cache_read_tokens
        self._active_probe_write += tel.cache_write_tokens
        self._active_probe_cost += probe_cost

        self._pending_probe_count -= 1
        if self._pending_probe_count <= 0:
            self._finalize_probe_window()

    def _finalize_probe_window(self) -> None:
        """Decide whether the just-completed probe window argued for switching back.

        We compute the savings the probe would have produced vs running the same window
        in switched-off mode. Positive → caching helped → vote 'switch back'. After
        auto_switch_consecutive_required consecutive 'switch back' votes, restore default.
        """
        actual = self._active_probe_cost
        no_cache_equivalent = cost_with_cache(
            self.model,
            input_uncached_tokens=(
                self._active_probe_input_uncached
                + self._active_probe_read
                + self._active_probe_write
            ),
            output_tokens=self._active_probe_output,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
        switch_back_vote = no_cache_equivalent > actual
        self._probe_verdict_history.append(switch_back_vote)

        recent = self._probe_verdict_history[-self.auto_switch_consecutive_required:]
        if (len(recent) >= self.auto_switch_consecutive_required
                and all(recent)
                and self._current_cache_mode == "switched_off"):
            action_msg = (
                f"[costscope] auto-switch: restoring cache_control markers after "
                f"{self.auto_switch_consecutive_required} consecutive probe windows showed "
                f"caching now saves money "
                f"(last window: probe cost ${actual:.4f} vs no-cache ${no_cache_equivalent:.4f})."
            )
            if self.auto_switch_dry_run:
                print(f"\n[costscope] auto-switch (dry run): would restore markers. "
                      f"Probe window saved ${no_cache_equivalent - actual:.4f}.",
                      file=sys.stderr)
            else:
                self._current_cache_mode = "default"
                self._probe_verdict_history.clear()
                print(f"\n{action_msg}", file=sys.stderr)

    def _cache_value_per_iter(self, input_uncached: int, output: int,
                              cache_read: int, cache_write: int, *, n: int) -> float:
        """Per-iter $ saved by caching vs the no-cache counterfactual.

        Positive = caching is helping; negative = caching is costing extra.
        """
        if n <= 0:
            return 0.0
        actual = cost_with_cache(
            self.model,
            input_uncached_tokens=input_uncached,
            output_tokens=output,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )
        # No-cache counterfactual: every cached/written byte gets billed at the plain
        # input rate. We use cost_with_cache with zeroed cache fields and an inflated
        # uncached count, so the same code path / rate lookup applies.
        no_cache = cost_with_cache(
            self.model,
            input_uncached_tokens=input_uncached + cache_read + cache_write,
            output_tokens=output,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
        return (no_cache - actual) / n

    def _dispatch(self, kwargs: dict) -> CallTelemetry:
        if self._synthetic:
            response = self._backend.completion(model=self.model, **kwargs)
            cost = self._backend.cost(response)
            usage = response.usage
            # Mirror the real OpenAI helpers: track whether the user passed
            # prompt_cache_key so the suggestion heuristic has data to work with
            # in synthetic demos.
            has_key = None
            if not _is_anthropic(self.model):
                has_key = "prompt_cache_key" in kwargs and bool(kwargs["prompt_cache_key"])
            return CallTelemetry(
                response=response,
                cost=cost,
                elapsed=response.latency_s,
                input_uncached_tokens=getattr(usage, "input_uncached_tokens", usage.prompt_tokens),
                output_tokens=usage.completion_tokens,
                cache_read_tokens=getattr(usage, "cache_read_tokens", 0),
                cache_write_tokens=getattr(usage, "cache_write_tokens", 0),
                image_output_tokens=getattr(usage, "image_output_tokens", 0),
                has_prompt_cache_key=has_key,
            )

        t0 = time.perf_counter()
        if _is_anthropic(self.model):
            tel = _call_anthropic(self.model, kwargs)
        elif self.api == _API_RESPONSES:
            tel = _call_openai_responses(self.model, kwargs)
        else:
            tel = _call_openai_chat(self.model, kwargs)
        tel.elapsed = time.perf_counter() - t0
        # Cost is computed here so it reflects cache pricing — fixes the prior bug where
        # Anthropic cache_read/write tokens were dropped on the floor.
        tel.cost = cost_with_cache(
            self.model,
            input_uncached_tokens=tel.input_uncached_tokens,
            output_tokens=tel.output_tokens,
            cache_read_tokens=tel.cache_read_tokens,
            cache_write_tokens=tel.cache_write_tokens,
            image_output_tokens=tel.image_output_tokens,
        )
        return tel

    def _update_sample_bar(self) -> None:
        if self._sample_bar is None:
            return
        if len(self._sample_costs) >= 2:
            est = compute_estimate(self._sample_costs, self.total_iterations, self.confidence)
            self._sample_bar.set_postfix_str(
                f"est ${est.total_estimate:,.2f} ±${est.margin:,.2f}",
                refresh=False,
            )
        self._sample_bar.update(1)

    def _finalize_sampling(self) -> None:
        if self._sample_bar is not None:
            self._sample_bar.close()
            self._sample_bar = None

        est = compute_estimate(self._sample_costs, self.total_iterations, self.confidence)
        time_est = compute_estimate(self._sample_times, self.total_iterations, self.confidence)
        self._final_estimate = est
        self._final_time_estimate = time_est
        sampled_actual = sum(self._sample_costs)

        avg_calls = sum(self._sample_call_counts) / len(self._sample_call_counts)
        cache_verdict = compute_cache_verdict(
            self.model,
            sample_cache_read_tokens=self._sample_cache_read_tokens,
            sample_cache_write_tokens=self._sample_cache_write_tokens,
            sample_input_uncached_tokens=self._sample_input_uncached_tokens,
            sample_output_tokens=self._sample_output_tokens,
            sample_size=len(self._sample_costs),
            total_iterations=self.total_iterations,
            sample_prompt_token_counts=self._sample_prompt_token_counts or None,
            sample_openai_calls_without_key=self._sample_openai_calls_without_key,
            sample_openai_calls=self._sample_openai_calls,
        )
        self._final_cache_verdict = cache_verdict
        print(
            format_estimate(
                est,
                self.model,
                sampled_actual,
                time_est=time_est,
                avg_calls_per_iter=avg_calls,
                cache_verdict=cache_verdict,
            ),
            file=sys.stderr,
        )
        self._maybe_post_sample_auto_switch(cache_verdict)

        if self.threshold_usd is not None and est.upper <= self.threshold_usd:
            print(
                f"  → auto-proceed: upper bound ${est.upper:,.2f} ≤ "
                f"threshold ${self.threshold_usd:,.2f}",
                file=sys.stderr,
            )
            proceed = True
        elif self.auto_confirm is not None:
            proceed = self.auto_confirm
        else:
            proceed = self.confirm_fn(est, self.model)

        if not proceed:
            self._cancelled = True
            self._maybe_run_cleanup()
            raise EstimationCancelled(
                f"User declined. Spent ${sampled_actual:.4f} on {self.sample_size} sample iterations."
            )

        remaining = self.total_iterations - self.sample_size
        if remaining > 0:
            self._exec_bar = tqdm(
                total=remaining,
                desc="Running ",
                unit="iter",
                leave=True,
            )

    @property
    def estimate(self) -> Optional[CostEstimate]:
        """Cost estimate computed at the end of the sampling phase. None before that."""
        return self._final_estimate

    @property
    def time_estimate(self) -> Optional[CostEstimate]:
        """Wall-time estimate (seconds) over total_iterations, run sequentially."""
        return self._final_time_estimate

    @property
    def cache_verdict(self) -> Optional[CacheVerdict]:
        """Prompt-cache effectiveness over the sample. None before sampling finishes."""
        return self._final_cache_verdict

    @property
    def current_cache_mode(self) -> str:
        """\"default\" (caller's original markers) or \"switched_off\" (markers stripped)."""
        return self._current_cache_mode

    def _maybe_post_sample_auto_switch(self, verdict: Optional[CacheVerdict]) -> None:
        """One-shot decision at end of sample: turn caching off if the sample shows it's wasteful.

        Skipped when auto_switch_caching is off, the provider isn't Anthropic (v1 scope),
        or the verdict can't conclude (no cache activity, missing ratio). The dry-run flag
        toggles between logging the decision and actually applying it.
        """
        if not self.auto_switch_caching:
            return
        if verdict is None or verdict.provider != "anthropic":
            return
        if not verdict.has_cache_activity or verdict.pays_off_5m is None:
            return
        if verdict.pays_off_5m:
            return  # Caching is paying off — leave it alone.

        action = "strip cache_control markers (turn caching off)"
        reason = (
            f"sample read/write ratio {verdict.read_write_ratio:.2f} "
            f"is below 5-min break-even {verdict.breakeven_5m:.2f}"
        )
        if self.auto_switch_dry_run:
            print(
                f"\n[costscope] auto-switch (dry run): would {action}. {reason}.",
                file=sys.stderr,
            )
            return
        self._current_cache_mode = "switched_off"
        print(
            f"\n[costscope] auto-switch: stripping cache_control markers going forward. "
            f"{reason}.",
            file=sys.stderr,
        )

    @property
    def actual_total_cost(self) -> float:
        return sum(self._sample_costs) + sum(self._exec_costs)

    @property
    def drift_detected(self) -> bool:
        """True if the running post-sample mean is currently outside the CI band."""
        return self._drift_detected

    @property
    def cache_drift_detected(self) -> bool:
        """True if the running per-iter cache savings has drifted past the threshold."""
        return self._cache_drift_detected

    @property
    def actual_total_time(self) -> float:
        """Sum of measured per-iteration wall times."""
        return sum(self._sample_times) + sum(self._exec_times)

    @property
    def iterations_done(self) -> int:
        return self._iterations_done

    @property
    def calls_made(self) -> int:
        """Total LLM calls dispatched (across all iterations)."""
        return self._total_calls_made

    @property
    def avg_calls_per_iteration(self) -> Optional[float]:
        """Mean calls per iteration over the sample. None before sampling finishes."""
        if not self._sample_call_counts:
            return None
        return sum(self._sample_call_counts) / len(self._sample_call_counts)

    @property
    def total_calls(self) -> int:
        """Back-compat alias for total_iterations."""
        return self.total_iterations


def _default_confirm(est: CostEstimate, model: str) -> bool:
    try:
        ans = input("  → Proceed? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


def _is_anthropic(model: str) -> bool:
    return model.lower().startswith("claude")


def _call_openai_chat(model: str, kwargs: dict) -> CallTelemetry:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise ImportError("openai is required for OpenAI models. Install with: pip install openai") from e
    has_key = "prompt_cache_key" in kwargs and bool(kwargs["prompt_cache_key"])
    response = OpenAI().chat.completions.create(model=model, **kwargs)
    usage = response.usage
    prompt = usage.prompt_tokens
    cached = _maybe_int(getattr(usage, "prompt_tokens_details", None), "cached_tokens")
    return CallTelemetry(
        response=response,
        cost=0.0,
        elapsed=0.0,
        input_uncached_tokens=max(prompt - cached, 0),
        output_tokens=usage.completion_tokens,
        cache_read_tokens=cached,
        has_prompt_cache_key=has_key,
    )


def _call_openai_responses(model: str, kwargs: dict) -> CallTelemetry:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise ImportError("openai is required for OpenAI models. Install with: pip install openai") from e
    # Translate chat-style kwargs: `messages` -> `input`.
    if "messages" in kwargs and "input" not in kwargs:
        kwargs["input"] = kwargs.pop("messages")
    has_key = "prompt_cache_key" in kwargs and bool(kwargs["prompt_cache_key"])
    response = OpenAI().responses.create(model=model, **kwargs)
    usage = response.usage
    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    image_tokens = 0
    details = getattr(usage, "output_tokens_details", None)
    if details is not None:
        image_tokens = getattr(details, "image_tokens", 0) or 0
        # When image_tokens are reported separately, treat the remaining output_tokens
        # as non-image so we don't double-bill at the text-output rate.
        if image_tokens and image_tokens <= output_tokens:
            output_tokens -= image_tokens
    # OpenAI responses API reports cached tokens under input_tokens_details.cached_tokens
    cached = _maybe_int(getattr(usage, "input_tokens_details", None), "cached_tokens")
    return CallTelemetry(
        response=response,
        cost=0.0,
        elapsed=0.0,
        input_uncached_tokens=max(input_tokens - cached, 0),
        output_tokens=output_tokens,
        cache_read_tokens=cached,
        image_output_tokens=image_tokens,
        has_prompt_cache_key=has_key,
    )


def _call_anthropic(model: str, kwargs: dict) -> CallTelemetry:
    try:
        from anthropic import Anthropic  # type: ignore
    except ImportError as e:
        raise ImportError("anthropic is required for Claude models. Install with: pip install anthropic") from e

    messages = list(kwargs.pop("messages", []))
    system = None
    if messages and messages[0].get("role") == "system":
        system = messages.pop(0)["content"]

    create_kwargs = {"model": model, "messages": messages, **kwargs}
    if system is not None:
        create_kwargs["system"] = system
    create_kwargs.setdefault("max_tokens", 4096)

    response = Anthropic().messages.create(**create_kwargs)
    usage = response.usage
    # Anthropic: usage.input_tokens is already the *uncached* portion (the API does the
    # subtraction for us). Cache reads/writes are separate fields.
    return CallTelemetry(
        response=response,
        cost=0.0,
        elapsed=0.0,
        input_uncached_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


def _maybe_int(obj: Any, attr: str) -> int:
    """Pull an int attribute, treating None / missing as 0."""
    if obj is None:
        return 0
    val = getattr(obj, attr, 0)
    return int(val or 0)


def _strip_cache_control(payload: Any) -> Any:
    """Recursively remove `cache_control` fields from a system block / messages payload.

    Anthropic's prompts may carry `cache_control` on system blocks or on individual
    content blocks inside messages. Removing the key turns caching off for that block
    while leaving the actual prompt content untouched (so the model sees the same text).
    Strings, ints, and other scalars pass through unchanged.
    """
    if isinstance(payload, dict):
        return {k: _strip_cache_control(v) for k, v in payload.items() if k != "cache_control"}
    if isinstance(payload, list):
        return [_strip_cache_control(item) for item in payload]
    return payload
