import sys
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

from tqdm.auto import tqdm

from .exceptions import EstimationCancelled
from .pricing import builtin_cost
from .stats import CostEstimate, compute_estimate
from .synthetic import SyntheticBackend, SyntheticConfig
from .ui import format_estimate


ConfirmFn = Callable[[CostEstimate, str], bool]

_API_AUTO = "auto"
_API_CHAT = "chat"
_API_RESPONSES = "responses"


class CostEstimator:
    """Sample → estimate → confirm → run, around a batched LLM job.

    Sampling unit is an *iteration*. By default each `.completion()` call is its
    own iteration (back-compat). Wrap a multi-call iteration in `with ce.iteration():`
    to roll several calls into one sample; the CI is then projected over
    `total_iterations`, not raw call count.

    Per-iteration wall time is recorded alongside cost. Projected wall time is
    sequential by default; pass `concurrency=N` to divide by N for a parallel
    estimate.

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
        api: str = _API_AUTO,
        concurrency: int = 1,
    ):
        total = self._pick_one(total_iterations, total_calls, "total_iterations", "total_calls")
        if total is None or total < 1:
            raise ValueError("total_iterations must be >= 1")
        sample = sample_iterations if sample_iterations is not None else sample_size
        if sample < 2:
            raise ValueError("sample size must be >= 2 to compute a CI")
        if confidence not in (0.90, 0.95, 0.99):
            raise ValueError("confidence must be one of 0.90, 0.95, 0.99")
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if api not in (_API_AUTO, _API_CHAT, _API_RESPONSES):
            raise ValueError("api must be 'auto', 'chat', or 'responses'")

        self.model = model
        self.total_iterations = total
        self.sample_size = min(sample, total)
        self.confidence = confidence
        self.threshold_usd = threshold_usd
        self.auto_confirm = auto_confirm
        self.confirm_fn = confirm_fn or _default_confirm
        self.concurrency = concurrency
        self.api = self._resolve_api(model, api)

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
        self._iterations_done = 0
        self._total_calls_made = 0
        self._cancelled = False
        self._sample_bar: Optional[tqdm] = None
        self._exec_bar: Optional[tqdm] = None
        self._final_estimate: Optional[CostEstimate] = None
        self._final_time_estimate: Optional[CostEstimate] = None

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

        response, cost, elapsed = self._dispatch(kwargs)
        self._total_calls_made += 1

        if self._in_iteration:
            self._iter_cost += cost
            self._iter_time += elapsed
            self._iter_calls += 1
        else:
            self._record_iteration(cost, elapsed, 1)

        return response

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

    def _dispatch(self, kwargs: dict) -> tuple[Any, float, float]:
        if self._synthetic:
            response = self._backend.completion(model=self.model, **kwargs)
            cost = self._backend.cost(response)
            elapsed = response.latency_s
            return response, cost, elapsed

        t0 = time.perf_counter()
        if _is_anthropic(self.model):
            response, prompt_tokens, completion_tokens, image_out = _call_anthropic(self.model, kwargs)
        elif self.api == _API_RESPONSES:
            response, prompt_tokens, completion_tokens, image_out = _call_openai_responses(self.model, kwargs)
        else:
            response, prompt_tokens, completion_tokens, image_out = _call_openai_chat(self.model, kwargs)
        elapsed = time.perf_counter() - t0
        cost = builtin_cost(self.model, prompt_tokens, completion_tokens, image_out)
        return response, cost, elapsed

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
        print(
            format_estimate(
                est,
                self.model,
                sampled_actual,
                time_est=time_est,
                concurrency=self.concurrency,
                avg_calls_per_iter=avg_calls,
            ),
            file=sys.stderr,
        )

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
        """Sequential wall-time estimate (seconds) over total_iterations."""
        return self._final_time_estimate

    @property
    def wall_time_estimate(self) -> Optional[float]:
        """Projected wall time in seconds, divided by concurrency."""
        if self._final_time_estimate is None:
            return None
        return self._final_time_estimate.total_estimate / max(self.concurrency, 1)

    @property
    def actual_total_cost(self) -> float:
        return sum(self._sample_costs) + sum(self._exec_costs)

    @property
    def actual_total_time(self) -> float:
        """Sum of measured per-iteration wall times. Sequential — does not account for concurrency."""
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


def _call_openai_chat(model: str, kwargs: dict) -> tuple[Any, int, int, int]:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise ImportError("openai is required for OpenAI models. Install with: pip install openai") from e
    response = OpenAI().chat.completions.create(model=model, **kwargs)
    return response, response.usage.prompt_tokens, response.usage.completion_tokens, 0


def _call_openai_responses(model: str, kwargs: dict) -> tuple[Any, int, int, int]:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise ImportError("openai is required for OpenAI models. Install with: pip install openai") from e
    # Translate chat-style kwargs: `messages` -> `input`.
    if "messages" in kwargs and "input" not in kwargs:
        kwargs["input"] = kwargs.pop("messages")
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
    return response, input_tokens, output_tokens, image_tokens


def _call_anthropic(model: str, kwargs: dict) -> tuple[Any, int, int, int]:
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
    return response, response.usage.input_tokens, response.usage.output_tokens, 0
