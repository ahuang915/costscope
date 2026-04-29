import sys
from typing import Any, Callable, Optional

from tqdm.auto import tqdm

from .exceptions import EstimationCancelled
from .stats import CostEstimate, compute_estimate
from .synthetic import SyntheticBackend, SyntheticConfig
from .ui import format_estimate


ConfirmFn = Callable[[CostEstimate, str], bool]


class CostEstimator:
    """Sample → estimate → confirm → run, around a batched LLM job.

    The first `sample_size` calls dispatched through `.completion()` are billed
    normally and used to build a per-call cost distribution. Once `sample_size`
    is reached, a confidence interval over the projected total is shown and the
    user is asked to proceed; on decline, subsequent calls raise
    `EstimationCancelled`.

    Pass `synthetic=True` to swap the litellm backend for an in-memory mock
    that draws token counts from configurable log-normal distributions —
    useful for tests, demos, and dev loops where real API calls would cost money.
    """

    def __init__(
        self,
        model: str,
        total_calls: int,
        sample_size: int = 20,
        confidence: float = 0.95,
        synthetic: bool = False,
        synthetic_config: Optional[SyntheticConfig] = None,
        auto_confirm: Optional[bool] = None,
        threshold_usd: Optional[float] = None,
        confirm_fn: Optional[ConfirmFn] = None,
    ):
        if total_calls < 1:
            raise ValueError("total_calls must be >= 1")
        if sample_size < 2:
            raise ValueError("sample_size must be >= 2 to compute a CI")
        if confidence not in (0.90, 0.95, 0.99):
            raise ValueError("confidence must be one of 0.90, 0.95, 0.99")

        self.model = model
        self.total_calls = total_calls
        self.sample_size = min(sample_size, total_calls)
        self.confidence = confidence
        self.threshold_usd = threshold_usd
        self.auto_confirm = auto_confirm
        self.confirm_fn = confirm_fn or _default_confirm

        self._synthetic = synthetic
        self._backend = SyntheticBackend(synthetic_config or SyntheticConfig()) if synthetic else None
        self._litellm = None

        self._sample_costs: list[float] = []
        self._exec_costs: list[float] = []
        self._calls_made = 0
        self._cancelled = False
        self._sample_bar: Optional[tqdm] = None
        self._exec_bar: Optional[tqdm] = None
        self._final_estimate: Optional[CostEstimate] = None

    def __enter__(self) -> "CostEstimator":
        self._sample_bar = tqdm(
            total=self.sample_size,
            desc="Sampling",
            unit="call",
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

    def completion(self, **kwargs: Any) -> Any:
        """Dispatch one LLM call. Routes to litellm or the synthetic backend.

        During the sampling phase, this counts toward the cost sample. When the
        sample is complete, the confirmation prompt fires before this call
        returns to the caller — so the user sees the estimate before any
        further calls are dispatched.
        """
        if self._cancelled:
            raise EstimationCancelled("Estimator was cancelled; refusing further calls")

        response, cost = self._dispatch(kwargs)
        self._calls_made += 1

        if self._calls_made <= self.sample_size:
            self._sample_costs.append(cost)
            self._update_sample_bar()
            if self._calls_made == self.sample_size:
                self._finalize_sampling()
        else:
            self._exec_costs.append(cost)
            if self._exec_bar is not None:
                self._exec_bar.update(1)

        return response

    def _dispatch(self, kwargs: dict) -> tuple[Any, float]:
        if self._synthetic:
            response = self._backend.completion(model=self.model, **kwargs)
            return response, self._backend.cost(response)

        litellm = self._lazy_litellm()
        kwargs.setdefault("model", self.model)
        response = litellm.completion(**kwargs)
        cost = float(litellm.completion_cost(completion_response=response) or 0.0)
        return response, cost

    def _lazy_litellm(self):
        if self._litellm is not None:
            return self._litellm
        try:
            import litellm  # type: ignore
        except ImportError as e:
            raise ImportError(
                "litellm is required for non-synthetic mode. "
                "Install with: pip install 'costscope[litellm]'"
            ) from e
        self._litellm = litellm
        return litellm

    def _update_sample_bar(self) -> None:
        if self._sample_bar is None:
            return
        if len(self._sample_costs) >= 2:
            est = compute_estimate(self._sample_costs, self.total_calls, self.confidence)
            self._sample_bar.set_postfix_str(
                f"est ${est.total_estimate:,.2f} ±${est.margin:,.2f}",
                refresh=False,
            )
        self._sample_bar.update(1)

    def _finalize_sampling(self) -> None:
        if self._sample_bar is not None:
            self._sample_bar.close()
            self._sample_bar = None

        est = compute_estimate(self._sample_costs, self.total_calls, self.confidence)
        self._final_estimate = est
        sampled_actual = sum(self._sample_costs)

        print(format_estimate(est, self.model, sampled_actual), file=sys.stderr)

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
                f"User declined. Spent ${sampled_actual:.4f} on {self.sample_size} sample calls."
            )

        remaining = self.total_calls - self.sample_size
        if remaining > 0:
            self._exec_bar = tqdm(
                total=remaining,
                desc="Running ",
                unit="call",
                leave=True,
            )

    @property
    def estimate(self) -> Optional[CostEstimate]:
        """Cost estimate computed at the end of the sampling phase. None before that."""
        return self._final_estimate

    @property
    def actual_total_cost(self) -> float:
        """Sum of observed costs across sampling + execution phases."""
        return sum(self._sample_costs) + sum(self._exec_costs)

    @property
    def calls_made(self) -> int:
        return self._calls_made


def _default_confirm(est: CostEstimate, model: str) -> bool:
    try:
        ans = input("  → Proceed? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")
