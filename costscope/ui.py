from .stats import CostEstimate


def format_estimate(est: CostEstimate, model: str, sampled_actual: float) -> str:
    pct = int(round(est.confidence * 100))
    rel = est.relative_margin * 100 if est.margin != float("inf") else float("inf")
    rel_str = f"±{rel:.1f}%" if rel != float("inf") else "n/a"
    width = 56
    sep = "─" * width

    lines = [
        f"┌{sep}┐",
        _row(width, " Cost Estimate"),
        f"├{sep}┤",
        _row(width, f"  Model:        {model}"),
        _row(width, f"  Sample:       {est.sample_size} of {est.total_calls} calls"
                    f"  (actual ${sampled_actual:.4f})"),
        _row(width, f"  Per call:     ${est.mean_per_call:.4f}"
                    f"  (σ ${est.stdev_per_call:.4f})"),
        _row(width, f"  Projected:    ${est.total_estimate:,.2f}"),
        _row(width, f"  {pct}% CI:        ${est.lower:,.2f} – ${est.upper:,.2f}  ({rel_str})"),
        f"└{sep}┘",
    ]
    return "\n".join(lines)


def _row(width: int, content: str) -> str:
    pad = max(0, width - _visible_len(content))
    return f"│{content}{' ' * pad}│"


def _visible_len(s: str) -> int:
    return len(s)
