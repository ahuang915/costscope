from typing import Optional

from .stats import CostEstimate


def format_estimate(
    est: CostEstimate,
    model: str,
    sampled_actual: float,
    time_est: Optional[CostEstimate] = None,
    concurrency: int = 1,
    unit_label: str = "iter",
    avg_calls_per_iter: Optional[float] = None,
) -> str:
    pct = int(round(est.confidence * 100))
    rel = est.relative_margin * 100 if est.margin != float("inf") else float("inf")
    rel_str = f"±{rel:.1f}%" if rel != float("inf") else "n/a"
    width = 60
    sep = "─" * width

    lines = [
        f"┌{sep}┐",
        _row(width, " Cost & Time Estimate"),
        f"├{sep}┤",
        _row(width, f"  Model:        {model}"),
        _row(width, f"  Sample:       {est.sample_size} of {est.total_calls} {unit_label}"
                    f"  (actual ${sampled_actual:.4f})"),
    ]
    if avg_calls_per_iter is not None:
        proj_calls = avg_calls_per_iter * est.total_calls
        lines.append(_row(width, f"  Calls/{unit_label}:    {avg_calls_per_iter:.2f}"
                                 f"  (projected total: {proj_calls:,.0f} calls)"))
    lines += [
        _row(width, f"  Per {unit_label}:     ${est.mean_per_call:.4f}"
                    f"  (σ ${est.stdev_per_call:.4f})"),
        _row(width, f"  Projected:    ${est.total_estimate:,.2f}"),
        _row(width, f"  {pct}% CI cost:   ${est.lower:,.2f} – ${est.upper:,.2f}  ({rel_str})"),
    ]
    if time_est is not None:
        seq_total = time_est.total_estimate
        seq_lo = time_est.lower
        seq_hi = time_est.upper
        c = max(concurrency, 1)
        wall = seq_total / c
        wall_lo = seq_lo / c
        wall_hi = seq_hi / c
        lines.append(_row(width, f"  Per {unit_label} time:  {_fmt_time(time_est.mean_per_call)}"))
        if c > 1:
            lines.append(_row(width, f"  Wall time:    {_fmt_time(wall)}  (concurrency={c})"))
            lines.append(_row(width, f"  {pct}% CI time:   {_fmt_time(wall_lo)} – {_fmt_time(wall_hi)}"))
        else:
            lines.append(_row(width, f"  Wall time:    {_fmt_time(wall)}  (sequential)"))
            lines.append(_row(width, f"  {pct}% CI time:   {_fmt_time(seq_lo)} – {_fmt_time(seq_hi)}"))

    lines.append(f"└{sep}┘")
    return "\n".join(lines)


def _row(width: int, content: str) -> str:
    pad = max(0, width - _visible_len(content))
    return f"│{content}{' ' * pad}│"


def _visible_len(s: str) -> int:
    return len(s)


def _fmt_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"
