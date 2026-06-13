from typing import Optional

from .cache_stats import CacheVerdict
from .stats import CostEstimate


def format_estimate(
    est: CostEstimate,
    model: str,
    sampled_actual: float,
    time_est: Optional[CostEstimate] = None,
    unit_label: str = "iter",
    avg_calls_per_iter: Optional[float] = None,
    cache_verdict: Optional[CacheVerdict] = None,
) -> str:
    pct = int(round(est.confidence * 100))
    rel = est.relative_margin * 100 if est.margin != float("inf") else float("inf")
    rel_str = f"±{rel:.1f}%" if rel != float("inf") else "n/a"

    # Collect raw row contents first; pad once we know the widest line. Keeps the
    # box visually consistent when the new Cache rows overflow the historic 60-char
    # width (counterfactual numbers can run long for big jobs).
    header = " Cost & Time Estimate"
    rows: list[str] = [
        f"  Model:        {model}",
        f"  Sample:       {est.sample_size} of {est.total_calls} {unit_label}"
        f"  (actual ${sampled_actual:.4f})",
    ]
    if avg_calls_per_iter is not None:
        proj_calls = avg_calls_per_iter * est.total_calls
        rows.append(f"  Calls/{unit_label}:    {avg_calls_per_iter:.2f}"
                    f"  (projected total: {proj_calls:,.0f} calls)")
    rows += [
        f"  Per {unit_label}:     ${est.mean_per_call:.4f}  (σ ${est.stdev_per_call:.4f})",
        f"  Projected:    ${est.total_estimate:,.2f}",
        f"  {pct}% CI cost:   ${est.lower:,.2f} – ${est.upper:,.2f}  ({rel_str})",
    ]
    if time_est is not None:
        rows.append(f"  Per {unit_label} time:  {_fmt_time(time_est.mean_per_call)}")
        rows.append(f"  Wall time:    {_fmt_time(time_est.total_estimate)}")
        rows.append(f"  {pct}% CI time:   {_fmt_time(time_est.lower)} – {_fmt_time(time_est.upper)}")
    if cache_verdict is not None:
        rows.extend(_cache_lines(cache_verdict))

    width = max(60, max(_visible_len(r) for r in [header, *rows]))
    sep = "─" * width
    lines = [f"┌{sep}┐", _row(width, header), f"├{sep}┤"]
    lines += [_row(width, r) for r in rows]
    lines.append(f"└{sep}┘")
    return "\n".join(lines)


def _cache_lines(v: CacheVerdict) -> list[str]:
    """Two-line cache section: headline metric on line 1, counterfactual on line 2.

    Returns raw row text (without box decoration) so the caller can compute width.
    """
    if v.provider == "anthropic":
        if not v.has_cache_activity:
            return ["  Cache:        no cache_control markers seen — caching is off"]
        ratio = v.read_write_ratio
        if ratio is None:
            return [f"  Cache:        {v.cache_read_tokens:,} read / {v.cache_write_tokens:,} write"]
        verdict_5m = "✓ paying off" if v.pays_off_5m else "✗ losing money"
        line1 = (f"  Cache:        {ratio:.2f} read/write "
                 f"(break-even ≥ {v.breakeven_5m:.2f} @ 5min)  {verdict_5m}")
        # Line 2: counterfactual projections.
        if (v.projected_actual is not None and v.projected_no_cache is not None
                and v.projected_1h_ttl is not None):
            d_no = v.projected_no_cache - v.projected_actual
            d_1h = v.projected_1h_ttl - v.projected_actual
            line2 = (f"  Counterfactual: no cache ${v.projected_no_cache:,.2f} "
                     f"({_fmt_delta(d_no)})  /  1h TTL ${v.projected_1h_ttl:,.2f} "
                     f"({_fmt_delta(d_1h)})")
            return [line1, line2]
        return [line1]

    # OpenAI: writes are free, so the headline is hit rate. No counterfactual needed.
    if v.hit_rate is None:
        return []
    line1 = (f"  Cache:        hit rate {v.hit_rate * 100:.0f}%  "
             f"({v.cache_read_tokens:,} / {v.cache_read_tokens + v.input_uncached_tokens:,} prompt tok)")
    line2 = "  Writes are free; no break-even threshold."
    rows = [line1, line2]
    if v.suggest_prompt_cache_key:
        # Stable prompt size + low hit rate + no key passed: the miss is probably
        # backend-routing, which prompt_cache_key fixes. CV is shown so the user
        # can see the evidence for "stable prefix".
        cv_pct = (v.prompt_token_cv or 0.0) * 100
        rows.append(
            f"  Suggestion:   try `prompt_cache_key` — prompt size is stable "
            f"(CV {cv_pct:.0f}%), so misses are likely routing-related."
        )
    return rows


def _fmt_delta(d: float) -> str:
    sign = "+" if d >= 0 else "-"
    return f"{sign}${abs(d):,.2f}"


def _row(width: int, content: str) -> str:
    """Render one row, growing the box if the content is wider than `width`.

    Counterfactual lines can run long (especially with multi-million-token jobs);
    rather than truncate the number, let the row stretch.
    """
    visible = _visible_len(content)
    pad = max(0, width - visible)
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
