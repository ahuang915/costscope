"""Run a 2-call-per-row pipeline over 500 rows with claude-opus-4-7, gated by a cost estimate.

If you decline at the sample-end prompt, costscope offers to save the partial
sample data — the rows already processed plus the per-iteration estimate.
"""

import json
from pathlib import Path

from costscope import CostEstimator, EstimationCancelled

# Replace with your actual data source (CSV read, DB query, etc.).
rows = [f"row {i}" for i in range(500)]


def extract_prompt(row: str) -> str:
    return f"Extract key facts from:\n\n{row}"


def summarize_prompt(facts: str) -> str:
    return f"Summarize in one sentence:\n\n{facts}"


results = []


def save_sample(ce: CostEstimator) -> None:
    out = Path("sample_run.json")
    out.write_text(json.dumps({
        "model": ce.model,
        "iterations_done": ce.iterations_done,
        "actual_cost_usd": ce.actual_total_cost,
        "estimate_total_usd": ce.estimate.total_estimate,
        "estimate_per_iter_usd": ce.estimate.mean_per_call,
        "results": results,
    }, indent=2))
    print(f"  → saved {len(results)} sample outputs to {out}")


try:
    with CostEstimator(
        model="claude-opus-4-7",
        total_iterations=len(rows),
        sample_iterations=20,
        confidence=0.95,
        on_cancel=save_sample,
    ) as ce:
        for row in rows:
            with ce.iteration():
                facts_resp = ce.completion(
                    max_tokens=1024,
                    messages=[{"role": "user", "content": extract_prompt(row)}],
                )
                facts = facts_resp.content[0].text
                summary_resp = ce.completion(
                    max_tokens=256,
                    messages=[{"role": "user", "content": summarize_prompt(facts)}],
                )
                results.append(summary_resp.content[0].text)
except EstimationCancelled as e:
    print(f"Cancelled: {e}")
    raise SystemExit(1)

print(f"Done. Processed {len(results)} rows.")
print(f"Actual total cost: ${ce.actual_total_cost:.4f}")
print(f"Actual wall time (sequential): {ce.actual_total_time:.1f}s")
