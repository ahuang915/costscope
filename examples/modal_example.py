"""Run the synthetic costscope demo on Modal to test remote interactivity.

Two things to verify:

1. Output streaming — tqdm bars and the formatted estimate block should appear
   in your local terminal as the remote function runs. This works with plain:

       modal run examples/modal_example.py

   With AUTO_CONFIRM=True (default below), the run finishes without prompting.

2. Stdin / interactive prompt — flip AUTO_CONFIRM to False and run:

       modal run -i examples/modal_example.py

   The `-i` flag is necessary but not sufficient: the remote function also has
   to explicitly claim stdin via `modal.interact()`. The custom confirm_fn
   below does that, so the input() prompt actually pauses for your reply.
"""

import modal

# Toggle this to test the two paths described above.
AUTO_CONFIRM: bool | None = None

app = modal.App("costscope-modal-test")

image = (
    modal.Image.debian_slim()
    .pip_install("tqdm>=4.65")
    .add_local_python_source("costscope")
)


def _modal_confirm(est, model):
    """Confirm prompt that works inside a Modal remote function.

    `modal.interact()` opens a PTY back to the local terminal so input()
    actually blocks for stdin instead of hitting EOF. Requires `modal run -i`.
    """
    import modal

    modal.interact()
    try:
        ans = input("  → Proceed? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


@app.function(image=image)
def run_estimate(auto_confirm: bool | None):
    from costscope import CostEstimator, EstimationCancelled, SyntheticConfig

    prompts = [f"prompt {i}" for i in range(500)]

    cfg = SyntheticConfig(
        input_median=800,
        output_median=300,
        reasoning_median=2000,
        reasoning_sigma=0.8,
        seed=2026,
    )

    try:
        with CostEstimator(
            model="o1",
            total_calls=len(prompts),
            sample_size=20,
            confidence=0.95,
            synthetic=True,
            synthetic_config=cfg,
            auto_confirm=auto_confirm,
            confirm_fn=_modal_confirm if auto_confirm is None else None,
        ) as ce:
            for prompt in prompts:
                ce.completion(messages=[{"role": "user", "content": prompt}])
    except EstimationCancelled as e:
        print(f"\nCancelled: {e}")
        return

    print(f"\nDone. Actual total: ${ce.actual_total_cost:.2f}")
    print(
        f"Projected was:   ${ce.estimate.total_estimate:.2f} "
        f"({int(ce.estimate.confidence * 100)}% CI ±${ce.estimate.margin:.2f})"
    )


@app.local_entrypoint()
def main():
    run_estimate.remote(AUTO_CONFIRM)
