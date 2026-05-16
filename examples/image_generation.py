"""Estimate cost for batch image generation with gpt-image-1 via the Responses API.

gpt-image-1 prices text-input tokens at $5/1M and image-output tokens at $40/1M
(image tokens dominate). costscope auto-routes gpt-image-* to the Responses API
and extracts image_tokens from `response.usage.output_tokens_details`.

For a dry run without spending anything:

    python examples/image_generation.py --synthetic
"""

import sys

from costscope import CostEstimator, EstimationCancelled, SyntheticConfig


def main(synthetic: bool):
    prompts = [f"A surreal scene #{i}" for i in range(200)]

    cfg = (
        SyntheticConfig(
            input_median=120,           # short text prompt
            output_median=0,            # no chat text output
            reasoning_median=0,
            image_output_median=4000,   # ~4k image tokens per 1024x1024 image
            image_output_sigma=0.2,
            latency_median=8.0,         # gpt-image-1 is slow
            latency_sigma=0.3,
            seed=42,
        )
        if synthetic else None
    )

    try:
        with CostEstimator(
            model="gpt-image-1",
            total_iterations=len(prompts),
            sample_iterations=10,
            confidence=0.95,
            synthetic=synthetic,
            synthetic_config=cfg,
            concurrency=5,             # 5 parallel image jobs
        ) as ce:
            for prompt in prompts:
                # Real call (synthetic=False): goes to OpenAI Responses API
                # because api='auto' routes gpt-image-* there.
                ce.completion(
                    input=prompt,
                    tools=[{"type": "image_generation"}],
                )
    except EstimationCancelled as e:
        print(f"\nCancelled: {e}")
        return

    print(f"\nDone. Actual total: ${ce.actual_total_cost:.2f}")
    print(
        f"Projected: ${ce.estimate.total_estimate:.2f} "
        f"({int(ce.estimate.confidence*100)}% CI ±${ce.estimate.margin:.2f})"
    )
    print(f"Wall time @ concurrency={ce.concurrency}: {ce.wall_time_estimate:.0f}s")


if __name__ == "__main__":
    main(synthetic="--synthetic" in sys.argv)
