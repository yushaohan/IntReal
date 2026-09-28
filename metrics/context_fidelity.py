"""Context fidelity metric: VLM-as-judge rubric verification against generated images."""

from .utils import call_llm, extract_images

# Matches {"passes": [true, false, ...]}
_PASSES_RE = r'"passes"\s*:\s*\['


def _build_rubric_checklist(rubrics):
    """Format a list of rubrics as a numbered checklist string."""
    return "\n".join(f"{i+1}. {r}" for i, r in enumerate(rubrics))


async def _verify_rubrics(engine, rubrics, image_path):
    """Send one image + numbered rubric list to VLM, return passes array."""
    cfg = engine.metrics_config["llm"]["context_fidelity"]
    api_config = engine.experts_config["api"]
    temperature = cfg["temperature"]
    max_tokens = cfg["max_tokens"]
    max_retries = engine.metrics_config.get("max_retry", 0)

    prompt_template = engine.prompts_config["context_fidelity"]
    checklist = _build_rubric_checklist(rubrics)
    prompt = prompt_template.replace("{Checklist}", checklist)

    result = await call_llm(api_config, prompt, temperature, max_tokens, max_retries, [image_path],
                            format_regex=_PASSES_RE)
    return result.get("passes", [])


async def compute_async(engine, benchmark_item, output_item, tf_item=None):
    golden_result = benchmark_item.get("golden_result", [])
    source_item = tf_item if tf_item is not None else output_item
    response_images = extract_images(source_item.get("response", []))

    # Collect rubric-bearing image blocks from golden_result
    rubric_entries = []
    for b in golden_result:
        if b.get("type") == "image" and b.get("rubrics"):
            rubric_entries.append((b["content"], b["rubrics"]))

    if len(rubric_entries) != len(response_images):
        return None

    image_scores = []
    for (_, rubrics), gen_image_path in zip(rubric_entries, response_images):
        try:
            passes = await _verify_rubrics(engine, rubrics, gen_image_path)
            n_pass = sum(1 for p in passes if p)
            n_total = len(rubrics)
            if n_pass == 0:
                image_scores.append(0.0)
            else:
                score = 0.5 + (n_pass - 1) * 0.5 / (n_total - 1)
                image_scores.append(score)
        except Exception:
            image_scores.append(0.0)

    return image_scores
