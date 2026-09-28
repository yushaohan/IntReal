"""Text coherence metric: LLM fact-checking with search."""

from .utils import call_llm

# Matches {"coherent": true} or {"coherent": false}
_COHERENT_RE = r'"coherent"\s*:\s*(?:true|false)'


def _response_to_text_with_markers(blocks):
    parts = []
    for b in blocks:
        if b.get("type") == "text":
            parts.append(b["content"])
        elif b.get("type") == "image":
            parts.append("<image>")
    return "\n".join(parts)


def _build_prompt(engine, benchmark_item, output_item):
    prompt_text = benchmark_item.get("prompt", {}).get("text", "")
    prompt_images = benchmark_item.get("prompt", {}).get("images", [])
    model_text = _response_to_text_with_markers(output_item.get("response", []))

    prompt_template = engine.prompts_config["text_logicality"]
    prompt = (
        prompt_template
        .replace("{prompt}", prompt_text)
        .replace("{model_text}", model_text)
    )
    return prompt, prompt_images


async def _call_and_score(engine, prompt, image_paths):
    cfg = engine.metrics_config["llm"]["text_logicality"]
    api_config = engine.experts_config["api"]
    temperature = cfg["temperature"]
    max_tokens = cfg["max_tokens"]
    max_retries = engine.metrics_config.get("max_retry", 0)

    result = await call_llm(api_config, prompt, temperature, max_tokens, max_retries,
                            format_regex=_COHERENT_RE)
    coherent = result["coherent"]
    return 1.0 if coherent else 0.0


async def compute_async(engine, benchmark_item, output_item) -> float:
    prompt, image_paths = _build_prompt(engine, benchmark_item, output_item)
    return await _call_and_score(engine, prompt, image_paths)
