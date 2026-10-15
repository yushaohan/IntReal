"""Average both response orders; ties score 1, and either failed direction fails the metric."""

import asyncio

from .utils import call_llm, extract_text

# Matches {"winner": 0}, {"winner": 1}, {"winner": 2}
_WINNER_RE = r'"winner"\s*:\s*[012]'


def _build_prompt(engine, benchmark_item, output_item, model_position):
    prompt_text = benchmark_item.get("prompt", {}).get("text", "")
    reference_text = extract_text(benchmark_item.get("golden_result", []))
    model_text = extract_text(output_item.get("response", []))

    if model_position == 2:
        response_1 = reference_text
        response_2 = model_text
    else:
        response_1 = model_text
        response_2 = reference_text

    prompt_template = engine.prompts_config["text_richness"]
    prompt = (
        prompt_template
        .replace("{prompt}", prompt_text)
        .replace("{response_1}", response_1)
        .replace("{response_2}", response_2)
    )
    return prompt


async def _call_and_score(engine, prompt, model_position):
    cfg = engine.metrics_config["llm"]["text_richness"]
    api_config = engine.experts_config["api"]
    temperature = cfg["temperature"]
    max_tokens = cfg["max_tokens"]
    max_retries = engine.metrics_config.get("max_retry", 0)

    result = await call_llm(api_config, prompt, temperature, max_tokens, max_retries,
                            format_regex=_WINNER_RE)
    winner = result.get("winner")
    if type(winner) is not int or winner not in (0, 1, 2):
        raise ValueError("Expected integer winner in {0, 1, 2}")

    if winner == 0:
        return 1.0

    return 1.0 if winner == model_position else 0.0


async def compute_async(engine, benchmark_item, output_item) -> float:
    scores = await asyncio.gather(*(
        _call_and_score(engine, _build_prompt(engine, benchmark_item, output_item, position), position)
        for position in (1, 2)
    ))
    return sum(scores) / 2
