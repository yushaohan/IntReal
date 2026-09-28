"""Text richness metric: LLM comparative evaluation with position swap."""

from .utils import call_llm, extract_text

# Matches {"winner": 0}, {"winner": 1}, {"winner": 2}
_WINNER_RE = r'"winner"\s*:\s*[012]'


def _build_prompt(engine, benchmark_item, output_item):
    benchmark_id = benchmark_item.get("id", 0)
    prompt_text = benchmark_item.get("prompt", {}).get("text", "")
    reference_text = extract_text(benchmark_item.get("golden_result", []))
    model_text = extract_text(output_item.get("response", []))

    # Position swap based on benchmark id parity to debias
    if benchmark_id % 2 == 0:
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
    return prompt, benchmark_id


async def _call_and_score(engine, prompt, benchmark_id):
    cfg = engine.metrics_config["llm"]["text_richness"]
    api_config = engine.experts_config["api"]
    temperature = cfg["temperature"]
    max_tokens = cfg["max_tokens"]
    max_retries = engine.metrics_config.get("max_retry", 0)

    result = await call_llm(api_config, prompt, temperature, max_tokens, max_retries,
                            format_regex=_WINNER_RE)
    winner = result.get("winner", 0)

    if winner == 0:
        return 1.0

    # id%2==0: pos1=reference, pos2=model → model wins when winner==2
    # id%2==1: pos1=model, pos2=reference → model wins when winner==1
    if benchmark_id % 2 == 0:
        return 1.0 if winner == 2 else 0.0
    else:
        return 1.0 if winner == 1 else 0.0


async def compute_async(engine, benchmark_item, output_item) -> float:
    prompt, benchmark_id = _build_prompt(engine, benchmark_item, output_item)
    return await _call_and_score(engine, prompt, benchmark_id)
