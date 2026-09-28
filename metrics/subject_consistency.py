"""Subject consistency metric: DINOv3 cosine similarity between generated and golden images."""

from .utils import extract_images


def compute(engine, benchmark_item, output_item, tf_item=None):
    source = benchmark_item.get("source", "")
    if source not in ("storystream", "xiachufang"):
        return None

    golden_result = benchmark_item.get("golden_result", [])
    golden_images = [b["content"] for b in golden_result if b.get("type") == "image"]
    source_item = tf_item if tf_item is not None else output_item
    response_images = extract_images(source_item.get("response", []))

    prompt_images = benchmark_item.get("prompt", {}).get("images", [])
    start_idx = 0 if len(prompt_images) > 0 else 1

    gen_subset = response_images[start_idx:]
    gold_subset = golden_images[start_idx:]

    if len(gen_subset) != len(gold_subset):
        return None

    if len(gen_subset) == 0:
        return None

    gen_features = engine.extract_dino_features(gen_subset)
    gold_features = engine.extract_dino_features(gold_subset)

    sims = []
    for gf, ff in zip(gen_features, gold_features):
        sim = (gf @ ff).item()
        sims.append(sim)

    return sims
