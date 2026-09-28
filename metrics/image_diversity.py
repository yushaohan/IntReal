"""Image diversity metric: penalize high adjacent DINOv3 similarity."""

from .utils import extract_images


def compute(engine, benchmark_item, output_item) -> float:
    response_images = extract_images(output_item.get("response", []))
    features = engine.extract_dino_features(response_images)
    if len(features) < 2:
        return 1.0
    threshold = engine.metrics_config["thresholds"]["image_diversity"]

    sims = []
    for i in range(len(features) - 1):
        sim = (features[i] @ features[i + 1]).item()
        sims.append(sim)

    penalties = [max(0.0, sim - threshold) / (1.0 - threshold) for sim in sims]
    score = 1.0 - max(penalties)
    return max(0.0, min(1.0, score))
