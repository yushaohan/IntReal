"""Image aesthetic quality metric using RALI."""

from PIL import Image

from .utils import extract_images


def compute(engine, benchmark_item, output_item) -> float | None:
    image_paths = extract_images(output_item.get("response", []))
    if not image_paths:
        return 1.0

    scorer = engine.get_rali_scorer()
    images = [Image.open(path).convert("RGB") for path in image_paths]
    raw_scores = scorer.model(images)
    cfg = engine.metrics_config["aesthetic"]
    aesthetic_min = cfg["aesthetic_min"]
    aesthetic_max = cfg["aesthetic_max"]
    normalized = ((raw_scores - aesthetic_min) / (aesthetic_max - aesthetic_min)).clamp(0.0, 1.0)
    return normalized.mean().item()
