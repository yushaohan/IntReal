"""Quantity compliance metric: check if the number of generated images matches golden."""

from .utils import extract_images


def compute(engine, benchmark_item, output_item) -> float:
    expected = len(extract_images(benchmark_item.get("golden_result", [])))
    actual = len(extract_images(output_item.get("response", [])))
    return 1.0 if expected == actual else 0.0
