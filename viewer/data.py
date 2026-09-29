"""JSONL discovery and ID-based joins, independent of Flask and model dependencies."""

import json
from pathlib import Path

METRICS = {
    "text_richness": "Text Richness",
    "text_logicality": "Text Logicality",
    "image_aesthetic": "Image Aesthetic",
    "image_diversity": "Image Diversity",
    "quantity_compliance": "Quantity Compliance",
    "context_fidelity": "Context Fidelity",
    "subject_consistency": "Subject Consistency",
}
CATEGORIES = ("story", "knowledge", "planning", "tutorial")
SOURCE_CATEGORY = {
    "storystream": "story", "wikipedia": "knowledge",
    "wikitravel": "planning", "wikivoyage": "planning", "xiachufang": "tutorial",
}
SEQUENCE_METRICS = {"context_fidelity", "subject_consistency"}


class DataError(ValueError):
    """A local dataset cannot be interpreted safely."""


def read_index(path, id_key):
    """Read on each request so appended evaluation results are visible on refresh."""
    path = Path(path)
    if not path.is_file():
        return {}
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                sample_id = record[id_key]
                if isinstance(sample_id, bool) or not isinstance(sample_id, (str, int)):
                    raise ValueError(f"{id_key} must be a string or integer")
                if record.get("metrics") is not None and not isinstance(record["metrics"], dict):
                    raise ValueError("metrics must be an object or null")
                key = str(sample_id)
                if key in records:
                    raise ValueError(f"duplicate {id_key}: {sample_id}")
                records[key] = record
            except (ValueError, TypeError, KeyError) as exc:
                raise DataError(f"{path.name}, line {line_no}: {exc}") from exc
    return records


def category(record):
    return SOURCE_CATEGORY.get(record.get("source"))


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def benchmarks(self):
        return sorted(p.stem for p in (self.root / "benchmark").glob("*.jsonl"))

    def benchmark(self, name):
        if name not in self.benchmarks():
            raise DataError("Unknown benchmark.")
        return read_index(self.root / "benchmark" / f"{name}.jsonl", "id")

    def models(self, benchmark, results_only=False):
        folders = ("results",) if results_only else ("outputs", "results")
        names = set()
        for folder in folders:
            for directory in (self.root / folder).glob("*"):
                if directory.is_dir() and (directory / f"{benchmark}.jsonl").is_file():
                    names.add(directory.name)
        return sorted(names)

    def records(self, folder, model, benchmark):
        if folder not in ("outputs", "results"):
            raise DataError("Unknown data directory.")
        if benchmark not in self.benchmarks() or model not in self.models(benchmark):
            raise DataError("Unknown model or benchmark.")
        return read_index(self.root / folder / model / f"{benchmark}.jsonl", "eval_set_id")

    def response(self, model, benchmark, sample_id):
        key = str(sample_id)
        output = self.records("outputs", model, benchmark).get(key)
        result = self.records("results", model, benchmark).get(key)
        # Prefer the original response; results also retain responses if outputs are absent.
        record = output if output is not None else result
        if record is None:
            return None
        return {
            "sf": record.get("response_sf", record.get("response", [])),
            "tf": record.get("response_tf"),
            "metrics": (result or {}).get("metrics") or {},
        }
