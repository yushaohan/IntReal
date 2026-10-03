"""Report aggregation. Scores are displayed on a 0–100 scale."""

import math

from categories import subject_consistency_applicable

from .data import CATEGORIES, METRICS, category


def valid_score(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean(values):
    return sum(values) / len(values) if values else None


def metric_summary(records, keys):
    cells = {}
    for key in keys:
        values = [r.get("metrics", {}).get(key) for r in records if isinstance(r.get("metrics"), dict)]
        values = [v for v in values if valid_score(v)]
        avg = mean(values)
        cells[key] = {"score": avg * 100 if avg is not None else None, "count": len(values)}
    overall = mean([cell["score"] for cell in cells.values() if cell["score"] is not None])
    return cells, overall


def benchmark_report(store, benchmark):
    reference = store.benchmark(benchmark)
    rows = []
    for model in store.models(benchmark, results_only=True):
        results = store.records("results", model, benchmark)
        # Only benchmark IDs count. Record count alone cannot detect missing/extra IDs.
        matched = {key: row for key, row in results.items() if key in reference}
        cells = {}
        for metric in METRICS:
            ids = {
                key for key, record in reference.items()
                if metric != "subject_consistency" or subject_consistency_applicable(record)
            }
            summary, _ = metric_summary([matched[key] for key in ids if key in matched], [metric])
            cells[metric] = {**summary[metric], "expected": len(ids)}
        overall = mean([cell["score"] for cell in cells.values() if cell["score"] is not None])
        incomplete = any(cell["count"] != cell["expected"] for cell in cells.values())
        categories = {}
        for name in CATEGORIES:
            ids = {key for key, row in reference.items() if category(row) == name}
            group = [matched[key] for key in ids if key in matched]
            keys = [k for k in METRICS if name not in ("knowledge", "planning") or k != "subject_consistency"]
            group_cells, score = metric_summary(group, keys)
            categories[name] = {
                "score": score, "count": len(group), "expected": len(ids),
                "valid": {k: v["count"] for k, v in group_cells.items()},
                "incomplete": any(v["count"] != len(ids) for v in group_cells.values()),
            }
        # Keep the original viewer's cross-benchmark definition: mean of sample means.
        sample_scores = []
        for record in matched.values():
            metrics = record.get("metrics") or {}
            values = [metrics.get(k) for k in METRICS]
            value = mean([v for v in values if valid_score(v)])
            if value is not None:
                sample_scores.append(value * 100)
        rows.append({
            "model": model, "count": len(matched), "expected": len(reference),
            "extra": len(results.keys() - reference.keys()),
            "metrics": cells, "overall": overall, "categories": categories,
            "incomplete": incomplete,
            "categories_incomplete": any(v["incomplete"] for v in categories.values()),
            "category_mean": mean([v["score"] for v in categories.values() if v["score"] is not None]),
            "sample_mean": mean(sample_scores), "sample_count": len(sample_scores),
        })
    return rows


def all_reports(store):
    return {name: benchmark_report(store, name) for name in store.benchmarks()}
