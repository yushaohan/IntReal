"""Run with: python -m viewer.app [--host 127.0.0.1] [--port 8420]."""

import argparse
import csv
import io
from pathlib import Path

from flask import Flask, Response, abort, render_template, request, send_file, url_for

from .data import CATEGORIES, METRICS, DataError, Store, category
from .reports import benchmark_report, valid_score

ROOT = Path(__file__).resolve().parents[1]
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def create_app(root=ROOT, image_roots=()):
    app = Flask(__name__)
    store = Store(root)
    allowed_images = [store.root / "images", store.root / "outputs"]
    allowed_images += [Path(p).resolve() for p in image_roots]
    allowed_images = [p.resolve() for p in allowed_images]

    @app.template_filter("score")
    def format_score(value):
        return f"{value:.2f}" if valid_score(value) else "—"

    @app.context_processor
    def shared_context():
        return {"metric_labels": METRICS, "categories": CATEGORIES}

    @app.errorhandler(DataError)
    def bad_data(error):
        return render_template("error.html", message=str(error)), 400

    @app.errorhandler(404)
    def missing(error):
        return render_template("error.html", message="The requested page or image was not found."), 404

    def selected_benchmark():
        names = store.benchmarks()
        name = request.args.get("benchmark")
        if name and name not in names:
            abort(404)
        return names, name or ("IntReal" if "IntReal" in names else next(iter(names), None))

    @app.get("/")
    @app.get("/benchmark")
    def benchmark():
        return browse("benchmark")

    @app.get("/response")
    def response():
        return browse("response")

    def browse(view):
        names, name = selected_benchmark()
        models = store.models(name) if name and view == "response" else []
        model = request.args.get("model")
        if model and model not in models:
            abort(404)
        model = model or next(iter(models), None)
        selected_category = request.args.get("category", "")
        if selected_category and selected_category not in CATEGORIES:
            raise DataError("Unknown category.")
        records = list(store.benchmark(name).values()) if name else []
        if selected_category:
            records = [r for r in records if category(r) == selected_category]
        sample_id = request.args.get("id", "").strip()
        try:
            page = int(request.args.get("page", "1"))
        except ValueError as exc:
            raise DataError("Page must be an integer.") from exc
        if sample_id:
            page = next((i + 1 for i, r in enumerate(records) if str(r["id"]) == sample_id), 0)
            if not page:
                raise DataError(f"Sample {sample_id} was not found in this selection.")
        page = max(1, min(page, len(records)))
        sample = records[page - 1] if records else None
        generated = store.response(model, name, sample["id"]) if model and sample else None
        params = {"benchmark": name, "category": selected_category}
        if model:
            params["model"] = model
        previous = url_for(view, **params, page=page - 1) if page > 1 else None
        following = url_for(view, **params, page=page + 1) if page < len(records) else None
        return render_template(
            "browse.html", view=view, benchmarks=names, benchmark=name, models=models,
            model=model, selected_category=selected_category, sample=sample,
            sample_category=category(sample) if sample else None, generated=generated,
            page=page, total=len(records), previous=previous, following=following,
        )

    @app.get("/report")
    def report():
        names, name = selected_benchmark()
        view = request.args.get("view", "metrics")
        if view not in ("metrics", "categories"):
            raise DataError("Unknown report view.")
        rows = benchmark_report(store, name) if name else []
        return render_template(
            "report.html", view=view, benchmarks=names, benchmark=name, rows=rows,
        )

    @app.get("/report.csv")
    def export_report():
        _, name = selected_benchmark()
        if not name:
            raise DataError("No benchmark is available.")
        view = request.args.get("view", "metrics")
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)

        def cell(value):
            if value is None:
                return ""
            if isinstance(value, float):
                return round(value, 6)
            # Treat model names as literal text when opened in spreadsheet software.
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                return "'" + value
            return value

        if view == "categories":
            writer.writerow(["model", "category", "score", "completed_samples", "expected_samples"] + [f"{k}_valid" for k in METRICS])
            for row in benchmark_report(store, name):
                for cat, entry in row["categories"].items():
                    writer.writerow([cell(v) for v in [row["model"], cat, entry["score"], entry["count"], entry["expected"]] + [entry["valid"].get(k, "") for k in METRICS]])
        elif view == "metrics":
            writer.writerow(["model", "completed_samples", "expected_samples", "extra_samples", "overall"] + list(METRICS) + [f"{k}_valid" for k in METRICS])
            for row in benchmark_report(store, name):
                values = [row["model"], row["count"], row["expected"], row["extra"], row["overall"]]
                values += [row["metrics"][k]["score"] for k in METRICS]
                values += [row["metrics"][k]["count"] for k in METRICS]
                writer.writerow([cell(v) for v in values])
        else:
            raise DataError("Unknown report view.")
        return Response(buffer.getvalue(), mimetype="text/csv", headers={"Content-Disposition": 'attachment; filename="intreal-report.csv"'})

    @app.get("/image")
    def image():
        value = request.args.get("path", "")
        try:
            path = Path(value)
            path = (path if path.is_absolute() else store.root / path).resolve()
            allowed = any(path.is_relative_to(root) for root in allowed_images)
            if not allowed or path.suffix.lower() not in IMAGE_TYPES or not path.is_file():
                abort(404)
        except (ValueError, OSError, RuntimeError):
            abort(404)
        return send_file(path, conditional=True)

    return app


def main():
    parser = argparse.ArgumentParser(description="Browse IntReal benchmarks, responses, and reports")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository containing benchmark/, outputs/, and results/")
    parser.add_argument("--image-root", type=Path, action="append", default=[], help="Additional allowed image directory; may be repeated")
    args = parser.parse_args()
    create_app(args.root, args.image_root).run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
