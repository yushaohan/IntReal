#!/usr/bin/env python3
"""Evaluate model outputs against a benchmark and write per-sample results."""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

from metrics import MetricsEngine


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


async def evaluate_sample(engine, bench_item, out_item, tf_item, pbar, start_time,
                          total_ref, semaphore, fout, write_lock,
                          is_real_mode=False):
    async with semaphore:
        t0 = time.time()
        sample_id = bench_item.get("id", "?")
        metrics = await engine.compute_all(bench_item, out_item, tf_item=tf_item)
        if is_real_mode:
            metrics["text_richness"] = 1.0

        result_item = dict(out_item)
        result_item["response_sf"] = result_item.pop("response", [])
        result_item["metrics"] = metrics
        if tf_item is not None:
            result_item["response_tf"] = tf_item.get("response", [])

        line = json.dumps(result_item, ensure_ascii=False) + "\n"
        async with write_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, fout.write, line)
            await loop.run_in_executor(None, fout.flush)

        total_ref[0] += 1
        elapsed = time.time() - t0
        avg_time = (time.time() - start_time) / total_ref[0]
        pbar.set_postfix(avg=f"{avg_time:.2f}s/sample", last=f"{elapsed:.2f}s", id=sample_id)
        pbar.update(1)


async def main_async(args):
    config_dir = Path(args.config_dir)
    output_dir = Path(args.output_dir)
    result_dir = Path(args.result_dir)

    with open(config_dir / "experts.yaml", "r", encoding="utf-8") as f:
        experts_config = yaml.safe_load(f)
    with open(config_dir / "metrics.yaml", "r", encoding="utf-8") as f:
        metrics_config = yaml.safe_load(f)
    with open(config_dir / "prompts.yaml", "r", encoding="utf-8") as f:
        prompts_config = yaml.safe_load(f)

    benchmark_path = Path("benchmark") / f"{args.benchmark}.jsonl"
    model_output_path = output_dir / args.model / f"{args.benchmark}.jsonl"
    result_path = result_dir / args.model / f"{args.benchmark}.jsonl"

    if not benchmark_path.exists():
        print(f"[FATAL] Benchmark not found: {benchmark_path}", file=sys.stderr)
        raise SystemExit(1)

    is_real_mode = args.model == "real"
    if is_real_mode and metrics_config.get("enabled", {}).get("text_richness", False):
        metrics_config["enabled"]["text_richness"] = False
        print("[INFO] real mode: text_richness auto-disabled to save API calls.", file=sys.stderr)
    if not is_real_mode and not model_output_path.exists():
        print(f"[FATAL] Model output not found: {model_output_path}", file=sys.stderr)
        raise SystemExit(1)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    engine = MetricsEngine(experts_config, metrics_config, prompts_config)
    print("[INFO] Pre-loading local expert models...", file=sys.stderr)
    enabled = metrics_config.get("enabled", {})
    if enabled.get("image_diversity", True) or enabled.get("subject_consistency", True):
        engine.get_dinov3()
    if enabled.get("image_aesthetic", True):
        engine.get_rali_scorer()
    print("[INFO] Models loaded.", file=sys.stderr)

    bench_items = list(load_jsonl(benchmark_path))
    if is_real_mode:
        out_items = [{"model": "real", "eval_set": args.benchmark,
                      "eval_set_id": b["id"], "response": b.get("golden_result", [])}
                     for b in bench_items]
        tf_by_id = {}
    else:
        raw_items = list(load_jsonl(model_output_path))
        has_tf = "response_tf" in raw_items[0] if raw_items else False
        if has_tf:
            tf_by_id = {}
            out_items = []
            for item in raw_items:
                out_items.append({"model": item["model"], "eval_set": item["eval_set"],
                                  "eval_set_id": item["eval_set_id"],
                                  "response": item["response_sf"]})
                tf_by_id[item["eval_set_id"]] = {"response": item["response_tf"]}
        else:
            out_items = raw_items
            tf_by_id = {}

    bench_by_id = {b["id"]: b for b in bench_items}
    out_by_id = {o["eval_set_id"]: o for o in out_items}
    common_ids = sorted(set(bench_by_id) & set(out_by_id))
    missing_in_output = sorted(set(bench_by_id) - set(out_by_id))
    missing_in_benchmark = sorted(set(out_by_id) - set(bench_by_id))
    if missing_in_output:
        print(f"[WARN] {len(missing_in_output)} benchmark samples missing in output: {missing_in_output[:20]}", file=sys.stderr)
    if missing_in_benchmark:
        print(f"[WARN] {len(missing_in_benchmark)} output samples missing in benchmark: {missing_in_benchmark[:20]}", file=sys.stderr)
    if not common_ids:
        print("[FATAL] No matching samples found between benchmark and output.", file=sys.stderr)
        raise SystemExit(1)

    already_done = set()
    if result_path.exists():
        try:
            for line in result_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    already_done.add(json.loads(line)["eval_set_id"])
        except Exception:
            pass
    if already_done:
        common_ids = [i for i in common_ids if i not in already_done]
        print(f"[INFO] Resume mode: {len(already_done)} samples already done, {len(common_ids)} remaining.", file=sys.stderr)
    if not common_ids:
        print("[DONE] All samples already evaluated. Nothing to do.", file=sys.stderr)
        return

    print(f"[INFO] Evaluating {len(common_ids)} samples. Results: {result_path}", file=sys.stderr)
    start_time = time.time()
    total_ref = [0]
    semaphore = asyncio.Semaphore(args.concurrency)
    pbar = tqdm(total=len(common_ids), desc="Evaluating", unit="sample", ncols=100)
    with open(result_path, "a", encoding="utf-8") as fout:
        write_lock = asyncio.Lock()
        tasks = [evaluate_sample(engine, bench_by_id[i], out_by_id[i], tf_by_id.get(i),
                                 pbar, start_time, total_ref, semaphore, fout, write_lock,
                                 is_real_mode)
                 for i in common_ids]
        await asyncio.gather(*tasks)
        pbar.close()

    avg_time = (time.time() - start_time) / len(common_ids)
    print(f"[DONE] Evaluated {len(common_ids)} samples, recorded {total_ref[0]}. Avg {avg_time:.2f}s/sample. Results written to {result_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="IntReal model evaluation")
    parser.add_argument("--model", required=True, help="Model name (must match outputs/ dir)")
    parser.add_argument("--benchmark", required=True, help="Benchmark name (must match benchmark/ file)")
    parser.add_argument("--config-dir", default="config", help="Path to config directory")
    parser.add_argument("--output-dir", default="outputs", help="Path to output directory")
    parser.add_argument("--result-dir", default="results", help="Path to result directory")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Evaluation concurrency for LLM API calls")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
