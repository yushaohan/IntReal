"""MetricsEngine: lazily loads experts and computes all metrics."""

import asyncio
import sys
import traceback
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


from . import context_fidelity, image_aesthetic, image_diversity, quantity_compliance, subject_consistency


def _tensor_from_image_model_output(out):
    """Normalize get_image_features / vision forward outputs to a 2D float tensor (B, D)."""
    if isinstance(out, torch.Tensor):
        return out
    pooler = getattr(out, "pooler_output", None)
    if pooler is not None:
        return pooler
    lhs = getattr(out, "last_hidden_state", None)
    if lhs is not None:
        return lhs[:, 0]
    for name in ("image_embeds",):
        emb = getattr(out, name, None)
        if emb is not None:
            return emb
    raise TypeError(f"Cannot extract image feature tensor from output type {type(out)!r}")


class MetricsEngine:
    def __init__(self, experts_config, metrics_config, prompts_config):
        self.experts_config = experts_config
        self.metrics_config = metrics_config
        self.prompts_config = prompts_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._dinov3_processor = None
        self._dinov3_model = None
        self._dinov3_device = None
        self._rali_scorer = None
        self._openai_clients = {}

    def get_dinov3(self):
        if self._dinov3_model is None:
            local_dir = Path(self.experts_config["local"]["dino"]["path"])
            device = self.experts_config["local"]["dino"].get("cuda", self.device)
            processor = AutoImageProcessor.from_pretrained(
                local_dir, local_files_only=True, backend="torchvision"
            )
            model = AutoModel.from_pretrained(
                local_dir, local_files_only=True, dtype=torch.float32
            )
            model.to(device)
            model.eval()
            self._dinov3_processor = processor
            self._dinov3_model = model
            self._dinov3_device = device
        return self._dinov3_processor, self._dinov3_model

    def get_rali_scorer(self):
        if self._rali_scorer is None:
            from experts.RALI.rali_scorer import ClipScorer

            rali_cfg = self.experts_config["local"]["aesthetic-predictor"]
            rali_path = Path(rali_cfg["path"])
            device = torch.device(rali_cfg.get("cuda", self.device))
            self._rali_scorer = ClipScorer(
                ckpt_path=str(rali_path / "ckpt.pt"),
                clip_model=str(rali_path / "best"),
                pca_path=str(rali_path / "pca.pkl"),
                basis_path=str(rali_path / "basis.npz"),
                device=device,
            )
        return self._rali_scorer

    def extract_dino_features(self, image_paths):
        processor, model = self.get_dinov3()

        if not image_paths:
            return []

        device = self._dinov3_device or self.device
        images = [Image.open(path).convert("RGB") for path in image_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model(**inputs)
        feat = out.pooler_output if hasattr(out, "pooler_output") else out.last_hidden_state[:, 0]
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return [f.cpu() for f in feat]

    async def compute_all(self, benchmark_item, output_item, tf_item=None) -> dict:
        """Compute all enabled metrics concurrently.

        Local-expert metrics are still sync (they are fast once batched);
        LLM metrics are dispatched concurrently.

        tf_item: optional separate output item whose response contains
        teacher-forcing images (used by context_fidelity / subject_consistency).
        """
        enabled = self.metrics_config.get("enabled", {})
        results = {}

        async def _run_llm(metric_name, module_name, **extra):
            mod = __import__(f"metrics.{module_name}", fromlist=["compute_async"])
            try:
                return metric_name, await mod.compute_async(self, benchmark_item, output_item, **extra)
            except Exception as e:
                print(f"[ERROR] {metric_name} failed for id={benchmark_item.get('id')}: {e}")
                traceback.print_exc()
                return metric_name, None

        loop = asyncio.get_running_loop()

        def _local_aesthetic():
            try:
                return "image_aesthetic", image_aesthetic.compute(self, benchmark_item, output_item)
            except Exception as e:
                print(f"[ERROR] image_aesthetic failed for id={benchmark_item.get('id')}: {e}")
                traceback.print_exc()
                return "image_aesthetic", None

        def _local_diversity():
            try:
                return "image_diversity", image_diversity.compute(self, benchmark_item, output_item)
            except Exception as e:
                print(f"[ERROR] image_diversity failed for id={benchmark_item.get('id')}: {e}")
                traceback.print_exc()
                return "image_diversity", None

        def _local_quantity():
            try:
                return "quantity_compliance", quantity_compliance.compute(self, benchmark_item, output_item)
            except Exception as e:
                print(f"[ERROR] quantity_compliance failed for id={benchmark_item.get('id')}: {e}")
                traceback.print_exc()
                return "quantity_compliance", None

        def _local_subject():
            try:
                return "subject_consistency", subject_consistency.compute(self, benchmark_item, output_item, tf_item=tf_item)
            except Exception as e:
                print(f"[ERROR] subject_consistency failed for id={benchmark_item.get('id')}: {e}")
                traceback.print_exc()
                return "subject_consistency", None

        coros = []
        if enabled.get("text_richness", True):
            coros.append(_run_llm("text_richness", "text_richness"))
        if enabled.get("text_logicality", True):
            coros.append(_run_llm("text_logicality", "text_logicality"))
        if enabled.get("image_aesthetic", True):
            coros.append(loop.run_in_executor(None, _local_aesthetic))
        if enabled.get("image_diversity", True):
            coros.append(loop.run_in_executor(None, _local_diversity))
        if enabled.get("quantity_compliance", True):
            coros.append(loop.run_in_executor(None, _local_quantity))
        if enabled.get("context_fidelity", True):
            coros.append(_run_llm("context_fidelity", "context_fidelity", tf_item=tf_item))
        if enabled.get("subject_consistency", True):
            coros.append(loop.run_in_executor(None, _local_subject))

        for coro in asyncio.as_completed(coros):
            name, value = await coro
            results[name] = value

        # CF / SC return per-position lists; keep mean as the scalar metric.
        for name in ("context_fidelity", "subject_consistency"):
            value = results.get(name)
            if isinstance(value, list):
                results[f"{name}_per_pos"] = value
                results[name] = (sum(value) / len(value)) if value else 0.0

        return results
