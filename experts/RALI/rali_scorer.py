# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vendored ClipScorer from RALI repo for self-contained usage."""

import os
from typing import List

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Patch CLIPModel.get_image_features for transformers>=4.56 compatibility.
_CLIP_GET_IMAGE_FEATURES_ORIG = None


def _patch_clip_get_image_features_for_rali() -> None:
    global _CLIP_GET_IMAGE_FEATURES_ORIG
    if _CLIP_GET_IMAGE_FEATURES_ORIG is not None:
        return
    from transformers.models.clip import modeling_clip as clip_mod

    _CLIP_GET_IMAGE_FEATURES_ORIG = clip_mod.CLIPModel.get_image_features

    def _wrapped(self, *args, **kwargs):
        out = _CLIP_GET_IMAGE_FEATURES_ORIG(self, *args, **kwargs)
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        raise TypeError(
            f"Unexpected CLIPModel.get_image_features return type: {type(out)}"
        )

    clip_mod.CLIPModel.get_image_features = _wrapped  # type: ignore[assignment]


def _restore_clip_get_image_features_patch() -> None:
    global _CLIP_GET_IMAGE_FEATURES_ORIG
    if _CLIP_GET_IMAGE_FEATURES_ORIG is None:
        return
    from transformers.models.clip import modeling_clip as clip_mod

    clip_mod.CLIPModel.get_image_features = _CLIP_GET_IMAGE_FEATURES_ORIG
    _CLIP_GET_IMAGE_FEATURES_ORIG = None


# Apply patch on import.
_patch_clip_get_image_features_for_rali()


# ============================================================
# PrototypeScorer
# ============================================================
class PrototypeScorer(nn.Module):
    def __init__(
        self,
        clip_emb_dim: int,
        pca_path: str = None,
        basis_path: str = None,
        num_prototypes: int = 32,
        init_std: float = 0.02,
        learnable_tau: bool = False,
        init_tau: float = 0.25,
        learnable_pca: bool = False,
        learnable_prototypes: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.clip_emb_dim = clip_emb_dim
        self.learnable_pca = learnable_pca
        self.learnable_tau = learnable_tau
        self.learnable_prototypes = learnable_prototypes

        self.pca_dim = clip_emb_dim
        pca_mean = np.zeros(clip_emb_dim)
        pca_scale = np.ones(clip_emb_dim)
        pca_components = np.eye(clip_emb_dim)

        if pca_path and os.path.exists(pca_path):
            try:
                pca = joblib.load(pca_path)
                pca_mean = pca.mean_ if hasattr(pca, "mean_") else pca_mean
                pca_scale = np.sqrt(pca.var_) if hasattr(pca, "var_") else pca_scale
                pca_components = (
                    pca.components_ if hasattr(pca, "components_") else pca_components
                )
                self.pca_dim = pca_components.shape[0]
                pca_scale[pca_scale < 1e-10] = 1.0
            except Exception as e:
                print(f"[Warning] Failed to load PCA, fallback to identity: {e}")

        self.pca_mean = torch.tensor(pca_mean, dtype=torch.float32, device=device)
        self.pca_scale = nn.Parameter(
            torch.tensor(pca_scale, dtype=torch.float32, device=device),
            requires_grad=learnable_pca,
        )
        self.pca_components = nn.Parameter(
            torch.tensor(pca_components, dtype=torch.float32, device=device),
            requires_grad=learnable_pca,
        )

        proto_init = torch.randn(num_prototypes, self.pca_dim) * init_std
        score_init = torch.empty(num_prototypes).uniform_(0.3, 0.7)

        if basis_path and os.path.exists(basis_path):
            try:
                data = np.load(basis_path)
                proto_init = torch.tensor(data["vectors"], dtype=torch.float32)
                score_init = torch.tensor(data["scores"], dtype=torch.float32)
                num_prototypes = proto_init.shape[0]
            except Exception as e:
                print(f"[Warning] Failed to load basis, using random init: {e}")

        self.prototypes = nn.Parameter(
            proto_init.to(device), requires_grad=learnable_prototypes
        )
        self.proto_scores = nn.Parameter(
            score_init.to(device), requires_grad=learnable_prototypes
        )

        self.logit_scale = nn.Parameter(
            torch.tensor(1.0 / max(init_tau, 1e-6), dtype=torch.float32, device=device),
            requires_grad=learnable_tau,
        )

    @property
    def tau(self):
        return 1.0 / self.logit_scale.clamp(min=1e-6)

    def pca_transform(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.pca_mean) / torch.abs(self.pca_scale)
        return torch.matmul(x_norm, self.pca_components.T)

    def forward(self, clip_feats: torch.Tensor) -> torch.Tensor:
        x_pca = self.pca_transform(clip_feats)
        x_norm = F.normalize(x_pca, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.matmul(x_norm, proto_norm.T)
        weights = F.softmax(sim / self.tau, dim=-1)
        return torch.matmul(weights, self.proto_scores)


# ============================================================
# CLIP + Prototype IQA model
# ============================================================
class CLIP_IQA_Proto(nn.Module):
    def __init__(
        self,
        clip_model_name: str,
        device: torch.device,
        pca_path: str = None,
        basis_path: str = None,
        num_prototypes: int = 32,
        tau: float = 0.07,
        learnable_tau: bool = False,
        learnable_pca: bool = False,
        learnable_prototypes: bool = True,
        prec: str = "fp32",
    ):
        super().__init__()
        self.device = device
        self.prec = prec

        self.clip = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip.requires_grad_(False)
        self.clip.eval()

        with torch.no_grad():
            dummy = Image.new("RGB", (336, 336))
            pixel = self.processor(images=[dummy], return_tensors="pt").to(device)
            clip_dim = self.clip.get_image_features(**pixel).shape[-1]

        self.scorer = PrototypeScorer(
            clip_emb_dim=clip_dim,
            pca_path=pca_path,
            basis_path=basis_path,
            num_prototypes=num_prototypes,
            init_tau=tau,
            learnable_tau=learnable_tau,
            learnable_pca=learnable_pca,
            learnable_prototypes=learnable_prototypes,
            device=device,
        )

    def encode_images(self, pil_images: List[Image.Image]):
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self.clip.get_image_features(**inputs)
        return feats

    def forward(self, pil_images: List[Image.Image]):
        feats = self.encode_images(pil_images)
        return self.scorer(feats)


# ============================================================
# ClipScorer wrapper
# ============================================================
class ClipScorer(nn.Module):
    def __init__(
        self,
        ckpt_path: str,
        clip_model: str,
        pca_path: str,
        basis_path: str,
        device: torch.device = None,
    ):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = CLIP_IQA_Proto(
            clip_model_name=clip_model,
            device=self.device,
            pca_path=pca_path,
            basis_path=basis_path,
        ).to(self.device)

        state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, pixels: torch.Tensor):
        images = []
        for x in pixels:
            img = (
                (x.clamp(0, 1) * 255)
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            images.append(Image.fromarray(img))
        return self.model(images)

    @torch.no_grad()
    def score_image(self, image: Image.Image) -> float:
        score = self.model([image])[0]
        return score.item()
