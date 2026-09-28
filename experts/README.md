# Expert Models

This document describes how to set up the expert models and their recommended locations.

We recommend using the Hugging Face CLI to download the models.

## DINOv3

We use [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) (DINOv3 ViT-L/16 pretrained on LVD-1689M).

Before downloading, request access from Meta on the model page and wait for approval. With the Hugging Face CLI installed and logged in to the approved account, run the following command from the repository root:

```bash
hf download facebook/dinov3-vitl16-pretrain-lvd1689m --local-dir experts/DINOv3
```

We recommend placing DINOv3 in `experts/DINOv3/`. If you use a different location, update the corresponding path in `config/experts.yaml`.

## RALI

Download all files from the [RALI subdirectory of ByteDance/Q-Insight](https://huggingface.co/ByteDance/Q-Insight/tree/main/RALI). With the Hugging Face CLI ready, run the following command from the repository root:

```bash
hf download ByteDance/Q-Insight --include "RALI/*" --local-dir experts
```

The command preserves the `RALI/` subdirectory, placing its contents in `experts/RALI/`, including `best/`, `basis.npz`, `ckpt.pt`, and `pca.pkl`.

Keep the existing `experts/RALI/rali_scorer.py` in the same directory: it contains our compatibility adaptations. Do not delete or overwrite it when placing the downloaded files. If you store the model files elsewhere, update the corresponding path in `config/experts.yaml` and retain the scorer at its existing location.

## VLM

Our VLM selection targets deployment on a single GPU with at least 80 GB of memory, such as an NVIDIA H100 80GB, A100 80GB, A800 80GB, H20 96GB, or a higher-memory GPU, to reduce deployment costs. Actual memory requirements depend on precision, context length, image inputs, and serving concurrency. In this paper, we use [Gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) as the default evaluator and also test evaluation results using [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The newer [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) is also worth trying.

By default, the evaluator sends API requests to a separately deployed vLLM service. This repository does not include the VLM serving environment, follow each model's official documentation linked above to download and deploy it, then configure the API connection in `config/experts.yaml`.
