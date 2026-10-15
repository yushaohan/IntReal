# BAGEL inference for IntReal

`bagel.py` produces both self-feeding (SF) and teacher-forcing (TF) responses for the IntReal evaluator. BAGEL source code and model weights are external dependencies and are not distributed in this repository.

## Setup

From the IntReal repository root:

```bash
git clone https://github.com/ByteDance-Seed/Bagel.git inference/Bagel
git -C inference/Bagel checkout a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f
```

Follow BAGEL's installation instructions in a separate inference environment. The tested environment uses Python 3.10, PyTorch 2.5.1, Transformers 4.49.0, Accelerate 1.12.0 and FlashAttention 2.8.3. IntReal's scoring environment does not need these inference dependencies. `inference/Bagel/` is gitignored.

Download the BAGEL model assets separately. `--model-path` must contain the tokenizer, `llm_config.json`, `vit_config.json`, `ae.safetensors`, and (by default) `ema.safetensors`. Use `--checkpoint` for a different fine-tuned weights file or Accelerate-compatible sharded checkpoint directory.

## Generate

Run from the IntReal repository root, using the BAGEL environment:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python inference/bagel.py \
  --model-path /path/to/model-assets \
  --checkpoint /path/to/finetuned/ema.safetensors \
  --benchmark benchmark/IntReal.jsonl \
  --image-root . \
  --output-dir outputs \
  --model-name bagel-model \
  --reasoning off --num-workers 8
```

The backend defaults to `Bagel/` beside the script, independent of the working directory. Override it with `--bagel-repo /path/to/Bagel`. Other relative command-line paths are relative to the working directory. Reference and golden image paths in the benchmark are resolved against `--image-root`.

Select `--reasoning on` for a checkpoint trained with full reasoning, or `off` for a checkpoint trained without reasoning. This selects the prompt protocol; it does not disable reasoning learned by a reasoning checkpoint and is not adaptive reasoning. TF always conditions on golden text and preceding golden images, without generated reasoning. SF feeds back its own generated content.

Both SF and TF always run. Defaults are 1024 x 1024 images, 50 diffusion steps, greedy text decoding, seed 42, a 2048-token text budget and a 12-image safety cap. The image cap does not force the output count to match the benchmark. With `--num-workers 8`, eight independent processes each load one model on one GPU and consume prompts from a shared queue, with batch size 1. The parent alone writes JSONL results after each phase. Completed phases can be resumed when changing worker count. With the default `--num-workers 1`, one model is partitioned across visible GPUs and samples run sequentially. `--gpu-memory-gib` defaults to 60 per GPU; CPU/disk model offload is not enabled.

Use `--sample-ids 302 --num-timesteps 10` for a smoke run in a separate output directory. Reduced diffusion steps are for validation, not final benchmark reporting.

## Outputs, resume and scoring

Outputs are written to `outputs/bagel-model/IntReal.jsonl` with `response_sf` and `response_tf`; PNG files are under the same model directory's `images/`. Image paths are absolute. Additional fields retain raw SF text, the stopping reason, generation settings and phase errors. Exported answer text excludes reasoning markers and reasoning content.

Add `--resume` with the same arguments to reuse completed phases and retry failed phases. Configuration mismatches are rejected. Each sample and phase has an independent deterministic seed. Exit code 0 means no phase failed, 2 means one or more generation phases failed, and uncaught setup/configuration errors exit nonzero. Check `sf_finish_reason` for truncation; successful execution does not imply output quality or a correct image count.

Switch to the IntReal scoring environment and configure its VLM/expert environment variables, then run:

```bash
python evaluate.py --model bagel-model --benchmark benchmark/IntReal.jsonl --output-dir outputs
```

For a custom benchmark, pass the same JSONL path to inference and evaluation. A smoke subset must be scored against a matching subset benchmark, not treated as a complete full-benchmark result.

`load_generator(...)` and `run_benchmark(...)` can also be imported. The runner accepts a generator implementing `generate_sf` and `generate_tf`; training-framework lifecycle and distributed/FSDP integration belong in the caller.
