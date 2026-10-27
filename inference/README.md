# Inference for IntReal

## BAGEL

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


## SenseNova-U1: IntReal SFT format only

`sensenova_u1.py` is a standalone adapter for IntReal U1 SFT / SFT-RL checkpoints.
It always enables SF reasoning and runs both SF and TF. There are no reasoning or
SF/TF mode switches. It does not support the original U1 `<imageN>` answer format.
The SFT protocol is a global `<think>` plan, answer text, and per-image
`<image_think>` descriptions. Exported answers omit reasoning and place images at
the backend's actual `<image>` generation markers; malformed alignments fail the
phase instead of silently moving images.

Clone the external backend separately (it is gitignored), without editing it:

```bash
git clone https://github.com/OpenSenseNova/SenseNova-U1.git inference/SenseNova-U1
git -C inference/SenseNova-U1 checkout 2d77fdd8ac9aadafe3fb44b4e20e17a42ab2f6b7
```

For an offline development machine, clone and checkout on a networked machine,
a sparse clone can omit unrelated assets and history blobs:

```bash
git clone --filter=blob:none --no-checkout https://github.com/OpenSenseNova/SenseNova-U1.git SenseNova-U1
git -C SenseNova-U1 sparse-checkout set src
git -C SenseNova-U1 checkout 2d77fdd8ac9aadafe3fb44b4e20e17a42ab2f6b7
```

If the destination has an older Git that rejects `worktreeConfig`, move the sparse
settings to the ordinary local config on the networked machine before transfer:

```bash
git -C SenseNova-U1 config --local core.sparseCheckout true
git -C SenseNova-U1 config --local core.sparseCheckoutCone true
git -C SenseNova-U1 config --local --unset extensions.worktreeConfig
git -C SenseNova-U1 config --local core.repositoryformatversion 0
```

Transfer this checkout, including `.git`, into `inference/SenseNova-U1`. The root
metadata and all `src` files are materialized before transfer. Other omitted
paths cannot be checked out offline without fetching their blobs first.
Use the upstream SenseNova-U1 inference environment, separately from IntReal's
scoring environment. The adapter imports the backend from this directory rather
than any editable installation elsewhere. Override with `--sensenova-repo`.
`--model-path` must be a complete local HF checkpoint with configuration,
tokenizer and all weight shards. The backend is checked for tracked modifications.
No backend source edits or runtime method patches are made by the adapter.

From the IntReal repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python inference/sensenova_u1.py \
  --model-path ../IntReal-U1-SFT \
  --benchmark benchmark/IntReal.jsonl --image-root . \
  --output-dir outputs --model-name intreal-u1-sft \
  --num-workers 2 --gpus-per-model 4
```

Defaults: fixed 1024 x 1024 output, 50 diffusion steps, CFG 4.0/1.0,
CFG interval [0, 1], no CFG renormalization, timestep shift 3.0, `t_eps=0.02`,
greedy text generation with an 8192-token total SF text budget, and seed 42.
Reference images are capped at a 1536-pixel long side before backend preprocessing.
The upstream SF implementation hardcodes a 10-image limit; no ineffective image
limit option is exposed here. TF sets its limit to the actual GT image count.
SF and TF share diffusion parameters, including `t_eps` and `cfg_norm`.
TF retains the SFT system prompt and the upstream empty think prefix; it does not
generate reasoning and conditions on GT text and preceding GT images.

`--num-workers` controls independent model replicas, each using
`--gpus-per-model` visible GPUs. Multiple-GPU replicas shard decoder layers while
keeping vision, flow and other modules on GPU 0, using the backend loader's
`device_map` support. GPU 0 needs additional headroom. The default is one worker
on one GPU. Full-weight GPU execution and multi-GPU memory fit must be validated
on the target hardware; CPU-only adapter checks do not establish those properties.

Outputs follow the same `outputs/MODEL/BENCHMARK.jsonl` layout as BAGEL. Each
phase is saved atomically with an explicit status. Use `--resume` with matching
settings to reuse completed phases and retry failures. Missing output images
invalidate the affected phase. Checkpoint shard metadata, asset content hashes,
backend revision and adapter hash are recorded; worker count can change on resume.
Seeds are derived separately for each sample and phase. This isolates scheduling
and SF/TF RNG state but does not promise bitwise equality across hardware.

The backend does not return a stopping reason. `sf_finish_reason` therefore says
`not_reported_by_backend`; `sf_image_cap_reached` is an additional diagnostic, not
a guarantee that an answer is complete. Raw SF text is retained. Exit code 2 means
one or more phases failed; setup failures also exit nonzero.

For a smoke run use `--sample-ids 302 --num-steps 2` with a separate output/model
directory. Then evaluate using the IntReal scoring environment and a matching
subset benchmark. Reduced steps are not suitable for reported benchmark scores.
