# Models

## Bagel

Follow the [official Bagel repository](https://github.com/ByteDance-Seed/Bagel) to configure the inference environment. Place its source code in `inference/Bagel/`.

Prepare the model assets following the official instructions, and download an IntReal checkpoint from our [Model Zoo](../README.md#-model-zoo). The model directory should contain the configuration files, tokenizer, VAE, and `ema.safetensors`. Use `--checkpoint` if the fine-tuned weights are stored separately.

## SenseNova-U1

Follow the [official SenseNova-U1 repository](https://github.com/OpenSenseNova/SenseNova-U1) to configure the inference environment. Place its source code in `inference/SenseNova-U1/`.

Download a complete IntReal U1 SFT or SFT-RL checkpoint from our [Model Zoo](../README.md#-model-zoo), including its configuration files, tokenizer, and weights.

# Inference

Run the following commands from the IntReal repository root in the corresponding model environment. Prepare the benchmark images as described in [benchmark/README.md](../benchmark/README.md), and replace the model paths with your local directories.

Both scripts run reasoning-enabled SF generation and GT-conditioned TF generation.

## Bagel

```bash
python inference/bagel.py \
  --model-path /path/to/IntReal-Bagel-SFT \
  --benchmark benchmark/IntReal.jsonl \
  --output-dir outputs \
  --model-name intreal-bagel-sft \
  --reasoning on
```

## SenseNova-U1

```bash
python inference/sensenova_u1.py \
  --model-path /path/to/IntReal-U1-8B-MoT-SFT \
  --benchmark benchmark/IntReal.jsonl \
  --output-dir outputs \
  --model-name intreal-u1-sft
```

U1 enables SF reasoning by default. Results are saved to `outputs/<model-name>/IntReal.jsonl`, with generated images in the same model directory. Add `--resume` to either command to continue an interrupted run.
