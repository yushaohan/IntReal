<div align="center">

# IntReal: A Real-World Dataset and Benchmark for Multimodal Interleaved Generation

[Shaohan Yu](https://yushaohan.github.io/)<sup>1,2</sup>,
[Jie Qin](https://jieqin-ai.top/)<sup>2</sup>,
[Wende Tan](https://scholar.google.com/citations?user=IDvZ0CcAAAAJ)<sup>3</sup>,
[Haiwen Diao](https://paranioar.github.io/)<sup>4</sup>,
[Limeng Qiao](https://scholar.google.com/citations?user=3PFZAg0AAAAJ&hl=en&oi=ao)<sup>2</sup>,
[Yueming Lyu](https://yueming-ai.top/)<sup>1</sup>,
[Chenyang Si](http://chenyangsi.top/)<sup>1,†</sup>

<sup>1</sup> [PRLab, Nanjing University](https://prlab-nju.com/nju/) &nbsp;
<sup>2</sup> [Meituan](https://longcat.ai/)
<br>
<sup>3</sup> [Imperial College London](https://www.imperial.ac.uk/) &nbsp;
<sup>4</sup> [S-Lab, Nanyang Technological University](https://www.ntu.edu.sg/s-lab)

📄 Paper (Coming Soon) · 🌐 [Project Page](https://yushaohan.github.io/IntReal/) · 🤗 [Models & Dataset](https://huggingface.co/collections/yushaohan/intreal)

</div>

## 🔥 News & Updates
Coming Soon

## 🚀 Quick Start

IntReal evaluation can run on a single common GPU with 80 GB of memory, such as an NVIDIA A100 80GB or H100 80GB. Before evaluation, complete the following preparation steps. Run all commands from the repository root.

1. **Prepare the environment with uv.** Install the project dependencies with `uv sync --locked`, then activate the environment with `source .venv/bin/activate`.

2. **Prepare the benchmark images.** Follow [benchmark/README.md](benchmark/README.md) to download the evaluation images from Hugging Face and extract them into the expected directory.

3. **Prepare the expert models.** Follow [experts/README.md](experts/README.md) to download DINOv3 and RALI to the recommended locations. You also need to download and deploy a VLM as an API service for judging. The VLM serving environment is separate from this project's evaluation environment.

Once everything is ready, create a `.env` file in the repository root with your deployed model name, API base URL, and API key:

```bash
cat > .env <<'EOF'
export INTREAL_API_MODEL=gemma4
export INTREAL_API_BASE_URL=http://localhost:8000/v1
export INTREAL_API_KEY=EMPTY
EOF
```

Use the model name exposed by your service and replace the URL with its actual address. If your service requires authentication, replace `EMPTY` with its API key. The `.env` file is excluded from Git. `evaluate.py` first checks the three variables in the shell environment, then automatically fills missing or empty values from the `.env` beside the script. Existing non-empty shell values take precedence. If any required value is still missing or empty, evaluation exits with an error before loading models.

Run the bundled single-sample test:

```bash
python evaluate.py --model test --benchmark test
```

If the environment is configured correctly, the evaluation results are saved to `results/test/test.jsonl`.

## 🦁 Model Zoo

Our released models are available in the [IntReal Hugging Face collection](https://huggingface.co/collections/yushaohan/intreal).

| Model family | SFT | SFT + RL |
| --- | --- | --- |
| Bagel | [IntReal-Bagel-SFT](https://huggingface.co/yushaohan/IntReal-Bagel-SFT) | [IntReal-Bagel-SFT-RL](https://huggingface.co/yushaohan/IntReal-Bagel-SFT-RL) |
| SenseNova-U1-8B-MoT | [IntReal-U1-8B-MoT-SFT](https://huggingface.co/yushaohan/IntReal-U1-8B-MoT-SFT) | [IntReal-U1-8B-MoT-SFT-RL](https://huggingface.co/yushaohan/IntReal-U1-8B-MoT-SFT-RL) |

## 🤖 Inference

We provide inference scripts for **Bagel** and **SenseNova-U1**, covering both self-forcing (SF) interleaved generation and teacher-forcing (TF) image generation conditioned on ground-truth context.

See [inference/README.md](inference/README.md) for backend installation, checkpoint preparation, generation commands, multi-GPU execution, and resuming interrupted runs. The scripts save outputs in the format expected by the evaluation pipeline below.

## 🔍 Evaluation

After generation, save model outputs as `outputs/<model_name>/<benchmark_name>.jsonl`, with one JSON object per sample:

```text
outputs/
└── my-model/
    ├── IntReal.jsonl
    └── images/
        ├── sf_0001.png
        └── tf_0001.png
```

Each record stores the sample ID and an ordered sequence of text and image blocks. For example (formatted across multiple lines for readability):

```json
{
  "model": "my-model",
  "eval_set": "IntReal",
  "eval_set_id": 1,
  "response_sf": [
    {"type": "text", "content": "The first step is..."},
    {"type": "image", "content": "outputs/my-model/images/sf_0001.png"},
    {"type": "text", "content": "Next..."}
  ],
  "response_tf": [
    {"type": "image", "content": "outputs/my-model/images/tf_0001.png"}
  ]
}
```

`eval_set_id` must match the sample's `id` in `benchmark/<benchmark_name>.jsonl`. Image paths must point to local files; relative paths are resolved from the repository root. Preserve the generation order of the blocks.

`response_sf` corresponds to **holistic evaluation** in our paper. Since this setting involves open-ended generation, we refer to it as **self-forcing (SF)** by analogy. `response_tf` corresponds to **fine-grained evaluation**, where generation is conditioned on ground-truth context; we refer to this setting as **teacher-forcing (TF)** by analogy.

Activate the environment and export the API settings from `.env`, then run evaluation from the repository root:

```bash
source .venv/bin/activate
python evaluate.py --model my-model --benchmark IntReal
```

The results are written to `results/my-model/IntReal.jsonl`.

Optional arguments:

| Argument | Default | Description |
| --- | --- | --- |
| `--concurrency` | `20` | Maximum number of samples evaluated concurrently. |
| `--config-dir` | `config` | Directory containing `experts.yaml`, `metrics.yaml`, and `prompts.yaml`. |
| `--output-dir` | `outputs` | Root directory of model outputs; the runner reads `<output-dir>/<model>/<benchmark>.jsonl`. |
| `--result-dir` | `results` | Root directory for evaluation results. |

For example: `python evaluate.py --model my-model --benchmark IntReal --concurrency 4 --result-dir results-new`.


## 🖼️ Viewer

Browse benchmarks, compare reference and SF/TF responses, and export evaluation reports as CSV:

```bash
source .venv/bin/activate
python -m viewer.app
```

Open `http://127.0.0.1:8420`. The viewer reads `benchmark/`, `outputs/`, and `results/` without loading models or calling APIs.

## 🙏 Acknowledgement

We thank the teams behind [Bagel](https://github.com/ByteDance-Seed/Bagel), [Emu3.5](https://github.com/baaivision/Emu3.5), and [SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1) for their contributions to open-source unified multimodal models (UMMs). We also thank the [Flow-Factory](https://github.com/X-GenGroup/Flow-Factory) team for providing the reinforcement learning training framework, and the [SEED-Story](https://github.com/TencentARC/SEED-Story) team for collecting and releasing interleaved image-text storytelling data.

## ©️ Licence

- **Code:** Released under the [Apache License 2.0](LICENSE). Third-party components remain subject to their respective licences.
- **Dataset:** Please refer to the licence and usage terms in the [IntReal-100K dataset card](https://huggingface.co/datasets/yushaohan/IntReal-100K) on Hugging Face.
- **Models:** Please refer to the licence and usage terms in each model card in our [Hugging Face collection](https://huggingface.co/collections/yushaohan/intreal).

If you plan to use IntReal for commercial purposes, please let me know at [shaohan.yu@smail.nju.edu.cn](mailto:shaohan.yu@smail.nju.edu.cn).

## 📧 Contact

For a faster response, we recommend contacting the first author Shaohan at [shaohan.yu@smail.nju.edu.cn](mailto:shaohan.yu@smail.nju.edu.cn). If you prefer to communicate in Chinese, you can also reach him on WeChat: `SeanShaohanYu`. Alternatively, you may contact the corresponding author Chenyang at [chenyang.si@nju.edu.cn](mailto:chenyang.si@nju.edu.cn).

## 📑 Citation
Coming Soon
