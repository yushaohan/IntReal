<div align="center">

# IntReal: A Real-World Dataset and Benchmark for Multimodal Interleaved Generation

[Shaohan Yu](https://yushaohan.github.io/)<sup>1,2</sup>,
[Qin Jie](https://qinjie-ai.top/)<sup>2</sup>,
[Wende Tan](https://scholar.google.com/citations?user=IDvZ0CcAAAAJ)<sup>3</sup>,
[Haiwen Diao](https://paranioar.github.io/)<sup>4</sup>,
[Limeng Qiao](https://scholar.google.com/citations?user=3PFZAg0AAAAJ&hl=en&oi=ao)<sup>2</sup>,
[Yueming Lyu](https://yueming-ai.top/)<sup>1</sup>,
[Chenyang Si](http://chenyangsi.top/)<sup>1,†</sup>

<sup>1</sup> [PRLab, Nanjing University](https://prlab-nju.com/) &nbsp;
<sup>2</sup> [Meituan LongCat](https://longcat.ai/)
<br>
<sup>3</sup> [Imperial College London](https://www.imperial.ac.uk/) &nbsp;
<sup>4</sup> [S-Lab, Nanyang Technological University](https://www.ntu.edu.sg/s-lab)

📄 Paper (Coming Soon) · 🌐 Project Page (Coming Soon) · 🤗 Models & Dataset (Coming Soon)

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
INTREAL_API_MODEL=gemma4
INTREAL_API_BASE_URL=http://localhost:8000/v1
INTREAL_API_KEY=EMPTY
EOF
```

Use the model name exposed by your service and replace the URL with its actual address. If your service requires authentication, replace `EMPTY` with its API key. The `.env` file is excluded from Git.

Run the bundled single-sample test:

```bash
uv run --locked --env-file .env python evaluate.py --model test --benchmark test
```

If the environment is configured correctly, the evaluation results are saved to `result/test/test.jsonl`.

## 📧 Contact

For a faster response, we recommend contacting the first author Shaohan at [shaohan.yu@smail.nju.edu.cn](mailto:shaohan.yu@smail.nju.edu.cn). If you prefer to communicate in Chinese, you can also reach him on WeChat: `SeanShaohanYu`. Alternatively, you may contact the corresponding author Chenyang at [chenyang.si@nju.edu.cn](mailto:chenyang.si@nju.edu.cn).

## 📑 Citation
Coming Soon