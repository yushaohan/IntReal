# Benchmark

The evaluation records are included in `benchmark/IntReal.jsonl`. The images will be available in the Hugging Face dataset [yushaohan/IntReal-100K](https://huggingface.co/datasets/yushaohan/IntReal-100K) (not uploaded yet).

Once the dataset is available, download and extract the evaluation images from the repository root:

```bash
hf download yushaohan/IntReal-100K archives/eval.tar --repo-type dataset --local-dir .
tar -xmf archives/eval.tar
```

The archive already contains the `images/eval/` directory structure. Extract it at the repository root so that image paths match the JSONL records. Images and downloaded archives are excluded from Git; the benchmark JSONL is included.
