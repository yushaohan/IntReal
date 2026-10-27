#!/usr/bin/env python3
"""IntReal SFT-only SenseNova-U1 inference: reasoning SF and GT-conditioned TF."""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import queue
import random
import re
import sys

LOG = logging.getLogger('intreal.u1')
SYSTEM = '''You are a multimodal instruction assistant that interleaves step-by-step text with generated images. When responding to user requests, follow these rules:
1. Begin with a high-level plan inside <think>...</think> tags. Outline the logical sequence and identify which steps require visual illustration.
2. Provide instructions in clear, concise, professional prose.
3. When a step benefits from visual aid, enclose the image description inside <image_think>...</image_think> tags. The description must be vivid, specific, and technically precise. Every <image_think> block must end with </image_think> and be immediately followed by an image.
4. Alternate naturally between text and images. Only generate images where visual representation genuinely clarifies a complex step, state, or outcome.
5. Maintain seamless transitions: guide the user logically from text to image and back to text.
6. You may conclude with a final text paragraph without generating an image.'''
SF_IMAGE_CAP = 10


def parse_sft(text, images):
    """Match the original SFT parser: split at image_think blocks."""
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.S)
    parts = re.split(r'<image_think>.*?</image_think>', clean, flags=re.S)
    blocks, image_index = [], 0
    for index, part in enumerate(parts):
        part = re.sub(r'\n{3,}', '\n\n', part.strip()).strip()
        part = part.removeprefix('<image>').strip()
        if part:
            blocks.append(part)
        if index < len(parts) - 1 and image_index < len(images):
            blocks.append(images[image_index])
            image_index += 1
    blocks.extend(images[image_index:])
    if not blocks:
        clean = re.sub(r'<image_think>.*?</image_think>', '', clean, flags=re.S)
        blocks.append(re.sub(r'\n{3,}', '\n\n', clean).strip())
    return blocks


@contextlib.contextmanager
def seeded(seed):
    import numpy as np
    import torch
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def read_image(path, max_long_side=0):
    from PIL import Image
    with Image.open(path) as image:
        image = image.convert('RGB').copy()
    if max_long_side and max(image.size) > max_long_side:
        scale = max_long_side / max(image.size)
        image = image.resize(tuple(max(1, round(x * scale)) for x in image.size), Image.Resampling.LANCZOS)
    return image


def to_pil(tensor):
    from PIL import Image
    pixels = tensor.detach().float().cpu()
    if pixels.ndim != 4 or pixels.shape[:2] != (1, 3):
        raise ValueError(f'Unexpected generated image shape: {pixels.shape}')
    pixels = ((pixels[0] * 0.5 + 0.5).clamp(0, 1) * 255).permute(1, 2, 0).numpy().astype('uint8')
    return Image.fromarray(pixels)


class SFTGenerator:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer

    def generate(self, sample, phase, settings, seed):
        from transformers import GenerationConfig
        images = [read_image(p, settings['max_input_image_long_side']) for p in sample['prompt'].get('images', [])]
        prompt = sample['prompt']['text']  # Preserve reference-image positions.
        kwargs = dict(image_size=(settings['width'], settings['height']),
                      cfg_scale=settings['cfg_scale'], img_cfg_scale=settings['img_cfg_scale'],
                      cfg_norm=settings['cfg_norm'], timestep_shift=settings['timestep_shift'],
                      cfg_interval=tuple(settings['cfg_interval']), num_steps=settings['num_steps'],
                      t_eps=settings['t_eps'], system_message=SYSTEM, verbose=False)
        with seeded(seed):
            if phase == 'sf':
                raw, tensors = self.model.interleave_gen(
                    self.tokenizer, prompt, images=images, think_mode=True, seed=seed, max_images=SF_IMAGE_CAP,
                    generation_config=GenerationConfig(max_new_tokens=settings['max_text_tokens']), **kwargs)
                return parse_sft(raw, [to_pil(t) for t in tensors]), {
                    'sf_raw_text': raw,
                    # The upstream API does not report EOS vs token/image-limit termination.
                    'sf_finish_reason': 'not_reported_by_backend',
                    'sf_image_cap_reached': len(tensors) >= SF_IMAGE_CAP,
                }
            gt_images, parts = [], []
            for block in sample['golden_result']:
                if len(gt_images) >= SF_IMAGE_CAP:
                    break
                if block['type'] == 'text':
                    if '<image>' in block['content']:
                        raise ValueError('GT text contains reserved <image> marker')
                    parts.append(block['content'])
                else:
                    parts.append('<image>')
                    gt_images.append(read_image(block['content']))
            if not gt_images:
                return [], {}
            tensors = self.model.interleave_gen_image_only(
                self.tokenizer, prompt, ''.join(parts), images=images, gt_images=gt_images,
                max_images=len(gt_images), **kwargs)
            if len(tensors) != len(gt_images):
                raise ValueError(f'TF returned {len(tensors)} images; expected {len(gt_images)}')
            return [to_pil(t) for t in tensors], {}


def patch_cross_device_flow(model):
    """Match SenseNova-U1/interleaved_gen_sft.py's multi-GPU flow handling."""
    import types
    original = model._t2i_predict_v

    def predict(self, input_embeds, indexes_image, attn_mask, past_key_values,
                t, z, image_token_num, timestep_embeddings=None, image_size=None):
        device = next(self.fm_modules['fm_head'].parameters()).device
        output_device = z.device
        z, t = z.to(device), t.to(device)
        if timestep_embeddings is not None:
            timestep_embeddings = timestep_embeddings.to(device)
        return original(input_embeds, indexes_image, attn_mask, past_key_values,
                        t, z, image_token_num, timestep_embeddings, image_size).to(output_device)

    model._t2i_predict_v = types.MethodType(predict, model)


def load_generator(repo, model_path, attn_backend='auto', gpus_per_model=1):
    """Use the original SFT loader and its cross-device flow fix for multi-GPU."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('U1 inference requires a CUDA GPU')
    sys.path.insert(0, str(Path(repo) / 'src'))
    import sensenova_u1
    if not Path(sensenova_u1.__file__).resolve().is_relative_to(Path(repo).resolve()):
        raise RuntimeError('A different sensenova_u1 package was already imported')
    sensenova_u1.set_attn_backend(attn_backend)
    from sensenova_u1.utils import load_model_and_tokenizer
    if gpus_per_model > 1:
        model, tokenizer = load_model_and_tokenizer(model_path, dtype=torch.bfloat16, device_map='auto')
        patch_cross_device_flow(model)
        LOG.info('Model device map: %s', getattr(model, 'hf_device_map', None))
    else:
        model, tokenizer = load_model_and_tokenizer(model_path, dtype=torch.bfloat16, device='cuda:0')
    for name in ('interleave_gen', 'interleave_gen_image_only'):
        if not callable(getattr(model, name, None)):
            raise TypeError(f'Backend missing {name}')
    return SFTGenerator(model, tokenizer)


def load_samples(benchmark, image_root):
    samples = [json.loads(line) for line in Path(benchmark).read_text().splitlines() if line.strip()]
    ids = [s['id'] for s in samples]
    if not samples or any(type(sid) is not int for sid in ids) or len(set(ids)) != len(ids):
        raise ValueError('Benchmark requires nonempty, unique integer IDs')
    for sample in samples:
        prompt = sample['prompt']
        if not isinstance(prompt['text'], str) or not prompt['text'].strip():
            raise ValueError('Empty benchmark prompt')
        prompt['images'] = [str((Path(image_root) / p).resolve()) for p in prompt.get('images', [])]
        if prompt['text'].count('<image>') > len(prompt['images']):
            raise ValueError('Prompt has more image markers than reference images')
        paths = list(prompt['images'])
        for block in sample['golden_result']:
            if block['type'] == 'image':
                block['content'] = str((Path(image_root) / block['content']).resolve())
                paths.append(block['content'])
            elif block['type'] != 'text' or not isinstance(block['content'], str):
                raise ValueError('Unsupported golden block')
        for path in paths:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    return samples


def phase_complete(row, sample, phase):
    if not row or row.get(phase + '_status') != 'complete':
        return False
    blocks = row.get('response_' + phase, [])
    if any(not Path(b['content']).is_file() for b in blocks if b['type'] == 'image'):
        return False
    if phase == 'tf':
        return len(blocks) == min(SF_IMAGE_CAP, sum(b['type'] == 'image' for b in sample['golden_result'])) and all(b['type'] == 'image' for b in blocks)
    return bool(blocks)


def read_rows(path, samples, model_name, eval_set, settings, resume):
    rows = {}
    if not path.exists():
        return rows
    if not resume:
        raise FileExistsError(f'Use --resume or another output directory: {path}')
    for line in path.read_text().splitlines():
        row = json.loads(line)
        sid = row['eval_set_id']
        # Older results included an informational protocol label.
        generation = row.get('generation')
        if isinstance(generation, dict):
            generation.pop('protocol', None)
        if sid in rows or row.get('generation') != settings or row['model'] != model_name or row['eval_set'] != eval_set:
            raise ValueError('Duplicate ID or incompatible resume configuration')
        rows[sid] = row
    if set(rows) - {s['id'] for s in samples}:
        raise ValueError('Existing output contains IDs outside the benchmark selection')
    return rows


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(rows[k], ensure_ascii=False) + '\n' for k in sorted(rows)))
    temporary.replace(path)


def run_sample(generator, sample, row, output_path, model_name, eval_set, settings, publish):
    from PIL import Image
    sid = sample['id']
    row = row or dict(model=model_name, eval_set=eval_set, eval_set_id=sid,
                      response_sf=[], response_tf=[], generation=settings)
    failures = 0
    for phase in ('sf', 'tf'):
        if phase_complete(row, sample, phase):
            continue
        key = 'response_' + phase
        try:
            LOG.info('id=%s starting %s', sid, phase)
            blocks, metadata = generator.generate(sample, phase, settings, settings['seed'])
            expected = min(SF_IMAGE_CAP, sum(b['type'] == 'image' for b in sample['golden_result']))
            if phase == 'tf' and (len(blocks) != expected or any(not isinstance(b, Image.Image) for b in blocks)):
                raise ValueError('TF output does not match golden image positions')
            if phase == 'sf' and not blocks:
                raise ValueError('Empty SF response')
            saved, image_count = [], 0
            directory = output_path.parent / 'images' / f'{sid:06}'
            directory.mkdir(parents=True, exist_ok=True)
            for block in blocks:
                if isinstance(block, str):
                    if block.strip():
                        saved.append({'type': 'text', 'content': block})
                elif isinstance(block, Image.Image):
                    image_count += 1
                    path = directory / f'{phase}_{image_count:03}.png'
                    temporary = path.with_suffix('.tmp.png')
                    block.save(temporary)
                    temporary.replace(path)
                    saved.append({'type': 'image', 'content': str(path.resolve())})
                else:
                    raise TypeError(f'Unexpected output block: {type(block)}')
            if phase == 'sf' and not saved:
                raise ValueError('Empty SF response after export')
            row.update(metadata)
            row[key], row[phase + '_status'] = saved, 'complete'
            row.pop(phase + '_error', None)
        except Exception as exc:
            LOG.exception('id=%s phase=%s failed', sid, phase)
            row[key], row[phase + '_status'] = [], 'failed'
            row[phase + '_error'] = f'{type(exc).__name__}: {exc}'
            failures += 1
        publish(row)
    return failures


def worker(device, tasks, results, args, output_path, settings):
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    logging.basicConfig(level=logging.INFO)
    generator = load_generator(args.sensenova_repo, args.model_path, args.attn_backend, args.gpus_per_model)
    while True:
        task = tasks.get()
        if task is None:
            return
        sample, row = task
        failures = run_sample(generator, sample, row, output_path, args.model_name, Path(args.benchmark).stem,
                              settings, lambda value: results.put(('row', json.dumps(value, ensure_ascii=False))))
        results.put(('done', failures))


def run_parallel(args, pending, rows, output_path, settings):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU is available in this session; run in a GPU allocation')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = [d.strip() for d in visible.split(',')] if visible is not None else [str(i) for i in range(torch.cuda.device_count())]
    count = min(args.num_workers, len(pending))
    if any(not d for d in devices) or len(set(devices)) != len(devices) or count * args.gpus_per_model > len(devices):
        raise ValueError('Not enough distinct visible GPUs: num_workers * gpus_per_model exceeds availability')
    LOG.info('Using %s workers with %s GPU(s) per model', count, args.gpus_per_model)
    ctx = mp.get_context('spawn')
    tasks, results = ctx.Queue(), ctx.Queue()
    processes = []
    try:
        for task in pending:
            tasks.put(task)
        for index in range(count):
            tasks.put(None)
            group = ','.join(devices[index * args.gpus_per_model:(index + 1) * args.gpus_per_model])
            process = ctx.Process(target=worker, args=(group, tasks, results, args, output_path, settings))
            process.start()
            processes.append(process)
        completed = failures = 0
        while completed < len(pending):
            try:
                kind, value = results.get(timeout=1)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in processes) or all(p.exitcode is not None for p in processes):
                    raise RuntimeError('Worker exited early; completed phases are saved')
                continue
            if kind == 'row':
                row = json.loads(value)
                rows[row['eval_set_id']] = row
                write_rows(output_path, rows)
            else:
                completed += 1
                failures += value
                LOG.info('Completed %s/%s pending samples', completed, len(pending))
        for process in processes:
            process.join()
        return 2 if failures or any(p.exitcode != 0 for p in processes) else 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        tasks.cancel_join_thread()
        tasks.close()
        results.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sensenova-repo', default=str(Path(__file__).resolve().parent / 'SenseNova-U1'))
    parser.add_argument('--model-path', required=True, help='Local complete HF SFT checkpoint')
    parser.add_argument('--benchmark', required=True)
    parser.add_argument('--image-root', default='.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--sample-ids', nargs='+', type=int)
    parser.add_argument('--num-workers', type=int, default=os.environ.get('NUM_WORKERS'), help='Default: visible GPU count divided by gpus_per_model')
    parser.add_argument('--gpus-per-model', '--gpus_per_model', type=int, default=1,
                        help='GPUs per model process (default: 1)')
    parser.add_argument('--attn-backend', choices=('auto', 'flash', 'sdpa'), default='auto')
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--max-text-tokens', type=int, default=8192,
                        help='Total generated text tokens, including reasoning and image descriptions; original SFT default: 8192')
    parser.add_argument('--max-input-image-long-side', type=int, default=1536)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cfg-scale', type=float, default=4.0)
    parser.add_argument('--img-cfg-scale', type=float, default=1.0)
    parser.add_argument('--cfg-norm', choices=('none', 'global', 'channel'), default='none')
    parser.add_argument('--cfg-interval', nargs=2, type=float, default=[0.0, 1.0])
    parser.add_argument('--timestep-shift', type=float, default=3.0)
    parser.add_argument('--t-eps', type=float, default=0.02)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if not args.model_name or Path(args.model_name).name != args.model_name or args.model_name in ('.', '..'):
        parser.error('model-name must be a single directory name')
    if args.gpus_per_model < 1:
        parser.error('gpus_per_model must be positive')
    if args.num_workers is None:
        import torch
        args.num_workers = torch.cuda.device_count() // args.gpus_per_model
    if min(args.num_workers, args.num_steps, args.max_text_tokens) < 1:
        parser.error('Worker, GPU, step and token counts must be positive')
    if any(size < 32 or size % 32 for size in (args.width, args.height)):
        parser.error('Image dimensions must be positive multiples of 32')
    if not 0 < args.t_eps < 1 or args.timestep_shift <= 0 or not 0 <= args.cfg_interval[0] <= args.cfg_interval[1] <= 1:
        parser.error('Invalid diffusion parameters')
    if args.max_input_image_long_side < 0:
        parser.error('max-input-image-long-side must be nonnegative')
    args.sensenova_repo = str(Path(args.sensenova_repo).resolve())
    args.model_path = str(Path(args.model_path).resolve())
    repo = Path(args.sensenova_repo)
    if not (repo / 'src/sensenova_u1').is_dir():
        raise FileNotFoundError(f'Clone the upstream repository into {repo}')
    samples = load_samples(args.benchmark, args.image_root)
    if args.sample_ids:
        wanted = set(args.sample_ids)
        samples = [s for s in samples if s['id'] in wanted]
        if {s['id'] for s in samples} != wanted:
            parser.error('Unknown sample IDs')
    keys = ('width', 'height', 'num_steps', 'max_text_tokens', 'seed', 'cfg_scale', 'img_cfg_scale',
            'cfg_norm', 'cfg_interval', 'timestep_shift', 't_eps', 'max_input_image_long_side', 'attn_backend')
    settings = {key: getattr(args, key) for key in keys}
    settings.update(reasoning=True, sf_image_cap=SF_IMAGE_CAP,
                    seed_scheme='fixed-v1', model_path=args.model_path,
                    image_root=str(Path(args.image_root).resolve()))
    path = (Path(args.output_dir) / args.model_name / (Path(args.benchmark).stem + '.jsonl')).resolve()
    rows = read_rows(path, samples, args.model_name, Path(args.benchmark).stem, settings, args.resume)
    pending = [(s, rows.get(s['id'])) for s in samples if not all(phase_complete(rows.get(s['id']), s, p) for p in ('sf', 'tf'))]
    if not pending:
        LOG.info('All phases already complete')
        return 0
    return run_parallel(args, pending, rows, path, settings)


if __name__ == '__main__':
    raise SystemExit(main())
