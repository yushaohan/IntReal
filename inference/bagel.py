# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Run IntReal SF+TF with official BAGEL; import run_benchmark to reuse an existing generator."""

import argparse
import contextlib
import hashlib
import json
import logging
import multiprocessing as mp
import os
import queue
import time
import random
import re
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image

LOG = logging.getLogger('intreal')
SYSTEM_ON = '''You are a multimodal instruction assistant that interleaves step-by-step text with generated images. When responding to user requests, follow these rules:
1. Begin with a high-level plan inside <global_think>...</global_think> tags. Outline the logical sequence and identify which steps require visual illustration.
2. Provide instructions in clear, concise, professional prose.
3. When a step benefits from visual aid, enclose the image description inside <image_think>...</image_think> tags. The description must be vivid, specific, and technically precise. Every <image_think> block must end with </image_think> and be immediately followed by an image.
4. Alternate naturally between text and images. Only generate images where visual representation genuinely clarifies a complex step, state, or outcome.
5. Maintain seamless transitions: guide the user logically from text to image and back to text.
6. You may conclude with a final text paragraph without generating an image.'''
SYSTEM_OFF = '''You are a multimodal instruction assistant that interleaves step-by-step text with generated images. When responding to user requests, follow these rules:
1. Provide instructions in clear, concise, professional prose.
2. When a step benefits from visual aid, enclose the image generation tag <image>. After the image generation tag, you should generate a image based on the previous text.
3. Alternate naturally between text and images. Only generate images where visual representation genuinely clarifies a complex step, state, or outcome.
4. Maintain seamless transitions: guide the user logically from text to image and back to text.
5. You may conclude with a final text paragraph without generating an image.'''


def read_image(path):
    """Read RGB pixels and release the input file immediately."""
    with Image.open(path) as image:
        return image.convert('RGB').copy()


def clean_text(text):
    """Remove reasoning and control markers only from the exported answer."""
    for tag in ('global_think', 'image_think', 'think'):
        text = re.sub(rf'<{tag}>.*?(?:</{tag}>|$)', '', text, flags=re.S)
    return text.replace('<image>', '').strip()


@contextlib.contextmanager
def sample_seed(seed, sample_id, phase):
    """Isolate RNG state and make SF/TF seeds independent of resume order."""
    value = int.from_bytes(hashlib.sha256(f'{seed}:{sample_id}:{phase}'.encode()).digest()[:4], 'big')
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            random.seed(value)
            np.random.seed(value)
            torch.manual_seed(value)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class IntRealGenerator:
    """Own the SF/TF policy while accepting an already loaded BAGEL inferencer."""

    def __init__(self, inferencer):
        self.inf = inferencer

    def _image(self, image, context, vit):
        image = self.inf.vae_transform.resize_transform(image)
        return self.inf.update_context_image(image, context, vae=True, vit=vit)

    def _prefix(self, sample, system=''):
        gen = self.inf.init_gen_context()
        cfg_text, cfg_img = deepcopy(gen), deepcopy(gen)
        if system:
            gen = self.inf.update_context_text(system, gen)
            cfg_img = self.inf.update_context_text(system, cfg_img)
        for path in sample['prompt'].get('images', []):
            image = read_image(path)
            gen = self._image(image, gen, True)
            cfg_img = self._image(image, cfg_img, True)
            cfg_text = deepcopy(gen)
        prompt = ' '.join(sample['prompt']['text'].replace('<image>', '').split())
        gen = self.inf.update_context_text(prompt, gen)
        cfg_img = self.inf.update_context_text(prompt, cfg_img)
        if cfg_text['kv_lens'] == [0]:
            cfg_text = deepcopy(gen)
        return gen, cfg_text, cfg_img

    def _text(self, context, max_tokens, temperature):
        """Decode batch size 1, committing every boundary token to KV before returning."""
        model, tokenizer = self.inf.model, self.inf.tokenizer
        device = model.language_model.model.embed_tokens.weight.device
        token = torch.tensor([self.inf.new_token_ids['bos_token_id']], device=device)
        ids = []
        for step in range(max_tokens + 1):
            length, position = context['kv_lens'][0], context['ropes'][0]
            extra = {'mode': 'und'} if model.use_moe else {}
            output = model.language_model.forward_inference(
                packed_query_sequence=model.language_model.model.embed_tokens(token),
                query_lens=torch.ones(1, device=device, dtype=torch.int),
                packed_query_position_ids=torch.tensor([position], device=device),
                packed_query_indexes=torch.tensor([length], device=device),
                past_key_values=context['past_key_values'],
                key_values_lens=torch.tensor([length], device=device, dtype=torch.int),
                packed_key_value_indexes=torch.arange(length, device=device),
                update_past_key_values=True, is_causal=True, **extra,
            )
            context['past_key_values'] = output.past_key_values
            context['kv_lens'][0] += 1
            context['ropes'][0] += 1
            if step:
                if token.item() == self.inf.new_token_ids['eos_token_id']:
                    return tokenizer.decode(ids), 'eos'
                ids.append(token.item())
                text = tokenizer.decode(ids)
                if any(marker in text for marker in ('</image_think>', '<image>')):
                    return text, 'image'
                if step == max_tokens:
                    return text, 'length'
            logits = model.language_model.lm_head(output.packed_query_sequence).float()
            if temperature > 0:
                token = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1).flatten().to(device)
            else:
                token = logits.argmax(dim=-1).to(device)
        raise AssertionError('Unreachable decoding state')

    @torch.no_grad()
    def generate_sf(self, sample, *, reasoning, max_text_tokens, max_images, temperature, **image_kwargs):
        """Generate a free-running answer; reasoning mode selects the checkpoint's prompt protocol."""
        with torch.autocast('cuda', dtype=torch.bfloat16):
            gen, cfg_text, cfg_img = self._prefix(sample, SYSTEM_ON if reasoning == 'on' else SYSTEM_OFF)
            blocks, image_count = [], 0
            while True:
                text, reason = self._text(gen, max_text_tokens, temperature)
                blocks.append(text)
                if reason != 'image':
                    return blocks, reason
                if image_count >= max_images:
                    return blocks, 'image_limit'
                image = self.inf.gen_image(
                    gen_context=gen, cfg_text_precontext=deepcopy(cfg_text),
                    cfg_img_precontext=deepcopy(cfg_img), **image_kwargs,
                )
                blocks.append(image)
                image_count += 1
                gen = self._image(image, gen, False)
                cfg_text, cfg_img = deepcopy(gen), deepcopy(gen)

    @torch.no_grad()
    def generate_tf(self, sample, **image_kwargs):
        """Generate every reference image position from GT prefixes, without reasoning or SF feedback."""
        with torch.autocast('cuda', dtype=torch.bfloat16):
            gen, cfg_text, cfg_img = self._prefix(sample)
            images = []
            for block in sample['golden_result']:
                if block['type'] == 'text':
                    gen = self.inf.update_context_text(block['content'], gen)
                    cfg_img = self.inf.update_context_text(block['content'], cfg_img)
                else:
                    images.append(self.inf.gen_image(
                        gen_context=gen, cfg_text_precontext=deepcopy(cfg_text),
                        cfg_img_precontext=deepcopy(cfg_img), **image_kwargs,
                    ))
                    gen = self._image(read_image(block['content']), gen, True)
                    cfg_text, cfg_img = deepcopy(gen), deepcopy(gen)
            return images


def load_generator(model_path, checkpoint=None, gpu_memory_gib=60, bagel_repo=None):
    """Load one model, balancing decoder layers across visible GPUs without touching upstream files."""
    repo = Path(bagel_repo).expanduser().resolve() if bagel_repo else Path(__file__).resolve().parent / 'Bagel'
    for relative in ('inferencer.py', 'modeling/bagel', 'data/data_utils.py'):
        if not (repo / relative).exists():
            raise FileNotFoundError(f'Missing BAGEL source: {repo / relative}; clone BAGEL into inference/Bagel or use --bagel-repo')
    sys.path.insert(0, str(repo))
    from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
    from accelerate.utils import get_balanced_memory
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    from inferencer import InterleaveInferencer
    from modeling.autoencoder import load_ae
    from modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM
    from modeling.bagel import SiglipVisionConfig, SiglipVisionModel
    from modeling.qwen2 import Qwen2Tokenizer

    if not torch.cuda.is_available():
        raise RuntimeError('BAGEL inference requires a CUDA GPU')
    root = Path(model_path)
    llm = Qwen2Config.from_json_file(str(root / 'llm_config.json'))
    llm.qk_norm, llm.tie_word_embeddings, llm.layer_module = True, False, 'Qwen2MoTDecoderLayer'
    vit = SiglipVisionConfig.from_json_file(str(root / 'vit_config.json'))
    vit.rope, vit.num_hidden_layers = False, vit.num_hidden_layers - 1
    vae, vae_config = load_ae(local_path=str(root / 'ae.safetensors'))
    config = BagelConfig(visual_gen=True, visual_und=True, llm_config=llm, vit_config=vit,
                         vae_config=vae_config, vit_max_num_patch_per_side=70,
                         connector_act='gelu_pytorch_tanh', latent_patch_size=2, max_latent_size=64)
    with init_empty_weights():
        model = Bagel(Qwen2ForCausalLM(llm), SiglipVisionModel(vit), config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit, meta=True)
    tokenizer, tokens, _ = add_special_tokens(Qwen2Tokenizer.from_pretrained(str(root)))
    weights = Path(checkpoint) if checkpoint else root / 'ema.safetensors'
    if not weights.exists():
        raise FileNotFoundError(weights)
    no_split = ['Qwen2MoTDecoderLayer', 'SiglipVisionModel']
    limits = {i: f'{gpu_memory_gib}GiB' for i in range(torch.cuda.device_count())}
    balanced = get_balanced_memory(model, max_memory=limits, no_split_module_classes=no_split,
                                   dtype=torch.bfloat16)
    device_map = infer_auto_device_map(model, max_memory=balanced, no_split_module_classes=no_split,
                                       dtype=torch.bfloat16)
    if any(device in ('cpu', 'disk') for device in device_map.values()):
        raise RuntimeError('Insufficient GPU memory; increase --gpu-memory-gib or visible GPU count')
    # These modules exchange tensors outside module forward hooks.
    tied = ('time_embedder', 'latent_pos_embed', 'vae2llm', 'llm2vae', 'connector', 'vit_pos_embed')
    first = next(iter(device_map.values()))
    if '' not in device_map:
        for name in tied:
            for key in list(device_map):
                if key == name or key.startswith(name + '.'):
                    del device_map[key]
            device_map[name] = first
    LOG.info('Model device map: %s', device_map)
    model = load_checkpoint_and_dispatch(model, str(weights), device_map=device_map,
                                        dtype=torch.bfloat16, force_hooks=True).eval()
    vae_device = f'cuda:{first}' if isinstance(first, int) else first
    vae = vae.to(device=vae_device, dtype=torch.float32).eval()
    for module in (vae.encoder, vae.decoder):
        add_hook_to_module(module, AlignDevicesHook(execution_device=vae_device))
    inferencer = InterleaveInferencer(model, vae, tokenizer, ImageTransform(1024, 512, 16),
                                      ImageTransform(980, 224, 14), tokens)
    return IntRealGenerator(inferencer)


def load_samples(benchmark, image_root):
    """Validate unique IDs and resolve input/GT image paths without changing the benchmark."""
    rows = [json.loads(line) for line in Path(benchmark).read_text().splitlines() if line.strip()]
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Benchmark must contain nonempty, unique IDs')
    for row in rows:
        if not isinstance(row['id'], int) or not row['prompt']['text'].strip():
            raise ValueError('Expected integer IDs and nonempty prompts')
        images = row['prompt'].get('images', [])
        row['prompt']['images'] = [str((Path(image_root) / p).resolve()) for p in images]
        for block in row['golden_result']:
            if block['type'] not in ('text', 'image'):
                raise ValueError(f"Unsupported block type: {block['type']}")
            if block['type'] == 'image':
                block['content'] = str((Path(image_root) / block['content']).resolve())
        paths = row['prompt']['images'] + [b['content'] for b in row['golden_result'] if b['type'] == 'image']
        for path in paths:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    return rows


def save_blocks(blocks, image_dir, prefix):
    """Save generated pixels and export only answer text to the standard IntReal schema."""
    image_dir.mkdir(parents=True, exist_ok=True)
    result, count = [], 0
    for block in blocks:
        if isinstance(block, str):
            text = clean_text(block)
            if text:
                result.append({'type': 'text', 'content': text})
        elif isinstance(block, Image.Image):
            count += 1
            path = image_dir / f'{prefix}_{count:03}.png'
            block.save(path)
            result.append({'type': 'image', 'content': str(path.resolve())})
        else:
            raise TypeError(f'Unexpected generated block: {type(block)}')
    return result


def read_rows(output_path, samples, model_name, eval_set, settings, resume):
    """Validate existing results before loading models."""
    rows = {}
    if output_path.exists():
        if not resume:
            raise FileExistsError(f'Use --resume or a new output directory: {output_path}')
        for line in output_path.read_text().splitlines():
            row = json.loads(line)
            if row['eval_set_id'] in rows:
                raise ValueError('Duplicate IDs in existing output')
            if row.get('generation') != settings or row['model'] != model_name or row['eval_set'] != eval_set:
                raise ValueError('Resume configuration differs; use a new output directory')
            rows[row['eval_set_id']] = row
        if set(rows) - {s['id'] for s in samples}:
            raise ValueError('Existing output contains IDs outside this benchmark selection')
    return rows


def run_benchmark(generator, samples, output_path, *, model_name, eval_set, settings, resume=False, initial_rows=None, on_row=None):
    """Always run SF and TF; accept an injected generator with generate_sf/generate_tf methods."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(output_path, samples, model_name, eval_set, settings, resume) if initial_rows is None else initial_rows
    image_kwargs = {k: settings[k] for k in ('image_shape', 'num_timesteps', 'timestep_shift',
                   'cfg_text_scale', 'cfg_img_scale', 'cfg_interval', 'cfg_renorm_min', 'cfg_renorm_type')}
    failures = 0
    for index, sample in enumerate(samples, 1):
        sid = sample['id']
        row = rows.get(sid, {'model': model_name, 'eval_set': eval_set, 'eval_set_id': sid,
                            'response_sf': [], 'response_tf': [], 'generation': settings})
        image_dir = output_path.parent / 'images' / f'{sid:06}'
        for phase in ('sf', 'tf'):
            key = 'response_' + phase
            existing = row[key]
            expected = sum(b['type'] == 'image' for b in sample['golden_result'])
            complete = bool(existing) and all(Path(b['content']).is_file() for b in existing if b['type'] == 'image')
            if phase == 'tf':
                complete = complete and len(existing) == expected and all(b['type'] == 'image' for b in existing)
            if complete:
                continue
            try:
                started = time.monotonic()
                LOG.info('id=%s starting %s', sid, phase.upper())
                with sample_seed(settings['seed'], sid, phase):
                    if phase == 'sf':
                        blocks, reason = generator.generate_sf(sample, reasoning=settings['reasoning'],
                            max_text_tokens=settings['max_text_tokens'], max_images=settings['max_images'],
                            temperature=settings['temperature'], **image_kwargs)
                        row['sf_finish_reason'] = reason
                        row['sf_raw_text'] = [b for b in blocks if isinstance(b, str)]
                    else:
                        blocks = generator.generate_tf(sample, **image_kwargs)
                        if len(blocks) != expected:
                            raise ValueError(f'TF returned {len(blocks)} images; expected {expected}')
                row[key] = save_blocks(blocks, image_dir, phase)
                if not row[key]:
                    raise ValueError(f'Empty {phase.upper()} response')
                row.pop(phase + '_error', None)
                LOG.info('id=%s %s completed in %.2fs', sid, phase.upper(), time.monotonic() - started)
            except Exception as exc:
                row[key] = []
                row[phase + '_error'] = f'{type(exc).__name__}: {exc}'
                failures += 1
                LOG.exception('id=%s %s failed', sid, phase.upper())
            rows[sid] = row
            if on_row is None:
                write_rows(output_path, rows)
            else:
                on_row(row)
        LOG.info('%d/%d id=%s SF=%s TF=%s', index, len(samples), sid,
                 bool(row['response_sf']), bool(row['response_tf']))
    return 2 if failures else 0


def write_rows(output_path, rows):
    """Atomically publish results from one writer."""
    temporary = output_path.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(rows[k], ensure_ascii=False) + '\n' for k in sorted(rows)))
    temporary.replace(output_path)


def inference_worker(device, tasks, results, args, output_path, settings):
    """Keep one model on one visible GPU and process one prompt at a time."""
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(processName)s %(levelname)s %(message)s')
    try:
        generator = load_generator(args.model_path, args.checkpoint, args.gpu_memory_gib, args.bagel_repo)
        LOG.info('Worker ready on physical GPU %s', device)
        while True:
            task = tasks.get()
            if task is None:
                break
            sample, row = task
            def publish(value):
                results.put(('row', json.dumps(value, ensure_ascii=False)))
            status = run_benchmark(generator, [sample], output_path, model_name=args.model_name,
                eval_set=Path(args.benchmark).stem, settings=settings,
                initial_rows={sample['id']: row} if row else {}, on_row=publish)
            results.put(('done', (sample['id'], status)))
    except Exception:
        LOG.exception('Worker on GPU %s failed', device)
        raise


def run_parallel(args, samples, output_path, settings):
    """Distribute prompts dynamically and serialize all JSONL writes in the parent."""
    rows = read_rows(output_path, samples, args.model_name, Path(args.benchmark).stem, settings, args.resume)
    pending = []
    for sample in samples:
        row = rows.get(sample['id'])
        complete = row is not None
        for phase in ('sf', 'tf'):
            blocks = row.get('response_' + phase, []) if row else []
            complete = complete and bool(blocks) and all(Path(b['content']).is_file() for b in blocks if b['type'] == 'image')
            if phase == 'tf':
                complete = complete and len(blocks) == sum(b['type'] == 'image' for b in sample['golden_result']) and all(b['type'] == 'image' for b in blocks)
        if not complete:
            pending.append((sample, row))
    if not pending:
        LOG.info('All %s samples already complete', len(samples))
        return 0
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = visible.split(',') if visible is not None else [str(i) for i in range(torch.cuda.device_count())]
    if args.num_workers > len(devices) or any(not d.strip() for d in devices):
        raise ValueError('--num-workers exceeds visible GPUs')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    context = mp.get_context('spawn')
    tasks, results = context.Queue(), context.Queue()
    workers = []
    failures = completed = 0
    started = time.monotonic()
    try:
        for task in pending:
            tasks.put(task)
        for index in range(min(args.num_workers, len(pending))):
            tasks.put(None)
            process = context.Process(target=inference_worker, name=f'gpu-{devices[index]}',
                args=(devices[index].strip(), tasks, results, args, output_path, settings))
            process.start()
            workers.append(process)
        while completed < len(pending):
            try:
                kind, value = results.get(timeout=1)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in workers) or all(p.exitcode is not None for p in workers):
                    raise RuntimeError('Inference worker exited before all samples completed; completed phases are saved')
                continue
            if kind == 'row':
                row = json.loads(value)
                rows[row['eval_set_id']] = row
                write_rows(output_path, rows)
            else:
                completed += 1
                failures += value[1] != 0
                LOG.info('Parallel completed %s/%s pending samples; id=%s elapsed=%.1fs', completed, len(pending), value[0], time.monotonic() - started)
        for process in workers:
            process.join()
        return 2 if failures or any(p.exitcode != 0 for p in workers) else 0
    finally:
        for process in workers:
            if process.is_alive():
                process.terminate()
        for process in workers:
            process.join()
        tasks.cancel_join_thread()
        tasks.close()
        results.close()


def main():
    """Load the official BAGEL backend and run the complete benchmark protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bagel-repo', help='External BAGEL source directory; default: Bagel/ beside this script')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--checkpoint', help='Weights file or sharded checkpoint directory; default: MODEL/ema.safetensors')
    parser.add_argument('--benchmark', required=True)
    parser.add_argument('--image-root', default='.', help='Root for relative reference images')
    parser.add_argument('--output-dir', required=True, help='Writes DIR/MODEL/BENCHMARK.jsonl and images/')
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--reasoning', choices=('on', 'off'), default='off', help='Match the fine-tuned checkpoint: on=full reasoning, off=no reasoning; not adaptive. TF is unchanged.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--sample-ids', type=int, nargs='+', help='Optional subset for smoke runs')
    parser.add_argument('--num-workers', type=int, default=1, help='Independent single-GPU workers; each uses batch size 1')
    parser.add_argument('--gpu-memory-gib', type=int, default=60)
    parser.add_argument('--image-shape', type=int, nargs=2, default=[1024, 1024], metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--max-text-tokens', type=int, default=2048)
    parser.add_argument('--max-images', type=int, default=12, help='Safety cap, not the requested image count')
    parser.add_argument('--temperature', type=float, default=0.0, help='0 selects greedy text decoding')
    parser.add_argument('--num-timesteps', type=int, default=50)
    parser.add_argument('--timestep-shift', type=float, default=3.0)
    parser.add_argument('--cfg-text-scale', type=float, default=3.0)
    parser.add_argument('--cfg-img-scale', type=float, default=1.5)
    parser.add_argument('--cfg-interval', type=float, nargs=2, default=[0.4, 1.0])
    parser.add_argument('--cfg-renorm-min', type=float, default=0.0)
    parser.add_argument('--cfg-renorm-type', choices=('global', 'channel', 'text_channel'), default='global')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if Path(args.model_name).name != args.model_name or args.model_name in ('.', '..'):
        parser.error('--model-name must be a single directory name')
    if args.num_workers < 1:
        parser.error('--num-workers must be positive')
    if args.max_text_tokens < 1 or args.max_images < 1 or args.num_timesteps < 2 or args.temperature < 0:
        parser.error('Invalid token/image/step limit or temperature')
    if any(size < 16 or size > 1024 or size % 16 for size in args.image_shape):
        parser.error('Image dimensions must be multiples of 16 between 16 and 1024')
    samples = load_samples(args.benchmark, args.image_root)
    if args.sample_ids:
        wanted = set(args.sample_ids)
        samples = [s for s in samples if s['id'] in wanted]
        if {s['id'] for s in samples} != wanted:
            parser.error('--sample-ids contains IDs absent from the benchmark')
    fields = ('reasoning', 'seed', 'max_text_tokens', 'max_images', 'temperature', 'image_shape',
              'num_timesteps', 'timestep_shift', 'cfg_text_scale', 'cfg_img_scale', 'cfg_interval',
              'cfg_renorm_min', 'cfg_renorm_type')
    settings = {key: getattr(args, key) for key in fields}
    weights = Path(args.checkpoint) if args.checkpoint else Path(args.model_path) / 'ema.safetensors'
    settings.update(model_path=str(Path(args.model_path).resolve()), checkpoint=str(weights.resolve()),
                    checkpoint_mtime_ns=weights.stat().st_mtime_ns,
                    benchmark_sha256=hashlib.sha256(Path(args.benchmark).read_bytes()).hexdigest())
    output_path = Path(args.output_dir) / args.model_name / (Path(args.benchmark).stem + '.jsonl')
    if output_path.exists() and not args.resume:
        raise FileExistsError(f'Use --resume or a new output directory: {output_path}')
    if args.num_workers > 1:
        return run_parallel(args, samples, output_path, settings)
    generator = load_generator(args.model_path, args.checkpoint, args.gpu_memory_gib, args.bagel_repo)
    return run_benchmark(generator, samples, output_path, model_name=args.model_name,
                         eval_set=Path(args.benchmark).stem, settings=settings, resume=args.resume)


if __name__ == '__main__':
    raise SystemExit(main())
