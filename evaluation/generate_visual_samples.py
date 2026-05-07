from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping
import sys

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

CURRENT_ROOT = Path(__file__).resolve().parents[1]
OLD_RESEARCH_ROOT = Path('/scratch/nickyun/diffusion-test01')
for path in (OLD_RESEARCH_ROOT, CURRENT_ROOT):
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)

from evaluation.evaluate_suite import _load_research_checkpoint, _prepare_effective_cfg
from evaluation.run_comparisons import _load_published_diffusion
from models.diffusion.sampling import edm_sampler_sr
from research.channels import ChannelMixture, ChannelSpec, build_default_channel_specs
from research.config import load_json_config
from research_image_token_pipeline import (
    build_eval_loader,
    build_swin,
    conditioning_labels,
    get_channel_input_size,
    load_research_modules,
    set_seed,
)


MODE_LABELS = {
    'original': 'Original',
    'channel': 'Channel',
    'published_baseline': 'Published EDM',
    'transport': 'Transport',
    'baseline': 'Baseline',
    'tokenizer_cond': 'Tokenizer cond',
    'diffusion_no_token': 'No token',
    'diffusion_token': 'Full token',
}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _select_specs(cfg: Mapping[str, Any], wanted: Iterable[str]) -> list[ChannelSpec]:
    channel_cfg = cfg.get('channel', {})
    specs = [ChannelSpec(**s) for s in channel_cfg['specs']] if channel_cfg.get('specs') else build_default_channel_specs()
    by_name = {spec.name: spec for spec in specs}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise ValueError(f'Unknown channel(s): {missing}; available={sorted(by_name)}')
    return [by_name[name] for name in wanted]


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    image = ((tensor.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    image = image.permute(1, 2, 0).numpy()
    return Image.fromarray(image)


def _pil_font(size: int) -> ImageFont.ImageFont:
    for path in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _draw_contact_sheet(
    columns: list[str],
    tensors: Mapping[str, torch.Tensor],
    *,
    title: str,
    output: Path,
) -> None:
    n_samples = next(iter(tensors.values())).shape[0]
    tile = int(next(iter(tensors.values())).shape[-1])
    label_h = 34
    row_label_w = 74
    pad = 8
    title_h = 40
    font = _pil_font(18)
    small_font = _pil_font(15)

    width = row_label_w + len(columns) * tile + (len(columns) + 1) * pad
    height = title_h + label_h + n_samples * tile + (n_samples + 1) * pad
    sheet = Image.new('RGB', (width, height), color=(248, 248, 248))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 8), title, fill=(20, 20, 20), font=font)

    y0 = title_h
    for ci, col in enumerate(columns):
        x = row_label_w + pad + ci * (tile + pad)
        draw.text((x + 4, y0 + 7), MODE_LABELS.get(col, col), fill=(25, 25, 25), font=small_font)

    for ri in range(n_samples):
        y = title_h + label_h + pad + ri * (tile + pad)
        draw.text((pad, y + 8), f'#{ri}', fill=(70, 70, 70), font=small_font)
        for ci, col in enumerate(columns):
            x = row_label_w + pad + ci * (tile + pad)
            sheet.paste(_tensor_to_pil(tensors[col][ri]), (x, y))

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def _write_readme(output_dir: Path, generated: list[Path], args: argparse.Namespace, sampler_steps: int) -> None:
    rel_paths = [path.relative_to(output_dir) for path in generated]
    lines = [
        '# V5 Login Visual Samples',
        '',
        f'Generated from `{args.config}` and checkpoint `{args.checkpoint}`.',
        '',
        f'Samples per channel: `{args.num_samples}`.',
        f'Sampler steps: `{sampler_steps}`.',
        '',
        '| File | Contents |',
        '|---|---|',
    ]
    for rel in rel_paths:
        lines.append(f'| `{rel}` | Original/channel plus comparison reconstructions |')
    lines.extend(
        [
            '',
            'Columns use the same channelized inputs per row. Diffusion columns share the same initial sampler noise within each channel so visual differences are attributable to conditioning/mode changes.',
            '',
        ]
    )
    (output_dir / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def _load_best_comparison_rows(run_root: Path) -> Dict[str, str]:
    comparison_csv = run_root / 'comparisons' / 'comparisons.csv'
    if not comparison_csv.exists():
        return {}
    rows: Dict[str, str] = {}
    with comparison_csv.open('r', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            method = row.get('method', '')
            mode = row.get('mode', '')
            if method and mode:
                rows[mode] = method
    return rows


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='outputs/research_image_token_v5_login/visual_samples')
    parser.add_argument('--channels', nargs='+', default=['awgn', 'rayleigh', 'uma'])
    parser.add_argument('--num-samples', type=int, default=4)
    parser.add_argument('--sampler-steps', type=int, default=None)
    parser.add_argument('--include-published', action='store_true')
    args = parser.parse_args()

    request_cfg = load_json_config(args.config)
    set_seed(int(request_cfg.get('seed', 42)))
    device = torch.device(request_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    run_root = Path(args.output).parents[0]

    state = _load_research_checkpoint(args.checkpoint, device)
    cfg = _prepare_effective_cfg(request_cfg, state)
    cfg = _deep_merge(cfg, {'val_batch_size': args.num_samples, 'num_workers': 0})
    strict_loading = bool(cfg.get('strict_checkpoint_loading', True))
    sampler_steps = int(args.sampler_steps if args.sampler_steps is not None else cfg.get('sampler_steps', 18))
    channel_input_size = get_channel_input_size(cfg)

    swin, receiver, edm_model, _ = load_research_modules(cfg, state, device, strict_loading=strict_loading, is_main=True)
    swin.eval()
    receiver.eval()
    edm_model.eval()

    published = None
    if args.include_published:
        for ent in request_cfg.get('comparison', {}).get('entries', []):
            if ent.get('kind') == 'published_baseline':
                published = _load_published_diffusion(ent['diffusion_ckpt'], device)
                published.eval()
                break
        if published is None:
            raise ValueError('No published_baseline entry found in config.comparison.entries')

    loader = build_eval_loader(cfg)
    imgs = next(iter(loader)).to(device)[: args.num_samples]
    specs = _select_specs(cfg, args.channels)
    mixture = ChannelMixture(specs=specs, backend=cfg.get('channel', {}).get('backend', 'sionna'))

    model_cfg = cfg.get('model', {})
    use_explicit_csi = str(model_cfg.get('edm_variant', 'pilotless')).strip().lower() == 'oracle_csi'

    columns = ['original', 'channel']
    if published is not None:
        columns.append('published_baseline')
    columns.extend(['transport', 'baseline', 'tokenizer_cond', 'diffusion_no_token', 'diffusion_token'])

    generated: list[Path] = []
    out_dir = Path(args.output)
    for si, spec in enumerate(specs):
        set_seed(int(request_cfg.get('seed', 42)) + si * 1000)
        clean_latent = swin.encoder(imgs)
        imgs_lr = F.interpolate(
            imgs,
            size=(channel_input_size, channel_input_size),
            mode='bicubic',
            align_corners=False,
            antialias=True,
        )
        noisy_lr, ch_meta = mixture.apply(imgs_lr, spec=spec)
        noisy_img = F.interpolate(noisy_lr, size=imgs.shape[-2:], mode='bilinear', align_corners=False)
        noisy_latent = swin.encoder(noisy_img)
        pout = receiver(noisy_latent)
        sample_noise = torch.randn_like(clean_latent)

        tensors: Dict[str, torch.Tensor] = {
            'original': imgs,
            'channel': noisy_img,
            'transport': swin.decoder(pout.restored).float(),
        }
        if published is not None:
            pred = edm_sampler_sr(
                published,
                latents=sample_noise.clone(),
                x_cond=noisy_latent,
                class_labels=None,
                num_steps=sampler_steps,
            ).float()
            tensors['published_baseline'] = swin.decoder(pred).float()

        pred = edm_sampler_sr(
            edm_model,
            latents=sample_noise.clone(),
            x_cond=noisy_latent,
            class_labels=None,
            h=ch_meta.get('h', None) if use_explicit_csi else None,
            num_steps=sampler_steps,
        ).float()
        tensors['baseline'] = swin.decoder(pred).float()

        pred = edm_sampler_sr(
            edm_model,
            latents=sample_noise.clone(),
            x_cond=pout.restored,
            class_labels=None,
            h=ch_meta.get('h', None) if use_explicit_csi else None,
            num_steps=sampler_steps,
        ).float()
        tensors['tokenizer_cond'] = swin.decoder(pred).float()

        pred = edm_sampler_sr(
            edm_model,
            latents=sample_noise.clone(),
            x_cond=pout.restored,
            class_labels=conditioning_labels(pout.token, ch_meta, label_mode='zero'),
            h=ch_meta.get('h', None) if use_explicit_csi else None,
            num_steps=sampler_steps,
        ).float()
        tensors['diffusion_no_token'] = swin.decoder(pred).float()

        pred = edm_sampler_sr(
            edm_model,
            latents=sample_noise.clone(),
            x_cond=pout.restored,
            class_labels=conditioning_labels(pout.token, ch_meta, label_mode='token'),
            h=ch_meta.get('h', None) if use_explicit_csi else None,
            num_steps=sampler_steps,
        ).float()
        tensors['diffusion_token'] = swin.decoder(pred).float()

        output = out_dir / f'v5_login_visual_{spec.name}.png'
        _draw_contact_sheet(columns, tensors, title=f'v5 login visual comparison - {spec.name}', output=output)
        generated.append(output)
        print(output)

    _write_readme(out_dir, generated, args, sampler_steps)

    methods = _load_best_comparison_rows(run_root)
    if methods:
        print(f'comparison methods found: {methods}')


if __name__ == '__main__':
    main()
