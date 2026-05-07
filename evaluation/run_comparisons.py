from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping
import sys

import torch

CURRENT_ROOT = Path(__file__).resolve().parents[1]
OLD_RESEARCH_ROOT = Path('/scratch/nickyun/diffusion-test01')
for path in (OLD_RESEARCH_ROOT, CURRENT_ROOT):
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)

from evaluation.evaluate_suite import (
    _build_loader,
    _evaluate_mode,
    _load_research_checkpoint,
    _load_research_modules,
    _prepare_effective_cfg,
)
from models.diffusion.networks import EDMPrecondSR
from models.diffusion.sampling import edm_sampler_sr
from research.channels import ChannelMixture, ChannelSpec, build_default_channel_specs
from research.config import load_json_config, save_json
from research.metrics import batch_psnr, batch_ssim, latent_cosine, linear_cka, summarize
from research_image_token_pipeline import build_swin, channelize_images, get_channel_input_size, set_seed


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row['method']), str(row['channel']), str(row['mode'])


def _load_existing_rows(out_dir: Path) -> Dict[tuple[str, str, str], Dict[str, float | str]]:
    rows: Dict[tuple[str, str, str], Dict[str, float | str]] = {}
    csv_path = out_dir / 'comparisons.csv'
    if not csv_path.exists():
        return rows
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if not {'method', 'channel', 'mode'} <= set(row):
                continue
            parsed: Dict[str, float | str] = {}
            for key, value in row.items():
                if key in {'method', 'channel', 'mode'}:
                    parsed[key] = value
                else:
                    try:
                        parsed[key] = float(value)
                    except (TypeError, ValueError):
                        parsed[key] = value
            rows[_row_key(parsed)] = parsed
    return rows


def _write_results(out_dir: Path, rows: List[Dict[str, float | str]], cfg: Dict[str, Any]) -> None:
    if not rows:
        return
    csv_path = out_dir / 'comparisons.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    save_json(str(out_dir / 'comparisons.json'), {'rows': rows, 'config': cfg})


def _read_external_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _load_published_diffusion(path: str, device: torch.device) -> torch.nn.Module:
    model = EDMPrecondSR(img_resolution=32, img_channels=24, cond_channels=24, sigma_data=0.06).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    return model


@torch.no_grad()
def _evaluate_published_baseline(
    *,
    swin: torch.nn.Module,
    diffusion: torch.nn.Module,
    loader,
    channel_mix: ChannelMixture,
    channel_spec: ChannelSpec,
    device: torch.device,
    sampler_steps: int,
    channel_input_size: int,
) -> Dict[str, float]:
    psnr_all, ssim_all, cos_all, cka_all = [], [], [], []
    for imgs in loader:
        imgs = imgs.to(device)
        clean_latent, noisy_latent, _ = channelize_images(
            imgs,
            swin,
            channel_mix,
            channel_input_size=channel_input_size,
            channel_spec=channel_spec,
        )
        pred_latent = edm_sampler_sr(
            diffusion,
            latents=torch.randn_like(clean_latent),
            x_cond=noisy_latent,
            class_labels=None,
            num_steps=sampler_steps,
        ).float()
        recon = swin.decoder(pred_latent)
        psnr_all.append(batch_psnr((imgs + 1) / 2, (recon + 1) / 2))
        ssim_all.append(batch_ssim(imgs, recon))
        cos_all.append(latent_cosine(clean_latent, pred_latent))
        cka_all.append(linear_cka(clean_latent, pred_latent).view(1))

    return summarize(
        {
            'psnr': torch.cat(psnr_all),
            'ssim': torch.cat(ssim_all),
            'latent_cosine': torch.cat(cos_all),
            'latent_cka': torch.cat(cka_all),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output', type=str, default='outputs/research/comparisons')
    args = parser.parse_args()

    request_cfg = load_json_config(args.config)
    base_seed = int(request_cfg.get('seed', 42))
    set_seed(base_seed)
    device = torch.device(request_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = _build_loader(request_cfg)
    channel_cfg = request_cfg.get('channel', {})
    specs = [ChannelSpec(**s) for s in channel_cfg['specs']] if channel_cfg.get('specs') else build_default_channel_specs()
    mixture = ChannelMixture(specs=specs, backend=channel_cfg.get('backend', 'sionna'))
    sampler_steps = int(request_cfg.get('sampler_steps', 18))
    channel_input_size = get_channel_input_size(request_cfg)

    entries = request_cfg.get('comparison', {}).get('entries', [])
    if not entries:
        raise ValueError('No comparison entries found in config.comparison.entries')

    existing = _load_existing_rows(out_dir)
    rows: List[Dict[str, float | str]] = list(existing.values())
    if rows:
        print(f"Resuming comparisons with {len(rows)} completed rows")
        _write_results(out_dir, rows, request_cfg)

    for ent in entries:
        name = ent['name']
        kind = ent.get('kind', 'research')
        mode = ent.get('mode', 'diffusion_token')
        if kind == 'published_baseline':
            diffusion_ckpt = ent['diffusion_ckpt']
            swin = build_swin(request_cfg, device)
            diffusion = _load_published_diffusion(diffusion_ckpt, device)
            for si, spec in enumerate(specs):
                if (name, spec.name, mode) in existing:
                    continue
                set_seed(base_seed + si * 1000)
                stats = _evaluate_published_baseline(
                    swin=swin,
                    diffusion=diffusion,
                    loader=loader,
                    channel_mix=mixture,
                    channel_spec=spec,
                    device=device,
                    sampler_steps=int(ent.get('sampler_steps', sampler_steps)),
                    channel_input_size=channel_input_size,
                )
                row: Dict[str, float | str] = {'method': name, 'channel': spec.name, 'mode': mode}
                row.update(stats)
                rows.append(row)
                existing[_row_key(row)] = row
                print(row)
                _write_results(out_dir, rows, request_cfg)
            continue
        if kind != 'research':
            raise ValueError(f'Unknown comparison entry kind: {kind}')

        checkpoint = ent['checkpoint']
        state = _load_research_checkpoint(checkpoint, device)
        model_overrides = {}
        if 'edm_variant' in ent:
            model_overrides['edm_variant'] = ent['edm_variant']
        if 'oracle_use_token' in ent:
            model_overrides['oracle_use_token'] = bool(ent['oracle_use_token'])

        ent_cfg = _prepare_effective_cfg(request_cfg, state, model_overrides=model_overrides)
        strict_loading = bool(ent_cfg.get('strict_checkpoint_loading', True))
        use_explicit_csi = str(ent_cfg.get('model', {}).get('edm_variant', 'pilotless')).strip().lower() == 'oracle_csi'
        oracle_use_token = bool(ent_cfg.get('model', {}).get('oracle_use_token', False))

        swin, receiver, dm, _ = _load_research_modules(ent_cfg, state, device, strict_loading=strict_loading, is_main=True)

        for si, spec in enumerate(specs):
            if (name, spec.name, mode) in existing:
                continue
            set_seed(base_seed + si * 1000)
            stats = _evaluate_mode(
                swin=swin,
                receiver=receiver,
                edm_model=dm,
                loader=loader,
                channel_mix=mixture,
                channel_spec=spec,
                device=device,
                mode=mode,
                sampler_steps=sampler_steps,
                use_explicit_csi=use_explicit_csi or mode == 'diffusion_oracle_csi',
                oracle_use_token=oracle_use_token,
                channel_input_size=channel_input_size,
            )
            row: Dict[str, float | str] = {'method': name, 'channel': spec.name, 'mode': mode}
            row.update(stats)
            rows.append(row)
            existing[_row_key(row)] = row
            print(row)
            _write_results(out_dir, rows, request_cfg)

    external_csvs = request_cfg.get('comparison', {}).get('external_csvs', [])
    for path in external_csvs:
        for row in _read_external_csv(path):
            if {'method', 'channel', 'mode'} <= set(row):
                key = _row_key(row)
                if key in existing:
                    continue
                existing[key] = row
            rows.append(row)
            _write_results(out_dir, rows, request_cfg)

    _write_results(out_dir, rows, request_cfg)


if __name__ == '__main__':
    main()
