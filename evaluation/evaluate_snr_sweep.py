from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Dict, List
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
from research.channels import ChannelMixture, ChannelSpec, build_default_channel_specs
from research.config import load_json_config, save_json
from research_image_token_pipeline import get_channel_input_size, set_seed


SNR_POINTS_DB = [0, 3, 6, 9, 12, 15, 18, 20]
DEFAULT_CHANNELS = ['awgn', 'rayleigh', 'rician', 'tdl', 'cdl', 'urban', 'rural', 'highway', 'umi', 'uma']


def _stable_seed(base_seed: int, channel: str, snr_db: float, mode: str) -> int:
    payload = f'{base_seed}|{channel}|{snr_db:.4f}|{mode}'.encode('utf-8')
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='outputs/research/snr_sweep')
    parser.add_argument('--snr-points', type=float, nargs='+', default=SNR_POINTS_DB)
    parser.add_argument('--channels', type=str, nargs='+', default=['all'])
    parser.add_argument('--modes', type=str, nargs='+', default=['baseline', 'tokenizer_cond', 'diffusion_no_token', 'diffusion_token'])
    args = parser.parse_args()

    request_cfg = load_json_config(args.config)
    base_seed = int(request_cfg.get('seed', 42))
    set_seed(base_seed)
    device = torch.device(request_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = _load_research_checkpoint(args.checkpoint, device)
    cfg = _prepare_effective_cfg(request_cfg, state)
    strict_loading = bool(cfg.get('strict_checkpoint_loading', True))
    model_cfg = cfg.get('model', {})
    use_explicit_csi = str(model_cfg.get('edm_variant', 'pilotless')).strip().lower() == 'oracle_csi'
    oracle_use_token = bool(model_cfg.get('oracle_use_token', False))

    swin, receiver, dm, _ = _load_research_modules(cfg, state, device, strict_loading=strict_loading)
    loader = _build_loader(cfg)
    backend = cfg.get('channel', {}).get('backend', 'sionna')
    sampler_steps = int(cfg.get('sampler_steps', 18))
    channel_input_size = get_channel_input_size(cfg)

    base_specs = {spec.name: spec for spec in build_default_channel_specs()}
    channels_to_sweep = DEFAULT_CHANNELS if 'all' in args.channels else args.channels
    for ch in channels_to_sweep:
        if ch not in base_specs:
            raise ValueError(f'Unsupported channel for sweep: {ch}')

    csv_path = out_dir / 'snr_sweep.csv'
    rows: List[Dict[str, float | str]] = []
    completed = set()
    if csv_path.exists():
        with csv_path.open('r', newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                rows.append(dict(row))
                completed.add((row['channel'], float(row['snr_db']), row['mode']))

    total = len(channels_to_sweep) * len(args.snr_points) * len(args.modes)
    done = len(completed)
    file_mode = 'a' if completed else 'w'

    with csv_path.open(file_mode, newline='', encoding='utf-8') as f:
        writer = None
        for ch_name in channels_to_sweep:
            template = base_specs[ch_name]
            for snr_db in args.snr_points:
                spec = ChannelSpec(
                    name=template.name,
                    probability=1.0,
                    snr_db_min=float(snr_db),
                    snr_db_max=float(snr_db),
                    k_factor=template.k_factor,
                    phase_noise_std=template.phase_noise_std,
                    iq_gain_imbalance=template.iq_gain_imbalance,
                    impulsive_prob=template.impulsive_prob,
                    impulsive_scale=template.impulsive_scale,
                    freq_selective=template.freq_selective,
                    max_taps=template.max_taps,
                )
                mixture = ChannelMixture(specs=[spec], backend=backend)
                for mode in args.modes:
                    key = (ch_name, float(snr_db), mode)
                    if key in completed:
                        continue
                    set_seed(_stable_seed(base_seed, ch_name, float(snr_db), mode))
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
                        use_explicit_csi=use_explicit_csi,
                        oracle_use_token=oracle_use_token,
                        channel_input_size=channel_input_size,
                    )
                    row: Dict[str, float | str] = {'channel': ch_name, 'snr_db': snr_db, 'mode': mode}
                    row.update(stats)
                    rows.append(row)
                    done += 1
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                        if not completed:
                            writer.writeheader()
                    writer.writerow(row)
                    f.flush()
                    print(
                        f'[{done}/{total}] {ch_name} SNR={snr_db:+.0f}dB {mode}: '
                        f'PSNR={stats["psnr_mean"]:.2f} SSIM={stats["ssim_mean"]:.3f}'
                    )

    save_json(
        str(out_dir / 'snr_sweep.json'),
        {
            'rows': rows,
            'checkpoint': args.checkpoint,
            'config': cfg,
            'snr_points': args.snr_points,
            'channels': channels_to_sweep,
            'strict_checkpoint_loading': strict_loading,
        },
    )


if __name__ == '__main__':
    main()
