from __future__ import annotations

import argparse
import csv
from copy import deepcopy
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
from research.channels import ChannelMixture, ChannelSpec, build_default_channel_specs
from research.config import load_json_config, save_json
from research_image_token_pipeline import get_channel_input_size, set_seed


def _row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row['ablation']), str(row['channel'])


def _ordered_fieldnames(rows: List[Dict[str, float | str]]) -> List[str]:
    names: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in names:
                names.append(key)
    return names


def _load_existing_rows(out_dir: Path, wanted: set[tuple[str, str]]) -> Dict[tuple[str, str], Dict[str, float | str]]:
    rows: Dict[tuple[str, str], Dict[str, float | str]] = {}
    csv_path = out_dir / 'ablations.csv'
    if not csv_path.exists():
        return rows
    with csv_path.open('r', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if 'ablation' not in row or 'channel' not in row or _row_key(row) not in wanted:
                continue
            parsed: Dict[str, float | str] = {}
            for key, value in row.items():
                if key in {'ablation', 'ablation_protocol', 'channel', 'mode'}:
                    parsed[key] = value
                else:
                    try:
                        parsed[key] = float(value)
                    except (TypeError, ValueError):
                        parsed[key] = value
            rows[_row_key(parsed)] = parsed
    return rows


def _write_results(
    out_dir: Path,
    rows: List[Dict[str, float | str]],
    checkpoint: str,
    cfg: Dict[str, Any],
    strict_loading: bool,
) -> None:
    if not rows:
        return
    csv_path = out_dir / 'ablations.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=_ordered_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)
    save_json(
        str(out_dir / 'ablations.json'),
        {
            'rows': rows,
            'checkpoint': checkpoint,
            'config': cfg,
            'strict_checkpoint_loading': strict_loading,
        },
    )


def _force_single_expert(receiver):
    orig_forward = receiver.forward

    def wrapped(y):
        out = orig_forward(y)
        probs = torch.zeros_like(out.router_probs)
        probs[:, 0] = 1.0
        stacked = torch.stack([e(y) for e in receiver.experts], dim=1)
        mixed = torch.sum(stacked * probs[:, :, None, None, None], dim=1)
        return out._replace(restored=mixed, router_probs=probs)

    receiver.forward = wrapped
    return orig_forward


def _disable_vq(receiver):
    orig_vq = receiver.vq.forward

    def wrapped(z_e):
        b, t, _ = z_e.shape
        idx = torch.zeros(b, t, dtype=torch.long, device=z_e.device)
        return z_e, idx, torch.tensor(0.0, device=z_e.device)

    receiver.vq.forward = wrapped
    return orig_vq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='outputs/research/ablations')
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

    swin, receiver, dm, _ = _load_research_modules(cfg, state, device, strict_loading=strict_loading, is_main=True)
    loader = _build_loader(cfg)
    channel_cfg = cfg.get('channel', {})
    specs = [ChannelSpec(**s) for s in channel_cfg['specs']] if channel_cfg.get('specs') else build_default_channel_specs()
    mixture = ChannelMixture(specs=specs, backend=channel_cfg.get('backend', 'sionna'))
    sampler_steps = int(cfg.get('sampler_steps', 18))
    channel_input_size = get_channel_input_size(cfg)

    ablations = cfg.get('ablation', {}).get(
        'items',
        ['baseline', 'tokenizer_cond', 'no_token', 'single_expert', 'no_vq'],
    )
    planned = [(ab, spec) for ab in ablations for spec in specs]
    wanted = {(ab, spec.name) for ab, spec in planned}
    existing = _load_existing_rows(out_dir, wanted)
    rows: List[Dict[str, float | str]] = [
        existing[(ab, spec.name)]
        for ab, spec in planned
        if (ab, spec.name) in existing
    ]
    if rows:
        print(f"Resuming ablations with {len(rows)}/{len(planned)} completed rows")
        _write_results(out_dir, rows, args.checkpoint, cfg, strict_loading)

    for ab in ablations:
        for si, spec in enumerate(specs):
            if (ab, spec.name) in existing:
                continue
            set_seed(base_seed + si * 1000)
            rx_variant = deepcopy(receiver)
            mode = 'diffusion_token'

            if ab == 'full':
                mode = 'diffusion_token'
            elif ab == 'baseline':
                mode = 'baseline'
            elif ab == 'tokenizer_cond':
                mode = 'tokenizer_cond'
            elif ab == 'no_token':
                mode = 'diffusion_no_token'
            elif ab == 'single_expert':
                _force_single_expert(rx_variant)
            elif ab == 'no_vq':
                _disable_vq(rx_variant)
            elif ab == 'oracle_csi':
                if not use_explicit_csi:
                    raise ValueError("Ablation 'oracle_csi' requires model.edm_variant='oracle_csi'.")
                mode = 'diffusion_oracle_csi'
            else:
                raise ValueError(f'Unsupported ablation item: {ab}')

            stats = _evaluate_mode(
                swin=swin,
                receiver=rx_variant,
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
            row: Dict[str, float | str] = {
                'ablation': ab,
                'ablation_protocol': 'posthoc_eval_only',
                'channel': spec.name,
                'mode': mode,
            }
            row.update(stats)
            rows.append(row)
            existing[_row_key(row)] = row
            print(row)
            _write_results(out_dir, rows, args.checkpoint, cfg, strict_loading)

    _write_results(out_dir, rows, args.checkpoint, cfg, strict_loading)


if __name__ == '__main__':
    main()
