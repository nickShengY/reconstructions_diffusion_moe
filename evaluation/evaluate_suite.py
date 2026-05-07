from __future__ import annotations

import argparse
import ast
import csv
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping
import random
import sys

import numpy as np
import torch

CURRENT_ROOT = Path(__file__).resolve().parents[1]
OLD_RESEARCH_ROOT = Path('/scratch/nickyun/diffusion-test01')
for path in (OLD_RESEARCH_ROOT, CURRENT_ROOT):
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)

from research.channels import ChannelMixture, ChannelSpec, build_default_channel_specs
from research.config import load_json_config, save_json
from research_image_token_pipeline import (
    build_eval_loader,
    evaluate_mode,
    get_channel_input_size,
    load_research_modules,
    set_seed,
)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_research_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def _prepare_effective_cfg(
    eval_cfg: Dict[str, Any],
    checkpoint_state: Mapping[str, Any],
    model_overrides: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    checkpoint_cfg = deepcopy(checkpoint_state.get('config', {}))
    effective = checkpoint_cfg if checkpoint_cfg else deepcopy(eval_cfg)

    if eval_cfg:
        protocol_override = deepcopy(eval_cfg)
        model_override = {}
        for key in ('swin_ckpt', 'oracle_use_token'):
            if key in protocol_override.get('model', {}):
                model_override[key] = protocol_override['model'][key]
        protocol_override.pop('model', None)
        effective = _deep_merge(effective, protocol_override)
        if model_override:
            effective['model'] = _deep_merge(effective.get('model', {}), model_override)

    if model_overrides:
        effective['model'] = _deep_merge(effective.get('model', {}), model_overrides)

    return effective


_build_loader = build_eval_loader
_load_research_modules = load_research_modules
_evaluate_mode = evaluate_mode


def _row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row['channel']), str(row['mode'])


def _ordered_fieldnames(rows: List[Dict[str, float | str]]) -> List[str]:
    names: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in names:
                names.append(key)
    return names


def _write_results(
    out_dir: Path,
    rows: List[Dict[str, float | str]],
    checkpoint: str,
    cfg: Dict[str, Any],
    strict_loading: bool,
) -> None:
    if not rows:
        return

    csv_path = out_dir / 'eval_suite.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=_ordered_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)

    save_json(
        str(out_dir / 'eval_suite.json'),
        {
            'rows': rows,
            'checkpoint': checkpoint,
            'config': cfg,
            'eval_root': cfg['dataset'].get('val_root', cfg['dataset'].get('eval_root', cfg['dataset'].get('test_root', ''))),
            'strict_checkpoint_loading': strict_loading,
        },
    )


def _load_existing_rows(out_dir: Path, wanted: set[tuple[str, str]]) -> Dict[tuple[str, str], Dict[str, float | str]]:
    rows: Dict[tuple[str, str], Dict[str, float | str]] = {}
    csv_path = out_dir / 'eval_suite.csv'
    if csv_path.exists():
        with csv_path.open('r', newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if 'channel' in row and 'mode' in row and _row_key(row) in wanted:
                    parsed: Dict[str, float | str] = {}
                    for key, value in row.items():
                        if key in {'channel', 'mode'}:
                            parsed[key] = value
                        else:
                            try:
                                parsed[key] = float(value)
                            except (TypeError, ValueError):
                                parsed[key] = value
                    rows[_row_key(parsed)] = parsed

    # Login-node evals can be killed before the first CSV write in older code.
    # Recover those completed rows from tee logs so the next run only computes
    # missing channel/mode pairs.
    logs_dir = out_dir.parent / 'logs'
    if logs_dir.exists():
        for log_path in sorted(logs_dir.glob('eval_suite_*.log')):
            with log_path.open('r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    text = line.strip()
                    if not text.startswith("{'channel':"):
                        continue
                    try:
                        row = ast.literal_eval(text)
                    except (SyntaxError, ValueError):
                        continue
                    if isinstance(row, dict) and 'channel' in row and 'mode' in row and _row_key(row) in wanted:
                        rows[_row_key(row)] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True, help='Research checkpoint from train_research_system.py')
    parser.add_argument('--output', type=str, default='outputs/research/eval')
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

    swin, receiver, dm, _ = _load_research_modules(cfg, state, device, strict_loading=strict_loading, is_main=True)
    loader = _build_loader(cfg)
    channel_cfg = cfg.get('channel', {})
    if channel_cfg.get('specs'):
        specs = [ChannelSpec(**s) for s in channel_cfg['specs']]
    else:
        specs = build_default_channel_specs()
    mixture = ChannelMixture(specs=specs, backend=channel_cfg.get('backend', 'sionna'))

    modes = cfg.get('eval', {}).get('modes', ['baseline', 'tokenizer_cond', 'diffusion_no_token', 'diffusion_token'])
    sampler_steps = int(cfg.get('sampler_steps', 18))
    model_cfg = cfg.get('model', {})
    use_explicit_csi = str(model_cfg.get('edm_variant', 'pilotless')).strip().lower() == 'oracle_csi'
    oracle_use_token = bool(model_cfg.get('oracle_use_token', False))
    channel_input_size = get_channel_input_size(cfg)

    planned = [(spec, mode) for spec in specs for mode in modes]
    wanted = {(spec.name, mode) for spec, mode in planned}
    existing = _load_existing_rows(out_dir, wanted)
    rows: List[Dict[str, float | str]] = [
        existing[(spec.name, mode)]
        for spec, mode in planned
        if (spec.name, mode) in existing
    ]
    if rows:
        print(f"Resuming eval_suite with {len(rows)}/{len(planned)} completed rows")
        _write_results(out_dir, rows, args.checkpoint, cfg, strict_loading)

    for si, spec in enumerate(specs):
        for mode in modes:
            if (spec.name, mode) in existing:
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
                use_explicit_csi=use_explicit_csi,
                oracle_use_token=oracle_use_token,
                channel_input_size=channel_input_size,
            )
            row: Dict[str, float | str] = {'channel': spec.name, 'mode': mode}
            row.update(stats)
            rows.append(row)
            existing[_row_key(row)] = row
            print(row)
            _write_results(out_dir, rows, args.checkpoint, cfg, strict_loading)

    _write_results(out_dir, rows, args.checkpoint, cfg, strict_loading)


if __name__ == '__main__':
    main()
