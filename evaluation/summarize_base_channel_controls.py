from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


CHANNELS = ['awgn', 'rayleigh']


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _fmt(value: str | float, digits: int = 3) -> str:
    try:
        return f'{float(value):.{digits}f}'
    except (TypeError, ValueError):
        return str(value)


def _metric_table(rows: Iterable[Dict[str, str]], *, include_mode: bool = False) -> List[str]:
    out = ['| Channel | SNR dB | PSNR | SSIM | Latent cosine | Latent CKA |', '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        channel = row.get('channel', '')
        snr = row.get('snr_db', 'train-mixture')
        if include_mode and row.get('mode') not in {'', 'baseline'}:
            channel = f'{channel} ({row.get("mode")})'
        out.append(
            '| '
            + ' | '.join(
                [
                    channel,
                    _fmt(snr, 0) if snr != 'train-mixture' else snr,
                    _fmt(row.get('psnr_mean', '')),
                    _fmt(row.get('ssim_mean', '')),
                    _fmt(row.get('latent_cosine_mean', ''), 6),
                    _fmt(row.get('latent_cka_mean', '')),
                ]
            )
            + ' |'
        )
    return out


def _best_checkpoint(run_dir: Path) -> str:
    for name in ('ckpt_best.pt', 'ckpt_last.pt'):
        path = run_dir / 'diffusion' / name
        if path.exists():
            return str(path)
    return 'missing'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='outputs/base_channel_diffusion_controls_login')
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()

    root = Path(args.root)
    report = Path(args.output) if args.output else root / 'reports' / 'base_channel_diffusion_controls.md'
    report.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = [
        '# Base Channel-Specific Diffusion Control Study',
        '',
        'This control study trains two diffusion-only receivers with the frozen Swin autoencoder. The receiver/tokenizer/MoE path is not used. One model is trained only on AWGN channel samples, and the other model is trained only on Rayleigh channel samples.',
        '',
        'The reported mode is `baseline`, meaning diffusion refinement conditioned on the noisy Swin latent with no learned channel token.',
        '',
        '## Artifact Inventory',
        '| Model | Checkpoint | Eval CSV | SNR sweep CSV |',
        '|---|---|---|---|',
    ]

    for channel in CHANNELS:
        run_dir = root / channel
        lines.append(
            f'| {channel.upper()} diffusion | `{_best_checkpoint(run_dir)}` | '
            f'`{run_dir / "eval" / "eval_suite.csv"}` | `{run_dir / "snr_sweep" / "snr_sweep.csv"}` |'
        )

    lines.extend(['', '## Same-Channel Validation', ''])
    eval_rows = []
    for channel in CHANNELS:
        eval_rows.extend(_read_csv(root / channel / 'eval' / 'eval_suite.csv'))
    if eval_rows:
        lines.extend(_metric_table(eval_rows))
    else:
        lines.append('Validation results are not available yet.')

    lines.extend(['', '## SNR Sweep Results', ''])
    any_snr = False
    for channel in CHANNELS:
        rows = _read_csv(root / channel / 'snr_sweep' / 'snr_sweep.csv')
        lines.extend([f'### {channel.upper()}', ''])
        if rows:
            any_snr = True
            rows = sorted(rows, key=lambda row: float(row.get('snr_db', 0.0)))
            lines.extend(_metric_table(rows))
        else:
            lines.append('SNR sweep results are not available yet.')
        lines.append('')

    if any_snr:
        lines.extend(
            [
                '## Interpretation Guide',
                '',
                '- PSNR measures pixel-level reconstruction quality; higher is better.',
                '- SSIM measures structural similarity; higher is better.',
                '- Latent cosine and latent CKA measure feature-space agreement between clean and reconstructed Swin latents.',
                '- These are channel-specific control models, so each model should primarily be interpreted on its own trained channel.',
                '',
            ]
        )

    report.write_text('\n'.join(lines), encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
