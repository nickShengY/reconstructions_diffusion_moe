
# Reconstruction Diffusion MoE

Code for channel-aware image reconstruction with Swin autoencoding,
channel tokenization, expert routing, and latent diffusion refinement.

## Repository Contents

- `research_image_token_pipeline.py`: core channel-tokenizer and receiver
  modules.
- `pipeline/`: training and evaluation entrypoints for Swin AE, latent
  diffusion, and the staged research system.
- `models/`: diffusion and Swin autoencoder model definitions.
- `dataset/` and `utils/`: dataset loading and training utilities.
- `configs/`: production research configs for transport, diffusion, joint
  training, and evaluation.
- `evaluation/`: metric suites, ablations, comparisons, SNR sweeps, table
  generation, and sample generation.
- `diffusion_model_ckpt/`: LFS-tracked model checkpoints.
- `paper_figures/`: generated channel-token visualization figures.

## Included Checkpoints

- `outputs/research_image_token_v5_channel_pretrain_login/diffusion/ckpt_best.pt`
  - epoch 475, validation PSNR 24.596, SSIM 0.675
- `outputs/research_image_token_v5_channel_pretrain_login/joint/ckpt_best.pt`
  - epoch 75, validation PSNR 24.144, SSIM 0.656

Evaluation artifacts for that run are included under the same output tree,
including `reports/v5_login_result_report.md`, `eval/`, `comparisons/`,
`ablations/`, `snr_sweep/`, and generated metric tables.

The joint v5 checkpoint contains the channel tokenizer, vector-quantization
codebook, router, 10-expert MoE receiver, low-rank expert adapters, token
affine layers, and token-conditioned diffusion state.

Channel-specific diffusion control checkpoints are also included:

- `outputs/base_channel_diffusion_controls_login/awgn/diffusion/ckpt_best.pt`
  - AWGN-only control, same-channel PSNR 25.183, SSIM 0.681
- `outputs/base_channel_diffusion_controls_login/rayleigh/diffusion/ckpt_best.pt`
  - Rayleigh-only control, same-channel PSNR 24.559, SSIM 0.679

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Checkpoints are stored with Git LFS. After cloning, make sure the checkpoint
files are downloaded before running evaluation:

```bash
git lfs install
git lfs pull
```

## Main Commands

Transport stage:

```bash
python pipeline/train_research_system.py --config configs/research_train_transport_v2.json
```

Diffusion stage:

```bash
python pipeline/train_research_system.py --config configs/research_train_diffusion_v2.json
```

Joint stage:

```bash
python pipeline/train_research_system.py --config configs/research_train_joint_v2.json
```

Evaluation:

```bash
python evaluation/evaluate_suite.py \
  --config configs/research_eval_v2.json \
  --checkpoint outputs/research_image_token_v5_channel_pretrain_login/joint/ckpt_best.pt \
  --output outputs/research_image_token_v5_channel_pretrain_login/eval
```

Visual samples matching the paper-style comparison:

```bash
python evaluation/generate_visual_samples.py \
  --config configs/research_eval_v2.json \
  --checkpoint outputs/research_image_token_v5_channel_pretrain_login/joint/ckpt_best.pt \
  --channels awgn rayleigh uma \
  --num-samples 4 \
  --sampler-steps 18 \
  --include-published \
  --output outputs/research_image_token_v5_channel_pretrain_login/visual_samples
```

This command writes one contact sheet per channel. The important point is that
`generate_visual_samples.py` expects the full receiver checkpoint when producing
the token-guided visual comparison. Use
`outputs/research_image_token_v5_channel_pretrain_login/joint/ckpt_best.pt`
for this. Do not use the AWGN-only or Rayleigh-only diffusion-control
checkpoints for the full token-guided visual sheet; those controls are
diffusion-only baselines and are meaningful only in the baseline/noisy-latent
diffusion mode.

The `--include-published` flag adds the published unconditional EDM baseline
from:

```text
diffusion_model_ckpt/v2_edm_ffhq_ckpt_latent_222_470_10.pt
```

Channel-specific diffusion-only controls can be evaluated separately:

```bash
python evaluation/evaluate_suite.py \
  --config outputs/base_channel_diffusion_controls_login/awgn/diffusion/resolved_config.json \
  --checkpoint outputs/base_channel_diffusion_controls_login/awgn/diffusion/ckpt_best.pt \
  --output outputs/base_channel_diffusion_controls_login/awgn/eval

python evaluation/evaluate_suite.py \
  --config outputs/base_channel_diffusion_controls_login/rayleigh/diffusion/resolved_config.json \
  --checkpoint outputs/base_channel_diffusion_controls_login/rayleigh/diffusion/ckpt_best.pt \
  --output outputs/base_channel_diffusion_controls_login/rayleigh/eval
```
