# Base Channel-Specific Diffusion Control Study

This control study trains two diffusion-only receivers with the frozen Swin autoencoder. The receiver/tokenizer/MoE path is not used. One model is trained only on AWGN channel samples, and the other model is trained only on Rayleigh channel samples.

The reported mode is `baseline`, meaning diffusion refinement conditioned on the noisy Swin latent with no learned channel token.

## Artifact Inventory
| Model | Checkpoint | Eval CSV | SNR sweep CSV |
|---|---|---|---|
| AWGN diffusion | `outputs/base_channel_diffusion_controls_login/awgn/diffusion/ckpt_best.pt` | `outputs/base_channel_diffusion_controls_login/awgn/eval/eval_suite.csv` | `outputs/base_channel_diffusion_controls_login/awgn/snr_sweep/snr_sweep.csv` |
| RAYLEIGH diffusion | `outputs/base_channel_diffusion_controls_login/rayleigh/diffusion/ckpt_best.pt` | `outputs/base_channel_diffusion_controls_login/rayleigh/eval/eval_suite.csv` | `outputs/base_channel_diffusion_controls_login/rayleigh/snr_sweep/snr_sweep.csv` |

## Same-Channel Validation

| Channel | SNR dB | PSNR | SSIM | Latent cosine | Latent CKA |
|---|---:|---:|---:|---:|---:|
| awgn | train-mixture | 25.183 | 0.681 | 0.999945 | 0.982 |
| rayleigh | train-mixture | 24.559 | 0.679 | 0.999949 | 0.975 |

## SNR Sweep Results

### AWGN

| Channel | SNR dB | PSNR | SSIM | Latent cosine | Latent CKA |
|---|---:|---:|---:|---:|---:|
| awgn | 0 | 22.637 | 0.547 | 0.999925 | 0.983 |
| awgn | 3 | 23.782 | 0.611 | 0.999936 | 0.983 |
| awgn | 6 | 24.696 | 0.660 | 0.999943 | 0.983 |
| awgn | 9 | 25.392 | 0.695 | 0.999948 | 0.983 |
| awgn | 12 | 25.871 | 0.717 | 0.999950 | 0.983 |
| awgn | 15 | 26.208 | 0.732 | 0.999952 | 0.983 |
| awgn | 18 | 26.405 | 0.741 | 0.999953 | 0.983 |
| awgn | 20 | 26.473 | 0.743 | 0.999954 | 0.983 |

### RAYLEIGH

| Channel | SNR dB | PSNR | SSIM | Latent cosine | Latent CKA |
|---|---:|---:|---:|---:|---:|
| rayleigh | 0 | 22.050 | 0.592 | 0.999938 | 0.963 |
| rayleigh | 3 | 23.091 | 0.623 | 0.999942 | 0.968 |
| rayleigh | 6 | 23.893 | 0.653 | 0.999946 | 0.972 |
| rayleigh | 9 | 24.667 | 0.682 | 0.999950 | 0.976 |
| rayleigh | 12 | 25.256 | 0.705 | 0.999952 | 0.978 |
| rayleigh | 15 | 25.651 | 0.723 | 0.999955 | 0.979 |
| rayleigh | 18 | 25.917 | 0.735 | 0.999956 | 0.980 |
| rayleigh | 20 | 26.035 | 0.741 | 0.999956 | 0.981 |

## Interpretation Guide

- PSNR measures pixel-level reconstruction quality; higher is better.
- SSIM measures structural similarity; higher is better.
- Latent cosine and latent CKA measure feature-space agreement between clean and reconstructed Swin latents.
- These are channel-specific control models, so each model should primarily be interpreted on its own trained channel.
