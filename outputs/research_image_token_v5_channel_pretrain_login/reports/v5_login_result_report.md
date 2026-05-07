# Login-node v5 Result Report

Output tree: `outputs/research_image_token_v5_channel_pretrain_login/`

Completion: `login eval complete` was written by the final chain at `Sun May 3 05:43:29 EDT 2026`; the watchdog exited at `2026-05-03 05:45:20 EDT`.

## Artifact Inventory
| Artifact | Rows / count | Path | Notes |
| --- | --- | --- | --- |
| Diffusion history | 500 | diffusion/history.json | 500 epochs complete |
| Joint history | 100 | joint/history.json | 100 epochs complete |
| Eval suite | 50 | eval/eval_suite.csv | 10 channels x 5 modes |
| Ablations | 50 | ablations/ablations.csv | 10 channels x 5 ablations |
| Comparisons | 60 | comparisons/comparisons.csv | 10 channels x 6 methods |
| SNR sweep | 320 | snr_sweep/snr_sweep.csv | 10 channels x 8 SNRs x 4 modes |
| Generated tables | 17 | tables/ | 16 md/tex metric tables + ARTIFACTS.txt |

## Training Checkpoints
| Stage | Epoch | Val PSNR | Val SSIM | Val latent CKA | Checkpoint |
| --- | --- | --- | --- | --- | --- |
| Diffusion best | 475 | 24.596 | 0.675 | 0.977 | diffusion/ckpt_best.pt |
| Diffusion final | 500 | 24.493 | 0.678 | 0.978 | diffusion/ckpt_last.pt / ckpt_diffusion_500.pt |
| Joint best | 75 | 24.144 | 0.656 | 0.797 | joint/ckpt_best.pt |
| Joint final | 100 | 21.971 | 0.529 | 0.842 | joint/ckpt_last.pt / ckpt_joint_100.pt |

## Eval Suite: Mean Across 10 Channels
| Mode | N | PSNR | SSIM | Latent cosine | Latent CKA |
| --- | --- | --- | --- | --- | --- |
| baseline | 10 | 18.141 | 0.287 | 0.999753 | 0.925 |
| diffusion_no_token | 10 | 17.089 | 0.240 | 0.999687 | 0.900 |
| diffusion_token | 10 | 23.836 | 0.656 | 0.999926 | 0.659 |
| tokenizer_cond | 10 | 17.084 | 0.240 | 0.999687 | 0.900 |
| transport | 10 | 22.980 | 0.661 | 0.997686 | 0.951 |

## Eval Suite: Channel-Level PSNR and Token Gain
| Channel | Baseline PSNR | No-token diff PSNR | Token diff PSNR | Token gain PSNR | Transport PSNR | Token diff SSIM | Token gain SSIM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| awgn | 18.392 | 17.300 | 24.462 | 7.162 | 23.730 | 0.663 | 0.411 |
| cdl | 18.360 | 17.256 | 24.507 | 7.252 | 24.160 | 0.670 | 0.426 |
| highway | 17.982 | 17.019 | 23.153 | 6.134 | 21.438 | 0.633 | 0.396 |
| rayleigh | 18.004 | 17.053 | 23.846 | 6.793 | 22.324 | 0.654 | 0.419 |
| rician | 18.295 | 17.226 | 24.277 | 7.050 | 23.583 | 0.662 | 0.420 |
| rural | 17.918 | 16.855 | 23.170 | 6.314 | 22.677 | 0.654 | 0.417 |
| tdl | 18.182 | 17.101 | 23.962 | 6.860 | 23.180 | 0.661 | 0.422 |
| uma | 18.034 | 16.995 | 23.513 | 6.518 | 22.688 | 0.651 | 0.415 |
| umi | 18.196 | 17.130 | 24.055 | 6.924 | 23.385 | 0.660 | 0.420 |
| urban | 18.043 | 16.954 | 23.421 | 6.467 | 22.635 | 0.651 | 0.414 |

## Comparison Summary
| Method | N | PSNR | SSIM | Latent cosine | Latent CKA |
| --- | --- | --- | --- | --- | --- |
| published_baseline | 10 | 6.462 | 0.018 | 0.871571 | 0.933 |
| v5_full | 10 | 23.832 | 0.656 | 0.999926 | 0.659 |
| v5_internal_baseline | 10 | 18.142 | 0.287 | 0.999753 | 0.925 |
| v5_no_token | 10 | 17.086 | 0.240 | 0.999687 | 0.900 |
| v5_tokenizer_cond | 10 | 17.085 | 0.240 | 0.999687 | 0.900 |
| v5_transport | 10 | 22.976 | 0.660 | 0.997686 | 0.950 |

## Ablation Summary
Deltas are relative to `v5_full` from the comparison suite.
| Ablation | N | PSNR | Delta PSNR | SSIM | Delta SSIM | Latent CKA | Delta CKA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 10 | 18.134 | -5.698 | 0.287 | -0.369 | 0.925 | 0.267 |
| no_token | 10 | 17.089 | -6.743 | 0.240 | -0.416 | 0.900 | 0.242 |
| no_vq | 10 | 20.738 | -3.094 | 0.439 | -0.217 | 0.797 | 0.139 |
| single_expert | 10 | 6.662 | -17.170 | 0.016 | -0.640 | 0.933 | 0.275 |
| tokenizer_cond | 10 | 17.086 | -6.745 | 0.240 | -0.416 | 0.900 | 0.242 |

## Tokenizer Contribution
| Contrast | Delta PSNR | Delta SSIM | Delta latent CKA | Interpretation |
| --- | --- | --- | --- | --- |
| Full tokenized diffusion vs no-token diffusion | 6.747 | 0.416 | -0.242 | Primary tokenizer-conditioned gain in image quality |
| Full tokenized diffusion vs internal baseline | 5.696 | 0.369 | -0.267 | Full path gain over non-diffusion baseline |
| Tokenizer-cond alone vs no-token diffusion | -0.005 | -0.000 | -0.000 | Near zero: tokenizer path alone does not help without diffusion token use |
| Full tokenized diffusion vs transport | 0.856 | -0.005 | -0.292 | Higher PSNR; transport has slightly higher mean SSIM/CKA |

## SNR Sweep: Tokenized Diffusion Gain by Channel
Gain is `diffusion_token - diffusion_no_token`, averaged over SNR levels 0, 3, 6, 9, 12, 15, 18, and 20 dB.
| Channel | N SNRs | Mean PSNR gain | Min PSNR gain | Max PSNR gain | Mean SSIM gain |
| --- | --- | --- | --- | --- | --- |
| awgn | 8 | 7.165 | 5.490 | 7.949 | 0.411 |
| cdl | 8 | 7.017 | 5.325 | 7.894 | 0.420 |
| highway | 8 | 5.167 | 3.944 | 5.661 | 0.358 |
| rayleigh | 8 | 6.688 | 4.693 | 7.833 | 0.416 |
| rician | 8 | 7.061 | 5.411 | 7.899 | 0.421 |
| rural | 8 | 5.745 | 3.981 | 6.898 | 0.403 |
| tdl | 8 | 6.534 | 4.655 | 7.722 | 0.413 |
| uma | 8 | 6.251 | 4.431 | 7.524 | 0.408 |
| umi | 8 | 6.632 | 4.813 | 7.740 | 0.413 |
| urban | 8 | 6.218 | 4.335 | 7.487 | 0.408 |

## SNR Sweep: Tokenized Diffusion Gain by SNR
| SNR dB | Mean PSNR gain | Mean SSIM gain |
| --- | --- | --- |
| 0 | 4.708 | 0.362 |
| 3 | 5.399 | 0.379 |
| 6 | 5.993 | 0.395 |
| 9 | 6.517 | 0.409 |
| 12 | 6.922 | 0.420 |
| 15 | 7.196 | 0.427 |
| 18 | 7.385 | 0.432 |
| 20 | 7.461 | 0.434 |

## Top Eval Rows by PSNR
| Channel | Mode | PSNR | SSIM | Latent CKA |
| --- | --- | --- | --- | --- |
| cdl | diffusion_token | 24.507 | 0.670 | 0.653 |
| awgn | diffusion_token | 24.462 | 0.663 | 0.663 |
| rician | diffusion_token | 24.277 | 0.662 | 0.660 |
| cdl | transport | 24.160 | 0.695 | 0.963 |
| umi | diffusion_token | 24.055 | 0.660 | 0.667 |
| tdl | diffusion_token | 23.962 | 0.661 | 0.659 |
| rayleigh | diffusion_token | 23.846 | 0.654 | 0.665 |
| awgn | transport | 23.730 | 0.663 | 0.952 |
| rician | transport | 23.583 | 0.669 | 0.961 |
| uma | diffusion_token | 23.513 | 0.651 | 0.657 |
| urban | diffusion_token | 23.421 | 0.651 | 0.650 |
| umi | transport | 23.385 | 0.672 | 0.952 |

## Generated Table Files
| File | Bytes |
| --- | --- |
| ARTIFACTS.txt | 1155 |
| ablation_latent_cka_mean.md | 659 |
| ablation_latent_cka_mean.tex | 669 |
| ablation_latent_cosine_mean.md | 659 |
| ablation_latent_cosine_mean.tex | 669 |
| ablation_psnr_mean.md | 699 |
| ablation_psnr_mean.tex | 709 |
| ablation_ssim_mean.md | 659 |
| ablation_ssim_mean.tex | 669 |
| comparison_latent_cka_mean.md | 795 |
| comparison_latent_cka_mean.tex | 800 |
| comparison_latent_cosine_mean.md | 795 |
| comparison_latent_cosine_mean.tex | 800 |
| comparison_psnr_mean.md | 845 |
| comparison_psnr_mean.tex | 850 |
| comparison_ssim_mean.md | 795 |
| comparison_ssim_mean.tex | 800 |

## Interpretation
- The strongest image-quality result is `diffusion_token`: mean PSNR `23.836` and SSIM `0.656` across channels.
- The tokenized diffusion path adds `+6.747` PSNR and `+0.416` SSIM over `diffusion_no_token` in the eval suite, and `+6.548` PSNR / `+0.405` SSIM averaged over the full SNR sweep.
- `tokenizer_cond` alone is effectively identical to `diffusion_no_token`, so the tokenizer is not contributing much as a standalone receiver condition. Its contribution appears when the diffusion path consumes the learned token representation.
- `no_vq` lands between baseline/no-token and full tokenized diffusion: this suggests the quantized/token representation contributes materially, but the full tokenized diffusion stack is needed for the best reconstruction quality.
- Latent CKA is higher for baseline/transport than full tokenized diffusion, so tokenized diffusion trades latent-space similarity for much stronger pixel/structural reconstruction metrics.
