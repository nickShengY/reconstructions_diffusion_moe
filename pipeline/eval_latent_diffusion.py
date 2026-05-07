import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

from dataset.ffhq_dataset import FFHQDataset
from models.diffusion.channel import Channel
from models.diffusion.networks import EDMPrecondSR
from models.diffusion.sampling import edm_sampler_sr
from models.swin_ae.swin_unet import SwinUnet
from utils.data import get_default_transforms


class ChannelConfig:
    def __init__(self, channel_type="rayleigh", cuda=True):
        self.channel_type = channel_type
        self.CUDA = cuda


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, default="/scratch/nickyun/hf_datasets/ffhq_raw/val")
    p.add_argument("--swin-ckpt", type=str, default="diffusion_model_ckpt/v1_ckpt_ffhq_222_500_09.pt")
    p.add_argument("--diffusion-ckpt", type=str, default="diffusion_model_ckpt/v2_edm_ffhq_ckpt_latent_222_470_10.pt")
    p.add_argument("--channel-type", type=str, default="rayleigh", choices=["awgn", "rayleigh"])
    p.add_argument("--snr-db", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="Optional cap on number of samples")
    p.add_argument("--mode", type=str, default="full", choices=["full", "original"], help="full = evaluate entire val split, original = one batch only")
    return p.parse_args()


def load_swin(ckpt_path, device):
    model = SwinUnet(depths=[6, 2], depths_decoder=[6, 2], embed_dim=12).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt


def load_diffusion(ckpt_path, device):
    model = EDMPrecondSR(img_resolution=32, img_channels=24, cond_channels=24, sigma_data=0.06).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt


def to_numpy_image_batch(tensor):
    return (tensor.clamp(-1, 1).mul(0.5).add(0.5)).permute(0, 2, 3, 1).detach().cpu().numpy()


def batch_metrics(target, pred):
    target_np = to_numpy_image_batch(target).astype(np.float32)
    pred_np = to_numpy_image_batch(pred).astype(np.float32)
    psnrs = []
    ssims = []
    for i in range(target_np.shape[0]):
        mse = np.mean((target_np[i] - pred_np[i]) ** 2)
        psnr = float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)
        psnrs.append(psnr)
        ssims.append(ssim(target_np[i], pred_np[i], channel_axis=-1, data_range=1.0, win_size=7))
    return psnrs, ssims


def main():
    args = parse_args()
    device = torch.device(args.device)

    dataset = FFHQDataset(root_dir=args.dataset_root, transform=get_default_transforms(img_size=224))
    if args.limit is not None:
        dataset.image_paths = dataset.image_paths[: args.limit]

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    swin, swin_ckpt = load_swin(args.swin_ckpt, device)
    diffusion, diff_ckpt = load_diffusion(args.diffusion_ckpt, device)

    channel = Channel(ChannelConfig(channel_type=args.channel_type, cuda=device.type == "cuda"))
    print(f"swin_ckpt_epoch={swin_ckpt.get('epoch')} loss={swin_ckpt.get('loss')}")
    print(f"diff_ckpt_epoch={diff_ckpt.get('epoch')} loss={diff_ckpt.get('loss')}")
    print(f"dataset={len(dataset)} channel={args.channel_type} snr={args.snr_db} steps={args.num_steps} device={device}")

    ae_psnr = []
    ae_ssim = []
    cond_psnr = []
    cond_ssim = []
    diff_psnr = []
    diff_ssim = []

    with torch.inference_mode():
        for batch_idx, imgs in enumerate(loader):
            imgs = imgs.to(device)

            recon = swin(imgs)
            p, s = batch_metrics(imgs, recon)
            ae_psnr.extend(p)
            ae_ssim.extend(s)

            imgs_lr = F.interpolate(imgs, size=(16, 16), mode="bicubic", align_corners=False, antialias=True)
            noisy_lr = channel(imgs_lr, chan_param=args.snr_db)
            noisy_lr = F.interpolate(noisy_lr, size=(224, 224), mode="bilinear", align_corners=False)

            x_cond = swin.encoder(noisy_lr)
            cond_img = swin.decoder(x_cond)
            p, s = batch_metrics(imgs, cond_img)
            cond_psnr.extend(p)
            cond_ssim.extend(s)

            latents = torch.randn_like(x_cond)
            out_latent = edm_sampler_sr(
                net=diffusion,
                latents=latents,
                x_cond=x_cond,
                class_labels=None,
                num_steps=args.num_steps,
            )
            out_img = swin.decoder(out_latent.float())
            p, s = batch_metrics(imgs, out_img)
            diff_psnr.extend(p)
            diff_ssim.extend(s)

            if args.mode == "original":
                break

            if batch_idx % 10 == 0:
                print(f"batch {batch_idx}: ae_psnr={np.mean(ae_psnr):.3f} cond_psnr={np.mean(cond_psnr):.3f} diff_psnr={np.mean(diff_psnr):.3f}")

    def summary(name, psnrs, ssims):
        print(f"{name}: n={len(psnrs)} psnr={np.mean(psnrs):.4f}±{np.std(psnrs):.4f} ssim={np.mean(ssims):.4f}±{np.std(ssims):.4f}")

    summary("AE clean recon", ae_psnr, ae_ssim)
    summary("Condition decode", cond_psnr, cond_ssim)
    summary("Diffusion output", diff_psnr, diff_ssim)


if __name__ == "__main__":
    main()
