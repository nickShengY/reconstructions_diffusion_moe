import torch
import os
from models.diffusion.networks import EDMPrecond

def parse_training_args(p):
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output-dir", type=str, default="outputs/diffusion")
    p.add_argument("--load-ckpt", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument(
        "--train-data-root",
        type=str,
        default="/scratch/nickyun/hf_datasets/ffhq_raw/train",
        help="Root directory for FFHQ training images",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cond-class", type=int, default=0, help="Number of conditional class labels")
    p.add_argument("--num-workers", type=int, default=4, help="Number of cpus")
    p.add_argument("--swin-path", type=str, default="", help="load pretrained encoder")
    return p

def save_checkpoint(path, epoch, model, optimizer, loss):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.item(),
    }, path)


def log_gpu(args):
    print(f"start training, cuda: {torch.cuda.is_available()}, using device {args.device}, num_workers: {args.num_workers}, classes {args.cond_class} lr {args.lr}")
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i), "VRAM:", torch.cuda.get_device_properties(i).total_memory/1024**3, "GB")

def get_edm_model(cond_class, shape, channel, sigma_data=0.5, device="cuda"):
    net = EDMPrecond(
        img_resolution=shape,
        img_channels=channel,
        label_dim=cond_class,
        model_type='SongUNet',
    ).to(device)
    return net

def load_checkpoint(ckpt_path, model, optimizer=None, device="cuda", resume_opt=False, strict=False):
    if not ckpt_path:
        print(f"No checkpoint path provided, starting from scratch.")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        print(f"No checkpoint found")
        raise FileNotFoundError

    print(f"load model state dict with loss: {ckpt['loss']:.8f}")
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    model.to(device)

    if optimizer is not None and "optimizer_state_dict" in ckpt and resume_opt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print("Resumed optimizer state")

    return ckpt


def log_info(model, args, optimizer):
    if args.load_ckpt:
        print(f"[resume] {args.load_ckpt} ")

    if args.swin_path:
        print(f"[load swin] {args.swin_path} ")

    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i), "VRAM:", torch.cuda.get_device_properties(i).total_memory/1024**3, "GB")
    print(f"start training, cond-class: {args.cond_class} cuda: {torch.cuda.is_available()} using device {args.device}, num_workers: {args.num_workers} lr {args.lr}")
