import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torchvision
import torch.nn.functional as F
from models.diffusion.networks import  EDMPrecondSR
from models.diffusion.loss import   EDMLossSR
from models.swin_ae.swin_unet import SwinUnet
from utils.training import get_edm_model, parse_training_args, log_info, load_checkpoint, save_checkpoint
from utils.data import get_default_transforms
from dataset.ffhq_dataset import FFHQDataset

from models.diffusion.channel import Channel

class ChannelConfig:
    """Minimal config to initialize CDDM's Channel class."""
    def __init__(self, channel_type='rayleigh', cuda=True):
        self.channel_type = channel_type
        self.CUDA = cuda



def main():
    args = parse_training_args(argparse.ArgumentParser()).parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    transform = get_default_transforms(img_size=224)
    ffhq_dataset = FFHQDataset(root_dir=args.train_data_root, transform=transform)
    train_loader = DataLoader(ffhq_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)

    print(f"Dataset: {len(ffhq_dataset)} images")

    swin_ae = SwinUnet(depths=[6,2], depths_decoder=[6,2], embed_dim=12).to(args.device)
    load_checkpoint(args.swin_path, swin_ae, args.device, strict=True)
    for param in swin_ae.parameters():
        param.requires_grad = False
    swin_ae.eval()

    edm_model = EDMPrecondSR(img_resolution=32, img_channels=24, cond_channels=24, sigma_data=0.06).to(args.device)
    load_checkpoint(args.load_ckpt, edm_model, args.device, strict=False)

    loss_fn = EDMLossSR()
    optimizer = optim.AdamW(edm_model.parameters(), lr=args.lr)

    log_info(edm_model, args, optimizer)

    channel_config = ChannelConfig(channel_type='awgn', cuda=args.device=='cuda')
    pass_channel = Channel(channel_config)

    device = args.device

    for epoch in range(args.epochs):
        edm_model.train()
        running_loss = 0.0

        # pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        # for imgs in pbar:
        for imgs in train_loader:
            imgs = imgs.to(device)
            imgs_lr = F.interpolate(imgs, size=(16, 16), mode='bicubic', align_corners=False, antialias=True)

            with torch.no_grad():
                # random between 0 and 20 snr
                chan_param = torch.randint(0, 21, (1,), device=device).item()
                noisy_lr_imgs = pass_channel(imgs_lr, chan_param=chan_param)
                noisy_lr_imgs = F.interpolate(noisy_lr_imgs, size=(224, 224), mode='bilinear', align_corners=False)

                ground_truth_latent_img = swin_ae.encoder(imgs)
                img_cond = swin_ae.encoder(noisy_lr_imgs)

            loss = loss_fn(net=edm_model, images=ground_truth_latent_img, x_cond=img_cond, labels=None).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = running_loss / len(train_loader)
        epoch_num = epoch + 1
        print(f"epoch {epoch_num} | loss {avg:.8f}")

        save_checkpoint(os.path.join(args.output_dir, "ckpt_last.pt"), epoch_num, edm_model, optimizer, torch.tensor(avg))
        if epoch_num % 10 == 0 or epoch_num == args.epochs:
            save_path = os.path.join(args.output_dir, f"awgn_edm_ffhq_ckpt_latent_222_{epoch_num}.pt")
            save_checkpoint(save_path, epoch_num, edm_model, optimizer, torch.tensor(avg))

if __name__ == "__main__":
    main()
