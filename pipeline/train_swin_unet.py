import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torchvision
from models.swin_ae.swin_unet import SwinUnet
from tqdm import tqdm
import os, glob, argparse, torch
from utils.training import get_edm_model, parse_training_args, log_info, load_checkpoint, save_checkpoint
from utils.data import get_default_transforms
from dataset.ffhq_dataset import FFHQDataset


def main():
    args = parse_training_args(argparse.ArgumentParser()).parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    transform = get_default_transforms(img_size=224)
    ffhq_dataset = FFHQDataset(root_dir=args.train_data_root, transform=transform)
    train_loader = DataLoader(ffhq_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    print(f"Dataset: {len(ffhq_dataset)} images")

    device = torch.device(args.device)
    model = SwinUnet(depths=[6,2], depths_decoder=[6,2], embed_dim=12).to(device)

    load_checkpoint(args.load_ckpt, model)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), args.lr)

    log_info(model, args, optimizer)
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        # pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        # for imgs in pbar:
        for imgs in train_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)

            loss = criterion(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = running_loss / len(train_loader)
        print(f"epoch {epoch} | loss {avg:.6f}")

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "encoder": model.encoder.state_dict(),
                "decoder": model.decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg,
            },os.path.join(args.output_dir, f"v1_ckpt_ffhq_222_{epoch}.pt"))

if __name__ == "__main__":
    main()
