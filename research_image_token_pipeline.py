
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

CURRENT_ROOT = Path(__file__).resolve().parent
OLD_RESEARCH_ROOT = Path('/scratch/nickyun/diffusion-test01')
for path in (CURRENT_ROOT, OLD_RESEARCH_ROOT):
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)

from dataset.ffhq_dataset import FFHQDataset
from models.diffusion.loss import EDMLossSR, EDMLossSRChannel
from models.diffusion.networks import EDMPrecondSR, EDMPrecondSRChannel
from models.diffusion.sampling import edm_sampler_sr
from models.swin_ae.swin_unet import SwinUnet
from research.channels import ChannelMixture, ChannelSpec, build_default_channel_specs
from research.metrics import batch_psnr, batch_ssim, latent_cosine, linear_cka, summarize
from research.pilotless import PilotlessReceiver
from utils.data import get_default_transforms
from utils.training import load_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_channel_input_size(cfg: Mapping[str, Any], default: int = 16) -> int:
    if 'channel_input_size' in cfg:
        return int(cfg['channel_input_size'])
    pipeline_cfg = cfg.get('pipeline', {})
    if isinstance(pipeline_cfg, Mapping) and 'channel_input_size' in pipeline_cfg:
        return int(pipeline_cfg['channel_input_size'])
    return default


def build_channel_mixture(cfg: Mapping[str, Any]) -> ChannelMixture:
    backend = cfg.get('backend', 'sionna')
    specs_cfg = cfg.get('specs', None)
    if not specs_cfg:
        specs = build_default_channel_specs()
    else:
        specs = [ChannelSpec(**spec) for spec in specs_cfg]
    return ChannelMixture(specs=specs, backend=backend)


def load_module_state(
    module: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    strict: bool,
    module_name: str,
    is_main: bool,
) -> None:
    if strict:
        module.load_state_dict(state_dict, strict=True)
        return
    current = module.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    skipped = sorted(set(state_dict.keys()) - set(filtered.keys()))
    incompatible = module.load_state_dict(filtered, strict=False)
    if is_main and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(
            f'Warning: partial load for {module_name}; '
            f'missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)} '
            f'shape_skipped={len(skipped)}'
        )
        if skipped:
            print(f'  shape-skipped keys ({module_name}): {skipped}')
        if incompatible.missing_keys:
            print(f'  missing keys ({module_name}): {incompatible.missing_keys}')
        if incompatible.unexpected_keys:
            print(f'  unexpected keys ({module_name}): {incompatible.unexpected_keys}')


def build_loaders(
    cfg: Mapping[str, Any],
    *,
    is_dist: bool,
    rank: int,
    world_size: int,
) -> Tuple[DataLoader, DataLoader, DistributedSampler | None]:
    ds_cfg = cfg['dataset']
    train_root = ds_cfg['train_root']
    val_root = ds_cfg.get('val_root', train_root)
    img_size = ds_cfg.get('img_size', 224)
    batch_size = int(cfg.get('batch_size', 16))
    val_batch_size = int(cfg.get('val_batch_size', batch_size))
    num_workers = int(cfg.get('num_workers', 4))

    transform = get_default_transforms(img_size=img_size)
    train_ds = FFHQDataset(root_dir=train_root, transform=transform)
    val_ds = FFHQDataset(root_dir=val_root, transform=transform)

    train_sampler = None
    if is_dist:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_sampler


def build_eval_loader(cfg: Mapping[str, Any]) -> DataLoader:
    ds_cfg = cfg['dataset']
    for key in ('test_root', 'eval_root', 'val_root'):
        path = ds_cfg.get(key, '')
        if path:
            root = path
            break
    else:
        raise KeyError('dataset must define one of: test_root, eval_root, val_root')

    transform = get_default_transforms(img_size=ds_cfg.get('img_size', 224))
    ds = FFHQDataset(root_dir=root, transform=transform)
    return DataLoader(ds, batch_size=int(cfg.get('val_batch_size', 8)), shuffle=False, num_workers=int(cfg.get('num_workers', 4)))


def build_swin(cfg: Mapping[str, Any], device: torch.device) -> SwinUnet:
    model_cfg = cfg.get('model', {})
    swin = SwinUnet(depths=[6, 2], depths_decoder=[6, 2], embed_dim=12).to(device)
    swin_ckpt = model_cfg.get('swin_ckpt', '')
    if swin_ckpt:
        load_checkpoint(swin_ckpt, swin, device=str(device), strict=False)
    for p in swin.parameters():
        p.requires_grad = False
    swin.eval()
    return swin


def build_receiver(cfg: Mapping[str, Any], device: torch.device) -> PilotlessReceiver:
    model_cfg = cfg.get('model', {})
    token_dim = int(model_cfg.get('token_dim', 64))
    num_experts = int(model_cfg.get('num_experts', 4))
    expert_rank = int(model_cfg.get('expert_rank', 8))
    codebook_size = int(model_cfg.get('codebook_size', 512))
    return PilotlessReceiver(
        channels=24,
        token_dim=token_dim,
        codebook_size=codebook_size,
        num_experts=num_experts,
        expert_rank=expert_rank,
    ).to(device)


def build_edm(cfg: Mapping[str, Any], device: torch.device) -> Tuple[torch.nn.Module, torch.nn.Module]:
    model_cfg = cfg.get('model', {})
    token_dim = int(model_cfg.get('token_dim', 64))
    edm_variant = str(model_cfg.get('edm_variant', 'pilotless')).strip().lower()
    diffusion_ckpt = str(model_cfg.get('diffusion_ckpt', '')).strip()
    if edm_variant == 'oracle_csi':
        edm = EDMPrecondSRChannel(
            img_resolution=32,
            img_channels=24,
            cond_channels=24,
            sigma_data=0.06,
            label_dim=token_dim,
        ).to(device)
        loss = EDMLossSRChannel()
    else:
        edm = EDMPrecondSR(
            img_resolution=32,
            img_channels=24,
            cond_channels=24,
            sigma_data=0.06,
            label_dim=token_dim,
        ).to(device)
        loss = EDMLossSR()
    if diffusion_ckpt:
        ckpt = load_checkpoint(diffusion_ckpt, edm, device=str(device), strict=False)
        state = ckpt.get('model_state_dict', {}) if ckpt is not None else {}
        map_label = getattr(getattr(edm, 'model', None), 'map_label', None)
        if map_label is not None and not any('map_label' in key for key in state):
            # Preserve the unconditional published EDM behavior at step 0; token
            # conditioning can then grow from a neutral adapter.
            torch.nn.init.zeros_(map_label.weight)
            if map_label.bias is not None:
                torch.nn.init.zeros_(map_label.bias)
    return edm, loss


def load_research_modules(
    cfg: Mapping[str, Any],
    checkpoint_state: Mapping[str, Any],
    device: torch.device,
    *,
    strict_loading: bool = True,
    is_main: bool = False,
) -> Tuple[SwinUnet, PilotlessReceiver, torch.nn.Module, torch.nn.Module]:
    swin = build_swin(cfg, device)
    receiver = build_receiver(cfg, device)
    edm_model, diffusion_loss = build_edm(cfg, device)

    if checkpoint_state:
        if 'receiver' in checkpoint_state:
            load_module_state(receiver, checkpoint_state['receiver'], strict=strict_loading, module_name='receiver', is_main=is_main)
        elif is_main:
            raise KeyError('Checkpoint is missing receiver state')
        if 'edm_model' in checkpoint_state:
            load_module_state(edm_model, checkpoint_state['edm_model'], strict=strict_loading, module_name='edm_model', is_main=is_main)
        elif is_main:
            raise KeyError('Checkpoint is missing edm_model state')
        if 'transmitter' in checkpoint_state and is_main:
            print('Note: ignoring legacy transmitter state; the corrected pipeline operates on noisy images directly.')

    return swin, receiver, edm_model, diffusion_loss


def conditioning_labels(
    token: torch.Tensor,
    ch_meta: Mapping[str, torch.Tensor],
    *,
    label_mode: str,
) -> torch.Tensor:
    if label_mode == 'zero':
        return torch.zeros_like(token)
    if label_mode == 'token':
        return token
    if label_mode == 'oracle':
        oracle_label = ch_meta.get('oracle_label', None)
        if oracle_label is None:
            raise KeyError("Missing ch_meta['oracle_label'] for oracle conditioning")
        return oracle_label.to(device=token.device, dtype=token.dtype)
    raise ValueError(f'Unknown label_mode: {label_mode}')


def channelize_images(
    imgs: torch.Tensor,
    swin: SwinUnet,
    channel_mix: ChannelMixture,
    *,
    channel_input_size: int = 16,
    channel_spec: ChannelSpec | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    clean_latent = swin.encoder(imgs)
    imgs_lr = F.interpolate(
        imgs,
        size=(channel_input_size, channel_input_size),
        mode='bicubic',
        align_corners=False,
        antialias=True,
    )
    noisy_lr, ch_meta = channel_mix.apply(imgs_lr, spec=channel_spec)
    noisy_img = F.interpolate(noisy_lr, size=imgs.shape[-2:], mode='bilinear', align_corners=False)
    noisy_latent = swin.encoder(noisy_img)
    return clean_latent, noisy_latent, ch_meta


@torch.no_grad()
def evaluate_mode(
    *,
    swin: SwinUnet,
    receiver: PilotlessReceiver | None,
    edm_model: torch.nn.Module | None,
    loader: DataLoader,
    channel_mix: ChannelMixture,
    channel_spec: ChannelSpec,
    device: torch.device,
    mode: str,
    sampler_steps: int,
    use_explicit_csi: bool = False,
    oracle_use_token: bool = False,
    channel_input_size: int = 16,
    max_batches: int | None = None,
) -> Dict[str, float]:
    swin.eval()
    if receiver is not None:
        receiver.eval()
    if edm_model is not None:
        edm_model.eval()

    psnr_all, ssim_all, cos_all, cka_all = [], [], [], []
    for batch_idx, imgs in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        imgs = imgs.to(device)
        clean_latent, noisy_latent, ch_meta = channelize_images(
            imgs,
            swin,
            channel_mix,
            channel_input_size=channel_input_size,
            channel_spec=channel_spec,
        )
        needs_receiver = mode in {
            'transport',
            'tokenizer_cond',
            'diffusion_no_token',
            'diffusion_token',
            'diffusion_oracle_csi',
        } or edm_model is None
        if needs_receiver:
            if receiver is None:
                raise ValueError(f"Mode '{mode}' requires a receiver checkpoint.")
            pout = receiver(noisy_latent)
        else:
            pout = None

        if mode == 'transport' or edm_model is None:
            if pout is None:
                raise ValueError(f"Mode '{mode}' requires receiver outputs.")
            pred_latent = pout.restored
        elif mode == 'baseline':
            noise = torch.randn_like(clean_latent)
            pred_latent = edm_sampler_sr(
                edm_model,
                latents=noise,
                x_cond=noisy_latent,
                class_labels=None,
                h=ch_meta.get('h', None) if use_explicit_csi else None,
                num_steps=sampler_steps,
            ).float()
        elif mode == 'tokenizer_cond':
            if pout is None:
                raise ValueError("Mode 'tokenizer_cond' requires receiver outputs.")
            noise = torch.randn_like(clean_latent)
            pred_latent = edm_sampler_sr(
                edm_model,
                latents=noise,
                x_cond=pout.restored,
                class_labels=None,
                h=ch_meta.get('h', None) if use_explicit_csi else None,
                num_steps=sampler_steps,
            ).float()
        elif mode == 'diffusion_no_token':
            if pout is None:
                raise ValueError("Mode 'diffusion_no_token' requires receiver outputs.")
            noise = torch.randn_like(clean_latent)
            labels = conditioning_labels(pout.token, ch_meta, label_mode='zero')
            pred_latent = edm_sampler_sr(
                edm_model,
                latents=noise,
                x_cond=pout.restored,
                class_labels=labels,
                h=ch_meta.get('h', None) if use_explicit_csi else None,
                num_steps=sampler_steps,
            ).float()
        elif mode == 'diffusion_token':
            if pout is None:
                raise ValueError("Mode 'diffusion_token' requires receiver outputs.")
            noise = torch.randn_like(clean_latent)
            labels = conditioning_labels(pout.token, ch_meta, label_mode='token')
            pred_latent = edm_sampler_sr(
                edm_model,
                latents=noise,
                x_cond=pout.restored,
                class_labels=labels,
                h=ch_meta.get('h', None) if use_explicit_csi else None,
                num_steps=sampler_steps,
            ).float()
        elif mode == 'diffusion_oracle_csi':
            if pout is None:
                raise ValueError("Mode 'diffusion_oracle_csi' requires receiver outputs.")
            if not use_explicit_csi:
                raise ValueError("Mode 'diffusion_oracle_csi' requires model.edm_variant='oracle_csi'.")
            noise = torch.randn_like(clean_latent)
            label_mode = 'token' if oracle_use_token else 'oracle'
            labels = conditioning_labels(pout.token, ch_meta, label_mode=label_mode)
            pred_latent = edm_sampler_sr(
                edm_model,
                latents=noise,
                x_cond=pout.restored,
                class_labels=labels,
                h=ch_meta.get('h', None),
                num_steps=sampler_steps,
            ).float()
        else:
            raise ValueError(f'Unknown eval mode: {mode}')

        recon = swin.decoder(pred_latent)
        psnr_all.append(batch_psnr((imgs + 1) / 2, (recon + 1) / 2))
        ssim_all.append(batch_ssim(imgs, recon))
        cos_all.append(latent_cosine(clean_latent, pred_latent))
        cka_all.append(linear_cka(clean_latent, pred_latent).view(1))

    metric_map = {
        'psnr': torch.cat(psnr_all),
        'ssim': torch.cat(ssim_all),
        'latent_cosine': torch.cat(cos_all),
        'latent_cka': torch.cat(cka_all),
    }
    return summarize(metric_map)
