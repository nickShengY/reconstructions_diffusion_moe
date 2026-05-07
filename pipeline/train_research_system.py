from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
from typing import Dict, Tuple
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

CURRENT_ROOT = Path(__file__).resolve().parents[1]
if str(CURRENT_ROOT) not in sys.path:
    sys.path.insert(0, str(CURRENT_ROOT))
OLD_RESEARCH_ROOT = Path('/scratch/nickyun/diffusion-test01')
if str(OLD_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(OLD_RESEARCH_ROOT))

from research.config import load_json_config, save_json
from research_image_token_pipeline import (
    build_channel_mixture,
    build_edm,
    build_loaders,
    build_receiver,
    build_swin,
    channelize_images,
    conditioning_labels,
    evaluate_mode,
    get_channel_input_size,
    load_module_state,
    set_seed,
)


def distributed_env() -> Tuple[bool, int, int, int]:
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def maybe_init_distributed(require_cuda: bool) -> Tuple[bool, int, int, int]:
    is_dist, rank, world_size, local_rank = distributed_env()
    if is_dist:
        if not torch.cuda.is_available():
            raise RuntimeError('Distributed training launch detected but CUDA is not available.')
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError('Config requires CUDA but CUDA is not available.')
        torch.cuda.set_device(local_rank)
        backend = 'nccl' if dist.is_nccl_available() else 'gloo'
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method='env://')
    return is_dist, rank, world_size, local_rank


def cleanup_distributed(is_dist: bool) -> None:
    if is_dist and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def load_state_if_present(module: torch.nn.Module, state: Dict[str, torch.Tensor], *, strict: bool, module_name: str, is_main: bool) -> None:
    if strict:
        module.load_state_dict(state, strict=True)
        return
    current = module.state_dict()
    filtered = {
        key: value
        for key, value in state.items()
        if key in current and tuple(current[key].shape) == tuple(value.shape)
    }
    skipped = sorted(set(state.keys()) - set(filtered.keys()))
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to JSON config')
    args = parser.parse_args()

    cfg = load_json_config(args.config)
    require_cuda = bool(cfg.get('require_cuda', False))
    is_dist, rank, world_size, local_rank = maybe_init_distributed(require_cuda=require_cuda)
    is_main = rank == 0

    try:
        set_seed(int(cfg.get('seed', 42)) + rank)

        cfg_device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        if is_dist:
            device = torch.device('cuda', local_rank)
        else:
            device = torch.device(cfg_device)
        if require_cuda and device.type != 'cuda':
            raise RuntimeError(f"CUDA required by config but resolved device is '{device}'.")

        output_dir = Path(cfg.get('output_dir', 'outputs/research'))
        if is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_json(str(output_dir / 'resolved_config.json'), cfg)

        train_loader, val_loader, train_sampler = build_loaders(cfg, is_dist=is_dist, rank=rank, world_size=world_size)
        channel_mix = build_channel_mixture(cfg.get('channel', {}))
        channel_input_size = get_channel_input_size(cfg)

        model_cfg = cfg.get('model', {})
        edm_variant = str(model_cfg.get('edm_variant', 'pilotless')).strip().lower()
        use_explicit_csi = edm_variant == 'oracle_csi'
        oracle_use_token = bool(model_cfg.get('oracle_use_token', False))

        swin = build_swin(cfg, device)
        receiver = build_receiver(cfg, device)
        edm_model, diffusion_loss = build_edm(cfg, device)

        stage = cfg.get('stage', 'joint')
        base_diffusion_active = stage in {'diffusion', 'edm_pretrain', 'base_diffusion'}
        transport_active = stage in {'transport'}
        diffusion_active = stage in {'diffusion', 'edm_pretrain', 'base_diffusion', 'joint', 'tokenizer'}
        receiver_active = stage in {'transport', 'joint', 'tokenizer'}

        freeze_edm_backbone = bool(model_cfg.get('freeze_edm_backbone', stage in {'diffusion'}))
        if freeze_edm_backbone:
            for name, param in edm_model.named_parameters():
                param.requires_grad = False
            for name, param in edm_model.named_parameters():
                if 'map_label' in name or 'map_augment' in name:
                    param.requires_grad = True

        optim_cfg = cfg.get('optim', {})
        lr_transport = float(optim_cfg.get('lr_transport', 2e-4))
        lr_diffusion = float(optim_cfg.get('lr_diffusion', 2e-4))
        lr_joint = float(optim_cfg.get('lr_joint', 1e-4))
        lr_edm_multiplier = float(optim_cfg.get('lr_edm_multiplier', 0.1))
        weight_decay = float(optim_cfg.get('weight_decay', 0.01))
        grad_clip_norm = float(optim_cfg.get('grad_clip_norm', 1.0))

        receiver_params = [p for p in receiver.parameters() if p.requires_grad]
        edm_params = [p for p in edm_model.parameters() if p.requires_grad]
        diff_groups = [{'params': receiver_params, 'lr': lr_diffusion}]
        joint_groups = [{'params': receiver_params, 'lr': lr_joint}]
        if edm_params:
            diff_groups.append({'params': edm_params, 'lr': lr_diffusion * lr_edm_multiplier})
            joint_groups.append({'params': edm_params, 'lr': lr_joint * lr_edm_multiplier})
        opt_transport = torch.optim.AdamW(receiver_params, lr=lr_transport, weight_decay=weight_decay)
        opt_diff = torch.optim.AdamW(diff_groups, weight_decay=weight_decay)
        opt_joint = torch.optim.AdamW(joint_groups, weight_decay=weight_decay)

        epochs = int(cfg.get('epochs', 100))
        save_every = int(cfg.get('save_every', 10))
        val_every = int(cfg.get('val_every', 5))
        max_val_batches_cfg = cfg.get('max_val_batches', None)
        max_val_batches = None if max_val_batches_cfg in (None, 0, '') else int(max_val_batches_cfg)
        sampler_steps = int(cfg.get('sampler_steps', 18))

        lam_vq = float(cfg.get('loss', {}).get('lambda_vq', 0.5))
        lam_router = float(cfg.get('loss', {}).get('lambda_router', 0.01))
        lam_diff = float(cfg.get('loss', {}).get('lambda_diffusion', 1.0))
        lam_h = float(cfg.get('loss', {}).get('lambda_h_privileged', 0.0))
        lam_cls = float(cfg.get('loss', {}).get('lambda_channel_cls', 1.0))
        lam_router_cls = float(cfg.get('loss', {}).get('lambda_router_channel_cls', 0.0))
        lam_aux = float(cfg.get('loss', {}).get('lambda_aux', 0.1))
        lam_img = float(cfg.get('loss', {}).get('lambda_image_recon', 0.0))
        use_privileged_h = bool(cfg.get('privileged_training', {}).get('enable_h_supervision', False))

        start_epoch = 0
        resume_history: list = []
        resume_best_psnr = float('-inf')
        if cfg.get('resume', ''):
            state = torch.load(cfg['resume'], map_location=device, weights_only=False)
            strict_resume_loading = bool(cfg.get('strict_resume_loading', True))
            if 'receiver' in state:
                load_state_if_present(receiver, state['receiver'], strict=strict_resume_loading, module_name='receiver', is_main=is_main)
            if 'edm_model' in state:
                load_state_if_present(edm_model, state['edm_model'], strict=strict_resume_loading, module_name='edm_model', is_main=is_main)
            if 'opt_transport' in state:
                try:
                    opt_transport.load_state_dict(state['opt_transport'])
                except (ValueError, RuntimeError) as exc:
                    if is_main:
                        print(f'Warning: could not restore opt_transport state ({exc}); optimizer starts fresh')
            if 'opt_diff' in state:
                try:
                    opt_diff.load_state_dict(state['opt_diff'])
                except (ValueError, RuntimeError) as exc:
                    if is_main:
                        print(f'Warning: could not restore opt_diff state ({exc}); optimizer starts fresh')
            if 'opt_joint' in state:
                try:
                    opt_joint.load_state_dict(state['opt_joint'])
                except (ValueError, RuntimeError) as exc:
                    if is_main:
                        print(f'Warning: could not restore opt_joint state ({exc}); optimizer starts fresh')
            if state.get('stage', '') == stage:
                start_epoch = int(state.get('epoch', 0))
                resume_history = list(state.get('history', []))
                if resume_history:
                    resume_best_psnr = max((h.get('val_psnr_mean', float('-inf')) for h in resume_history), default=float('-inf'))
                if is_main:
                    print(f'Resuming {stage} from epoch {start_epoch}/{epochs}')

        if is_dist:
            if receiver_active:
                receiver = DDP(receiver, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            if diffusion_active:
                edm_model = DDP(edm_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        if is_main:
            print(
                json.dumps(
                    {
                        'distributed': is_dist,
                        'rank': rank,
                        'world_size': world_size,
                        'local_rank': local_rank,
                        'device': str(device),
                        'stage': stage,
                        'edm_variant': edm_variant,
                        'channel_input_size': channel_input_size,
                        'pipeline_mode': 'noisy_image_tokenizer',
                    }
                )
            )

        history = list(resume_history)
        best_psnr = resume_best_psnr
        for epoch in range(start_epoch, epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            receiver.train() if receiver_active else receiver.eval()
            if diffusion_active and not freeze_edm_backbone:
                edm_model.train()
            else:
                edm_model.eval()

            running = {
                'loss': 0.0,
                'receiver_aux': 0.0,
                'transport': 0.0,
                'image_recon': 0.0,
                'diffusion': 0.0,
                'privileged_h': 0.0,
                'channel_cls': 0.0,
                'router_channel_cls': 0.0,
                'router_entropy': 0.0,
                'vq_perplexity': 0.0,
                'batches': 0.0,
            }
            iterator = tqdm(train_loader, desc=f'epoch {epoch+1}/{epochs}', leave=False) if is_main else train_loader

            for imgs in iterator:
                imgs = imgs.to(device, non_blocking=True)
                with torch.no_grad():
                    clean_latent, noisy_latent, ch_meta = channelize_images(
                        imgs,
                        swin,
                        channel_mix,
                        channel_input_size=channel_input_size,
                        channel_spec=None,
                    )

                with contextlib.nullcontext():
                    pout = receiver(noisy_latent) if receiver_active else None

                transport_loss = torch.tensor(0.0, device=device)
                image_recon_loss = torch.tensor(0.0, device=device)
                receiver_aux = torch.tensor(0.0, device=device)
                diff_loss = torch.tensor(0.0, device=device)
                privileged_h_loss = torch.tensor(0.0, device=device)
                channel_cls_loss = torch.tensor(0.0, device=device)
                router_channel_cls_loss = torch.tensor(0.0, device=device)

                if stage in {'transport', 'joint'}:
                    if pout is None:
                        raise RuntimeError(f"Stage '{stage}' requires the receiver but receiver_active is false.")
                    transport_loss = F.mse_loss(pout.restored, clean_latent)
                    if lam_img > 0:
                        recon_img = swin.decoder(pout.restored)
                        image_recon_loss = F.mse_loss((recon_img + 1.0) * 0.5, (imgs + 1.0) * 0.5)

                if transport_active:
                    receiver_aux = lam_vq * pout.vq_loss + lam_router * pout.router_loss
                    if use_privileged_h and lam_h > 0 and 'h' in ch_meta:
                        h_target = ch_meta['h'].detach().abs().reshape(ch_meta['h'].shape[0], -1).mean(dim=1, keepdim=True)
                        privileged_h_loss = F.mse_loss(torch.log1p(pout.h_pred), torch.log1p(h_target))
                        receiver_aux = receiver_aux + lam_h * privileged_h_loss
                    if lam_cls > 0 and 'channel_id' in ch_meta:
                        ch_id = ch_meta['channel_id'].detach().long().flatten()
                        channel_cls_loss = F.cross_entropy(pout.channel_logits, ch_id)
                        receiver_aux = receiver_aux + lam_cls * channel_cls_loss
                        if lam_router_cls > 0:
                            router_target = ch_id.remainder(pout.router_logits.shape[-1])
                            router_channel_cls_loss = F.cross_entropy(pout.router_logits, router_target)
                            receiver_aux = receiver_aux + lam_router_cls * router_channel_cls_loss

                if diffusion_active:
                    if base_diffusion_active:
                        labels = None
                        cond_latent = noisy_latent
                    else:
                        if pout is None:
                            raise RuntimeError(f"Stage '{stage}' requires the receiver but receiver_active is false.")
                        label_mode = 'token'
                        if use_explicit_csi and not oracle_use_token:
                            label_mode = 'oracle'
                        labels = conditioning_labels(pout.token, ch_meta, label_mode=label_mode)
                        cond_latent = pout.restored
                        receiver_aux = receiver_aux + lam_vq * pout.vq_loss + lam_router * pout.router_loss
                        if use_privileged_h and lam_h > 0 and 'h' in ch_meta:
                            h_target = ch_meta['h'].detach().abs().reshape(ch_meta['h'].shape[0], -1).mean(dim=1, keepdim=True)
                            privileged_h_loss = F.mse_loss(torch.log1p(pout.h_pred), torch.log1p(h_target))
                            receiver_aux = receiver_aux + lam_h * privileged_h_loss
                        if lam_cls > 0 and 'channel_id' in ch_meta:
                            ch_id = ch_meta['channel_id'].detach().long().flatten()
                            channel_cls_loss = F.cross_entropy(pout.channel_logits, ch_id)
                            receiver_aux = receiver_aux + lam_cls * channel_cls_loss
                            if lam_router_cls > 0:
                                router_target = ch_id.remainder(pout.router_logits.shape[-1])
                                router_channel_cls_loss = F.cross_entropy(pout.router_logits, router_target)
                                receiver_aux = receiver_aux + lam_router_cls * router_channel_cls_loss
                    diff_loss = diffusion_loss(
                        net=edm_model,
                        images=clean_latent,
                        x_cond=cond_latent,
                        labels=labels,
                        h=ch_meta.get('h', None) if use_explicit_csi else None,
                    ).mean()

                if stage == 'transport':
                    loss = transport_loss + lam_img * image_recon_loss + receiver_aux
                    opt_transport.zero_grad()
                    loss.backward()
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(receiver_params, grad_clip_norm)
                    opt_transport.step()
                elif stage in {'diffusion', 'edm_pretrain', 'base_diffusion'}:
                    loss = diff_loss
                    opt_diff.zero_grad()
                    loss.backward()
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(edm_params, grad_clip_norm)
                    opt_diff.step()
                elif stage == 'tokenizer':
                    loss = diff_loss + lam_aux * receiver_aux
                    opt_diff.zero_grad()
                    loss.backward()
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(receiver_params + edm_params, grad_clip_norm)
                    opt_diff.step()
                elif stage == 'joint':
                    loss = transport_loss + lam_img * image_recon_loss + lam_aux * receiver_aux + lam_diff * diff_loss
                    opt_joint.zero_grad()
                    loss.backward()
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(receiver_params + edm_params, grad_clip_norm)
                    opt_joint.step()
                else:
                    raise ValueError(f'Unknown stage: {stage}')

                running['loss'] += float(loss.item())
                running['receiver_aux'] += float(receiver_aux.item())
                running['transport'] += float(transport_loss.item())
                running['image_recon'] += float(image_recon_loss.item())
                running['diffusion'] += float(diff_loss.item())
                running['privileged_h'] += float(privileged_h_loss.item())
                running['channel_cls'] += float(channel_cls_loss.item())
                running['router_channel_cls'] += float(router_channel_cls_loss.item())
                if pout is not None:
                    running['router_entropy'] += float(pout.router_entropy.item())
                if receiver_active and pout is not None:
                    with torch.no_grad():
                        perp = unwrap_model(receiver).vq.perplexity(pout.vq_indices.detach())
                        running['vq_perplexity'] += float(perp.item())
                running['batches'] += 1.0

            stats_tensor = torch.tensor(
                [
                    running['loss'],
                    running['receiver_aux'],
                    running['transport'],
                    running['image_recon'],
                    running['diffusion'],
                    running['privileged_h'],
                    running['channel_cls'],
                    running['router_channel_cls'],
                    running['router_entropy'],
                    running['vq_perplexity'],
                    running['batches'],
                ],
                device=device,
                dtype=torch.float64,
            )
            if is_dist:
                dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
            denom = max(1.0, float(stats_tensor[10].item()))
            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': float(stats_tensor[0].item() / denom),
                'train_receiver_aux': float(stats_tensor[1].item() / denom),
                'train_transport': float(stats_tensor[2].item() / denom),
                'train_image_recon': float(stats_tensor[3].item() / denom),
                'train_diffusion': float(stats_tensor[4].item() / denom),
                'train_privileged_h': float(stats_tensor[5].item() / denom),
                'train_channel_cls': float(stats_tensor[6].item() / denom),
                'train_router_channel_cls': float(stats_tensor[7].item() / denom),
                'train_router_entropy': float(stats_tensor[8].item() / denom),
                'train_vq_perplexity': float(stats_tensor[9].item() / denom),
            }

            should_eval = (epoch + 1) % val_every == 0 or epoch == start_epoch
            if is_main and should_eval:
                val_stats = evaluate_mode(
                    swin=swin,
                    receiver=unwrap_model(receiver) if receiver_active else None,
                    edm_model=unwrap_model(edm_model) if diffusion_active else None,
                    loader=val_loader,
                    channel_mix=channel_mix,
                    channel_spec=None,
                    device=device,
                    mode='baseline' if base_diffusion_active else ('transport' if stage == 'transport' else 'diffusion_token'),
                    sampler_steps=sampler_steps,
                    use_explicit_csi=use_explicit_csi,
                    oracle_use_token=oracle_use_token,
                    channel_input_size=channel_input_size,
                    max_batches=max_val_batches,
                )
                epoch_log.update({f'val_{k}': v for k, v in val_stats.items()})

            if is_main:
                print(epoch_log)
                history.append(epoch_log)
                ckpt = {
                    'epoch': epoch + 1,
                    'stage': stage,
                    'receiver': unwrap_model(receiver).state_dict(),
                    'edm_model': unwrap_model(edm_model).state_dict(),
                    'opt_transport': opt_transport.state_dict(),
                    'opt_diff': opt_diff.state_dict(),
                    'opt_joint': opt_joint.state_dict(),
                    'history': history,
                    'config': cfg,
                    'distributed_world_size': world_size,
                    'pipeline_mode': 'noisy_image_tokenizer',
                    'channel_input_size': channel_input_size,
                }
                if 'val_psnr_mean' in epoch_log and float(epoch_log['val_psnr_mean']) > best_psnr:
                    best_psnr = float(epoch_log['val_psnr_mean'])
                    torch.save(ckpt, output_dir / 'ckpt_best.pt')

                if (epoch + 1) % save_every == 0 or epoch + 1 == epochs:
                    torch.save(ckpt, output_dir / f'ckpt_{stage}_{epoch+1}.pt')
                    save_json(str(output_dir / 'history.json'), {'history': history})

                torch.save(ckpt, output_dir / 'ckpt_last.pt')

            if is_dist:
                dist.barrier()
    finally:
        cleanup_distributed(is_dist)


if __name__ == '__main__':
    main()
