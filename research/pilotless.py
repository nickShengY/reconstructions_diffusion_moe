from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, NamedTuple, Tuple


def power_normalize(x: torch.Tensor, target_power: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    dims = tuple(range(1, x.ndim))
    pwr = x.pow(2).mean(dim=dims, keepdim=True).clamp_min(eps)
    return x * (target_power / pwr).sqrt()


class JSCCTransmitter(nn.Module):
    """Simple learned mapper with UEP-like per-channel gain."""

    def __init__(self, channels: int = 24):
        super().__init__()
        self.mapper = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.uep = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mapper(x) * self.uep
        y = torch.tanh(y)
        return power_normalize(y)


class ChannelTokenizer(nn.Module):
    """Pilotless channel signature extractor from received latent."""

    def __init__(self, in_channels: int = 24, hidden: int = 128, code_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, code_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = x.view(b, c, h * w)
        z = self.encoder(seq).transpose(1, 2).contiguous()  # [B, T, D]
        return z


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int = 512,
        code_dim: int = 64,
        beta: float = 0.35,
        temperature: float = 0.35,
        diversity_weight: float = 0.20,
        confidence_weight: float = 0.01,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook = nn.Embedding(codebook_size, code_dim)
        # Spread initialisation wide enough that entries don't start degenerate.
        self.codebook.weight.data.uniform_(-1.0 / code_dim ** 0.5, 1.0 / code_dim ** 0.5)
        self.beta = beta
        self.temperature = temperature
        self.diversity_weight = diversity_weight
        self.confidence_weight = confidence_weight

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z_e: [B, T, D]
        b, t, d = z_e.shape
        flat = z_e.reshape(-1, d)
        code = self.codebook.weight
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            + code.pow(2).sum(dim=1)
            - 2.0 * flat @ code.t()
        )
        idx = torch.argmin(dist, dim=1)
        probs = torch.softmax(-dist / max(self.temperature, 1e-4), dim=1)
        hard = F.one_hot(idx, num_classes=self.codebook_size).to(dtype=probs.dtype)
        st_probs = hard + probs - probs.detach()
        z_q = (st_probs @ code).view(b, t, d)
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        avg_probs = probs.mean(dim=0).clamp_min(1e-8)
        diversity_loss = torch.sum(avg_probs * (avg_probs.log() + math.log(self.codebook_size)))
        sample_entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1).mean()
        sample_entropy = sample_entropy / math.log(self.codebook_size)
        vq_loss = (
            codebook_loss
            + self.beta * commitment_loss
            + self.diversity_weight * diversity_loss
            + self.confidence_weight * sample_entropy
        )
        return z_q, idx.view(b, t), vq_loss

    def perplexity(self, indices: torch.Tensor) -> torch.Tensor:
        """Effective number of codebook entries used (exp of entropy)."""
        counts = torch.bincount(indices.flatten(), minlength=self.codebook_size).float()
        probs = counts / counts.sum().clamp_min(1e-8)
        entropy = -(probs * (probs + 1e-8).log()).sum()
        return entropy.exp()


class Router(nn.Module):
    def __init__(self, token_dim: int = 64, num_experts: int = 4, sharpness_weight: float = 0.10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_experts),
        )
        self.num_experts = num_experts
        self.sharpness_weight = sharpness_weight

    def forward(self, token_global: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.mlp(token_global)
        return logits, torch.softmax(logits, dim=-1)

    def balance_loss(self, probs: torch.Tensor) -> torch.Tensor:
        """Encourage globally balanced but per-sample sharp expert usage."""
        p_mean = probs.mean(dim=0).clamp_min(1e-8)
        target = torch.full_like(p_mean, 1.0 / self.num_experts)
        balance = torch.sum(p_mean * (p_mean.log() - target.log()))
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1).mean()
        entropy = entropy / math.log(self.num_experts)
        return balance + self.sharpness_weight * entropy


class ExpertBlock(nn.Module):
    def __init__(self, channels: int = 24, rank: int = 8):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        # LoRA-style lightweight residual adapter.
        self.lora_down = nn.Conv2d(channels, rank, kernel_size=1, bias=False)
        self.lora_up = nn.Conv2d(rank, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.base[-1].weight)
        nn.init.zeros_(self.base[-1].bias)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.base(x) + self.lora_up(self.lora_down(x))


class PilotlessOutputs(NamedTuple):
    restored: torch.Tensor
    token: torch.Tensor
    h_pred: torch.Tensor
    vq_loss: torch.Tensor
    router_loss: torch.Tensor
    router_probs: torch.Tensor
    router_logits: torch.Tensor
    router_entropy: torch.Tensor
    # Channel-type classification logits (for auxiliary supervised loss).
    # Shape [B, num_channels] during training; zeros at inference if channel_id unavailable.
    channel_logits: torch.Tensor
    vq_indices: torch.Tensor   # [B, T] raw VQ assignments, used for perplexity logging


class PilotlessReceiver(nn.Module):
    def __init__(
        self,
        channels: int = 24,
        token_dim: int = 64,
        codebook_size: int = 512,   # increased from 256 — more capacity for 10 channel types
        num_experts: int = 4,
        expert_rank: int = 8,
        num_channel_types: int = 10,
    ):
        super().__init__()
        self.tokenizer = ChannelTokenizer(in_channels=channels, code_dim=token_dim)
        self.vq = VectorQuantizer(codebook_size=codebook_size, code_dim=token_dim, beta=0.5)
        self.router = Router(token_dim=token_dim, num_experts=num_experts)
        self.experts = nn.ModuleList([ExpertBlock(channels=channels, rank=expert_rank) for _ in range(num_experts)])
        self.token_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
        )
        self.token_affine = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 2 * channels),
        )
        # Privileged-training head: predict scalar channel magnitude proxy from token.
        self.h_regressor = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 1),
            nn.Softplus(),
        )
        # Auxiliary channel-type classification head.
        # Forces the VQ token to encode channel-discriminative information,
        # preventing codebook collapse and making the MoE routing channel-sensitive.
        self.channel_classifier = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_channel_types),
        )

    def forward(self, y: torch.Tensor) -> PilotlessOutputs:
        z_e = self.tokenizer(y)
        z_q, vq_idx, vq_loss = self.vq(z_e)
        token_global = z_q.mean(dim=1)           # [B, 64] — mean-pooled channel signature
        router_logits, probs = self.router(token_global)
        router_loss = self.router.balance_loss(probs)
        router_entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=1).mean()
        router_entropy = router_entropy / math.log(self.router.num_experts)

        expert_outs = [e(y) for e in self.experts]
        stacked = torch.stack(expert_outs, dim=1)  # [B, E, C, H, W]
        mixed = torch.sum(stacked * probs[:, :, None, None, None], dim=1)

        token = self.token_proj(token_global)
        scale, shift = self.token_affine(token).chunk(2, dim=1)
        scale = 0.10 * torch.tanh(scale).view(y.shape[0], y.shape[1], 1, 1)
        shift = 0.10 * torch.tanh(shift).view(y.shape[0], y.shape[1], 1, 1)
        mixed = mixed * (1.0 + scale) + shift
        h_pred = self.h_regressor(token)
        channel_logits = self.channel_classifier(token_global)  # [B, num_channel_types]

        return PilotlessOutputs(
            restored=mixed,
            token=token,
            h_pred=h_pred,
            vq_loss=vq_loss,
            router_loss=router_loss,
            router_probs=probs,
            router_logits=router_logits,
            router_entropy=router_entropy,
            channel_logits=channel_logits,
            vq_indices=vq_idx,
        )
