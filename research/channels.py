from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ChannelSpec:
    name: str
    probability: float = 1.0
    snr_db_min: float = 0.0
    snr_db_max: float = 20.0
    k_factor: float = 4.0  # used by Rician-like channels
    phase_noise_std: float = 0.0
    iq_gain_imbalance: float = 0.0
    impulsive_prob: float = 0.0
    impulsive_scale: float = 5.0
    freq_selective: bool = False
    max_taps: int = 7


def _sample_snr(spec: ChannelSpec, batch_size: int, device: torch.device) -> torch.Tensor:
    snr = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(spec.snr_db_min, spec.snr_db_max)
    return snr

ORACLE_LABEL_DIM = 64


def _resize_profile(profile: torch.Tensor, out_dim: int) -> torch.Tensor:
    profile = profile.to(torch.float32)
    if profile.ndim != 2:
        raise ValueError(f"Expected [B, T] profile, got shape {tuple(profile.shape)}")
    if profile.shape[1] == out_dim:
        return profile
    return F.interpolate(profile.unsqueeze(1), size=out_dim, mode="linear", align_corners=False).squeeze(1)


def _normalize_profile(profile: torch.Tensor) -> torch.Tensor:
    return profile / profile.mean(dim=1, keepdim=True).clamp_min(1e-6)


def _build_oracle_label(
    snr_db: torch.Tensor,
    h_scalar: torch.Tensor,
    time_profile: torch.Tensor | None = None,
    delay_profile: torch.Tensor | None = None,
    out_dim: int = ORACLE_LABEL_DIM,
) -> torch.Tensor:
    if out_dim < 4:
        raise ValueError("oracle label dimension must be at least 4")
    b = snr_db.shape[0]
    feature_dim = out_dim - 1
    if time_profile is None and delay_profile is None:
        base = h_scalar.reshape(b, 1).expand(b, feature_dim)
    else:
        time_bins = feature_dim // 2
        delay_bins = feature_dim - time_bins
        parts = []
        if time_profile is not None:
            parts.append(_resize_profile(_normalize_profile(time_profile), time_bins))
        else:
            parts.append(h_scalar.reshape(b, 1).expand(b, time_bins))
        if delay_profile is not None:
            parts.append(_resize_profile(_normalize_profile(delay_profile), delay_bins))
        else:
            parts.append(h_scalar.reshape(b, 1).expand(b, delay_bins))
        base = torch.cat(parts, dim=1)
    snr_norm = ((snr_db.reshape(b, 1) + 5.0) / 25.0).clamp(0.0, 1.0)
    return torch.cat([base, snr_norm], dim=1)


def _awgn(x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
    power = x.pow(2).mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
    snr_lin = torch.pow(10.0, snr_db / 10.0)
    noise_power = power / snr_lin
    noise = torch.randn_like(x) * torch.sqrt(noise_power)
    return x + noise


def _rayleigh_like(x: torch.Tensor, snr_db: torch.Tensor, rician_k: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    b = x.shape[0]
    device = x.device
    ray = torch.sqrt(torch.randn(b, 1, 1, 1, device=device).pow(2) + torch.randn(b, 1, 1, 1, device=device).pow(2)) / math.sqrt(2)
    if rician_k is not None and rician_k > 0:
        k = float(rician_k)
        los = math.sqrt(k / (k + 1.0))
        nlos = math.sqrt(1.0 / (k + 1.0))
        h = los + nlos * ray
    else:
        h = ray
    y = _awgn(h * x, snr_db)
    return y, h


def _freq_selective_distortion(x: torch.Tensor, max_taps: int = 7) -> torch.Tensor:
    _, c, _, w = x.shape
    max_taps = max(3, int(max_taps))
    if max_taps % 2 == 0:
        max_taps -= 1
    tap_choices = list(range(3, max_taps + 1, 2))
    taps = random.choice(tap_choices)
    kernel = torch.randn(c, 1, 1, taps, device=x.device)
    kernel = kernel / kernel.abs().sum(dim=-1, keepdim=True).clamp_min(1e-6)
    pad = taps // 2
    y = F.conv2d(x, kernel, padding=(0, pad), groups=c)
    if y.shape[-1] > w:
        y = y[..., :w]
    elif y.shape[-1] < w:
        y = F.pad(y, (0, w - y.shape[-1]))
    return y


def _phase_iq_impulsive(x: torch.Tensor, spec: ChannelSpec) -> torch.Tensor:
    y = x
    if spec.phase_noise_std > 0:
        theta = torch.randn(y.shape[0], 1, 1, 1, device=y.device) * spec.phase_noise_std
        y = y * torch.cos(theta)
    if spec.iq_gain_imbalance > 0:
        scale = 1.0 + torch.empty(y.shape[0], 1, 1, 1, device=y.device).uniform_(-spec.iq_gain_imbalance, spec.iq_gain_imbalance)
        y = y * scale
    if spec.impulsive_prob > 0:
        mask = (torch.rand_like(y) < spec.impulsive_prob).to(y.dtype)
        impulses = torch.randn_like(y) * spec.impulsive_scale
        y = y + mask * impulses
    return y


class TorchChannelBackend:
    """PyTorch-native channel simulation for mixed impairments."""

    def apply(self, x: torch.Tensor, spec: ChannelSpec) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b = x.shape[0]
        device = x.device
        snr_db = _sample_snr(spec, b, device=device)
        meta: Dict[str, torch.Tensor] = {"snr_db": snr_db}

        if spec.name == "awgn":
            y = _awgn(x, snr_db)
            h = torch.ones(b, 1, 1, 1, device=device)
        elif spec.name == "rayleigh":
            y, h = _rayleigh_like(x, snr_db, rician_k=None)
        elif spec.name == "rician":
            y, h = _rayleigh_like(x, snr_db, rician_k=spec.k_factor)
        elif spec.name in {"tdl", "cdl", "umi", "uma", "rma", "highway", "rural", "urban"}:
            # Frequency-selective surrogate used in PyTorch fallback.
            y, h = _rayleigh_like(x, snr_db, rician_k=spec.k_factor if spec.name in {"urban", "uma"} else None)
            y = _freq_selective_distortion(y, max_taps=spec.max_taps)
        else:
            raise ValueError(f"Unknown channel name: {spec.name}")

        if spec.freq_selective:
            y = _freq_selective_distortion(y, max_taps=spec.max_taps)

        y = _phase_iq_impulsive(y, spec)
        meta["h"] = h
        meta["oracle_label"] = _build_oracle_label(snr_db=snr_db, h_scalar=h)
        meta["backend_used"] = "torch"
        return y, meta


class SionnaChannelBackend:
    """
    Sionna-backed channel simulation using 3GPP TR 38.901 channel models.

    Channel mapping:
      awgn    → AWGN (TF ops, power-normalised)
      rayleigh → RayleighBlockFading (flat, single-tap Rayleigh)
      rician   → TDL-D  (LOS, K≈13 dB, 10 ns DS, pedestrian)
      tdl      → TDL-A  (NLOS, 100 ns DS, pedestrian)
      cdl      → CDL-A  (NLOS, 100 ns DS, pedestrian, SISO)
      urban    → TDL-B  (NLOS, 200 ns DS, slow vehicle ≤36 km/h)
      rural    → TDL-C  (NLOS, 1000 ns DS, fast vehicle ≤120 km/h)
      highway  → TDL-D  (LOS, 30 ns DS, high mobility 60–180 km/h)
      umi      → TDL-A  (UMi-like NLOS, 65 ns DS, pedestrian)
      uma      → TDL-C  (UMa-like NLOS, 363 ns DS, slow vehicle ≤36 km/h)

    All channels add AWGN after fading, scaled to the instantaneous
    post-fading signal power so the receiver SNR matches spec.snr_db.

    Falls back to TorchChannelBackend if Sionna/TF is unavailable.
    """

    # Physical constants
    _CARRIER_HZ  = 3.5e9    # 3.5 GHz (sub-6 GHz 5G NR)
    _BANDWIDTH_HZ = 15.36e6  # 10 MHz equivalent sampling rate
    _L_MAX = 20              # covers up to ~1.3 µs delay spread

    # TDL configs: profile, delay_spread_s, min_speed_mps, max_speed_mps
    _TDL_CFG: Dict[str, Tuple] = {
        "rician":  ("D",  10e-9,    0.0,  3.0),
        "tdl":     ("A", 100e-9,    0.0,  3.0),
        "urban":   ("B", 200e-9,    0.0, 10.0),
        "rural":   ("C", 1000e-9,   0.0, 33.3),
        "highway": ("D",  30e-9,   16.7, 50.0),
        "umi":     ("A",  65e-9,    0.0,  3.0),
        "uma":     ("C", 363e-9,    0.0, 10.0),
    }
    # CDL configs: profile, delay_spread_s, min_speed_mps, max_speed_mps
    _CDL_CFG: Dict[str, Tuple] = {
        "cdl":     ("A", 100e-9,    0.0,  3.0),
    }

    def __init__(self):
        self.torch_backend = TorchChannelBackend()
        self._sionna_ok = False
        self._tf = None
        self._warned_fallback = False
        # Lazy cache: (ch_name, N) → (GenerateTimeChannel, ApplyTimeChannel)
        self._pipeline_cache: Dict[Tuple[str, int], Tuple] = {}
        self._ray_ch = None   # RayleighBlockFading instance
        self._ut_array = None  # CDL antenna arrays (SISO)
        self._bs_array = None
        self._l_min: int = -6  # updated from DEFAULT_L_MIN after import
        self._init_sionna()

    def _warn_once(self, message: str) -> None:
        if not self._warned_fallback:
            print(f"[sionna-backend] {message}")
            self._warned_fallback = True

    def _init_sionna(self) -> None:
        try:
            import os as _os
            _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            _os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
            _os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
            _os.environ.setdefault("OMP_NUM_THREADS", "2")
            import tensorflow as tf  # type: ignore
            try:
                tf.config.set_visible_devices([], "GPU")
                tf.config.threading.set_intra_op_parallelism_threads(2)
                tf.config.threading.set_inter_op_parallelism_threads(2)
            except Exception:
                pass
        except Exception:
            self._warn_once("TensorFlow not available. Falling back to torch backend.")
            return

        try:
            import sionna.channel as _sch        # type: ignore
            import sionna.channel.tr38901 as _tr  # type: ignore
        except Exception as e:
            self._warn_once(f"Sionna not available ({e}). Falling back to torch backend.")
            return

        self._tf = tf
        self._l_min = int(_sch.DEFAULT_L_MIN)

        # Pre-build flat-fading model (used by rayleigh)
        self._ray_ch = _sch.RayleighBlockFading(
            num_rx=1, num_rx_ant=1, num_tx=1, num_tx_ant=1)

        # Pre-build SISO antenna arrays for CDL
        ant_kwargs = dict(num_rows=1, num_cols=1, polarization="single",
                          polarization_type="V", antenna_pattern="38.901",
                          carrier_frequency=self._CARRIER_HZ)
        self._ut_array = _tr.AntennaArray(**ant_kwargs)
        self._bs_array = _tr.AntennaArray(**ant_kwargs)

        # Store references for lazy pipeline creation
        self._sch = _sch
        self._tr  = _tr
        self._sionna_ok = True

    # ------------------------------------------------------------------
    # Static helpers (unchanged from original)
    # ------------------------------------------------------------------
    @staticmethod
    def _to_complex(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b = x.shape[0]
        flat = x.reshape(b, -1)
        if flat.shape[1] % 2 == 1:
            flat = torch.cat([flat, torch.zeros(b, 1, device=x.device, dtype=x.dtype)], dim=1)
        real = flat[:, 0::2]
        imag = flat[:, 1::2]
        return torch.complex(real, imag), x.numel() // b

    @staticmethod
    def _from_complex(z: torch.Tensor, ref_shape: Tuple[int, ...], original_numel: int) -> torch.Tensor:
        b = ref_shape[0]
        cat = torch.stack([z.real, z.imag], dim=-1).reshape(b, -1)
        return cat[:, :original_numel].reshape(ref_shape)

    @staticmethod
    def _mmse_equalize_flat(y: np.ndarray, h: np.ndarray, snr_lin: np.ndarray) -> np.ndarray:
        """Known-CSI one-tap MMSE equalizer for flat fading."""
        h = h.reshape(y.shape[0], 1).astype(np.complex64)
        snr = snr_lin.reshape(y.shape[0], 1).astype(np.float32)
        denom = (np.abs(h) ** 2 + 1.0 / np.maximum(snr, 1e-6)).astype(np.float32)
        return (np.conj(h) / denom) * y

    @staticmethod
    def _mmse_equalize_frequency_selective(
        y: np.ndarray,
        h_time: np.ndarray,
        snr_lin: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """Approximate known-CSI frequency-domain MMSE equalizer for SISO TDL/CDL."""
        # h_time from Sionna: [B, 1, 1, 1, 1, T, L]. Average over time to get a
        # stable per-sample channel impulse response for the image-sized block.
        taps = h_time[:, 0, 0, 0, 0, :n, :].mean(axis=1).astype(np.complex64)
        h_freq = np.fft.fft(taps, n=n, axis=1).astype(np.complex64)
        y_freq = np.fft.fft(y, n=n, axis=1).astype(np.complex64)
        snr = snr_lin.reshape(y.shape[0], 1).astype(np.float32)
        denom = (np.abs(h_freq) ** 2 + 1.0 / np.maximum(snr, 1e-6)).astype(np.float32)
        x_hat_freq = (np.conj(h_freq) / denom) * y_freq
        return np.fft.ifft(x_hat_freq, n=n, axis=1).astype(np.complex64)

    # ------------------------------------------------------------------
    # Lazy pipeline factory
    # ------------------------------------------------------------------
    def _get_pipeline(self, ch_name: str, N: int) -> Tuple:
        """Return cached (GenerateTimeChannel, ApplyTimeChannel) for ch_name and N."""
        key = (ch_name, N)
        if key in self._pipeline_cache:
            return self._pipeline_cache[key]

        l_min, l_max = self._l_min, self._L_MAX
        l_tot = l_max - l_min + 1

        if ch_name in self._TDL_CFG:
            profile, ds, min_spd, max_spd = self._TDL_CFG[ch_name]
            model = self._tr.TDL(
                profile,
                delay_spread=ds,
                carrier_frequency=self._CARRIER_HZ,
                min_speed=min_spd,
                max_speed=max_spd,
                num_rx_ant=1,
                num_tx_ant=1,
            )
        elif ch_name in self._CDL_CFG:
            profile, ds, min_spd, max_spd = self._CDL_CFG[ch_name]
            model = self._tr.CDL(
                profile,
                delay_spread=ds,
                carrier_frequency=self._CARRIER_HZ,
                ut_array=self._ut_array,
                bs_array=self._bs_array,
                direction="uplink",
                min_speed=min_spd,
                max_speed=max_spd,
            )
        else:
            raise ValueError(f"No Sionna pipeline defined for channel '{ch_name}'")

        gen = self._sch.GenerateTimeChannel(
            model,
            bandwidth=self._BANDWIDTH_HZ,
            num_time_samples=N,
            l_min=l_min,
            l_max=l_max,
            normalize_channel=True,   # unit expected energy → predictable SNR
        )
        apply_ch = self._sch.ApplyTimeChannel(
            num_time_samples=N, l_tot=l_tot, add_awgn=False)

        self._pipeline_cache[key] = (gen, apply_ch)
        return gen, apply_ch

    # ------------------------------------------------------------------
    # Core apply
    # ------------------------------------------------------------------
    def _add_awgn(self, tf_y: "tf.Tensor", snr_lin_np: np.ndarray) -> "tf.Tensor":
        """Add AWGN scaled so receiver SNR = snr_lin (per batch sample)."""
        tf = self._tf
        # Measure instantaneous post-fading signal power per batch element
        sig_pwr = tf.reduce_mean(tf.abs(tf_y) ** 2, axis=-1)            # [B]
        snr_lin_tf = tf.constant(snr_lin_np, dtype=tf.float32)           # [B]
        # Per-component noise std: sigma² = sig_pwr / (2 * snr_lin)
        noise_std = tf.sqrt(sig_pwr / (2.0 * snr_lin_tf))                # [B]
        N = tf.shape(tf_y)[-1]
        n_re = tf.random.normal([tf.shape(tf_y)[0], N], dtype=tf.float32) * noise_std[:, None]
        n_im = tf.random.normal([tf.shape(tf_y)[0], N], dtype=tf.float32) * noise_std[:, None]
        return tf_y + tf.complex(n_re, n_im)

    def apply(self, x: torch.Tensor, spec: ChannelSpec) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self._sionna_ok:
            out, meta = self.torch_backend.apply(x, spec)
            meta["backend_used"] = "torch_fallback"
            return out, meta

        try:
            return self._apply_sionna(x, spec)
        except Exception as e:
            self._warn_once(
                f"Sionna runtime failure on channel '{spec.name}' ({e}). "
                "Falling back to torch backend.")
            out, meta = self.torch_backend.apply(x, spec)
            meta["backend_used"] = "torch_fallback"
            return out, meta

    def _apply_sionna(self, x: torch.Tensor, spec: ChannelSpec) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        tf = self._tf
        b = x.shape[0]
        device = x.device

        snr_db     = _sample_snr(spec, b, device=device)                       # [B,1,1,1]
        snr_lin_pt = torch.pow(10.0, snr_db / 10.0).reshape(b)                 # [B]
        snr_lin_np = snr_lin_pt.cpu().numpy().astype(np.float32)

        z_pt, numel = self._to_complex(x)          # [B, N]  complex torch
        N = z_pt.shape[1]
        tf_z = tf.convert_to_tensor(z_pt.detach().cpu().numpy().astype(np.complex64))

        h_np = np.ones((b, 1), dtype=np.float32)   # default scalar channel gain
        time_profile_np = None
        delay_profile_np = None

        if spec.name == "awgn":
            # No fading — AWGN only (power-normalised to actual signal power)
            tf_y = self._add_awgn(tf_z, snr_lin_np)
            z_equalized_np = tf_y.numpy().astype(np.complex64)

        elif spec.name == "rayleigh":
            # Flat Rayleigh block fading: one complex coefficient per batch sample
            h_tf, _ = self._ray_ch(batch_size=b, num_time_steps=1)  # [B,1,1,1,1,1,1]
            h_coeff  = tf.reshape(h_tf, [b, 1])                      # [B, 1]
            z_faded  = h_coeff * tf_z                                 # [B, N]
            tf_y     = self._add_awgn(z_faded, snr_lin_np)
            h_coeff_np = h_coeff.numpy().astype(np.complex64)
            z_equalized_np = self._mmse_equalize_flat(tf_y.numpy().astype(np.complex64), h_coeff_np, snr_lin_np)
            h_np     = np.abs(h_coeff_np).astype(np.float32)          # [B, 1]

        else:
            # Time-domain 3GPP channels: TDL-A/B/C/D or CDL-A
            gen, apply_ch = self._get_pipeline(spec.name, N)
            h_time   = gen(batch_size=b)                              # [B,1,1,1,1, N+extra, l_tot]
            x_4d     = tf.reshape(tf_z, [b, 1, 1, N])                # [B,1,1,N]
            y_4d     = apply_ch([x_4d, h_time])                      # [B,1,1, N+l_max-l_min]
            z_faded  = y_4d[:, 0, 0, :N]                             # [B, N]  trim
            tf_y     = self._add_awgn(z_faded, snr_lin_np)
            h_time_np = h_time.numpy().astype(np.complex64)
            z_equalized_np = self._mmse_equalize_frequency_selective(
                tf_y.numpy().astype(np.complex64),
                h_time_np,
                snr_lin_np,
                N,
            )
            h_abs = np.abs(h_time_np[:, 0, 0, 0, 0, :, :])
            h_mag = h_abs.mean(axis=(-1, -2))
            h_np  = h_mag.reshape(b, 1).astype(np.float32)
            time_profile_np = h_abs[:, :N, :].mean(axis=-1).astype(np.float32)
            delay_profile_np = h_abs[:, :N, :].mean(axis=1).astype(np.float32)

        # Convert back to PyTorch real tensor
        y_pt = torch.from_numpy(z_equalized_np).to(device=device)
        out  = self._from_complex(y_pt, tuple(x.shape), numel).to(dtype=x.dtype)

        # Hardware impairments (phase noise, IQ imbalance, impulsive noise) are
        # additional distortions on top of the propagation channel.
        out = _phase_iq_impulsive(out, spec)

        h = torch.from_numpy(h_np).to(device=device).view(b, 1, 1, 1)
        time_profile = None if time_profile_np is None else torch.from_numpy(time_profile_np).to(device=device)
        delay_profile = None if delay_profile_np is None else torch.from_numpy(delay_profile_np).to(device=device)
        meta: Dict[str, torch.Tensor] = {
            "snr_db": snr_db,
            "h": h,
            "oracle_label": _build_oracle_label(snr_db=snr_db, h_scalar=h, time_profile=time_profile, delay_profile=delay_profile),
            "backend_used": "sionna",
        }
        return out, meta


@dataclass
class ChannelMixture:
    specs: List[ChannelSpec] = field(default_factory=list)
    backend: str = "torch"  # "torch" or "sionna"

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("ChannelMixture requires at least one ChannelSpec")
        prob = sum(s.probability for s in self.specs)
        if prob <= 0:
            raise ValueError("Sum of channel probabilities must be positive")
        for s in self.specs:
            s.probability = s.probability / prob
        self._channel_id_map = {s.name: i for i, s in enumerate(self.specs)}
        self._backend_impl = SionnaChannelBackend() if self.backend == "sionna" else TorchChannelBackend()

    def sample_spec(self) -> ChannelSpec:
        names = [s.name for s in self.specs]
        probs = [s.probability for s in self.specs]
        idx = np.random.choice(len(names), p=probs)
        return self.specs[int(idx)]

    def apply(self, x: torch.Tensor, spec: Optional[ChannelSpec] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        chosen = spec if spec is not None else self.sample_spec()
        out, meta = self._backend_impl.apply(x, chosen)
        channel_id = self._channel_id_map[chosen.name]
        meta["channel_id"] = torch.full((x.shape[0], 1), float(channel_id), device=x.device)
        meta["channel_name"] = chosen.name
        meta["backend_requested"] = self.backend
        return out, meta


def build_default_channel_specs() -> List[ChannelSpec]:
    return [
        ChannelSpec(name="awgn", probability=0.15, snr_db_min=0, snr_db_max=20),
        ChannelSpec(name="rayleigh", probability=0.15, snr_db_min=0, snr_db_max=20),
        ChannelSpec(name="rician", probability=0.10, snr_db_min=0, snr_db_max=20, k_factor=6.0),
        ChannelSpec(name="tdl", probability=0.10, snr_db_min=0, snr_db_max=20, freq_selective=True, max_taps=9),
        ChannelSpec(name="cdl", probability=0.10, snr_db_min=0, snr_db_max=20, freq_selective=True, max_taps=11),
        ChannelSpec(name="urban", probability=0.10, snr_db_min=-2, snr_db_max=18, k_factor=6.0, phase_noise_std=0.05, iq_gain_imbalance=0.08),
        ChannelSpec(name="rural", probability=0.10, snr_db_min=2, snr_db_max=22, freq_selective=True, max_taps=7),
        ChannelSpec(name="highway", probability=0.10, snr_db_min=-4, snr_db_max=16, phase_noise_std=0.08, impulsive_prob=0.01, impulsive_scale=8.0),
        ChannelSpec(name="umi", probability=0.05, snr_db_min=-2, snr_db_max=18, freq_selective=True, max_taps=9),
        ChannelSpec(name="uma", probability=0.05, snr_db_min=-2, snr_db_max=18, freq_selective=True, max_taps=11),
    ]
