import torch
import numpy as np

class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t
        print(f"Init VP Loss beta_d {beta_d}, beta_min {beta_min}, epsilon_t {epsilon_t}")

    def __call__(self, net, images, labels, augment_pipe=None, step=0):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        # if step % 10 ==0 :
        #     print(f"img min {images.min().item():.4f} max {images.max().item():.4f} mean {images.mean().item():.4f} std {images.std().item():.4f}")
        #     print(f"D_yn min {D_yn.min().item():.4f} max {D_yn.max().item():.4f} mean {D_yn.mean().item():.4f} std {D_yn.std().item():.4f}")
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()


class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

class EDMLossSRAugment:
    """EDM loss for conditional latent super-resolution."""
    def __init__(self, P_mean=-2.8, P_std=1.2, sigma_data=0.06):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, x_cond, labels=None, augment_labels=None):
        """
        Args:
            images: clean latent [B, 24, 32, 32] — ground truth
            x_cond: degraded/LR latent [B, 24, 32, 32] — condition
        """
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y = images
        # y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, x_cond=x_cond, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


class EDMLossSR:
    """EDM loss for conditional latent super-resolution."""
    def __init__(self, P_mean=-2.8, P_std=1.2, sigma_data=0.06):
        # P_mean=-2.8 → median sigma ≈ exp(-2.8) ≈ 0.06, matching sigma_data
        # This ensures training focuses on the right noise levels
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, x_cond, labels=None, h=None, augment_pipe=None):
        """
        Args:
            images: clean latent [B, 24, 32, 32] — ground truth
            x_cond: degraded/LR latent [B, 24, 32, 32] — condition
        """
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        if h is not None:
            try:
                D_yn = net(y + n, sigma, labels, x_cond=x_cond, h=h, augment_labels=augment_labels)
            except TypeError:
                D_yn = net(y + n, sigma, labels, x_cond=x_cond, augment_labels=augment_labels)
        else:
            D_yn = net(y + n, sigma, labels, x_cond=x_cond, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


class EDMLossSRChannel:
    """EDM loss for conditional latent super-resolution with optional CSI."""

    def __init__(self, P_mean=-2.8, P_std=1.2, sigma_data=0.06):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, x_cond, labels=None, h=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        if h is not None:
            D_yn = net(y + n, sigma, labels, x_cond=x_cond, h=h, augment_labels=augment_labels)
        else:
            D_yn = net(y + n, sigma, labels, x_cond=x_cond, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


class EDMLossV2:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, noisy_images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(noisy_images, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


class EDMLossV3:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5,
                  sigma_min=0.002, sigma_max=80, snr_min_db=-5, snr_max_db=5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.sigma_min =  sigma_min
        self.sigma_max = sigma_max
        self.snr_max_db = snr_max_db
        self.snr_min_db = snr_min_db
        self.loss_type = 'l2'


    def _get_sigma_schedule(self, num_steps):
        rho = 7
        step_indices = torch.arange(num_steps)
        t_steps = (self.sigma_max ** (1/rho) + step_indices / (num_steps - 1) *
                   (self.sigma_min ** (1/rho) - self.sigma_max ** (1/rho))) ** rho
        return t_steps.flip(0)  # Flip so t=0 is clean, t=T-1 is most degraded

    def sigma_to_snr_db(self, sigma):
        """Map sigma to SNR: larger sigma → lower SNR"""
        # Linear interpolation in log-sigma space
        log_sigma = torch.log(sigma)
        log_sigma_min = torch.log(torch.tensor(self.sigma_min))
        log_sigma_max = torch.log(torch.tensor(self.sigma_max))

        # Normalize to [0, 1]
        t_normalized = (log_sigma - log_sigma_min) / (log_sigma_max - log_sigma_min)
        t_normalized = t_normalized.clamp(0, 1)

        # Interpolate SNR
        snr_db = self.snr_max_db + t_normalized * (self.snr_min_db - self.snr_max_db)
        return snr_db

    def add_rayleigh_fading(self, images, sigma):
        """
        Apply Rayleigh fading with SNR based on sigma schedule.
        y = h * x + n
        """
        batch_size = images.shape[0]
        device = images.device

        # Rayleigh fading coefficient (one per sample)
        real = torch.randn(batch_size, 1, 1, 1, device=device)
        imag = torch.randn(batch_size, 1, 1, 1, device=device)
        h = torch.sqrt(real**2 + imag**2)  # Rayleigh distributed

        # Apply fading
        faded = h * images

        # Convert sigma to SNR
        snr_db = self.sigma_to_snr_db(sigma)

        # Handle per-sample sigma (batch of sigmas)
        if snr_db.dim() == 0:
            snr_db = snr_db.expand(batch_size)
        snr_db = snr_db.view(batch_size, 1, 1, 1)

        # Calculate noise for target SNR
        signal_power = faded.pow(2).mean(dim=[1, 2, 3], keepdim=True)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        sigma_n = torch.sqrt(noise_power + 1e-8)

        # Add AWGN
        noise = torch.randn_like(images) * sigma_n
        y = faded + noise

        return y, sigma_n

    def q_sample(self, x_start, t):
        """Apply Rayleigh fading based on timestep t"""
        # Get sigma for each sample in batch
        sigma = self.sigmas[t]  # Shape: (batch_size,)

        # Apply Rayleigh fading
        x_degraded, sigma_n = self.add_rayleigh_fading(x_start, sigma)
        return x_degraded

    # def p_losses(self, x_start, t):
    #     """Training loss: reconstruct clean image from degraded"""
    #     x_degraded = self.q_sample(x_start=x_start, t=t)
    #     x_recon = self.defade_fn(x_degraded, t)

    #     if self.loss_type == 'l1':
    #         loss = (x_start - x_recon).abs().mean()
    #     elif self.loss_type == 'l2':
    #         loss = torch.nn.functional.mse_loss(x_start, x_recon)
    #     else:
    #         raise NotImplementedError()
    #     return loss

    # def forward(self, x, *args, **kwargs):
    #     b, c, h, w, device, img_size = *x.shape, x.device, self.image_size
    #     assert h == img_size and w == img_size
    #     t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
    #     return self.p_losses(x, t, *args, **kwargs)

    # def __call__(self, net, images, labels=None, augment_pipe=None):
    #     rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
    #     sigma = (rnd_normal * self.P_std + self.P_mean).exp()
    #     weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
    #     y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
    #     n = torch.randn_like(y) * sigma
    #     D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
    #     loss = weight * ((D_yn - y) ** 2)
    #     return loss

class EDMLossv5:
    def __init__(self, P_mean=-2.8, P_std=1.2, sigma_data=0.06):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, x_cond=None, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, x_cond=x_cond, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss
