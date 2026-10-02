import torch
import torch.nn as nn
import torch.nn.functional as F

from ocsvm_guidance_torch import OCSVMGuidedAutoencoderBase


class OCSVMguidedAutoencoder(OCSVMGuidedAutoencoderBase):
    """OgAE for Experiment 1 (MNIST-C). OCSVM-guidance arguments are documented in OCSVMGuidedAutoencoderBase.

    Defaults are the best Experiment 1 hyperparameters (lambda = 1e-2, nu = 0.03, gamma = 1e-2), with the decision
    function given by the mean of the last M = 10 OC-SVMs. Use set_betas() at mid-training for the
    expander -> expander + compactor schedule.
    """

    def __init__(self, batch_size_train, batch_size_valid, latent_dim=32, n_last_ocsvms=10, **ogae_kwargs):
        super().__init__(batch_size_train, batch_size_valid, n_last_ocsvms=n_last_ocsvms, **ogae_kwargs)
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            EncoderBlock(1, 4),
            EncoderBlock(4, 8),
            nn.Flatten(),
            nn.Linear(7 * 7 * 8, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 7 * 7 * 8),
            nn.Unflatten(1, (8, 7, 7)),
            DecoderBlock(8, 8),
            DecoderBlock(8, 4),
            nn.ConvTranspose2d(4, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )


class DeepSVDDAutoEncoderHard(nn.Module):
    def __init__(self, center, latent_dim=32, balance_coeff=0.1):
        super().__init__()
        self.center = center
        self.balance_coeff = balance_coeff

        self.encoder = nn.Sequential(
            EncoderBlock(1, 4, use_bias=False),
            EncoderBlock(4, 8, use_bias=False),
            nn.Flatten(),
            nn.Linear(7 * 7 * 8, latent_dim, bias=False)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 7 * 7 * 8),
            nn.Unflatten(1, (8, 7, 7)),
            DecoderBlock(8, 8, use_bias=False),
            DecoderBlock(8, 4, use_bias=False),
            nn.ConvTranspose2d(4, 1, kernel_size=5, padding=2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss_center = torch.sum((z - self.center) ** 2, dim=1)
        return loss_center, x_hat


class DeepSVDDEncoderHard(nn.Module):
    def __init__(self, center, latent_dim=32):
        super().__init__()
        self.center = center
        self.encoder = nn.Sequential(
            EncoderBlock(1, 4, use_bias=False),
            EncoderBlock(4, 8, use_bias=False),
            nn.Flatten(),
            nn.Linear(7 * 7 * 8, latent_dim, bias=False)
        )

    def forward(self, x):
        z = self.encoder(x)
        return torch.sum((z - self.center) ** 2, dim=1)


class DeepSVDDEncoderSoft(nn.Module):
    def __init__(self, center, nu, latent_dim=32):
        super().__init__()
        self.center = center
        self.nu = nu
        self.R = nn.Parameter(torch.tensor(1e-4))
        self.train_r_only = False

        self.encoder = nn.Sequential(
            EncoderBlock(1, 4, use_bias=False),
            EncoderBlock(4, 8, use_bias=False),
            nn.Flatten(),
            nn.Linear(7 * 7 * 8, latent_dim, bias=False)
        )

    def forward(self, x):
        z = self.encoder(x)
        l2_sq = torch.sum((z - self.center) ** 2, dim=1)
        return l2_sq

    def compute_loss(self, x):
        l2_sq = self(x)
        second_term = (1 / self.nu) * F.relu(l2_sq - self.R ** 2).mean()
        return self.R ** 2 + second_term


class DeepSVDDVariationalAutoEncoderHard(nn.Module):
    def __init__(self, latent_dim=32, beta_kl=1.0, balance_coeff=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta_kl = beta_kl
        self.balance_coeff = balance_coeff
        self.register_buffer("center", torch.zeros(latent_dim))

        self.encoder_conv = nn.Sequential(
            EncoderBlock(1, 4, use_bias=False),
            EncoderBlock(4, 8, use_bias=False),
            nn.Flatten()
        )
        self.fc_mean = nn.Linear(7 * 7 * 8, latent_dim, bias=False)
        self.fc_logvar = nn.Linear(7 * 7 * 8, latent_dim, bias=False)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 7 * 7 * 8),
            nn.Unflatten(1, (8, 7, 7)),
            DecoderBlock(8, 8, use_bias=False),
            DecoderBlock(8, 4, use_bias=False),
            nn.ConvTranspose2d(4, 1, kernel_size=5, padding=2, bias=False),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x_enc = self.encoder_conv(x)
        mu = self.fc_mean(x_enc)
        logvar = self.fc_logvar(x_enc)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, z

    def compute_loss(self, x):
        x_hat, mu, logvar, z = self(x)
        if self.training:  # center c is the mean of the current batch
            self.center.copy_(z.detach().mean(dim=0))
        mse_recons = F.mse_loss(x_hat, x, reduction='mean')
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        z_center = F.mse_loss(z, self.center.expand_as(z), reduction='mean')
        return self.balance_coeff * (mse_recons + self.beta_kl * kl) + z_center


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=use_bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, padding=2, bias=use_bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.block(x)
