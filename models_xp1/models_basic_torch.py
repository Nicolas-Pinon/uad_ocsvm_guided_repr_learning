import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.block(x)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
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

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim=32, beta_kl=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta_kl = beta_kl

        self.encoder_conv = nn.Sequential(
            EncoderBlock(1, 4),
            EncoderBlock(4, 8),
            nn.Flatten()
        )
        self.fc_mean = nn.Linear(7 * 7 * 8, latent_dim)
        self.fc_log_var = nn.Linear(7 * 7 * 8, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 7 * 7 * 8),
            nn.Unflatten(1, (8, 7, 7)),
            DecoderBlock(8, 8),
            DecoderBlock(8, 4),
            nn.ConvTranspose2d(4, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def compute_kl_loss(self, mean, log_var):
        return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)

    def forward(self, x):
        x_enc = self.encoder_conv(x)
        mean = self.fc_mean(x_enc)
        log_var = self.fc_log_var(x_enc)
        z = self.reparameterize(mean, log_var)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var


class SiameseAutoencoder(nn.Module):
    def __init__(self, latent_dim=32, similarity_coeff=0.1):
        super().__init__()
        self.similarity_coeff = similarity_coeff
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

    def contrastive_loss(self, z1, z2):
        return F.mse_loss(z1, z2)

    def forward(self, x):
        x1, x2 = torch.chunk(x, chunks=2, dim=0)
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        x1_hat = self.decoder(z1)
        x2_hat = self.decoder(z2)
        return x1_hat, x2_hat, z1, z2
