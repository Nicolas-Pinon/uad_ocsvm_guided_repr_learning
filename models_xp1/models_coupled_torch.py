import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp


class OCSVMguidedAutoencoder(nn.Module):
    def __init__(self, batch_size_train, batch_size_valid, latent_dim=32, ocsvm_coeff=0.1,
                 nu_ocsvm_coeff=0.03, gamma_rbf_coeff="scale", jz_mode="StopGradLoss",
                 jean_zad_linear=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.ocsvm_coeff = ocsvm_coeff
        self.nu = nu_ocsvm_coeff
        self.gamma_mode = gamma_rbf_coeff
        self.jz_mode = jz_mode
        self.linear = jean_zad_linear

        # Encoder
        self.encoder = nn.Sequential(
            EncoderBlock(1, 4),
            EncoderBlock(4, 8),
            nn.Flatten(),
            nn.Linear(7 * 7 * 8, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 7 * 7 * 8),
            nn.Unflatten(1, (8, 7, 7)),
            DecoderBlock(8, 8),
            DecoderBlock(8, 4),
            nn.ConvTranspose2d(4, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )

        # CVXPY problem for training
        n = batch_size_train // 2
        alpha = cp.Variable(n)
        k_sqrt = cp.Parameter((n, n), PSD=True)
        constraints = [cp.sum(alpha) == self.nu * n, alpha >= 0, alpha <= 1]
        objective = cp.Minimize(0.5 * cp.sum_squares(k_sqrt @ alpha))
        prob = cp.Problem(objective, constraints)
        self.ocsvm_layer_train = CvxpyLayer(prob, parameters=[k_sqrt], variables=[alpha])

        # CVXPY problem for validation
        n_val = batch_size_valid // 2
        alpha_val = cp.Variable(n_val)
        k_sqrt_val = cp.Parameter((n_val, n_val), PSD=True)
        constraints_val = [cp.sum(alpha_val) == self.nu * n_val, alpha_val >= 0, alpha_val <= 1]
        objective_val = cp.Minimize(0.5 * cp.sum_squares(k_sqrt_val @ alpha_val))
        prob_val = cp.Problem(objective_val, constraints_val)
        self.ocsvm_layer_valid = CvxpyLayer(prob_val, parameters=[k_sqrt_val], variables=[alpha_val])

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def solve_ocsvm(self, z, training=True):
        n = z.shape[0] // 2
        z_sv, _ = torch.split(z, n, dim=0)
        z_sv = standardize(z_sv)
        
        if self.gamma_mode == "scale":
            var = z_sv.var(dim=0, unbiased=False).mean()
            gamma = 1 / (self.latent_dim * var + 1e-6)
        elif self.gamma_mode == "auto":
            gamma = 1 / (self.latent_dim * 4)
        else:
            gamma = float(self.gamma_mode)

        if self.linear:
            K = torch.matmul(z_sv, z_sv.t())
        else:
            dist_sq = torch.cdist(z_sv, z_sv, p=2).pow(2)
            K = torch.exp(-gamma * dist_sq)

        eps = 1e-8 / gamma if not self.linear else 1e-8
        K_sqrt = torch.linalg.cholesky(K + eps * torch.eye(n, device=z.device))
        layer = self.ocsvm_layer_train if training else self.ocsvm_layer_valid
        alpha, = layer(K_sqrt.double())
        alpha = alpha.float() / (self.nu * n)
        return alpha, K

    def ocsvm_objective(self, alpha, z, K_sv):
        n = z.shape[0] // 2
        z_sv, z_loss = torch.split(z, n, dim=0)
        z_sv, z_loss = standardize(z_sv), standardize(z_loss)

        if self.gamma_mode == "scale":
            var = z_sv.var(dim=0, unbiased=False).mean()
            gamma = 1 / (self.latent_dim * var + 1e-6)
        elif self.gamma_mode == "auto":
            gamma = 1 / (self.latent_dim * 4)
        else:
            gamma = float(self.gamma_mode)

        if self.linear:
            K_sv_loss = torch.matmul(z_sv, z_loss.t())
        else:
            if "StopGradSV" in self.jz_mode:
                dists = torch.cdist(z_sv.detach(), z_loss, p=2).pow(2)
            elif "StopGradLoss" in self.jz_mode:
                dists = torch.cdist(z_sv, z_loss.detach(), p=2).pow(2)
            else:
                dists = torch.cdist(z_sv, z_loss, p=2).pow(2)
            K_sv_loss = torch.exp(-gamma * dists)

        sv_mask = ((alpha - 1 / (self.nu * n))**2 < (1 / (self.nu * n) - 1e-6)**2).float()
        rho = torch.sum(alpha.view(1, -1) @ K_sv @ sv_mask.view(-1, 1)) / (sv_mask.sum() + 1e-6)

        decision = (alpha.view(1, -1) @ K_sv_loss - rho) * self.nu * n
        if "StopGradSV" in self.jz_mode:
            decision = decision.detach()
        elif "StopGradLoss" in self.jz_mode:
            decision = decision

        obj = (1 / self.nu) * F.relu(-decision).sum()
        return obj.squeeze()

    def compute_loss(self, x, training=True):
        x_hat, z = self(x)
        mse = F.mse_loss(x_hat, x, reduction='mean')
        alpha, K_sv = self.solve_ocsvm(z, training=training)
        ocsvm_obj = self.ocsvm_objective(alpha, z, K_sv)
        total = mse + self.ocsvm_coeff * ocsvm_obj
        return total, mse, ocsvm_obj


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
        mse_recons = F.mse_loss(x_hat, x, reduction='mean')
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        z_center = F.mse_loss(z, self.center.expand_as(z), reduction='mean')
        return self.balance_coeff * (mse_recons + self.beta_kl * kl) + z_center


def standardize(x):
    return (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)


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
