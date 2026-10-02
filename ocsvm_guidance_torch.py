"""OCSVM-guidance of autoencoder training (OgAE), PyTorch implementation.

Notation follows the paper (Sec. III-B, eq. 7): each batch z is split into z_sv, used to solve the OC-SVM dual
problem (alpha*, rho*), and z_L (here `z_loss`), on which misclassified samples are penalized. The penalty is split into
  - the expander term (weight beta1): gradient flows through z_sv (and alpha*, rho*), z_loss is stopped,
  - the compactor term (weight beta2): gradient flows through z_loss only, z_sv, alpha* and rho* are stopped.
Both terms have the same forward value, so the loss value does not depend on (beta1, beta2): only the gradient does
(beta1 * g_expander + beta2 * g_compactor). (beta1, beta2) = (1, 1) is the full, unsplit gradient.

The paper's best schedule "(1, 0) -> (0.5, 0.5)" was run with the full gradient: call set_betas(1, 0) for the first half
of the epochs, then set_betas(1, 1). (0.5, 0.5) gives the same direction with half the OCSVM-guidance gradient magnitude.
"""
from collections import deque

import cvxpy as cp
import torch
import torch.nn as nn
import torch.nn.functional as F
from cvxpylayers.torch import CvxpyLayer

STD_EPS = 1e-6  # avoids division by zero when standardizing a collapsed latent dimension
FREE_SV_TOL = 1e-6  # tolerance on alpha bounds to identify free support vectors (used to compute rho)


def build_ocsvm_dual_layer(n, nu):
    """Differentiable solver of the scaled OC-SVM dual problem for n points (alpha_scaled = nu * n * alpha).

    sum_squares(K^1/2 @ alpha) == alpha.T @ K @ alpha, written this way so the problem is linear in its parameter (DPP).
    """
    alpha = cp.Variable(n)
    k_sqrt = cp.Parameter((n, n))
    constraints = [cp.sum(alpha) == nu * n, alpha >= 0, alpha <= 1]
    problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(k_sqrt @ alpha)), constraints)
    return CvxpyLayer(problem, parameters=[k_sqrt], variables=[alpha])


def standardize(z):
    mean, std = z.mean(dim=0), z.std(dim=0, unbiased=False) + STD_EPS
    return (z - mean) / std, mean, std


class OCSVMGuidedAutoencoderBase(nn.Module):
    """Autoencoder trained with reconstruction + lambda * OCSVM-guidance loss. Subclasses define self.encoder and
    self.decoder modules.

    Args:
        batch_size_train, batch_size_valid: batch sizes (must be even, one OC-SVM layer is built per batch size).
        ocsvm_coeff: lambda, weight of the OCSVM-guidance term.
        nu_ocsvm_coeff: nu of the OC-SVM.
        gamma_rbf_coeff: RBF gamma, a number, "scale" (1 / (dim * var(z_sv))) or "auto" (1 / dim), as in sklearn.
        beta1, beta2: weights of the expander and compactor gradients (see module docstring), can be changed during
            training with set_betas().
        differentiate_dual: if False, the expander gradient does not flow through alpha* and rho*.
        standardize_z: standardize z_sv and z_loss (each with its own batch statistics) before the kernel.
        linear_kernel: use a linear kernel instead of the RBF kernel.
        n_last_ocsvms: number M of OC-SVMs of the last training iterations kept for decision_function() (0: none,
            a final OC-SVM must then be trained on encode() outputs).
    """

    def __init__(self, batch_size_train, batch_size_valid, ocsvm_coeff=1e-2, nu_ocsvm_coeff=0.03, gamma_rbf_coeff=1e-2,
                 beta1=1.0, beta2=0.0, differentiate_dual=True, standardize_z=True, linear_kernel=False,
                 n_last_ocsvms=0):
        super().__init__()
        self.ocsvm_coeff = ocsvm_coeff
        self.nu = nu_ocsvm_coeff
        if gamma_rbf_coeff not in ("scale", "auto") and not isinstance(gamma_rbf_coeff, (int, float)):
            raise ValueError(f"gamma_rbf_coeff must be a number, 'scale' or 'auto', got {gamma_rbf_coeff!r}")
        self.gamma_rbf_coeff = gamma_rbf_coeff
        self.set_betas(beta1, beta2)
        self.differentiate_dual = differentiate_dual
        self.standardize_z = standardize_z
        self.linear_kernel = linear_kernel
        self.n_last_ocsvms = n_last_ocsvms
        self.ocsvm_buffer = deque(maxlen=n_last_ocsvms)

        # Half of each batch is used to solve the OC-SVM problem, the other half for the loss
        if batch_size_train % 2 or batch_size_valid % 2:
            raise ValueError("Batch sizes must be even (each batch is split in z_sv and z_loss)")
        self.ocsvm_layers = {n // 2: build_ocsvm_dual_layer(n // 2, nu_ocsvm_coeff)
                             for n in {batch_size_train, batch_size_valid}}

    def set_betas(self, beta1, beta2):
        self.beta1, self.beta2 = float(beta1), float(beta2)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    @torch.no_grad()
    def encode(self, x):
        return self.encoder(x)

    def _gamma(self, z_sv):
        if self.gamma_rbf_coeff == "scale":
            return 1 / (z_sv.shape[1] * z_sv.detach().var(unbiased=False).clamp_min(1e-32))  # clamp: collapse
        if self.gamma_rbf_coeff == "auto":
            return 1 / z_sv.shape[1]
        return self.gamma_rbf_coeff

    def _kernel(self, a, b, gamma):
        """Kernel matrix K_ij = k(a_i, b_j)."""
        if self.linear_kernel:
            return a @ b.T
        # Explicit squared distances rather than cdist, whose gradient is unstable at zero distance
        return torch.exp(-gamma * ((a[:, None, :] - b[None, :, :]) ** 2).sum(dim=-1))

    def solve_ocsvm_problem(self, z_sv, gamma):
        """Returns the (unscaled) dual solution alpha* and the kernel matrix of z_sv."""
        n_half = z_sv.shape[0]
        k_sv = self._kernel(z_sv, z_sv, gamma)
        num_stability_coeff = 1e-8 if self.linear_kernel else 1e-8 / gamma
        eye = torch.eye(n_half, dtype=torch.float64, device=z_sv.device)
        # K = L @ L.T, so ||L.T @ alpha||^2 = alpha.T @ K @ alpha
        k_sqrt = torch.linalg.cholesky(k_sv.double() + num_stability_coeff * eye).mT
        alpha_scaled, = self.ocsvm_layers[n_half](k_sqrt)
        return alpha_scaled.to(z_sv.dtype) / (self.nu * n_half), k_sv

    def _rho(self, alpha, k_sv):
        """rho* averaged over free support vectors (0 < alpha_j < 1 / (nu * n)) for stability, as in LIBSVM."""
        upper_bound = 1 / (self.nu * alpha.shape[0])
        free_sv = ((alpha > FREE_SV_TOL) & (alpha < upper_bound - FREE_SV_TOL)).to(alpha.dtype)
        # Guard: with no free SV, 0/0 would give a NaN that poisons the loss
        return ((k_sv @ alpha) * free_sv).sum() / free_sv.sum().clamp_min(1.)

    def _decision_function(self, alpha, rho, k):
        """Decision function of the OC-SVM on the columns of k, de-normalized from the scaled problem."""
        return (alpha @ k - rho) * self.nu * alpha.shape[0]

    def ocsvm_guidance_loss(self, z, store_ocsvm=False):
        z_sv, z_loss = torch.chunk(z, 2, dim=0)
        z_sv_std, sv_mean, sv_std = standardize(z_sv)
        if self.standardize_z:
            z_sv, z_loss = z_sv_std, standardize(z_loss)[0]
        gamma = self._gamma(z_sv)
        alpha, k_sv = self.solve_ocsvm_problem(z_sv, gamma)
        rho = self._rho(alpha, k_sv)

        if store_ocsvm:
            self.ocsvm_buffer.append(dict(alpha=alpha.detach(), rho=rho.detach(), z_sv=z_sv.detach(), gamma=gamma,
                                          mean=sv_mean.detach(), std=sv_std.detach()))

        alpha_exp, rho_exp = (alpha, rho) if self.differentiate_dual else (alpha.detach(), rho.detach())
        decision_exp = self._decision_function(alpha_exp, rho_exp, self._kernel(z_sv, z_loss.detach(), gamma))
        decision_comp = self._decision_function(alpha.detach(), rho.detach(),
                                                self._kernel(z_sv.detach(), z_loss, gamma))
        # Penalize only misclassified z_loss (negative decision function), nu as upper bound of outliers fraction
        loss_exp = F.relu(-decision_exp).sum() / self.nu
        loss_comp = F.relu(-decision_comp).sum() / self.nu

        # Straight-through weighting: forward value is the unsplit loss, gradient is beta1 * g_exp + beta2 * g_comp.
        # Zero-weighted terms are skipped so they can never propagate a NaN.
        loss = loss_exp.detach()
        if self.beta1:
            loss = loss + self.beta1 * (loss_exp - loss_exp.detach())
        if self.beta2:
            loss = loss + self.beta2 * (loss_comp - loss_comp.detach())
        return loss

    @torch.no_grad()
    def decision_function(self, x):
        """Anomaly score (negative outside the support) as the mean of the last M stored OC-SVMs' decision functions.

        New samples are standardized with the statistics of each stored z_sv.
        """
        if not self.ocsvm_buffer:
            raise RuntimeError("No stored OC-SVM: train with n_last_ocsvms > 0, or fit a final OC-SVM on encode()")
        z = self.encode(x)
        scores = []
        for ocsvm in self.ocsvm_buffer:
            z_ = (z - ocsvm["mean"]) / ocsvm["std"] if self.standardize_z else z
            k = self._kernel(ocsvm["z_sv"], z_, ocsvm["gamma"])
            scores.append(self._decision_function(ocsvm["alpha"], ocsvm["rho"], k))
        return torch.stack(scores).mean(dim=0)

    def compute_loss(self, x):
        """Returns total loss, reconstruction loss and OCSVM-guidance loss. OC-SVMs are stored in training mode."""
        x_hat, z = self(x)
        mse = F.mse_loss(x_hat, x)
        ocsvm_obj = self.ocsvm_guidance_loss(z, store_ocsvm=self.training and self.n_last_ocsvms > 0)
        return mse + self.ocsvm_coeff * ocsvm_obj, mse, ocsvm_obj
