"""OCSVM-guidance of autoencoder training (OgAE), PyTorch implementation.

Notation follows the paper (Sec. III-B, eq. 7): each batch z is split into z_sv, used to solve the OC-SVM dual
problem (alpha*, rho*), and z_L (here `z_loss`), on which misclassified samples are penalized. The penalty is split into
  - the expander term (weight beta1): gradient flows through z_sv (and alpha*, rho*), z_loss is stopped,
  - the compactor term (weight beta2): gradient flows through z_loss only, z_sv, alpha* and rho* are stopped.
Both terms have the same forward value, so the loss value does not depend on (beta1, beta2): only the gradient does
(beta1 * g_expander + beta2 * g_compactor). The expander -> expander + compactor schedule is obtained by calling
set_betas(1, 0), then set_betas(0.5, 0.5) at mid-training.
"""
from collections import deque

import cvxpy as cp
import torch
import torch.nn as nn
import torch.nn.functional as F
from cvxpylayers.torch import CvxpyLayer


def build_ocsvm_layer(n, nu):
    """CVXPyLayers OC-SVM dual problem for n samples, scaled problem with alpha_scaled = nu * n * alpha"""
    alpha = cp.Variable(n)
    k_sqrt = cp.Parameter((n, n))  # K^1/2 such that K^1/2.T @ K^1/2 = K
    constraints = [cp.sum(alpha) == nu * n, alpha >= 0, alpha <= 1]
    # ||K^1/2 @ alpha||^2 == alpha.T @ K @ alpha
    objective = cp.Minimize(0.5 * cp.sum_squares(k_sqrt @ alpha))
    return CvxpyLayer(cp.Problem(objective, constraints), parameters=[k_sqrt], variables=[alpha])


def standardize(x, x_ref):
    return (x - x_ref.mean(dim=0)) / (x_ref.std(dim=0, unbiased=False) + 1e-6)


class OCSVMGuidedAutoencoderBase(nn.Module):
    """Autoencoder trained with reconstruction + lambda * OCSVM-guidance loss. Subclasses define self.encoder and
    self.decoder modules.

    Args:
        batch_size_train, batch_size_valid: batch sizes, split in two (z_sv and z_loss).
        ocsvm_coeff: lambda, weight of the OCSVM-guidance term.
        nu_ocsvm_coeff: nu of the OC-SVM.
        gamma_rbf_coeff: RBF gamma, a number, "scale" or "auto" (as in sklearn).
        beta1, beta2: weights of the expander and compactor gradients (see module docstring), can be changed during
            training with set_betas().
        differentiate_dual: if False, the expander gradient does not flow through alpha and rho.
        standardize_z: standardize z_sv and z_loss (each with its own batch statistics) before the kernel.
        linear_kernel: linear kernel instead of the RBF kernel.
        n_last_ocsvms: number M of OC-SVMs of the last training iterations kept for decision_function() (0: none,
            a final OC-SVM must then be trained on the latent representations).
    """

    def __init__(self, batch_size_train, batch_size_valid, ocsvm_coeff=1e-2, nu_ocsvm_coeff=0.03, gamma_rbf_coeff=1e-2,
                 beta1=1.0, beta2=0.0, differentiate_dual=True, standardize_z=True, linear_kernel=False,
                 n_last_ocsvms=0):
        super().__init__()
        self.ocsvm_coeff = ocsvm_coeff
        self.nu = nu_ocsvm_coeff
        if gamma_rbf_coeff not in ("scale", "auto") and not isinstance(gamma_rbf_coeff, (int, float)):
            raise ValueError(str(gamma_rbf_coeff) + " not implemented or non-valid")
        self.gamma_mode = gamma_rbf_coeff
        self.set_betas(beta1, beta2)
        self.differentiate_dual = differentiate_dual
        self.standardize_z = standardize_z
        self.linear = linear_kernel
        # OC-SVMs (alpha, rho, z_sv, gamma) of the last M training iterations, for the decision function
        self.n_last_ocsvms = n_last_ocsvms
        self.last_ocsvms = deque(maxlen=n_last_ocsvms)

        # CVXPY problems for training and validation, each batch is split in two (z_sv and z_loss)
        self.ocsvm_layer_train = build_ocsvm_layer(batch_size_train // 2, self.nu)
        self.ocsvm_layer_valid = build_ocsvm_layer(batch_size_valid // 2, self.nu)

    def set_betas(self, beta1, beta2):
        self.beta1, self.beta2 = float(beta1), float(beta2)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def gamma(self, z_sv):
        if self.gamma_mode == "scale":
            var = z_sv.detach().var(unbiased=False)
            return 1 / (z_sv.shape[1] * var + 1e-6)
        elif self.gamma_mode == "auto":
            return 1 / z_sv.shape[1]
        return float(self.gamma_mode)

    def kernel(self, z_a, z_b, gamma):
        if self.linear:
            return torch.matmul(z_a, z_b.t())
        dist_sq = ((z_a[:, None, :] - z_b[None, :, :]) ** 2).sum(dim=-1)  # not cdist, its gradient is nan at distance 0
        return torch.exp(-gamma * dist_sq)

    def solve_ocsvm(self, z, training=True):
        n = z.shape[0] // 2
        z_sv, _ = torch.split(z, n, dim=0)
        if self.standardize_z:
            z_sv = standardize(z_sv, z_sv)
        gamma = self.gamma(z_sv)
        K = self.kernel(z_sv, z_sv, gamma)

        eps = 1e-8 / gamma if not self.linear else 1e-8
        # Cholesky K = L @ L.T, so K^1/2 = L.T (in float64 for stability)
        K_sqrt = torch.linalg.cholesky(K.double() + eps * torch.eye(n, dtype=torch.float64, device=z.device)).mT
        layer = self.ocsvm_layer_train if training else self.ocsvm_layer_valid
        alpha, = layer(K_sqrt)
        alpha = alpha.to(z.dtype) / (self.nu * n)  # return to the unscaled problem
        return alpha, K

    def rho(self, alpha, K_sv):
        n = alpha.shape[0]
        # SV in the band 0 < alpha < 1/(nu * n), rho is their mean (clamp : no SV in the band would give 0/0)
        sv_mask = ((alpha - 1 / (2 * self.nu * n))**2 < (1 / (2 * self.nu * n) - 1e-6)**2).to(alpha.dtype)
        return torch.sum(alpha.view(1, -1) @ K_sv @ sv_mask.view(-1, 1)) / sv_mask.sum().clamp_min(1.)

    def decision(self, alpha, rho, K):
        return (alpha.view(1, -1) @ K - rho) * self.nu * alpha.shape[0]  # de-normalization of the scaled problem

    def ocsvm_objective(self, alpha, z, K_sv, store_ocsvm=False):
        n = z.shape[0] // 2
        z_sv_raw, z_loss = torch.split(z, n, dim=0)
        z_sv = z_sv_raw
        if self.standardize_z:
            z_sv, z_loss = standardize(z_sv, z_sv), standardize(z_loss, z_loss)
        gamma = self.gamma(z_sv)
        rho = self.rho(alpha, K_sv)
        if store_ocsvm:
            self.last_ocsvms.append({"alpha": alpha.detach(), "rho": rho.detach(), "z_sv": z_sv_raw.detach(), "gamma": gamma})

        # Expander : gradient only through z_sv (and alpha, rho if differentiate_dual)
        alpha_exp, rho_exp = (alpha, rho) if self.differentiate_dual else (alpha.detach(), rho.detach())
        decision_exp = self.decision(alpha_exp, rho_exp, self.kernel(z_sv, z_loss.detach(), gamma))
        # Compactor : gradient only through z_loss
        decision_comp = self.decision(alpha.detach(), rho.detach(), self.kernel(z_sv.detach(), z_loss, gamma))

        obj_exp = (1 / self.nu) * F.relu(-decision_exp).sum()
        obj_comp = (1 / self.nu) * F.relu(-decision_comp).sum()
        # Both terms have the same value, only their gradients differ : the value of the objective is kept and its
        # gradient is beta1 * grad_expander + beta2 * grad_compactor (a term weighted by 0 is skipped : never gives nan)
        obj = obj_exp.detach()
        if self.beta1:
            obj = obj + self.beta1 * (obj_exp - obj_exp.detach())
        if self.beta2:
            obj = obj + self.beta2 * (obj_comp - obj_comp.detach())
        return obj.squeeze()

    @torch.no_grad()
    def decision_function(self, x):
        """Anomaly score (negative outside the support), mean of the decision functions of the last M OC-SVMs"""
        if not self.last_ocsvms:
            raise RuntimeError("No stored OC-SVM : train with n_last_ocsvms > 0, or train a final OC-SVM on the latent representations")
        z = self.encoder(x)
        decisions = []
        for ocsvm in self.last_ocsvms:
            z_sv, z_ = ocsvm["z_sv"], z
            if self.standardize_z:  # new samples standardized with the statistics of z_sv
                z_sv, z_ = standardize(z_sv, ocsvm["z_sv"]), standardize(z, ocsvm["z_sv"])
            decisions.append(self.decision(ocsvm["alpha"], ocsvm["rho"], self.kernel(z_sv, z_, ocsvm["gamma"])).squeeze(0))
        return torch.stack(decisions).mean(dim=0)

    def compute_loss(self, x, training=True):
        x_hat, z = self(x)
        mse = F.mse_loss(x_hat, x, reduction='mean')
        alpha, K_sv = self.solve_ocsvm(z, training=training)
        ocsvm_obj = self.ocsvm_objective(alpha, z, K_sv, store_ocsvm=training and self.n_last_ocsvms > 0)
        total = mse + self.ocsvm_coeff * ocsvm_obj
        return total, mse, ocsvm_obj
