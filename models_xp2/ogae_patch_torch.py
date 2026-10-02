import torch.nn as nn

from ocsvm_guidance_torch import OCSVMGuidedAutoencoderBase


def conv_block(conv, out_channels):
    return nn.Sequential(conv, nn.BatchNorm2d(out_channels), nn.GELU())


class OCSVMguidedPatchAutoencoder(OCSVMGuidedAutoencoderBase):
    """OgAE with localized OCSVM for Experiment 2 (brain MRI), architecture of supplementary material S-F2.

    Takes 15x15 patches; each training batch must contain co-localized patches (same voxel location across subjects),
    so that one OC-SVM per location is estimated. The latent representation (16x5x5 feature map, flattened) is associated
    to the central voxel of the patch. After training, a final OC-SVM is fitted per location on encode() outputs.

    Defaults are the ones used for the paper: no standardization of z and a fixed gamma = 1e-2. With an unconstrained
    latent scale, expander-heavy settings can saturate the RBF kernel (k ~ 0 between subjects) and collapse the decision
    function to -rho (supplementary S-D); gamma_rbf_coeff="scale" or standardize_z=True avoid this.
    """

    def __init__(self, batch_size_train, batch_size_valid, nb_channels=1, standardize_z=False, **ogae_kwargs):
        super().__init__(batch_size_train, batch_size_valid, standardize_z=standardize_z, **ogae_kwargs)
        # Unpadded convolutions: 15 -> 11 -> 9 -> 7 -> 5
        self.encoder = nn.Sequential(
            conv_block(nn.Conv2d(nb_channels, 3, kernel_size=5), 3),
            conv_block(nn.Conv2d(3, 4, kernel_size=3), 4),
            conv_block(nn.Conv2d(4, 12, kernel_size=3), 12),
            conv_block(nn.Conv2d(12, 16, kernel_size=3), 16),
            nn.Flatten()
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (16, 5, 5)),
            conv_block(nn.ConvTranspose2d(16, 12, kernel_size=3), 12),
            conv_block(nn.ConvTranspose2d(12, 4, kernel_size=3), 4),
            conv_block(nn.ConvTranspose2d(4, 3, kernel_size=3), 3),
            nn.ConvTranspose2d(3, nb_channels, kernel_size=5),
            nn.Sigmoid()
        )
