import torch.nn as nn

from ocsvm_guidance_torch import OCSVMGuidedAutoencoderBase


class OCSVMguidedPatchAutoencoder(OCSVMGuidedAutoencoderBase):
    """OgAE with localized OCSVM for Experiment 2 (brain MRI).

    Takes 15x15 patches, encoded into a latent vector of dimension 16 associated to the central voxel of the patch. Each
    training batch must contain co-localized patches (same voxel location across subjects), so that one OC-SVM per
    location is estimated. After training, a final OC-SVM is trained per location on the latent representations.

    Defaults are the ones used for the paper: no standardization of z and a fixed gamma = 1e-2. With an unconstrained
    latent scale, expander-heavy settings can saturate the RBF kernel (k ~ 0 between subjects) and collapse the decision
    function to -rho (supplementary S-D); gamma_rbf_coeff="scale" avoids this.
    """

    def __init__(self, batch_size_train, batch_size_valid, nb_channels=1, **ogae_kwargs):
        super().__init__(batch_size_train, batch_size_valid, **ogae_kwargs)
        # Encoder : 15x15 -> 11x11 -> 9x9 -> 3x3 -> 1x1, BatchNorm after the activation
        self.encoder = nn.Sequential(
            nn.Conv2d(nb_channels, 3, kernel_size=5), nn.GELU(), nn.BatchNorm2d(3),
            nn.Conv2d(3, 4, kernel_size=3), nn.GELU(), nn.BatchNorm2d(4),
            nn.Conv2d(4, 12, kernel_size=3, stride=3), nn.GELU(), nn.BatchNorm2d(12),
            nn.Conv2d(12, 16, kernel_size=3), nn.GELU(), nn.BatchNorm2d(16),
            nn.Flatten()  # 16x1x1 -> 16
        )
        # Decoder : 1x1 -> 3x3 -> 9x9 -> 11x11 -> 15x15
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (16, 1, 1)),
            nn.ConvTranspose2d(16, 12, kernel_size=3), nn.GELU(), nn.BatchNorm2d(12),
            nn.ConvTranspose2d(12, 4, kernel_size=3, stride=3), nn.GELU(), nn.BatchNorm2d(4),
            nn.ConvTranspose2d(4, 3, kernel_size=3), nn.GELU(), nn.BatchNorm2d(3),
            nn.ConvTranspose2d(3, nb_channels, kernel_size=5), nn.Sigmoid()
        )
