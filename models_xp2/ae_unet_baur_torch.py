import torch
import torch.nn as nn
import torchvision.transforms as T


def standardize(x):
    return (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)


class UnetBaur(nn.Module):
    def __init__(self, out_chan=2, filters=(64, 128, 256, 512, 64), preprocessing=True):
        super().__init__()
        self.preprocessing = preprocessing
        if preprocessing:
            self.preprocess = T.Compose([
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=0),
                RandomBrightnessPyTorch((-0.1, 0.1), (0.0, 1.0)),
                T.RandomAdjustSharpness(1.0, p=0.5)  # Approximate random contrast
            ])

        self.encoder_blocks = nn.ModuleList()
        for i, nf in enumerate(filters):
            stride = 2 if i != 0 else 1
            block = nn.Sequential(
                nn.Conv2d(filters[i - 1] if i > 0 else 1, nf, kernel_size=5, stride=stride, padding=2),
                nn.BatchNorm2d(nf),
                nn.ReLU()
            )
            self.encoder_blocks.append(block)

        self.decoder_blocks = nn.ModuleList()
        self.decoder_blocks.append(nn.Sequential(
            nn.ConvTranspose2d(filters[-1], filters[-2], kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(filters[-2]),
            nn.ReLU()
        ))
        self.decoder_blocks.append(nn.Sequential(
            nn.ConvTranspose2d(filters[-2], filters[-3], kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(filters[-3]),
            nn.ReLU()
        ))
        self.decoder_blocks.append(nn.Sequential(
            nn.ConvTranspose2d(filters[-3] * 2, filters[-4], kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(filters[-4]),
            nn.ReLU()
        ))
        self.decoder_blocks.append(nn.Sequential(
            nn.ConvTranspose2d(filters[-4] * 2, filters[-5], kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(filters[-5]),
            nn.ReLU()
        ))
        self.final_conv = nn.Conv2d(filters[0], out_chan, kernel_size=1, stride=1)

    def forward(self, x):
        if self.preprocessing:
            x = self.preprocess(x)

        x0 = self.encoder_blocks[0](x)
        x1 = self.encoder_blocks[1](x0)
        x2 = self.encoder_blocks[2](x1)
        x3 = self.encoder_blocks[3](x2)
        x4 = self.encoder_blocks[4](x3)

        d0 = self.decoder_blocks[0](x4)
        d1 = self.decoder_blocks[1](d0)
        d2 = self.decoder_blocks[2](torch.cat([d1, x2], dim=1))
        d3 = self.decoder_blocks[3](torch.cat([d2, x1], dim=1))

        return torch.tanh(self.final_conv(d3))


class RandomBrightnessPyTorch(nn.Module):
    def __init__(self, factor, value_range):
        super().__init__()
        self.factor = factor
        self.min_val, self.max_val = value_range

    def forward(self, x):
        if not self.training:
            return x
        b = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(self.factor[0], self.factor[1])
        x = x + b
        return torch.clamp(x, self.min_val, self.max_val)
