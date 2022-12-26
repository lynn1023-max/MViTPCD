from typing import List, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn.modules import Module, ModuleList, Sequential

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "nn.Module"
ModuleList.__module__ = "nn.ModuleList"
Sequential.__module__ = "nn.Sequential"


class ConvBlock(Module):
    """N-layer convolutional encoder block N x (Conv2d->BN->ReLU->Dropout)."""

    def __init__(
        self,
        channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
        pool: bool = True,
    ) -> None:
        """Initializes the convolutional encoder block.
        Args:
            channels: number of filters per conv layer
                (first element is the input channels)
            kernel_size: kernel size for each conv layer
            dropout: probability for each dropout layer
            pool: max pool last conv layer output if True
        """
        super().__init__()  # type: ignore[no-untyped-call]
        layers = []
        for i in range(1, len(channels)):
            layers.extend(
                [
                    nn.modules.Conv2d(
                        channels[i - 1],
                        channels[i],
                        kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                    ),
                    nn.modules.BatchNorm2d(channels[i]),  # type: ignore[no-untyped-call]  # noqa: E501
                    nn.modules.ReLU(),
                    nn.modules.Dropout(dropout),
                ]
            )
        self.model = Sequential(*layers)

        if pool:
            self.pool = nn.modules.MaxPool2d(kernel_size=2)
        else:
            self.pool = nn.Identity()  # type: ignore[attr-defined]

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass of the model.
        Args:
            x: input tensor
        Returns:
            pool: max pooled output of last conv layer
            x: output of last conv layer
        """
        x = self.model(x)
        return self.pool(x), x


class DeConvBlock(Sequential):
    """N-layer convolutional decoder block: N x (ConvTranspose2d->BN->ReLU->Dropout)."""

    def __init__(
        self, channels: List[int], kernel_size: int = 3, dropout: float = 0.2
    ) -> None:
        """Initializes the convolutional decoder block.
        Args:
            channels: number of filters per conv layer
                (first element is the input channels)
            kernel_size: kernel size for each conv layer
            dropout: probability for each dropout layer
        """
        super().__init__(
            *[
                Sequential(
                    nn.modules.ConvTranspose2d(
                        channels[i - 1],
                        channels[i],
                        kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.modules.BatchNorm2d(channels[i]),  # type: ignore[no-untyped-call] # noqa: E501
                    nn.modules.ReLU(),
                    nn.modules.Dropout(dropout),
                )
                for i in range(1, len(channels))
            ]
        )


class UpsampleBlock(Sequential):
    """Wrapper for nn.ConvTranspose2d upsampling layer."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        """Initializes the upsampling block.
        Args:
            channels: number of filters for the ConvTranspose2d layer
            kernel_size: kernel size for the ConvTranspose2d layer
        """
        super().__init__(
            nn.modules.ConvTranspose2d(
                channels,
                channels,
                kernel_size,
                padding=kernel_size // 2,
                stride=2,
                output_padding=1,
            )
        )


class Encoder(ModuleList):
    """4-layer convolutional encoder."""

    def __init__(self, in_channels: int = 3, pool: bool = True) -> None:
        """Initializes the encoder.
        Args:
            in_channels: number of input channels
            pool: max pool last conv block output if True
        """
        super().__init__(
            [
                ConvBlock([in_channels, 16, 16]),
                ConvBlock([16, 32, 32]),
                ConvBlock([32, 64, 64, 64]),
                ConvBlock([64, 128, 128, 128], pool=pool),
            ]
        )


class Decoder(ModuleList):
    """4-layer convolutional decoder."""

    def __init__(self, classes: int = 2) -> None:
        """Initializes the decoder.
        Args:
            classes: number of output segmentation classes
                (default=2 for binary segmentation)
        """
        super().__init__(
            [
                DeConvBlock([256, 128, 128, 64]),
                DeConvBlock([128, 64, 64, 32]),
                DeConvBlock([64, 32, 16]),
                DeConvBlock([32, 16, classes]),
            ]
        )


class ConcatDecoder(ModuleList):
    """4-layer convolutional decoder supporting concatenated inputs from encoder."""

    def __init__(self, t: int = 2, classes: int = 2) -> None:
        """Initializes the decoder.
        Args:
            t: number of input images being compared for change
            classes: number of output segmentation classes
                (default=2 for binary segmentation)
        """
        scale = 0.5 * (t + 1)
        super().__init__(
            [
                DeConvBlock([int(256 * scale), 128, 128, 64]),
                DeConvBlock([int(128 * scale), 64, 64, 32]),
                DeConvBlock([int(64 * scale), 32, 16]),
                DeConvBlock([int(32 * scale), 16, classes]),
            ]
        )


class Upsample(ModuleList):
    """Upsampling layers in decoder."""

    def __init__(self) -> None:
        """Initializes the upsampling module."""
        super().__init__(
            [
                UpsampleBlock(128),
                UpsampleBlock(64),
                UpsampleBlock(32),
                UpsampleBlock(16),
            ]
        )


class dconv(nn.Module):
    def __init__(self, dims):
        super(dconv,self).__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv2d(dims[0], dims[-1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[-1]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        y = self.conv_branch(x)
        return y


class FCSiamDiff(nn.modules.Module):
    """Fully-convolutional Siamese Difference (FC-Siam-diff).
    'Fully Convolutional Siamese Networks for Change Detection', Daudt et al. (2018)
    If you use this model in your research, please cite the following paper:
    * https://arxiv.org/abs/1810.08462
    """

    def __init__(self, in_channels: int = 3, t: int = 2, classes: int = 2) -> None:
        """Initializes the FCSiamDiff module.
        Args:
            in_channels: number of channels per input image
            t: number of input images being compared for change
            classes: number of output segmentation classes
                (default=2 for binary segmentation)
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.encoder = Encoder(in_channels, pool=False)
        self.decoder = Decoder(classes)
        self.upsample = Upsample()
        self.pool = nn.modules.MaxPool2d(kernel_size=2)

        self.deconv1 = dconv(*[[16, 32]])
        self.deconv2 = dconv(*[[64, 64]])
        self.deconv3 = dconv(*[[128, 128]])
        self.deconv4 = dconv(*[[256, 256]])
        self.pool = nn.AvgPool2d(3)
        self.fc = nn.Linear(256, 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        b, t, c, h, w = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")

        skips = []
        for block in self.encoder:
            x, skip = block(x)
            skips.append(skip)

        # Diff skips
        skips = [rearrange(skip, "(b t) c h w -> b t c h w", t=t)
                 for skip in skips]
        diffs = []
        for skip in skips:
            diff, xt = skip[:, 0, ...], skip[:, 1:, ...]
            for i in range(t - 1):
                # type: ignore[attr-defined]
                diff = torch.abs(diff - xt[:, i, ...])
            diffs.append(diff)

        # Only first input encoding is passed directly to decoder
        x = rearrange(x, "(b t) c h w -> b t c h w", t=t)
        x = x[:, 0, ...]
        x = self.pool(x)

        d1 = self.deconv1(diffs[0])
        r1 = torch.cat([d1, diffs[1]], dim=1)
        d2 = self.deconv2(r1)
        r2 = torch.cat([d2, diffs[2]], dim=1)
        d3 = self.deconv3(r2)
        r3 = torch.cat([d3, diffs[3]], dim=1)
        d4 = self.deconv4(r3)
        d5 = self.deconv4(d4)
        r5 = self.pool(d5).view(-1, d5.shape[1])
        y  = self.fc(r5)

        # for block, upsample, skip in zip(self.decoder, self.upsample, reversed(diffs)):
        #     x = upsample(x)
        #     x = torch.cat([x, skip], dim=1)  # type: ignore[attr-defined]
        #     x = block(x)

        return y


if __name__ == '__main__':
    input = torch.randn((1, 3, 128, 128))
    B,C,H,W = input.shape
    a = input.reshape(B,1,C,H,W)

    incat = torch.cat((a,a),dim=1)
    print(incat.shape)
    # input=torch.randn((2, 2, 3, 128, 128))
    model = FCSiamDiff()  # FCEF 参数量：1.350582 计算量： 0.894369792
    output = model(incat) # FCSiamConc 参数量：1.54599 计算量： 1.332805632
    print(output.shape)   # FCSiamDiff 参数量：1.35015 计算量： 1.181810688
    #
    from thop import profile
    sflops, sparams = profile(model, inputs=(incat,))
    print("参数量：{} 计算量： {}".format(sparams/1e+6, sflops/1e+9))