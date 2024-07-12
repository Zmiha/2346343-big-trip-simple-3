import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from typing import Type, Callable, List, Optional
from pytorch_symbolic import Input, SymbolicModel


# Определение Conv2dAuto
class Conv2dAuto(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: Optional[int] = None, dilation: int = 1, groups: int = 1, bias: bool = True) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)


conv3x3: Callable[..., Conv2dAuto] = partial(Conv2dAuto, kernel_size=3, bias=False)


# Объединенный класс ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expansion: int = 1, downsampling: int = 1,
                 conv: Callable[..., nn.Conv2d] = conv3x3) -> None:
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv

        self.blocks = self._make_blocks()
        self.shortcut = self._make_shortcut() if self.should_apply_shortcut else nn.Identity()

    def _make_blocks(self) -> nn.Sequential:
        raise NotImplementedError("This method should be overridden by subclasses")

    def _make_shortcut(self) -> nn.Sequential:
        return nn.Sequential(OrderedDict({
            'conv': nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1, stride=self.downsampling,
                              bias=False),
            'bn': nn.BatchNorm2d(self.expanded_channels)
        }))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self) -> bool:
        return self.in_channels != self.expanded_channels

    @property
    def expanded_channels(self) -> int:
        return self.out_channels * self.expansion


# Определение ResNetBasicBlock
class ResNetBasicBlock(ResidualBlock):
    expansion = 1

    def _make_blocks(self) -> nn.Sequential:
        return nn.Sequential(
            self._conv_bn(self.in_channels, self.out_channels, stride=self.downsampling),
            nn.ReLU(),
            self._conv_bn(self.out_channels, self.expanded_channels),
        )

    def _conv_bn(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> nn.Sequential:
        return nn.Sequential(OrderedDict({
            'conv': self.conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False),
            'bn': nn.BatchNorm2d(out_channels)
        }))


# Определение ResNetBottleNeckBlock
class ResNetBottleNeckBlock(ResidualBlock):
    expansion = 1

    def _make_blocks(self) -> nn.Sequential:
        return nn.Sequential(
            self._conv_bn(self.in_channels, self.out_channels, kernel_size=1),
            nn.ReLU(),
            self._conv_bn(self.out_channels, self.out_channels, kernel_size=3, stride=self.downsampling),
            nn.ReLU(),
            self._conv_bn(self.out_channels, self.expanded_channels, kernel_size=1),
        )

    def _conv_bn(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> nn.Sequential:
        return nn.Sequential(OrderedDict({
            'conv': self.conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False),
            'bn': nn.BatchNorm2d(out_channels)
        }))


# Определение ResNetLayer
class ResNetLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, block: Type[ResidualBlock], n: int, expansion: int = 1,
                 downsampling: int = 1) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            block(in_channels, out_channels, downsampling=downsampling),
            *[block(out_channels * expansion, out_channels, downsampling=1) for _ in range(n - 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return x


# Определение ResNetEncoder
class ResNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, blocks_sizes=None,
                 depths=None, activation: Type[nn.Module] = nn.ReLU,
                 block: Type[ResidualBlock] = ResNetBasicBlock) -> None:
        super().__init__()
        if depths is None:
            depths = [2, 2, 2, 2]
        if blocks_sizes is None:
            blocks_sizes = [64, 128, 256, 512]
        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], block, n=depths[0]),
            *[ResNetLayer(in_channels * block.expansion, out_channels, block, n=n)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


# Определение ResNetDecoder
class ResNetDecoder(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


# Определение ResNet
class ResNet(SymbolicModel):
    def __init__(self, in_channels: int, blocks_sizes=None,
                 depths=None, activation: Type[nn.Module] = nn.ReLU,
                 block: Type[ResidualBlock] = ResNetBasicBlock) -> None:
        if depths is None:
            depths = [2, 2, 2, 2]
        if blocks_sizes is None:
            blocks_sizes = [64, 128, 256, 512]
        input_tensor = Input(shape=(in_channels, 224, 224))
        encoder = ResNetEncoder(in_channels, blocks_sizes=blocks_sizes, depths=depths, activation=activation,
                                block=block)
        encoded = encoder(input_tensor)
        decoder = ResNetDecoder(encoder.blocks[-1].blocks[-1].expanded_channels)
        output_tensor = decoder(encoded)
        super().__init__(inputs=input_tensor, outputs=output_tensor)


# Функции для создания моделей ResNet
def resnet(in_channels: int, architecture: str, depths=None) -> ResNet:
    default_depths = {
        'resnet18': [2, 2, 2, 2],
        'resnet34': [3, 4, 6, 3],
        'resnet50': [3, 4, 6, 3],
        'resnet101': [3, 4, 23, 3],
        'resnet152': [3, 8, 36, 3]
    }

    blocks = {
        'resnet18': ResNetBasicBlock,
        'resnet34': ResNetBasicBlock,
        'resnet50': ResNetBottleNeckBlock,
        'resnet101': ResNetBottleNeckBlock,
        'resnet152': ResNetBottleNeckBlock
    }

    if depths is None:
        depths = default_depths[architecture]

    block = blocks[architecture]

    return ResNet(in_channels, block=block, depths=depths)
