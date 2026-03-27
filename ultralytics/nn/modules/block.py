# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CIB",
    "DFL",
    "ELAN1",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "AConv",
    "ADown",
    "Attention",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ContrastiveHead",
    "GhostBottleneck",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Proto",
    "RepC3",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "TorchVision",
    "WaveletSpatialAttention",
    "PrototypeAttention",
    "C2fWP",
)


class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        """Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of pooling iterations.
            shortcut (bool): Whether to use shortcut connection.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """Initialize RepC3 module with RepConv blocks.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skip connection and addition to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP bottleneck with 4 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(
        self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """Forward pass of ImagePoolingAttn.

        Args:
            x (list[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    @staticmethod
    def forward_fuse(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Passes image features through unchanged after fusing."""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (list[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx: list[int]):
        """Initialize CBFuse module.

        Args:
            idx (list[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through CBFuse layer.

        Args:
            xs (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize CSP bottleneck layer with three convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            attn (bool): Whether to use attention blocks.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth-wise convolutional block in RepVGG architecture."""

    def __init__(self, ed: int) -> None:
        """Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth-wise convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the fused RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth-wise convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        if not hasattr(self, "conv1"):
            return  # already fused
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """Compact Inverted Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use large kernel. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use large kernel.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c1.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature
    extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c2.
        m (nn.ModuleList): List of PSABlock modules for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.block import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n))


class SCDown(nn.Module):
    """SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules.block import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and
    customize the model by truncating or unwrapping layers.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): Unwraps the model to a sequential containing all but the last `truncate` layers.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.
    """

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | list[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided into.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided into.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided into.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided into.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor.

        Args:
            gc (int): Guide channels.
            ec (int): Embedding channels.
            e (int): Expansion factor.
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m: nn.Module) -> None:
        """Initialize residual module with the wrapped module.

        Args:
            m (nn.Module): Module to wrap with residual connection.
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch: list[int], c3: int, embed: int):
        """Initialize SAVPE module with channels, intermediate channels, and embedding dimension.

        Args:
            ch (list[int]): List of input channel dimensions.
            c3 (int): Intermediate channels.
            embed (int): Embedding dimension.
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: list[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
        score = F.softmax(score, dim=-1).to(y.dtype)
        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)


class Proto26(Proto):
    """Ultralytics YOLO26 models mask Proto module for segmentation models."""

    def __init__(self, ch: tuple = (), c_: int = 256, c2: int = 32, nc: int = 80):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            ch (tuple): Tuple of channel sizes from backbone feature maps.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
            nc (int): Number of classes for semantic segmentation.
        """
        super().__init__(c_, c_, c2)
        self.feat_refine = nn.ModuleList(Conv(x, ch[0], k=1) for x in ch[1:])
        self.feat_fuse = Conv(ch[0], c_, k=3)
        self.semseg = nn.Sequential(Conv(ch[0], c_, k=3), Conv(c_, c_, k=3), nn.Conv2d(c_, nc, 1))

    def forward(self, x: torch.Tensor, return_semseg: bool = True) -> torch.Tensor:
        """Perform a forward pass by fusing multi-scale feature maps and generating proto masks."""
        feat = x[0]
        for i, f in enumerate(self.feat_refine):
            up_feat = f(x[i + 1])
            up_feat = F.interpolate(up_feat, size=feat.shape[2:], mode="nearest")
            feat = feat + up_feat
        p = super().forward(self.feat_fuse(feat))
        if self.training and return_semseg:
            semseg = self.semseg(feat)
            return (p, semseg)
        return p

    def fuse(self):
        """Fuse the model for inference by removing the semantic segmentation head."""
        self.semseg = None


class RealNVP(nn.Module):
    """RealNVP: a flow-based generative model.

    References:
        https://arxiv.org/abs/1605.08803
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/realnvp.py
    """

    @staticmethod
    def nets():
        """Get the scale model in a single invertible mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2), nn.Tanh())

    @staticmethod
    def nett():
        """Get the translation model in a single invertible mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))

    @property
    def prior(self):
        """The prior distribution."""
        return torch.distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self):
        super().__init__()

        self.register_buffer("loc", torch.zeros(2))
        self.register_buffer("cov", torch.eye(2))
        self.register_buffer("mask", torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))

        self.s = torch.nn.ModuleList([self.nets() for _ in range(len(self.mask))])
        self.t = torch.nn.ModuleList([self.nett() for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """Apply mapping from the data space to the latent space and calculate the log determinant of the Jacobian
        matrix.
        """
        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """Calculate the log probability of given sample in data space."""
        if x.dtype == torch.float32 and self.s[0][0].weight.dtype != torch.float32:
            self.float()
        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KL_LOSS_CONTAINER = []
# ===================== 小波变换核心工具函数 =====================


def get_haar_wavelet_recon_kernel(in_channels, dtype=torch.float32):
    """生成Haar小波重构核（转置卷积用）"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL.T).unsqueeze(0).to(dtype)
    filter_LH = torch.from_numpy(harr_wav_LH.T).unsqueeze(0).to(dtype)
    filter_HL = torch.from_numpy(harr_wav_HL.T).unsqueeze(0).to(dtype)
    filter_HH = torch.from_numpy(harr_wav_HH.T).unsqueeze(0).to(dtype)

    filter_LL = filter_LL.expand(in_channels, -1, -1, -1)
    filter_LH = filter_LH.expand(in_channels, -1, -1, -1)
    filter_HL = filter_HL.expand(in_channels, -1, -1, -1)
    filter_HH = filter_HH.expand(in_channels, -1, -1, -1)

    return filter_LL, filter_LH, filter_HL, filter_HH


# ===================== 核心注意力模块（精准匹配你的逻辑） =====================
# -------------------------- 小波空间注意力（带残差） --------------------------
class WaveletSpatialAttention_old(nn.Module):
    """
    最终精准版：LL与每个方向高频单独融合→生成对应权重→加权对应高频 + 残差连接
    核心流程：
    1. 小波分解 → LL + LH(垂直)/HL(水平)/HH(对角)
    2. LL+LH → 多尺度卷积 → attn_LH → LH*attn_LH
    3. LL+HL → 多尺度卷积 → attn_HL → HL*attn_HL
    4. LL+HH → 多尺度卷积 → attn_HH → HH*attn_HH
    5. 小波重构（LL + 加权后LH/HL/HH）
    6. 残差连接：重构输出 + 原始输入
    """

    def __init__(self, in_channels, dilation=2, dtype=torch.float32):
        super(WaveletSpatialAttention_old, self).__init__()
        self.in_channels = in_channels
        self.dilation = dilation

        # 1. 加载小波分解/重构核（注册为缓冲区，不参与训练）
        self.register_buffer("LL_k", get_haar_wavelet_kernel(in_channels, dtype)[0])
        self.register_buffer("LH_k", get_haar_wavelet_kernel(in_channels, dtype)[1])
        self.register_buffer("HL_k", get_haar_wavelet_kernel(in_channels, dtype)[2])
        self.register_buffer("HH_k", get_haar_wavelet_kernel(in_channels, dtype)[3])

        self.register_buffer("LL_recon_k", get_haar_wavelet_recon_kernel(in_channels)[0])
        self.register_buffer("LH_recon_k", get_haar_wavelet_recon_kernel(in_channels)[1])
        self.register_buffer("HL_recon_k", get_haar_wavelet_recon_kernel(in_channels)[2])
        self.register_buffer("HH_recon_k", get_haar_wavelet_recon_kernel(in_channels)[3])

        # 2. 分方向多尺度卷积（每个方向独立的局部+空洞卷积，保证权重精准对应）
        # 垂直方向（LH）：LL+LH → 卷积生成attn_LH
        self.conv_LH_local = nn.Conv2d(in_channels, in_channels, 3, 1, 1, dilation=1, bias=False, groups=in_channels)
        self.conv_LH_atrous = nn.Conv2d(
            in_channels, in_channels, 3, 1, 2, dilation=dilation, bias=False, groups=in_channels
        )
        self.attn_conv_LH = nn.Conv2d(in_channels, 1, 1, 1, 0)

        # 水平方向（HL）：LL+HL → 卷积生成attn_HL
        self.conv_HL_local = nn.Conv2d(in_channels, in_channels, 3, 1, 1, dilation=1, bias=False, groups=in_channels)
        self.conv_HL_atrous = nn.Conv2d(
            in_channels, in_channels, 3, 1, 2, dilation=dilation, bias=False, groups=in_channels
        )
        self.attn_conv_HL = nn.Conv2d(in_channels, 1, 1, 1, 0)

        # 对角方向（HH）：LL+HH → 卷积生成attn_HH
        self.conv_HH_local = nn.Conv2d(in_channels, in_channels, 3, 1, 1, dilation=1, bias=False, groups=in_channels)
        self.conv_HH_atrous = nn.Conv2d(
            in_channels, in_channels, 3, 1, 2, dilation=dilation, bias=False, groups=in_channels
        )
        self.attn_conv_HH = Conv(in_channels, 0.5 * in_channels, 1, 1, 0)

        # 共享批归一化和激活
        self.bn = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.mish = nn.Mish()
        self.conv_out = nn.Conv2d(in_channels, in_channels, 3, 1, 1, dilation=1, bias=False, groups=1)

    def wavelet_decompose(self, x):
        """小波分解：输入特征 → LL(低频) + LH/HL/HH(分方向高频)"""
        B, C, H_ori, W_ori = x.shape

        # 尺寸补全（适配stride=2的小波分解）
        pad_h = (2 - H_ori % 2) % 2
        pad_w = (2 - W_ori % 2) % 2
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h + pad_w) > 0 else x

        # 分组卷积实现小波分解（保留分方向高频）
        LL = F.conv2d(x_pad, self.LL_k, groups=C, stride=2, padding=0)
        LH = F.conv2d(x_pad, self.LH_k, groups=C, stride=2, padding=0)  # 垂直高频
        HL = F.conv2d(x_pad, self.HL_k, groups=C, stride=2, padding=0)  # 水平高频
        HH = F.conv2d(x_pad, self.HH_k, groups=C, stride=2, padding=0)  # 对角高频

        return LL, LH, HL, HH, (H_ori, W_ori)

    def wavelet_reconstruct(self, LL, LH, HL, HH, ori_size):
        """小波重构：频域分量 → 空间特征"""
        B, C = LL.shape[0], LL.shape[1]
        H_ori, W_ori = ori_size

        # 转置卷积实现小波重构（上采样回原始尺寸）
        LL_recon = F.conv_transpose2d(LL, self.LL_recon_k, groups=C, stride=2, padding=0)
        LH_recon = F.conv_transpose2d(LH, self.LH_recon_k, groups=C, stride=2, padding=0)
        HL_recon = F.conv_transpose2d(HL, self.HL_recon_k, groups=C, stride=2, padding=0)
        HH_recon = F.conv_transpose2d(HH, self.HH_recon_k, groups=C, stride=2, padding=0)

        # 重构特征求和 + 裁剪回原始尺寸
        x_recon = LL_recon + LH_recon + HL_recon + HH_recon
        if x_recon.shape[2] > H_ori or x_recon.shape[3] > W_ori:
            x_recon = x_recon[:, :, :H_ori, :W_ori]
        return x_recon

    def forward(self, x):
        """
        严格按照你的逻辑执行 + 残差连接：
        1. 小波分解得到LL、LH、HL、HH
        2. 对每个方向：LL+高频 → 多尺度卷积 → sigmoid生成权重 → 加权对应高频
        3. 小波重构输出
        4. 残差连接：重构输出 + 原始输入
        """
        # ========== 关键改动1：保存原始输入（残差分支） ==========
        residual = x

        # 步骤1：小波分解（空间→频域，保留分方向高频）
        LL, LH, HL, HH, ori_size = self.wavelet_decompose(x)
        B, C, H, W = LL.shape

        # ===================== 垂直方向（LH）：LL+LH → 生成attn_LH → 加权LH =====================
        # LL与LH逐像素相加（你的核心要求）
        F_add_LH = LL + LH
        # 多尺度卷积（局部+空洞）提取垂直方向特征
        LH_local = self.conv_LH_local(F_add_LH)
        LH_atrous = self.conv_LH_atrous(F_add_LH)
        LH_feat = self.sigmoid(self.bn(LH_local + LH_atrous))
        # 生成垂直方向空间权重
        # attn_LH = self.sigmoid(self.attn_conv_LH(LH_feat))
        # # 加权垂直高频LH
        # LH_weighted = LH * attn_LH
        LH_weighted = HL * LH_feat

        # ===================== 水平方向（HL）：LL+HL → 生成attn_HL → 加权HL =====================
        # LL与HL逐像素相加
        F_add_HL = LL + HL
        # 多尺度卷积提取水平方向特征
        HL_local = self.conv_HL_local(F_add_HL)
        HL_atrous = self.conv_HL_atrous(F_add_HL)
        HL_feat = self.mish(self.bn(HL_local + HL_atrous))
        # # 生成水平方向空间权重
        # attn_HL = self.sigmoid(self.attn_conv_HL(HL_feat))
        # # 加权水平高频HL
        # HL_weighted = HL * attn_HL
        HL_weighted = HL * HL_feat

        # ===================== 对角方向（HH）：LL+HH → 生成attn_HH → 加权HH =====================
        # LL与HH逐像素相加
        F_add_HH = LL + HH
        # 多尺度卷积提取对角方向特征
        HH_local = self.conv_HH_local(F_add_HH)
        HH_atrous = self.conv_HH_atrous(F_add_HH)
        HH_feat = self.mish(self.bn(HH_local + HH_atrous))
        # 生成对角方向空间权重
        # attn_HH = self.sigmoid(self.attn_conv_HH(HH_feat))
        # # 加权对角高频HH
        # HH_weighted = HH * attn_HH
        HH_weighted = HH * HH_feat

        # 步骤2：小波重构（低频LL + 三个方向加权后的高频）
        out = self.wavelet_reconstruct(LL, LH_weighted, HL_weighted, HH_weighted, ori_size)

        # ========== 关键改动2：残差连接（重构输出 + 原始输入） ==========
        out = out + residual
        # out = self.conv_out(out)

        return out


# -------------------------- 原型注意力（带残差） --------------------------
class PrototypeAttention(nn.Module):
    """
    终极修复版PA模块（彻底解决维度不匹配+保留所有你的逻辑）：
    1. 核心修复：4维重塑+expand替代repeat，避免维度错位
    2. 保留：C≠M/3维BN/torch.max降维/Q_L2广播/3×3卷积/Sigmoid
    3. 全维度校验+数值稳定性兜底
    """

    def __init__(self, in_channels, num_prototypes=8):
        super().__init__()
        self.C = in_channels  # 输入通道数（C≠M）
        self.M = num_prototypes  # 原型数量M
        self.N = num_prototypes  # 查询数量N=M
        self.eps = 1e-8  # 防除零常数

        # ========== 1. PLU编码器（C→M维度，3×3卷积） ==========
        self.proto_prob = nn.Conv2d(self.C, self.M, kernel_size=3, padding=1)  # 保持H/W不变
        # 初始化强制float32
        self.proto_prob.weight.data = self.proto_prob.weight.data.float()
        if self.proto_prob.bias is not None:
            self.proto_prob.bias.data = self.proto_prob.bias.data.float()

        # ========== 2. 3维BN + Softmax + Sigmoid ==========
        self.bn_dim1 = nn.BatchNorm1d(1)  # 3维BN（特征维度=1）
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # ========== 3. 查询解码器（可学习查询Q: N×C） ==========
        self.query = nn.Parameter(torch.randn(self.N, self.C, dtype=torch.float32) * 0.001, requires_grad=True)
        self.q_initialized = False  # 标记Q是否已初始化（仅首次前向传播执行）

    def forward(self, x):
        # 保存残差+记录原始dtype（兼容混合精度）
        # residual = x
        input_dtype = x.dtype
        x_float32 = x.to(dtype=torch.float32)  # 内部强制float32计算
        B, C, H, W = x_float32.shape
        N_pixel = H * W

        # ===================== Step1: PLU编码器（C→M维度） =====================
        # 1.1 3×3卷积生成原型概率logits（B×C×H×W → B×M×H×W）
        proto_logits = self.proto_prob(x_float32)  # 核心：C→M维度融合
        proto_logits = torch.clamp(proto_logits, min=-5.0, max=5.0)  # 防溢出

        # 1.2 展平+3维BN（特征维度=1）
        proto_flat = proto_logits.contiguous().view(B, self.M, N_pixel)  # B×M×N_pixel
        # 3维BN适配：(B×N_pixel)×1×M → BN → 还原为B×M×N_pixel
        proto_flat_bn = proto_flat.permute(0, 2, 1).reshape(B * N_pixel, 1, self.M)
        proto_flat_bn = self.bn_dim1(proto_flat_bn)
        proto_flat_bn = proto_flat_bn.reshape(B, N_pixel, self.M).permute(0, 2, 1)  # 还原维度

        # 1.3 稳定版Softmax（dim=1: 原型维度）
        P_softmax = self.softmax(proto_flat_bn)
        P_softmax = torch.nan_to_num(P_softmax, nan=self.eps, posinf=1.0, neginf=0.0)

        # 1.4 输入特征展平 + 生成原型特征V（B×M×C）
        F_flat = x_float32.contiguous().view(B, C, N_pixel)  # B×C×N_pixel
        V = torch.matmul(F_flat, P_softmax.transpose(1, 2))  # B×C×M
        V = V.transpose(1, 2)  # 转置为 B×M×C
        V = self.tanh(V)

        if not self.q_initialized and self.training:  # 仅训练模式下初始化
            with torch.no_grad():  # 初始化阶段不计算梯度
                # 策略1：取batch内原型特征V的均值作为Q的初始值（推荐，稳定）
                # V_mean: M×C（对batch维度求平均）
                V_mean = V.mean(dim=0)  # B×M×C → M×C（N=M，维度匹配）
                # 替换Q的初始值
                self.query.data.copy_(V_mean)
                # 标记初始化完成，后续不再执行
                self.q_initialized = True

        # ===================== Step2: 查询解码器（torch.max降维 + Q_L2广播） =====================

        # 2.1 归一化：查询Q + 原型特征V（只算一次）
        Q = self.query.float()  # N×C (N=M)
        Q_norm = F.normalize(Q, dim=-1, eps=self.eps)  # N×C（归一化后范数=1）
        V_norm = F.normalize(V, dim=-1, eps=self.eps)  # B×M×C（归一化后范数=1）

        # 2.2 余弦相似度计算（归一化后点积=余弦，无冗余）
        similarity = torch.matmul(V_norm, Q_norm.transpose(0, 1))  # B×M×N → 直接是余弦相似度

        # 2.3 torch.max降维：M×N → M×1（核心逻辑保留）        # 3.2 横向Softmax (dim=1: 原型维度) → A_proto: B×C×M
        S_max, _ = torch.max(similarity, dim=-1, keepdim=True)  # B×M×1 → 范围[-1,1]

        # ✅ 删掉重复的V_L2/Q_L2计算 + 多余的除法
        S = torch.clamp(S_max, min=-1.0, max=1.0)  # 直接用max后的余弦相似度
        S = torch.nan_to_num(S, nan=0.0, posinf=1.0, neginf=-1.0)

        # ===================== Step3: 原型转置+横向Softmax =====================
        # 3.1 原型特征转置 → V_T: B×C×M
        V_T = V.transpose(1, 2)  # B×C×M

        # max_vt = torch.max(V_T, dim=1, keepdim=True)[0]
        # V_T_stable = V_T - max_vt
        # exp_vt = torch.exp(V_T_stable)
        # sum_exp_vt = torch.sum(exp_vt, dim=1, keepdim=True) + self.eps
        # A_proto = exp_vt / sum_exp_vt  # B×C×M
        # A_proto = torch.nan_to_num(A_proto, nan=self.eps, posinf=1.0, neginf=0.0)
        A_proto = self.tanh(V_T)

        # 3.3 矩阵相乘: A_proto (B×C×M) × S (B×M×1) → B×C×1
        channel_att_raw = torch.matmul(A_proto, S)  # B×C×1

        # 3.4 Softmax得到最终通道注意力向量α: B×C×1
        max_ca = torch.max(channel_att_raw, dim=1, keepdim=True)[0]
        ca_stable = channel_att_raw - max_ca
        exp_ca = torch.exp(ca_stable)
        sum_exp_ca = torch.sum(exp_ca, dim=1, keepdim=True) + self.eps
        alpha = exp_ca / sum_exp_ca  # B×C×1
        alpha = torch.nan_to_num(alpha, nan=self.eps, posinf=1.0, neginf=0.0)

        # ===================== Step4: 核心修复：4维重塑+expand广播 =====================
        # 错误根源：3维alpha直接repeat会导致维度错位，先转为4维再广播
        alpha_4d = alpha.view(B, C, 1, 1)  # B×C×1×1（4维重塑）
        alpha_broadcast = alpha_4d.expand(B, C, H, W)  # 广播到B×C×H×W（无数据复制，效率高）

        # ===================== Step5: 注意力加权 + 残差连接 =====================
        # 注意力加权特征（维度完全匹配：B×C×H×W × B×C×H×W）
        x_att = x_float32 * alpha_broadcast

        # 转回原始dtype + 残差连接
        out = x_att.to(dtype=input_dtype)

        return out


# ===================== 集成到C2fWP模块 =====================
class C2fW(nn.Module):
    """适配YOLO的C2fWP模块（集成最终精准版小波注意力）"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(c1, self.c, 1, 1)
        # self.gc1 = GSConv(c_, c_, 1, 1)
        # self.gc2 = GSConv(c_, c_, 1, 1)
        self.gsb = GSBottleneck(self.c, self.c, 1, 1)
        # self.res = Conv(self.c, self.c, 3, 1, act=False)
        self.cv3 = Conv(2 * self.c, c2, 1)  #
        self.Wast = WaveletSpatialAttention(self.c * 2)
        # self.m = nn.ModuleList(
        #     nn.Sequential(
        #         WaveletSpatialAttention(self.c),  # 输入输出都是self.c维
        #         # Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0),  # 官方Bottleneck
        #         self.gsb,
        #     ) for _ in range(n)
        # )
        # self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """完全复用官方C2f的前向逻辑"""
        return self.Wast(x)
        # x1 = self.gsb(self.Wast(self.cv1(x)))
        # y = self.cv2(x)
        # # y.extend(m(y[-1]) for m in self.m)
        # return self.cv3(torch.cat((y, x1),dim=1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """可选：拆分式前向（用于部署/调试）"""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2fP(nn.Module):
    """适配YOLO的C2fWP模块（集成最终精准版小波注意力）"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(c1, self.c, 1, 1)
        self.gsb = GSBottleneck(self.c, self.c, 1, 1)
        self.shortcut = shortcut
        self.PA = PrototypeAttention(self.c * 2)
        self.Wast = WaveletSpatialAttention(self.c)
        self.cv3 = Conv(2 * self.c, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """完全复用官方C2f的前向逻辑"""
        return self.PA(x)
        # x1 = self.gsb(self.cv1(x))
        # y = self.cv2(x)
        # return self.PA(self.cv3(torch.cat((y,x1), dim=1)))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """可选：拆分式前向（用于部署/调试）"""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class ChannelSplitAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # 确保通道数是偶数（避免拆分不均，若为奇数可补1或向下取整）
        assert channels % 2 == 0, f"通道数{channels}必须为偶数，否则无法拆分50%/50%"

        # 初始化两种注意力（各处理一半通道）
        self.wavelet_att = WaveletSpatialAttention(channels // 2)
        self.prototype_att = PrototypeAttention(channels // 2)

    def forward(self, x):
        # 步骤1：拆分输入通道为前50%和后50%
        # x.shape: [B, C, H, W] → x1: [B, C//2, H, W], x2: [B, C//2, H, W]
        x1, x2 = torch.chunk(x, 2, dim=1)

        # 步骤2：分别执行注意力
        x1 = self.wavelet_att(x1)  # 小波注意力处理前50%通道
        x2 = self.prototype_att(x2)  # 原型注意力处理后50%通道

        # 步骤3：拼接通道（恢复原通道数）
        out = torch.cat([x1, x2], dim=1)  # [B, C//2 + C//2, H, W] = [B, C, H, W]

        return out


class C2fWP(nn.Module):
    """适配YOLO的C2fWP模块（集成最终精准版小波注意力）"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(c1, self.c, 1, 1)
        self.gsb = GSBottleneck(self.c, self.c, 1, 1)
        self.shortcut = shortcut
        self.PA = PrototypeAttention(self.c * 2)
        self.Wast = WaveletSpatialAttention(self.c)
        self.cv3 = Conv(2 * self.c, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """完全复用官方C2f的前向逻辑"""
        x1 = self.gsb(self.Wast(self.cv1(x)))
        # x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.PA(self.cv3(torch.cat((y, x1), dim=1)))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """可选：拆分式前向（用于部署/调试）"""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class GSConv(nn.Module):
    """
    GSConv enhancement for representation learning: generate various receptive-fields and
    texture-features only in one Conv module
    https://github.com/AlanLi1997/slim-neck-by-gsconv
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, d, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, d, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        y = torch.cat((x1, x2), dim=1)
        # shuffle
        y = y.reshape(y.shape[0], 2, y.shape[1] // 2, y.shape[2], y.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])


class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        # for lighting
        self.conv_lighting = nn.Sequential(GSConvE(c1, c_, 3, 1), GSConvE(c_, c2, 3, 1, act=False))
        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)


class GSBottleneckC(GSBottleneck):
    # cheap GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__(c1, c2, k, s)
        self.shortcut = DWConv(c1, c2, 3, 1, act=False)


class VoVGSCSP(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = DWConv(c1, c_, 1, 1)
        # self.gc1 = GSConv(c_, c_, 1, 1)
        # self.gc2 = GSConv(c_, c_, 1, 1)
        # self.gsb = GSBottleneck(c_, c_, 1, 1)
        # self.res = GSConvE(c_, c_, 3, 1, act=False)
        self.cv3 = GSConv(2 * c_, c2, 1)  #
        self.gsb_stack = nn.Module(GSBottleneckC(c_, c_, 1, 1) for _ in range(n))

    def forward(self, x):
        x1 = self.gsb_stack(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))


# class VoVGSCSPC(VoVGSCSP):
#     # cheap VoVGSCSP module with GSBottleneck
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__(c1, c2, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.gsb = GSBottleneckC(c_, c_, 3, 1)


class SNI(nn.Module):
    """
    https://github.com/AlanLi1997/rethinking-fpn
    soft nearest neighbor interpolation for up-sampling
    secondary features aligned
    """

    def __init__(self, c1=0, c2=0, up_f=2):
        super(SNI, self).__init__()
        self.us = nn.Upsample(None, up_f, "nearest")
        self.alpha = 1 / math.sqrt(up_f)
        # self.alpha = 1/math.sqrt(math.sqrt(up_f)

    def forward(self, x):
        return np.sqrt(self.alpha) * self.us(x)


class DSConv(nn.Module):
    def __init__(self, c1, c2, act=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU() if act else nn.Identity(),
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU() if act else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class GSConvE(nn.Module):
    """
    GSConv enhancement for representation learning: generate various receptive-fields and
    texture-features only in one Conv module
    https://github.com/AlanLi1997/slim-neck-by-gsconv
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, d, act)
        self.cv2 = DSConv(c_, c_, act=True)
        self.cv3 = nn.Sequential(
            nn.Conv2d(c2, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU() if act else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        y = torch.cat((x1, x2), dim=1)
        # shuffle
        # y = y.reshape(y.shape[0], 2, y.shape[1] // 2, y.shape[2], y.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])
        return self.cv3(y)


class GSConvE2(nn.Module):
    # enhancement lightweight conv

    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 4
        self.cv1 = Conv(c1, c_, k, s, k // 2, g, d, act)
        self.cv2 = Conv(c_, c_, 9, 1, k // 2, c_, d, act)
        self.cv3 = Conv(c_, c_, 13, 1, k // 2, c_, d, act)
        self.cv4 = Conv(c_, c_, 17, 1, k // 2, c_, d, act)

    def forward(self, x):
        y = torch.cat((self.cv1(x), self.cv2(self.cv1(x)), self.cv3(self.cv1(x)), self.cv4(self.cv1(x))), dim=1)
        # shuffle
        y = y.reshape(y.shape[0], 2, y.shape[1] // 2, y.shape[2], y.shape[3])
        output = y.permute(0, 2, 1, 3, 4)

        return output.reshape(output.shape[0], -1, output.shape[3], output.shape[4])


class ESD(nn.Module):
    """
    https://github.com/AlanLi1997/rethinking-fpn
    Extended spatial window for down-sampling
    lightweight fusion
    """

    def __init__(self, c1, c2, k=3, s=2, g=1, d=1, act=True):
        super().__init__()
        self.out_c = c2
        self.dense_feature = Conv(c1, c2, k, s, k // 2, g, d, act)  # window_dense_f
        self.global_feature = nn.AvgPool2d(4, 2, 1)  # window_global_f
        self.local_feature = nn.MaxPool2d(4, 2, 1)  # window_local_f

    def forward(self, x):
        if self.out_c == x.shape[1]:
            return self.global_feature(x) + self.local_feature(x) + self.dense_feature(x)
        else:
            return torch.cat((self.global_feature(x), self.local_feature(x)), dim=1) + self.dense_feature(x)


class ESD2(nn.Module):
    """
    https://github.com/AlanLi1997/rethinking-fpn
    Extended spatial window for down-sampling
    learnable linearly fusion
    """

    def __init__(self, c1, c2, k=3, s=2, g=1, d=1, act=True):
        super().__init__()
        self.dense_feature = Conv(c1, c2, k, s, None, g, d, act)  # window_dense_f
        self.global_feature = nn.AvgPool2d(4, 2, 1)  # window_global_f
        self.local_feature = nn.MaxPool2d(4, 2, 1)  # window_local_f
        self.fuse = nn.Conv2d(3 * c2, c2, 1, 1, bias=False)

    def forward(self, x):
        return self.fuse(torch.cat((self.global_feature(x), self.local_feature(x), self.dense_feature(x)), dim=1))


class VoVGSCSPW(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb_stack = nn.Sequential(*[GSBottleneckC(c_, c_, 1, 1) for _ in range(n)])
        self.Wast = WaveletSpatialAttention(in_channels=c_, out_channels=c_)
        self.cv3 = GSConvE(2 * c_, c2, 1)

    def forward(self, x):
        x1 = self.gsb_stack(self.cv1(x))  # branch 1                  # branch 2
        y = self.Wast(self.cv2(x))  # WTSA branch
        return self.cv3(torch.cat((x1, y), dim=1))


class PSPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of pooling iterations.
            shortcut (bool): Whether to use shortcut connection.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2
        self.PGCA = PrototypeAttention(c1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(self.PGCA(x))]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y


# ================= Haar Wavelet Kernel =================
def get_haar_wavelet_kernel(in_channels):
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] *= -1

    LL = np.transpose(harr_wav_L) * harr_wav_L
    LH = np.transpose(harr_wav_L) * harr_wav_H
    HL = np.transpose(harr_wav_H) * harr_wav_L
    HH = np.transpose(harr_wav_H) * harr_wav_H

    LL = torch.tensor(LL).float().unsqueeze(0).unsqueeze(0)
    LH = torch.tensor(LH).float().unsqueeze(0).unsqueeze(0)
    HL = torch.tensor(HL).float().unsqueeze(0).unsqueeze(0)
    HH = torch.tensor(HH).float().unsqueeze(0).unsqueeze(0)

    LL = LL.repeat(in_channels, 1, 1, 1)
    LH = LH.repeat(in_channels, 1, 1, 1)
    HL = HL.repeat(in_channels, 1, 1, 1)
    HH = HH.repeat(in_channels, 1, 1, 1)

    return LL, LH, HL, HH


# ================= Depthwise Separable Conv =================
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depth = nn.Conv2d(
            in_channels,
            in_channels,
            3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.point = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Mish()

    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        x = self.bn(x)
        return self.act(x)


# ================= Window MHSA =================
class WindowMHSA(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8):
        super().__init__()

        self.dim = dim
        self.window = window_size

        self.num_heads = max(1, min(num_heads, dim))
        self.head_dim = max(1, dim // self.num_heads)
        self.inner_dim = self.head_dim * self.num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = DepthwiseSeparableConv(dim, self.inner_dim * 3)
        self.proj = nn.Conv2d(self.inner_dim, dim, 1)

    def _get_window_size(self, H, W):
        ws = min(self.window, H, W)
        ws = max(ws, 1)
        return ws

    def _safe_pad(self, x, pad_w, pad_h):
        if pad_h == 0 and pad_w == 0:
            return x

        H, W = x.shape[-2], x.shape[-1]

        # reflect 要求 pad < input_size
        if pad_h < H and pad_w < W:
            return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            # 小尺寸特征图时改用 replicate，更稳定
            return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    def window_partition(self, x):
        B, C, H, W = x.shape
        ws = self._get_window_size(H, W)

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        x = self._safe_pad(x, pad_w, pad_h)

        Hp = H + pad_h
        Wp = W + pad_w

        x = x.view(B, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        windows = x.view(-1, C, ws, ws)

        return windows, H, W, Hp, Wp, ws, pad_h, pad_w

    def window_reverse(self, windows, H, W, Hp, Wp, ws, pad_h, pad_w):
        B = int(windows.shape[0] // ((Hp // ws) * (Wp // ws)))

        x = windows.view(B, Hp // ws, Wp // ws, self.dim, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, self.dim, Hp, Wp)

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        return x

    def forward(self, x):
        B, C, H, W = x.shape

        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        q_win, H, W, Hp, Wp, ws, pad_h, pad_w = self.window_partition(q)
        k_win, _, _, _, _, _, _, _ = self.window_partition(k)
        v_win, _, _, _, _, _, _, _ = self.window_partition(v)

        area = ws * ws

        q = q_win.view(-1, self.num_heads, self.head_dim, area).transpose(2, 3)
        k = k_win.view(-1, self.num_heads, self.head_dim, area).transpose(2, 3)
        v = v_win.view(-1, self.num_heads, self.head_dim, area).transpose(2, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.transpose(2, 3).contiguous()
        out = out.view(-1, self.inner_dim, ws, ws)

        x = self.window_reverse(out, H, W, Hp, Wp, ws, pad_h, pad_w)
        x = self.proj(x)

        return x


class ECAAttention(nn.Module):
    """
    Efficient Channel Attention
    Reference idea:
        GAP -> 1D Conv -> Sigmoid -> Channel Reweight
    """

    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv1d(y)  # [B, 1, C]
        y = self.act(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y.expand_as(x)


# ================= Conv-BN-Act =================
class BasicConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Mish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ================= Wavelet + ECA (Lightweight) =================
class WaveletSpatialAttention(nn.Module):
    """
    改进点：
    1. 轻量化：去掉 WindowMHSA，改用 ECA
    2. 残差式增强
    3. 高频使用 cat(LH, HL, HH) 后 1x1 压缩，保留方向信息
    4. 适合接在降维后的分支上
    """

    def __init__(self, in_channels, out_channels=None, eca_kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels

        LL, LH, HL, HH = get_haar_wavelet_kernel(in_channels)
        self.register_buffer("LL", LL)
        self.register_buffer("LH", LH)
        self.register_buffer("HL", HL)
        self.register_buffer("HH", HH)

        # 高频 cat 后压缩回 in_channels
        self.high_reduce = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False), nn.BatchNorm2d(in_channels), nn.Mish()
        )

        # 低频与高频分别做轻量通道注意力
        self.eca_ll = ECAAttention(in_channels, k_size=eca_kernel_size)
        self.eca_high = ECAAttention(in_channels, k_size=eca_kernel_size)

        # 融合
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.Mish(),
        )

        # 残差支路
        if self.out_channels == in_channels:
            self.short = nn.Identity()
        else:
            self.short = nn.Sequential(
                nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(self.out_channels)
            )

    def dwt(self, x):
        """
        Haar DWT with stride=2
        """
        B, C, H, W = x.shape

        pad_h = H % 2
        pad_w = W % 2

        if pad_h or pad_w:
            # 小尺寸时 reflect 仍可能稳定；若你训练时报错可改为 replicate
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        LL = F.conv2d(x, self.LL, groups=C, stride=2)
        LH = F.conv2d(x, self.LH, groups=C, stride=2)
        HL = F.conv2d(x, self.HL, groups=C, stride=2)
        HH = F.conv2d(x, self.HH, groups=C, stride=2)

        return LL, LH, HL, HH, H, W

    def forward(self, x):
        shortcut = self.short(x)

        LL, LH, HL, HH, H, W = self.dwt(x)

        # 高频保留方向信息，不直接相加
        high = torch.cat([LH, HL, HH], dim=1)
        high = self.high_reduce(high)

        # 轻量注意力
        LL = self.eca_ll(LL)
        high = self.eca_high(high)

        # 融合低频和高频
        y = torch.cat([LL, high], dim=1)
        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
        y = self.fusion(y)

        # 残差输出
        return shortcut + y


class MSCM(nn.Module):
    """
    Mixed Spatial-Channel Modulation
    输入: 融合后的频域特征
    输出: 综合 gate (B, C, H, W)
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 16)

        # 通道注意力分支
        self.channel_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )

        # 局部注意力分支
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ca = self.channel_branch(x)  # [B,C,1,1]
        la = self.local_branch(x)  # [B,C,H,W]
        gate = self.sigmoid(ca + la)  # broadcast
        return gate


# ================= WCA Head Input =================
class WCALite(nn.Module):
    """
    更贴近论文思路的 WCA:
    1) DWT 分成低频 LL 和 高频(H)
    2) 高低频融合
    3) 经过 MSCM 得到 channel+local 综合门控
    4) gate 与高频逐像素相乘
    5) 与低频相加，保留低频内容
    6) 上采样回原尺寸，作为 head 输入
    """

    def __init__(self, channels, reduction=4, preserve_residual=True):
        super().__init__()
        self.channels = channels
        self.preserve_residual = preserve_residual

        LL, LH, HL, HH = get_haar_wavelet_kernel(channels)
        self.register_buffer("LL", LL)
        self.register_buffer("LH", LH)
        self.register_buffer("HL", HL)
        self.register_buffer("HH", HH)

        # 高频三方向拼接后压缩
        self.high_proj = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.SiLU()
        )

        # 低频投影
        self.low_proj = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.SiLU())

        # 高低频融合
        self.fuse = nn.Sequential(nn.Conv2d(channels * 2, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.SiLU())

        # MSCM 生成综合 gate
        self.mscm = MSCM(channels, reduction=reduction)

        # 输出投影
        self.out_proj = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.SiLU())

    def dwt(self, x):
        B, C, H, W = x.shape
        pad_h = H % 2
        pad_w = W % 2

        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        LL = F.conv2d(x, self.LL, stride=2, groups=C)
        LH = F.conv2d(x, self.LH, stride=2, groups=C)
        HL = F.conv2d(x, self.HL, stride=2, groups=C)
        HH = F.conv2d(x, self.HH, stride=2, groups=C)

        return LL, LH, HL, HH, H, W

    def forward(self, x):
        shortcut = x

        LL, LH, HL, HH, H, W = self.dwt(x)

        # 低频
        low = self.low_proj(LL)

        # 高频：保留方向信息
        high = torch.cat([LH, HL, HH], dim=1)
        high = self.high_proj(high)

        # 高低频融合，用于生成 gate
        mix = torch.cat([low, high], dim=1)
        mix = self.fuse(mix)

        # 综合门控
        gate = self.mscm(mix)

        # 高频筛选
        high_refined = high * gate

        # 保留低频内容
        out = low + high_refined

        # 恢复原尺寸
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        out = self.out_proj(out)

        # 是否加原始残差
        if self.preserve_residual:
            out = out + shortcut

        return out


class PGCR(nn.Module):
    """
    Prototype-Guided Classification Refinement
    适用于 YOLOv8 Detect 中 cls 分支前的特征增强

    输入:
        x: [B, C, H, W]
    输出:
        out: [B, C, H, W]

    核心流程:
        1) prototype assignment
        2) prototype aggregation
        3) prototype write-back
        4) 与原特征融合
        5) 通道+局部门控
        6) residual refinement
    """

    def __init__(self, channels, num_prototypes=8, reduction=4, use_norm=True):
        super().__init__()
        self.channels = channels
        self.num_prototypes = num_prototypes
        self.use_norm = use_norm

        hidden = max(channels // reduction, 16)

        # -------- Prototype Learning Unit (PLU) --------
        # 生成 prototype assignment map: [B, M, H, W]
        self.proto_assign = nn.Conv2d(channels, num_prototypes, kernel_size=1, bias=True)

        # prototype 回写后的投影
        self.proto_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False), nn.BatchNorm2d(channels), nn.SiLU()
        )

        # 融合原始特征和 prototype 回写特征
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False), nn.BatchNorm2d(channels), nn.SiLU()
        )

        # -------- MSCM-like gating --------
        # 全局通道分支: GAP -> 1x1 -> 1x1
        self.channel_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

        # 局部分支: local modeling
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

        self.sigmoid = nn.Sigmoid()

        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False), nn.BatchNorm2d(channels), nn.SiLU()
        )

        # 可选归一化
        if use_norm:
            self.norm = nn.BatchNorm2d(channels)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        HW = H * W

        # --------------------------------------------------
        # 1) 生成 prototype assignment
        # --------------------------------------------------
        # assign_map: [B, M, H, W]
        assign_map = self.proto_assign(x)

        # reshape 到 [B, HW, M]
        assign_map = assign_map.view(B, self.num_prototypes, HW).permute(0, 2, 1).contiguous()

        # 对 prototype 维度做 softmax
        # 每个空间位置属于不同 prototype 的权重
        assign_map = F.softmax(assign_map, dim=-1)  # [B, HW, M]

        # --------------------------------------------------
        # 2) 聚合 prototype
        # --------------------------------------------------
        # x_flat: [B, HW, C]
        x_flat = x.view(B, C, HW).permute(0, 2, 1).contiguous()

        # prototypes: [B, M, C]
        # 相当于 A^T @ X
        prototypes = torch.bmm(assign_map.transpose(1, 2), x_flat)

        # 做一个归一化，避免 prototype 幅值随 HW 变化太大
        denom = assign_map.transpose(1, 2).sum(dim=-1, keepdim=True) + 1e-6
        prototypes = prototypes / denom

        # --------------------------------------------------
        # 3) prototype 回写到空间特征
        # --------------------------------------------------
        # recon: [B, HW, C] = A @ P
        recon = torch.bmm(assign_map, prototypes)

        # reshape 回 [B, C, H, W]
        recon = recon.permute(0, 2, 1).contiguous().view(B, C, H, W)
        recon = self.proto_proj(recon)

        # --------------------------------------------------
        # 4) 与原始特征融合
        # --------------------------------------------------
        z = x + recon
        z = self.fuse(z)

        # --------------------------------------------------
        # 5) MSCM-like gate
        # --------------------------------------------------
        wg = self.channel_branch(z)  # [B, C, 1, 1]
        wl = self.local_branch(z)  # [B, C, H, W]
        gate = self.sigmoid(wg + wl)  # broadcast

        # --------------------------------------------------
        # 6) residual refinement
        # --------------------------------------------------
        out = gate * x + x
        out = self.out_proj(out)
        out = self.norm(out)

        return out


class NeckNoiseGate(nn.Module):
    def __init__(self, in_channels, reduction=4, sigma=0.05, learnable_sigma=False, preserve_residual=True):
        super().__init__()

        self.channels = in_channels
        self.hidden = max(self.channels // reduction, 16)

        self.preserve_residual = preserve_residual

        if learnable_sigma:
            self.sigma = nn.Parameter(torch.tensor(float(sigma)))
        else:
            self.register_buffer("sigma", torch.tensor(float(sigma)), persistent=False)

        # ✅ 通道门控（正确）
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels, self.hidden, kernel_size=1, bias=False),  # ✔ 输入必须是 self.channels
            nn.SiLU(),
            nn.Conv2d(self.hidden, self.channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # ✅ 空间门控（重点修复）
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(self.channels, self.hidden, kernel_size=3, padding=1, bias=False),  # ✔ 这里必须是 self.channels
            nn.BatchNorm2d(self.hidden),
            nn.SiLU(),
            nn.Conv2d(self.hidden, 1, 1, bias=True),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 1, bias=False), nn.BatchNorm2d(self.channels), nn.SiLU()
        )

    def forward(self, x):
        identity = x

        # gate in [0,1]

        cg = self.channel_gate(x)  # [B, C, 1, 1]
        sg = self.spatial_gate(x)  # [B, 1, H, W]
        gate = cg * sg  # [B, C, H, W] (broadcast)

        if self.training:
            sigma = torch.clamp(self.sigma, min=0.0)
            noise = torch.randn_like(x) * sigma

            # gate越大，噪声越小
            x = x + noise * (1.0 - gate)

        x = self.out_proj(x)

        if self.preserve_residual:
            x = x + identity
        # print("我在这里NOISE")
        return x


# class PGCR_EMA_Cosine_MSCM(nn.Module):
#     def __init__(
#         self,
#         channels,
#         num_prototypes=10,
#         reduction=4,
#         momentum_start=0.80,
#         momentum_end=0.95,
#         temperature=0.1,
#         eps=1e-6,
#         update_in_train=True,
#         use_norm=True,
#         proto_noise_std=0.05,          # 新增：prototype噪声强度
#         noise_on_eval=False,           # 新增：推理时是否也加噪
#         renorm_after_noise=True,       # 新增：加噪后是否重新归一化
#     ):
#         super().__init__()
#         self.channels = channels
#         self.num_prototypes = num_prototypes
#         self.momentum = momentum_start
#         self.momentum_end = momentum_end
#         self.total_iters = 7000
#         self.momentum_step = (momentum_end - momentum_start) / self.total_iters
#         self.temperature = temperature
#         self.eps = eps
#         self.update_in_train = update_in_train
#
#         self.proto_noise_std = proto_noise_std
#         self.noise_on_eval = noise_on_eval
#         self.renorm_after_noise = renorm_after_noise
#
#         hidden = max(channels // reduction, 16)
#
#         self.query_proj = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.SiLU()
#         )
#
#         self.recon_proj = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.SiLU()
#         )
#
#         self.channel_branch = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
#             nn.SiLU(),
#             nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
#         )
#
#         self.local_branch = nn.Sequential(
#             nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(),
#             nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
#         )
#
#         self.sigmoid = nn.Sigmoid()
#
#         self.out_proj = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.SiLU()
#         )
#
#         self.norm = nn.BatchNorm2d(channels) if use_norm else nn.Identity()
#
#         init_proto = torch.randn(num_prototypes, channels)
#         init_proto = F.normalize(init_proto, dim=-1)
#         self.register_buffer("prototypes", init_proto)
#         self.register_buffer("proto_counter", torch.zeros(num_prototypes))
#
#     @torch.no_grad()
#     def reset_prototypes(self):
#         proto = torch.randn(self.num_prototypes, self.channels, device=self.prototypes.device)
#         proto = F.normalize(proto, dim=-1)
#         self.prototypes.copy_(proto)
#         self.proto_counter.zero_()
#
#     def _get_noisy_prototypes(self, B):
#         """
#         返回用于当前forward的 prototype 副本
#         shape: [B, M, C]
#         """
#         P = self.prototypes.detach().clone().unsqueeze(0).expand(B, -1, -1)  # [B, M, C]
#
#         use_noise = self.training or self.noise_on_eval
#         if use_noise and self.proto_noise_std > 0:
#             noise = torch.randn_like(P) * self.proto_noise_std
#             P = P + noise
#
#             if self.renorm_after_noise:
#                 P = F.normalize(P, dim=-1, eps=self.eps)
#
#         return P
#
#     def _cosine_assign(self, q_flat):
#         """
#         q_flat: [B, HW, C]
#         return:
#             A: [B, HW, M]
#             P: [B, M, C]
#         """
#         B, HW, C = q_flat.shape
#
#         # 取带噪声的prototype副本
#         P = self._get_noisy_prototypes(B)  # [B, M, C]
#
#         q_norm = F.normalize(q_flat, dim=-1, eps=self.eps)
#         p_norm = F.normalize(P, dim=-1, eps=self.eps)
#
#         sim = torch.bmm(q_norm, p_norm.transpose(1, 2))  # [B, HW, M]
#         sim = sim / self.temperature
#         A = F.softmax(sim, dim=-1)
#
#         return A, P
#
#     @torch.no_grad()
#     def _ema_update_prototypes(self, q_flat, A):
#         """
#         q_flat: [B, HW, C]
#         A: [B, HW, M]
#         """
#         q_norm = F.normalize(q_flat, dim=-1, eps=self.eps)  # [B, HW, C]
#
#         # [B, M, HW] x [B, HW, C] -> [B, M, C]
#         proto_batch = torch.bmm(A.transpose(1, 2), q_norm)
#
#         denom = A.transpose(1, 2).sum(dim=-1, keepdim=True) + self.eps
#         proto_batch = proto_batch / denom
#
#         proto_now = proto_batch.mean(dim=0)  # [M, C]
#         proto_now = F.normalize(proto_now, dim=-1, eps=self.eps)
#
#         updated = self.momentum * self.prototypes + (1.0 - self.momentum) * proto_now
#         updated = F.normalize(updated, dim=-1, eps=self.eps)
#         self.prototypes.copy_(updated)
#
#         usage = A.sum(dim=(0, 1))
#         self.proto_counter.add_(usage)
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         HW = H * W
#
#         self.update_momentum()
#
#         # 1) query feature
#         q = self.query_proj(x)
#         q_flat = q.view(B, C, HW).permute(0, 2, 1).contiguous()  # [B, HW, C]
#
#         # 2) cosine assignment with noisy prototype prior
#         A, P = self._cosine_assign(q_flat)
#
#         # 3) reconstruct channel vectors
#         q_rec_flat = torch.bmm(A, P)  # [B, HW, C]
#         q_rec = q_rec_flat.permute(0, 2, 1).contiguous().view(B, C, H, W)
#         q_rec = self.recon_proj(q_rec)
#
#         # 4) MSCM gate on original x
#         wg = self.channel_branch(x)   # [B, C, 1, 1]
#         wl = self.local_branch(x)     # [B, C, H, W]
#         gate = self.sigmoid(wg + wl)
#
#         # 5) residual fusion
#         out = x + gate * q_rec
#         out = self.out_proj(out)
#         out = self.norm(out)
#
#         # 6) EMA update
#         if self.training and self.update_in_train:
#             self._ema_update_prototypes(q_flat.detach(), A.detach())
#
#         return out
#
#     def update_momentum(self):
#         if not self.training:
#             return
#
#         if self.momentum < self.momentum_end:
#             self.momentum += self.momentum_step
#             if self.momentum > self.momentum_end:
#                 self.momentum = self.momentum_end

# class PGCR_EMA_Cosine_MSCM(nn.Module):
#     def __init__(self, channels,
#                  num_prototypes=10,
#                  reduction=4,
#                  momentum_start=0.65,
#                  momentum_end=0.90,
#                  temperature=0.1,
#                  eps=1e-6,
#                  update_in_train=True,
#                  use_norm=True,
#                  proto_noise_std=0.05,
#                  noise_on_eval=False,
#                  renorm_after_noise=True):
#         super().__init__()
#         self.channels = channels
#         self.num_prototypes = num_prototypes
#         self.momentum = momentum_start
#         self.momentum_end = momentum_end
#         self.total_iters = 7000
#         self.momentum_step = (momentum_end - momentum_start) / self.total_iters
#         self.temperature = temperature
#         self.eps = eps
#         self.update_in_train = update_in_train
#         self.proto_noise_std = proto_noise_std
#         self.noise_on_eval = noise_on_eval
#         self.renorm_after_noise = renorm_after_noise
#
#         hidden = max(channels // reduction, 16)
#
#         self.query_proj = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.SiLU()
#         )
#
#         self.recon_proj = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.SiLU()
#         )
#
#         self.channel_branch = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
#             nn.SiLU(),
#             nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
#         )
#
#         self.local_branch = nn.Sequential(
#             nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(),
#             nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
#         )
#
#         self.sigmoid = nn.Sigmoid()
#
#         self.out_proj = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.SiLU()
#         )
#
#         self.norm = nn.BatchNorm2d(channels) if use_norm else nn.Identity()
#
#         # **初始化原型**：从标准正态分布初始化原型
#         init_mean = torch.randn(num_prototypes, channels) * 0.1  # 初始均值
#         init_var = torch.ones(num_prototypes, channels) * 0.1  # 初始方差
#         self.register_buffer("prototypes_mean", init_mean)  # 原型的均值
#         self.register_buffer("prototypes_var", init_var)    # 原型的方差
#         self.register_buffer("proto_counter", torch.zeros(num_prototypes))
#
#     @torch.no_grad()
#     def reset_prototypes(self):
#         """
#         重新初始化原型，并使用标准正态分布进行初始化
#         """
#         proto_mean = torch.randn(self.num_prototypes, self.channels, device=self.prototypes_mean.device) * 0.1  # 均值
#         proto_var = torch.ones(self.num_prototypes, self.channels, device=self.prototypes_var.device) * 0.1  # 方差
#         self.prototypes_mean.copy_(proto_mean)
#         self.prototypes_var.copy_(proto_var)
#         self.proto_counter.zero_()
#
#     def _get_prototypes(self, B):
#         """
#         返回当前用于计算的原型副本
#         """
#         P_mean = self.prototypes_mean.detach().clone().unsqueeze(0).expand(B, -1, -1)  # [B, M, C]
#         P_var = self.prototypes_var.detach().clone().unsqueeze(0).expand(B, -1, -1)  # [B, M, C]
#
#         return P_mean, P_var
#
#     def _cosine_assign(self, q_flat):
#         """
#         q_flat: [B, HW, C]
#         return:
#             A: [B, HW, M]
#             P_mean: [B, M, C]  (原型均值)
#             P_var: [B, M, C]   (原型方差)
#         """
#         B, HW, C = q_flat.shape
#
#         # 获取当前的原型副本
#         P_mean, P_var = self._get_prototypes(B)  # [B, M, C] 均值和方差
#         P_samples = torch.randn_like(P_mean) * P_var + P_mean
#
#         q_norm = F.normalize(q_flat, dim=-1, eps=self.eps)
#         p_norm = F.normalize(P_samples, dim=-1, eps=self.eps)
#
#         sim = torch.bmm(q_norm, p_norm.transpose(1, 2))  # [B, HW, M]
#         sim = sim / self.temperature
#         A = F.softmax(sim, dim=-1)
#
#         return A, P_mean, P_var
#
#     @torch.no_grad()
#     def _ema_update_prototypes(self, q_flat, A):
#         """
#         q_flat: [B, HW, C]
#         A: [B, HW, M]
#         """
#         q_norm = F.normalize(q_flat, dim=-1, eps=self.eps)  # [B, HW, C]
#
#         proto_batch = torch.bmm(A.transpose(1, 2), q_norm)
#
#         denom = A.transpose(1, 2).sum(dim=-1, keepdim=True) + self.eps
#         proto_batch = proto_batch / denom
#
#         # 计算新的均值和方差
#         mean_now = proto_batch.mean(dim=0)  # [M, C] 均值
#         var_now = proto_batch.var(dim=0)    # [M, C] 方差
#
#         # 使用EMA更新均值和方差
#         self.prototypes_mean = self.momentum * self.prototypes_mean + (1.0 - self.momentum) * mean_now
#         self.prototypes_var = self.momentum * self.prototypes_var + (1.0 - self.momentum) * var_now
#
#         # 归一化
#         self.prototypes_mean = F.normalize(self.prototypes_mean, dim=-1, eps=self.eps)
#         self.prototypes_var = F.normalize(self.prototypes_var, dim=-1, eps=self.eps)
#
#     def _kl_divergence(self):
#         """
#         计算原型与标准正态分布之间的 KL 散度损失
#         """
#         p_mean = self.prototypes_mean  # 原型的均值
#         p_var = self.prototypes_var    # 原型的方差
#
#         # 计算标准正态分布的均值和方差
#         mean_0 = torch.zeros_like(p_mean)
#         var_1 = torch.ones_like(p_var)
#
#         # 计算KL散度： 0.5 * (var + mean^2 - log(var) - 1)
#         kl_loss = 0.5 * torch.sum(p_mean**2 + p_var - torch.log(p_var + self.eps) - 1)
#
#         return kl_loss
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         HW = H * W
#
#         self.update_momentum()
#
#         # 1) query feature
#         q = self.query_proj(x)
#         q_flat = q.view(B, C, HW).permute(0, 2, 1).contiguous()  # [B, HW, C]
#
#         # 2) cosine assignment with noisy prototype prior
#         A, P_mean, P_var = self._cosine_assign(q_flat)
#
#         # 3) reconstruct channel vectors
#         # P_samples = torch.randn_like(P_mean) * P_var + P_mean  # 从高斯分布中重采样
#         q_rec_flat = torch.bmm(A, P_mean)  # [B, HW, C]
#         q_rec = q_rec_flat.permute(0, 2, 1).contiguous().view(B, C, H, W)
#         q_rec = self.recon_proj(q_rec)
#
#         # 4) MSCM gate on original x
#         wg = self.channel_branch(x)   # [B, C, 1, 1]
#         wl = self.local_branch(x)     # [B, C, H, W]
#         gate = self.sigmoid(wg + wl)
#
#         # 5) residual fusion
#         out = x + gate * q_rec
#         out = self.out_proj(out)
#         out = self.norm(out)
#
#         # 6) EMA update
#         if self.training and self.update_in_train:
#             self._ema_update_prototypes(q_flat.detach(), A.detach())
#
#         # 7) 计算并返回 KL 损失
#         kl_loss = self._kl_divergence()
#         KL_LOSS_CONTAINER.append(kl_loss)
#
#         return out
#
#     def update_momentum(self):
#         if not self.training:
#             return
#
#         if self.momentum < self.momentum_end:
#             self.momentum += self.momentum_step
#             if self.momentum > self.momentum_end:
#                 self.momentum = self.momentum_end


class PGCR_EMA_Cosine_MSCM_old(nn.Module):
    """
    改动说明：
    1. 保留原型重构逻辑：q -> cosine assign -> prototype reconstruction -> q_rec
    2. gate 不再来自 channel/local branch
    3. gate 改成：只对“位置编码序列”做 self-attention，得到纯空间注意力
    4. 最终输出：out = x + gate * q_rec

    注意：
    - 这里 gate 只依赖位置，不依赖图像内容
    - 所以同一层、同一分辨率下，所有图片的 gate 是一样的
    - 这是“固定空间先验”
    """

    def __init__(
        self,
        channels,
        num_prototypes=10,
        reduction=4,
        momentum_start=0.65,
        momentum_end=0.90,
        temperature=0.1,
        eps=1e-6,
        update_in_train=True,
        use_norm=True,
        proto_noise_std=0.05,
        noise_on_eval=False,
        renorm_after_noise=True,
        num_heads=8,
        use_gaussian_scaling=True,
        gate_use_softmax=False,
    ):
        super().__init__()

        assert channels % num_heads == 0, f"channels({channels}) 必须能被 num_heads({num_heads}) 整除"
        assert channels % 4 == 0, f"为了构造2D sin-cos位置编码，channels({channels}) 最好能被4整除"

        self.channels = channels
        self.num_prototypes = num_prototypes
        self.momentum = momentum_start
        self.momentum_end = momentum_end
        self.total_iters = 7000
        self.momentum_step = (momentum_end - momentum_start) / self.total_iters

        self.temperature = temperature
        self.eps = eps
        self.update_in_train = update_in_train
        self.proto_noise_std = proto_noise_std
        self.noise_on_eval = noise_on_eval
        self.renorm_after_noise = renorm_after_noise
        self.use_gaussian_scaling = use_gaussian_scaling
        self.gate_use_softmax = gate_use_softmax

        hidden = max(channels // reduction, 16)

        # --------------------------------
        # feature projection
        # --------------------------------
        self.query_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False), nn.BatchNorm2d(channels), nn.SiLU()
        )

        self.recon_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False), nn.BatchNorm2d(channels), nn.SiLU()
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False), nn.BatchNorm2d(channels), nn.SiLU()
        )

        self.norm = nn.BatchNorm2d(channels) if use_norm else nn.Identity()

        # --------------------------------
        # 位置编码 -> Self-Attention -> gate
        # 只看位置，不看内容
        # --------------------------------
        self.pos_attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.pos_ln = nn.LayerNorm(channels)
        self.gate_proj = nn.Linear(channels, 1)

        # 一个可学习缩放，控制位置编码强度
        self.pos_scale = nn.Parameter(torch.tensor(1.0))

        # --------------------------------
        # prototype buffers
        # --------------------------------
        init_mean = torch.randn(num_prototypes, channels) * 0.1
        init_var = torch.ones(num_prototypes, channels) * 0.1

        self.register_buffer("prototypes_mean", init_mean)  # [M, C]
        self.register_buffer("prototypes_var", init_var)  # [M, C]
        self.register_buffer("proto_counter", torch.zeros(num_prototypes))

    # =========================================================
    # prototype utils
    # =========================================================
    @torch.no_grad()
    def reset_prototypes(self):
        proto_mean = torch.randn(self.num_prototypes, self.channels, device=self.prototypes_mean.device) * 0.1
        proto_var = torch.ones(self.num_prototypes, self.channels, device=self.prototypes_var.device) * 0.1

        self.prototypes_mean.copy_(proto_mean)
        self.prototypes_var.copy_(proto_var)
        self.proto_counter.zero_()

    def _get_prototypes(self, B):
        P_mean = self.prototypes_mean.detach().clone().unsqueeze(0).expand(B, -1, -1)  # [B, M, C]
        P_var = self.prototypes_var.detach().clone().unsqueeze(0).expand(B, -1, -1)  # [B, M, C]
        return P_mean, P_var

    def _gaussian_scale(self, q_flat, P_mean, P_var):
        """
        根据 prototype 的高斯分布，对 cosine similarity 做缩放。
        这里不用完整pdf乘积，避免高维下严重下溢，
        改用“按通道均值后的负二次项”来构造稳定的概率型缩放。

        q_flat: [B, HW, C]
        P_mean: [B, M, C]
        P_var:  [B, M, C]

        return:
            scale: [B, HW, M], 数值在(0,1]
        """
        q_exp = q_flat.unsqueeze(2)  # [B, HW, 1, C]
        mu_exp = P_mean.unsqueeze(1)  # [B, 1, M, C]
        var_exp = P_var.unsqueeze(1).clamp_min(1e-4)  # [B, 1, M, C]

        mahal = ((q_exp - mu_exp) ** 2) / (var_exp + self.eps)  # [B, HW, M, C]
        mahal = mahal.mean(dim=-1)  # [B, HW, M]

        scale = torch.exp(-0.5 * mahal)  # [B, HW, M]
        return scale

    def _cosine_assign(self, q_flat):
        """
        q_flat: [B, HW, C]
        return:
            A:      [B, HW, M]
            P_mean: [B, M, C]
            P_var:  [B, M, C]
        """
        B, HW, C = q_flat.shape

        P_mean, P_var = self._get_prototypes(B)

        # 按“标准差”采样，不是直接乘方差
        std = torch.sqrt(P_var.clamp_min(1e-4))
        P_samples = torch.randn_like(P_mean) * std + P_mean  # [B, M, C]

        q_norm = F.normalize(q_flat, dim=-1, eps=self.eps)  # [B, HW, C]
        p_norm = F.normalize(P_samples, dim=-1, eps=self.eps)  # [B, M, C]

        sim = torch.bmm(q_norm, p_norm.transpose(1, 2))  # [B, HW, M]

        # 你想要的：用高斯概率型缩放 cosine similarity
        if self.use_gaussian_scaling:
            scale = self._gaussian_scale(q_flat, P_mean, P_var)  # [B, HW, M]
            sim = sim * scale

        sim = sim / self.temperature
        A = F.softmax(sim, dim=-1)

        return A, P_mean, P_var

    @torch.no_grad()
    def _ema_update_prototypes(self, q_flat, A):
        """
        q_flat: [B, HW, C]
        A:      [B, HW, M]
        """
        q_norm = F.normalize(q_flat, dim=-1, eps=self.eps)  # [B, HW, C]

        proto_batch = torch.bmm(A.transpose(1, 2), q_norm)  # [B, M, C]
        denom = A.transpose(1, 2).sum(dim=-1, keepdim=True) + self.eps
        proto_batch = proto_batch / denom

        mean_now = proto_batch.mean(dim=0)  # [M, C]
        var_now = proto_batch.var(dim=0, unbiased=False)  # [M, C]

        new_mean = self.momentum * self.prototypes_mean + (1.0 - self.momentum) * mean_now
        new_var = self.momentum * self.prototypes_var + (1.0 - self.momentum) * var_now

        # mean 可以归一化
        new_mean = F.normalize(new_mean, dim=-1, eps=self.eps)
        # var 只保持正，不做 normalize
        new_var = new_var.clamp_min(1e-4)

        self.prototypes_mean.copy_(new_mean)
        self.prototypes_var.copy_(new_var)

    def _kl_divergence(self):
        """
        KL( N(mu, var) || N(0,1) )
        对角高斯版本
        """
        p_mean = self.prototypes_mean
        p_var = self.prototypes_var.clamp_min(1e-4)

        kl_loss = 0.5 * torch.sum(p_mean**2 + p_var - torch.log(p_var + self.eps) - 1.0)
        return kl_loss

    # =========================================================
    # positional self-attention gate
    # =========================================================
    def _build_2d_sincos_pos_embed(self, H, W, C, device, dtype):
        """
        生成 [1, HW, C] 的 2D sin-cos 位置编码
        """
        assert C % 4 == 0, "channels 必须能被 4 整除"

        y = torch.arange(H, device=device, dtype=dtype)
        x = torch.arange(W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")  # [H, W]

        yy = yy.reshape(-1, 1)  # [HW, 1]
        xx = xx.reshape(-1, 1)  # [HW, 1]

        dim = C // 4
        omega = torch.arange(dim, device=device, dtype=dtype)
        omega = 1.0 / (10000 ** (omega / dim))  # [dim]

        out_y = yy * omega.unsqueeze(0)  # [HW, dim]
        out_x = xx * omega.unsqueeze(0)  # [HW, dim]

        pos = torch.cat([torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=1)  # [HW, C]

        return pos.unsqueeze(0)  # [1, HW, C]

    def _position_self_attention_gate(self, B, H, W, device, dtype):
        """
        只对位置编码做 SA，生成纯空间 gate

        return:
            gate: [B, 1, H, W]
        """
        pos = self._build_2d_sincos_pos_embed(H, W, self.channels, device, dtype)  # [1, HW, C]
        pos = self.pos_scale * pos
        pos = pos.expand(B, -1, -1).contiguous()  # [B, HW, C]

        pos = self.pos_ln(pos)
        pos_attn_out, _ = self.pos_attn(pos, pos, pos, need_weights=False)  # [B, HW, C]

        gate_logits = self.gate_proj(pos_attn_out)  # [B, HW, 1]

        if self.gate_use_softmax:
            gate = F.softmax(gate_logits, dim=1)  # [B, HW, 1]
        else:
            gate = torch.sigmoid(gate_logits)  # [B, HW, 1]

        gate = gate.transpose(1, 2).contiguous().view(B, 1, H, W)  # [B,1,H,W]
        return gate

    # =========================================================
    # forward
    # =========================================================
    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        HW = H * W

        self.update_momentum()

        # 1) query feature
        q = self.query_proj(x)  # [B, C, H, W]
        q_flat = q.view(B, C, HW).permute(0, 2, 1).contiguous()  # [B, HW, C]

        # 2) prototype assignment + prototype reconstruction
        A, P_mean, P_var = self._cosine_assign(q_flat)  # A:[B,HW,M], P_mean:[B,M,C]
        q_rec_flat = torch.bmm(A, P_mean)  # [B, HW, C]
        q_rec = q_rec_flat.permute(0, 2, 1).contiguous().view(B, C, H, W)
        q_rec = self.recon_proj(q_rec)  # [B, C, H, W]

        # 3) 只用位置编码做 self-attention，生成 spatial gate
        gate = self._position_self_attention_gate(B, H, W, x.device, x.dtype)  # [B,1,H,W]

        # 4) 用 gate 筛选 q_rec，再与 x 残差相加
        out = x + gate * q_rec
        out = self.out_proj(out)
        out = self.norm(out)

        # 5) EMA update
        if self.training and self.update_in_train:
            self._ema_update_prototypes(q_flat.detach(), A.detach())

        # 6) KL loss
        kl_loss = self._kl_divergence()
        if "KL_LOSS_CONTAINER" in globals():
            KL_LOSS_CONTAINER.append(kl_loss)

        return out

    def update_momentum(self):
        if not self.training:
            return

        if self.momentum < self.momentum_end:
            self.momentum += self.momentum_step
            if self.momentum > self.momentum_end:
                self.momentum = self.momentum_end


class PGCR_EMA_Cosine_MSCM(nn.Module):
    """
    改进版：
    1. 保留 prototype reconstruction
    2. gate 使用 q_rec + hybrid positional embedding
    3. hybrid positional embedding = sin-cos + learnable(可插值)
    4. 输出: out = x + gate * q_rec
    """

    def __init__(
        self,
        channels,
        num_prototypes=10,
        reduction=4,
        momentum_start=0.25,
        momentum_end=0.50,
        temperature=0.1,
        eps=1e-6,
        update_in_train=True,
        use_norm=True,
        proto_noise_std=0.05,
        noise_on_eval=False,
        renorm_after_noise=True,
        num_heads=2,
        use_gaussian_scaling=True,
        gate_use_softmax=False,
        learnable_pos_base=20,
    ):
        super().__init__()

        assert channels % num_heads == 0, f"channels({channels}) 必须能被 num_heads({num_heads}) 整除"
        assert channels % 4 == 0, f"channels({channels}) 最好能被4整除，便于2D sin-cos位置编码"

        self.channels = channels
        self.num_prototypes = num_prototypes
        self.momentum = momentum_start
        self.momentum_end = momentum_end
        self.total_iters = 7000
        self.momentum_step = (momentum_end - momentum_start) / self.total_iters

        self.temperature = temperature
        self.eps = eps
        self.update_in_train = update_in_train
        self.proto_noise_std = proto_noise_std
        self.noise_on_eval = noise_on_eval
        self.renorm_after_noise = renorm_after_noise
        self.use_gaussian_scaling = use_gaussian_scaling
        self.gate_use_softmax = gate_use_softmax
        self.learnable_pos_base = learnable_pos_base

        hidden = max(channels // reduction, 16)

        # ----------------------------
        # q feature projection
        # ----------------------------
        self.query_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False), nn.BatchNorm2d(channels), nn.SiLU()
        )

        self.recon_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False), nn.BatchNorm2d(channels), nn.SiLU()
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False), nn.BatchNorm2d(channels), nn.SiLU()
        )

        self.norm = nn.BatchNorm2d(channels) if use_norm else nn.Identity()

        # ----------------------------
        # gate: SA over (q_rec + hybrid_pos)
        # ----------------------------
        self.gate_ln1 = nn.LayerNorm(channels)
        self.gate_attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.gate_ln2 = nn.LayerNorm(channels)

        self.gate_ffn = nn.Sequential(nn.Linear(channels, hidden), nn.SiLU(), nn.Linear(hidden, 1))

        # sin-cos 强度
        self.sincos_scale = nn.Parameter(torch.tensor(1.0))
        # learnable pos 强度
        self.learnable_pos_scale = nn.Parameter(torch.tensor(1.0))

        # learnable pos: 用一个基础尺寸，再插值到当前 H,W
        self.learnable_pos = nn.Parameter(torch.randn(1, channels, learnable_pos_base, learnable_pos_base) * 0.02)

        # ----------------------------
        # prototypes
        # ----------------------------
        init_mean = torch.randn(num_prototypes, channels) * 0.1
        init_var = torch.ones(num_prototypes, channels) * 0.1

        self.register_buffer("prototypes_mean", init_mean)  # [M, C]
        self.register_buffer("prototypes_var", init_var)  # [M, C]
        self.register_buffer("proto_counter", torch.zeros(num_prototypes))

    # =========================================================
    # prototype utils
    # =========================================================
    @torch.no_grad()
    def reset_prototypes(self):
        proto_mean = torch.randn(self.num_prototypes, self.channels, device=self.prototypes_mean.device) * 0.1
        proto_var = torch.ones(self.num_prototypes, self.channels, device=self.prototypes_var.device) * 0.1

        self.prototypes_mean.copy_(proto_mean)
        self.prototypes_var.copy_(proto_var)
        self.proto_counter.zero_()

    def _get_prototypes(self, B):
        P_mean = self.prototypes_mean.detach().clone().unsqueeze(0).expand(B, -1, -1)  # [B,M,C]
        P_var = self.prototypes_var.detach().clone().unsqueeze(0).expand(B, -1, -1)  # [B,M,C]
        return P_mean, P_var

    def _gaussian_scale(self, q_flat, P_mean, P_var):
        """
        用 prototype 的高斯统计，对 cosine sim 做缩放
        q_flat: [B,HW,C]
        P_mean: [B,M,C]
        P_var: [B,M,C]
        return: [B,HW,M]
        """
        q_exp = q_flat.unsqueeze(2)  # [B,HW,1,C]
        mu_exp = P_mean.unsqueeze(1)  # [B,1,M,C]
        var_exp = P_var.unsqueeze(1).clamp_min(1e-4)  # [B,1,M,C]

        mahal = ((q_exp - mu_exp) ** 2) / (var_exp + self.eps)  # [B,HW,M,C]
        mahal = mahal.mean(dim=-1)  # [B,HW,M]

        scale = torch.exp(-0.5 * mahal)  # [B,HW,M]
        return scale

    def _cosine_assign(self, q_flat):
        """
        q_flat: [B, HW, C]
        return:
            A:      [B, HW, M]
            P_mean: [B, M, C]
            P_var:  [B, M, C]
        """
        B, HW, C = q_flat.shape
        P_mean, P_var = self._get_prototypes(B)

        std = torch.sqrt(P_var.clamp_min(1e-4))
        P_samples = torch.randn_like(P_mean) * std + P_mean  # [B,M,C]

        q_norm = F.normalize(q_flat, dim=-1, eps=self.eps)  # [B,HW,C]
        p_norm = F.normalize(P_samples, dim=-1, eps=self.eps)  # [B,M,C]

        sim = torch.bmm(q_norm, p_norm.transpose(1, 2))  # [B,HW,M]

        if self.use_gaussian_scaling:
            scale = self._gaussian_scale(q_flat, P_mean, P_var)  # [B,HW,M]
            sim = sim * scale

        sim = sim / self.temperature
        A = F.softmax(sim, dim=-1)

        return A, P_mean, P_var

    @torch.no_grad()
    def _ema_update_prototypes(self, q_flat, A):
        """
        q_flat: [B, HW, C]
        A: [B, HW, M]
        """
        q_norm = F.normalize(q_flat, dim=-1, eps=self.eps)  # [B,HW,C]

        proto_batch = torch.bmm(A.transpose(1, 2), q_norm)  # [B,M,C]
        denom = A.transpose(1, 2).sum(dim=-1, keepdim=True) + self.eps
        proto_batch = proto_batch / denom

        mean_now = proto_batch.mean(dim=0)  # [M,C]
        var_now = proto_batch.var(dim=0, unbiased=False)  # [M,C]

        new_mean = self.momentum * self.prototypes_mean + (1.0 - self.momentum) * mean_now
        new_var = self.momentum * self.prototypes_var + (1.0 - self.momentum) * var_now

        new_mean = F.normalize(new_mean, dim=-1, eps=self.eps)
        new_var = new_var.clamp_min(1e-4)

        self.prototypes_mean.copy_(new_mean)
        self.prototypes_var.copy_(new_var)

    def _kl_divergence(self):
        """
        KL( N(mu,var) || N(0,1) )
        """
        p_mean = self.prototypes_mean
        p_var = self.prototypes_var.clamp_min(1e-4)

        kl_loss = 0.5 * torch.sum(p_mean**2 + p_var - torch.log(p_var + self.eps) - 1.0)
        return kl_loss

    # =========================================================
    # hybrid positional embedding
    # =========================================================
    def _build_2d_sincos_pos_embed(self, H, W, C, device, dtype):
        """
        return: [1, HW, C]
        """
        assert C % 4 == 0, "channels 必须能被4整除"

        y = torch.arange(H, device=device, dtype=dtype)
        x = torch.arange(W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")  # [H,W]

        yy = yy.reshape(-1, 1)  # [HW,1]
        xx = xx.reshape(-1, 1)  # [HW,1]

        dim = C // 4
        omega = torch.arange(dim, device=device, dtype=dtype)
        omega = 1.0 / (10000 ** (omega / dim))  # [dim]

        out_y = yy * omega.unsqueeze(0)  # [HW,dim]
        out_x = xx * omega.unsqueeze(0)  # [HW,dim]

        pos = torch.cat([torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=1)  # [HW,C]

        return pos.unsqueeze(0)  # [1,HW,C]

    def _get_learnable_pos_embed(self, H, W, B, device, dtype):
        """
        learnable pos 从基础尺寸插值到当前尺寸
        return: [B, HW, C]
        """
        pos = F.interpolate(self.learnable_pos, size=(H, W), mode="bilinear", align_corners=False)  # [1,C,H,W]
        pos = pos.to(device=device, dtype=dtype)
        pos = pos.flatten(2).transpose(1, 2).contiguous()  # [1,HW,C]
        pos = pos.expand(B, -1, -1)  # [B,HW,C]
        return pos

    def _get_hybrid_pos_embed(self, H, W, B, device, dtype):
        """
        sin-cos + learnable
        return: [B, HW, C]
        """
        pos_sincos = self._build_2d_sincos_pos_embed(H, W, self.channels, device, dtype)  # [1,HW,C]
        pos_sincos = pos_sincos.expand(B, -1, -1)  # [B,HW,C]

        pos_learn = self._get_learnable_pos_embed(H, W, B, device, dtype)  # [B,HW,C]

        pos = self.sincos_scale * pos_sincos + self.learnable_pos_scale * pos_learn
        return pos

    # =========================================================
    # gate
    # =========================================================
    def _generate_gate(self, q_rec, H, W):
        """
        q_rec: [B,C,H,W]
        return gate: [B,1,H,W]
        """
        B, C, _, _ = q_rec.shape

        q_rec_flat = q_rec.flatten(2).transpose(1, 2).contiguous()  # [B,HW,C]
        pos = self._get_hybrid_pos_embed(H, W, B, q_rec.device, q_rec.dtype)  # [B,HW,C]

        tokens = q_rec_flat + pos  # [B,HW,C]

        # Pre-LN + SA + residual
        tokens_ln = self.gate_ln1(tokens)
        attn_out, _ = self.gate_attn(tokens_ln, tokens_ln, tokens_ln, need_weights=False)
        tokens = tokens + attn_out

        tokens = self.gate_ln2(tokens)
        gate_logits = self.gate_ffn(tokens)  # [B,HW,1]

        if self.gate_use_softmax:
            gate = F.softmax(gate_logits, dim=1)  # [B,HW,1]
        else:
            gate = torch.sigmoid(gate_logits)  # [B,HW,1]

        gate = gate.transpose(1, 2).contiguous().view(B, 1, H, W)  # [B,1,H,W]
        return gate

    # =========================================================
    # forward
    # =========================================================
    def forward(self, x):
        """
        x: [B,C,H,W]
        """
        B, C, H, W = x.shape
        HW = H * W

        self.update_momentum()

        # 1) query feature
        q = self.query_proj(x)  # [B,C,H,W]
        q_flat = q.view(B, C, HW).permute(0, 2, 1).contiguous()  # [B,HW,C]

        # 2) prototype reconstruction
        A, P_mean, P_var = self._cosine_assign(q_flat)  # A:[B,HW,M]
        q_rec_flat = torch.bmm(A, P_mean)  # [B,HW,C]
        q_rec = q_rec_flat.permute(0, 2, 1).contiguous().view(B, C, H, W)
        q_rec = self.recon_proj(q_rec)  # [B,C,H,W]

        # 3) gate from (q_rec + hybrid_pos)
        gate = self._generate_gate(q_rec, H, W)  # [B,1,H,W]

        # 4) residual fusion
        out = x + gate * q_rec
        out = self.out_proj(out)
        # out = self.norm(out)

        # 5) EMA update
        if self.training and self.update_in_train:
            self._ema_update_prototypes(q_flat.detach(), A.detach())

        # 6) KL
        kl_loss = self._kl_divergence()
        KL_LOSS_CONTAINER.append(kl_loss)

        return out

    def update_momentum(self):
        if not self.training:
            return

        if self.momentum < self.momentum_end:
            self.momentum += self.momentum_step
            if self.momentum > self.momentum_end:
                self.momentum = self.momentum_end


class C2f_DSC(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        # self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv1 = DepthwiseSeparableConv(c1, 2 * self.c)
        # self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.cv2 = DepthwiseSeparableConv((2 + n) * self.c, c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class ProtoGRUCell(nn.Module):
    """
    对每个 prototype 槽位做共享参数的 GRU-style 更新
    输入:
        p_old: [B, M, C]
        p_new: [B, M, C]
    输出:
        p:     [B, M, C]
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 16)

        self.z_net = nn.Sequential(nn.Linear(channels * 2, hidden), nn.SiLU(), nn.Linear(hidden, channels))
        self.r_net = nn.Sequential(nn.Linear(channels * 2, hidden), nn.SiLU(), nn.Linear(hidden, channels))
        self.h_net = nn.Sequential(nn.Linear(channels * 2, hidden), nn.SiLU(), nn.Linear(hidden, channels))

    def forward(self, p_old, p_new):
        x = torch.cat([p_old, p_new], dim=-1)  # [B, M, 2C]

        z = torch.sigmoid(self.z_net(x))
        r = torch.sigmoid(self.r_net(x))

        h_in = torch.cat([r * p_old, p_new], dim=-1)
        h_tilde = torch.tanh(self.h_net(h_in))

        p = (1.0 - z) * p_old + z * h_tilde
        return p


class PGCR_Recurrent(nn.Module):
    """
    基于你当前 SOTA PGCR 的“单图递归精炼版”

    核心思想：
    1) prototype 仍然来自当前图像
    2) prototype 聚合仍然基于原始 x
    3) assignment 通过 recon-conditioned feature 递归细化
    4) prototype 更新用 GRU-style，而不是手工 momentum
    5) 可选弱全局参考 prototype，但默认关闭
    """

    def __init__(
        self,
        channels,
        num_prototypes=8,
        reduction=2,
        use_norm=True,
        refine_steps=3,
        use_global_ref=False,
        global_ref_weight=0.01,
    ):
        super().__init__()

        self.channels = channels
        self.num_prototypes = num_prototypes
        self.use_norm = use_norm
        self.refine_steps = refine_steps
        self.use_global_ref = use_global_ref
        self.global_ref_weight = global_ref_weight

        hidden = max(channels // reduction, 16)

        # -------- Prototype Learning Unit --------
        self.proto_assign = nn.Conv2d(channels, num_prototypes, kernel_size=1, bias=True)

        self.proto_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False), nn.BatchNorm2d(channels), nn.SiLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False), nn.BatchNorm2d(channels), nn.SiLU()
        )

        # -------- Recurrent prototype updater --------
        self.proto_update = ProtoGRUCell(channels, reduction=reduction)

        # -------- Optional weak global reference --------
        if use_global_ref:
            self.global_proto = nn.Parameter(torch.randn(1, num_prototypes, channels) * 0.02)

        # -------- MSCM-like gating --------
        self.channel_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

        self.sigmoid = nn.Sigmoid()

        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False), nn.BatchNorm2d(channels), nn.SiLU()
        )

        self.norm = nn.BatchNorm2d(channels) if use_norm else nn.Identity()

    def _get_assign_map(self, feat):
        """
        feat: [B, C, H, W]
        return:
            assign_map: [B, HW, M]
        """
        B, _, H, W = feat.shape
        HW = H * W

        assign_map = self.proto_assign(feat)  # [B, M, H, W]
        assign_map = assign_map.view(B, self.num_prototypes, HW).permute(0, 2, 1).contiguous()
        assign_map = F.softmax(assign_map, dim=-1)  # [B, HW, M]
        return assign_map

    def _aggregate_prototypes(self, x, assign_map):
        """
        用原始 x 聚合 prototype
        x: [B, C, H, W]
        assign_map: [B, HW, M]
        return:
            prototypes: [B, M, C]
        """
        B, C, H, W = x.shape
        HW = H * W

        x_flat = x.view(B, C, HW).permute(0, 2, 1).contiguous()  # [B, HW, C]

        prototypes = torch.bmm(assign_map.transpose(1, 2), x_flat)  # [B, M, C]
        denom = assign_map.transpose(1, 2).sum(dim=-1, keepdim=True) + 1e-6
        prototypes = prototypes / denom

        return prototypes

    def _write_back(self, assign_map, prototypes, H, W):
        """
        assign_map: [B, HW, M]
        prototypes: [B, M, C]
        return:
            recon: [B, C, H, W]
        """
        B, HW, _ = assign_map.shape
        C = prototypes.shape[-1]

        recon = torch.bmm(assign_map, prototypes)  # [B, HW, C]
        recon = recon.permute(0, 2, 1).contiguous().view(B, C, H, W)
        recon = self.proto_proj(recon)
        return recon

    def _maybe_add_global_ref(self, prototypes):
        """
        prototypes: [B, M, C]
        用弱全局 prototype 做初始化参考
        """
        if not self.use_global_ref:
            return prototypes

        B = prototypes.shape[0]
        g = self.global_proto.expand(B, -1, -1)  # [B, M, C]
        beta = self.global_ref_weight
        prototypes = (1.0 - beta) * prototypes + beta * g
        return prototypes

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # --------------------------------------------------
        # Step 0: 初始 assignment 与 prototype
        # --------------------------------------------------
        assign_map = self._get_assign_map(x)  # [B, HW, M]
        prototypes = self._aggregate_prototypes(x, assign_map)  # [B, M, C]
        prototypes = self._maybe_add_global_ref(prototypes)

        # --------------------------------------------------
        # Recurrent refinement inside one image
        # --------------------------------------------------
        # 只细化 assignment；prototype 仍然从原始 x 聚合
        for _ in range(self.refine_steps - 1):
            recon = self._write_back(assign_map, prototypes, H, W)  # [B, C, H, W]
            z = self.fuse(x + recon)  # [B, C, H, W]

            # 用 z 更新 assignment，全局学习变成局部学习
            assign_new = self._get_assign_map(z)  # [B, HW, M]

            # 但 prototype 仍然从原始 x 聚合，保持与你原版 PGCR 更一致
            proto_new = self._aggregate_prototypes(x, assign_new)  # [B, M, C]

            # GRU-style update
            prototypes = self.proto_update(prototypes, proto_new)  # [B, M, C]

            # 可选：轻微参考全局 prototype
            prototypes = self._maybe_add_global_ref(prototypes)

            assign_map = assign_new

        # --------------------------------------------------
        # Final reconstruction
        # --------------------------------------------------
        recon = self._write_back(assign_map, prototypes, H, W)
        z = self.fuse(x + recon)

        # --------------------------------------------------
        # MSCM-like gate
        # --------------------------------------------------
        wg = self.channel_branch(z)  # [B, C, 1, 1]
        wl = self.local_branch(z)  # [B, C, H, W]
        gate = self.sigmoid(wg + wl)

        # --------------------------------------------------
        # residual refinement
        # 保持你原始 SOTA 写法
        # --------------------------------------------------
        out = gate * x + x
        out = self.out_proj(out)
        out = self.norm(out)

        return out
