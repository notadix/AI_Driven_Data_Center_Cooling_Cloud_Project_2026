import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.w1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def _cmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, : self.modes1, : self.modes2] = self._cmul(x_ft[:, :, : self.modes1, : self.modes2], self.w1)
        out_ft[:, :, -self.modes1 :, : self.modes2] = self._cmul(x_ft[:, :, -self.modes1 :, : self.modes2], self.w2)

        return torch.fft.irfft2(out_ft, s=(H, W))


class FNO2d(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        modes1: int = 4,
        modes2: int = 4,
        width: int = 32,
        num_layers: int = 4,
    ):
        super().__init__()
        self.fc0 = nn.Linear(in_channels, width)

        self.spec_layers = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(num_layers)]
        )
        self.bypass = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=1) for _ in range(num_layers)]
        )

        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        for spec, bypass in zip(self.spec_layers, self.bypass):
            x = F.gelu(spec(x) + bypass(x))
        x = F.gelu(self.fc1(x.permute(0, 2, 3, 1)))
        return self.fc2(x).permute(0, 3, 1, 2)


def create_fno(in_channels: int = 3, out_channels: int = 1, width: int = 32) -> FNO2d:
    return FNO2d(in_channels=in_channels, out_channels=out_channels, width=width)
