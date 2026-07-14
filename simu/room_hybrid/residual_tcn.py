"""Causal residual branch for V5 slow-model correction.

The grey-box V5 model remains the plant backbone. This branch predicts the
remaining room-temperature residual from the slow trajectory and actuator
history, so low-frequency bias and PID-maintenance waves can be learned without
forcing the physical branch to become unrealistically fast.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ResidualTCNConfig:
    input_size: int
    hidden_size: int = 64
    levels: int = 5
    kernel_size: int = 5
    dropout: float = 0.05
    max_residual_c: float = 2.0

    def to_dict(self) -> dict:
        return asdict(self)


class CausalConv1d(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels_in, channels_out, kernel_size,
            dilation=dilation, padding=self.left_padding,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.conv(values)
        if self.left_padding:
            output = output[..., :-self.left_padding]
        return output


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.net(values)


class ResidualTCN(nn.Module):
    """Causal sequence model that predicts ``truth - slow_prediction``."""

    def __init__(self, config: ResidualTCNConfig):
        super().__init__()
        self.config = config
        self.register_buffer("feature_mean", torch.zeros(config.input_size))
        self.register_buffer("feature_scale", torch.ones(config.input_size))
        self.input_projection = nn.Conv1d(config.input_size, config.hidden_size, 1)
        self.blocks = nn.Sequential(*[
            ResidualBlock(
                config.hidden_size,
                config.kernel_size,
                dilation=2 ** level,
                dropout=config.dropout,
            )
            for level in range(config.levels)
        ])
        self.head = nn.Sequential(
            nn.Conv1d(config.hidden_size, config.hidden_size, 1),
            nn.GELU(),
            nn.Conv1d(config.hidden_size, 1, 1),
        )

    def set_feature_normalization(self, mean, scale) -> None:
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
        scale_tensor = torch.as_tensor(scale, dtype=torch.float32).clamp_min(1e-4)
        if mean_tensor.numel() != self.config.input_size:
            raise ValueError("feature mean size does not match model input size")
        self.feature_mean.copy_(mean_tensor)
        self.feature_scale.copy_(scale_tensor)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return residual temperatures for ``features`` shaped [batch, time, input]."""
        normalized = (features - self.feature_mean) / self.feature_scale
        values = normalized.transpose(1, 2)
        hidden = self.input_projection(values)
        hidden = self.blocks(hidden)
        residual = self.head(hidden).squeeze(1)
        return self.config.max_residual_c * torch.tanh(residual / self.config.max_residual_c)
