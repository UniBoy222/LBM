#!/usr/bin/env python3
"""Fifteen-channel h_i initializer model for the book pressure operator."""

from __future__ import annotations

import torch
import torch.nn as nn


INPUT_CHANNELS = ("rho", "u", "v", "w", "fei", "press", "div_u_source")
OUTPUT_CHANNELS = tuple(f"h{i}" for i in range(15))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.GELU()

    def forward(self, value):
        return self.activation(value + self.net(value))


class HInitializer(nn.Module):
    def __init__(self, width: int = 32, depth: int = 4, input_channels: int | None = None) -> None:
        super().__init__()
        input_channels = len(INPUT_CHANNELS) if input_channels is None else input_channels
        layers: list[nn.Module] = [
            nn.Conv3d(input_channels, width, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        layers.extend(ResidualBlock(width) for _ in range(depth))
        layers.append(nn.Conv3d(width, len(OUTPUT_CHANNELS), kernel_size=1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        hh = self.network(inputs)
        if hh.shape[1] != 15:
            raise RuntimeError("h initializer must produce exactly 15 channels")
        return hh


def pressure_from_h(hh):
    if hh.shape[1] != 15:
        raise ValueError(f"expected 15 h channels, got {hh.shape[1]}")
    return torch.sum(hh, dim=1, keepdim=True)
