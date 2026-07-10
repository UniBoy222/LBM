#!/usr/bin/env python3
"""Differentiable pressure iteration from book Eqs. (6.42)-(6.44)."""

from __future__ import annotations

from typing import NamedTuple

import torch


EX = (0, 1, 0, 0, -1, 0, 0, 1, -1, 1, 1, -1, 1, -1, -1)
EY = (0, 0, 1, 0, 0, -1, 0, 1, 1, -1, 1, -1, -1, 1, -1)
EZ = (0, 0, 0, 1, 0, 0, -1, 1, 1, 1, -1, -1, -1, -1, 1)
EI = (
    2.0 / 9.0,
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 72.0, 1.0 / 72.0, 1.0 / 72.0, 1.0 / 72.0,
    1.0 / 72.0, 1.0 / 72.0, 1.0 / 72.0, 1.0 / 72.0,
)


class BookPressureStages(NamedTuple):
    divergence: object
    collision: object
    streamed: object
    bounced: object
    pressure: object


def _neighbor(field, ex: int, ey: int, ez: int):
    """Fortran firstord neighbor: symmetric x, periodic y/z."""
    shifted = field.roll(shifts=(-ez, -ey), dims=(2, 3))
    if ex == 1:
        return torch.cat((shifted[..., 1:], shifted[..., -2:-1]), dim=4)
    if ex == -1:
        return torch.cat((shifted[..., 1:2], shifted[..., :-1]), dim=4)
    return shifted


def firstord(field):
    """Book/Fortran firstord for a [B,1,Z,Y,X] tensor."""
    px = field.new_zeros(field.shape)
    py = field.new_zeros(field.shape)
    pz = field.new_zeros(field.shape)
    for q in range(1, 15):
        neighbor = _neighbor(field, EX[q], EY[q], EZ[q])
        px = px + neighbor * EX[q]
        py = py + neighbor * EY[q]
        pz = pz + neighbor * EZ[q]
    return px / 10.0, py / 10.0, pz / 10.0


def divergence_firstord(u, v, w):
    upx, _, _ = firstord(u)
    _, vpy, _ = firstord(v)
    _, _, wpz = firstord(w)
    return upx + vpy + wpz


def collision(hh, rho, divergence, dx: float = 1.0):
    """Eq. (6.42) collision/correction with Eq. (6.43)."""
    if hh.shape[1] != 15:
        raise ValueError(f"expected 15 h channels, got {hh.shape[1]}")
    pressure = hh.sum(dim=1, keepdim=True)
    tauh = 1.0 / rho + 0.5
    weights = hh.new_tensor(EI).view(1, 15, 1, 1, 1)
    return hh - (hh - weights * pressure) / tauh - (weights / 3.0) * divergence * dx


def stream(hh_collision):
    """Fortran stream(hh): periodic propagation in x/y/z."""
    channels = [
        hh_collision[:, q:q + 1].roll(
            shifts=(EZ[q], EY[q], EX[q]), dims=(2, 3, 4))
        for q in range(15)
    ]
    return torch.cat(channels, dim=1)


def slip_bounceback(hh_streamed):
    """Fortran slip_bounceback(hh) at x=1 and x=lx."""
    left = {1: 4, 7: 8, 9: 14, 10: 13, 12: 11}
    right = {4: 1, 8: 7, 14: 9, 13: 10, 11: 12}
    channels = []
    for q in range(15):
        value = hh_streamed[:, q:q + 1]
        if q in left:
            source = hh_streamed[:, left[q]:left[q] + 1, ..., :1]
            value = torch.cat((source, value[..., 1:]), dim=4)
        if q in right:
            source = hh_streamed[:, right[q]:right[q] + 1, ..., -1:]
            value = torch.cat((value[..., :-1], source), dim=4)
        channels.append(value)
    return torch.cat(channels, dim=1)


def t_book(hh, rho, u, v, w, dx: float = 1.0, divergence=None) -> BookPressureStages:
    """One complete correction -> stream -> bounce -> getp iteration."""
    div_u = divergence_firstord(u, v, w) if divergence is None else divergence
    collided = collision(hh, rho, div_u, dx)
    streamed = stream(collided)
    bounced = slip_bounceback(streamed)
    pressure = bounced.sum(dim=1, keepdim=True)
    return BookPressureStages(div_u, collided, streamed, bounced, pressure)


def fixed_point_loss(hh_pred, rho, u, v, w, dx: float = 1.0, divergence=None):
    """L_fixed_point = mean(||T_book(h_pred)-h_pred||^2)."""
    hh_next = t_book(hh_pred, rho, u, v, w, dx, divergence).bounced
    return (hh_next - hh_pred).square().mean()
