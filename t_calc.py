from enum import IntEnum

import torch


class Component(IntEnum):
    """
    Electromagnetic field component enumeration

    Defines the seven field components used in magnetotelluric measurements:
    - Magnetic field: Hx, Hy, Hz
    - Electric field: Ex, Ey
    - Remote reference: Rx, Ry

    Index order of the standard 7x7 power spectral matrix:
    0:Hx, 1:Hy, 2:Hz, 3:Ex, 4:Ey, 5:Rx, 6:Ry

    Attributes:
        Hx: magnetic field X component (index 0)
        Hy: magnetic field Y component (index 1)
        Hz: magnetic field Z component (index 2)
        Ex: electric field X component (index 3)
        Ey: electric field Y component (index 4)
        Rx: remote reference X component (index 5)
        Ry: remote reference Y component (index 6)
    """

    Hx = 0
    Hy = 1
    Hz = 2
    Ex = 3
    Ey = 4
    Rx = 5
    Ry = 6


def get_complex_at(matrix: torch.Tensor, row: Component, col: Component):
    """
    Retrieve the complex element S(r, c) from an unpacked matrix
    matrix: (7, 7); the input here should be a weighted real-valued matrix
    """
    # Follow the same logic as the previous at() method:
    # Diagonal: real
    # Lower triangle (r > c): real part
    # Upper triangle (r < c): -imaginary part
    r = row.value
    c = col.value
    if r == c:
        return torch.complex(matrix[..., r, c], torch.zeros_like(matrix[..., r, c]))
    elif r > c:
        re = matrix[..., r, c]
        im = matrix[..., c, r]
        return torch.complex(re, im)
    else:
        re = matrix[..., c, r]
        im = -matrix[..., r, c]
        return torch.complex(re, im)


def calc_zxy_zyx(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the ZXY and ZYX components
    matrix: (7, 7); the input here should be a weighted real-valued matrix
    """
    # 2. Extract components and compute Z
    s_hxrx = get_complex_at(matrix, Component.Hx, Component.Rx)
    s_hyry = get_complex_at(matrix, Component.Hy, Component.Ry)
    s_hxry = get_complex_at(matrix, Component.Hx, Component.Ry)
    s_hyrx = get_complex_at(matrix, Component.Hy, Component.Rx)
    s_exrx = get_complex_at(matrix, Component.Ex, Component.Rx)
    s_exry = get_complex_at(matrix, Component.Ex, Component.Ry)
    s_eyrx = get_complex_at(matrix, Component.Ey, Component.Rx)
    s_eyry = get_complex_at(matrix, Component.Ey, Component.Ry)

    denom = s_hxrx * s_hyry - s_hxry * s_hyrx + 1e-12

    zxy = (s_exry * s_hxrx - s_exrx * s_hxry) / denom
    zyx = (s_eyrx * s_hyry - s_eyry * s_hyrx) / denom

    return zxy, zyx


def calc_zxx_zyy(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the ZXX and ZYY components
    matrix: (7, 7); the input here should be a weighted real-valued matrix
    """
    s_hxrx = get_complex_at(matrix, Component.Hx, Component.Rx)
    s_hyry = get_complex_at(matrix, Component.Hy, Component.Ry)
    s_hxry = get_complex_at(matrix, Component.Hx, Component.Ry)
    s_hyrx = get_complex_at(matrix, Component.Hy, Component.Rx)
    s_exrx = get_complex_at(matrix, Component.Ex, Component.Rx)
    s_exry = get_complex_at(matrix, Component.Ex, Component.Ry)
    s_eyrx = get_complex_at(matrix, Component.Ey, Component.Rx)
    s_eyry = get_complex_at(matrix, Component.Ey, Component.Ry)

    denom = s_hxrx * s_hyry - s_hxry * s_hyrx + 1e-12

    zxx = (s_exrx * s_hyry - s_exry * s_hyrx) / denom
    zyy = (s_eyry * s_hxrx - s_eyrx * s_hxry) / denom

    return zxx, zyy


def calc_log10(rho):
    # Take the log for training stability
    # Clamp rho to a minimum tiny value (e.g. 1e-8) to avoid log of zero or negative numbers
    return torch.log10(torch.clamp(rho, min=1e-10))


def calc_rho_phs(
    freq: float, matrix: torch.Tensor, is_log: bool = True
) -> torch.Tensor:
    """
    Compute the rho and phi components
    matrix: (7, 7); the input here should be a weighted real-valued matrix
    return rxy (log or not), ryx, pxy, pyx (radians)
    """
    zxy, zyx = calc_zxy_zyx(matrix)

    rxy = 0.2 * (zxy.real**2 + zxy.imag**2) / freq
    ryx = 0.2 * (zyx.real**2 + zyx.imag**2) / freq
    pxy = torch.angle(zxy)
    pyx = torch.angle(zyx)

    if is_log:
        return torch.stack(
            [
                calc_log10(rxy),
                calc_log10(ryx),
                pxy,
                pyx,
            ]
        )
    else:
        return torch.stack(
            [
                rxy + 1e-10,
                ryx + 1e-10,
                pxy,
                pyx,
            ]
        )
