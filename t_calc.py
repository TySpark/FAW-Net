from enum import IntEnum

import torch


class Component(IntEnum):
    """
    电磁场分量枚举

    定义大地电磁测量中的七个场分量：
    - 磁场分量: Hx, Hy, Hz
    - 电场分量: Ex, Ey
    - 远程参考: Rx, Ry

    标准7x7功率谱矩阵的索引顺序：
    0:Hx, 1:Hy, 2:Hz, 3:Ex, 4:Ey, 5:Rx, 6:Ry

    Attributes:
        Hx: 磁场X分量 (索引 0)
        Hy: 磁场Y分量 (索引 1)
        Hz: 磁场Z分量 (索引 2)
        Ex: 电场X分量 (索引 3)
        Ey: 电场Y分量 (索引 4)
        Rx: 远程参考X分量 (索引 5)
        Ry: 远程参考Y分量 (索引 6)
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
    从解包后的矩阵中获取复数元素 S(r, c)
    matrix: (7, 7) 这里的输入应是加权后的实数矩阵
    """
    # 遵循你之前的 at() 逻辑：
    # 对角线：实数
    # 下三角 (r > c)：实部
    # 上三角 (r < c)：-虚部
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
    计算 ZXY 和 ZYX 分量
    matrix: (7, 7) 这里的输入应是加权后的实数矩阵
    """
    # 2. 提取分量并计算 Z
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
    计算 ZXX 和 ZYY 分量
    matrix: (7, 7) 这里的输入应是加权后的实数矩阵
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
    # 取对数提高训练稳定性
    # 将 rho 限制在最小为一个极小值（比如 1e-8），防止对 0 或负数取对数
    return torch.log10(torch.clamp(rho, min=1e-10))


def calc_rho_phs(
    freq: float, matrix: torch.Tensor, is_log: bool = True
) -> torch.Tensor:
    """
    计算 rho 和 phi 分量
    matrix: (7, 7) 这里的输入应是加权后的实数矩阵
    return rxy(是否对数), ryx, pxy, pyx(弧度)
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
