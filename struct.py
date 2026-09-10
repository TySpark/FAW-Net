from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator, Sequence, Union

import numpy as np
from numpy.typing import NDArray


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


class RefChannel(IntEnum):
    """
    参考通道枚举

    定义阻抗计算时可用的参考通道类型：
    - E: 电场参考 (用于电场参考法)
    - H: 磁场参考 (标准方法，默认)
    - R: 远程参考 (用于远程参考处理)
    """

    E = 0
    H = 1
    R = 2


def calc_resistivity(freq: float, z: complex) -> float:
    """
    计算视电阻率

    公式: ρ = |Z|² / (ωμ) = |Z|² * 0.2 / f
    其中 μ = 4π×10⁻⁷ H/m

    Args:
        freq: 频率 (Hz)
        z: 阻抗值

    Returns:
        视电阻率 (Ω·m)
    """
    return abs(z) ** 2 * 0.2 / freq


def calc_phase_deg(z: complex) -> float:
    """
    计算相位 (度)

    Args:
        z: 阻抗值

    Returns:
        相位角 (度)
    """
    return np.angle(z, deg=True)


def calc_power_density(psm: "SinglePSM", c1: Component, c2: Component) -> float:
    """
    计算功率密度

    公式: P = |S(c1, c2)| / window_length

    Args:
        psm: 功率谱矩阵
        c1: 第一个通道索引
        c2: 第二个通道索引

    Returns:
        功率密度
    """

    return abs(psm.at(c1, c2)) / psm.window_length


def calc_phase_tensor(zxx: complex, zxy: complex, zyx: complex, zyy: complex):
    """
    根据 Caldwell 等 (2004) 实现的大地电磁相位张量计算。

    返回:
        Phi: 2x2 相位张量矩阵
        alpha: 坐标依赖角 (度)
        beta: 偏斜角 (度)
    """
    # 1. 构造阻抗张量矩阵 Z 并分解为实部 X 和虚部 Y (Eq. 18)
    # Z = X + iY
    Z = np.array([[zxx, zxy], [zyx, zyy]])
    X = Z.real
    Y = Z.imag

    # 2. 计算相位张量 Phi = X^-1 * Y (Eq. 18)
    # 使用 solve(X, Y) 在数值上比 inv(X) @ Y 更稳健
    try:
        Phi = np.linalg.solve(X, Y)
    except np.linalg.LinAlgError:
        # 如果 X 奇异（例如在合成数据或某些极端频率点），返回空值
        return None, np.nan, np.nan, np.nan, np.nan, np.nan

    p11, p12 = Phi[0, 0], Phi[0, 1]
    p21, p22 = Phi[1, 0], Phi[1, 1]

    # 3. 计算坐标不变量 (Eq. 21-24)
    tr = p11 + p22  # trace
    sk = p12 - p21  # skew
    d2 = p11 - p22  # difference
    s2 = p12 + p21  # sum

    # 4. 计算偏斜角 beta (Eq. 29)
    # beta 衡量 3-D 效应，与坐标系无关
    beta = 0.5 * np.arctan2(sk, tr)

    # 5. 计算依赖角 alpha (Eq. 30)
    # alpha 衡量张量相对于坐标系的旋转
    alpha = 0.5 * np.arctan2(s2, d2)

    return Phi, np.degrees(alpha), np.degrees(beta)


@dataclass
class SinglePSM:
    """
    单个7x7功率谱矩阵

    存储和管理单个频率点的功率谱矩阵，遵循EDI标准格式：
    - 对角线: 自功率谱（实数）
    - 下三角: 互功率谱实部
    - 上三角: 互功率谱虚部

    矩阵布局 (7通道含远参考):
                  | Hx       | Hy       | Hz       | Ex       | Ey       | Rx       | Ry
        ----------|----------|----------|----------|----------|----------|----------|----------
        Hx (0)    | <HxH*x>  | Imag部分 | Imag部分 | Imag部分 | Imag部分 | Imag部分 | Imag部分
        Hy (1)    | Real部分 | <HyH*y>  | Imag部分 | Imag部分 | Imag部分 | Imag部分 | Imag部分
        Hz (2)    | Real部分 | Real部分 | <HzH*z>  | Imag部分 | Imag部分 | Imag部分 | Imag部分
        Ex (3)    | Real部分 | Real部分 | Real部分 | <ExE*x>  | Imag部分 | Imag部分 | Imag部分
        Ey (4)    | Real部分 | Real部分 | Real部分 | Real部分 | <EyE*y>  | Imag部分 | Imag部分
        Rx (5)    | Real部分 | Real部分 | Real部分 | Real部分 | Real部分 | <RxR*x>  | Imag部分
        Ry (6)    | Real部分 | Real部分 | Real部分 | Real部分 | Real部分 | Real部分 | <RyR*y>
    Attributes:
        freq: 频率 (Hz)
        spec_num: 原始频谱数量（用于方差计算）
        matrix: 7x7功率谱矩阵
    """

    freq: float  # 频率
    window_length: int  # fft时候的窗口长度
    seg_num: int  # 谱数量(分段数量)
    matrix: NDArray  # 功率谱数据 7*7

    def at(self, row: Component, col: Component) -> np.complex128:
        """
        获取矩阵元素对应的复数功率谱值

        Args:
            row: 行通道
            col: 列通道

        Returns:
            复数功率谱值 S(row, col)

        Note:
            S(i,j) = conj(S(j,i))，矩阵存储格式：
            - 对角线: S(ii) 实数
            - 下三角: Re(S(ji)) where j > i
            - 上三角: Im(S(ij)) where i < j
        """
        r = row.value
        c = col.value

        if r == c:
            # 对角线：自功率谱，虚部为0
            return np.complex128(self.matrix[r, c])
        elif r > c:
            # 下三角：实部在此，虚部在对应的上三角点取反
            # S(ji) = Re(S(ji)) + j*Im(S(ji)), 其中Im(S(ji)) = -Im(S(ij))
            re = self.matrix[r, c]
            im = self.matrix[c, r]
            return np.complex128(re + 1j * im)
        else:
            # 上三角：实部在对应的下三角点，虚部在此（取反）
            # S(ij) = Re(S(ji)) + j*Im(S(ij)), 其中Re(S(ji)) = Re(S(ij))
            re = self.matrix[c, r]
            im = -self.matrix[r, c]
            return np.complex128(re + 1j * im)

    def _validate_operand(self, other: object) -> bool:
        """验证操作数是否兼容"""
        if not isinstance(other, SinglePSM):
            return False
        # 容差：绝对 1e-6 Hz（适合低频），相对 0.01%（适合高频）
        FREQ_TOL = 1e-6
        FREQ_REL_TOL = 1e-4
        diff = abs(self.freq - other.freq)
        tol = max(FREQ_TOL, FREQ_REL_TOL * abs(self.freq))
        if diff >= tol:
            raise ValueError(
                f"Frequency mismatch: {self.freq} vs {other.freq} (tolerance: {tol})"
            )
        return True

    def __add__(self, other: "SinglePSM") -> "SinglePSM":
        """功率谱叠加"""
        if not isinstance(other, SinglePSM):
            return NotImplemented
        self._validate_operand(other)
        return SinglePSM(
            self.freq, self.window_length, self.seg_num, self.matrix + other.matrix
        )

    def __radd__(self, other: Union[int, "SinglePSM"]) -> "SinglePSM":
        """
        右加法，支持 sum() 从0开始累加

        当 sum([psm1, psm2, ...]) 时，Python 执行 0 + psm1，
        0 的 __add__ 失败，会回退到 psm1.__radd__(0)
        """
        if other == 0:
            return self
        if isinstance(other, SinglePSM):
            return self.__add__(other)
        return NotImplemented

    def __mul__(self, other: "SinglePSM") -> "SinglePSM":
        """逐元素相乘"""
        if not isinstance(other, SinglePSM):
            return NotImplemented
        self._validate_operand(other)
        return SinglePSM(
            self.freq, self.window_length, self.seg_num, self.matrix * other.matrix
        )

    def __truediv__(self, scalar: float) -> "SinglePSM":
        """标量除法（用于计算平均）"""
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Can only divide by scalar, not {type(scalar)}")
        return SinglePSM(
            freq=self.freq,
            window_length=self.window_length,
            seg_num=self.seg_num,
            matrix=self.matrix / scalar,
        )

    @classmethod
    def average(cls, psms: Sequence["SinglePSM"]) -> "SinglePSM":
        """
        计算平均值 - 修复类型推断问题

        Args:
            psms: SinglePSM 序列（列表或元组）
        """
        if not psms:
            raise ValueError("Cannot compute average of empty sequence")

        # 显式累加避免 sum() 的类型推断问题
        total = psms[0]
        for psm in psms[1:]:
            total += psm

        # 显式转换为 float 消除类型歧义
        return total / float(len(psms))

    def least_squares(self, ref_channel: RefChannel = RefChannel.H) -> "Impedance":
        """
        最小二乘法阻抗估计

        Args:
            ref_channel: 参考通道类型

        Returns:
            Impedance对象
        """
        # 根据参考通道选择不同的互功率谱组合
        if ref_channel == RefChannel.E:
            # E参考
            d = self.at(Component.Hx, Component.Ex) * self.at(
                Component.Hy, Component.Ey
            ) - self.at(Component.Hx, Component.Ey) * self.at(
                Component.Hy, Component.Ex
            )
            zxx = (
                self.at(Component.Ex, Component.Ex)
                * self.at(Component.Hy, Component.Ey)
                - self.at(Component.Ex, Component.Ey)
                * self.at(Component.Hy, Component.Ex)
            ) / d
            zxy = (
                self.at(Component.Ex, Component.Ey)
                * self.at(Component.Hx, Component.Ex)
                - self.at(Component.Ex, Component.Ex)
                * self.at(Component.Hx, Component.Ey)
            ) / d
            zyx = (
                self.at(Component.Ey, Component.Ex)
                * self.at(Component.Hy, Component.Ey)
                - self.at(Component.Ey, Component.Ey)
                * self.at(Component.Hy, Component.Ex)
            ) / d
            zyy = (
                self.at(Component.Ey, Component.Ey)
                * self.at(Component.Hx, Component.Ex)
                - self.at(Component.Ey, Component.Ex)
                * self.at(Component.Hx, Component.Ey)
            ) / d
        elif ref_channel == RefChannel.H:
            # H参考
            d = self.at(Component.Hx, Component.Hx) * self.at(
                Component.Hy, Component.Hy
            ) - self.at(Component.Hx, Component.Hy) * self.at(
                Component.Hy, Component.Hx
            )
            zxx = (
                self.at(Component.Ex, Component.Hx)
                * self.at(Component.Hy, Component.Hy)
                - self.at(Component.Ex, Component.Hy)
                * self.at(Component.Hy, Component.Hx)
            ) / d
            zxy = (
                self.at(Component.Ex, Component.Hy)
                * self.at(Component.Hx, Component.Hx)
                - self.at(Component.Ex, Component.Hx)
                * self.at(Component.Hx, Component.Hy)
            ) / d
            zyx = (
                self.at(Component.Ey, Component.Hx)
                * self.at(Component.Hy, Component.Hy)
                - self.at(Component.Ey, Component.Hy)
                * self.at(Component.Hy, Component.Hx)
            ) / d
            zyy = (
                self.at(Component.Ey, Component.Hy)
                * self.at(Component.Hx, Component.Hx)
                - self.at(Component.Ey, Component.Hx)
                * self.at(Component.Hx, Component.Hy)
            ) / d
        else:  # RefChannel.R
            # R参考
            d = self.at(Component.Hx, Component.Rx) * self.at(
                Component.Hy, Component.Ry
            ) - self.at(Component.Hx, Component.Ry) * self.at(
                Component.Hy, Component.Rx
            )
            zxx = (
                self.at(Component.Ex, Component.Rx)
                * self.at(Component.Hy, Component.Ry)
                - self.at(Component.Ex, Component.Ry)
                * self.at(Component.Hy, Component.Rx)
            ) / d
            zxy = (
                self.at(Component.Ex, Component.Ry)
                * self.at(Component.Hx, Component.Rx)
                - self.at(Component.Ex, Component.Rx)
                * self.at(Component.Hx, Component.Ry)
            ) / d
            zyx = (
                self.at(Component.Ey, Component.Rx)
                * self.at(Component.Hy, Component.Ry)
                - self.at(Component.Ey, Component.Ry)
                * self.at(Component.Hy, Component.Rx)
            ) / d
            zyy = (
                self.at(Component.Ey, Component.Ry)
                * self.at(Component.Hx, Component.Rx)
                - self.at(Component.Ey, Component.Rx)
                * self.at(Component.Hx, Component.Ry)
            ) / d

        # 计算倾子
        t_det = self.at(Component.Hx, Component.Hx) * self.at(
            Component.Hy, Component.Hy
        ) - self.at(Component.Hx, Component.Hy) * self.at(Component.Hy, Component.Hx)
        tzx = (
            self.at(Component.Hz, Component.Hx) * self.at(Component.Hy, Component.Hy)
            - self.at(Component.Hz, Component.Hy) * self.at(Component.Hy, Component.Hx)
        ) / t_det
        tzy = (
            self.at(Component.Hz, Component.Hy) * self.at(Component.Hx, Component.Hx)
            - self.at(Component.Hz, Component.Hx) * self.at(Component.Hx, Component.Hy)
        ) / t_det

        return Impedance(
            freq=self.freq,
            zxx=zxx,
            zxy=zxy,
            zyx=zyx,
            zyy=zyy,
            tzx=tzx,
            tzy=tzy,
        )


@dataclass
class PowerSpectrumMatrix:
    """功率谱矩阵数据"""

    freq: float  # 频率
    window_length: int  # fft时候的窗口长度
    seg_num: int  # 谱数量(分段数量)
    psms: tuple[SinglePSM, ...]  # 功率谱矩阵数据

    # ==================== 迭代器协议实现 ====================

    def __iter__(self) -> Iterator[SinglePSM]:
        """
        实现迭代器协议
        支持: for psm in matrix: ...
        """
        return iter(self.psms)

    def __len__(self) -> int:
        """
        返回功率谱矩阵段数
        支持: len(matrix)
        """
        return len(self.psms)

    def __getitem__(self, index: int | slice) -> SinglePSM | tuple[SinglePSM, ...]:
        """
        支持索引访问和切片
        支持:
            matrix[0]      # 获取第一个
            matrix[-1]     # 获取最后一个
            matrix[:5]     # 获取前5个
            matrix[1:10:2] # 切片
        """
        return self.psms[index]

    def __contains__(self, item: SinglePSM) -> bool:
        """
        支持成员检查
        支持: if psm in matrix: ...
        """
        return item in self.psms

    def over(self, num: int = 0) -> "PowerSpectrumMatrix":
        """
        叠加功率谱矩阵
        :param num: 叠加后谱的数量
        :return: 叠加后的功率谱矩阵
        """
        item_num = len(self.psms)
        if item_num == 0:
            return self

        # 计算分组的谱数量
        if num > 0 and item_num > 1 and item_num > num * 2:
            group_size = item_num // num
        else:
            group_size = 2
            num = item_num // group_size

        group_size = max(group_size, 2)
        reminder = item_num - (group_size * num)

        result = []
        cursor = 0
        for i in range(num):
            end = cursor + group_size + (1 if i < reminder else 0)
            end = min(end, item_num)

            if cursor >= end:
                break

            if end - cursor > 1:
                t_item = self.psms[cursor]
                for item in self.psms[cursor + 1 : end]:
                    t_item += item
                result.append(t_item / (end - cursor))
            else:
                result.append(self.psms[cursor])
            cursor = end

        return PowerSpectrumMatrix(
            freq=self.freq,
            window_length=self.window_length,
            seg_num=self.seg_num,
            psms=tuple(result),
        )

    def to_matrix(self) -> NDArray:
        """
        将功率谱矩阵转换为numpy数组

        Returns:
            numpy数组 (num, 7, 7)
        """
        return np.array([psm.matrix for psm in self.psms])

    def binary_mask(self, mask: list[int]) -> "PowerSpectrumMatrix":
        """
        根据掩码筛选功率谱并计算平均

        Args:
            mask: 掩码列表，1表示保留，0表示丢弃

        Returns:
            筛选后的平均PowerSpectralMatrix
        """
        if len(mask) != len(self.psms):
            raise ValueError(f"掩码长度({len(mask)})与谱数量({len(self.psms)})不匹配")

        if all(mask[i] == 0 for i in range(len(mask))):
            raise ValueError("mask内值全为0")

        selected_psm: list[SinglePSM] = []
        for i, psm in enumerate(self.psms):
            if i < len(mask) and mask[i] == 1:
                selected_psm.append(self.psms[i])

        return PowerSpectrumMatrix(
            freq=self.freq,
            window_length=self.window_length,
            seg_num=self.seg_num,
            psms=tuple(selected_psm),
        )

    def to_spsm(self) -> SinglePSM:
        """前当前所有功率谱矩阵的平均功率谱矩阵"""
        return SinglePSM.average(self.psms)

    def weighting(self, w: np.ndarray | list[float]) -> SinglePSM:
        """对功率谱矩阵进行加权，权重 w 的和为 1（理论上不为1也行应为除法上下同时缩放不影响结果，权重数量必须和谱数量一致"""
        if len(w) != len(self.psms):
            raise ValueError(f"权重长度({len(w)})与谱数量({len(self.psms)})不匹配")
        w = np.asarray(w)
        matrixs = np.stack([spsm.matrix for spsm in self.psms], axis=0)
        m_w = np.sum(matrixs * w[:, None, None], axis=0)
        return SinglePSM(
            freq=self.freq,
            window_length=self.window_length,
            seg_num=self.seg_num,
            matrix=m_w,
        )


@dataclass
class Impedance:
    freq: float  # 频率
    zxx: complex
    zxy: complex
    zyx: complex
    zyy: complex

    tzx: complex
    tzy: complex

    def to_rho_phi(self) -> "ResistivityPhase":
        return ResistivityPhase(
            freq=self.freq,
            rxx=calc_resistivity(self.freq, self.zxx),
            rxy=calc_resistivity(self.freq, self.zxy),
            ryx=calc_resistivity(self.freq, self.zyx),
            ryy=calc_resistivity(self.freq, self.zyy),
            pxx=calc_phase_deg(self.zxx),
            pxy=calc_phase_deg(self.zxy),
            pyx=calc_phase_deg(self.zyx),
            pyy=calc_phase_deg(self.zyy),
        )


@dataclass
class ResistivityPhase:
    """电阻率和相位角"""

    freq: float  # 频率
    rxx: float
    rxy: float
    ryx: float
    ryy: float
    pxx: float
    pxy: float
    pyx: float
    pyy: float
