from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator, Sequence, Union

import numpy as np
from numpy.typing import NDArray


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


class RefChannel(IntEnum):
    """
    Reference channel enumeration

    Defines the reference channel types available for impedance calculation:
    - E: electric field reference (electric-field reference method)
    - H: magnetic field reference (standard method, default)
    - R: remote reference (used in remote-reference processing)
    """

    E = 0
    H = 1
    R = 2


def calc_resistivity(freq: float, z: complex) -> float:
    """
    Compute apparent resistivity

    Formula: ρ = |Z|² / (ωμ) = |Z|² * 0.2 / f
    where μ = 4π×10⁻⁷ H/m

    Args:
        freq: frequency (Hz)
        z: impedance value

    Returns:
        apparent resistivity (Ω·m)
    """
    return abs(z) ** 2 * 0.2 / freq


def calc_phase_deg(z: complex) -> float:
    """
    Compute phase (degrees)

    Args:
        z: impedance value

    Returns:
        phase angle (degrees)
    """
    return np.angle(z, deg=True)


def calc_power_density(psm: "SinglePSM", c1: Component, c2: Component) -> float:
    """
    Compute power density

    Formula: P = |S(c1, c2)| / window_length

    Args:
        psm: power spectral matrix
        c1: first channel index
        c2: second channel index

    Returns:
        power density
    """

    return abs(psm.at(c1, c2)) / psm.window_length


def calc_phase_tensor(zxx: complex, zxy: complex, zyx: complex, zyy: complex):
    """
    Magnetotelluric phase tensor calculation following Caldwell et al. (2004).

    Returns:
        Phi: 2x2 phase tensor matrix
        alpha: coordinate-dependent angle (degrees)
        beta: skew angle (degrees)
    """
    # 1. Build impedance tensor Z and split into real X and imaginary Y (Eq. 18)
    # Z = X + iY
    Z = np.array([[zxx, zxy], [zyx, zyy]])
    X = Z.real
    Y = Z.imag

    # 2. Compute phase tensor Phi = X^-1 * Y (Eq. 18)
    # solve(X, Y) is numerically more robust than inv(X) @ Y
    try:
        Phi = np.linalg.solve(X, Y)
    except np.linalg.LinAlgError:
        # If X is singular (e.g. on synthetic data or at some extreme frequencies), return nulls
        return None, np.nan, np.nan, np.nan, np.nan, np.nan

    p11, p12 = Phi[0, 0], Phi[0, 1]
    p21, p22 = Phi[1, 0], Phi[1, 1]

    # 3. Compute coordinate invariants (Eq. 21-24)
    tr = p11 + p22  # trace
    sk = p12 - p21  # skew
    d2 = p11 - p22  # difference
    s2 = p12 + p21  # sum

    # 4. Compute skew angle beta (Eq. 29)
    # beta measures 3-D effects and is coordinate-independent
    beta = 0.5 * np.arctan2(sk, tr)

    # 5. Compute dependent angle alpha (Eq. 30)
    # alpha measures rotation of the tensor relative to the coordinate system
    alpha = 0.5 * np.arctan2(s2, d2)

    return Phi, np.degrees(alpha), np.degrees(beta)


@dataclass
class SinglePSM:
    """
    Single 7x7 power spectral matrix

    Stores and manages the power spectral matrix at a single frequency, following the
    EDI standard format:
    - Diagonal: auto-power spectra (real)
    - Lower triangle: real part of cross-power spectra
    - Upper triangle: imaginary part of cross-power spectra

    Matrix layout (7 channels including remote reference):
                  | Hx       | Hy       | Hz       | Ex       | Ey       | Rx       | Ry
        ----------|----------|----------|----------|----------|----------|----------|----------
        Hx (0)    | <HxH*x>  | Imag     | Imag     | Imag     | Imag     | Imag     | Imag
        Hy (1)    | Real     | <HyH*y>  | Imag     | Imag     | Imag     | Imag     | Imag
        Hz (2)    | Real     | Real     | <HzH*z>  | Imag     | Imag     | Imag     | Imag
        Ex (3)    | Real     | Real     | Real     | <ExE*x>  | Imag     | Imag     | Imag
        Ey (4)    | Real     | Real     | Real     | Real     | <EyE*y>  | Imag     | Imag
        Rx (5)    | Real     | Real     | Real     | Real     | Real     | <RxR*x>  | Imag
        Ry (6)    | Real     | Real     | Real     | Real     | Real     | Real     | <RyR*y>
    Attributes:
        freq: frequency (Hz)
        spec_num: number of original spectra (used for variance calculation)
        matrix: 7x7 power spectral matrix
    """

    freq: float  # frequency
    window_length: int  # FFT window length
    seg_num: int  # number of spectra (segments)
    matrix: NDArray  # power spectral data 7*7

    def at(self, row: Component, col: Component) -> np.complex128:
        """
        Retrieve the complex power spectral value for a matrix element

        Args:
            row: row channel
            col: column channel

        Returns:
            complex power spectral value S(row, col)

        Note:
            S(i,j) = conj(S(j,i)); storage layout:
            - Diagonal: S(ii) real
            - Lower triangle: Re(S(ji)) where j > i
            - Upper triangle: Im(S(ij)) where i < j
        """
        r = row.value
        c = col.value

        if r == c:
            # Diagonal: auto-power spectrum, imaginary part is 0
            return np.complex128(self.matrix[r, c])
        elif r > c:
            # Lower triangle: real part here; imaginary part is the negated upper-triangle entry
            # S(ji) = Re(S(ji)) + j*Im(S(ji)), where Im(S(ji)) = -Im(S(ij))
            re = self.matrix[r, c]
            im = self.matrix[c, r]
            return np.complex128(re + 1j * im)
        else:
            # Upper triangle: real part is in the corresponding lower-triangle entry;
            # imaginary part is here (negated)
            # S(ij) = Re(S(ji)) + j*Im(S(ij)), where Re(S(ji)) = Re(S(ij))
            re = self.matrix[c, r]
            im = -self.matrix[r, c]
            return np.complex128(re + 1j * im)

    def _validate_operand(self, other: object) -> bool:
        """Validate that the operand is compatible"""
        if not isinstance(other, SinglePSM):
            return False
        # Tolerance: absolute 1e-6 Hz (suited to low frequencies), relative 0.01% (suited to high frequencies)
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
        """Power spectral stacking"""
        if not isinstance(other, SinglePSM):
            return NotImplemented
        self._validate_operand(other)
        return SinglePSM(
            self.freq, self.window_length, self.seg_num, self.matrix + other.matrix
        )

    def __radd__(self, other: Union[int, "SinglePSM"]) -> "SinglePSM":
        """
        Right addition, supporting sum() accumulation from 0

        For sum([psm1, psm2, ...]), Python evaluates 0 + psm1; 0's __add__ fails
        and falls back to psm1.__radd__(0)
        """
        if other == 0:
            return self
        if isinstance(other, SinglePSM):
            return self.__add__(other)
        return NotImplemented

    def __mul__(self, other: "SinglePSM") -> "SinglePSM":
        """Element-wise multiplication"""
        if not isinstance(other, SinglePSM):
            return NotImplemented
        self._validate_operand(other)
        return SinglePSM(
            self.freq, self.window_length, self.seg_num, self.matrix * other.matrix
        )

    def __truediv__(self, scalar: float) -> "SinglePSM":
        """Scalar division (used for averaging)"""
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
        Compute the average — fixes type-inference issues

        Args:
            psms: sequence of SinglePSM (list or tuple)
        """
        if not psms:
            raise ValueError("Cannot compute average of empty sequence")

        # Explicit accumulation avoids sum() type-inference issues
        total = psms[0]
        for psm in psms[1:]:
            total += psm

        # Explicit float conversion removes type ambiguity
        return total / float(len(psms))

    def least_squares(self, ref_channel: RefChannel = RefChannel.H) -> "Impedance":
        """
        Least-squares impedance estimation

        Args:
            ref_channel: reference channel type

        Returns:
            Impedance object
        """
        # Select the cross-power spectral combination based on the reference channel
        if ref_channel == RefChannel.E:
            # E reference
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
            # H reference
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
            # R reference
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

        # Compute the tipper
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
    """Power spectral matrix data"""

    freq: float  # frequency
    window_length: int  # FFT window length
    seg_num: int  # number of spectra (segments)
    psms: tuple[SinglePSM, ...]  # power spectral matrix data

    # ==================== Iterator protocol ====================

    def __iter__(self) -> Iterator[SinglePSM]:
        """
        Implement the iterator protocol
        Supports: for psm in matrix: ...
        """
        return iter(self.psms)

    def __len__(self) -> int:
        """
        Return the number of power spectral segments
        Supports: len(matrix)
        """
        return len(self.psms)

    def __getitem__(self, index: int | slice) -> SinglePSM | tuple[SinglePSM, ...]:
        """
        Support indexing and slicing
        Supports:
            matrix[0]      # first
            matrix[-1]     # last
            matrix[:5]     # first 5
            matrix[1:10:2] # slice
        """
        return self.psms[index]

    def __contains__(self, item: SinglePSM) -> bool:
        """
        Support membership tests
        Supports: if psm in matrix: ...
        """
        return item in self.psms

    def over(self, num: int = 0) -> "PowerSpectrumMatrix":
        """
        Stack (average) power spectral matrices
        :param num: number of spectra after stacking
        :return: stacked power spectral matrix
        """
        item_num = len(self.psms)
        if item_num == 0:
            return self

        # Compute the group size
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
        Convert the power spectral matrices to a numpy array

        Returns:
            numpy array (num, 7, 7)
        """
        return np.array([psm.matrix for psm in self.psms])

    def binary_mask(self, mask: list[int]) -> "PowerSpectrumMatrix":
        """
        Filter power spectra by a mask and compute the average

        Args:
            mask: mask list; 1 keeps a spectrum, 0 discards it

        Returns:
            filtered averaged PowerSpectralMatrix
        """
        if len(mask) != len(self.psms):
            raise ValueError(
                f"Mask length ({len(mask)}) does not match the number of spectra ({len(self.psms)})"
            )

        if all(mask[i] == 0 for i in range(len(mask))):
            raise ValueError("All values in the mask are 0")

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
        """Average power spectral matrix of all current power spectral matrices"""
        return SinglePSM.average(self.psms)

    def weighting(self, w: np.ndarray | list[float]) -> SinglePSM:
        """Weight the power spectral matrices; weights w need not sum to 1 (numerator and denominator
        scale together, so the result is unchanged). The number of weights must match the number of spectra."""
        if len(w) != len(self.psms):
            raise ValueError(
                f"Weight length ({len(w)}) does not match the number of spectra ({len(self.psms)})"
            )
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
    freq: float  # frequency
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
    """Resistivity and phase angles"""

    freq: float  # frequency
    rxx: float
    rxy: float
    ryx: float
    ryy: float
    pxx: float
    pxy: float
    pyx: float
    pyy: float
