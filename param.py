import math
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .struct import (
    Component,
    PowerSpectrumMatrix,
    calc_phase_tensor,
    calc_power_density,
)


def robust_scaler(arr, is_log=False):
    valid = np.asarray(arr)
    if is_log:
        valid = np.log10(np.abs(valid) + 1e-10)

    median = np.median(valid, axis=0, keepdims=True)
    q75, q25 = np.percentile(valid, [75, 25], axis=0, keepdims=True)
    iqr = q75 - q25
    # Prevent division by zero: replace a too-small IQR with 1.0 (or a tiny value)
    eps = 1e-8
    iqr_safe = np.where(np.abs(iqr) < eps, 1e-8, iqr)
    scaled = (valid - median) / iqr_safe
    # Return the scaled valid data
    return scaled


def _decompose_complex(z: NDArray[np.complex128], is_log: bool = True):
    """Decompose a complex array into scaled amplitude, sin(phase), and cos(phase) components"""
    amp = np.abs(z)
    amp_scaled = robust_scaler(amp, is_log=is_log)
    phs = np.angle(z)
    return amp_scaled, np.sin(phs), np.cos(phs)


@dataclass
class Params:
    """
    Multi-parameter container
    """

    zxx: NDArray[np.complex128] = field(default_factory=list)  # impedance
    zyy: NDArray[np.complex128] = field(default_factory=list)  # impedance
    zxy: NDArray[np.complex128] = field(default_factory=list)  # impedance
    zyx: NDArray[np.complex128] = field(default_factory=list)  # impedance
    tzx: NDArray[np.complex128] = field(default_factory=list)  # tipper
    tzy: NDArray[np.complex128] = field(default_factory=list)  # tipper
    ex: NDArray[np.float64] = field(default_factory=list)  # auto-power spectral density
    ey: NDArray[np.float64] = field(default_factory=list)  # auto-power spectral density
    hx: NDArray[np.float64] = field(default_factory=list)  # auto-power spectral density
    hy: NDArray[np.float64] = field(default_factory=list)  # auto-power spectral density

    @classmethod
    def from_psms(cls, psm: PowerSpectrumMatrix) -> "Params":
        imps = [p.least_squares() for p in psm]
        zxx = np.array([imp.zxx for imp in imps])
        zyy = np.array([imp.zyy for imp in imps])
        zxy = np.array([imp.zxy for imp in imps])
        zyx = np.array([imp.zyx for imp in imps])
        tzx = np.array([imp.tzx for imp in imps])
        tzy = np.array([imp.tzy for imp in imps])
        ex = np.array([calc_power_density(s, Component.Ex, Component.Ex) for s in psm])
        ey = np.array([calc_power_density(s, Component.Ey, Component.Ey) for s in psm])
        hx = np.array([calc_power_density(s, Component.Hx, Component.Hx) for s in psm])
        hy = np.array([calc_power_density(s, Component.Hy, Component.Hy) for s in psm])

        return cls(
            zxx=zxx,
            zyy=zyy,
            zxy=zxy,
            zyx=zyx,
            tzx=tzx,
            tzy=tzy,
            ex=ex,
            ey=ey,
            hx=hx,
            hy=hy,
        )

    def to_matrix(self) -> np.ndarray:
        return np.stack(
            [
                np.real(self.zxx),
                np.imag(self.zxx),
                np.real(self.zyy),
                np.imag(self.zyy),
                np.real(self.zxy),
                np.imag(self.zxy),
                np.real(self.zyx),
                np.imag(self.zyx),
                np.real(self.tzx),
                np.imag(self.tzx),
                np.real(self.tzy),
                np.imag(self.tzy),
                self.ex,
                self.ey,
                self.hx,
                self.hy,
            ],
            axis=1,
        )

    def to_features(
        self,
        use_impedance: bool = True,
        use_tipper: bool = True,
        use_psd: bool = True,
        use_phase_tensor_angles: bool = True,
        use_phase_tensor_main: bool = True,
    ) -> np.ndarray:
        """Add phase-tensor features; support selective feature output by physical module

        Parameters
        ----------
        use_impedance : impedance tensor (Zxx, Zyy, Zxy, Zyx) → 12 dimensions
        use_tipper : tipper (Tzx, Tzy) → 6 dimensions
        use_psd : power spectral density (Ex, Ey, Hx, Hy) → 4 dimensions
        use_phase_tensor_angles : phase tensor angles (Alpha, Beta) → 4 dimensions
        use_phase_tensor_main : phase tensor principal values (P11,P12,P21,P22) → 4 dimensions
        """
        arrays_to_stack = []

        # --- Impedance tensor (12 dims) ---
        if use_impedance:
            for z in (self.zxx, self.zyy, self.zxy, self.zyx):
                amp, sin_, cos_ = _decompose_complex(z, is_log=True)
                arrays_to_stack.extend([amp, sin_, cos_])

        # --- Tipper (6 dims) ---
        if use_tipper:
            for z in (self.tzx, self.tzy):
                amp, sin_, cos_ = _decompose_complex(z, is_log=True)
                arrays_to_stack.extend([amp, sin_, cos_])

        # --- Power spectral density (4 dims) ---
        if use_psd:
            for arr in (self.ex, self.ey, self.hx, self.hy):
                arrays_to_stack.append(robust_scaler(arr, is_log=True))

        # --- Phase tensor (angles 4 dims + main 4 dims, sharing calc_phase_tensor) ---
        if use_phase_tensor_angles or use_phase_tensor_main:
            p11s, p12s, p21s, p22s = [], [], [], []
            alphas, betas = [], []
            for i in range(self.zxx.shape[0]):
                phase_tensor, alpha, beta = calc_phase_tensor(
                    zxx=self.zxx[i], zxy=self.zxy[i], zyx=self.zyx[i], zyy=self.zyy[i]
                )
                p11s.append(phase_tensor[0, 0])
                p12s.append(phase_tensor[0, 1])
                p21s.append(phase_tensor[1, 0])
                p22s.append(phase_tensor[1, 1])
                alphas.append(alpha)
                betas.append(beta)

            if use_phase_tensor_angles:
                alphas = np.array(alphas)
                betas = np.array(betas)
                arrays_to_stack.extend(
                    [
                        np.sin(alphas),
                        np.cos(alphas),
                        np.sin(betas),
                        np.cos(betas),
                    ]
                )

            if use_phase_tensor_main:
                arrays_to_stack.extend(
                    [
                        robust_scaler(np.array(p11s), is_log=True),
                        robust_scaler(np.array(p12s), is_log=True),
                        robust_scaler(np.array(p21s), is_log=True),
                        robust_scaler(np.array(p22s), is_log=True),
                    ]
                )

        return np.stack(arrays_to_stack, axis=1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

RAW_LABELS = [
    "Zxx Real",
    "Zxx Imag",
    "Zyy Real",
    "Zyy Imag",
    "Zxy Real",
    "Zxy Imag",
    "Zyx Real",
    "Zyx Imag",
    "Tzx Real",
    "Tzx Imag",
    "Tzy Real",
    "Tzy Imag",
    "Ex PSD",
    "Ey PSD",
    "Hx PSD",
    "Hy PSD",
]

FEATURE_LABELS = [
    "Zxx Amp (Scaled)",
    "Zxx Sin",
    "Zxx Cos",
    "Zyy Amp (Scaled)",
    "Zyy Sin",
    "Zyy Cos",
    "Zxy Amp (Scaled)",
    "Zxy Sin",
    "Zxy Cos",
    "Zyx Amp (Scaled)",
    "Zyx Sin",
    "Zyx Cos",
    "Tzx Amp (Scaled)",
    "Tzx Sin",
    "Tzx Cos",
    "Tzy Amp (Scaled)",
    "Tzy Sin",
    "Tzy Cos",
    "Ex PSD (Scaled)",
    "Ey PSD (Scaled)",
    "Hx PSD (Scaled)",
    "Hy PSD (Scaled)",
    "Alpha Sin",
    "Alpha Cos",
    "Beta Sin",
    "Beta Cos",
    "Phase Tensor P11",
    "Phase Tensor P12",
    "Phase Tensor P21",
    "Phase Tensor P22",
]

RAW_LOG_INDICES = {12, 13, 14, 15}  # PSD columns use a log scale


def _plot_grid(
    matrix: np.ndarray,
    labels: list[str],
    log_indices: set[int] | None = None,
    title: str = "",
):
    """Generic feature-grid plot"""
    if log_indices is None:
        log_indices = set()

    n_features = matrix.shape[1]
    n_cols = 4
    n_rows = math.ceil(n_features / n_cols)
    last_row_start = (n_rows - 1) * n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True
    )
    axes = np.array(axes).flatten()

    cmap = plt.get_cmap("tab20")
    for i in range(n_features):
        color = cmap(i % 20)
        axes[i].plot(
            matrix[:, i], linestyle="--", color=color, alpha=0.5, linewidth=1.0
        )
        axes[i].scatter(
            range(len(matrix)),
            matrix[:, i],
            marker="o",
            s=25,
            color=color,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.7,
            label=labels[i],
        )
        axes[i].legend(loc="upper right", fontsize="small", framealpha=0.8)
        axes[i].grid(True, linestyle="--", alpha=0.7)

        if i in log_indices:
            axes[i].set_yscale("log")

        if i < last_row_start:
            axes[i].tick_params(labelbottom=False)

    # Remove unused empty subplots
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])

    if title:
        fig.suptitle(title, fontsize=14)
    fig.supxlabel("Samples")
    fig.supylabel("Amplitude")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.show()


def plot_raw(params: Params, title: str = ""):
    """Plot the raw data from Params.to_matrix()"""
    _plot_grid(params.to_matrix(), RAW_LABELS, log_indices=RAW_LOG_INDICES, title=title)


def plot_features(features: np.ndarray, title: str = ""):
    """Plot features after to_deal_two() processing"""
    _plot_grid(features, FEATURE_LABELS, title=title)


if __name__ == "__main__":
    # Demo only: set these paths to your local MT archive before running.
    from pathlib import Path

    from mt_py_lib import MTCal, MTFreqGroup, Spectrum, TimeSeries

    temp_dir = Path(r"E:\path\to\temp_dir")
    tbl = Path(r"E:\path\to\YB006A.TBL")

    ts3 = tbl.with_suffix(".TS3")
    ts4 = tbl.with_suffix(".TS4")
    ts5 = tbl.with_suffix(".TS5")

    t3 = TimeSeries.from_file(ts3)
    t4 = TimeSeries.from_file(ts4)
    t5 = TimeSeries.from_file(ts5)

    t3.freq_groups = MTFreqGroup.new_ts3().default()
    t4.freq_groups = MTFreqGroup.new_ts4().default()
    t5.freq_groups = MTFreqGroup.new_ts5().default_precise()

    cts_file = temp_dir / f"{tbl.stem}.cts"
    mt_cal = MTCal.from_cts_file(cts_file)

    specs: list[Spectrum] = []
    specs.extend(t3.to_spec())
    specs.extend(t4.to_spec())
    specs.extend(t5.to_spec())

    for s in specs:
        s.calibrate(mt_cal)

    psms = [s.to_psm().over(500) for s in specs]

    params = {}

    for psm in psms:
        print(psm.freq)
        param = Params.from_psms(psm)
        features = param.to_features()

        params[psm.freq] = features

        plot_raw(param, title=f"Raw — Freq={psm.freq}")
        plot_features(features, title=f"Features — Freq={psm.freq}")
