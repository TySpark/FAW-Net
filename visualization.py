"""MT denoise result visualization module.

Provides both single-frequency and all-frequency visualization granularities,
for use together with mt_py_lib.visualization.draw_rho_phi.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .param import FEATURE_LABELS
from .struct import PowerSpectrumMatrix, SinglePSM

# ---------------------------------------------------------------------------
# Helpers: index list[SinglePSM] / list[PowerSpectrumMatrix] by frequency
# ---------------------------------------------------------------------------


def _find_psm(psm_list: list, freq: float):
    """Find the PSM object at the specified frequency in the list."""
    for p in psm_list:
        if p.freq == freq:
            return p
    raise KeyError(f"Frequency {freq} not found in PSM list")


# ---------------------------------------------------------------------------
# Single-frequency visualization
# ---------------------------------------------------------------------------


def plot_single_freq_weights(
    res_w: dict[float, np.ndarray],
    freq: float,
    title: str = "",
    show: bool = True,
):
    """Single-frequency weight distribution bar chart.

    Each bar represents the weight of one PSM sample. Colors map the weight
    magnitude; mean and max reference lines highlight which samples the model
    trusts most.
    """
    weights = res_w[freq]
    n = len(weights)
    indices = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, 4))

    # Color map: higher weight → warmer color
    norm = plt.Normalize(vmin=weights.min(), vmax=weights.max())
    cmap = plt.get_cmap("RdYlGn")
    colors = cmap(norm(weights))

    bars = ax.bar(
        indices, weights, color=colors, edgecolor="black", linewidth=0.3, width=1.0
    )

    # Mean and max reference lines
    mean_w = np.mean(weights)
    max_w = np.max(weights)
    ax.axhline(
        mean_w,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {mean_w:.4f}",
    )
    ax.axhline(
        max_w, color="#d62728", linestyle=":", linewidth=1.5, label=f"Max = {max_w:.4f}"
    )

    ax.set_xlabel("Sample Index")
    ax.set_yscale("log")
    ax.set_ylabel("Weight (log)")
    ax.set_xlim(-0.5, n - 0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", which="both")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title(
            f"Weight Distribution — Freq = {freq} Hz", fontsize=12, fontweight="bold"
        )

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_single_freq_impedance(
    single_psms: list[SinglePSM],
    psms: list[PowerSpectrumMatrix],
    freq: float,
    title: str = "",
    show: bool = True,
):
    """Single-frequency impedance comparison before/after processing.

    Grouped bar chart of the real/imaginary impedance component amplitudes for
    the raw PSM and the weighted SinglePSM, showing which components the model
    preserves and which it suppresses.
    """
    raw = _find_psm(psms, freq)
    weighted = _find_psm(single_psms, freq)

    raw_imp = raw.to_spsm().least_squares()
    wt_imp = weighted.least_squares()

    components = ["Zxx", "Zyy", "Zxy", "Zyx", "Tzx", "Tzy"]
    raw_vals = [
        raw_imp.zxx,
        raw_imp.zyy,
        raw_imp.zxy,
        raw_imp.zyx,
        raw_imp.tzx,
        raw_imp.tzy,
    ]
    wt_vals = [
        wt_imp.zxx,
        wt_imp.zyy,
        wt_imp.zxy,
        wt_imp.zyx,
        wt_imp.tzx,
        wt_imp.tzy,
    ]

    # Use amplitudes for comparison
    raw_amp = np.abs(np.array(raw_vals))
    wt_amp = np.abs(np.array(wt_vals))

    x = np.arange(len(components))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left panel: amplitude comparison
    ax = axes[0]
    ax.bar(
        x - width / 2,
        raw_amp,
        width,
        label="Raw",
        color="#1f77b4",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.bar(
        x + width / 2,
        wt_amp,
        width,
        label="Weighted",
        color="#2ca02c",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.set_ylabel("|Component|")
    ax.set_title("Impedance Amplitude", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Right panel: phase comparison
    raw_phase = np.angle(np.array(raw_vals), deg=True)
    wt_phase = np.angle(np.array(wt_vals), deg=True)

    ax = axes[1]
    ax.bar(
        x - width / 2,
        raw_phase,
        width,
        label="Raw",
        color="#1f77b4",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.bar(
        x + width / 2,
        wt_phase,
        width,
        label="Weighted",
        color="#2ca02c",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.set_ylabel("Phase (°)")
    ax.set_title("Impedance Phase", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    else:
        fig.suptitle(
            f"Raw vs Weighted Impedance — Freq = {freq} Hz",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    if show:
        plt.show()
    return fig, axes


def plot_single_freq_features(
    out_params: dict[float, np.ndarray],
    res_w: dict[float, np.ndarray],
    freq: float,
    title: str = "",
    show: bool = True,
):
    """Single-frequency input-feature heatmap.

    Sorts out_params[freq] (shape: psm_num × n_features) by weight in
    descending order and plots a heatmap. High-weight samples appear at the
    top, revealing the feature patterns the model focuses on.
    """
    features = out_params[freq]  # (psm_num, n_features)
    weights = res_w[freq]

    # Sort by weight in descending order
    order = np.argsort(weights)[::-1]
    features_sorted = features[order]
    weights_sorted = weights[order]

    n_features = features_sorted.shape[1]
    labels = (
        FEATURE_LABELS[:n_features]
        if n_features <= len(FEATURE_LABELS)
        else [f"F{i}" for i in range(n_features)]
    )

    fig, ax = plt.subplots(figsize=(12, max(6, len(features_sorted) * 0.15)))

    im = ax.imshow(
        features_sorted, aspect="auto", cmap="RdBu_r", interpolation="nearest"
    )

    ax.set_yticks(range(len(features_sorted)))
    # Label only high-weight sample indices
    tick_labels = []
    for i, w in enumerate(weights_sorted):
        if i % max(1, len(features_sorted) // 20) == 0:
            tick_labels.append(f"{order[i]} (w={w:.3f})")
        else:
            tick_labels.append("")
    ax.set_yticklabels(tick_labels, fontsize=7)

    ax.set_xticks(range(n_features))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Sample (sorted by weight ↓)")
    ax.set_title(
        title or f"Input Features (sorted by weight) — Freq = {freq} Hz",
        fontsize=11,
        fontweight="bold",
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Feature Value", fontsize=9)

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# All-frequency visualization
# ---------------------------------------------------------------------------


def plot_weights_heatmap(
    res_w: dict[float, np.ndarray],
    title: str = "",
    show: bool = True,
):
    """Weight heatmap across all frequencies.

    x = sample index, y = frequency (low frequency at top), color = weight.
    """
    freqs = sorted(res_w.keys())  # low frequency on top, high frequency at bottom
    n_freqs = len(freqs)
    n_samples = max(len(res_w[f]) for f in freqs)

    # Build the matrix; pad missing entries with NaN
    matrix = np.full((n_freqs, n_samples), np.nan)
    for i, f in enumerate(freqs):
        w = res_w[f]
        matrix[i, : len(w)] = w

    fig, ax = plt.subplots(figsize=(12, max(4, n_freqs * 0.3)))

    # Plot on a log scale (weight values span a wide range)
    matrix_log = np.where(np.isnan(matrix), np.nan, np.log10(np.maximum(matrix, 1e-10)))
    vmin = np.nanmin(matrix_log)
    vmax = np.nanmax(matrix_log)
    norm = LogNorm(vmin=10**vmin, vmax=10**vmax) if vmin != vmax else None

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdYlGn",
        interpolation="nearest",
        norm=norm,
        origin="upper",
    )

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_yticks(range(n_freqs))
    ax.set_yticklabels([f"{f}" for f in freqs], fontsize=7)

    fig.colorbar(im, ax=ax, label="Weight (log scale)", shrink=0.8)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title("Weight Heatmap — All Frequencies", fontsize=12, fontweight="bold")

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_weight_curve(
    res_w: dict[float, np.ndarray],
    title: str = "",
    show: bool = True,
):
    """Weight trend versus frequency.

    Each translucent line is one sample; the bold lines show mean and median.
    """
    freqs = sorted(res_w.keys())
    n_samples = max(len(res_w[f]) for f in freqs)

    # Build an (n_samples, n_freqs) matrix
    matrix = np.full((n_samples, len(freqs)), np.nan)
    for j, f in enumerate(freqs):
        w = res_w[f]
        matrix[: len(w), j] = w

    fig, ax = plt.subplots(figsize=(10, 5))

    # One line per sample
    for i in range(n_samples):
        ax.plot(freqs, matrix[i], color="gray", alpha=0.15, linewidth=0.5)

    # Mean and median
    mean_curve = np.nanmean(matrix, axis=0)
    median_curve = np.nanmedian(matrix, axis=0)
    ax.plot(freqs, mean_curve, color="#d62728", linewidth=2, label="Mean")
    ax.plot(
        freqs,
        median_curve,
        color="#1f77b4",
        linewidth=2,
        linestyle="--",
        label="Median",
    )

    ax.set_xscale("log")
    ax.invert_xaxis()  # high frequency on the left
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Weight (log)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", which="both")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title("Weight vs Frequency", fontsize=12, fontweight="bold")

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_before_after_rho_phi(
    single_psms: list[SinglePSM],
    psms: list[PowerSpectrumMatrix],
    name: str = "Denoised",
    show: bool = True,
):
    """Resistivity/phase comparison before and after processing.

    Raw data uses thin dashed lines with low opacity (background);
    weighted data uses thick solid lines at full opacity (highlighted);
    the frequency axis has high frequencies on the left.
    """
    # Pair raw and weighted entries by frequency
    freq_map_raw = {p.freq: p for p in psms}
    freq_map_wt = {p.freq: p for p in single_psms}
    common_freqs = sorted(set(freq_map_raw) & set(freq_map_wt))

    raw_rps = [
        freq_map_raw[f].to_spsm().least_squares().to_rho_phi() for f in common_freqs
    ]
    wt_rps = [freq_map_wt[f].least_squares().to_rho_phi() for f in common_freqs]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    ax11, ax12, ax21, ax22 = axes.flatten()

    # Color scheme: raw uses light tones, weighted uses the same color but more prominent
    comp_colors = {
        "rxx": "#1f77b4",
        "ryy": "#ff7f0e",
        "rxy": "#2ca02c",
        "ryx": "#d62728",
        "pxx": "#9467bd",
        "pyy": "#8c564b",
        "pxy": "#e377c2",
        "pyx": "#7f7f7f",
    }

    # --- Plot raw: thin dashed line + low opacity + no marker (background reference) ---
    for comp, ax in [
        ("rxx", ax11),
        ("ryy", ax11),
        ("rxy", ax12),
        ("ryx", ax12),
        ("pxx", ax21),
        ("pyy", ax21),
        ("pxy", ax22),
        ("pyx", ax22),
    ]:
        vals = [getattr(rp, comp) for rp in raw_rps]
        ax.plot(
            common_freqs,
            vals,
            color=comp_colors[comp],
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            marker=".",
            markersize=3,
            label=f"Raw {comp.upper()}",
        )

    # --- Plot weighted: thick solid line + full opacity + markers (highlight processed result) ---
    for comp, ax in [
        ("rxx", ax11),
        ("ryy", ax11),
        ("rxy", ax12),
        ("ryx", ax12),
        ("pxx", ax21),
        ("pyy", ax21),
        ("pxy", ax22),
        ("pyx", ax22),
    ]:
        vals = [getattr(rp, comp) for rp in wt_rps]
        ax.plot(
            common_freqs,
            vals,
            color=comp_colors[comp],
            linestyle="-",
            linewidth=2.5,
            alpha=1.0,
            marker="o",
            markersize=4,
            label=f"{name} {comp.upper()}",
        )

    # Configure axes
    for ax in [ax11, ax12, ax21, ax22]:
        ax.set_xscale("log")
        ax.invert_xaxis()  # high frequency on the left
        ax.grid(True, which="both", alpha=0.3, linestyle="--")
        # Put the processed-result legend first (last-added labels end up at the legend tail; reorder)
        handles, labels = ax.get_legend_handles_labels()
        # Separate raw and processed; processed entries go first
        raw_h, raw_l, den_h, den_l = [], [], [], []
        for h, l in zip(handles, labels):
            if l.startswith("Raw "):
                raw_h.append(h)
                raw_l.append(l)
            else:
                den_h.append(h)
                den_l.append(l)
        ax.legend(den_h + raw_h, den_l + raw_l, loc="best", fontsize=7, framealpha=0.9)

    # Log scale for resistivity
    ax11.set_yscale("log")
    ax12.set_yscale("log")

    # Subplot titles and axis labels
    ax11.set_ylabel("Resistivity (Ω·m)", fontsize=11)
    ax11.set_title("Resistivity: Rxx / Ryy", fontsize=11, fontweight="bold")
    ax12.set_ylabel("Resistivity (Ω·m)", fontsize=11)
    ax12.set_title("Resistivity: Rxy / Ryx", fontsize=11, fontweight="bold")
    ax21.set_ylabel("Phase (°)", fontsize=11)
    ax21.set_xlabel("Frequency (Hz)", fontsize=11)
    ax21.set_title("Phase: Pxx / Pyy", fontsize=11, fontweight="bold")
    ax22.set_ylabel("Phase (°)", fontsize=11)
    ax22.set_xlabel("Frequency (Hz)", fontsize=11)
    ax22.set_title("Phase: Pxy / Pyx", fontsize=11, fontweight="bold")

    fig.suptitle(
        f"Raw vs {name} — Resistivity & Phase", fontsize=14, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    if show:
        plt.show()
    return ax11, ax12, ax21, ax22


# ---------------------------------------------------------------------------
# Combined dashboard
# ---------------------------------------------------------------------------


def plot_denoise_dashboard(
    single_psms: list[SinglePSM],
    psms: list[PowerSpectrumMatrix],
    res_w: dict[float, np.ndarray],
    out_params: dict[float, np.ndarray] | None = None,
    freq: float | None = None,
    title: str = "",
    show: bool = True,
):
    """Combined dashboard.

    When freq is not None: show 4 subplots for that frequency
      - weight bar chart / impedance comparison / weight distribution box / global weight curve
    When freq is None: show an all-frequency overview
      - weight heatmap / weight curve / rho_phi comparison / global weight box plot
    """
    if freq is not None:
        return _dashboard_single_freq(
            single_psms, psms, res_w, out_params, freq, title, show
        )
    else:
        return _dashboard_overview(single_psms, psms, res_w, title, show)


def _dashboard_single_freq(
    single_psms,
    psms,
    res_w,
    out_params,
    freq,
    title,
    show,
):
    """Single-frequency dashboard: 2×2 layout"""
    weights = res_w[freq]
    n = len(weights)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    # Top-left: weight bar chart
    ax = axes[0, 0]
    indices = np.arange(n)
    norm = plt.Normalize(vmin=weights.min(), vmax=weights.max())
    cmap = plt.get_cmap("RdYlGn")
    colors = cmap(norm(weights))
    ax.bar(indices, weights, color=colors, edgecolor="black", linewidth=0.3, width=1.0)
    mean_w = np.mean(weights)
    ax.axhline(
        mean_w,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean={mean_w:.4f}",
    )
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Weight (log)")
    ax.set_yscale("log")
    ax.set_title("Weight Distribution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Top-right: impedance comparison
    ax = axes[0, 1]
    raw = _find_psm(psms, freq)
    weighted = _find_psm(single_psms, freq)
    raw_imp = raw.to_spsm().least_squares()
    wt_imp = weighted.least_squares()
    components = ["Zxx", "Zyy", "Zxy", "Zyx", "Tzx", "Tzy"]
    raw_amp = np.abs(
        [raw_imp.zxx, raw_imp.zyy, raw_imp.zxy, raw_imp.zyx, raw_imp.tzx, raw_imp.tzy]
    )
    wt_amp = np.abs(
        [wt_imp.zxx, wt_imp.zyy, wt_imp.zxy, wt_imp.zyx, wt_imp.tzx, wt_imp.tzy]
    )
    x = np.arange(len(components))
    w = 0.35
    ax.bar(
        x - w / 2,
        raw_amp,
        w,
        label="Raw",
        color="#1f77b4",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.bar(
        x + w / 2,
        wt_amp,
        w,
        label="Weighted",
        color="#2ca02c",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.set_ylabel("|Component|")
    ax.set_title("Impedance Amplitude", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    # Bottom-left: weight distribution histogram
    ax = axes[1, 0]
    ax.hist(
        weights,
        bins=min(30, n // 2 + 1),
        color="#8c564b",
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axvline(
        mean_w,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean={mean_w:.4f}",
    )
    ax.axvline(
        np.median(weights),
        color="#1f77b4",
        linestyle=":",
        linewidth=1.5,
        label=f"Median={np.median(weights):.4f}",
    )
    ax.set_xlabel("Weight (log)")
    ax.set_xscale("log")
    ax.set_ylabel("Count")
    ax.set_title("Weight Histogram", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", which="both")

    # Bottom-right: feature heatmap (if out_params is available)
    ax = axes[1, 1]
    if out_params is not None and freq in out_params:
        features = out_params[freq]
        order = np.argsort(weights)[::-1]
        features_sorted = features[order]
        n_feat = features_sorted.shape[1]
        labels = (
            FEATURE_LABELS[:n_feat]
            if n_feat <= len(FEATURE_LABELS)
            else [f"F{i}" for i in range(n_feat)]
        )

        im = ax.imshow(
            features_sorted, aspect="auto", cmap="RdBu_r", interpolation="nearest"
        )
        ax.set_xticks(range(n_feat))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Sample (sorted ↓)")
        ax.set_title("Features (by weight)", fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8)
    else:
        # Fall back to a global weight box plot when feature data is unavailable
        all_weights = np.concatenate([res_w[f] for f in sorted(res_w)])
        bp = ax.boxplot(
            all_weights,
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor="#8c564b", alpha=0.5),
        )
        ax.set_yscale("log")
        ax.set_ylabel("Weight (log)")
        ax.set_title("Global Weight Distribution", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--", which="both")

    suptitle = title or f"Denoise Dashboard — Freq = {freq} Hz"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    if show:
        plt.show()
    return fig, axes


def _dashboard_overview(
    single_psms,
    psms,
    res_w,
    title,
    show,
):
    """All-frequency overview dashboard: 2×2 layout"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    # Top-left: weight heatmap
    ax = axes[0, 0]
    freqs = sorted(res_w.keys())  # low frequency on top, high frequency at bottom
    n_freqs = len(freqs)
    n_samples = max(len(res_w[f]) for f in freqs)
    matrix = np.full((n_freqs, n_samples), np.nan)
    for i, f in enumerate(freqs):
        w = res_w[f]
        matrix[i, : len(w)] = w
    # Plot on a log scale (weight values span a wide range)
    matrix_log = np.where(np.isnan(matrix), np.nan, np.log10(np.maximum(matrix, 1e-10)))
    vmin = np.nanmin(matrix_log)
    vmax = np.nanmax(matrix_log)
    norm = LogNorm(vmin=10**vmin, vmax=10**vmax) if vmin != vmax else None

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdYlGn",
        interpolation="nearest",
        norm=norm,
        origin="upper",
    )
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_yticks(range(n_freqs))
    ax.set_yticklabels([f"{f}" for f in freqs], fontsize=7)
    fig.colorbar(im, ax=ax, label="Weight (log scale)", shrink=0.8)
    ax.set_title("Weight Heatmap", fontsize=11, fontweight="bold")

    # Top-right: weight curve
    ax = axes[0, 1]
    freqs_asc = sorted(res_w.keys())
    matrix_asc = np.full((n_samples, len(freqs_asc)), np.nan)
    for j, f in enumerate(freqs_asc):
        w = res_w[f]
        matrix_asc[: len(w), j] = w
    for i in range(n_samples):
        ax.plot(freqs_asc, matrix_asc[i], color="gray", alpha=0.15, linewidth=0.5)
    ax.plot(
        freqs_asc,
        np.nanmean(matrix_asc, axis=0),
        color="#d62728",
        linewidth=2,
        label="Mean",
    )
    ax.plot(
        freqs_asc,
        np.nanmedian(matrix_asc, axis=0),
        color="#1f77b4",
        linewidth=2,
        linestyle="--",
        label="Median",
    )
    ax.set_xscale("log")
    ax.invert_xaxis()  # high frequency on the left
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Weight (log)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", which="both")
    ax.set_title("Weight vs Frequency", fontsize=11, fontweight="bold")

    # Bottom-left: rho_phi comparison (highlight processed result)
    ax = axes[1, 0]
    freq_map_raw = {p.freq: p for p in psms}
    freq_map_wt = {p.freq: p for p in single_psms}
    common_freqs = sorted(set(freq_map_raw) & set(freq_map_wt))
    if common_freqs:
        raw_rps = [
            freq_map_raw[f].to_spsm().least_squares().to_rho_phi() for f in common_freqs
        ]
        wt_rps = [freq_map_wt[f].least_squares().to_rho_phi() for f in common_freqs]
        freq_vals = [rp.freq for rp in raw_rps]
        # raw: thin dashed line + low opacity (background)
        ax.plot(
            freq_vals,
            [rp.rxy for rp in raw_rps],
            "--",
            color="#2ca02c",
            linewidth=1.5,
            alpha=0.7,
            marker=".",
            markersize=3,
            label="Raw Rxy",
        )
        ax.plot(
            freq_vals,
            [rp.ryx for rp in raw_rps],
            "--",
            color="#d62728",
            linewidth=1.5,
            alpha=0.7,
            marker=".",
            markersize=3,
            label="Raw Ryx",
        )
        # weighted: thick solid line + full opacity (highlighted)
        ax.plot(
            freq_vals,
            [rp.rxy for rp in wt_rps],
            "-",
            color="#2ca02c",
            linewidth=2.5,
            marker="o",
            markersize=4,
            label="Denoised Rxy",
        )
        ax.plot(
            freq_vals,
            [rp.ryx for rp in wt_rps],
            "-",
            color="#d62728",
            linewidth=2.5,
            marker="o",
            markersize=4,
            label="Denoised Ryx",
        )
        ax.set_xscale("log")
        ax.invert_xaxis()  # high frequency on the left
        ax.set_yscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Resistivity (Ω·m)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_title("Raw vs Weighted Rho", fontsize=11, fontweight="bold")

    # Bottom-right: global weight box plot (grouped by frequency)
    ax = axes[1, 1]
    freqs_for_box = sorted(res_w.keys(), reverse=True)
    data_for_box = [res_w[f] for f in freqs_for_box]
    bp = ax.boxplot(
        data_for_box,
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="#8c564b", alpha=0.5),
    )
    ax.set_xticks(range(1, len(freqs_for_box) + 1))
    ax.set_xticklabels(
        [f"{f}" for f in freqs_for_box], rotation=45, ha="right", fontsize=7
    )
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Weight (log)")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", which="both")
    ax.set_title("Weight Distribution by Freq", fontsize=11, fontweight="bold")

    suptitle = title or "Denoise Overview — All Frequencies"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    if show:
        plt.show()
    return fig, axes
