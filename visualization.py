"""MT 去噪结果可视化模块

提供单频点和全频点两种粒度的可视化，配合 mt_py_lib.visualization.draw_rho_phi 使用。
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .param import FEATURE_LABELS
from .struct import PowerSpectrumMatrix, SinglePSM

# ---------------------------------------------------------------------------
# 辅助：按频点索引 list[SinglePSM] / list[PowerSpectrumMatrix]
# ---------------------------------------------------------------------------


def _find_psm(psm_list: list, freq: float):
    """在 PSM 列表中找到指定频点的对象"""
    for p in psm_list:
        if p.freq == freq:
            return p
    raise KeyError(f"Frequency {freq} not found in PSM list")


# ---------------------------------------------------------------------------
# 单频点可视化
# ---------------------------------------------------------------------------


def plot_single_freq_weights(
    res_w: dict[float, np.ndarray],
    freq: float,
    title: str = "",
    show: bool = True,
):
    """单频点权重分布柱状图

    每根柱子代表一个 PSM 样本的权重，颜色映射权重高低，
    标注均值和最大值线，直观看到哪些样本被模型信任。
    """
    weights = res_w[freq]
    n = len(weights)
    indices = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, 4))

    # 颜色映射：权重越高越暖
    norm = plt.Normalize(vmin=weights.min(), vmax=weights.max())
    cmap = plt.get_cmap("RdYlGn")
    colors = cmap(norm(weights))

    bars = ax.bar(
        indices, weights, color=colors, edgecolor="black", linewidth=0.3, width=1.0
    )

    # 均值和最大值参考线
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
    """单频点去噪前后阻抗分量对比

    用分组柱状图展示 raw PSM 和 weighted SinglePSM 的
    阻抗实部/虚部均值，看模型保留了哪些分量、压制了哪些。
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

    # 取幅值进行对比
    raw_amp = np.abs(np.array(raw_vals))
    wt_amp = np.abs(np.array(wt_vals))

    x = np.arange(len(components))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：幅值对比
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

    # 右图：相位对比
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
    """单频点输入特征热力图

    将 out_params[freq]（shape: psm_num × n_features）按权重从高到低排序后
    绘制热力图，高权重样本排在上方，展示模型关注的特征模式。
    """
    features = out_params[freq]  # (psm_num, n_features)
    weights = res_w[freq]

    # 按权重降序排列
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
    # 只标注高权重样本的索引
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
# 全频点可视化
# ---------------------------------------------------------------------------


def plot_weights_heatmap(
    res_w: dict[float, np.ndarray],
    title: str = "",
    show: bool = True,
):
    """所有频点权重热力图

    x = 样本索引，y = 频点（从低频到高频，低频在上），color = 权重值。
    """
    freqs = sorted(res_w.keys())  # 低频在上，高频在下
    n_freqs = len(freqs)
    n_samples = max(len(res_w[f]) for f in freqs)

    # 构建矩阵，不足的用 NaN 填充
    matrix = np.full((n_freqs, n_samples), np.nan)
    for i, f in enumerate(freqs):
        w = res_w[f]
        matrix[i, : len(w)] = w

    fig, ax = plt.subplots(figsize=(12, max(4, n_freqs * 0.3)))

    # 对权重取对数后绘图（权重值跨度大）
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
    """权重随频率变化趋势

    每条半透明线是一个样本，加粗线为均值和中位数。
    """
    freqs = sorted(res_w.keys())
    n_samples = max(len(res_w[f]) for f in freqs)

    # 构建 (n_samples, n_freqs) 矩阵
    matrix = np.full((n_samples, len(freqs)), np.nan)
    for j, f in enumerate(freqs):
        w = res_w[f]
        matrix[: len(w), j] = w

    fig, ax = plt.subplots(figsize=(10, 5))

    # 每个样本一条线
    for i in range(n_samples):
        ax.plot(freqs, matrix[i], color="gray", alpha=0.15, linewidth=0.5)

    # 均值和中位数
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
    ax.invert_xaxis()  # 高频在左
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
    """去噪前后电阻率/相位对比

    raw 用细虚线+低透明度（背景），
    weighted 用粗实线+全透明度（突出），
    频率轴高频在左。
    """
    # 按频点配对 raw 和 weighted
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

    # 颜色方案：raw 用浅色，weighted 用同色但更醒目
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

    # --- 画 raw：细虚线 + 低透明度 + 无 marker（背景参考）---
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

    # --- 画 weighted：粗实线 + 全透明度 + marker（突出去噪结果）---
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

    # 设置坐标轴
    for ax in [ax11, ax12, ax21, ax22]:
        ax.set_xscale("log")
        ax.invert_xaxis()  # 高频在左
        ax.grid(True, which="both", alpha=0.3, linestyle="--")
        # 将去噪结果的 legend 放在前面（最后添加的 label 排在 legend 末尾，需调整）
        handles, labels = ax.get_legend_handles_labels()
        # 分离 raw 和 denoised，denoised 放前面
        raw_h, raw_l, den_h, den_l = [], [], [], []
        for h, l in zip(handles, labels):
            if l.startswith("Raw "):
                raw_h.append(h)
                raw_l.append(l)
            else:
                den_h.append(h)
                den_l.append(l)
        ax.legend(den_h + raw_h, den_l + raw_l, loc="best", fontsize=7, framealpha=0.9)

    # 电阻率取对数
    ax11.set_yscale("log")
    ax12.set_yscale("log")

    # 子图标题和轴标签
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
# 综合仪表板
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
    """综合仪表板

    freq 不为 None 时：展示该频点的 4 个子图
      - 权重柱状图 / 阻抗对比 / 权重分布箱线 / 全局权重曲线
    freq 为 None 时：展示全频点概览
      - 权重热力图 / 权重曲线 / rho_phi 对比 / 全局权重箱线
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
    """单频点仪表板：2×2 布局"""
    weights = res_w[freq]
    n = len(weights)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    # 左上：权重柱状图
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

    # 右上：阻抗对比
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

    # 左下：权重分布直方图
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

    # 右下：特征热力图（如果有 out_params）
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
        # 没有特征数据时画全局权重箱线图
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
    """全频点概览仪表板：2×2 布局"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    # 左上：权重热力图
    ax = axes[0, 0]
    freqs = sorted(res_w.keys())  # 低频在上，高频在下
    n_freqs = len(freqs)
    n_samples = max(len(res_w[f]) for f in freqs)
    matrix = np.full((n_freqs, n_samples), np.nan)
    for i, f in enumerate(freqs):
        w = res_w[f]
        matrix[i, : len(w)] = w
    # 对权重取对数后绘图（权重值跨度大）
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

    # 右上：权重曲线
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
    ax.invert_xaxis()  # 高频在左
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Weight (log)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", which="both")
    ax.set_title("Weight vs Frequency", fontsize=11, fontweight="bold")

    # 左下：rho_phi 对比（突出去噪结果）
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
        # raw：细虚线 + 低透明度（背景）
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
        # weighted：粗实线 + 全透明度（突出）
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
        ax.invert_xaxis()  # 高频在左
        ax.set_yscale("log")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Resistivity (Ω·m)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_title("Raw vs Weighted Rho", fontsize=11, fontweight="bold")

    # 右下：全局权重箱线图（按频点分组）
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
