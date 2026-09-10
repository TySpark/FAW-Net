import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .struct import PowerSpectrumMatrix, SinglePSM


# ==========================================
# 1. 动态特征筛选生成器
# ==========================================
def create_feature_selector(
    use_impedance: bool,
    use_tipper: bool,
    use_psd: bool,
    use_phase_tensor_angles: bool,
    use_phase_tensor_main: bool,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    根据物理模块开关，生成一个裁剪特征的函数。
    返回的函数接收 (n, 30) 的 Tensor，返回 (n, selected_dim) 的 Tensor。
    """
    # 按照你的 to_deal_two 方法的特征拼装顺序
    indices = []

    if use_impedance:
        indices.extend(range(0, 12))  # Zxx, Zyy, Zxy, Zyx (amp, sin, cos)
    if use_tipper:
        indices.extend(range(12, 18))  # Tzx, Tzy (amp, sin, cos)
    if use_psd:
        indices.extend(range(18, 22))  # Ex, Ey, Hx, Hy (scaled)
    if use_phase_tensor_angles:
        indices.extend(range(22, 26))  # Alpha, Beta (sin, cos)
    if use_phase_tensor_main:
        indices.extend(range(26, 30))  # P11, P12, P21, P22

    idx_array = np.array(indices)

    def selector(x: np.ndarray) -> np.ndarray:
        # 将 idx_tensor 放到与数据相同的设备上再切片
        return x[:, idx_array]

    return selector


def _calc_model_output(
    model: torch.nn.Module,
    features: dict[float, torch.Tensor],
    psms: list[PowerSpectrumMatrix],
) -> tuple[dict[float, np.ndarray], list[SinglePSM]]:
    # 开启推理模式
    model.eval()
    with torch.no_grad():
        out = model(features)
        if isinstance(out, tuple | list):
            weights = out[0]
        elif isinstance(out, dict):
            weights = out["weights"]
        else:
            weights = out

    weights: dict[float, np.ndarray] = {f: weights[f].cpu().numpy() for f in weights}
    single_psms: list[SinglePSM] = []

    for psm in psms:
        f = psm.freq
        weight = weights[f]
        single_psms.append(psm.weighting(weight))
    return weights, single_psms


def denoise(
    psms: list[PowerSpectrumMatrix],
    params: dict[float, np.ndarray],
    model: torch.nn.Module | list[torch.nn.Module] | dict[str, torch.nn.Module],
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> (
    tuple[dict[float, np.ndarray], list[SinglePSM]]
    | tuple[list[dict[float, np.ndarray]], list[list[SinglePSM]]]
    | tuple[dict[str, dict[float, np.ndarray]], dict[str, list[SinglePSM]]]
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = {
        f: torch.tensor(feat, dtype=dtype, device=device) for f, feat in params.items()
    }

    if isinstance(model, dict):
        res_w = {}
        res_spsm = {}
        for name, m in model.items():
            m.to(device)
            weights, single_psms = _calc_model_output(m, features, psms)
            res_w[name] = weights
            res_spsm[name] = single_psms
        return res_w, res_spsm

    elif isinstance(model, list):
        res_w = []
        res_spsm = []
        for m in model:
            if isinstance(m, torch.nn.Module):
                m.to(device)
                weights, single_psms = _calc_model_output(m, features, psms)
                res_w.append(weights)
                res_spsm.append(single_psms)
        return res_w, res_spsm

    else:
        model.to(device)
        weights, single_psms = _calc_model_output(model, features, psms)

        return weights, single_psms


def denoise_by_pkl(
    pkl: Path,
    model: torch.nn.Module | list[torch.nn.Module] | dict[str, torch.nn.Module],
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    use_impedance: bool = True,
    use_tipper: bool = True,
    use_psd: bool = True,
    use_phase_tensor_angles: bool = True,
    use_phase_tensor_main: bool = True,
) -> tuple[
    list[PowerSpectrumMatrix],
    dict[float, np.ndarray],
    dict[float, np.ndarray]
    | list[dict[float, np.ndarray]]
    | dict[str, dict[float, np.ndarray]],
    list[SinglePSM] | list[list[SinglePSM]] | dict[str, list[SinglePSM]],
]:
    print("---计算参数")
    with open(pkl, "rb") as f:
        dataset_dict = pickle.load(f)

    # edi_target = dataset_dict["target"]
    matrix = dataset_dict["matrix"]
    origin_param = dataset_dict["param"]

    selecter = create_feature_selector(
        use_impedance=use_impedance,
        use_tipper=use_tipper,
        use_psd=use_psd,
        use_phase_tensor_angles=use_phase_tensor_angles,
        use_phase_tensor_main=use_phase_tensor_main,
    )

    params = {k: selecter(m) for k, m in origin_param.items()}

    psms = []
    for k, m in matrix.items():
        spms = [
            SinglePSM(freq=k, window_length=0, seg_num=len(m), matrix=im) for im in m
        ]
        psm = PowerSpectrumMatrix(
            freq=k, window_length=0, seg_num=len(m), psms=tuple(spms)
        )
        psms.append(psm)

    print("---加权计算")
    weights, single_psms = denoise(
        psms=psms,
        params=params,
        model=model,
        device=device,
        dtype=dtype,
    )

    return psms, params, weights, single_psms


if __name__ == "__main__":
    from .model import FreqAdaptWeighter
    from .visualization import (
        plot_before_after_rho_phi,
        plot_denoise_dashboard,
        plot_single_freq_weights,
        plot_weight_curve,
        plot_weights_heatmap,
    )

    model = FreqAdaptWeighter(n_features=20)

    best_model = torch.load(Path(r"src_2_github\best_model.pth"))
    model.load_state_dict(best_model["model_state_dict"])

    pkl = Path(r"src_2_github\pkl\ANH0203A.pkl")
    psms, params, weights, single_psms = denoise_by_pkl(
        pkl=pkl,
        model=model,
        use_tipper=False,
        use_phase_tensor_angles=False,
    )

    # 去噪前后 rho_phi 对比
    plot_before_after_rho_phi(single_psms, psms, name=f"{pkl.stem} Denoised")

    # 权重可视化
    plot_weights_heatmap(weights, title=f"{pkl.stem} — Weight Heatmap")
    plot_weight_curve(weights, title=f"{pkl.stem} — Weight vs Frequency")
