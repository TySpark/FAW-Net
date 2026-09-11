import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .t_calc import calc_log10, calc_rho_phs, calc_zxy_zyx


class Loss(nn.Module):
    """
    Multi-task loss function with 4 components:
    1. Loss 1 — MSE supervision: mean squared error between predictions and targets.
    2. Loss 2 — second-order continuity constraint under non-uniform frequency sampling:
       central-difference second derivative in the log-frequency domain, ensuring
       physical smoothness.
    3. Loss 3 — L2 weight regularization: prevents weights from concentrating too heavily
       on a single sample; drives them toward uniformity when there is no clear
       discriminative signal.
    4. Loss 4 — causality constraint: ensures Re[Z] and Im[Z] satisfy causality.
    """

    def __init__(
        self,
        lambda_sup: float = 1.0,
        lambda_smooth: float = 0.01,
        lambda_polar: float = 0.5,
        lambda_kk: float = 0.1,
    ):
        super().__init__()
        self.lambda_sup = lambda_sup
        self.lambda_smooth = lambda_smooth
        self.lambda_polar = lambda_polar
        self.lambda_kk = lambda_kk

    def forward(
        self,
        model_output: dict[str, dict[float, torch.Tensor]],
        # model_output: dict[float, torch.Tensor],
        matrix_dict: dict[float, torch.Tensor],
        target_dict: dict[float, torch.Tensor],
        device: torch.device | None = None,
        *args,
        **kwargs,
    ):
        weights_dict = model_output["weights"]
        sorted_freqs = sorted(weights_dict.keys(), reverse=False)

        pred_list, target_list = [], []
        z_list, freq_list = [], []

        for f in sorted_freqs:
            w_raw = weights_dict[f]
            if w_raw.shape[0] == 0:
                continue

            weights = w_raw.unsqueeze(-1).unsqueeze(-1)
            matrix = matrix_dict[f]
            s_avg = torch.sum(weights * matrix, dim=0)

            # MSE supervision target
            preds = calc_rho_phs(f, s_avg, is_log=True)
            pred_list.append(preds)

            # Analyticity-constraint features
            zxy, zyx = calc_zxy_zyx(s_avg)
            z_feat = torch.stack([zxy.real, zxy.imag, zyx.real, zyx.imag])
            z_list.append(z_feat)

            target = target_dict[f]
            targets = torch.stack(
                [
                    calc_log10(target[0]),
                    calc_log10(target[2]),
                    target[1],
                    target[3],
                ]
            )
            target_list.append(targets)
            freq_list.append(f)

        if not pred_list:
            return torch.tensor(0.0, device=device)

        # --- Loss 1: MSE supervision ---
        predicted_tensor = torch.stack(pred_list)
        target_tensor = torch.stack(target_list)
        loss_sup = F.mse_loss(predicted_tensor, target_tensor)

        # --- Loss 2: Strict second-order continuity under non-uniform frequency sampling (physical essence) ---
        loss_smooth = torch.zeros(1, device=device, dtype=predicted_tensor.dtype)
        if len(z_list) >= 3:
            z_tensor = torch.stack(z_list)  # (n_freqs, 4)
            f_tensor = torch.tensor(
                freq_list, device=device, dtype=z_tensor.dtype
            ).unsqueeze(-1)
            f_tensor = torch.log10(f_tensor.clamp(min=1e-12))

            # Normalize only inside the smoothness computation
            z_mean = z_tensor.mean(dim=0, keepdim=True)
            z_std = z_tensor.std(dim=0, keepdim=True).clamp(min=1e-6)
            z_norm = (z_tensor - z_mean) / z_std

            # Subsequent diff2 is computed from z_norm (not z_tensor)
            df1 = f_tensor[1:-1] - f_tensor[:-2]
            df2 = f_tensor[2:] - f_tensor[1:-1]
            grad1 = (z_norm[1:-1] - z_norm[:-2]) / df1.clamp(min=1e-8)
            grad2 = (z_norm[2:] - z_norm[1:-1]) / df2.clamp(min=1e-8)
            diff2 = (grad2 - grad1) / ((df1 + df2) / 2.0).clamp(min=1e-8)

            diff2 = (
                torch.clamp(diff2.abs(), max=1 / self.lambda_smooth * 0.1)
                * diff2.sign()
            )

            # L1
            # loss_smooth = torch.mean(torch.abs(diff2))
            # L2, more sensitive to interference
            loss_smooth = torch.mean(diff2**2)

        # --- Loss 3: One-sided constraint — it only blocks extreme one-hot solutions and does not
        # force uniformity, so the model can freely choose the optimal solution between the safe
        # regions of "not concentrated" and "concentrated". ---
        loss_polar = torch.zeros(1, device=device)
        n_freqs = 0

        for f in sorted_freqs:
            w = weights_dict[f]
            n_spectra = w.shape[0]

            if n_spectra <= 1:
                continue

            min_effective_points = max(1.0, n_spectra * 0.1)
            # Theoretical upper bound on the sum of squared weights
            max_l2_norm = 1.0 / min_effective_points
            # Sum of squares of the current weights
            current_l2 = torch.sum(w_raw**2)
            # ReLU truncation: penalize concentration beyond the allowed level
            sparsity_penalty = F.relu(current_l2 - max_l2_norm)
            loss_polar += sparsity_penalty
            n_freqs += 1

        if n_freqs > 0:
            loss_polar /= n_freqs

        # --- Loss 4: Kramers-Kronig causality constraint (central difference) ---
        # KK relations (Bode gain-phase relation):
        # ϕ(f_i) ≈ (π/2) × [ln|Z(f_{i+1})| - ln|Z(f_{i-1})|] / [ln(f_{i+1}) - ln(f_{i-1})]
        # Computed separately for zxy and zyx to ensure Re[Z] and Im[Z] satisfy causality
        loss_kk = torch.zeros(1, device=device, dtype=predicted_tensor.dtype)
        if len(z_list) >= 3:
            z_tensor = torch.stack(
                z_list
            )  # (n_freqs, 4): [zxy_re, zxy_im, zyx_re, zyx_im]
            # Note: freq_log_list must use ln(f); see explanation below
            f_tensor = torch.tensor(freq_list, device=device, dtype=z_tensor.dtype)
            f_tensor = torch.log(f_tensor.clamp(min=1e-12))
            # Handle zxy and zyx separately
            for z_re_idx, z_im_idx, need_shift in [(0, 1, False), (2, 3, True)]:
                z_re = z_tensor[:, z_re_idx]  # (n_freqs,)
                z_im = z_tensor[:, z_im_idx]  # (n_freqs,)

                # Compute magnitude and phase
                abs_z = torch.sqrt(z_re**2 + z_im**2) + 1e-12
                log_abs_z = torch.log(abs_z)
                phase_z = torch.atan2(z_im, z_re)

                # ---------------------Non-uniform three-point central difference: d(ln|Z|) / d(ln f)------------------
                log_f = f_tensor  # already ln(f)
                h_left = log_f[1:-1] - log_f[:-2]  # ln f_i - ln f_{i-1}
                h_right = log_f[2:] - log_f[1:-1]  # ln f_{i+1} - ln f_i

                numerator = (
                    h_left**2 * log_abs_z[2:]
                    + (h_right**2 - h_left**2) * log_abs_z[1:-1]
                    - h_right**2 * log_abs_z[:-2]
                )
                denominator = h_left * h_right * (h_left + h_right)

                # Predicted phase at the original frequency point f_i
                predicted_phase = (
                    (math.pi / 2) * numerator / denominator.clamp(min=1e-10)
                )
                # --------------------------------------------------------------------------

                # Zyx component: subtract π from the predicted phase to align with arctan2's third quadrant
                if need_shift:
                    predicted_phase = predicted_phase - math.pi
                # Actual phase is taken directly at f_i (not midpoint-averaged)
                actual_phase = phase_z[1:-1]

                # Residual
                kk_residual = predicted_phase - actual_phase
                loss_kk = loss_kk + torch.mean(kk_residual**2)
            loss_kk = loss_kk / 2.0  # average over zxy and zyx

        total_loss = (
            self.lambda_sup * loss_sup
            + self.lambda_smooth * loss_smooth
            + self.lambda_polar * loss_polar
            + self.lambda_kk * loss_kk
        )

        return total_loss
