"""FreqAdaptWeighter — deep-learning-based intelligent weighting of magnetotelluric spectra.

Supports unequal numbers of spectral segments per frequency (e.g. 500, 100, 20, etc.).

"""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionGate(nn.Module):
    """
    Adaptive gated fusion: lets the network decide how much global information
    each position needs.

    Inputs: local_feat [S, D], global_feat [D]
    Output: gated_feat [S, D]
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        # Gate is initialized to favor keeping local features (output ≈ 0), i.e. less global information
        for m in self.gate_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, -2.0)  # sigmoid(-2) ≈ 0.12

    def forward(
        self, local_feat: torch.Tensor, global_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        local_feat: [S, D]  segment-level features
        global_feat: [D]    frequency-level features (CFT output)
        """
        global_expanded = global_feat.unsqueeze(0).expand_as(local_feat)  # [S, D]
        gate = self.gate_proj(
            torch.cat([local_feat, global_expanded], dim=-1)
        )  # [S, D]
        # gate ≈ 0 → keep local; gate ≈ 1 → prefer global
        return (1 - gate) * local_feat + gate * global_expanded


class CrossSegmentAttention(nn.Module):
    """
    Intra-frequency cross-segment self-attention (CSA; input [n_seg, d_model])
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [n_seg, d_model]

        Returns
        -------
        out : Tensor [n_seg, d_model]
        """
        # Add batch dimension → [1, S, D]
        x_b = x.unsqueeze(0)

        # Self-Attention + residual
        attn_out, _ = self.self_attn(x_b, x_b, x_b)
        x_b = self.norm1(x_b + attn_out)

        # FFN + residual
        ffn_out = self.ffn(x_b)
        x_b = self.norm2(x_b + ffn_out)

        return x_b.squeeze(0)  # [S, D]


class FreqDynamicConv1D(nn.Module):
    """
    Frequency-conditioned dynamic convolution (FreqDynamicConv1D / FDC)
    - Low frequency → large receptive field (deep structures are continuous)
    - High frequency → small receptive field (preserves shallow details)
    """

    def __init__(self, d_model: int, dropout: float = 0.1, kernel_size: int = 3):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size

        # Frequency → convolution kernel generator
        # Two-layer MLP: expand first, then generate kernel parameters
        hidden = d_model // 2
        self.kernel_generator = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # regularization to prevent frequency overfitting
            nn.Linear(hidden, d_model * kernel_size),
        )

        # Lightweight refinement (nonlinearity in the residual path)
        self.refine = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """Initialize: convolution kernel = identity (center = 1), refinement = 0"""
        # Final kernel-generator layer: zero-init, then set the center position to 1
        nn.init.zeros_(self.kernel_generator[-1].weight)  # type: ignore
        nn.init.zeros_(self.kernel_generator[-1].bias)  # type: ignore
        # Depthwise conv1d kernel layout: [D, 1, K]
        # Center position index
        center_start = (self.kernel_size // 2) * self.d_model
        nn.init.constant_(
            self.kernel_generator[-1].bias[center_start : center_start + self.d_model],  # type: ignore
            1.0,
        )

        # Zero-init refinement (identity)
        nn.init.zeros_(self.refine[-1].weight)  # type: ignore
        nn.init.zeros_(self.refine[-1].bias)  # type: ignore

    def forward(self, H: torch.Tensor, freq: float) -> torch.Tensor:
        """
        Args:
            H:    [n_seg, d_model]  input segment features
            freq: float             current frequency (Hz)
        Returns:
            adapted: [n_seg, d_model]  frequency-modulated features
        """

        # n_seg = H.shape[0]
        D = self.d_model
        K = self.kernel_size
        device = H.device

        # 1. Frequency generates the convolution kernel
        freq_log = torch.tensor([[math.log(freq)]], device=device)
        kernel_flat = self.kernel_generator(freq_log)  # [1, D*K]
        kernel = kernel_flat.view(D, 1, K)  # [D, 1, K]
        # groups=D → requires C_in=D

        # 2. Depthwise 1D convolution
        H_conv = H.T.unsqueeze(0)  # [1, D, n_seg]  ← fixed this line
        pad = K // 2
        H_padded = F.pad(H_conv, (pad, pad), mode="replicate")  # [1, D, n_seg+pad]
        H_out = F.conv1d(H_padded, kernel, groups=D)  # [1, D, n_seg] ✅
        H_out = H_out.squeeze(0).T  # [n_seg, D]  ← fixed this line

        # 3. Residual + refinement
        adapted = self.norm(H + self.dropout(H_out))
        adapted = adapted + self.dropout(self.refine(adapted))

        return adapted


class CrossFrequencyTransformer(nn.Module):
    """
    Physics-guided cross-frequency attention (Cross-Frequency Attention, CFA)

    - MT impedance is a continuous function of frequency. Neighboring frequencies
      can corroborate each other:
    - A segment anomalous at the current band but normal at neighboring bands →
      the anomaly is likely local noise.
    - Skin depth δ ∝ √(ρ/f) determines the strength of physical correlation between frequencies.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        n_freq_neighbors: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_neighbors = n_freq_neighbors

        # Physics bias: attention decay determined by skin depth
        self.physics_proj = nn.Linear(1, d_model)
        self.physics_gate = nn.Sequential(
            nn.Linear(d_model * 2, 1),
            nn.Sigmoid(),
        )

        # Multi-head attention (with physics bias)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # sigmoid(0.5) ≈ 0.62 → sigmoid(-2.0) ≈ 0.12
        # Keep CFT nearly inactive early in training; gradually learn to participate
        self.gamma = nn.Parameter(torch.tensor(-2.0))

    def _build_physics_bias(self, freqs: list, device: torch.device) -> torch.Tensor:
        """Build a skin-depth physics bias"""
        n = len(freqs)
        log_f = torch.tensor([math.log(f) for f in freqs], device=device)

        # Skin-depth distance: δ ∝ √(1/f), so log(δ) ∝ -0.5·log(f)
        # Physical distance: degree of skin-depth overlap between neighboring frequencies
        log_depth = -0.5 * log_f  # relative measure of log(δ)
        depth_dist = torch.abs(log_depth.unsqueeze(0) - log_depth.unsqueeze(1))

        # Physics bias: closer (more skin-depth overlap) → stronger attention
        # Gaussian decay: bias = exp(-d²/(2σ²)), with σ controlled by a learnable parameter
        sigma = F.softplus(self.physics_proj(log_f.unsqueeze(-1))).mean()
        physics_bias = -(depth_dist**2) / (2 * sigma**2 + 1e-8)

        # Neighbor mask
        freq_dist = torch.abs(log_f.unsqueeze(0) - log_f.unsqueeze(1))
        k = min(self.n_neighbors, n)
        neighbor_mask = torch.full((n, n), float("-inf"), device=device)
        for i in range(n):
            _, idx = freq_dist[i].topk(k, largest=False)
            neighbor_mask[i, idx] = 0.0

        # Combine physics bias and neighbor mask
        combined_bias = physics_bias + neighbor_mask

        return combined_bias

    def forward(self, freq_feat: torch.Tensor, freqs: list) -> torch.Tensor:
        """
        Parameters
        ----------
        freq_feat : Tensor [n_freq, d_model]
        freqs     : list[float]

        Returns
        -------
        out : Tensor [n_freq, d_model]
        """
        n_freq = len(freqs)
        device = freq_feat.device

        if n_freq <= 1:
            return freq_feat

        # Physics bias
        attn_bias = self._build_physics_bias(freqs, device)  # [F, F]

        # Cross-frequency attention (with physics bias)
        seq_b = freq_feat.unsqueeze(0)  # [1, F, D]
        attn_out, attn_weights = self.cross_attn(
            seq_b, seq_b, seq_b, attn_mask=attn_bias
        )
        attn_out = attn_out.squeeze(0)  # [F, D]

        # Residual + normalization
        out = self.norm1(attn_out + freq_feat)

        # FFN
        out = self.norm2(self.ffn(out) + out)

        # Gated residual: controls CFA contribution strength
        gamma = torch.sigmoid(self.gamma)
        out = freq_feat + gamma * (out - freq_feat)

        return out


# ================================================================
#  Main Model
# ================================================================


class FreqAdaptWeighter(nn.Module):
    """
    Frequency-adaptive weighting model — supports unequal numbers of segments per frequency

    Parameters
    ----------
    n_features       : input feature dimension m
    d_model          : hidden dimension
    n_heads          : number of attention heads
    n_freq_neighbors : number of neighbors each frequency attends to in CFA
    dropout          : dropout rate

    Input
    -----
    input_dict : dict[float, Tensor[n_seg_i, n_features]]
        key = frequency (Hz), value = features of each spectral segment at that frequency
        n_seg_i may differ across frequencies (e.g. 500, 100, 20)

    Output
    ------
    dict {
        'weights' : dict[float, Tensor[n_seg_i]]  normalized weights for each segment at each frequency
    }
    """

    # Hyperparameters chosen via Optuna: 64 - 8 - 4 - 0.07
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 8,
        n_freq_neighbors: int = 4,
        dropout: float = 0.07,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_features + 1, d_model),  # +1: frequency feature
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.csas = CrossSegmentAttention(d_model, n_heads, dropout)
        self.fcd = FreqDynamicConv1D(d_model, dropout)
        self.cft = CrossFrequencyTransformer(
            d_model, n_heads, n_freq_neighbors, dropout
        )
        self.fusion_gate = FusionGate(d_model)

        self.weight_heads = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # temp = exp(0.0) = 1.0, giving the softmax more discriminative power early on
        self.log_temps = nn.Parameter(torch.tensor(0.0))

        self._init_weights()

    # ----------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def _pool_freqs(self, seg_feats: dict) -> torch.Tensor:
        """Mean-pool per frequency → [n_freq, d_model]"""
        return torch.stack([sf.mean(dim=0) for sf in seg_feats.values()], dim=0)

    # ----------------------------------------------------------
    def forward(self, input_dict: Dict[float, torch.Tensor]) -> dict:
        """
        Parameters
        ----------
        input_dict : {freq: Tensor[n_seg_i, n_features]}
                     n_seg_i may differ across frequencies
        """
        freqs = sorted(input_dict.keys())

        # ---- Encode per frequency (freq concatenated into features) → dict ----
        encoded = {}
        for f in freqs:
            feat = input_dict[f]  # [S_i, n_feat]
            freq_col = torch.full(
                (feat.shape[0], 1), math.log(f), device=feat.device
            )  # [S_i, 1]
            feat_with_freq = torch.cat([feat, freq_col], dim=-1)  # [S_i, n_feat+1]
            encoded[f] = self.encoder(feat_with_freq)  # [S_i, D]

        # Features at the current layer (dict preserves frequency indexing)
        current = {f: encoded[f].clone() for f in freqs}

        # 1. Per frequency:
        for f in freqs:
            current[f] = self.csas(current[f])
            current[f] = self.fcd(current[f], f)

        # 2. Pool → cross-frequency → gated fusion back to segment level
        freq_feat = self._pool_freqs(current)  # [F, D]
        freq_feat = self.cft(freq_feat, freqs)  # [F, D]

        # CFA broadcast: direct addition → gated fusion
        for fi, f in enumerate(freqs):
            # Adaptive fusion via FusionGate
            current[f] = self.fusion_gate(current[f], freq_feat[fi])

        # 3. Per frequency: weight generation
        temp = torch.exp(self.log_temps)
        weights: dict = {}
        for f in freqs:
            logits = self.weight_heads(current[f]).squeeze(-1)
            w = F.softmax(logits / temp, dim=0)
            weights[f] = w

        return {
            "weights": weights,
        }

    # ----------------------------------------------------------
    @torch.no_grad()
    def predict_weights(
        self, input_dict: Dict[float, torch.Tensor]
    ) -> dict[float, torch.Tensor]:
        """Inference: returns {freq: Tensor[n_seg_i]}"""
        self.eval()
        out = self.forward(input_dict)
        return out["weights"]


# ================================================================
#  Usage Example
# ================================================================

if __name__ == "__main__":
    # ---- Hyperparameters ----
    n_features = 26

    # ---- Model ----
    model = FreqAdaptWeighter(
        n_features=n_features,
    )

    # ---- Dummy-data verification -------
    print(f"\n{'=' * 65}")
    print("  Dummy-data forward-pass test")
    print(f"{'=' * 65}")

    # Build 3 frequencies with different segment counts (simulating a realistic scenario)
    fake_input: Dict[float, torch.Tensor] = {
        0.1: torch.randn(100, n_features),  # 0.1 Hz, 100 segments
        1.0: torch.randn(300, n_features),  # 1.0 Hz, 300 segments
        10.0: torch.randn(1000, n_features),  # 10.0 Hz, 1000 segments
    }

    model.eval()
    with torch.no_grad():
        output = model(fake_input)

    weights = output["weights"]
    for f, w in weights.items():
        print(
            f"  freq={f:5.1f} Hz  |  segments={w.shape[0]:4d}  |  "
            f"sum={w.sum().item():.4f}  |  mean={w.mean().item():.4f}  |  "
            f"min={w.min().item():.4f}  max={w.max().item():.4f}"
        )

    print("\n  Forward pass OK ✓")
