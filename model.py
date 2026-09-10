"""FreqAdaptWeighter — 深度学习驱动的大地电磁频谱智能加权模型

支持每个频率下不等长的谱段数量 (e.g. 500, 100, 20 等)。

"""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionGate(nn.Module):
    """
    自适应门控融合: 让网络决定每个位置需要多少全局信息

    输入: local_feat [S, D], global_feat [D]
    输出: gated_feat [S, D]
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        # 门控初始偏向"保留局部"(输出≈0)，即少用全局信息
        for m in self.gate_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, -2.0)  # sigmoid(-2) ≈ 0.12

    def forward(
        self, local_feat: torch.Tensor, global_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        local_feat: [S, D]  段级特征
        global_feat: [D]     频率级特征(CFT输出)
        """
        global_expanded = global_feat.unsqueeze(0).expand_as(local_feat)  # [S, D]
        gate = self.gate_proj(
            torch.cat([local_feat, global_expanded], dim=-1)
        )  # [S, D]
        # gate ≈ 0 时 → 保留local; gate ≈ 1 时 → 倾向global
        return (1 - gate) * local_feat + gate * global_expanded


class CrossSegmentAttention(nn.Module):
    """
    频率内段间自注意力 (CSA, 输入 [n_seg, d_model])
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
        # 加 batch 维度 → [1, S, D]
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
    频率条件动态卷积 (FreqDynamicConv1D / FDC)
    - 低频 → 大感受野（深部结构连续）
    - 高频 → 小感受野（浅部细节保留）
    """

    def __init__(self, d_model: int, dropout: float = 0.1, kernel_size: int = 3):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size

        # 频率 → 卷积核生成器
        # 两层 MLP：先扩展再生成核参数
        hidden = d_model // 2
        self.kernel_generator = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # 正则化防止频率过拟合
            nn.Linear(hidden, d_model * kernel_size),
        )

        # 轻量精调（残差路径中的非线性）
        self.refine = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """初始化：卷积核 = identity（中心为 1），精调 = 0"""
        # 核生成器最后一层：全部初始化为 0，然后中心位置设为 1
        nn.init.zeros_(self.kernel_generator[-1].weight)  # type: ignore
        nn.init.zeros_(self.kernel_generator[-1].bias)  # type: ignore
        # depthwise conv1d 核格式：[D, 1, K]
        # 中心位置索引
        center_start = (self.kernel_size // 2) * self.d_model
        nn.init.constant_(
            self.kernel_generator[-1].bias[center_start : center_start + self.d_model],  # type: ignore
            1.0,
        )

        # 精调初始化为零（identity）
        nn.init.zeros_(self.refine[-1].weight)  # type: ignore
        nn.init.zeros_(self.refine[-1].bias)  # type: ignore

    def forward(self, H: torch.Tensor, freq: float) -> torch.Tensor:
        """
        Args:
            H:    [n_seg, d_model]  输入的段特征
            freq: float             当前频率 (Hz)
        Returns:
            adapted: [n_seg, d_model]  频率调制后的特征
        """

        # n_seg = H.shape[0]
        D = self.d_model
        K = self.kernel_size
        device = H.device

        # 1. 频率生成卷积核
        freq_log = torch.tensor([[math.log(freq)]], device=device)
        kernel_flat = self.kernel_generator(freq_log)  # [1, D*K]
        kernel = kernel_flat.view(D, 1, K)  # [D, 1, K]
        # groups=D → 需要 C_in=D

        # 2. Depthwise 1D 卷积
        H_conv = H.T.unsqueeze(0)  # [1, D, n_seg]  ← 修复这行
        pad = K // 2
        H_padded = F.pad(H_conv, (pad, pad), mode="replicate")  # [1, D, n_seg+pad]
        H_out = F.conv1d(H_padded, kernel, groups=D)  # [1, D, n_seg] ✅
        H_out = H_out.squeeze(0).T  # [n_seg, D]  ← 修复这行

        # 3. 残差 + 精调
        adapted = self.norm(H + self.dropout(H_out))
        adapted = adapted + self.dropout(self.refine(adapted))

        return adapted


class CrossFrequencyTransformer(nn.Module):
    """
    物理引导的跨频率注意力 —— (Cross-Frequency Attention, CFA)

    - MT 阻抗是频率的连续函数。相邻频率可以互相"作证":
    - 某段在当前频段异常但在相邻频段正常 → 该异常可能是局部噪声。
    - 趋肤深度 δ ∝ √(ρ/f) 决定了频率间的物理关联强度。
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

        # 物理偏置：趋肤深度决定的注意力衰减
        self.physics_proj = nn.Linear(1, d_model)
        self.physics_gate = nn.Sequential(
            nn.Linear(d_model * 2, 1),
            nn.Sigmoid(),
        )

        # 多头注意力（带物理偏置）
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
        # 让CFT训练初期几乎不参与，逐步学习介入
        self.gamma = nn.Parameter(torch.tensor(-2.0))

    def _build_physics_bias(self, freqs: list, device: torch.device) -> torch.Tensor:
        """构建趋肤深度物理偏置"""
        n = len(freqs)
        log_f = torch.tensor([math.log(f) for f in freqs], device=device)

        # 趋肤深度距离：δ ∝ √(1/f)，所以 log(δ) ∝ -0.5·log(f)
        # 物理距离：相邻频率的趋肤深度重叠程度
        log_depth = -0.5 * log_f  # log(δ) 的相对度量
        depth_dist = torch.abs(log_depth.unsqueeze(0) - log_depth.unsqueeze(1))

        # 物理偏置：距离越近（趋肤深度重叠越多），注意力越强
        # 使用高斯衰减：bias = exp(-d²/(2σ²))，σ 由可学习参数控制
        sigma = F.softplus(self.physics_proj(log_f.unsqueeze(-1))).mean()
        physics_bias = -(depth_dist**2) / (2 * sigma**2 + 1e-8)

        # 邻域 mask
        freq_dist = torch.abs(log_f.unsqueeze(0) - log_f.unsqueeze(1))
        k = min(self.n_neighbors, n)
        neighbor_mask = torch.full((n, n), float("-inf"), device=device)
        for i in range(n):
            _, idx = freq_dist[i].topk(k, largest=False)
            neighbor_mask[i, idx] = 0.0

        # 结合物理偏置和邻域 mask
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

        # 物理偏置
        attn_bias = self._build_physics_bias(freqs, device)  # [F, F]

        # 跨频率注意力（带物理偏置）
        seq_b = freq_feat.unsqueeze(0)  # [1, F, D]
        attn_out, attn_weights = self.cross_attn(
            seq_b, seq_b, seq_b, attn_mask=attn_bias
        )
        attn_out = attn_out.squeeze(0)  # [F, D]

        # 残差 + 归一化
        out = self.norm1(attn_out + freq_feat)

        # FFN
        out = self.norm2(self.ffn(out) + out)

        # 门控残差：控制 CFA 贡献强度
        gamma = torch.sigmoid(self.gamma)
        out = freq_feat + gamma * (out - freq_feat)

        return out


# ================================================================
#  Main Model
# ================================================================


class FreqAdaptWeighter(nn.Module):
    """
    频率自适应加权模型 — 支持每个频率不等长的谱段

    Parameters
    ----------
    n_features       : 输入特征维度 m
    d_model          : 隐层维度
    n_heads          : attention 头数
    n_freq_neighbors : CFA 每个频率关注的邻居数
    dropout          : dropout rate

    Input
    -----
    input_dict : dict[float, Tensor[n_seg_i, n_features]]
        key = 频率 (Hz), value = 该频率下各谱段的特征
        不同频率的 n_seg_i 可以不同 (e.g. 500, 100, 20)

    Output
    ------
    dict {
        'weights' : dict[float, Tensor[n_seg_i]]  每个频率各谱段的归一化权重
    }
    """

    # 使用 optuna 确定参数 64 - 8 - 4 - 0.07
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
            nn.Linear(n_features + 1, d_model),  # +1: 频率特征
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

        # temp = exp(0.0) = 1.0，让softmax初期更有区分度
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
        """逐频率 mean 池化 → [n_freq, d_model]"""
        return torch.stack([sf.mean(dim=0) for sf in seg_feats.values()], dim=0)

    # ----------------------------------------------------------
    def forward(self, input_dict: Dict[float, torch.Tensor]) -> dict:
        """
        Parameters
        ----------
        input_dict : {freq: Tensor[n_seg_i, n_features]}
                     不同频率的 n_seg_i 可以不同
        """
        freqs = sorted(input_dict.keys())

        # ---- 逐频率编码 (freq concat 到特征) → dict ----
        encoded = {}
        for f in freqs:
            feat = input_dict[f]  # [S_i, n_feat]
            freq_col = torch.full(
                (feat.shape[0], 1), math.log(f), device=feat.device
            )  # [S_i, 1]
            feat_with_freq = torch.cat([feat, freq_col], dim=-1)  # [S_i, n_feat+1]
            encoded[f] = self.encoder(feat_with_freq)  # [S_i, D]

        # 当前层特征 (用 dict 保持频率索引)
        current = {f: encoded[f].clone() for f in freqs}

        # 1. 逐频率:
        for f in freqs:
            current[f] = self.csas(current[f])
            current[f] = self.fcd(current[f], f)

        # 2. 池化 → 跨频率 → 门控融合回段级
        freq_feat = self._pool_freqs(current)  # [F, D]
        freq_feat = self.cft(freq_feat, freqs)  # [F, D]

        # CFA广播: 直接加法 → 门控融合
        for fi, f in enumerate(freqs):
            # 用FusionGate自适应融合
            current[f] = self.fusion_gate(current[f], freq_feat[fi])

        # 3. 逐频率: 权重生成
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
        """推理: 返回 {freq: Tensor[n_seg_i]}"""
        self.eval()
        out = self.forward(input_dict)
        return out["weights"]


# ================================================================
#  Usage Example
# ================================================================

if __name__ == "__main__":
    # ---- 超参数 ----
    n_features = 26

    # ---- 模型 ----
    model = FreqAdaptWeighter(
        n_features=n_features,
    )

    # ---- 虚假数据验证 -------
    print(f"\n{'=' * 65}")
    print("  虚假数据前向传播测试")
    print(f"{'=' * 65}")

    # 构造3个频率，各频率谱段数不同 (模拟真实场景)
    fake_input: Dict[float, torch.Tensor] = {
        0.1: torch.randn(100, n_features),  # 0.1 Hz, 100个谱段
        1.0: torch.randn(300, n_features),  # 1.0 Hz, 300个谱段
        10.0: torch.randn(1000, n_features),  # 10.0 Hz, 1000个谱段
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

    print("\n  前向传播通过 ✓")
