"""
FAW_Net Viewer — PySide6 交互式推理结果查看器
"""

import math
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyqtgraph as pg
import torch
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QLinearGradient
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .model import FreqAdaptWeighter
from .struct import PowerSpectrumMatrix, ResistivityPhase, SinglePSM

# ── 完整 30 维特征标签 (抄自 param.py, 避免导入 param.py) ──
FEATURE_LABELS_30 = [
    "Zxx Amp",
    "Zxx Sin",
    "Zxx Cos",
    "Zyy Amp",
    "Zyy Sin",
    "Zyy Cos",
    "Zxy Amp",
    "Zxy Sin",
    "Zxy Cos",
    "Zyx Amp",
    "Zyx Sin",
    "Zyx Cos",
    "Tzx Amp",
    "Tzx Sin",
    "Tzx Cos",
    "Tzy Amp",
    "Tzy Sin",
    "Tzy Cos",
    "Ex PSD",
    "Ey PSD",
    "Hx PSD",
    "Hy PSD",
    "Alpha Sin",
    "Alpha Cos",
    "Beta Sin",
    "Beta Cos",
    "P11",
    "P12",
    "P21",
    "P22",
]

# 30→20 维索引映射 (关闭 tipper + phase_tensor_angles)
SELECTED_INDICES = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    18,
    19,
    20,
    21,
    26,
    27,
    28,
    29,
]
FEATURE_LABELS_20 = [FEATURE_LABELS_30[i] for i in SELECTED_INDICES]


class FeaturePair(NamedTuple):
    """一个特征对散点图的配置"""

    x_idx: int  # X 轴在 20 维特征中的索引
    y_idx: int  # Y 轴在 20 维特征中的索引
    x_label: str  # X 轴显示标签
    y_label: str  # Y 轴显示标签


# 5 个 Tab × 2 个特征对
FEATURE_PAIRS: list[list[FeaturePair]] = [
    # Tab 1
    [
        FeaturePair(0, 6, FEATURE_LABELS_20[0], FEATURE_LABELS_20[6]),
        FeaturePair(3, 9, FEATURE_LABELS_20[3], FEATURE_LABELS_20[9]),
    ],
    # Tab 2
    [
        FeaturePair(6, 9, FEATURE_LABELS_20[6], FEATURE_LABELS_20[9]),
        FeaturePair(0, 3, FEATURE_LABELS_20[0], FEATURE_LABELS_20[3]),
    ],
    # Tab 3
    [
        FeaturePair(7, 8, FEATURE_LABELS_20[7], FEATURE_LABELS_20[8]),
        FeaturePair(10, 11, FEATURE_LABELS_20[10], FEATURE_LABELS_20[11]),
    ],
    # Tab 4
    [
        FeaturePair(12, 13, FEATURE_LABELS_20[12], FEATURE_LABELS_20[13]),
        FeaturePair(14, 15, FEATURE_LABELS_20[14], FEATURE_LABELS_20[15]),
    ],
    # Tab 5
    [
        FeaturePair(12, 14, FEATURE_LABELS_20[12], FEATURE_LABELS_20[14]),
        FeaturePair(16, 19, FEATURE_LABELS_20[16], FEATURE_LABELS_20[19]),
    ],
]


@dataclass
class StationData:
    """单个测点的推理结果

    - weights: {freq: (n_seg,)} 模型对各谱段分配的 softmax 权重
    - params:  {freq: (n_seg, 20)} 裁剪后的 20 维输入特征
    - raw_psms: 按频点排序的原始 PowerSpectrumMatrix 列表
    - denoised_psms: 各频点加权后合并的 SinglePSM 列表
    - raw_rho_phi / denoised_rho_phi: 原始/加权后的视电阻率和相位
    """

    name: str
    weights: dict[float, np.ndarray]
    params: dict[float, np.ndarray]
    raw_psms: list[PowerSpectrumMatrix]
    denoised_psms: list[SinglePSM]
    raw_rho_phi: list[ResistivityPhase] | None = None
    denoised_rho_phi: list[ResistivityPhase] | None = None
    frequencies: list[float] = field(init=False)

    def __post_init__(self):
        self.frequencies = sorted(self.weights.keys())


class InferenceEngine:
    """加载模型 + PKL → 前向推理 → 返回 StationData"""

    def __init__(self, model_path: str | Path):
        self.model = FreqAdaptWeighter(n_features=20)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    @staticmethod
    def _crop_features(params_30: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
        return {f: m[:, SELECTED_INDICES] for f, m in params_30.items()}

    @staticmethod
    def _to_tensors(params: dict[float, np.ndarray]) -> dict[float, torch.Tensor]:
        return {f: torch.from_numpy(m).float() for f, m in params.items()}

    def _infer(self, tensors: dict[float, torch.Tensor]) -> dict[float, np.ndarray]:
        with torch.no_grad():
            out = self.model(tensors)
        return {f: out["weights"][f].cpu().numpy() for f in out["weights"]}

    @staticmethod
    def _compute_rho_phi(
        raw_psms: list[PowerSpectrumMatrix],
        denoised_psms: list[SinglePSM],
    ) -> tuple[list[ResistivityPhase], list[ResistivityPhase]]:
        raw_rp = [psm.to_spsm().least_squares().to_rho_phi() for psm in raw_psms]
        den_rp = [sp.least_squares().to_rho_phi() for sp in denoised_psms]
        return raw_rp, den_rp

    def load_pkl(self, pkl_path: str | Path) -> StationData:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        name = Path(pkl_path).stem
        if "matrix" not in data or "param" not in data:
            raise ValueError(f"PKL 文件缺少 matrix 或 param 键: {pkl_path}")

        matrix_dict = data["matrix"]
        param_30 = data["param"]

        # 验证 param 维度
        for f, m in param_30.items():
            if m.shape[1] < max(SELECTED_INDICES) + 1:
                raise ValueError(f"{name} 频率 {f} 的特征维度 {m.shape[1]} < 30")

        params_20 = self._crop_features(param_30)
        tensors = self._to_tensors(params_20)
        weights = self._infer(tensors)

        # 对齐 matrix_dict 与 weights 的频点集合
        common_freqs = sorted(set(matrix_dict) & set(weights))
        if not common_freqs:
            raise ValueError(f"{name}: matrix 与 weights 无共同频点")

        raw_psms = []
        for freq in common_freqs:
            matrices = matrix_dict[freq]
            spsms = [
                SinglePSM(freq=freq, window_length=0, seg_num=i, matrix=m)
                for i, m in enumerate(matrices)
            ]
            psm = PowerSpectrumMatrix(
                freq=freq, window_length=0, seg_num=len(matrices), psms=tuple(spsms)
            )
            raw_psms.append(psm)

        denoised_psms = []
        for psm in raw_psms:
            w = weights[psm.freq]
            # 手动加权平均各谱段的交叉功率谱矩阵
            matrices = [s.matrix for s in psm.psms]
            weighted_sum = sum(w[i] * matrices[i] for i in range(len(matrices)))
            total_weight = w.sum()
            avg_matrix = (
                weighted_sum / total_weight if total_weight > 0 else matrices[0]
            )
            denoised_psms.append(
                SinglePSM(
                    freq=psm.freq,
                    window_length=psm.window_length,
                    seg_num=0,
                    matrix=avg_matrix,
                )
            )

        raw_rp, den_rp = self._compute_rho_phi(raw_psms, denoised_psms)

        return StationData(
            name=name,
            weights=weights,
            params=params_20,
            raw_psms=raw_psms,
            denoised_psms=denoised_psms,
            raw_rho_phi=raw_rp,
            denoised_rho_phi=den_rp,
        )

    def load_directory(
        self,
        dir_path: str | Path,
        num: int | None = None,
        parent=None,  # ← 新增：传 MainWindow 进来
    ) -> list[StationData]:
        path = Path(dir_path)
        pkl_files = sorted(path.glob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"在 {dir_path} 中未找到 .pkl 文件")
        if num is not None:
            pkl_files = pkl_files[:num]

        total = len(pkl_files)
        dlg = QProgressDialog("正在加载台站数据…", "取消", 0, total, parent)
        dlg.setWindowTitle("加载进度")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)  # 立即显示，默认要等 4 秒
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setWindowFlag(
            Qt.WindowType.WindowContextHelpButtonHint, False
        )  # 去掉标题栏 "?"

        stations = []
        canceled = False
        for i, pf in enumerate(pkl_files):
            dlg.setValue(i)
            dlg.setLabelText(f"({i + 1}/{total}) {pf.name}")
            QApplication.processEvents()
            if dlg.wasCanceled():
                canceled = True
                break
            try:
                stations.append(self.load_pkl(pf))
            except Exception as e:
                print(f"⚠ 跳过 {pf.name}: {e}")

        dlg.setValue(total)
        dlg.close()
        if canceled:
            print(f"用户取消，已加载 {len(stations)}/{total}")
        return stations


class RhoPhiView(pg.PlotWidget):
    """视电阻率 / 相位去噪前后对比图，支持点击选频和 2/4 分量切换"""

    frequency_selected = Signal(object)  # float

    # 颜色: xx=紫, xy=红, yx=蓝, yy=深绿
    COMP_COLORS = {
        "rxx": "#9b59b6",
        "rxy": "#e74c3c",
        "ryx": "#3498db",
        "ryy": "#27ae60",
        "pxx": "#9b59b6",
        "pxy": "#e74c3c",
        "pyx": "#3498db",
        "pyy": "#27ae60",
    }
    # 符号: xx=△(上三角), xy=○(圆), yx=□(方), yy=▽(下三角)
    COMP_SYMBOLS = {
        "rxx": ("t", 0),
        "rxy": ("o", 0),
        "ryx": ("s", 0),
        "ryy": ("t", 180),
        "pxx": ("t", 0),
        "pxy": ("o", 0),
        "pyx": ("s", 0),
        "pyy": ("t", 180),
    }
    COMP_LABELS = {
        "rxx": "xx",
        "rxy": "xy",
        "ryx": "yx",
        "ryy": "yy",
        "pxx": "xx",
        "pxy": "xy",
        "pyx": "yx",
        "pyy": "yy",
    }

    def __init__(self, title: str = "", is_rho: bool = True, parent=None):
        super().__init__(parent)
        self._is_rho = is_rho
        self._mode = 2
        self._data = None
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._highlight_line: pg.InfiniteLine | None = None

        self.setBackground("w")
        self.setLabel("bottom", "Frequency (Hz)")
        self.setLabel("left", "Resistivity (Ω·m)" if is_rho else "Phase (°)")
        self.setLogMode(x=True, y=True if is_rho else False)
        self.getPlotItem().vb.state["xInverted"] = True
        self.showGrid(x=True, y=True, alpha=0.3)
        self._legend = self.addLegend(size=None, offset=(10, 10))

        self.scene().sigMouseClicked.connect(self._on_click)

    def set_component_mode(self, mode: int):
        self._mode = mode
        self._update_visibility()

    def update_data(self, rho_phi_list_raw, rho_phi_list_denoised, freqs):
        self.clear()
        self._curves = {}
        self._highlight_line = None
        self._data = (list(freqs), rho_phi_list_raw, rho_phi_list_denoised)
        self._plot_all()

    def _plot_all(self):
        if self._data is None:
            return
        freqs, raw_list, den_list = self._data
        comps = (
            ["rxx", "rxy", "ryx", "ryy"]
            if self._is_rho
            else ["pxx", "pxy", "pyx", "pyy"]
        )

        for c in comps:
            raw_vals = [getattr(rp, c) for rp in raw_list]
            den_vals = [getattr(rp, c) for rp in den_list]
            color = self.COMP_COLORS[c]
            sym, rot = self.COMP_SYMBOLS[c]
            label = self.COMP_LABELS[c]

            # raw: 虚线 + 空心符号
            self._curves[f"{c}_raw"] = self.plot(
                freqs,
                raw_vals,
                pen=pg.mkPen(color, width=1.0, style=Qt.DashLine),
                symbol=sym,
                symbolRot=rot,
                symbolSize=5,
                symbolPen=pg.mkPen(color, width=1.2),
                symbolBrush=None,
            )
            # denoised: 实线 + 实心符号 (只有 denoised 进图例)
            self._curves[f"{c}_den"] = self.plot(
                freqs,
                den_vals,
                pen=pg.mkPen(color, width=1.8),
                symbol=sym,
                symbolRot=rot,
                symbolSize=7,
                symbolPen=pg.mkPen(color, width=1.2),
                symbolBrush=color,
                name=label,
            )

        self._update_visibility()
        self.autoRange()

    def _update_visibility(self):
        if self._mode == 2:
            visible = {"rxy", "ryx", "pxy", "pyx"}
        else:
            visible = {"rxx", "rxy", "ryx", "ryy", "pxx", "pxy", "pyx", "pyy"}
        for key, curve in self._curves.items():
            comp = key.rsplit("_", 1)[0]
            curve.setVisible(comp in visible)

    def highlight_freq(self, freq: float | None):
        if self._highlight_line is not None:
            self.removeItem(self._highlight_line)
            self._highlight_line = None
        if freq is not None:
            y_range = self.viewRange()[1]
            self._highlight_line = self.plot(
                [freq, freq],
                [y_range[0], y_range[1]],
                pen=pg.mkPen("#e74c3c", width=2.0, style=Qt.DashLine),
            )
            self._highlight_line.setZValue(50)

    def _on_click(self, event):
        if self._data is None:
            return
        pos = self.getPlotItem().vb.mapSceneToView(event.scenePos())
        click_x = pos.x()
        if click_x <= 0:
            return
        freqs = self._data[0]
        idx = np.argmin(np.abs(np.log10(freqs) - math.log10(click_x)))
        selected = freqs[idx]
        self.frequency_selected.emit(selected)


class ScatterView(pg.PlotWidget):
    """特征对散点图，按模型权重着色 (绿高红低)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pair = None
        self._scatter: pg.ScatterPlotItem | None = None
        self._pair = None

        self.setBackground("w")
        self.showGrid(x=True, y=True, alpha=0.3)

        # 色标: jet (蓝→青→绿→黄→红), top-right, log 0.001~1
        grad = QLinearGradient(0, 1, 0, 0)
        for pos, rgb in [
            (0.00, (0, 0, 128)),
            (0.15, (0, 0, 255)),
            (0.35, (0, 255, 255)),
            (0.50, (0, 200, 0)),
            (0.65, (255, 255, 0)),
            (0.85, (255, 0, 0)),
            (1.00, (128, 0, 0)),
        ]:
            grad.setColorAt(pos, QColor(*rgb))
        self._cbar = pg.GradientLegend((18, 70), (-35, 10))
        self._cbar.setGradient(grad)
        self._cbar.setLabels({"1": 0, "0.001": 1})
        self._cbar.setParentItem(self.getPlotItem().vb)
        self._cbar.setZValue(1000)

    def set_feature_pair(self, pair: FeaturePair):
        self.setLabel("bottom", pair.x_label)
        self.setLabel("left", pair.y_label)
        self._pair = pair

    def update_data(self, features: np.ndarray, weights: np.ndarray):
        """features: (n_seg, 20), weights: (n_seg,)"""
        if self._scatter is not None:
            self.removeItem(self._scatter)
        self._scatter = None
        if self._pair is None or len(features) == 0:
            return

        x = features[:, self._pair.x_idx]
        y = features[:, self._pair.y_idx]
        w = weights

        # 对数归一化: log10(weight) ∈ [-3, 0] 对应 0.001~1
        log_w_low = -3.0
        log_w_high = 0.0

        spots = []
        order = np.argsort(w)
        for idx in order:
            if w[idx] < 0.001:
                # 低权重: 黑色空圈 (透明填充, 小一号)
                spots.append(
                    {
                        "pos": (x[idx], y[idx]),
                        "size": 4,
                        "brush": None,
                        "pen": pg.mkPen(0, 0, 0, 160, width=1.0),
                    }
                )
            else:
                val = np.clip(
                    (np.log10(w[idx]) - log_w_low) / (log_w_high - log_w_low), 0, 1
                )
                spots.append(
                    {
                        "pos": (x[idx], y[idx]),
                        "size": 7,
                        "brush": self._weight_color(val),
                        "pen": pg.mkPen(None),
                    }
                )

        self._scatter = pg.ScatterPlotItem(spots=spots)
        self.addItem(self._scatter)
        self.autoRange()

    @staticmethod
    def _weight_color(ratio: float):
        """ratio ∈ [0,1] → jet 色标"""
        # jet: blue(0) → cyan(0.25) → green(0.5) → yellow(0.75) → red(1)
        if ratio < 0.25:
            r = 0.0
            g = ratio * 4.0
            b = 1.0
        elif ratio < 0.5:
            r = 0.0
            g = 1.0
            b = 1.0 - (ratio - 0.25) * 4.0
        elif ratio < 0.75:
            r = (ratio - 0.5) * 4.0
            g = 1.0
            b = 0.0
        else:
            r = 1.0
            g = 1.0 - (ratio - 0.75) * 4.0
            b = 0.0
        return pg.mkBrush(int(r * 255), int(g * 255), int(b * 255), 200)


class ControlBar(QWidget):
    """顶部控制栏: 测点导航 + Tab 切换 + 分量模式 + 自动播放"""

    station_changed = Signal(int)  # delta: +1 or -1
    tab_changed = Signal(int)  # 0-based tab index
    component_mode_changed = Signal(int)  # 2 or 4
    open_directory = Signal()
    play_toggled = Signal(bool)  # True=play, False=pause

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        # 测点导航
        self.btn_prev = QPushButton("\u2190")
        self.btn_prev.setFixedWidth(32)
        self.lbl_station = QLabel("\u2014")
        self.lbl_station.setMinimumWidth(100)
        self.btn_next = QPushButton("\u2192")
        self.btn_next.setFixedWidth(32)
        self.btn_dir = QPushButton("打开目录")

        # Tab 按钮组 (1-5)
        self.tab_btns: list[QPushButton] = []
        for i in range(5):
            btn = QPushButton(str(i + 1))
            btn.setCheckable(True)
            btn.setFixedSize(30, 26)
            self.tab_btns.append(btn)

        # 分量模式
        self.btn_2comp = QPushButton("2分量")
        self.btn_2comp.setCheckable(True)
        self.btn_2comp.setChecked(True)
        self.btn_4comp = QPushButton("4分量")
        self.btn_4comp.setCheckable(True)

        # 自动播放
        self.btn_play = QPushButton("\u25b6 自动")
        self.btn_play.setCheckable(True)
        self.cmb_interval = QComboBox()
        self.cmb_interval.addItems(["1s", "2s", "3s", "5s"])
        self.cmb_interval.setCurrentIndex(2)

        # 组装
        layout.addWidget(self.btn_prev)
        layout.addWidget(self.lbl_station)
        layout.addWidget(self.btn_next)
        layout.addWidget(self.btn_dir)
        layout.addWidget(self._sep())
        for btn in self.tab_btns:
            layout.addWidget(btn)
        layout.addWidget(self._sep())
        layout.addWidget(self.btn_2comp)
        layout.addWidget(self.btn_4comp)
        layout.addWidget(self._sep())
        layout.addWidget(self.btn_play)
        layout.addWidget(QLabel("间隔:"))
        layout.addWidget(self.cmb_interval)
        layout.addStretch()

        # 信号连接
        self.btn_prev.clicked.connect(lambda: self.station_changed.emit(-1))
        self.btn_next.clicked.connect(lambda: self.station_changed.emit(1))
        self.btn_dir.clicked.connect(self.open_directory.emit)
        self.btn_play.toggled.connect(self._on_play_toggled)
        self.btn_2comp.clicked.connect(lambda: self._set_comp_mode(2))
        self.btn_4comp.clicked.connect(lambda: self._set_comp_mode(4))
        for i, btn in enumerate(self.tab_btns):
            btn.clicked.connect(lambda checked, idx=i: self._on_tab(idx))

        # 分组 2/4 分量按钮 (互斥)
        self._comp_group = [self.btn_2comp, self.btn_4comp]

    @staticmethod
    def _sep():
        lbl = QLabel("|")
        lbl.setStyleSheet("color: #555;")
        return lbl

    def set_station_name(self, name: str):
        self.lbl_station.setText(name)

    def set_tab(self, idx: int):
        for i, btn in enumerate(self.tab_btns):
            btn.setChecked(i == idx)

    def get_interval_ms(self) -> int:
        return int(self.cmb_interval.currentText().replace("s", "")) * 1000

    def _on_tab(self, idx: int):
        self.set_tab(idx)
        self.tab_changed.emit(idx)

    def _set_comp_mode(self, mode: int):
        self.btn_2comp.setChecked(mode == 2)
        self.btn_4comp.setChecked(mode == 4)
        self.component_mode_changed.emit(mode)

    def _on_play_toggled(self, checked: bool):
        self.btn_play.setText("\u23f8" if checked else "\u25b6 自动")
        self.play_toggled.emit(checked)


class MainWindow(QMainWindow):
    """主窗口: 控制栏 + 2x2 网格 + 状态栏"""

    def __init__(self, engine: InferenceEngine, stations: list[StationData]):
        super().__init__()
        self.engine = engine
        self.stations = stations
        self.current_idx = 0
        self.current_tab = 0
        self.current_freq: float | None = None

        self.setWindowTitle("FAW_Net Viewer — MT 选谱权重查看器")
        self.resize(1400, 900)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 控制栏
        self.control = ControlBar()
        main_layout.addWidget(self.control)

        # 2x2 网格
        grid = QGridLayout()
        grid.setSpacing(4)
        main_layout.addLayout(grid, 1)

        self.rho_view = RhoPhiView(is_rho=True)
        self.phi_view = RhoPhiView(is_rho=False)
        self.scatter_a = ScatterView()
        self.scatter_b = ScatterView()

        grid.addWidget(self.rho_view, 0, 0)
        grid.addWidget(self.scatter_a, 0, 1)
        grid.addWidget(self.phi_view, 1, 0)
        grid.addWidget(self.scatter_b, 1, 1)

        # 状态栏
        status_bar = QStatusBar()
        status_bar.setStyleSheet("QStatusBar{font-size:11px; color:#aaa;}")
        self.setStatusBar(status_bar)
        self.status_label = QLabel("Ready")
        status_bar.addWidget(self.status_label)

        # 自动播放
        self._timer = QTimer()
        self._timer.timeout.connect(self._auto_next)

        # 信号连接
        self.control.station_changed.connect(self._on_station_delta)
        self.control.tab_changed.connect(self._on_tab)
        self.control.component_mode_changed.connect(self._on_component_mode)
        self.control.open_directory.connect(self._open_dir)
        self.control.play_toggled.connect(self._on_play)
        self.rho_view.frequency_selected.connect(self._on_freq_selected)
        self.phi_view.frequency_selected.connect(self._on_freq_selected)

        # 初始显示
        self._on_tab(0)
        if self.stations:
            self.show_station(0)

    # ── 快捷键 ──

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self._on_freq_delta(1)  # -> 下一个(更高频)
        elif event.key() == Qt.Key_Right:
            self._on_freq_delta(-1)  # -> 上一个(更低频)
        elif event.key() == Qt.Key_Up:
            self._on_station_delta(-1)
        elif event.key() == Qt.Key_Down:
            self._on_station_delta(1)
        elif Qt.Key_1 <= event.key() <= Qt.Key_5:
            self._on_tab(event.key() - Qt.Key_1)
        elif event.key() == Qt.Key_Space:
            self.control.btn_play.toggle()
        else:
            super().keyPressEvent(event)

    # ── 测点 ──

    def show_station(self, idx: int):
        if idx < 0 or idx >= len(self.stations):
            return
        self.current_idx = idx
        station = self.stations[idx]
        self.control.set_station_name(station.name)

        self.rho_view.update_data(
            station.raw_rho_phi,
            station.denoised_rho_phi,
            station.frequencies,
        )
        self.phi_view.update_data(
            station.raw_rho_phi,
            station.denoised_rho_phi,
            station.frequencies,
        )

        # 默认选中中间频率
        mid_freq = station.frequencies[len(station.frequencies) // 2]
        self._on_freq_selected(mid_freq)
        self._update_status()

    def _on_station_delta(self, delta: int):
        n = len(self.stations)
        if n == 0:
            return
        self.show_station((self.current_idx + delta) % n)

    # ── 频率 ──

    def _on_freq_selected(self, freq: float):
        self.current_freq = freq
        station = self.stations[self.current_idx]

        features = station.params[freq]
        weights = station.weights[freq]

        self.scatter_a.update_data(features, weights)
        self.scatter_b.update_data(features, weights)
        self.rho_view.highlight_freq(freq)
        self.phi_view.highlight_freq(freq)
        self._update_status()

    def _on_freq_delta(self, delta: int):
        station = self.stations[self.current_idx]
        freqs = station.frequencies
        if self.current_freq is None:
            self._on_freq_selected(freqs[len(freqs) // 2])
            return
        try:
            idx = freqs.index(self.current_freq)
        except ValueError:
            idx = len(freqs) // 2
        idx = (idx + delta) % len(freqs)
        self._on_freq_selected(freqs[idx])

    # ── Tab ──

    def _on_tab(self, idx: int):
        self.current_tab = idx
        self.control.set_tab(idx)
        pairs = FEATURE_PAIRS[idx]
        self.scatter_a.set_feature_pair(pairs[0])
        self.scatter_b.set_feature_pair(pairs[1])
        if self.current_freq is not None:
            self._on_freq_selected(self.current_freq)

    # ── 分量模式 ──

    def _on_component_mode(self, mode: int):
        self.rho_view.set_component_mode(mode)
        self.phi_view.set_component_mode(mode)

    # ── 自动播放 ──

    def _on_play(self, playing: bool):
        if playing:
            self._timer.start(self.control.get_interval_ms())
        else:
            self._timer.stop()

    def _auto_next(self):
        self._timer.stop()
        self._on_station_delta(1)
        if self.control.btn_play.isChecked():
            self._timer.start(self.control.get_interval_ms())

    # ── 打开目录 ──

    def _open_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择 PKL 目录")
        if not dir_path:
            return
        try:
            new_stations = self.engine.load_directory(dir_path)
            if not new_stations:
                QMessageBox.warning(self, "警告", "未找到可用的 .pkl 文件")
                return
            self.stations = new_stations
            self.show_station(0)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载目录失败: {e}")

    # ── 状态 ──

    def _update_status(self):
        station = self.stations[self.current_idx]
        freq = self.current_freq
        if freq is None:
            return
        n_seg = len(station.weights[freq])
        w = station.weights[freq]
        self.status_label.setText(
            f"{station.name} | Freq: {freq:.6g} Hz | Segments: {n_seg} | "
            f"Weight: [{w.min():.2e}, {w.max():.2e}]"
        )


def main():

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    model_path = Path(__file__).parent / "best_model.pth"
    if not model_path.exists():
        QMessageBox.critical(None, "错误", "未找到 best_model.pth")
        sys.exit(1)

    engine = InferenceEngine(model_path)

    default_dir = Path(__file__).parent / "pkl"
    if default_dir.exists():
        stations = engine.load_directory(default_dir)
    else:
        stations = []

    if not stations:
        dir_path = QFileDialog.getExistingDirectory(None, "选择 PKL 目录")
        if dir_path:
            stations = engine.load_directory(dir_path)

    window = MainWindow(engine, stations)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    print("Starting viewer...")
    main()
