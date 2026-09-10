import pickle
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split


def load_site_data(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        dataset_dict = pickle.load(f)

    edi_target = dataset_dict["target"]
    matrix = dataset_dict["matrix"]
    origin_param = dataset_dict["param"]

    return edi_target, matrix, origin_param


# ==========================================
# 1. 动态特征筛选生成器
# ==========================================
def create_feature_selector(
    use_impedance: bool,
    use_tipper: bool,
    use_psd: bool,
    use_phase_tensor_angles: bool,
    use_phase_tensor_main: bool,
) -> Callable[[torch.Tensor], torch.Tensor]:
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

    # 转为 tensor 索引，避免在 forward 里反复创建
    idx_tensor = torch.tensor(indices, dtype=torch.long)

    def selector(x: torch.Tensor) -> torch.Tensor:
        # 将 idx_tensor 放到与数据相同的设备上再切片
        return x[:, idx_tensor.to(x.device)]

    return selector


# ====================== Dataset ======================
class CustomDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        max_num: int | None = None,
        min_freq: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        pre_deal_feature: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """
        初始化 Dataset
        :param data_dir: 存放所有 pkl 数据集的目录
        :param max_num: 限制使用的数据集最大数量，用于部分测试
        :param min_freq: 最小频率，用于过滤数据集
        :param device: 设备，默认为 None，自动选择
        :param pre_deal_feature: 预处理特征的函数，输入为 torch.Tensor，输出为 torch.Tensor
        """
        self.data_dir = Path(data_dir)
        # 获取目录下所有的 pkl 文件路径
        self.pkl_files = list(self.data_dir.glob("*.pkl"))

        if max_num is not None:
            self.pkl_files = self.pkl_files[:max_num]

        self.min_freq = min_freq

        if len(self.pkl_files) == 0:
            raise ValueError(f"在目录 {self.data_dir} 中没有找到任何 .pkl 文件！")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.dtype = dtype

        self.pre_deal_feature = pre_deal_feature

    def __len__(self):
        # 数据集的长度就是站点的数量（pkl文件的数量）
        return len(self.pkl_files)

    def __getitem__(self, idx: int) -> dict:  # type: ignore
        """
        获取一个站点的所有数据。
        为了配合神经网络和你的 MTLoss，这里直接将 numpy 数组转换为 PyTorch Tensor。
        """
        pkl_path = self.pkl_files[idx]
        targets, matrixs, params = load_site_data(pkl_path)

        # 将字典中的值转换为 Tensor
        tensor_targets = {}
        tensor_matrixs = {}
        tensor_params = {}

        for freq in matrixs.keys():
            if self.min_freq is not None and freq < self.min_freq:
                continue

            # target: (rxy, pxy, ryx, pyx) -> 形状 (4,) 的 float64 tensor
            tensor_targets[freq] = torch.tensor(
                targets[freq], dtype=self.dtype, device=self.device
            )

            # matrix: 功率谱矩阵 -> 形状 (N, 7, 7) 的 float64 tensor
            tensor_matrixs[freq] = torch.tensor(
                matrixs[freq], dtype=self.dtype, device=self.device
            )

            # params: 输入数据 -> 形状为（N, features）的 float64 tensor
            # 转换为 PyTorch Tensor 并存储
            feature = torch.tensor(params[freq], dtype=self.dtype, device=self.device)
            if self.pre_deal_feature is not None:
                feature = self.pre_deal_feature(feature)
            tensor_params[freq] = feature

        return {
            "site_name": pkl_path.stem,  # 记录一下站点名字方便调试
            "target": tensor_targets,  # {freq: Tensor(4,)}
            "matrix": tensor_matrixs,  # {freq: Tensor(N, 7, 7)}
            "param": tensor_params,  # {freq: Tensor(N, channels)}
        }


# ====================== Collate Function ======================
def site_collate_fn(batch):
    """
    因为每个站点的数据是包含了多个不同长度 tensor 的字典，
    PyTorch 默认的 default_collate 无法处理。
    我们直接返回这个 batch（一个包含字典的列表）即可，在 train 循环中去遍历它。
    """
    return batch


def split_dataset(
    dataset: Dataset,
    batch_size: int = 1,
    val_ratio: float = 0.2,
    seed: int = 42,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    将数据集按比例切分为训练集和验证集。
    """
    total_size = len(dataset)  # type: ignore
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=site_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=site_collate_fn,
    )

    print(f"Dataset split: {train_size} for training, {val_size} for validation.")
    return train_loader, val_loader


# ====================== 测试代码 ======================


def loader_dataset(
    dataset: Dataset, batch_size: int, shuffle: bool = True
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=site_collate_fn,
    )


if __name__ == "__main__":
    # 假设你的 pkl 文件保存在这个目录
    test_dir = Path(r"F:\MainMission\DLSpec\temp_data")

    # 1. 实例化 Dataset
    dataset = CustomDataset(test_dir)
    print(f"总共找到 {len(dataset)} 个站点数据。")

    # 2. 测试读取第 0 个样本
    sample = dataset[0]
    print(f"成功读取站点: {sample['site_name']}")

    # 取一个频率看看形状
    test_freq = list(sample["target"].keys())[0]
    print(f"频率 {test_freq} Hz 的 Target 形状: {sample['target'][test_freq].shape}")
    print(f"频率 {test_freq} Hz 的 Matrix 形状: {sample['matrix'][test_freq].shape}")
    print(f"频率 {test_freq} Hz 的 Param  形状: {sample['param'][test_freq].shape}")

    # 3. 构建 DataLoader
    # 注意：batch_size 可以大于 1，但必须使用自定义的 site_collate_fn
    train_loader, val_loader = split_dataset(
        dataset, batch_size=2, val_ratio=0.0, seed=42, shuffle=True
    )

    for batch_idx, batch in enumerate(train_loader):
        print(f"Batch {batch_idx}: 包含 {len(batch)} 个站点。")
        break
