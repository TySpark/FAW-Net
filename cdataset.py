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
# 1. Dynamic feature-selection generator
# ==========================================
def create_feature_selector(
    use_impedance: bool,
    use_tipper: bool,
    use_psd: bool,
    use_phase_tensor_angles: bool,
    use_phase_tensor_main: bool,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build a feature-slicing function controlled by physical-module switches.
    The returned function takes an (n, 30) Tensor and returns an (n, selected_dim) Tensor.
    """
    # Follow the feature-assembly order of the to_deal_two method
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

    # Convert to a tensor index to avoid recreating it inside forward
    idx_tensor = torch.tensor(indices, dtype=torch.long)

    def selector(x: torch.Tensor) -> torch.Tensor:
        # Move idx_tensor to the same device as the data before slicing
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
        Initialize the Dataset
        :param data_dir: directory containing all pkl datasets
        :param max_num: limit on the maximum number of datasets used, for partial testing
        :param min_freq: minimum frequency used to filter datasets
        :param device: device; default None selects automatically
        :param pre_deal_feature: feature-preprocessing function; takes a torch.Tensor and returns a torch.Tensor
        """
        self.data_dir = Path(data_dir)
        # Collect all pkl file paths under the directory
        self.pkl_files = list(self.data_dir.glob("*.pkl"))

        if max_num is not None:
            self.pkl_files = self.pkl_files[:max_num]

        self.min_freq = min_freq

        if len(self.pkl_files) == 0:
            raise ValueError(f"No .pkl files found in directory {self.data_dir}!")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.dtype = dtype

        self.pre_deal_feature = pre_deal_feature

    def __len__(self):
        # Dataset length equals the number of sites (number of pkl files)
        return len(self.pkl_files)

    def __getitem__(self, idx: int) -> dict:  # type: ignore
        """
        Get all data for one site.
        To interface with the neural network and MTLoss, numpy arrays are converted
        directly to PyTorch Tensors.
        """
        pkl_path = self.pkl_files[idx]
        targets, matrixs, params = load_site_data(pkl_path)

        # Convert dictionary values to Tensors
        tensor_targets = {}
        tensor_matrixs = {}
        tensor_params = {}

        for freq in matrixs.keys():
            if self.min_freq is not None and freq < self.min_freq:
                continue

            # target: (rxy, pxy, ryx, pyx) → float64 tensor of shape (4,)
            tensor_targets[freq] = torch.tensor(
                targets[freq], dtype=self.dtype, device=self.device
            )

            # matrix: power spectral matrix → float64 tensor of shape (N, 7, 7)
            tensor_matrixs[freq] = torch.tensor(
                matrixs[freq], dtype=self.dtype, device=self.device
            )

            # params: input data → float64 tensor of shape (N, features)
            # Convert to a PyTorch Tensor and store
            feature = torch.tensor(params[freq], dtype=self.dtype, device=self.device)
            if self.pre_deal_feature is not None:
                feature = self.pre_deal_feature(feature)
            tensor_params[freq] = feature

        return {
            "site_name": pkl_path.stem,  # keep the site name for debugging
            "target": tensor_targets,  # {freq: Tensor(4,)}
            "matrix": tensor_matrixs,  # {freq: Tensor(N, 7, 7)}
            "param": tensor_params,  # {freq: Tensor(N, channels)}
        }


# ====================== Collate Function ======================
def site_collate_fn(batch):
    """
    Each site's data is a dict of multiple tensors of different lengths, so
    PyTorch's default_collate cannot handle it.
    Return the batch as-is (a list of dicts) and iterate over it in the training loop.
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
    Split the dataset into training and validation sets by ratio.
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


# ====================== Test code ======================


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
    # Assume the pkl files are saved in this directory
    test_dir = Path(r"F:\MainMission\DLSpec\temp_data")

    # 1. Instantiate the Dataset
    dataset = CustomDataset(test_dir)
    print(f"Found {len(dataset)} site datasets in total.")

    # 2. Test reading sample 0
    sample = dataset[0]
    print(f"Successfully loaded site: {sample['site_name']}")

    # Inspect shapes at one frequency
    test_freq = list(sample["target"].keys())[0]
    print(f"Target shape at {test_freq} Hz: {sample['target'][test_freq].shape}")
    print(f"Matrix shape at {test_freq} Hz: {sample['matrix'][test_freq].shape}")
    print(f"Param  shape at {test_freq} Hz: {sample['param'][test_freq].shape}")

    # 3. Build the DataLoader
    # Note: batch_size may be greater than 1, but the custom site_collate_fn must be used
    train_loader, val_loader = split_dataset(
        dataset, batch_size=2, val_ratio=0.0, seed=42, shuffle=True
    )

    for batch_idx, batch in enumerate(train_loader):
        print(f"Batch {batch_idx}: contains {len(batch)} sites.")
        break
