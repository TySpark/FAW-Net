from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from .cdataset import CustomDataset, create_feature_selector, split_dataset
from .loss import Loss
from .model import FreqAdaptWeighter
from .trainer import Trainer

if __name__ == "__main__":
    # 1. 基础配置
    devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    batch_size = 4
    max_lr = 3e-3
    min_lr = 4e-5
    weight_decay = 1e-4
    save_dir = Path(__file__).absolute().parent / "checkpoints"
    data_dir = Path(__file__).absolute().parent / "pkl"

    pre_deal_feature = create_feature_selector(
        use_impedance=True,
        use_tipper=False,
        use_psd=True,
        use_phase_tensor_angles=False,
        use_phase_tensor_main=True,
    )

    dummy_input = torch.zeros((1, 30))
    actual_feature_dim = pre_deal_feature(dummy_input).shape[1]
    print(f"实际特征维度: {actual_feature_dim}")

    # 2. 准备数据
    dataset = CustomDataset(
        data_dir=data_dir,
        max_num=100,
        device=devices,
        pre_deal_feature=pre_deal_feature,
    )

    train_loader, val_loader = split_dataset(dataset, batch_size=4)

    # 3. 初始化模型与损失
    model = FreqAdaptWeighter(n_features=actual_feature_dim).to(devices)
    criterion = Loss().to(devices)

    # 4. 初始化优化器与调度器
    optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    # 5. 实例化并运行 Trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=devices,
        scheduler=scheduler,
        save_dir=save_dir,
    )

    trainer.run(train_loader=train_loader, epochs=epochs, val_loader=val_loader)
