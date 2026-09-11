import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .t_calc import calc_rho_phs


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler=None,
        save_dir: Path | None = None,
        save_interval: int = 5,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_interval = save_interval

        # Create the save directory if needed
        self.save_dir = save_dir
        if self.save_dir is not None and not self.save_dir.exists():
            print(f"Creating save directory: {self.save_dir}")
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def compute_cell_loss(self, site_data: dict) -> torch.Tensor:
        """
        [Overridable] Process a single site (or batch unit): run a forward pass
        and compute the loss.

        Subclasses may override this method to support different model I/O
        structures, multi-task losses, etc.
        """
        params_dict = site_data["param"]
        matrix_dict = site_data["matrix"]
        target_dict = site_data["target"]

        model_output = self.model(params_dict)

        loss = self.criterion(
            model_output=model_output,
            matrix_dict=matrix_dict,
            target_dict=target_dict,
            rate=1.0,
            device=self.device,
        )
        return loss.squeeze()

    def _calc_roughness(self, freq, v) -> float:
        log_f = np.log10(freq)
        v = np.array(v)
        d2y = v[:-2] - 2.0 * v[1:-1] + v[2:]
        delta_sq = ((log_f[2:] - log_f[:-2]) / 2.0) ** 2
        delta_sq = np.maximum(delta_sq, 1e-16)
        return float(np.sqrt(np.mean((d2y / delta_sq) ** 2)))

    def compute_roughness(self, site_data: dict) -> float:
        params_dict = site_data["param"]
        matrix_dict = site_data["matrix"]
        target_dict = site_data["target"]

        model_output = self.model(params_dict)

        weights_dict = model_output["weights"]
        sorted_freqs = sorted(weights_dict.keys(), reverse=False)

        rxys, ryxs, pxys, pyxs, freqs = [], [], [], [], []
        for f in sorted_freqs:
            w_raw = weights_dict[f]
            if w_raw.shape[0] == 0:
                continue

            weights = w_raw.unsqueeze(-1).unsqueeze(-1)
            matrix = matrix_dict[f]
            s_avg = torch.sum(weights * matrix, dim=0)

            # MSE supervision target
            res = calc_rho_phs(f, s_avg, is_log=True)
            rxy, ryx, pxy, pyx = res.detach().cpu().numpy()
            rxys.append(rxy)
            ryxs.append(ryx)
            pxys.append(pxy)
            pyxs.append(pyx)
            freqs.append(f)

        r1 = self._calc_roughness(freqs, rxys)
        r2 = self._calc_roughness(freqs, ryxs)
        r3 = self._calc_roughness(freqs, pxys)
        r4 = self._calc_roughness(freqs, pyxs)

        return np.mean([r1, r2, r3, r4])

    def train_epoch(self, dataloader: DataLoader, epoch_idx: int) -> float:
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Train Epoch {epoch_idx}",
            leave=False,
        )

        for batch_idx, batch in pbar:
            self.optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=self.device)

            # Call the extracted compute unit
            for site_data in batch:
                batch_loss += self.compute_cell_loss(site_data)

            batch_loss = batch_loss / len(batch)
            batch_loss.backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Record and update progress-bar info
            current_loss = batch_loss.item()
            total_loss += current_loss
            avg_loss = total_loss / (batch_idx + 1)

            pbar.set_postfix(
                {"loss": f"{current_loss:.4f}", "avg_loss": f"{avg_loss:.4f}"}
            )

        return total_loss / len(dataloader)

    def validate_epoch(
        self, dataloader: DataLoader, epoch_idx: int
    ) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        roughness = 0.0

        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Val Epoch {epoch_idx}",
            leave=False,
        )

        with torch.no_grad():
            for batch_idx, batch in pbar:
                batch_loss = torch.tensor(0.0, device=self.device)
                r_cell = 0.0

                # Call the extracted compute unit
                for site_data in batch:
                    batch_loss += self.compute_cell_loss(site_data)
                    r_cell += self.compute_roughness(site_data)

                batch_loss = batch_loss / len(batch)
                roughness += r_cell / len(batch)

                # Record and update progress-bar info
                current_loss = batch_loss.item()
                total_loss += current_loss
                avg_loss = total_loss / (batch_idx + 1)

                pbar.set_postfix(
                    {
                        "loss": f"{current_loss:.4f}",
                        "avg_loss": f"{avg_loss:.4f}",
                        "avg_roughness": f"{roughness / (batch_idx + 1):.4f}",
                    }
                )

        return total_loss / len(dataloader), roughness / len(dataloader)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        if self.save_dir is None:
            return

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        last_path = self.save_dir / "last_checkpoint.pth"
        torch.save(checkpoint, last_path)

        if (epoch + 1) % self.save_interval == 0:
            interval_path = self.save_dir / f"checkpoint_e{epoch + 1}.pth"
            torch.save(checkpoint, interval_path)

        if is_best:
            best_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)

    def load_model(self, checkpoint_path: Path):
        """Load the model state dict"""
        if not checkpoint_path.exists():
            print(f"⚠️ Warning: checkpoint file not found at {checkpoint_path}; training will start from scratch.")
            return
        print(f"🔄 Loading model: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def load_checkpoint(self, checkpoint_path: Path, only_model: bool = True):
        """Load a checkpoint, restore model/optimizer/scheduler state, and return the next epoch start index"""
        if not checkpoint_path.exists():
            print(f"⚠️ Warning: checkpoint file not found at {checkpoint_path}; training will start from scratch.")
            return

        print(f"🔄 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if not only_model:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # checkpoint["epoch"] is the last completed epoch index, so the next start epoch is epoch + 1
        start_epoch = checkpoint.get("epoch", -1) + 1
        print(f"✅ State restored successfully; training will continue from Epoch {start_epoch + 1}.")
        return start_epoch

    def save_history(self, history: dict):
        if self.save_dir is None:
            return
        history_path = self.save_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)

    def load_history(self) -> dict:
        """Load previous training history so plotted curves stay continuous"""
        if self.save_dir is None:
            return {"train_loss": [], "val_loss": [], "roughness": [], "lr": []}

        history_path = self.save_dir / "training_history.json"
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"train_loss": [], "val_loss": [], "roughness": [], "lr": []}

    def save_history_plot(self, history: dict):
        if self.save_dir is None:
            return
        with plt.style.context(["science", "no-latex"]):
            plt.figure(figsize=(10, 6))
            plt.plot(history["train_loss"], label="Train Loss")
            if "val_loss" in history and len(history["val_loss"]) > 0:
                plt.plot(history["val_loss"], label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(self.save_dir / "training_history.png")
            plt.close()

    def run(
        self,
        train_loader: DataLoader,
        epochs: int,
        val_loader: DataLoader | None = None,
        resume_checkpoint: Path | None = None,
        save_checkpoint: bool = True,
        save_history: bool = True,
        save_history_plot: bool = True,
    ) -> dict[str, list[float]]:
        start_epoch = 0
        history = {"train_loss": [], "val_loss": [], "roughness": [], "lr": []}

        # If a resume checkpoint path is provided, load it
        if resume_checkpoint is not None:
            start_epoch = self.load_checkpoint(resume_checkpoint, only_model=False)
            history = self.load_history()

        print(f"🚀 Starting training on {self.device}, planned total epochs: {epochs}...")
        if self.save_dir is not None:
            if save_checkpoint or save_history:
                print(f"📁 Checkpoints/history will be saved to: {self.save_dir.absolute()}")
        else:
            print("📁 No save directory configured")

        # Try to restore the best loss from history
        best_loss = float("inf")
        if history.get("val_loss"):
            best_loss = min(history["val_loss"])
        elif history.get("train_loss"):
            best_loss = min(history["train_loss"])
        elif history.get("roughness"):
            best_loss = min(history["roughness"])

        # Loop from start_epoch
        for epoch in range(start_epoch, epochs):
            current_lr = self.optimizer.param_groups[0]["lr"]

            train_loss = self.train_epoch(train_loader, epoch + 1)
            history["train_loss"].append(train_loss)
            history["lr"].append(current_lr)

            val_loss = None
            val_loss_str = ""
            if val_loader is not None:
                val_loss, roughness = self.validate_epoch(val_loader, epoch + 1)
                history["val_loss"].append(val_loss)
                history["roughness"].append(roughness)

                val_loss_str = f"| Val Loss: {val_loss:.6f} "

            current_loss = val_loss if val_loss is not None else train_loss

            is_best = current_loss < best_loss
            if is_best:
                best_loss = current_loss
                best_msg = "(🔥 New Best!)"
            else:
                best_msg = ""

            # Controlled saving
            if save_checkpoint:
                self.save_checkpoint(epoch, is_best)
            if save_history:
                self.save_history(history)

            if self.scheduler is not None:
                self.scheduler.step()

            print(
                f"Epoch [{epoch + 1:03d}/{epochs:03d}] | "
                f"Train Loss: {train_loss:.6f} {val_loss_str}| "
                f"Roughness: {roughness:.6f} | "
                f"LR: {current_lr:.6e} {best_msg}"
            )

        if save_history_plot:
            self.save_history_plot(history)

        print("🎉 Training Complete!")
        return history
