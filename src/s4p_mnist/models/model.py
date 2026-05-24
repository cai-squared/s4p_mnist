from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import joblib
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, TensorDataset

from s4p_mnist.models.base import BaseModel

try:
    # Import Rich progress only when available; keep optional so imports don't fail
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
except Exception:  # pragma: no cover - optional dependency
    Progress = None  # type: ignore

# Provide a static typing alias so Pylance/typing tools can resolve the
# `Progress` type without requiring `rich` at runtime.
if TYPE_CHECKING:
    from rich.progress import Progress as RichProgress  # type: ignore
    from rich.progress import TaskID as RichTaskID  # type: ignore
else:
    RichProgress = Any  # type: ignore
    RichTaskID = Any  # type: ignore

_BUNDLE_KIND = "s4p_mnist_torch_cnn_v1"


class MnistConvNet(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.head(self.body(x)))


class Model(BaseModel):
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        dropout = float(self.config.get("dropout", 0.3))
        self._net = MnistConvNet(num_classes=10, dropout=dropout)
        self._fitted = False

    def _device(self) -> torch.device:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _is_docker() -> bool:
        """Check if running inside a Docker container."""
        return Path("/.dockerenv").exists()

    @staticmethod
    def _prepare_x(X: np.ndarray) -> torch.Tensor:
        x = np.asarray(X, dtype=np.float32)
        if x.ndim == 2 and x.shape[1] == 784:
            x = x.reshape(-1, 1, 28, 28)
        elif x.ndim == 3 and x.shape[1:] == (28, 28):
            x = x[:, None, :, :]
        elif x.ndim == 4 and x.shape[1:] == (1, 28, 28):
            pass
        else:
            raise ValueError(
                "Expected X with shape (N, 784), (N, 28, 28), or (N, 1, 28, 28)."
            )
        if float(x.max()) > 2.0:
            x = x / 255.0
        x = (x - 0.1307) / 0.3081
        return cast(
            torch.Tensor,
            torch.from_numpy(np.ascontiguousarray(x)),
        )

    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        progress: RichProgress | None = None,
        batch_task_id: RichTaskID | None = None,
    ) -> None:
        self._net.train()
        non_blocking = device.type == "cuda"
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=non_blocking)
            yb = yb.to(device, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            logits = self._net(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            if progress is not None and batch_task_id is not None:
                try:
                    progress.advance(batch_task_id)
                except Exception:
                    pass

    def fit(self, X: Any, y: Any, *, show_progress: bool = False) -> Model:
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy.ndarray of pixels.")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy.ndarray of labels.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")

        device = self._device()
        self._net.to(device)
        xt = self._prepare_x(X)
        yt = torch.from_numpy(np.asarray(y, dtype=np.int64).copy()).to(device)

        n = xt.size(0)
        seed = int(self.config.get("seed", 42))
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        val_frac = float(self.config.get("val_fraction", 0.1))
        n_val = max(1, int(round(n * val_frac)))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        train_ds = TensorDataset(xt[train_idx], yt[train_idx])
        val_ds = TensorDataset(xt[val_idx], yt[val_idx])

        batch_size = int(self.config.get("batch_size", 128))
        pin = device.type == "cuda"
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin,
        )

        epochs = int(self.config.get("epochs", 15))
        lr = float(self.config.get("learning_rate", 1.2e-3))
        weight_decay = float(self.config.get("weight_decay", 1e-4))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self._net.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)

        best_val = -1.0
        best_state: dict[str, torch.Tensor] | None = None

        out_dir = Path(self.config.get("out_dir", ""))
        if out_dir != Path(""):
            out_dir.mkdir(parents=True, exist_ok=True)

        # Optionally display a Rich progress UI for epochs and batches
        if show_progress and Progress is not None:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
            )
            progress.start()
            epoch_task = progress.add_task("Epochs", total=epochs)
            batch_task = progress.add_task("Batches", total=0)
        else:
            progress = None
            epoch_task = None
            batch_task = None

        for epoch in range(epochs):
            if epoch == 0 and not self._is_docker():
                # Only profile on first epoch, and only outside Docker
                # update batch task total before running epoch
                if progress is not None and batch_task is not None:
                    progress.update(batch_task, total=len(train_loader), completed=0)

                try:
                    with profile(
                        activities=[ProfilerActivity.CPU],
                        record_shapes=False,
                        profile_memory=False,
                        with_stack=False,
                    ) as prof:
                        with record_function("train_epoch"):
                            self._train_epoch(
                                train_loader,
                                optimizer,
                                criterion,
                                device,
                                progress=progress,
                                batch_task_id=batch_task,
                            )

                    if out_dir != Path(""):
                        try:
                            with open(out_dir / "profile.txt", "w") as f:
                                f.write(
                                    prof.key_averages().table(
                                        sort_by="self_cpu_time_total", row_limit=30
                                    )
                                )
                        except Exception:
                            pass  # Silently skip profiling output if write fails
                except Exception:
                    # Profiler may fail; train without profiling
                    self._train_epoch(
                        train_loader,
                        optimizer,
                        criterion,
                        device,
                        progress=progress,
                        batch_task_id=batch_task,
                    )

            else:
                # update batch task total before running epoch
                if progress is not None and batch_task is not None and epoch == 0:
                    progress.update(batch_task, total=len(train_loader), completed=0)

                self._train_epoch(
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                    progress=progress,
                    batch_task_id=batch_task,
                )

            scheduler.step()

            self._net.eval()
            non_blocking = device.type == "cuda"
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=non_blocking)
                    yb = yb.to(device, non_blocking=non_blocking)
                    pred = self._net(xb).argmax(dim=1)
                    correct += int((pred == yb).sum().item())
                    total += yb.size(0)
            val_acc = correct / max(total, 1)
            if val_acc > best_val:
                best_val = val_acc
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self._net.state_dict().items()
                }
            # advance epoch progress
            if progress is not None and epoch_task is not None:
                try:
                    progress.advance(epoch_task)
                except Exception:
                    pass

        if progress is not None:
            try:
                progress.stop()
            except Exception:
                pass

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._net.to(device)
        self._fitted = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit(...) or load(...) before predict.")
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy.ndarray of pixels.")
        device = self._device()
        self._net.to(device)
        self._net.eval()
        xt = self._prepare_x(X)
        preds: list[int] = []
        batch_size = int(self.config.get("batch_size", 512))
        non_blocking = device.type == "cuda"
        with torch.no_grad():
            for start in range(0, xt.size(0), batch_size):
                chunk = xt[start : start + batch_size].to(
                    device, non_blocking=non_blocking
                )
                logits = self._net(chunk)
                preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        return np.asarray(preds, dtype=np.int64)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cpu_state = {k: v.detach().cpu() for k, v in self._net.state_dict().items()}
        bundle: dict[str, Any] = {
            "kind": _BUNDLE_KIND,
            "state_dict": cpu_state,
            "config": dict(self.config),
        }
        joblib.dump(bundle, path)

    @classmethod
    def load(cls, path: Path) -> Model:
        blob: Any = joblib.load(path)
        if not isinstance(blob, dict) or blob.get("kind") != _BUNDLE_KIND:
            raise TypeError("File is not a saved s4p_mnist CNN model bundle.")
        cfg = blob.get("config") or {}
        model = cls(cfg if isinstance(cfg, dict) else {})
        state = blob.get("state_dict")
        if not isinstance(state, dict):
            raise TypeError("Invalid state_dict in model file.")
        model._net.load_state_dict(state)
        model._net.to(model._device())
        model._fitted = True
        return model
