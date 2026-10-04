"""Experiment configuration for BiasPruner on HAM10000."""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple
import json

@dataclass
class Config:
    data_root: str = "/kaggle/input/ham10000"
    metadata_csv: Optional[str] = None
    image_dirs: Tuple[str, ...] = ()
    output_dir: str = "results"
    seed: int = 42
    task_order: Tuple[str, ...] = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
    train_fraction: float = 0.65
    val_fraction: float = 0.12
    image_size: int = 224
    batch_size: int = 32
    workers: int = 4
    pretrained: bool = False
    amp: bool = True
    gce_epochs: int = 200
    finetune_epochs: int = 20
    patience: int = 15
    learning_rate: float = 1e-4
    finetune_lr: float = 1e-4
    q: float = 0.7
    tau: float = 0.70
    gamma: float = 0.60
    alpha: float = 0.5
    checkpoint_priority: str = "balanced_accuracy_then_eod"
    num_task_orders: int = 3
    device: str = "cuda"
    def validate(self):
        expected={"akiec","bcc","bkl","df","mel","nv","vasc"}
        if len(self.task_order) != 7 or set(self.task_order) != expected:
            raise ValueError("task_order must contain each HAM diagnosis class exactly once")
        if not (0 < self.train_fraction < 1 and 0 < self.val_fraction < 1 and self.train_fraction + self.val_fraction < 1):
            raise ValueError("train_fraction and val_fraction must leave a nonempty test split")
        if not (0 < self.alpha < 1): raise ValueError("alpha must be in (0,1)")
        if not (0 < self.q <= 1): raise ValueError("q must be in (0,1]")
        if not (0 <= self.tau <= 1): raise ValueError("tau must be in [0,1]")
        if not (0 <= self.gamma < 1): raise ValueError("gamma must be in [0,1)")
        if self.checkpoint_priority not in {"balanced_accuracy_then_eod", "eod_then_balanced_accuracy"}:
            raise ValueError("unsupported checkpoint_priority")
        if self.num_task_orders < 1: raise ValueError("num_task_orders must be at least 1")
    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
