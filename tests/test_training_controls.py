from __future__ import annotations

import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from cellfm.training.loop import TrainConfig, Trainer


class ScriptedValidationModel(nn.Module):
    def __init__(self, val_losses: list[float]):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.val_losses = val_losses
        self.val_calls = 0

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        if self.training:
            return {"loss": self.weight.square()}
        idx = min(self.val_calls, len(self.val_losses) - 1)
        self.val_calls += 1
        return {"loss": self.weight.new_tensor(self.val_losses[idx])}


def _loader(n_batches: int = 1) -> DataLoader:
    return DataLoader([{"x": torch.tensor([i])} for i in range(n_batches)], batch_size=None)


def test_periodic_checkpoints_and_early_stopping(tmp_path):
    model = ScriptedValidationModel([1.0, 0.9, 0.91, 0.92, 0.93])
    cfg = TrainConfig(
        out_dir=tmp_path / "run",
        encoder="dummy",
        size="tiny",
        n_steps=20,
        eval_every=2,
        val_max_batches=1,
        checkpoint_every=4,
        early_stopping_patience=2,
        early_stopping_min_delta=0.0,
        warmup_steps=0,
        lr=0.0,
        amp=False,
        log_every=1,
    )

    trainer = Trainer(
        model=model,
        loaders={"train": _loader(), "val": _loader()},
        cfg=cfg,
        device=torch.device("cpu"),
    )
    trainer.fit()

    assert (cfg.out_dir / "best.pt").exists()
    assert (cfg.out_dir / "final.pt").exists()
    assert (cfg.out_dir / "step_000004.pt").exists()
    assert (cfg.out_dir / "step_000008.pt").exists()

    final = torch.load(cfg.out_dir / "final.pt", map_location="cpu")
    assert final["step"] == 8

    history = json.loads((cfg.out_dir / "train_history.json").read_text())
    early_stop = [rec for rec in history if rec.get("event") == "early_stop"]
    assert len(early_stop) == 1
    assert early_stop[0]["step"] == 8
    assert early_stop[0]["best_step"] == 4
    assert early_stop[0]["bad_eval_count"] == 2
    assert early_stop[0]["best_val_loss"] == pytest.approx(0.9)
    assert early_stop[0]["val_loss"] == pytest.approx(0.92)


def test_val_max_batches_limits_validation_work(tmp_path):
    model = ScriptedValidationModel([1.0, 2.0, 3.0])
    cfg = TrainConfig(
        out_dir=tmp_path / "run",
        encoder="dummy",
        size="tiny",
        n_steps=2,
        eval_every=2,
        val_max_batches=2,
        warmup_steps=0,
        lr=0.0,
        amp=False,
        log_every=1,
    )

    trainer = Trainer(
        model=model,
        loaders={"train": _loader(), "val": _loader(n_batches=3)},
        cfg=cfg,
        device=torch.device("cpu"),
    )
    trainer.fit()

    assert model.val_calls == 2
    best = torch.load(cfg.out_dir / "best.pt", map_location="cpu")
    assert best["val_metrics"] == {"loss": 1.5}
