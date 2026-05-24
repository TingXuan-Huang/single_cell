"""Generic training loop.

Encoder-agnostic: a Trainer takes any nn.Module whose forward(batch) returns a
dict with a 'loss' key. All four model classes in cellfm.models satisfy this.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from cellfm.training.schedules import cosine_schedule_with_warmup

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    out_dir: Path
    encoder: str
    size: str
    n_steps: int = 5000
    eval_every: int = 500
    val_max_batches: int | None = 20
    checkpoint_every: int | None = 0
    early_stopping_patience: int | None = 0
    early_stopping_min_delta: float = 0.0
    warmup_steps: int = 200
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    amp: bool = True
    log_every: int = 50
    seed: int = 0
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def _move_to(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loaders: dict[str, DataLoader],
        cfg: TrainConfig,
        device: torch.device | None = None,
    ):
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loaders = loaders
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.scheduler = cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=cfg.n_steps,
        )
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if cfg.amp and self.device.type == "cuda"
            else None
        )
        self.history: list[dict] = []
        self._best_val = float("inf")
        self._best_step = 0
        self._early_stop_best_val = float("inf")
        self._bad_eval_count = 0
        self._resume_step = 0
        self.cfg.out_dir = Path(cfg.out_dir)
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)

        # wandb (lazy)
        self.wandb = None
        if cfg.wandb_project:
            try:
                import wandb
                wandb.init(
                    project=cfg.wandb_project,
                    name=cfg.wandb_run_name or f"{cfg.encoder}_{cfg.size}",
                    config=asdict(cfg),
                )
                self.wandb = wandb
            except Exception as e:
                logger.warning("wandb init failed (%s). Continuing without wandb.", e)

    def load_checkpoint(self, path: Path | str, *, strict: bool = True) -> int:
        ckpt_path = Path(path)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=strict)
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if self.scaler is not None and ckpt.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        self._resume_step = int(ckpt.get("step", 0))
        val = ckpt.get("val_metrics")
        if isinstance(val, dict):
            v = float(val.get("loss", float("inf")))
            if math.isfinite(v):
                self._best_val = v
                self._best_step = self._resume_step
                self._early_stop_best_val = v

        history_path = self.cfg.out_dir / "train_history.json"
        if history_path.exists():
            try:
                self.history = json.loads(history_path.read_text())
            except json.JSONDecodeError:
                logger.warning("Could not parse existing train history: %s", history_path)
        self.history.append(
            {"step": self._resume_step, "event": "resume", "checkpoint": str(ckpt_path)}
        )
        logger.info("Resumed checkpoint %s at step=%d", ckpt_path, self._resume_step)
        return self._resume_step

    def _step(self, batch: dict) -> dict[str, float]:
        self.model.train()
        batch = _move_to(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)

        optimizer_stepped = True
        if self.scaler is not None:
            scale_before = self.scaler.get_scale()
            with torch.amp.autocast("cuda"):
                out = self.model(batch)
                loss = out["loss"]
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            optimizer_stepped = self.scaler.get_scale() >= scale_before
        else:
            out = self.model(batch)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
        if optimizer_stepped:
            self.scheduler.step()

        metrics = {k: float(v.detach()) if isinstance(v, torch.Tensor) else float(v)
                   for k, v in out.items() if k != "embedding" and k != "logits"}
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return metrics

    @torch.no_grad()
    def evaluate(self, split: str = "val", max_batches: int | None = None) -> dict[str, float]:
        self.model.eval()
        sums: dict[str, float] = {}
        n = 0
        for bi, batch in enumerate(self.loaders[split]):
            if max_batches is not None and bi >= max_batches:
                break
            batch = _move_to(batch, self.device)
            out = self.model(batch)
            for k, v in out.items():
                if k in ("embedding", "logits"):
                    continue
                if isinstance(v, torch.Tensor):
                    sums[k] = sums.get(k, 0.0) + float(v.detach())
            n += 1
        return {k: v / max(1, n) for k, v in sums.items()}

    def fit(self) -> None:
        torch.manual_seed(self.cfg.seed)
        if self._resume_step >= self.cfg.n_steps:
            raise ValueError(
                f"Checkpoint step {self._resume_step} is already >= target "
                f"n_steps {self.cfg.n_steps}. Increase --n-steps to continue."
            )
        train_loader = self.loaders["train"]
        train_iter = iter(train_loader)

        t0 = time.time()
        last_step = self._resume_step
        for step in range(self._resume_step + 1, self.cfg.n_steps + 1):
            last_step = step
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            metrics = self._step(batch)

            if step % self.cfg.log_every == 0:
                rec = {"step": step, **metrics, "elapsed": time.time() - t0}
                self.history.append(rec)
                logger.info(
                    "step=%d loss=%.4f lr=%.2e",
                    step, metrics.get("loss", float("nan")), metrics["lr"]
                )
                if self.wandb:
                    self.wandb.log({"train/" + k: v for k, v in metrics.items()}, step=step)

            val: dict[str, float] | None = None
            if step % self.cfg.eval_every == 0 or step == self.cfg.n_steps:
                val = self.evaluate("val", max_batches=self.cfg.val_max_batches)
                logger.info("step=%d val=%s", step, val)
                rec = {"step": step, **{"val_" + k: v for k, v in val.items()}}
                self.history.append(rec)
                if self.wandb:
                    self.wandb.log({"val/" + k: v for k, v in val.items()}, step=step)
                improved = self._maybe_save_best(val, step)
                if self._should_save_periodic(step):
                    self._save_checkpoint(f"step_{step:06d}.pt", step=step, val=val)
                if self._should_stop_early(improved, step, val):
                    break

            elif self._should_save_periodic(step):
                self._save_checkpoint(f"step_{step:06d}.pt", step=step, val=val)

        self._save_final(step=last_step)
        if self.wandb:
            self.wandb.finish()

    def _maybe_save_best(self, val_metrics: dict[str, float], step: int) -> bool:
        v = val_metrics.get("loss", float("inf"))
        improved_for_patience = self._is_meaningful_val_improvement(v)
        if improved_for_patience:
            self._early_stop_best_val = v
        if math.isfinite(v) and v < self._best_val:
            self._best_val = v
            self._best_step = step
            self._save_checkpoint("best.pt", step=step, val=val_metrics)
        return improved_for_patience

    def _is_meaningful_val_improvement(self, value: float) -> bool:
        return math.isfinite(value) and value < self._early_stop_best_val - max(
            0.0, self.cfg.early_stopping_min_delta
        )

    def _should_stop_early(self, improved: bool, step: int, val_metrics: dict[str, float]) -> bool:
        patience = self.cfg.early_stopping_patience
        if patience is None or patience <= 0:
            return False
        if improved:
            self._bad_eval_count = 0
            return False

        self._bad_eval_count += 1
        if self._bad_eval_count < patience:
            return False

        logger.info(
            "Early stopping at step=%d: no val loss improvement >= %.4g for %d evals "
            "(best %.6f at step %d)",
            step,
            self.cfg.early_stopping_min_delta,
            patience,
            self._best_val,
            self._best_step,
        )
        self.history.append(
            {
                "step": step,
                "event": "early_stop",
                "best_step": self._best_step,
                "best_val_loss": self._best_val,
                "bad_eval_count": self._bad_eval_count,
                **{"val_" + k: v for k, v in val_metrics.items()},
            }
        )
        return True

    def _should_save_periodic(self, step: int) -> bool:
        interval = self.cfg.checkpoint_every
        return interval is not None and interval > 0 and step % interval == 0

    def _save_final(self, *, step: int) -> None:
        self._save_checkpoint("final.pt", step=step, val=None)
        with (self.cfg.out_dir / "train_history.json").open("w") as fh:
            json.dump(self.history, fh, indent=2)
        with (self.cfg.out_dir / "train_config.json").open("w") as fh:
            cfg_dict = asdict(self.cfg)
            cfg_dict["out_dir"] = str(cfg_dict["out_dir"])
            json.dump(cfg_dict, fh, indent=2)

    def _save_checkpoint(self, name: str, *, step: int, val: dict | None) -> None:
        path = self.cfg.out_dir / name
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
                "step": step,
                "val_metrics": val,
            },
            path,
        )
        logger.info("Saved checkpoint -> %s", path)
