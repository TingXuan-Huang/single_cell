"""Train one (encoder, size) configuration on a built cache.

Usage (Hyak):
    python -m scripts.train_one \\
        --cache-dir /gscratch/.../cache/wmb_isocortex_v1 \\
        --encoder-config configs/encoder/value_bin.yaml \\
        --model-config   configs/model/tiny_1m.yaml \\
        --train-config   configs/experiment/main_grid.yaml \\
        --run-id value_bin_tiny_1m \\
        --out-dir /gscratch/.../runs/cellfm/v1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from scripts._config import load_yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _optional_int(value, default: int | None = None) -> int | None:
    if value is None:
        return None
    if value == "":
        return default
    return int(value)


def _build_tokenizer(encoder_cfg: dict, manifest, hvg_indices=None):
    from cellfm.tokenizers import ENCODERS
    from cellfm.tokenizers.base import TokenizerConfig

    name = encoder_cfg["name"]
    tcfg = TokenizerConfig(
        n_genes=int(manifest.n_genes),
        L=int(manifest.L),
        hvg_indices=hvg_indices if hvg_indices is not None else manifest.hvg_indices,
        n_bins=int(encoder_cfg.get("n_bins", 51)),
        log1p_normalize=bool(encoder_cfg.get("log1p_normalize", True)),
        target_sum=float(encoder_cfg.get("target_sum", 1e4)),
        mask_ratio=float(encoder_cfg.get("mask_ratio", 0.15)),
        add_cls=bool(encoder_cfg.get("add_cls", True)),
    )
    return ENCODERS[name](tcfg)


def _build_model(encoder_cfg, model_cfg, manifest, tokenizer):
    from cellfm.models import build_model, count_params

    name = encoder_cfg["name"]
    # Compose a SIZE override on top of the registered preset by passing kwargs.
    # We simply pull dims from the model yaml — the factory uses them via SIZE_CONFIGS,
    # so we register the model_cfg into SIZE_CONFIGS dynamically.
    from cellfm.models.factory import SIZE_CONFIGS, SizeConfig

    sz_name = model_cfg["name"]
    SIZE_CONFIGS[sz_name] = SizeConfig(
        name=sz_name,
        d_model=int(model_cfg["d_model"]),
        n_layers=int(model_cfg["n_layers"]),
        n_heads=int(model_cfg["n_heads"]),
        ffn_mult=int(model_cfg["ffn_mult"]),
        mlp_hidden=tuple(model_cfg["mlp_hidden"]),
        mlp_d_embed=int(model_cfg["mlp_d_embed"]),
    )

    model = build_model(
        encoder=name,
        size=sz_name,
        n_genes=int(manifest.n_genes),
        n_classes=len(manifest.label_vocab),
        L=int(manifest.L),
        n_hvg=int(len(manifest.hvg_indices)),
        gene_vocab_size=tokenizer.gene_vocab_size,
        value_vocab_size=tokenizer.value_vocab_size,
        dropout=float(model_cfg.get("dropout", 0.1)),
        mlm_weight=float(encoder_cfg.get("mlm_weight", 1.0)),
        ce_weight=float(encoder_cfg.get("ce_weight", 0.5)),
    )
    pcounts = count_params(model)
    logger.info("Param counts: %s", pcounts)
    return model, pcounts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train one cellfm encoder.")
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--encoder-config", required=True, type=Path)
    p.add_argument("--model-config", required=True, type=Path)
    p.add_argument("--train-config", required=True, type=Path,
                   help="Either a per-run YAML or the main_grid YAML (then needs --run-id).")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--val-max-batches", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=None)
    p.add_argument("--early-stopping-patience", type=int, default=None)
    p.add_argument("--early-stopping-min-delta", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--wandb-project", type=str, default=None)
    args = p.parse_args(argv)

    encoder_cfg = load_yaml(args.encoder_config)
    model_cfg = load_yaml(args.model_config)
    raw_train = load_yaml(args.train_config)

    # If we got main_grid.yaml, pull its `base.train` block; else treat raw_train as the train block.
    if "base" in raw_train and "train" in raw_train["base"]:
        train_cfg = dict(raw_train["base"]["train"])
    else:
        train_cfg = dict(raw_train)

    # CLI overrides
    if args.batch_size is not None: train_cfg["batch_size"] = args.batch_size
    if args.n_steps is not None: train_cfg["n_steps"] = args.n_steps
    if args.val_max_batches is not None: train_cfg["val_max_batches"] = args.val_max_batches
    if args.checkpoint_every is not None: train_cfg["checkpoint_every"] = args.checkpoint_every
    if args.early_stopping_patience is not None:
        train_cfg["early_stopping_patience"] = args.early_stopping_patience
    if args.early_stopping_min_delta is not None:
        train_cfg["early_stopping_min_delta"] = args.early_stopping_min_delta
    if args.lr is not None: train_cfg["lr"] = args.lr
    if args.seed is not None: train_cfg["seed"] = args.seed
    if args.wandb_project is not None: train_cfg["wandb_project"] = args.wandb_project

    run_id = args.run_id or f"{encoder_cfg['name']}_{model_cfg['name']}"
    out_dir = args.out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    from cellfm.data.cache import CacheManifest
    manifest = CacheManifest.from_json(args.cache_dir / "manifest.json")

    tokenizer = _build_tokenizer(encoder_cfg, manifest)
    model, pcounts = _build_model(encoder_cfg, model_cfg, manifest, tokenizer)

    # Dataloaders
    from cellfm.training import build_dataloaders
    loaders = build_dataloaders(
        cache_dir=args.cache_dir,
        tokenizer=tokenizer,
        batch_size=int(train_cfg.get("batch_size", 128)),
        num_workers=args.num_workers,
        eval_batch_size=int(train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 128))),
        drop_last=True,
    )

    # Train
    from cellfm.training import Trainer
    from cellfm.training.loop import TrainConfig
    tcfg = TrainConfig(
        out_dir=out_dir,
        encoder=encoder_cfg["name"],
        size=model_cfg["name"],
        n_steps=int(train_cfg.get("n_steps", 5000)),
        eval_every=int(train_cfg.get("eval_every", 500)),
        val_max_batches=_optional_int(train_cfg.get("val_max_batches", 20)),
        checkpoint_every=_optional_int(train_cfg.get("checkpoint_every", 0)),
        early_stopping_patience=_optional_int(train_cfg.get("early_stopping_patience", 0)),
        early_stopping_min_delta=float(train_cfg.get("early_stopping_min_delta", 0.0)),
        warmup_steps=int(train_cfg.get("warmup_steps", 200)),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        grad_clip=float(train_cfg.get("grad_clip", 1.0)),
        amp=bool(train_cfg.get("amp", True)),
        seed=int(train_cfg.get("seed", 0)),
        wandb_project=train_cfg.get("wandb_project"),
        wandb_run_name=run_id,
        extras={"param_counts": pcounts, "run_id": run_id},
    )

    # Persist resolved configs for reproducibility BEFORE training
    (out_dir / "encoder_config.json").write_text(json.dumps(encoder_cfg, indent=2))
    (out_dir / "model_config.json").write_text(json.dumps(model_cfg, indent=2))
    (out_dir / "param_counts.json").write_text(json.dumps(pcounts, indent=2))

    # F3 mitigation: 1-step VRAM probe for the large transformer sizes.
    # Aborts with a clear error before submitting a 12h SLURM job that would OOM
    # mid-run. CPU/non-transformer encoders skip cheaply (probe returns zeros).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_sizes = {"tiny_5m", "tiny_10m"}
    if device.type == "cuda" and model_cfg["name"] in probe_sizes:
        from cellfm.training.vram_probe import probe_vram
        model.to(device)
        stats = probe_vram(
            model, loaders["train"], device,
            max_frac=0.90,
            amp=bool(train_cfg.get("amp", True)),
        )
        (out_dir / "vram_probe.json").write_text(json.dumps(stats, indent=2))

    trainer = Trainer(model=model, loaders=loaders, cfg=tcfg)
    trainer.fit()

    # Save HVG indices for downstream eval (PCA baseline, etc.)
    import numpy as np
    np.save(out_dir / "hvg_indices.npy", np.asarray(manifest.hvg_indices, dtype=np.int64))

    logger.info("Done: %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
