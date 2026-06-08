"""Optional adapters for external single-cell foundation model packages."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cellfm.eval.foundation_embedding_io import load_embedding_matrix, safe_unique_names


def extract_scgpt_embeddings(adata, args: argparse.Namespace) -> np.ndarray:
    if args.scgpt_model_dir is None:
        raise ValueError("--scgpt-model-dir is required for --models scgpt")
    try:
        from scgpt.tasks.cell_emb import embed_data
    except ImportError as exc:
        raise RuntimeError("Install scGPT before using --models scgpt.") from exc

    adata = adata.copy()
    adata.obs["cell_type"] = adata.obs["cellfm_label"].astype(str)
    adata.var["feature_name"] = adata.var["gene_symbol"].astype(str)
    embedded = embed_data(
        adata,
        model_dir=args.scgpt_model_dir,
        cell_type_key="cell_type",
        gene_col="feature_name",
        max_length=args.scgpt_max_length,
        batch_size=args.batch_size,
        obs_to_save=["cellfm_label", "cellfm_label_id"],
        device=args.device,
        return_new_adata=True,
    )
    if "X_scGPT" not in embedded.obsm:
        raise RuntimeError("scGPT finished but did not write .obsm['X_scGPT']")
    return np.asarray(embedded.obsm["X_scGPT"], dtype=np.float32)


def _load_matrix_from_recent_outputs(directory: Path) -> np.ndarray:
    candidates = sorted(
        [p for p in directory.rglob("*") if p.suffix.lower() in {".npz", ".npy", ".csv", ".pt"}],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    errors = []
    for path in candidates:
        try:
            return load_embedding_matrix(path, h5ad_obsm_key="X")
        except Exception as exc:  # pragma: no cover - exercised only with external outputs
            errors.append(f"{path}: {exc}")
    raise RuntimeError("Could not find/load a 2D embedding output. Tried:\n" + "\n".join(errors))


def extract_geneformer_embeddings(adata, work_dir: Path, args: argparse.Namespace) -> np.ndarray:
    if args.geneformer_model_dir is None:
        raise ValueError("--geneformer-model-dir is required for --models geneformer")
    if "ensembl_id" not in adata.var.columns:
        raise ValueError("Geneformer requires ensembl_id in cache manifest/adata.var")
    try:
        from geneformer import EmbExtractor, TranscriptomeTokenizer
    except ImportError as exc:
        raise RuntimeError("Install Geneformer before using --models geneformer.") from exc

    input_dir = work_dir / "geneformer_h5ad"
    tokenized_dir = work_dir / "geneformer_tokenized"
    emb_dir = work_dir / "geneformer_raw_output"
    input_dir.mkdir(parents=True, exist_ok=True)
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    h5ad_path = input_dir / "cellfm_geneformer_input.h5ad"
    adata.write_h5ad(h5ad_path)

    tokenizer = TranscriptomeTokenizer(
        custom_attr_name_dict={"cellfm_label": "cellfm_label"},
        nproc=args.nproc,
        model_version=args.geneformer_version,
    )
    tokenizer.tokenize_data(input_dir, tokenized_dir, "cellfm_geneformer", file_format="h5ad")
    dataset_path = tokenized_dir / "cellfm_geneformer.dataset"

    extractor = EmbExtractor(
        model_type="Pretrained",
        num_classes=0,
        emb_mode=args.geneformer_emb_mode,
        max_ncells=None,
        emb_layer=args.geneformer_emb_layer,
        emb_label=["cellfm_label"],
        forward_batch_size=args.batch_size,
        nproc=args.nproc,
        model_version=args.geneformer_version,
    )
    returned = extractor.extract_embs(
        args.geneformer_model_dir,
        dataset_path,
        emb_dir,
        "geneformer",
        output_torch_embs=True,
    )
    if isinstance(returned, pd.DataFrame):
        arr = returned.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] > 0:
            return arr
    return _load_matrix_from_recent_outputs(emb_dir)


def extract_uce_embeddings(adata, args: argparse.Namespace) -> np.ndarray:
    try:
        from helical.models.uce import UCE, UCEConfig
    except ImportError as exc:
        raise RuntimeError("Install helical before using --models uce.") from exc

    cfg_kwargs: dict[str, Any] = {"batch_size": args.batch_size}
    if args.device:
        cfg_kwargs["device"] = args.device
    if args.uce_species:
        cfg_kwargs["species"] = args.uce_species
    try:
        configurer = UCEConfig(**cfg_kwargs)
    except TypeError:
        cfg_kwargs.pop("species", None)
        configurer = UCEConfig(**cfg_kwargs)

    adata = adata.copy()
    adata.var_names = safe_unique_names(adata.var["gene_symbol"].astype(str).tolist())
    model = UCE(configurer=configurer)
    dataset = model.process_data(
        adata,
        gene_names="index",
        name="cellfm_foundation_check",
        use_raw_counts=True,
    )
    return np.asarray(model.get_embeddings(dataset), dtype=np.float32)


def extract_scfoundation_embeddings(adata, work_dir: Path, args: argparse.Namespace) -> np.ndarray:
    if args.scfoundation_repo is None:
        raise ValueError("--scfoundation-repo is required for --models scfoundation")
    repo = Path(args.scfoundation_repo).expanduser()
    script = repo / "get_embedding.py"
    if not script.exists():
        raise FileNotFoundError(f"Could not find scFoundation get_embedding.py at {script}")

    input_dir = work_dir / "scfoundation_input"
    output_dir = work_dir / "scfoundation_raw_output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5ad_path = input_dir / "cellfm_scfoundation_input.h5ad"
    adata = adata.copy()
    adata.var["gene_name"] = adata.var["gene_symbol"].astype(str)
    adata.write_h5ad(h5ad_path)

    cmd = [
        sys.executable,
        str(script),
        "--task_name",
        "cellfm",
        "--input_type",
        "singlecell",
        "--output_type",
        "cell",
        "--pool_type",
        args.scfoundation_pool_type,
        "--tgthighres",
        args.scfoundation_tgthighres,
        "--data_path",
        str(h5ad_path),
        "--save_path",
        str(output_dir),
        "--pre_normalized",
        args.scfoundation_pre_normalized,
        "--version",
        args.scfoundation_version,
        "--ckpt_name",
        args.scfoundation_ckpt_name,
    ]
    if args.scfoundation_model_path is not None:
        cmd.extend(["--model_path", str(args.scfoundation_model_path)])
    subprocess.run(cmd, cwd=repo, check=True)
    return _load_matrix_from_recent_outputs(output_dir)


def run_requested_model(name: str, adata, work_dir: Path, args: argparse.Namespace) -> np.ndarray:
    if name == "scgpt":
        return extract_scgpt_embeddings(adata, args)
    if name == "geneformer":
        return extract_geneformer_embeddings(adata, work_dir, args)
    if name == "uce":
        return extract_uce_embeddings(adata, args)
    if name == "scfoundation":
        return extract_scfoundation_embeddings(adata, work_dir, args)
    raise ValueError(f"Unknown model {name!r}")
