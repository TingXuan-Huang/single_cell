"""1-step VRAM probe.

Runs one forward + backward on a synthetic batch, measures peak GPU memory,
and aborts with a clear message (not a deep CUDA OOM stack) if it exceeds
``max_frac`` of available VRAM.

Used by `scripts.train_one` for the larger size presets (tiny_5m, tiny_10m)
where head-room on a 48 GB A40 is tight. Cheap to run (~1 s) and prevents
silent SLURM-array OOMs.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _sample_batch(loader) -> dict:
    """Pull one batch from a DataLoader without consuming the training iterator."""
    it = iter(loader)
    return next(it)


def assert_flash_sdpa_available() -> None:
    """Best-effort check that PyTorch 2 SDPA + flash attention is wired up.

    We don't crash if the flash backend isn't available (e.g. on CPU), since
    SDPA falls back to the math kernel and the model still runs. We *do* log
    so reviewers can see whether the run used flash attention or not.
    """
    if not torch.cuda.is_available():
        logger.info("[vram-probe] CUDA unavailable -> SDPA running on CPU math backend.")
        return
    try:
        # PyTorch 2.x: query whether the flash kernel is selectable.
        from torch.backends.cuda import sdp_kernel  # noqa: F401
    except Exception:
        logger.warning("[vram-probe] torch.backends.cuda.sdp_kernel not found; "
                       "cannot confirm flash attention. Continuing.")
        return
    # The kernel is chosen at runtime by F.scaled_dot_product_attention.
    # Just print the available backends for the audit log.
    try:
        is_flash = torch.backends.cuda.flash_sdp_enabled()
        is_mem = torch.backends.cuda.mem_efficient_sdp_enabled()
        is_math = torch.backends.cuda.math_sdp_enabled()
        logger.info("[vram-probe] SDPA backends enabled: flash=%s mem_eff=%s math=%s",
                    is_flash, is_mem, is_math)
        if not (is_flash or is_mem):
            logger.warning("[vram-probe] Neither flash nor mem-efficient SDPA is enabled; "
                           "expect higher VRAM use. tiny_10m may OOM at L=2048.")
    except AttributeError:
        # Older PyTorch versions; non-fatal.
        logger.info("[vram-probe] PyTorch < 2.2; flash_sdp_enabled() not exposed.")


def probe_vram(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    max_frac: float = 0.90,
    amp: bool = True,
) -> dict[str, float]:
    """Forward + backward on one batch, return peak memory stats.

    Aborts via RuntimeError (not OOM) if peak memory > ``max_frac`` of the
    visible GPU total. On CPU this is a no-op that returns zeros.

    Returns
    -------
    dict with keys ``peak_mb``, ``total_mb``, ``frac`` (float).
    """
    if device.type != "cuda":
        return {"peak_mb": 0.0, "total_mb": 0.0, "frac": 0.0}

    assert_flash_sdpa_available()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    batch = _sample_batch(loader)
    batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
    model.train()
    try:
        if amp:
            with torch.amp.autocast("cuda"):
                out = model(batch)
                loss = out["loss"]
            # Use a real GradScaler-less scaled backward to keep memory profile honest.
            loss.backward()
        else:
            out = model(batch)
            loss = out["loss"]
            loss.backward()
    except torch.cuda.OutOfMemoryError as e:
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        total = torch.cuda.get_device_properties(device).total_memory / 1024**2
        raise RuntimeError(
            f"VRAM probe OOM at peak {peak:.0f} MB / {total:.0f} MB. "
            f"Reduce batch_size or n_steps; tiny_10m needs batch_size<=64 on a 48 GB A40. "
            f"Original error: {e}"
        ) from e

    # Reset grads so we don't poison the real first training step.
    model.zero_grad(set_to_none=True)

    peak = torch.cuda.max_memory_allocated(device)
    total = torch.cuda.get_device_properties(device).total_memory
    frac = peak / total
    stats = {
        "peak_mb": peak / 1024**2,
        "total_mb": total / 1024**2,
        "frac": frac,
    }
    logger.info(
        "[vram-probe] peak=%.0f MB / %.0f MB (%.1f%%)",
        stats["peak_mb"], stats["total_mb"], 100 * stats["frac"],
    )
    if frac > max_frac:
        raise RuntimeError(
            f"VRAM probe exceeded {max_frac:.0%} of GPU memory "
            f"({stats['peak_mb']:.0f} / {stats['total_mb']:.0f} MB = {100*frac:.1f}%). "
            f"Reduce batch_size before launching n_steps={getattr(loader.batch_sampler, 'batch_size', '?')} steps."
        )
    return stats
