"""Allen ABC Atlas download wrapper.

Uses ``abc_atlas_access`` to fetch the WMB-10X isocortex slice plus the
matching cell metadata from the AWS S3 Public Dataset bucket. The full
WMB-10X expression release is ~100+ GB; just the four isocortex shards of
WMB-10Xv2 plus the WMB-10Xv3 isocortex shards is the relevant subset for
this project (~30-50 GB depending on release).

Intended to run on a Hyak CPU node (see ``slurm/download.sbatch``). Do NOT
invoke this on a laptop unless you really want that much disk traffic.

Allen API reference
-------------------
- Listing top-level directories: ``abc_cache.list_directories`` (property)
- Listing expression-matrix file names in a directory:
  ``abc_cache.list_data_files(directory)``  (method) -- name in older releases
  ``abc_cache.list_expression_matrix_files(directory)``  -- newer alias
- Downloading one expression matrix:
  ``abc_cache.get_data_path(directory=DIR, file_name=DIR-Region-N/raw)``
- Listing metadata files: ``abc_cache.list_metadata_files(directory)``
- Downloading one metadata table:
  ``abc_cache.get_metadata_path(directory='WMB-10X', file_name='cell_metadata')``

Concretely, the isocortex slice of WMB-10Xv2 enumerates as
``WMB-10Xv2-Isocortex-1/raw`` .. ``WMB-10Xv2-Isocortex-4/raw``; WMB-10Xv3
enumerates analogously. We always pull raw counts (not the ``/log2`` copies)
because the cellfm pipeline does its own normalization.

References
----------
- https://github.com/AllenInstitute/abc_atlas_access
- https://alleninstitute.github.io/abc_atlas_access/descriptions/WMB-10X.html
- Yao et al., Nature 2023 (memory/LEARNINGS.md#wmb-10x-scrnaseq)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# Defaults -- the user can override on the CLI.
DEFAULT_EXPR_DIRS: tuple[str, ...] = ("WMB-10Xv2", "WMB-10Xv3")
DEFAULT_METADATA_DIRECTORY: str = "WMB-10X"
DEFAULT_METADATA_FILES: tuple[str, ...] = (
    # Cluster annotations and donor info -- needed for labels + donor split.
    "cell_metadata_with_cluster_annotation",
    # Gene metadata: maps ensembl IDs -> symbols.
    "gene",
)
DEFAULT_REGION_KEYWORD: str = "Isocortex"
DEFAULT_FILE_SUFFIX: str = "/raw"


def get_cache(download_dir: Path):
    """Lazy import abc_atlas_access and return its S3-backed cache."""
    try:
        from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
    except ImportError as e:  # pragma: no cover - explicit user-facing message
        raise ImportError(
            "abc_atlas_access is not installed. Install with `pip install -e .[allen]`."
        ) from e

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    return AbcProjectCache.from_s3_cache(download_dir)


def _list_data_files(abc_cache, directory: str) -> list[str]:
    """Compat wrapper: older releases expose ``list_data_files``, newer ones
    rename to ``list_expression_matrix_files``. Accept either."""
    for fn_name in ("list_data_files", "list_expression_matrix_files"):
        fn = getattr(abc_cache, fn_name, None)
        if callable(fn):
            return list(fn(directory))
    raise AttributeError(
        "Neither `list_data_files` nor `list_expression_matrix_files` is "
        "available on AbcProjectCache; check your abc_atlas_access version."
    )


def list_isocortex_packages(
    download_dir: Path,
    expr_dirs: tuple[str, ...] = DEFAULT_EXPR_DIRS,
    region_keyword: str = DEFAULT_REGION_KEYWORD,
    file_suffix: str = DEFAULT_FILE_SUFFIX,
) -> list[tuple[str, str]]:
    """Enumerate isocortex expression-matrix entries across one or more
    top-level directories.

    Returns a list of ``(directory, file_name)`` tuples suitable for
    ``abc_cache.get_data_path(directory=..., file_name=...)``.
    """
    abc_cache = get_cache(download_dir)
    out: list[tuple[str, str]] = []
    for directory in expr_dirs:
        try:
            files = _list_data_files(abc_cache, directory)
        except Exception as e:  # pragma: no cover - manifest miss is informational
            logger.warning("Failed to list data files for %s: %s", directory, e)
            continue
        for fn in files:
            if region_keyword in fn and fn.endswith(file_suffix):
                out.append((directory, fn))
    return sorted(out)


def download_expression(
    download_dir: Path,
    packages: list[tuple[str, str]],
) -> dict[str, str]:
    """Download a list of (directory, file_name) expression matrices.

    Returns ``{file_name: local_h5ad_path}``.
    """
    abc_cache = get_cache(download_dir)
    results: dict[str, str] = {}
    for directory, file_name in packages:
        logger.info("Downloading expression matrix %s :: %s", directory, file_name)
        path = abc_cache.get_data_path(directory=directory, file_name=file_name)
        results[file_name] = str(path)
        logger.info("  -> %s", path)
    return results


def download_metadata(
    download_dir: Path,
    directory: str = DEFAULT_METADATA_DIRECTORY,
    files: tuple[str, ...] = DEFAULT_METADATA_FILES,
) -> dict[str, str]:
    """Download metadata CSVs (cell metadata + gene table)."""
    abc_cache = get_cache(download_dir)
    results: dict[str, str] = {}
    for fn in files:
        logger.info("Downloading metadata %s :: %s", directory, fn)
        path = abc_cache.get_metadata_path(directory=directory, file_name=fn)
        results[fn] = str(path)
        logger.info("  -> %s", path)
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Download Allen WMB-10X isocortex expression matrices "
        "and matching cell metadata from the public S3 bucket."
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/abc"),
        help="Local cache directory (will be created if missing).",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Only enumerate matching isocortex packages, do not download.",
    )
    p.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip cell-metadata / gene-table downloads.",
    )
    p.add_argument(
        "--no-expression",
        action="store_true",
        help="Skip expression-matrix downloads (e.g. to only refresh metadata).",
    )
    p.add_argument(
        "--expr-dirs",
        nargs="+",
        default=list(DEFAULT_EXPR_DIRS),
        help="Top-level ABC directories to enumerate (default: WMB-10Xv2 WMB-10Xv3).",
    )
    p.add_argument(
        "--region-keyword",
        default=DEFAULT_REGION_KEYWORD,
        help="Substring used to filter file names (default: 'Isocortex').",
    )
    p.add_argument(
        "--file-suffix",
        default=DEFAULT_FILE_SUFFIX,
        help="File-name suffix to keep (default: '/raw'). Use '/log2' for the "
        "pre-log-normalized variants.",
    )
    p.add_argument(
        "--metadata-directory",
        default=DEFAULT_METADATA_DIRECTORY,
        help="ABC metadata directory key (default: WMB-10X).",
    )
    p.add_argument(
        "--metadata-files",
        nargs="+",
        default=list(DEFAULT_METADATA_FILES),
        help="ABC metadata file names to fetch.",
    )
    p.add_argument(
        "--packages",
        nargs="+",
        default=None,
        help="Explicit packages to download as DIR:FILENAME pairs, e.g. "
        "'WMB-10Xv2:WMB-10Xv2-Isocortex-1/raw'. If absent, downloads all "
        "matched isocortex packages.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without doing it.",
    )
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    available = list_isocortex_packages(
        args.out,
        expr_dirs=tuple(args.expr_dirs),
        region_keyword=args.region_keyword,
        file_suffix=args.file_suffix,
    )
    logger.info(
        "Found %d candidate %s packages with suffix %r across %s:",
        len(available), args.region_keyword, args.file_suffix, args.expr_dirs,
    )
    for d, f in available:
        logger.info("  - %s :: %s", d, f)

    if args.list:
        return 0

    # Resolve which expression packages to download.
    if args.packages:
        to_download: list[tuple[str, str]] = []
        for spec in args.packages:
            if ":" not in spec:
                logger.error("Bad --packages entry %r (expected DIR:FILE_NAME)", spec)
                return 1
            d, f = spec.split(":", 1)
            to_download.append((d, f))
    else:
        to_download = available

    results: dict[str, Any] = {
        "expression": {},
        "metadata": {},
        "expr_dirs": list(args.expr_dirs),
        "region_keyword": args.region_keyword,
        "file_suffix": args.file_suffix,
        "out": str(args.out),
    }

    if args.no_expression:
        logger.info("--no-expression set; skipping expression downloads.")
    else:
        if not to_download:
            logger.error(
                "No expression packages selected. Use --list to inspect or "
                "pass explicit --packages."
            )
            return 1
        logger.info("Will download %d expression matrices to %s", len(to_download), args.out)
        if not args.dry_run:
            results["expression"] = download_expression(args.out, to_download)

    if args.no_metadata:
        logger.info("--no-metadata set; skipping metadata downloads.")
    else:
        logger.info("Will download %d metadata files to %s", len(args.metadata_files), args.out)
        if not args.dry_run:
            results["metadata"] = download_metadata(
                args.out,
                directory=args.metadata_directory,
                files=tuple(args.metadata_files),
            )

    if args.dry_run:
        logger.info("Dry run complete; no files written.")
        return 0

    manifest_path = args.out / "download_manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Wrote %s", manifest_path)
    logger.info(
        "SUCCESS: downloaded %d expression + %d metadata files.",
        len(results["expression"]), len(results["metadata"]),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
