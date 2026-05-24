"""Torch DataLoader on top of CellShardDataset + Tokenizer.

The collate_fn is the only place tokenization happens; everything else is just
random-access reads into the cached CSR.
"""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from cellfm.data.cache import CellShardDataset
from cellfm.tokenizers.base import Tokenizer


class TorchCellShard(Dataset):
    """Thin torch.utils.data.Dataset wrapper around CellShardDataset."""

    def __init__(self, inner: CellShardDataset):
        self.inner = inner

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> dict:
        return self.inner[idx]


class _CollateFn:
    """Picklable collate that calls a tokenizer's encode_batch."""

    def __init__(self, tokenizer: Tokenizer, train: bool):
        self.tokenizer = tokenizer
        self.train = train

    def __call__(self, items: list[dict]) -> dict:
        return self.tokenizer.encode_batch(items, train=self.train)


def build_dataloaders(
    cache_dir,
    tokenizer: Tokenizer,
    *,
    batch_size: int = 64,
    num_workers: int = 2,
    eval_batch_size: int | None = None,
    drop_last: bool = True,
    eval_mode: bool = False,
) -> dict[str, DataLoader]:
    """Build {'train','val','test'} DataLoaders sharing one tokenizer.

    Args:
        eval_mode: if True, the train split is built with shuffle=False and the
            tokenizer's `train=False` flag (no MLM masking), suitable for
            embedding extraction.
    """
    eval_batch_size = eval_batch_size or batch_size
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        inner = CellShardDataset(cache_dir=cache_dir, split=split)
        ds = TorchCellShard(inner)
        is_train_split = split == "train"
        tokenizer_train = is_train_split and not eval_mode
        shuffle = is_train_split and not eval_mode
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size if is_train_split else eval_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=_CollateFn(tokenizer, train=tokenizer_train),
            drop_last=drop_last if (is_train_split and not eval_mode) else False,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    return loaders
