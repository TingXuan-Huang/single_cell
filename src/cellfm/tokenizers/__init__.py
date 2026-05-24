"""Input encoders / tokenizers.

Each encoder produces a batch dict consumable by its matching model head.
All encoders consume the same per-cell records from CellShardDataset:

    {"gene_idx": np.int64[K], "values": np.float32[K], "label": int}

This is the only place tokenization logic lives. Ablations across encoder
choice are clean.
"""

from cellfm.tokenizers.base import Tokenizer, TokenizerConfig
from cellfm.tokenizers.embedding_bag import EmbeddingBagTokenizer
from cellfm.tokenizers.hvg_dense import HVGDenseTokenizer
from cellfm.tokenizers.rank import RankTokenizer
from cellfm.tokenizers.value_bin import ValueBinTokenizer

ENCODERS = {
    "hvg_dense": HVGDenseTokenizer,
    "embedding_bag": EmbeddingBagTokenizer,
    "rank": RankTokenizer,
    "value_bin": ValueBinTokenizer,
}

__all__ = [
    "ENCODERS",
    "Tokenizer",
    "TokenizerConfig",
    "HVGDenseTokenizer",
    "EmbeddingBagTokenizer",
    "RankTokenizer",
    "ValueBinTokenizer",
]
