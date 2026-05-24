"""Models: shared transformer body + four encoder-specific heads.

Factory:
    from cellfm.models import build_model
    model = build_model(encoder='rank', size='tiny_1m', n_genes=20000, n_classes=300, L=2048)
"""

from cellfm.models.factory import (
    SIZE_CONFIGS,
    SizeConfig,
    build_model,
    count_params,
)

__all__ = ["build_model", "count_params", "SIZE_CONFIGS", "SizeConfig"]
