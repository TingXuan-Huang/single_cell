"""Deterministic hash helper tests."""

from __future__ import annotations

import pytest
import torch

from cellfm.tokenizers.hash_utils import stable_multi_hash


def test_stable_multi_hash_expected_values():
    gene_ids = torch.tensor([0, 1, 5])
    buckets = stable_multi_hash(gene_ids, n_buckets=13, n_hashes=3)
    expected = torch.tensor(
        [
            [3, 6, 9],
            [12, 10, 8],
            [9, 0, 4],
        ]
    )
    assert torch.equal(buckets, expected)


def test_stable_multi_hash_is_reproducible_and_range_bounded():
    gene_ids = torch.tensor([[0, 1], [2, 299]])
    a = stable_multi_hash(gene_ids, n_buckets=17, n_hashes=4, salt=11)
    b = stable_multi_hash(gene_ids.clone(), n_buckets=17, n_hashes=4, salt=11)
    assert torch.equal(a, b)
    assert a.shape == (2, 2, 4)
    assert int(a.min()) >= 0
    assert int(a.max()) < 17


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_buckets": 0, "n_hashes": 4},
        {"n_buckets": 8, "n_hashes": 0},
    ],
)
def test_stable_multi_hash_rejects_invalid_shape_params(kwargs):
    with pytest.raises(ValueError):
        stable_multi_hash(torch.tensor([1, 2, 3]), **kwargs)
