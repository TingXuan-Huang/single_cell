from __future__ import annotations

from cellfm.data.splits import SplitConfig, donor_stratified_split


def test_donor_stratified_no_donor_overlap(synthetic_adata):
    cfg = SplitConfig(
        donor_col="donor_id", label_col="subclass",
        train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=0,
        min_donors_per_split=1,
    )
    idx = donor_stratified_split(synthetic_adata.copy(), cfg)
    train_donors = set(synthetic_adata.obs.iloc[idx["train"]]["donor_id"])
    val_donors = set(synthetic_adata.obs.iloc[idx["val"]]["donor_id"])
    test_donors = set(synthetic_adata.obs.iloc[idx["test"]]["donor_id"])
    assert train_donors.isdisjoint(val_donors)
    assert train_donors.isdisjoint(test_donors)
    assert val_donors.isdisjoint(test_donors)


def test_donor_stratified_covers_all_cells(synthetic_adata):
    cfg = SplitConfig(donor_col="donor_id", label_col="subclass",
                      train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=0,
                      min_donors_per_split=1)
    idx = donor_stratified_split(synthetic_adata.copy(), cfg)
    n = sum(len(v) for v in idx.values())
    assert n == synthetic_adata.n_obs
