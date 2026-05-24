from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from cellfm.data.allen import attach_cell_metadata_csv


def test_attach_cell_metadata_csv_matches_concat_suffixed_index(tmp_path):
    adata = AnnData(
        X=np.ones((2, 3)),
        obs=pd.DataFrame(index=[
            "cell_a-/gscratch/stf/raw/WMB-10Xv2-Isocortex-1-raw.h5ad",
            "cell_b-/gscratch/stf/raw/WMB-10Xv2-Isocortex-1-raw.h5ad",
        ]),
    )
    csv_path = tmp_path / "cell_metadata.csv"
    pd.DataFrame(
        {
            "cell_label": ["cell_a", "cell_b"],
            "donor_label": ["d1", "d2"],
            "subclass": ["s1", "s2"],
            "class": ["c1", "c2"],
            "cluster": ["k1", "k2"],
            "region_of_interest_acronym": ["ISO", "ISO"],
            "library_method": ["10x", "10x"],
        }
    ).to_csv(csv_path, index=False)

    out = attach_cell_metadata_csv(adata, csv_path)

    assert out.n_obs == 2
    assert out.obs["donor_label"].tolist() == ["d1", "d2"]
    assert out.obs["subclass"].tolist() == ["s1", "s2"]


def test_attach_cell_metadata_csv_keeps_rows_with_missing_optional_metadata(tmp_path):
    adata = AnnData(
        X=np.ones((2, 3)),
        obs=pd.DataFrame(index=["cell_a", "cell_b"]),
    )
    csv_path = tmp_path / "cell_metadata.csv"
    pd.DataFrame(
        {
            "cell_label": ["cell_a", "cell_b"],
            "donor_label": ["d1", "d2"],
            "subclass": ["s1", "s2"],
            "class": [np.nan, np.nan],
            "cluster": [np.nan, np.nan],
            "region_of_interest_acronym": ["ISO", "ISO"],
            "library_method": ["10x", "10x"],
        }
    ).to_csv(csv_path, index=False)

    out = attach_cell_metadata_csv(adata, csv_path)

    assert out.n_obs == 2
    assert out.obs["donor_label"].tolist() == ["d1", "d2"]
