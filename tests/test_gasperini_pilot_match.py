from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad
import mudata as md

from perturbo.preprocessing.gasperini_pilot_match import build_at_scale_pilot_matched_mudata


def test_build_at_scale_pilot_matched_mudata_subsets_shared_guides_and_cells() -> None:
    at_scale = md.MuData(
        {
            "gene": ad.AnnData(
                X=np.arange(24, dtype=np.float32).reshape(6, 4),
                obs=pd.DataFrame(index=[f"cell_{i}" for i in range(6)]),
                var=pd.DataFrame(index=[f"gene_{i}" for i in range(4)]),
            ),
            "guide": ad.AnnData(
                X=np.array(
                    [
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ],
                    dtype=np.float32,
                ),
                obs=pd.DataFrame(index=[f"cell_{i}" for i in range(6)]),
                var=pd.DataFrame(index=["guide_b", "guide_d", "guide_x", "guide_y"]),
            ),
        }
    )
    pilot = md.MuData(
        {
            "gene": ad.AnnData(
                X=np.ones((4, 4), dtype=np.float32),
                obs=pd.DataFrame(index=[f"pilot_{i}" for i in range(4)]),
                var=pd.DataFrame(index=[f"gene_{i}" for i in range(4)]),
            ),
            "guide": ad.AnnData(
                X=np.array(
                    [
                        [1, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                    dtype=np.float32,
                ),
                obs=pd.DataFrame(index=[f"pilot_{i}" for i in range(4)]),
                var=pd.DataFrame(index=["guide_d", "guide_b", "guide_missing"]),
            ),
        }
    )

    matched, summary = build_at_scale_pilot_matched_mudata(at_scale, pilot, random_seed=0)

    assert matched["guide"].var_names.tolist() == ["guide_d", "guide_b"]
    assert matched.n_obs < at_scale.n_obs
    assert matched["gene"].n_obs == matched["guide"].n_obs
    assert matched["guide"].n_vars == 2
    assert summary.shared_guides == 2
    assert summary.at_scale_cells_with_shared_guides == 4
    assert summary.pilot_cells_with_shared_guides == 3
    assert summary.pilot_median_cells_per_shared_guide == 1.5
    assert abs(summary.at_scale_median_cells_per_shared_guide_after - summary.pilot_median_cells_per_shared_guide) <= 0.5
