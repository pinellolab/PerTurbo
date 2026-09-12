"""Reject invalid count matrices before the loaders can truncate them."""

from __future__ import annotations

import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from perturbo.core import _select_count_dtype, load_analysis_cells, load_controls


@pytest.mark.parametrize("bad", [0.5, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_interior_count_is_rejected(bad):
    counts = np.array([[0.0, bad], [1.0, 2.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="finite, non-negative integers"):
        _select_count_dtype(counts)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint32, np.int64])
@pytest.mark.parametrize("maximum, expected", [
    (65535, np.uint16), (65536, np.int32), (2**31, np.uint32), (2**32, np.int64),
])
def test_integral_counts_keep_existing_dtype_selection(dtype, maximum, expected):
    if np.issubdtype(dtype, np.integer) and maximum > np.iinfo(dtype).max:
        pytest.skip("value is outside the input dtype")
    counts = np.array([[0, maximum], [1, 2]], dtype=dtype)
    selected = _select_count_dtype(counts)
    assert selected == expected
    np.testing.assert_array_equal(counts.astype(selected), counts)


def test_fractional_value_after_first_validation_block_is_rejected():
    counts = np.ones((2, (1 << 20) + 3), dtype=np.float32)[:, ::2]
    counts[-1, -1] = 0.5
    assert not counts.flags.c_contiguous
    with pytest.raises(ValueError, match="finite, non-negative integers"):
        _select_count_dtype(counts)


@pytest.mark.parametrize("maximum", [float(2**63), np.uint64(2**63)])
def test_count_that_would_overflow_int64_is_rejected(maximum):
    with pytest.raises(ValueError, match="int64 range"):
        _select_count_dtype(np.array([[0, maximum]]))


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("backed", [False, True])
@pytest.mark.parametrize("loader", [load_controls, load_analysis_cells])
def test_public_loaders_refuse_fractional_expression(tmp_path, sparse, backed, loader):
    values = np.array([[0.5, 2.0], [1.0, 0.0]], dtype=np.float32)
    data = ad.AnnData(
        X=sp.csr_matrix(values) if sparse else values,
        obs=pd.DataFrame({"pert": ["NTC", "A"]}, index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    if backed:
        path = tmp_path / "fractional.h5ad"
        data.write_h5ad(path)
        data = ad.read_h5ad(path, backed="r")
    kwargs = {"control_selector": "NTC"} if loader is load_controls else {}
    try:
        with pytest.raises(ValueError, match="raw, unnormalized counts"):
            loader(data, perturbation_key="pert", **kwargs)
    finally:
        if backed:
            data.file.close()


def test_validation_is_not_disabled_by_optimized_python():
    result = subprocess.run(
        [sys.executable, "-O", "-c", (
            "import numpy as np\n"
            "from perturbo.core import _select_count_dtype\n"
            "try:\n"
            "    _select_count_dtype(np.array([[0.5, 2.0]]))\n"
            "except ValueError:\n"
            "    print('rejected')\n"
            "else:\n"
            "    raise RuntimeError('fractional counts were accepted')\n"
        )],
        check=True, capture_output=True, text=True,
    )
    assert result.stdout.strip() == "rejected"
