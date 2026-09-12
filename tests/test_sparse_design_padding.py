from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from perturbo.sparse_design import IndexedDesignMatrix, design_matrix_product


INDICES = jnp.array(
    [
        [0, 2, -1, -1],
        [1, 1, 3, -1],
        [2, -1, -1, -1],
        [-1, -1, -1, -1],
    ],
    dtype=jnp.int32,
)


def _dense_reference(values, coefficients):
    valid = INDICES >= 0
    rows = jnp.arange(INDICES.shape[0], dtype=jnp.int32)[:, None]
    columns = jnp.maximum(INDICES, 0)
    dense = jnp.zeros((INDICES.shape[0], coefficients.shape[0]), dtype=values.dtype)
    dense = dense.at[rows, columns].add(jnp.where(valid, values, 0))
    return dense.astype(coefficients.dtype) @ coefficients


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_padded_product_matches_dense_weighted_duplicate_reference(dtype) -> None:
    values = jnp.array(
        [[1.5, -0.25, 7.0, -3.0], [0.5, 1.25, 2.0, 9.0], [-2.0, 4.0, 5.0, 6.0], [3.0, 2.0, 1.0, 0.0]],
        dtype=dtype,
    )
    coefficients = jnp.arange(20, dtype=dtype).reshape(4, 5) / 7
    design = IndexedDesignMatrix(indices=INDICES, values=values, num_columns=4)

    actual = jax.jit(design_matrix_product)(design, coefficients)
    expected = _dense_reference(values, coefficients)

    tolerance = 1e-6 if dtype == jnp.float32 else 1e-12
    assert actual.dtype == dtype
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_padded_product_coefficient_and_value_gradients_match_dense_reference(dtype) -> None:
    values = jnp.array(
        [[1.5, -0.25, 7.0, -3.0], [0.5, 1.25, 2.0, 9.0], [-2.0, 4.0, 5.0, 6.0], [3.0, 2.0, 1.0, 0.0]],
        dtype=dtype,
    )
    coefficients = jnp.linspace(-1.0, 1.0, 20, dtype=dtype).reshape(4, 5)

    def indexed_loss(vals, coefs):
        design = IndexedDesignMatrix(indices=INDICES, values=vals, num_columns=4)
        return jnp.square(design_matrix_product(design, coefs)).sum()

    def dense_loss(vals, coefs):
        return jnp.square(_dense_reference(vals, coefs)).sum()

    actual = jax.jit(jax.grad(indexed_loss, argnums=(0, 1)))(values, coefficients)
    expected = jax.grad(dense_loss, argnums=(0, 1))(values, coefficients)

    tolerance = 1e-6 if dtype == jnp.float32 else 1e-12
    for actual_gradient, expected_gradient in zip(actual, expected, strict=True):
        assert actual_gradient.dtype == dtype
        np.testing.assert_allclose(
            np.asarray(actual_gradient), np.asarray(expected_gradient), rtol=tolerance, atol=tolerance
        )


def test_padding_does_not_read_unreferenced_column_zero() -> None:
    design = IndexedDesignMatrix(
        indices=jnp.array([[2, -1, -1]], dtype=jnp.int32),
        values=jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        num_columns=3,
    )
    coefficients = jnp.array([[jnp.nan], [4.0], [7.0]], dtype=jnp.float32)

    result = jax.jit(design_matrix_product)(design, coefficients)

    np.testing.assert_array_equal(np.asarray(result), np.array([[7.0]], dtype=np.float32))
