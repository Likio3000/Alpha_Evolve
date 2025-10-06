import numpy as np
import pytest

from alpha_framework import Op


def test_vec_div_scalar_small_denominator():
    """Guard vector/scalar division by clamping very small positive denominators."""
    v = np.array([1.0, -2.0, 3.0])
    buf = {"v": v, "s": 1e-12}
    Op("out", "vec_div_scalar", ("v", "s")).execute(buf, n_stocks=3)
    expected = v / 1e-3
    assert np.allclose(buf["out"], expected)


def test_vec_div_scalar_small_negative_denominator():
    """Guard division when the scalar denominator is tiny and negative."""
    v = np.array([1.0, -2.0, 3.0])
    buf = {"v": v, "s": -1e-12}
    Op("out", "vec_div_scalar", ("v", "s")).execute(buf, n_stocks=3)
    expected = v / 1e-3
    assert np.allclose(buf["out"], expected)


def test_vec_div_scalar_zero_denominator():
    """Fallback to epsilon denominator when the scalar is exactly zero."""
    v = np.array([1.0, -2.0, 3.0])
    buf = {"v": v, "s": 0.0}
    Op("out", "vec_div_scalar", ("v", "s")).execute(buf, n_stocks=3)
    expected = v / 1e-3
    assert np.allclose(buf["out"], expected)


def test_scalar_to_vector_broadcast():
    """Broadcast scalar operands into vector slots to form elementwise products."""
    buf = {"vec": 2.0, "s": 3.0}
    op = Op("out", "vec_mul_scalar", ("vec", "s"))
    op.execute(buf, n_stocks=4)
    assert np.allclose(buf["out"], np.full(4, 6.0))


def test_vector_promoted_in_elementwise_scalar_slot():
    """Promote scalar arguments so elementwise ops can add vector inputs correctly."""
    buf = {"v": np.array([1.0, 2.0, 3.0]), "s": 1.0}
    op = Op("out", "add", ("v", "s"))
    op.execute(buf, n_stocks=3)
    assert np.allclose(buf["out"], np.array([2.0, 3.0, 4.0]))


def test_vector_size_adjustment():
    """Truncate oversized vectors to match the requested number of stocks."""
    buf = {"v": np.array([1.0, 2.0, 3.0, 4.0]), "s": 1.0}
    op = Op("out", "vec_mul_scalar", ("v", "s"))
    op.execute(buf, n_stocks=2)
    assert np.allclose(buf["out"], np.array([1.0, 2.0]))


def test_invalid_vector_type_raises():
    """Raise TypeError when vectors are multidimensional arrays instead of 1-D."""
    buf = {"v": np.array([[1.0, 2.0], [3.0, 4.0]]), "s": 2.0}
    op = Op("out", "vec_mul_scalar", ("v", "s"))
    with pytest.raises(TypeError):
        op.execute(buf, n_stocks=2)

