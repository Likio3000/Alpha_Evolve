import numpy as np
import pytest

from alpha_framework import Op


def test_scalar_to_vector_broadcast():
    buf = {"vec": 2.0, "s": 3.0}
    op = Op("out", "vec_mul_scalar", ("vec", "s"))
    op.execute(buf, n_stocks=4)
    assert np.allclose(buf["out"], np.full(4, 6.0))


def test_vector_promoted_in_elementwise_scalar_slot():
    buf = {"v": np.array([1.0, 2.0, 3.0]), "s": 1.0}
    op = Op("out", "add", ("v", "s"))
    op.execute(buf, n_stocks=3)
    assert np.allclose(buf["out"], np.array([2.0, 3.0, 4.0]))


def test_vector_size_adjustment():
    buf = {"v": np.array([1.0, 2.0, 3.0, 4.0]), "s": 1.0}
    op = Op("out", "vec_mul_scalar", ("v", "s"))
    op.execute(buf, n_stocks=2)
    assert np.allclose(buf["out"], np.array([1.0, 2.0]))


def test_invalid_vector_type_raises():
    buf = {"v": np.array([[1.0, 2.0], [3.0, 4.0]]), "s": 2.0}
    op = Op("out", "vec_mul_scalar", ("v", "s"))
    with pytest.raises(TypeError):
        op.execute(buf, n_stocks=2)


