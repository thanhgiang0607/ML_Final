"""Test cho meanshift/kernels.py — người phụ trách: B.

Đây là vài test MẪU để bắt đầu; B bổ sung thêm test của mình.
"""
import numpy as np
import pytest

from meanshift.kernels import FlatKernel, GaussianKernel, make_kernel


def test_flat_weights_all_one():
    w = FlatKernel().weights(np.array([0.0, 0.5, 1.0]), 1.0)
    np.testing.assert_array_equal(w, [1.0, 1.0, 1.0])


def test_gaussian_weights_formula():
    h = 2.0
    sq = np.array([0.0, h**2, 4 * h**2])
    w = GaussianKernel().weights(sq, h)
    np.testing.assert_allclose(w, np.exp([-0.0, -0.5, -2.0]))


def test_support():
    assert FlatKernel().support(1.5) == 1.5
    assert GaussianKernel(cutoff=3.0).support(1.5) == pytest.approx(4.5)


def test_make_kernel():
    assert isinstance(make_kernel("flat"), FlatKernel)
    assert isinstance(make_kernel("gaussian"), GaussianKernel)
    with pytest.raises(ValueError):
        make_kernel("abc")
