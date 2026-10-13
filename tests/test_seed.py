import numpy as np

from src.utils.seed import set_seed


def test_set_seed_is_reproducible():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)