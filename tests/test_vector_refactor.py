import numpy as np

from omnirefactor.transforms import vector as tvector


def test_vector_field_from_zero_divergence():
    divergence = np.zeros((4, 5), dtype=np.float32)
    vec = tvector.compute_vector_field_from_divergence(divergence, grid_spacing=1)
    assert vec.shape == (2, 4, 5)
    assert np.allclose(vec, 0.0)
