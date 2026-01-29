import numpy as np
import pytest

from omnirefactor.misc import labels as labels_mod


def test_nd_grid_hypercube_labels_basic_centered():
    labels = labels_mod.nd_grid_hypercube_labels((6, 6), side=2, center=True)
    assert labels.shape == (6, 6)
    assert labels.min() == 1
    assert labels.max() == 9
    assert np.unique(labels).size == 9


def test_nd_grid_hypercube_labels_center_false_margins():
    labels = labels_mod.nd_grid_hypercube_labels((7, 7), side=3, center=False)
    assert labels.shape == (7, 7)
    # Last row/col should be outside grid span -> zeros.
    assert np.all(labels[-1, :] == 0)
    assert np.all(labels[:, -1] == 0)


def test_nd_grid_hypercube_labels_invalid_inputs():
    with pytest.raises(ValueError):
        labels_mod.nd_grid_hypercube_labels([[4, 4]], side=2)
    with pytest.raises(ValueError):
        labels_mod.nd_grid_hypercube_labels((4, 4), side=0)
    with pytest.raises(ValueError):
        labels_mod.nd_grid_hypercube_labels((2, 2), side=3)


def test_make_label_matrix_quadrants():
    labels = labels_mod.make_label_matrix(2, 2)
    assert labels.shape == (4, 4)
    assert labels[0, 0] == 0
    assert labels[0, -1] == 2
    assert labels[-1, 0] == 1
    assert labels[-1, -1] == 3


def test_make_label_matrix_invalid():
    with pytest.raises(ValueError):
        labels_mod.make_label_matrix(0, 2)


def test_create_pill_mask_shape_and_content():
    mask = labels_mod.create_pill_mask(R=2, L=4, f=1)
    assert mask.dtype == np.uint8
    assert mask.shape == (11, 15)
    assert mask.sum() > 0


def test_split_and_reconstruct_round_trip():
    array = np.arange(20).reshape(5, 4)
    parts = labels_mod.split_array(array, parts=2)
    rebuilt = labels_mod.reconstruct_array(parts)
    assert np.array_equal(rebuilt, array)


def test_split_array_warns_on_uneven(capsys):
    array = np.arange(20).reshape(5, 4)
    _ = labels_mod.split_array(array, parts=(2,), axes=0)
    captured = capsys.readouterr()
    assert "Warning: Axis 0" in captured.out


def test_split_array_axes_length_mismatch():
    array = np.arange(24).reshape(6, 4)
    with pytest.raises(ValueError):
        labels_mod.split_array(array, parts=(2, 2), axes=(0,))


def test_enumerate_nested_pairs():
    a = [[1, 2], [3, 4]]
    b = [[10, 20], [30, 40]]
    results = list(labels_mod.enumerate_nested(a, b))
    assert results[0] == ([0, 0], 1, 10)
    assert results[1] == ([0, 1], 2, 20)
    assert results[2] == ([1, 0], 3, 30)
    assert results[3] == ([1, 1], 4, 40)
