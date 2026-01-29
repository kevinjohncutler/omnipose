import numpy as np
import torch

from omnirefactor.utils import bits
from dask import array as da


def test_to_8_bit_and_16_bit_ranges():
    im = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    out8 = bits.to_8_bit(im)
    out16 = bits.to_16_bit(im)
    assert out8.dtype == np.uint8
    assert out16.dtype == np.uint16
    assert out8.min() == 0 and out8.max() == 255
    assert out16.min() == 0 and out16.max() == 65535


def test_is_integer_variants(tmp_path):
    assert bits.is_integer(3)
    assert bits.is_integer(np.int32(4))

    arr = np.array([1, 2, 3], dtype=np.int64)
    assert bits.is_integer(arr)

    mmap_path = tmp_path / "ints.dat"
    mmap = np.memmap(mmap_path, dtype=np.int16, mode="w+", shape=(4,))
    mmap[:] = 1
    assert bits.is_integer(mmap)

    darr = da.from_array(arr)
    assert bits.is_integer(darr)

    t = torch.tensor([1, 2], dtype=torch.int64)
    assert bits.is_integer(t)

    assert not bits.is_integer(3.14)
    assert not bits.is_integer(torch.tensor([1.0]))
