from __future__ import annotations

import time
import unittest
from contextlib import contextmanager

import numpy as np
from scipy.ndimage import median_filter

from ndimagetools.filter import jit8bit_3d_median_filter


@contextmanager
def timer():
    t = time.perf_counter()
    yield None
    print("Elapsed:", time.perf_counter() - t)


class Test3DMedian(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)

    def test_processing(self):
        for kernel_size in [1, 3, 5, 7, 9, 11]:
            for trial in range(1, 4):
                print(f"{kernel_size=}, {trial=}")
                test_img = np.random.randint(0, 256, (32, 32, 32), dtype=np.uint8)

                print("scipy.ndimage")
                with timer():
                    ary1 = median_filter(test_img, kernel_size)
                print("ndimagetools.filter")
                with timer():
                    ary2 = jit8bit_3d_median_filter(test_img, kernel_size)

                self.assertTrue(np.all(ary1 == ary2))
            print()

    def test_dtype(self):
        for dtype in [np.uint16, np.int8, np.float32, np.complex64]:
            test_img = np.zeros((32, 32, 32), dtype=dtype)
            with self.assertRaises(AssertionError):
                _ = jit8bit_3d_median_filter(test_img, 3)

    def test_kernel_size(self):
        for kernel_size in [2, 4, 6, 8, 10]:
            test_img = np.random.randint(0, 256, (32, 32, 32), dtype=np.uint8)
            with self.assertRaises(AssertionError):
                _ = jit8bit_3d_median_filter(test_img, kernel_size)

    def test_ndim(self):
        with self.assertRaises(AssertionError):
            test_img = np.random.randint(0, 256, (32), dtype=np.uint8)
            _ = jit8bit_3d_median_filter(test_img, 3)
        with self.assertRaises(AssertionError):
            test_img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
            _ = jit8bit_3d_median_filter(test_img, 3)
        with self.assertRaises(AssertionError):
            test_img = np.random.randint(0, 256, (32, 32, 32, 32), dtype=np.uint8)
            _ = jit8bit_3d_median_filter(test_img, 3)


if __name__ == "__main__":
    unittest.main()
