"""This module provides filtering functions for ndimage data.

Copyright (c) 2026 chatalympics
Licensed under the MIT License
"""

import numpy as np
from numba import jit


def jit8bit_3d_median_filter(src_ary: np.ndarray, kernel_size: int):
    """Performs a fast median filtering on a 3D uint8 volume using a histogram-based approach.

    This function is optimized with Numba's JIT compilation and assumes that the input volume
    has been appropriately padded. It updates the destination array in-place.

    Args:
        src_ary (np.ndarray): Input image to be processed. Requirements: ndim == 3, dtype == np.uint8.
        kernel_size (int): Size of the filter kernel. Must be an odd integer greater than or equal to 1.

    Returns:
        np.ndarray: The filtered 3D volume.
    """

    assert kernel_size >= 1 and kernel_size % 2 == 1
    assert src_ary.ndim == 3
    assert src_ary.dtype == np.uint8

    if kernel_size == 1:
        return src_ary.copy()

    pad_width = kernel_size // 2
    padded_ary = np.pad(src_ary, pad_width, mode="symmetric")
    dst_ary = np.empty_like(src_ary)

    __jit8bit_3d_median_filter(padded_ary, dst_ary, kernel_size)

    return dst_ary


@jit(cache=True)
def __jit8bit_3d_median_filter(padded_ary: np.ndarray, dst_ary: np.ndarray, kernel_size: int):
    """Internal function to perform fast median filtering on a 3D uint8 volume using a histogram-based sliding window.

    This function is JIT-compiled with Numba for performance. It computes the median value within a moving 3D kernel
    by maintaining a histogram of pixel intensities, which is updated incrementally as the window slides along the depth axis.

    Args:
        padded_ary (np.ndarray): Padded input 3D volume. Must be of dtype np.uint8.
        dst_ary (np.ndarray): Output array to store the filtered result. Must be the same shape as the unpadded input.
        kernel_size (int): Size of the cubic kernel. Must be an odd integer ≥ 1.

    Returns:
        None: The result is written directly into `dst_ary`.
    """

    for k in range(dst_ary.shape[2]):
        for j in range(dst_ary.shape[1]):
            i = 0
            window = padded_ary[i : i + kernel_size, j : j + kernel_size, k : k + kernel_size]
            hist_block = np.bincount(window.flatten(), minlength=256).astype(np.uint16)

            for i in range(dst_ary.shape[0]):
                if i != 0:
                    for k1 in range(kernel_size):
                        for k2 in range(kernel_size):
                            hist_block[padded_ary[i - 1, j + k1, k + k2]] -= 1
                            hist_block[padded_ary[i - 1 + kernel_size, j + k1, k + k2]] += 1

                cumsum = 0
                if padded_ary[i, j, k] < 128:
                    for median_value in range(256):
                        cumsum += hist_block[median_value]
                        if cumsum >= (kernel_size**3 + 1) // 2:
                            break
                    dst_ary[i, j, k] = median_value
                else:
                    for median_value in range(256):
                        cumsum += hist_block[-median_value - 1]
                        if cumsum >= (kernel_size**3 + 1) // 2:
                            break
                    dst_ary[i, j, k] = 255 - median_value
