

# ndimagetools

`ndimagetools` is a lightweight Python library that provides high‑performance utilities
for processing multi‑dimensional images.  

## install

```bash
uv add https://github.com/chatalympics/ndimagetools.git
```

or

```bash
pip install https://github.com/chatalympics/ndimagetools.git
```

## ndimagetools.filter

The `ndimagetools.filter` module contains fast implementations of commonly used
image‑filtering operations.  

### jit8bit_3d_median_filter

This function is optimized with Numba's JIT compilation and assumes that the input volume
has been appropriately padded. It updates the destination array in-place.

#### Args:

src_ary (np.ndarray): <br>Input image to be processed. Requirements: ndim == 3, dtype == np.uint8.

kernel_size (int): <br>Size of the filter kernel. Must be an odd integer greater than or equal to 1.

#### Returns:
np.ndarray: The filtered 3D volume.

#### Usage

```python
import numpy as np
from ndimagetools.filter import jit8bit_3d_median_filter

test_img = np.random.randint(0, 256, (32, 32, 32), dtype=np.uint8)
kernel_size = 3

filtered_img = jit8bit_3d_median_filter(test_img, kernel_size)
```

#### Note

Performance comparison with scipy.ndimage.median_filter
(Tested on an Apple M1 Pro chip)

```python
test_img = np.random.randint(0, 256, (128, 128, 128), dtype=np.uint8)
```

| Kernel Size| scipy.ndimage| ndimagetools | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.158 [ms] | 0.049 [ms] | 3.22x | 
| 3 | 724 [ms] | 159 [ms] | 4.55x |
| 5 | 2.80 [s] | 0.19 [s] | 14.7x |
| 7 | 7.14 [s] | 0.22 [s] | 32.45x |
| 9 | 14.56 [s] | 0.29 [s] | 50.20x |
| 11 | 26.00 [s] | 0.37 [s] | 70.27x |
