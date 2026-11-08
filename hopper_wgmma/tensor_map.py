"""CUDA tensor-map construction for the GEMM's bf16 3-D TMA layout."""
from __future__ import annotations

import ctypes
import numpy as np


def _check(status: int, operation: str) -> None:
  if status != 0:
    raise RuntimeError(f"{operation} failed with CUDA status {status}")


def make_tensor_map(tensor, box_h: int, box_w: int, height: int, width: int):
  if box_h <= 0 or box_w <= 0 or height <= 0 or width <= 0:
    raise ValueError("tensor-map dimensions must be positive")
  if box_w % 64 or width % 64:
    raise ValueError("tensor-map tile and width must be multiples of 64 bf16 elements")
  from tinygrad import Device, Tensor
  from tinygrad.runtime.autogen import cuda

  tensor.realize()
  Device["CUDA"].synchronize()
  try:
    pointer = int(tensor.uop.base.buffer._buf)
  except (AttributeError, TypeError, ValueError) as exc:
    raise RuntimeError("tensor must be a realized CUDA tensor") from exc
  descriptor = (ctypes.c_uint8 * 128)()
  dimensions = (cuda.cuuint64_t * 5)(64, height, width // 64, 1, 1)
  strides = (cuda.cuuint64_t * 5)(2 * width, 128, 0, 0, 0)
  box = (cuda.cuuint32_t * 5)(64, box_h, box_w // 64, 1, 1)
  unit = (cuda.cuuint32_t * 5)(1, 1, 1, 1, 1)
  _check(cuda.cuTensorMapEncodeTiled(ctypes.cast(descriptor, ctypes.POINTER(cuda.CUtensorMap)), cuda.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, ctypes.c_void_p(pointer), dimensions, strides, box, unit, cuda.CU_TENSOR_MAP_INTERLEAVE_NONE, cuda.CU_TENSOR_MAP_SWIZZLE_128B, 0, cuda.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "cuTensorMapEncodeTiled")
  return Tensor(np.frombuffer(bytes(descriptor), dtype=np.uint8).copy())
