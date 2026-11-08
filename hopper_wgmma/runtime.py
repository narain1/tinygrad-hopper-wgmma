"""Narrow CUDA runtime hooks required by the Hopper GEMM."""
from __future__ import annotations

import ctypes
import functools
import re

_DYNAMIC_SHARED_LIMIT = 48 * 1024
_MAX_EXPERIMENT_DYNAMIC_SHARED = 227 * 1024
_dynamic_bytes_by_kernel: dict[str, int] = {}
_original_launch = None
_installed = False


def _check(status: int, operation: str) -> None:
  if status != 0:
    raise RuntimeError(f"{operation} failed with CUDA status {status}")


def set_dynamic_smem_bytes(kernel_name: str, nbytes: int) -> None:
  if not kernel_name.startswith("hopper_"):
    raise ValueError("dynamic shared memory can only be configured for hopper_ kernels")
  if not isinstance(nbytes, int) or not 0 <= nbytes <= _MAX_EXPERIMENT_DYNAMIC_SHARED:
    raise ValueError(f"dynamic shared-memory size must be an integer from 0 to {_MAX_EXPERIMENT_DYNAMIC_SHARED}")
  _dynamic_bytes_by_kernel[kernel_name] = nbytes


def _is_hopper_kernel(name: str) -> bool:
  name = re.sub(r"\x1b\[[0-9;]*m", "", name)
  return name.startswith("hopper_")


def _dynamic_launch(ctx, call, prg):
  args = list(_original_launch(ctx, call, prg))
  name = prg.src[0].arg.name
  if _is_hopper_kernel(name):
    nbytes = _dynamic_bytes_by_kernel.get(name)
    if nbytes is None:
      raise RuntimeError(f"no dynamic shared-memory size configured for {name!r}")
    args[8] = nbytes
  return tuple(args)


def install_runtime_hooks() -> None:
  """Install process-local hooks. Called automatically by ``build_gemm_ws``."""
  global _installed, _original_launch
  if _installed:
    return
  from tinygrad import helpers
  from tinygrad.renderer.cstyle import CUDARenderer
  from tinygrad.runtime import ops_cuda
  from tinygrad.runtime.autogen import cuda
  from tinygrad.uop.ops import Ops, PatternMatcher, UPat

  # The renderer and CUDA-launch hooks below are process-local. Compile in this process so workers cannot miss them.
  helpers.PARALLEL.value = 0

  old_render_buffer = CUDARenderer.render_buffer
  def render_buffer(renderer, uop):
    # The experiment runs in a dedicated process. The one large LOCAL arena belongs to the Hopper GEMM.
    if uop.addrspace.name == "LOCAL" and uop.max_numel() * uop.dtype.itemsize > _DYNAMIC_SHARED_LIMIT:
      return f"extern __shared__ __align__(128) {renderer._render_dtype(uop.dtype)} {renderer[uop]}[];"
    return old_render_buffer(renderer, uop)

  old_render_kernel = CUDARenderer.render_kernel
  def render_kernel(renderer, function_name, kernel, bufs, uops, prefix=None):
    renderer._hopper_kernel_name = function_name
    source = old_render_kernel(renderer, function_name, kernel, bufs, uops, prefix)
    match = re.search(r"_cl(2)$", function_name)
    if match:
      source = source.replace(f") {function_name}(", f") __cluster_dims__({match.group(1)},1,1) {function_name}(", 1)
    return source
  CUDARenderer.render_buffer = render_buffer
  CUDARenderer.render_kernel = render_kernel

  old_function = ops_cuda.CUDADevice.function
  @functools.cache
  def function(device, lib: bytes, name: str) -> bytes:
    if not _is_hopper_kernel(name):
      return old_function(device, lib, name)
    _check(device.api.cuCtxSetCurrent(device.context), "cuCtxSetCurrent")
    module = device.api.CUmodule()
    _check(device.api.cuModuleLoadData(ctypes.byref(module), lib), "cuModuleLoadData")
    func = device.api.CUfunction()
    _check(device.api.cuModuleGetFunction(ctypes.byref(func), module, name.encode()), "cuModuleGetFunction")
    nbytes = _dynamic_bytes_by_kernel.get(name)
    if nbytes is None:
      raise RuntimeError(f"no dynamic shared-memory size configured for {name!r}")
    if nbytes > _DYNAMIC_SHARED_LIMIT:
      _check(device.api.cuFuncSetAttribute(func, cuda.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, nbytes), "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)")
      _check(device.api.cuFuncSetAttribute(func, cuda.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100), "cuFuncSetAttribute(PREFERRED_SHARED_MEMORY_CARVEOUT)")
    return bytes(func)
  ops_cuda.CUDADevice.function = function

  _original_launch = ops_cuda.cu_launch_kernel
  ops_cuda.pm_cuda = PatternMatcher([(UPat(Ops.CALL, src=(UPat(Ops.PROGRAM, name="prg"),), name="call", allow_any_len=True), _dynamic_launch)]) + ops_cuda.pm_cuda
  _installed = True
