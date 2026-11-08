"""Deterministic host-facing GEMM benchmark (not DEBUG timing)."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tinygrad"), str(ROOT)]

from hopper_wgmma import WSCfg, build_gemm_ws
from hopper_wgmma.runtime import install_runtime_hooks


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--size", type=int, default=4096)
  parser.add_argument("--iterations", type=int, default=20)
  args = parser.parse_args()
  install_runtime_hooks()
  from tinygrad import Device, Tensor, dtypes
  Tensor.manual_seed(0)
  a = Tensor.randn(args.size, args.size, dtype=dtypes.bfloat16).realize()
  b = Tensor.randn(args.size, args.size, dtype=dtypes.bfloat16).realize()
  c = Tensor.empty(args.size, args.size, dtype=dtypes.bfloat16).realize()
  launch = build_gemm_ws(args.size, WSCfg())(a, b, c)
  launch(); Device["CUDA"].synchronize()
  start = time.perf_counter()
  for _ in range(args.iterations): launch()
  Device["CUDA"].synchronize()
  seconds = (time.perf_counter() - start) / args.iterations
  print(f"N={args.size} host-synchronized average={seconds * 1e3:.3f} ms, {2 * args.size**3 / seconds / 1e12:.1f} TFLOP/s")

if __name__ == "__main__": main()
