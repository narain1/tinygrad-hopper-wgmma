"""Correctness smoke test. Exits nonzero on a numerical mismatch."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tinygrad"), str(ROOT)]

import numpy as np
from hopper_wgmma import WSCfg, build_gemm_ws
from hopper_wgmma.runtime import install_runtime_hooks


def main() -> int:
  install_runtime_hooks()
  from tinygrad import Device, Tensor, dtypes
  n = 256  # one default tile: deterministic, small enough for a smoke test
  Tensor.manual_seed(0)
  a = Tensor.randn(n, n, dtype=dtypes.bfloat16).realize()
  b = Tensor.randn(n, n, dtype=dtypes.bfloat16).realize()
  c = Tensor.empty(n, n, dtype=dtypes.bfloat16).realize()
  # N=256 has two 128x256 output tiles. Use two CTAs so this smoke matches the verified one-tile-per-CTA schedule.
  out = build_gemm_ws(n, WSCfg(num_sm=2))(a, b, c)()
  Device["CUDA"].synchronize()
  got = (out.bitcast(dtypes.uint16).numpy().astype(np.uint32) << 16).view(np.float32)
  ref = a.float().numpy() @ b.float().numpy()
  relative = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-9))
  print(f"smoke N={n} relative_error={relative:.6g}")
  if relative >= 2e-2:
    print("WGMMA GEMM mismatch", file=sys.stderr)
    return 1
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
