"""Correctness check on a Hopper GPU. Every config below runs on poisoned output and is compared with an fp32 host
reference of the same bf16 inputs. Exits 1 if any config is wrong or fails to run."""
from __future__ import annotations

import _path  # noqa: F401  (must come first)

import numpy as np

from hopper_wgmma.runtime import install_runtime_hooks

install_runtime_hooks()   # before any CUDA work: the launch hook must be in place when the first kernel is lowered

from tinygrad import Device, Tensor, dtypes  # noqa: E402

from hopper_wgmma import WSCfg  # noqa: E402
from hopper_wgmma.ladder import LADDER, build  # noqa: E402

# (label, config, (M, N, K)). 512^3 has 8 tiles of 128x256, so every schedule (hilbert, cluster) is valid.
CASES = [(name, cfg, (512, 512, 512)) for name, cfg in LADDER.items()] + [
  ("rectangular", WSCfg(), (512, 1024, 256)),
  ("rectangular_tma_store", WSCfg(tma_store=True), (1024, 512, 384)),
  ("bn192", WSCfg(BN=192, tma_store=True), (256, 768, 256)),
  ("group_m_1d", WSCfg(group_m=3), (768, 512, 256)),   # 6 row tiles, not divisible by 3 supertiles: 1D GROUP_M order
  ("one_consumer", WSCfg(consumers=1), (512, 512, 512)),
]
REPS = 3
TOL = 2e-2


def to_f32(t: Tensor) -> np.ndarray:
  return (t.bitcast(dtypes.uint16).numpy().astype(np.uint32) << 16).view(np.float32)


def main() -> int:
  failures = []
  for label, cfg, (M, N, K) in CASES:
    try:
      Tensor.manual_seed(0)
      a, b = Tensor.randn(M, K, dtype=dtypes.bfloat16).realize(), Tensor.randn(K, N, dtype=dtypes.bfloat16).realize()
      c = Tensor.empty(M, N, dtype=dtypes.bfloat16).realize()
      ref = a.float().numpy() @ b.float().numpy()
      launch = build(N, cfg, M, K)(a, b, c)
      worst = 0.0
      for _ in range(REPS):
        c.assign(Tensor.full((M, N), 12345.0, dtype=dtypes.bfloat16)).realize()   # stale or missing writes can't pass
        out = launch()
        Device["CUDA"].synchronize()
        worst = max(worst, float(np.max(np.abs(to_f32(out) - ref)) / (np.max(np.abs(ref)) + 1e-9)))
      status = "ok" if worst < TOL else "MISMATCH"
      if worst >= TOL: failures.append(label)
      print(f"{status:9s} {label:24s} {M}x{N}x{K}  relative_error={worst:.5f}")
    except Exception as exc:   # report every case instead of stopping at the first
      failures.append(label)
      print(f"ERROR     {label:24s} {M}x{N}x{K}  {type(exc).__name__}: {str(exc)[:300]}")
      if "CUDA" in str(exc) or "hang" in str(exc): break   # a faulted context cannot run later cases
  print(f"{len(CASES) - len(failures)}/{len(CASES)} passed")
  return 1 if failures else 0


if __name__ == "__main__":
  raise SystemExit(main())
