"""Launch GEMMs for Nsight Compute. This script does no timing of its own: run it under ``ncu`` (see modal_runner.py).

  --shape MxNxK   problem size (or a single N for a square GEMM)
  --configs a,b   names from hopper_wgmma.ladder.LADDER / SWEEP / EXTRA; invalid ones for the shape are skipped and reported
  --torch         also launch cuBLAS (torch) on the same inputs
  --stock         instead launch tinygrad's own bf16 matmul on the same shape (tensor cores via mma.sync)
  --check         compare each config with cuBLAS once before the timed launches
"""
from __future__ import annotations

import _path  # noqa: F401
import argparse

from hopper_wgmma.runtime import install_runtime_hooks

install_runtime_hooks()

from tinygrad import Device, Tensor, dtypes  # noqa: E402

from hopper_wgmma.ladder import EXTRA, LADDER, SWEEP, TUNE, build  # noqa: E402


def parse_shape(text: str) -> tuple[int, int, int]:
  dims = [int(x) for x in text.lower().split("x")]
  if len(dims) == 1: dims *= 3
  if len(dims) != 3: raise SystemExit(f"bad --shape {text!r}")
  return dims[0], dims[1], dims[2]


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--shape", default="4096")
  ap.add_argument("--configs", default="")
  ap.add_argument("--reps", type=int, default=5)
  ap.add_argument("--torch", action="store_true")
  ap.add_argument("--stock", action="store_true")
  ap.add_argument("--check", action="store_true")
  args = ap.parse_args()
  M, N, K = parse_shape(args.shape)
  Tensor.manual_seed(0)
  a, b = Tensor.randn(M, K, dtype=dtypes.bfloat16).realize(), Tensor.randn(K, N, dtype=dtypes.bfloat16).realize()
  c = Tensor.empty(M, N, dtype=dtypes.bfloat16).realize()
  ta = tb = ref = None
  if args.torch or args.check:
    import torch
    ta, tb = (torch.from_numpy(x.float().numpy()).cuda().bfloat16() for x in (a, b))
    ref = (ta @ tb).float()
  failed = 0
  if args.stock:
    for _ in range(args.reps): (a @ b).realize()
    Device["CUDA"].synchronize()
    print(f"KERNEL stock {M}x{N}x{K}")
  for name in [n for n in args.configs.split(",") if n]:
    cfg = {**LADDER, **SWEEP, **EXTRA, **TUNE}[name]
    try:
      run = build(N, cfg, M, K)
      launch = run(a, b, c)
      c.assign(Tensor.full((M, N), 12345.0, dtype=dtypes.bfloat16)).realize()   # a skipped write can't pass the check
      out = launch(); Device["CUDA"].synchronize()
      if args.check:
        got = torch.from_numpy(((out.bitcast(dtypes.uint16).numpy().astype("uint32")) << 16).view("float32")).cuda()
        rel = float((got - ref).abs().max() / (ref.abs().max() + 1e-9))
        print(f"CHECK {name} rel={rel:.5f} max|ref|={float(ref.abs().max()):.1f} {'ok' if rel < 2e-2 else 'MISMATCH'}")
        failed += rel >= 2e-2
      for _ in range(args.reps): launch()
      Device["CUDA"].synchronize()
      print(f"KERNEL {name} {run.kernel_name}")
    except ValueError as exc:
      print(f"SKIP {name}: {exc}")
  if args.torch:
    import torch
    for _ in range(args.reps): ta @ tb
    torch.cuda.synchronize()
    print("KERNEL cublas torch")
  return 1 if failed else 0


if __name__ == "__main__":
  raise SystemExit(main())
