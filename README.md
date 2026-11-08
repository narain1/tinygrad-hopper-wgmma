# tinygrad Hopper WGMMA GEMM

A compact experiment for one operation: bf16 GEMM on NVIDIA Hopper using TMA, WGMMA, staged shared memory, and tinygrad UOps. It is not a tinygrad fork or a general kernel framework.

## Support matrix

| Item | Status |
| --- | --- |
| GPU | Hopper SM90a only (H100/H200) |
| OS | Linux with a CUDA driver exposing tensor maps |
| Python | 3.11+ |
| GEMM | bf16 `A[M,K] @ B[K,N] -> C[M,N]`, dimensions divisible by `BM x BN x BK` |
| Default configuration | `BM=128`, `BN=256`, `BK=64`, 3 stages, two consumer warpgroups; 128 persistent CTAs is tuned for H100/H200 |
| Runtime hooks | Process-global and tied to this pinned tinygrad revision; run the experiment in its own process |
| macOS | source import and syntax checks only; no CUDA run |
| Cluster multicast / TMA store / attention | intentionally excluded |

## Clean checkout setup

The `tinygrad` git submodule is pinned to `4e45845d0244e6496b107377ffd70af4e171370e`. Initialize it, apply the tracked WGMMA overlay, then install both local packages:

```sh
git submodule update --init --recursive
patch -p1 --directory=tinygrad < patches/0001-hopper-wgmma-uop.patch
python -m pip install -e tinygrad -e .
```

The overlay adds the `WGMMA` UOp, its CUDA renderer lowering, SM90a architecture spelling, and UOp validation. The
package automatically installs the remaining **process-global, fork-version-sensitive** hooks before its included
scripts initialize CUDA, and `build_gemm_ws` repeats the idempotent setup for library callers. They are deliberately restricted to generated `hopper_` kernels: those kernels alone receive dynamic shared
memory, and requested CUDA function attributes are checked. Build all kernels through this package in a dedicated
process; this is an experiment, not a stable tinygrad extension.

Run a correctness smoke test on Hopper:

```sh
DEV=CUDA python scripts/smoke.py
```

The script uses deterministic inputs and returns exit status 1 for a numerical mismatch. Benchmarking is also deterministic, but its result is a host-wall-clock, synchronized average. It deliberately does not claim tinygrad `DEBUG` timing:

```sh
DEV=CUDA python scripts/benchmark.py --size 4096 --iterations 20
```

Optional Modal execution uses only this package, the pinned tinygrad submodule, and the two included scripts. It accepts only `smoke` or `benchmark`; it does not run arbitrary repository scripts:

```sh
modal run modal_runner.py --action smoke
```

## Layout

- `hopper_wgmma/gemm.py`: primitive effects, `Ring`, validated `WSCfg`, and `build_gemm_ws`.
- `hopper_wgmma/tensor_map.py`: checked CUDA tensor-map creation.
- `hopper_wgmma/runtime.py`: narrow dynamic-SMEM and launch setup.
- `patches/`: the upstream tinygrad overlay.

No attention, OKC, historical ladder scripts, or CUDA/C++ benchmark sources are included.
