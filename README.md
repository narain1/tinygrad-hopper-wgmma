# tinygrad Hopper WGMMA GEMM

A bf16 matrix multiply for NVIDIA Hopper (H100/H200), written as tinygrad UOps with TMA, mbarriers, `wgmma`, warp
specialization, a TMA-store epilogue, tile orders and thread-block clusters. It reaches **97% of cuBLAS at N=4096 and
99.5% at N=8192** on an H200 (Nsight Compute, table below). It is an experiment, not a library: see
[Limits](#limits).

It accompanies the post [Replicating "Outperforming cuBLAS on H100" in tinygrad](https://blog.narain.dev/posts/hopper-matmul-tinygrad/),
which rebuilds the [fast.cu](https://github.com/pranjalssh/fast.cu) ladder step by step.

## Results

H200, CUDA 12.8, Nsight Compute `gpu__time_duration`, best of 5 launches. cuBLAS is `torch.matmul` on the same inputs.
Every config was also checked against cuBLAS on poisoned output (max relative error printed as 0.00000, 56 of 56 checks). The full tables (all shapes, fast.cu,
tool versions) and the raw Nsight output are in [`results/`](results/summary.md).

| N=4096 | us | TFLOPS | % of cuBLAS |
|---|---:|---:|---:|
| stock tinygrad `A @ B` | 3147 | 44 | 6% |
| TMA + 4-stage ring (`tma_ring`) | 331.8 | 414 | 53% |
| warp specialization, gated (`gated_ws`) | 233.8 | 588 | 76% |
| role-split loops, persistent (`role_split`) | 201.2 | 683 | 88% |
| + TMA store (`tma_store`) | 181.6 | 757 | 97.5% |
| + Hilbert order (`hilbert`) | 180.9 | 760 | 97.9% |
| cuBLAS | 177.1 | 776 | 100% |
| fast.cu k12, same GPU | 177.9 | 773 | |

At N=8192 the best config is 781 TFLOPS against 785 for cuBLAS (99.5%) and 809 for fast.cu k12. This implementation is
1-3% behind fast.cu's last kernel.

## Setup

Needs Linux, a Hopper GPU, CUDA 12.4+ and Python 3.11+. The repo pins tinygrad (git submodule) and applies one patch
that adds `Ops.WGMMA` and the `sm_90a` target:

```sh
git clone --recurse-submodules https://github.com/narain1/tinygrad-hopper-wgmma
cd tinygrad-hopper-wgmma
patch -p1 --directory=tinygrad < patches/0001-hopper-wgmma-uop.patch
python -m pip install -e tinygrad numpy torch     # torch is only needed for the cuBLAS comparison
DEV=CUDA python scripts/smoke.py                  # 13 configs, exits 1 on any mismatch
```

```python
from hopper_wgmma.runtime import install_runtime_hooks
install_runtime_hooks()                           # BEFORE any other CUDA work in the process (see Limits)
from tinygrad import Tensor, dtypes
from hopper_wgmma import WSCfg, build_gemm_ws

cfg = WSCfg(tma_store=True, hilbert=True)
a, b = Tensor.randn(4096, 4096, dtype=dtypes.bfloat16), Tensor.randn(4096, 4096, dtype=dtypes.bfloat16)
c = Tensor.empty(4096, 4096, dtype=dtypes.bfloat16)
launch = build_gemm_ws(4096, cfg)(a, b, c)        # M, N, K must be multiples of the tile (128 x 256 x 64 by default)
out = launch()
```

## Reproduce the measurements on Modal

Modal builds the image (pinned tinygrad + patch, torch 2.9.1, and fast.cu at a pinned commit) and runs on an H200:

```sh
modal run modal_runner.py --action smoke     # correctness
modal run modal_runner.py --action ladder    # N=4096 and 8192 + cuBLAS + stock tinygrad + a poisoned-output check
modal run modal_runner.py --action shapes    # 12 shapes vs cuBLAS
modal run modal_runner.py --action fastcu    # fast.cu kernels 4,5,7,8,10,11,12
python scripts/report.py                     # results/summary.md from the raw Nsight output
```

Absolute numbers move a few percent between H200 instances (two full runs of the ladder differ by at most 1.7% per
row, cuBLAS included), so compare the ratios.

## Layout

| path | what |
|---|---|
| `hopper_wgmma/gemm.py` | PTX primitives, `Ring`, `region`, tile orders, `WSCfg`, `build_gemm_ws` |
| `hopper_wgmma/gated.py` | the first two ladder steps: a TMA ring, then role-gated warp specialization |
| `hopper_wgmma/ladder.py` | the ladder as named configs |
| `hopper_wgmma/runtime.py` | process-wide hooks: dynamic shared memory, launch size, cluster dims |
| `hopper_wgmma/tensor_map.py` | `CUtensorMap` construction through tinygrad's CUDA bindings |
| `patches/0001-hopper-wgmma-uop.patch` | `Ops.WGMMA` and `sm_90a` for the pinned tinygrad |
| `patches/fastcu-env.patch` | env-var knobs (`MN`, `KERNELS`, `REPEAT`) for fast.cu's `h100/matmul.cu` |
| `scripts/` | `smoke.py`, `ncu_workload.py`, `ncu_table.py`, `report.py` |
| `results/` | raw Nsight output, tool versions, `summary.md` |

## Limits

- **bf16 in, fp32 accumulate, bf16 out; K-major tiles; M, N, K must be multiples of the tile.** No edge handling, no
  split-K (so small problems under-fill the GPU: 46% of cuBLAS at 512 and 1024, 84% at 1536).
- **Process-global hooks.** `runtime.py` patches tinygrad's CUDA renderer, function loader and launch lowering so that
  `hopper_*` kernels get dynamic shared memory and cluster dims, and it forces in-process compilation. Call
  `install_runtime_hooks()` before any other CUDA work (tinygrad caches launch lowering), and keep this out of
  processes that run other custom kernels. It depends on tinygrad internals at the pinned commit.
- **Cluster size comes from the kernel name** (`..._cl2`). That is a hack; it belongs in `KernelInfo`.
- **Shared-memory alignment.** The 128-byte swizzle wants 1024-byte aligned tiles. The dynamic buffer is declared
  `__align__(128)`; it has been 1024-aligned on every H200 tested but CUDA does not promise it.
- **`WSCfg.regs` (`setmaxnreg`) does nothing here** (201.4 us vs 201.2 us). The builder gates it per role instead of
  placing it in a `region`, which is what ptxas needs. It is kept as a documented dead end.
- `num_sm=128` is tuned so that 512 tiles split exactly 4 per CTA at N=4096. It is not read from the device.
- Measured on H200 only. H100 is expected to work and has not been run.

## License

MIT. tinygrad (submodule) is MIT. fast.cu is MIT and is fetched, not copied.
