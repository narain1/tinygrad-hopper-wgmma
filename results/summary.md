# Results

Nsight Compute `gpu__time_duration.sum`, `--clock-control none --cache-control none`, best of 5 launches.
TFLOPS = 2*M*N*K / kernel time. Raw output and tool versions are next to this file.

```
NVIDIA H200, 610.57.04, 1980 MHz, 143771 MiB
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
Version 2025.1.1.0 (build 35528883) (public-release)
torch 2.9.1+cu128 numpy 2.2.6
fast.cu 2dfe5e26aecfd9e5f27bf9d5837deea01acda24b
```

## Ladder, N=4096

| step | us | TFLOPS | % of cuBLAS |
|---|---:|---:|---:|
| stock tinygrad `A @ B` (mma.sync tensor cores) | 3147 | 44 | 6% |
| 1. TMA + 4-stage ring (`GatedCfg(BN=128, stages=4, consumers=0)`) | 331.8 | 414 | 53.4% |
| 2a. warp specialization, role-gated (`GatedCfg()`) | 233.8 | 588 | 75.7% |
| 2b+3. role-split loops, persistent, 16x8 order (`WSCfg()`) | 201.2 | 683 | 88.0% |
| `WSCfg(regs=(24, 240))`: setmaxnreg on top of 2b+3 | 201.4 | 682 | 87.9% |
| 4. + TMA store epilogue (`tma_store=True`) | 181.6 | 757 | 97.5% |
| 5. + Hilbert order (`hilbert=True`) | 180.9 | 760 | 97.9% |
| clusters of 2 instead of Hilbert (`cluster=2`) | 184.0 | 747 | 96.2% |
| Hilbert and clusters (`hilbert=True, cluster=2`) | 183.1 | 750 | 96.7% |
| cuBLAS via torch (`nvjet`) | 177.1 | 776 | 100% |

## Ladder, N=8192

| step | us | TFLOPS | % of cuBLAS |
|---|---:|---:|---:|
| 1. TMA + 4-stage ring (`GatedCfg(BN=128, stages=4, consumers=0)`) | 3621.9 | 304 | 38.7% |
| 2a. warp specialization, role-gated (`GatedCfg()`) | 1784.0 | 616 | 78.5% |
| 2b+3. role-split loops, persistent, 16x8 order (`WSCfg()`) | 1495.2 | 735 | 93.7% |
| `WSCfg(regs=(24, 240))`: setmaxnreg on top of 2b+3 | 1494.6 | 736 | 93.7% |
| 4. + TMA store epilogue (`tma_store=True`) | 1414.5 | 777 | 99.0% |
| 5. + Hilbert order (`hilbert=True`) | 1408.5 | 781 | 99.5% |
| clusters of 2 instead of Hilbert (`cluster=2`) | 1408.1 | 781 | 99.5% |
| Hilbert and clusters (`hilbert=True, cluster=2`) | 1405.5 | 782 | 99.7% |
| cuBLAS via torch (`nvjet`) | 1401.0 | 785 | 100% |

A second full run of the N=4096 ladder (`ladder-repeat/`) differs from the first by at most 1.7% on any row.

## fast.cu on the same GPU

[pranjalssh/fast.cu](https://github.com/pranjalssh/fast.cu) at the commit in `env.txt`, `h100/matmul.cu` with the env-var patch in `patches/`.

| kernel | N | us | TFLOPS |
|---|---:|---:|---:|
| k4 | 4096 | 266.8 | 515 |
| k5 | 4096 | 223.7 | 614 |
| k7 | 4096 | 189.5 | 725 |
| k8 | 4096 | 187.6 | 733 |
| k10 | 4096 | 182.0 | 755 |
| k11 | 4096 | 180.5 | 761 |
| k12 | 4096 | 177.9 | 773 |
| k12 | 8192 | 1359.3 | 809 |

## 128 vs 132 persistent CTAs, N=4096

| CTAs | us | TFLOPS |
|---:|---:|---:|
| 128 | 181.8 | 756 |
| 132 | 185.7 | 740 |

## Shapes

Best of `role_split`, `tma_store`, `hilbert`, and the BN=192 variants that divide the shape, against cuBLAS (torch).

| M x N x K | best config | us | TFLOPS | cuBLAS TFLOPS | % of cuBLAS |
|---|---|---:|---:|---:|---:|
| 512x512x512 | `tma_store` | 8.7 | 31 | 66 | 46.9% |
| 1024x1024x1024 | `tma_store` | 13.7 | 157 | 346 | 45.4% |
| 1024x16384x4096 | `hilbert` | 181.0 | 759 | 771 | 98.5% |
| 1536x1536x1536 | `tma_store_n192` | 14.9 | 486 | 582 | 83.5% |
| 2048x2048x2048 | `tma_store` | 25.1 | 685 | 705 | 97.1% |
| 3072x3072x3072 | `hilbert_n192` | 80.1 | 724 | 754 | 96.0% |
| 4096x4096x4096 | `hilbert` | 179.5 | 766 | 788 | 97.2% |
| 4096x8192x4096 | `tma_store` | 347.2 | 792 | 808 | 97.9% |
| 4096x11008x4096 | `tma_store` | 488.9 | 756 | 755 | 100.0% |
| 6144x6144x6144 | `hilbert` | 575.9 | 805 | 802 | 100.4% |
| 8192x8192x1024 | `hilbert` | 190.0 | 724 | 708 | 102.1% |
| 8192x8192x8192 | `hilbert` | 1385.3 | 794 | 797 | 99.6% |
