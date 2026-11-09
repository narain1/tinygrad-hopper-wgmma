"""Build results/summary.md from the raw Nsight Compute output that ``modal_runner.py`` saved under results/.

  python scripts/report.py [results_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ncu_table as T  # noqa: E402

LADDER_ROWS = [("tma_ring", "1. TMA + 4-stage ring (`GatedCfg(BN=128, stages=4, consumers=0)`)"),
               ("gated_ws", "2a. warp specialization, role-gated (`GatedCfg()`)"),
               ("role_split", "2b+3. role-split loops, persistent, 16x8 order (`WSCfg()`)"),
               ("role_split_regs", "`WSCfg(regs=(24, 240))`: setmaxnreg on top of 2b+3"),
               ("tma_store", "4. + TMA store epilogue (`tma_store=True`)"),
               ("hilbert", "5. + Hilbert order (`hilbert=True`)"),
               ("cluster", "clusters of 2 instead of Hilbert (`cluster=2`)"),
               ("cluster_hilbert", "Hilbert and clusters (`hilbert=True, cluster=2`)")]
FASTCU = [4, 5, 7, 8, 10, 11, 12]


def shape_of(text: str) -> tuple[int, int, int]:
  d = [int(x) for x in text.split("x")]
  return (d[0], d[0], d[0]) if len(d) == 1 else (d[0], d[1], d[2])


def cublas(t: dict[str, float]) -> float:
  return min(v for k, v in t.items() if "nvjet" in k or "sm90" in k)


def main(root: Path) -> str:
  out = ["# Results", "", "Nsight Compute `gpu__time_duration.sum`, `--clock-control none --cache-control none`, best of 5 launches.",
         "TFLOPS = 2*M*N*K / kernel time. Raw output and tool versions are next to this file.", "", "```", (root / "ladder/env.txt").read_text().strip(), "```", ""]
  for n in (4096, 8192):
    txt = (root / f"ladder/ladder-{n}.txt").read_text(); t, lab = T.parse(txt), T.labels(txt)
    by = {lab[k]: v for k, v in t.items() if k in lab}
    cub = cublas(t)
    out += [f"## Ladder, N={n}", "", "| step | us | TFLOPS | % of cuBLAS |", "|---|---:|---:|---:|"]
    if n == 4096 and (root / "ladder/stock-4096.txt").exists():
      s = max(T.parse((root / "ladder/stock-4096.txt").read_text()).values())
      out.append(f"| stock tinygrad `A @ B` (mma.sync tensor cores) | {s:.0f} | {T.tflops(s, n, n, n):.0f} | {100 * cub / s:.0f}% |")
    for key, title in LADDER_ROWS:
      v = by[key]; out.append(f"| {title} | {v:.1f} | {T.tflops(v, n, n, n):.0f} | {100 * cub / v:.1f}% |")
    out += [f"| cuBLAS via torch (`nvjet`) | {cub:.1f} | {T.tflops(cub, n, n, n):.0f} | 100% |", ""]
  if (root / "ladder-repeat/ladder-4096.txt").exists():
    a, b = T.parse((root / "ladder/ladder-4096.txt").read_text()), T.parse((root / "ladder-repeat/ladder-4096.txt").read_text())
    spread = max(abs(b[k] - a[k]) / a[k] for k in a if k in b)
    out += [f"A second full run of the N=4096 ladder (`ladder-repeat/`) differs from the first by at most {100 * spread:.1f}% on any row.", ""]
  fc = {}
  for k in FASTCU:
    p = root / f"fastcu/fastcu-k{k}-4096.txt"
    if p.exists(): fc[(k, 4096)] = next(v for kk, v in T.parse(p.read_text()).items() if f"matmulKernel{k}<" in kk)
  if (root / "fastcu/fastcu-k12-8192.txt").exists():
    t = T.parse((root / "fastcu/fastcu-k12-8192.txt").read_text())
    fc[(12, 8192)] = next(v for kk, v in t.items() if "matmulKernel12<" in kk)
  if fc:
    out += ["## fast.cu on the same GPU", "", "[pranjalssh/fast.cu](https://github.com/pranjalssh/fast.cu) at the commit in `env.txt`, `h100/matmul.cu` with the env-var patch in `patches/`.", "",
            "| kernel | N | us | TFLOPS |", "|---|---:|---:|---:|"]
    for (k, n), v in sorted(fc.items(), key=lambda kv: (kv[0][1], kv[0][0])):
      out.append(f"| k{k} | {n} | {v:.1f} | {T.tflops(v, n, n, n):.0f} |")
    out.append("")
  if (root / "sm132/sm132-4096.txt").exists():
    txt = (root / "sm132/sm132-4096.txt").read_text(); t, lab = T.parse(txt), T.labels(txt)
    by = {lab[k]: v for k, v in t.items() if k in lab}
    out += ["## 128 vs 132 persistent CTAs, N=4096", "", "| CTAs | us | TFLOPS |", "|---:|---:|---:|"]
    out += [f"| {c} | {by[k]:.1f} | {T.tflops(by[k], 4096, 4096, 4096):.0f} |" for c, k in ((128, "tma_store"), (132, "tma_store_132"))] + [""]
  shapes = sorted((root / "shapes").glob("shape-*.txt"), key=lambda p: [int(x) for x in p.stem[6:].split("x")][::-1] and shape_of(p.stem[6:]))
  if shapes:
    out += ["## Shapes", "", "Best of `role_split`, `tma_store`, `hilbert`, and the BN=192 variants that divide the shape, against cuBLAS (torch).", "",
            "| M x N x K | best config | us | TFLOPS | cuBLAS TFLOPS | % of cuBLAS |", "|---|---|---:|---:|---:|---:|"]
    for p in shapes:
      m, n, k = shape_of(p.stem[6:]); txt = p.read_text(); t, lab = T.parse(txt), T.labels(txt)
      cands = {lab[kk]: v for kk, v in t.items() if kk in lab}
      name, v = min(cands.items(), key=lambda kv: kv[1]); cub = cublas(t)
      out.append(f"| {m}x{n}x{k} | `{name}` | {v:.1f} | {T.tflops(v, m, n, k):.0f} | {T.tflops(cub, m, n, k):.0f} | {100 * cub / v:.1f}% |")
    out.append("")
  return "\n".join(out)


if __name__ == "__main__":
  root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
  text = main(root)
  (root / "summary.md").write_text(text.rstrip("\n") + "\n")
  print(text)
