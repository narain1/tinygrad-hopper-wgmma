"""Draw the blog's three charts (SVG, text kept as text) from the raw Nsight output under results/.

  python scripts/charts.py [results_dir] [out_dir]        needs matplotlib
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import ncu_table as T  # noqa: E402

plt.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif", "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, GREY, ORANGE, RED = "#3b82f6", "#9ca3af", "#f59e0b", "#ef4444"


def ladder_times(root: Path, n: int) -> tuple[dict[str, float], float]:
  txt = (root / f"ladder/ladder-{n}.txt").read_text(); t, lab = T.parse(txt), T.labels(txt)
  return {lab[k]: v for k, v in t.items() if k in lab}, min(v for k, v in t.items() if "nvjet" in k or "sm90" in k)


def fastcu_us(root: Path, k: int, n: int = 4096) -> float:
  t = T.parse((root / f"fastcu/fastcu-k{k}-{n}.txt").read_text())
  return next(v for name, v in t.items() if f"matmulKernel{k}<" in name)


def main(root: Path, out: Path) -> None:
  out.mkdir(parents=True, exist_ok=True)
  n = 4096; flops = lambda us, m=n, nn=n, k=n: T.tflops(us, m, nn, k)  # noqa: E731
  by, cub = ladder_times(root, n)
  stock = max(T.parse((root / "ladder/stock-4096.txt").read_text()).values())

  # 1. the ladder
  steps = [("stock tinygrad\n(mma.sync)", stock), ("TMA\nring", by["tma_ring"]), ("warp spec.\n(gated)", by["gated_ws"]),
           ("role-split\n+ persistent", by["role_split"]), ("+ TMA\nstore", by["tma_store"]), ("+ Hilbert\norder", by["hilbert"])]
  vals = [flops(us) for _, us in steps]
  fig, ax = plt.subplots(figsize=(10, 4.6))
  bars = ax.bar([s for s, _ in steps], vals, color=[GREY] + [BLUE] * 5)
  ax.axhline(flops(cub), color=RED, ls="--", lw=1.5); ax.text(-0.4, flops(cub) + 12, f"cuBLAS: {flops(cub):.0f}", color=RED)
  for b, v in zip(bars, vals):
    inside = v > 650
    ax.text(b.get_x() + b.get_width() / 2, v - 45 if inside else v + 12, f"{v:.0f}", ha="center",
            color="white" if inside else "black", fontweight="bold" if inside else "normal")
  ax.set_ylim(0, 880); ax.set_ylabel("TFLOPS (bf16, N=4096, H200)"); ax.set_title("Each step of the ladder", loc="left")
  fig.tight_layout(); fig.savefig(out / "ladder.svg"); plt.close(fig)

  # 2. same optimization, fast.cu vs this repo
  pairs = [("warp spec.", by["gated_ws"], fastcu_us(root, 5)), ("persistent", by["role_split"], fastcu_us(root, 7)),
           ("TMA store", by["tma_store"], fastcu_us(root, 10)), ("Hilbert", by["hilbert"], fastcu_us(root, 11)),
           ("clusters", by["cluster"], fastcu_us(root, 8)), ("best", by["hilbert"], fastcu_us(root, 12))]
  fig, ax = plt.subplots(figsize=(9.5, 4.2)); w = 0.38; x = range(len(pairs))
  b1 = ax.bar([i - w / 2 for i in x], [flops(f) for _, _, f in pairs], w, color=ORANGE, label="fast.cu (CUDA C++)")
  b2 = ax.bar([i + w / 2 for i in x], [flops(t) for _, t, _ in pairs], w, color=BLUE, label="this repo (tinygrad UOps)")
  for bs in (b1, b2):
    for b in bs: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 4, f"{b.get_height():.0f}", ha="center", fontsize=11)
  ax.set_xticks(list(x), [p[0] for p in pairs]); ax.set_ylim(500, 830); ax.set_ylabel("TFLOPS (N=4096)")
  ax.set_title("Same optimization, same GPU (y-axis starts at 500)", loc="left"); ax.legend(frameon=False, loc="upper left")
  fig.tight_layout(); fig.savefig(out / "fastcu-vs-tinygrad.svg"); plt.close(fig)

  # 3. shapes: best config as a percentage of cuBLAS
  labels, pct = [], []
  files = sorted((root / "shapes").glob("shape-*.txt"), key=lambda p: (lambda d: (d[0] * d[1] * d[2]))(
    [int(x) for x in (p.stem[6:] + ("x" + p.stem[6:]) * 2 if "x" not in p.stem[6:] else p.stem[6:]).split("x")]))
  for p in files:
    d = [int(x) for x in p.stem[6:].split("x")]; m, nn, k = (d * 3)[:3] if len(d) == 1 else d
    txt = p.read_text(); t, lab = T.parse(txt), T.labels(txt)
    best = min(v for kk, v in t.items() if kk in lab); cb = min(v for kk, v in t.items() if "nvjet" in kk or "sm90" in kk)
    labels.append(f"{m}³" if m == nn == k else f"{m}×{nn}×{k}"); pct.append(100 * cb / best)
  fig, ax = plt.subplots(figsize=(10, 4.4))
  bars = ax.bar(labels, pct, color=[GREY if v < 90 else BLUE for v in pct]); ax.axhline(100, color=RED, ls="--", lw=1.2)
  for b, v in zip(bars, pct): ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}", ha="center", fontsize=10)
  ax.set_ylim(0, 115); ax.set_ylabel("% of cuBLAS (best config)"); plt.xticks(rotation=35, ha="right")
  ax.set_title("At cuBLAS from 2048 up, behind on small shapes (no split-K)", loc="left")
  fig.tight_layout(); fig.savefig(out / "shapes.svg"); plt.close(fig)
  print("wrote", sorted(p.name for p in out.glob("*.svg")), "| shapes:", [f"{l}={v:.1f}" for l, v in zip(labels, pct)])


if __name__ == "__main__":
  main(Path(sys.argv[1] if len(sys.argv) > 1 else "results"), Path(sys.argv[2] if len(sys.argv) > 2 else "results/charts"))
