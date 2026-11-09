"""Turn ``ncu --csv --metrics gpu__time_duration.sum`` output into best-of-reps kernel times and TFLOPS.

Usage as a library: ``rows = parse(csv_text)`` returns {kernel name: best microseconds}."""
from __future__ import annotations

import csv
import io

UNIT = {"nsecond": 1e-3, "ns": 1e-3, "usecond": 1.0, "us": 1.0, "msecond": 1e3, "ms": 1e3, "second": 1e6, "s": 1e6}


def parse(text: str) -> dict[str, float]:
  best: dict[str, float] = {}
  rows = [line for line in text.splitlines() if line.startswith('"') and "gpu__time_duration" in line]
  for r in csv.reader(io.StringIO("\n".join(rows))):
    name = r[4].split("(")[0]   # drop the argument list of demangled templates
    us = float(r[-1].replace(",", "")) * UNIT[r[-2]]
    best[name] = min(best.get(name, us), us)
  return best


def labels(text: str) -> dict[str, str]:
  """KERNEL <label> <kernel name> lines printed by ncu_workload.py -> {kernel name: label}"""
  out = {}
  for line in text.splitlines():
    if line.startswith("KERNEL "):
      parts = line.split(maxsplit=2)
      if len(parts) == 3: out[parts[2].strip()] = parts[1]
  return out


def tflops(us: float, m: int, n: int, k: int) -> float:
  return 2 * m * n * k / us / 1e6
