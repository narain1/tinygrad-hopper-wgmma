"""Run the checks and measurements on a Modal H200.

  modal run modal_runner.py --action smoke    # correctness of every config (exit 1 on any mismatch)
  modal run modal_runner.py --action ladder   # Nsight Compute: the ladder + cuBLAS + stock tinygrad, N=4096 and 8192
  modal run modal_runner.py --action shapes   # Nsight Compute: configs vs cuBLAS over 12 shapes
  modal run modal_runner.py --action fastcu   # Nsight Compute: pranjalssh/fast.cu kernels on the same GPU

Raw Nsight output and the tool versions land in results/. Only the actions above exist: this is not a generic runner.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

FASTCU_COMMIT = "2dfe5e26aecfd9e5f27bf9d5837deea01acda24b"   # pranjalssh/fast.cu (MIT)
NVCC_FLAGS = ("-std=c++17 -O3 -DNDEBUG -w --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Xcompiler=-fPIE "
              "-Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -gencode arch=compute_90a,code=sm_90a -lcublas -lcuda")

image = (
  modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
  .apt_install("patch", "git", "clang", "llvm")
  .pip_install("numpy==2.2.6", "torch==2.9.1")
  .add_local_dir("tinygrad", remote_path="/work/tinygrad", copy=True)
  .add_local_file("patches/0001-hopper-wgmma-uop.patch", remote_path="/work/0001-hopper-wgmma-uop.patch", copy=True)
  .add_local_file("patches/fastcu-env.patch", remote_path="/work/fastcu-env.patch", copy=True)
  .run_commands(
    "patch --batch --forward -p1 --directory=/work/tinygrad < /work/0001-hopper-wgmma-uop.patch",
    "pip install -e /work/tinygrad",
    f"git clone https://github.com/pranjalssh/fast.cu /opt/fast.cu && git -C /opt/fast.cu checkout {FASTCU_COMMIT}"
    " && git -C /opt/fast.cu apply /work/fastcu-env.patch"
    f" && cd /opt/fast.cu/h100 && nvcc {NVCC_FLAGS} matmul.cu -o /opt/fastcu_mm",
  )
  .add_local_dir("hopper_wgmma", remote_path="/work/hopper_wgmma", copy=True)
  .add_local_dir("scripts", remote_path="/work/scripts", copy=True)
)
app = modal.App("tinygrad-hopper-wgmma")

NCU = ["ncu", "--clock-control", "none", "--cache-control", "none", "--metrics", "gpu__time_duration.sum", "--csv"]
REMOTE_ENV = {"PYTHONPATH": "/work/tinygrad:/work", "DEV": "CUDA", "PYTHONUNBUFFERED": "1"}


def _sh(cmd: str) -> str:
  r = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
  return (r.stdout + r.stderr).strip()


@app.function(image=image, gpu="H200", timeout=3600)
def smoke() -> tuple[int, str]:
  r = subprocess.run([sys.executable, "/work/scripts/smoke.py"], cwd="/work", env={**os.environ, **REMOTE_ENV},
                     text=True, capture_output=True, check=False)
  return r.returncode, r.stdout + r.stderr[-3000:]


@app.function(image=image, gpu="H200", timeout=3600)
def measure(jobs: list[dict]) -> dict:
  """Each job is {label, kind: 'tinygrad'|'fastcu', ...}. Returns the environment and the raw ncu output per job."""
  env_info = "\n".join([
    _sh("nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,memory.total --format=csv,noheader"),
    _sh("nvcc --version | tail -2"), _sh("ncu --version | tail -1"),
    _sh("python -c 'import torch,numpy;print(\"torch\",torch.__version__,\"numpy\",numpy.__version__)'"),
    f"fast.cu {FASTCU_COMMIT}",
  ])
  out = []
  for job in jobs:
    if job["kind"] == "fastcu":
      cmd, env = NCU + ["--kernel-name", "regex:.*", "/opt/fastcu_mm"], {**os.environ, **job["env"]}
    else:
      cmd = NCU + ["--kernel-name", f"regex:{job['regex']}", sys.executable, "/work/scripts/ncu_workload.py", *job["args"]]
      env = {**os.environ, **REMOTE_ENV}
    r = subprocess.run(cmd, cwd="/work", env=env, text=True, capture_output=True, check=False)
    out.append({"label": job["label"], "rc": r.returncode, "stdout": r.stdout, "stderr": r.stderr[-2000:]})
  return {"env": env_info, "jobs": out}


# ---- local side ----------------------------------------------------------------------------------------------------
SHAPES = ["512", "1024", "1536", "2048", "3072", "4096", "6144", "8192",
          "4096x8192x4096", "4096x11008x4096", "1024x16384x4096", "8192x8192x1024"]   # M x N x K
ALL_LADDER = "tma_ring,gated_ws,role_split,role_split_regs,tma_store,hilbert,cluster,cluster_hilbert"
SWEEP_CONFIGS = "role_split,tma_store,hilbert,tma_store_n192,hilbert_n192"
FASTCU_KERNELS = [4, 5, 7, 8, 10, 11, 12]


def _jobs(action: str) -> list[dict]:
  def tg(label, shape, configs, regex, extra=()):
    return {"label": label, "kind": "tinygrad", "regex": regex,
            "args": ["--shape", shape, "--configs", configs, "--reps", "5", *extra]}
  if action == "ladder":
    jobs = [tg(f"ladder-{n}", n, ALL_LADDER, "hopper|nvjet|sm90", ["--torch", "--check"]) for n in ("4096", "8192")]
    return jobs + [tg("stock-4096", "4096", "", "^r_", ["--stock"])]
  if action == "shapes":
    return [tg(f"shape-{s}", s, SWEEP_CONFIGS, "hopper|nvjet|sm90", ["--torch", "--check"]) for s in SHAPES]
  if action == "fastcu":
    jobs = [{"label": f"fastcu-k{k}-4096", "kind": "fastcu", "env": {"MN": "4096", "KERNELS": str(k), "REPEAT": "3"}}
            for k in FASTCU_KERNELS]
    return jobs + [{"label": "fastcu-k12-8192", "kind": "fastcu", "env": {"MN": "8192", "KERNELS": "12", "REPEAT": "3"}}]
  raise SystemExit(f"unknown action {action!r}")


@app.local_entrypoint()
def main(action: str = "smoke"):
  if action == "smoke":
    rc, text = smoke.remote()
    print(text)
    sys.exit(rc)
  jobs = _jobs(action)
  # one container per job for the sweeps (each job carries its own cuBLAS reference, so cross-instance noise cancels in the ratios);
  # the ladder stays in one container so its rows are directly comparable
  runs = [measure.remote(jobs)] if action == "ladder" else list(measure.map([[j] for j in jobs]))
  result = {"env": "\n--\n".join(dict.fromkeys(r["env"] for r in runs)), "jobs": [j for r in runs for j in r["jobs"]]}
  out = Path("results") / action
  out.mkdir(parents=True, exist_ok=True)
  (out / "env.txt").write_text(result["env"] + "\n")
  for job in result["jobs"]:
    (out / f"{job['label']}.txt").write_text(job["stdout"])
    if job["rc"]: print(f"[{job['label']}] exit {job['rc']}: {job['stderr']}")
  (out / "index.json").write_text(json.dumps([{"label": j["label"], "rc": j["rc"]} for j in result["jobs"]], indent=1))
  print(result["env"], f"\nraw output in {out}/")
  sys.exit(1 if any(j["rc"] for j in result["jobs"]) else 0)
