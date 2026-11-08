"""Optional, narrowly scoped Modal runner for the included smoke or benchmark."""
import modal

image = (
  modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
  .apt_install("patch", "clang", "llvm")
  .pip_install("numpy")
  .add_local_dir("tinygrad", remote_path="/work/tinygrad", copy=True)
  .add_local_file("patches/0001-hopper-wgmma-uop.patch", remote_path="/work/0001-hopper-wgmma-uop.patch", copy=True)
  .add_local_dir("hopper_wgmma", remote_path="/work/hopper_wgmma", copy=True)
  .add_local_file("scripts/smoke.py", remote_path="/work/smoke.py", copy=True)
  .add_local_file("scripts/benchmark.py", remote_path="/work/benchmark.py", copy=True)
  .run_commands(
    "patch --batch --forward -p1 --directory=/work/tinygrad < /work/0001-hopper-wgmma-uop.patch",
    "pip install -e /work/tinygrad",
  )
)
app = modal.App("tinygrad-hopper-wgmma")


@app.function(image=image, gpu="H200", timeout=1200)
def run(action: str, size: int = 4096):
  if action not in {"smoke", "benchmark"}:
    raise ValueError("action must be smoke or benchmark")
  if size <= 0:
    raise ValueError("size must be positive")
  import os
  import subprocess
  import sys

  script = "/work/smoke.py" if action == "smoke" else "/work/benchmark.py"
  command = [sys.executable, script] + (["--size", str(size)] if action == "benchmark" else [])
  result = subprocess.run(command, check=False, cwd="/work", env={**os.environ, "PYTHONPATH": "/work/tinygrad:/work", "DEV": "CUDA"}, text=True, capture_output=True)
  if result.returncode:
    raise RuntimeError(result.stderr[-4000:])
  return result.stdout


@app.local_entrypoint()
def main(action: str = "smoke", size: int = 4096):
  print(run.remote(action, size))
