"""Make the pinned tinygrad submodule and this repo importable when a script is run from a checkout."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tinygrad"), str(ROOT)]
