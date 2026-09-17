"""Launch the trajectory viewer with an explicit backend banner (GPU when
started with CUDA_VISIBLE_DEVICES=1)."""
import jax

print("VIEWER BACKEND:", jax.default_backend(), jax.devices(), flush=True)

import runpy
import sys

sys.argv = ["view_trajectories.py", "--port", "8090"]
runpy.run_path("view_trajectories.py", run_name="__main__")
