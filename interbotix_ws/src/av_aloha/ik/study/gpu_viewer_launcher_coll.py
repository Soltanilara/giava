"""Launch the collision-geometry viewer with an explicit backend banner (GPU
when started with CUDA_VISIBLE_DEVICES=1)."""
import jax

print("VIEWER BACKEND:", jax.default_backend(), jax.devices(), flush=True)

import runpy
import sys

sys.argv = ["view_collision.py", "--port", "8091"]
runpy.run_path("view_collision.py", run_name="__main__")
