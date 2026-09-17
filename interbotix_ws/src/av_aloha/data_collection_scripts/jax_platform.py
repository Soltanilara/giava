"""One switch for which backend the coupled IK solver runs on.

WHY THIS IS A FILE AND NOT A LINE
=================================
`JAX_PLATFORMS` has to be in the environment BEFORE jax is imported anywhere,
and jax gets pulled in transitively (jaxlie, pyroki, yourdfpy's viewer...).
Every entry point therefore has to set it as its very first act, and the
several places that did so had drifted into disagreeing with each other --
`data_collection.py` pinned CPU at line 6 while `study_ik.py` pinned CPU again
at import.  One place, one rule, so a change lands everywhere at once.

CPU FOR THE BARE SOLVER, GPU FOR THE FULL ONE
=============================================
The ik_study measurement that pinned CPU was taken on the BARE pose-tracking
solver: three SE3 residuals, ~50 optimisation variables, ~5 ms/solve on CPU
and slower on GPU -- at that size the kernel launch overhead and the
host<->device round trip per LM iteration dominate the arithmetic completely.

The deployed solver is not that solver.  It carries the 180-sphere
self-collision cost (thousands of pairwise distances per residual evaluation,
re-evaluated every LM iteration) and now the tabletop half-space term as
well.  That is the regime where the arithmetic finally outweighs the launch
overhead, which is why this defaults to GPU-first while the study's CPU
finding remains true of what it measured.

    GIAVA_JAX_PLATFORM=gpu    (default) try CUDA, fall back to CPU
    GIAVA_JAX_PLATFORM=cpu              force CPU (the study's configuration)
    GIAVA_JAX_PLATFORM=cuda             GPU only -- fail loudly if absent

or per run:  python data_collection.py --jax cpu

FALLBACK IS SILENT BY CONSTRUCTION, SO IT IS REPORTED
=====================================================
`JAX_PLATFORMS="cuda,cpu"` is jax's own priority list: CUDA if it initialises,
CPU otherwise, with no error either way.  That is the behaviour we want (a
missing driver must not stop data collection) and also the failure mode that
hides a GPU that quietly never engaged -- so `describe()` below is printed
after the solver is built, naming the device actually in use.
"""

from __future__ import annotations

import os
from typing import List, Optional

## Presets -> the JAX_PLATFORMS priority list jax consumes.
PLATFORMS = {
    "gpu": "cuda,cpu",
    "auto": "cuda,cpu",
    "cuda": "cuda",
    "cpu": "cpu",
}

DEFAULT = "gpu"


def apply(default: Optional[str] = None) -> str:
    """Set JAX_PLATFORMS from the environment only -- no argv parsing.

    For LIBRARY modules (study_ik and friends) that are imported rather than
    run: they must not consume `--jax` from sys.argv, but they still have to
    put the variable in place in case they are the first thing to reach jax.
    `select()` calls this after resolving the flag, so an entry point that
    parses the flag and a library that only reads the environment agree."""
    name = str(os.environ.get("GIAVA_JAX_PLATFORM", default or DEFAULT)).strip().lower()
    if name not in PLATFORMS:
        print(f"[jax] ignoring unknown GIAVA_JAX_PLATFORM={name!r}; using {DEFAULT}")
        name = DEFAULT
    os.environ.setdefault("JAX_PLATFORMS", PLATFORMS[name])
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    ## Persistent compilation cache: the coupled IK and the gates are jitted
    ## for a fixed robot, so every launch was re-compiling the same programs.
    ## jax reads these as config flags; the cache survives across sessions
    ## and is keyed on program + jax/XLA version + platform, so a CPU and a
    ## GPU launch keep separate entries.  Delete the dir to force a rebuild.
    os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "giava_jax"))
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.5")
    os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
    ## GIAVA_JAX_CACHE_DEBUG=1 makes jax print WHY a cache lookup missed --
    ## the only way to tell a genuinely new program from a key that shifts
    ## run to run (a constant baked in from measured hardware, say).
    if os.environ.get("GIAVA_JAX_CACHE_DEBUG", "").strip() == "1":
        os.environ["JAX_EXPLAIN_CACHE_MISSES"] = "1"
    os.environ["GIAVA_JAX_PLATFORM"] = name
    return name


def select(argv: Optional[List[str]] = None, default: Optional[str] = None) -> str:
    """Read `--jax X` / GIAVA_JAX_PLATFORM and apply it to os.environ.

    MUST run before jax is imported anywhere.  Returns the preset name.
    An explicit JAX_PLATFORMS in the environment always wins -- that is the
    documented escape hatch and nothing here should override an operator who
    set the real variable by hand."""
    try:
        from .collision_modes import take_option
    except ImportError:
        from collision_modes import take_option

    name = take_option(
        "jax", argv,
        os.environ.get("GIAVA_JAX_PLATFORM", default or DEFAULT))
    name = str(name).strip().lower()
    if name not in PLATFORMS:
        raise SystemExit(
            f"--jax must be one of {sorted(PLATFORMS)}, got '{name}'")

    ## apply() does the environment work, including declining to preallocate
    ## 75 % of VRAM next to torch on a small GPU.
    os.environ["GIAVA_JAX_PLATFORM"] = name
    return apply(name)


def describe() -> str:
    """One line naming the backend jax ACTUALLY initialised.

    Call this AFTER the solver is constructed: `jax.devices()` initialises the
    backend, so calling it early would both cost startup time and freeze the
    choice before the solver's own imports have run."""
    requested = os.environ.get("GIAVA_JAX_PLATFORM", DEFAULT)
    want = os.environ.get("JAX_PLATFORMS", "")
    try:
        import jax
        devices = jax.devices()
        kind = devices[0].platform if devices else "none"
        detail = ", ".join(str(d) for d in devices[:4])
    except Exception as exc:
        return f"[jax] could not query devices ({exc}); requested {want!r}"

    line = f"[jax] backend: {kind.upper()}  ({detail})"
    if requested in ("gpu", "auto") and kind == "cpu":
        line += ("\n[jax] asked for the GPU and got CPU -- CUDA did not "
                 "initialise (no jax cuda plugin, or no driver). The solver "
                 "still runs; expect the study's CPU solve times.")
    return line
