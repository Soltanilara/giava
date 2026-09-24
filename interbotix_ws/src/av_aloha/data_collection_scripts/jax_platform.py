"""One switch for which backend the coupled IK solver runs on.

This is a file, not a line, because `JAX_PLATFORMS` must be set before jax is
imported ANYWHERE -- and jax arrives transitively (jaxlie, pyroki, yourdfpy).
Entry points and library modules both have to set it, and when they each did
it themselves they drifted into disagreeing.

    GIAVA_JAX_PLATFORM=gpu    (default) try CUDA, fall back to CPU
    GIAVA_JAX_PLATFORM=cpu              force CPU
    GIAVA_JAX_PLATFORM=cuda             GPU only -- fail loudly if absent

or per run:  python data_collection.py --jax cpu

GPU is the default even though ik_study measured CPU faster: that measurement
was on the BARE pose solver (three SE3 residuals), where kernel-launch
overhead dominates.  The deployed solver carries the 180-sphere self-collision
cost and the tabletop term, which is the regime where arithmetic wins.

`JAX_PLATFORMS="cuda,cpu"` falls back silently by design (a missing driver
must not stop data collection), which is also how a GPU that never engaged
stays invisible -- so describe() reports the device actually in use.
"""

from __future__ import annotations

import os
from typing import List, Optional

## Presets -> the JAX_PLATFORMS priority list jax consumes.
PLATFORMS = {
    "gpu": "cuda,cpu",
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
    ## Keyed on program + jax/XLA version + platform, so CPU and GPU launches
    ## keep separate entries.  Delete the dir to force a rebuild.
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

    MUST run before jax is imported anywhere.  An explicit JAX_PLATFORMS in
    the environment always wins -- that is the documented escape hatch."""
    from collision_modes import take_option

    name = take_option(
        "jax", argv,
        os.environ.get("GIAVA_JAX_PLATFORM", default or DEFAULT))
    name = str(name).strip().lower()
    if name not in PLATFORMS:
        raise SystemExit(
            f"--jax must be one of {sorted(PLATFORMS)}, got '{name}'")
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
    if requested == "gpu" and kind == "cpu":
        line += ("\n[jax] asked for the GPU and got CPU -- CUDA did not "
                 "initialise (no jax cuda plugin, or no driver). The solver "
                 "still runs; expect the study's CPU solve times.")
    return line
