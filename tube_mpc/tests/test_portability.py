"""Tests for the portability layer: config round trip, adapter loop,
command modes, and W calibration."""

import numpy as np
import pytest

from tube_mpc.adapter import FilterRunner, SimAdapter
from tube_mpc.calibrate_w import calibrate
from tube_mpc.config import FilterConfig


@pytest.fixture()
def cfg():
    # Built-in defaults: the relative "giava.urdf" resolves package-relative
    # (see config.resolve_urdf), so this works from any cwd. On a machine
    # without a URDF next to the package, skip instead of failing the whole
    # portability suite on a missing fixture file.
    from tube_mpc.config import resolve_urdf

    try:
        resolve_urdf(FilterConfig().urdf_path)
    except FileNotFoundError as e:
        pytest.skip(f"no URDF for the runner tests: {e}")
    return FilterConfig()


def test_config_round_trip(cfg, tmp_path):
    p = tmp_path / "cfg.yaml"
    cfg.save(p)
    cfg2 = FilterConfig.load(p)
    assert cfg2 == cfg


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("dt: 0.02\nnot_a_key: 1\n")
    with pytest.raises(ValueError, match="unknown config keys"):
        FilterConfig.load(p)


@pytest.mark.parametrize("mode", ["position", "velocity", "acceleration"])
def test_runner_all_command_modes(cfg, mode):
    cfg.command_mode = mode
    adapter = SimAdapter(cfg)
    runner = FilterRunner(cfg, adapter)
    lim = runner.model.limits
    n = runner.model.n
    for k in range(60):
        adapter.target = lim.q_max + 1.0 if k > 30 else adapter.target
        out = runner.step()
        assert out["u"] is not None
        q, v = adapter.read_state()
        assert np.all(q <= lim.q_max + 1e-6)
        assert np.all(np.abs(v) <= lim.v_max + 1e-6)


def test_calibrate_recovers_small_disturbance():
    """A well-tracked log should yield a small W; the bound must cover the
    injected noise level but not be wildly inflated."""
    rng = np.random.default_rng(0)
    T, n, dt = 3000, 6, 0.02
    t = np.arange(T) * dt
    q_cmd = 0.6 * np.sin(2 * np.pi * 0.3 * t)[:, None] * np.ones(n)
    noise = 5e-4
    q_meas = q_cmd + noise * rng.standard_normal((T, n))
    v_meas = np.gradient(q_cmd, dt, axis=0)  # clean velocities
    w_q, w_v = calibrate(q_cmd, q_meas, dt, v_meas=v_meas)
    assert np.all(w_q >= noise)          # must cover the noise
    assert np.all(w_q < 50 * noise)      # but stay the same order of magnitude
    assert np.all(w_v < 0.5)
