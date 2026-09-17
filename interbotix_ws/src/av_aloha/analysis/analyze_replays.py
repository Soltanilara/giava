"""Compare replay runs of one episode: how well does this hardware repeat?

WHAT THIS ANSWERS
=================
replay_episode.py --record --repeat N sends the SAME commanded trajectory at
the SAME rate N times and records what the arms actually did.  Two different
numbers come out of that, and conflating them is the usual mistake:

  TRACKING     measured - commanded, within a run.  How well the arm follows an
               order.  Dominated by controller lag: most of it is the arm being
               a few ticks behind, not the arm being in the wrong place.

  REPEATABILITY  measured(run i) - measured(run j).  How reproducible the
               hardware is given an identical command stream.  Backlash,
               gravity sag, thermal drift and driver-side clamping all land
               here.  THIS IS THE FLOOR: a policy trained on this rig cannot be
               asked to reproduce a trajectory more precisely than the rig
               reproduces its own, so it is the number to quote when a rollout
               "nearly" hits a target.

  LAG          the sample shift that best aligns measured to commanded, per
               joint, from cross-correlation.  Subtracting it splits tracking
               error into "late" (fixable by phase, or by feeding the policy
               the same lag it trained with) and "wrong" (not fixable that way).

DRIFT is reported too: run 1 vs run N tells you whether the rig is stationary
or warming up.  If repeatability degrades monotonically with run index, the
servos are heating and the early runs are not comparable to the late ones.

Joint space, not pixels: encoders resolve ~0.001 rad, the scene cameras resolve
a few millimetres at best, and pixel differences confound arm error with
lighting and object placement.  The recorded video is there to be looked at,
not differenced.
"""

from pathlib import Path
import argparse
import json

import numpy as np

try:
    from lerobot.datasets import LeRobotDataset
except ImportError as exc:
    raise SystemExit(f"lerobot is required: {exc}")


def open_local_dataset(root, video_backend="pyav"):
    """Open a dataset that lives on disk, without asking the hub about it.

    LeRobotDataset's first positional argument is a REPO ID, not a path.
    Passing the path there sends it to huggingface_hub, which rejects it as a
    malformed repo id -- so a purely local dataset could not be opened at all.
    The repo id in meta/info.json is the dataset's own record of what it would
    be called on the hub; `root` is what actually gets read.
    """
    from lerobot.datasets import LeRobotDataset

    root = Path(root)
    if not (root / "meta" / "episodes").is_dir():
        raise SystemExit(
            f"{root} has no meta/episodes -- it was never finalized, so its "
            "episode boundaries were never written and nothing can read it "
            "back.\n  Episode metadata is buffered (10 episodes) and flushed "
            "by finalize(), which runs when data collection exits with 'q' or "
            "through atexit.\n  A session killed with SIGKILL never gets "
            "there.  A session still RUNNING has not got there yet -- quit it "
            "first, then replay.")
    repo_id = "local/dataset"
    info = root / "meta" / "info.json"
    if info.is_file():
        try:
            repo_id = json.load(open(info)).get("repo_id") or repo_id
        except Exception:
            pass
    return LeRobotDataset(repo_id, root=str(root), video_backend=video_backend)


def episode_rows(ds, ep):
    """[from, to) row indices of one episode, whichever accessor exists."""
    edi = getattr(ds, "episode_data_index", None)
    if edi is not None:
        return int(edi["from"][ep]), int(edi["to"][ep])
    meta_ep = ds.meta.episodes[ep]
    return (int(meta_ep["dataset_from_index"]), int(meta_ep["dataset_to_index"]))


def columns(ds, ep, keys):
    """Requested columns for one episode as float64 arrays, no video decoded."""
    lo, hi = episode_rows(ds, ep)
    hf = getattr(ds, "hf_dataset", None)
    if hf is None:
        raise SystemExit("this dataset exposes no hf_dataset; cannot read "
                         "columns without decoding video")
    rows = hf.select(range(lo, hi))
    out = {}
    for k in keys:
        out[k] = np.asarray([np.asarray(v, dtype=np.float64).reshape(-1)
                             for v in rows[k]], dtype=np.float64)
    return out


def best_lag(cmd, meas, max_lag):
    """Samples `meas` trails `cmd` by, maximising correlation. 0 if flat.

    Both series are mean-removed first: a constant offset (gravity sag on a
    held pose) correlates with everything and would otherwise pin the answer
    at whatever lag the trend happens to favour.
    """
    c = cmd - cmd.mean()
    m = meas - meas.mean()
    if np.std(c) < 1e-9 or np.std(m) < 1e-9:
        return 0
    best, best_r = 0, -np.inf
    for lag in range(0, max_lag + 1):
        a = c[:len(c) - lag] if lag else c
        b = m[lag:] if lag else m
        if len(a) < 10:
            break
        r = float(np.corrcoef(a, b)[0, 1])
        if r > best_r:
            best_r, best = r, lag
    return best


def by_arm(names):
    """joint names -> {arm: [column indices]}, from the name prefix.

    A 21-row per-joint table buries the one fact that matters most -- WHICH ARM
    is not tracking.  The rollup is printed always; --per-joint stays for when
    the arm is known and the joint is the question.
    """
    groups = {}
    for i, nm in enumerate(names):
        groups.setdefault(nm.split("_")[0], []).append(i)
    return groups


def wrap_pi(x):
    """Differences into [-pi, pi].

    The middle waist is a MULTITURN joint: the driver records it relative to
    whichever 2 pi-equivalent frame the servo booted into, so the same physical
    pose can read 2 pi apart between two sessions (replay_episode.align_middle_waist
    exists for exactly this).  An unwrapped comparison reports that frame
    difference as a 360-degree error, which is the largest number in the table
    and means nothing.
    """
    return (x + np.pi) % (2 * np.pi) - np.pi


def rms(x, axis=0):
    return np.sqrt(np.mean(np.square(x), axis=axis))


def fmt(v):
    """rad and degrees, because one of them is always the intuitive one."""
    return f"{v:.4f} rad ({np.degrees(v):6.2f} deg)"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("replay_root", type=str,
                   help="A dataset written by replay_episode.py --record")
    p.add_argument("--max-lag", type=int, default=20,
                   help="Largest lag searched, in samples (default 20).")
    p.add_argument("--per-joint", action="store_true",
                   help="Break every number down by joint.")
    p.add_argument("--no-source", action="store_true",
                   help="Skip the comparison against the source episode.")
    args = p.parse_args()

    root = Path(args.replay_root).expanduser().resolve()
    ds = open_local_dataset(root)
    prov = {}
    prov_path = root / "replay_meta.json"
    if prov_path.is_file():
        prov = json.loads(prov_path.read_text())
        print(f"replay of {prov.get('source_root')} episode "
              f"{prov.get('source_episode')}  ({prov.get('created')})")
        if prov.get("note"):
            print(f"note: {prov['note']}")
    else:
        print("[warn] no replay_meta.json -- cannot identify the source "
              "episode; run-to-run numbers are still valid.")

    n_runs = ds.num_episodes
    fps = ds.fps
    names = prov.get("state_names") or [f"j{i}" for i in range(
        len(columns(ds, 0, ["observation.state"])["observation.state"][0]))]
    print(f"{n_runs} run(s) at {fps} Hz, {len(names)} state dims\n")

    runs = []
    for ep in range(n_runs):
        c = columns(ds, ep, ["observation.state", "action"])
        runs.append((c["observation.state"], c["action"]))
        print(f"  run {ep}: {len(c['action'])} steps")
    n = min(len(a) for a, _ in runs)
    if any(len(a) != n for a, _ in runs):
        print(f"[warn] runs differ in length; comparing the first {n} steps")
    print()

    # ---- lag ------------------------------------------------------------
    print("EXECUTION LAG (measured trails commanded)")
    lags = np.zeros((n_runs, len(names)), dtype=int)
    for ep, (meas, cmd) in enumerate(runs):
        for j in range(len(names)):
            lags[ep, j] = best_lag(cmd[:n, j], meas[:n, j], args.max_lag)
    med = np.median(lags, axis=0)
    print(f"  median over runs: {np.median(med):.0f} samples "
          f"= {1000 * np.median(med) / fps:.0f} ms")
    ## A joint whose best lag IS the search ceiling has not been measured, it
    ## has been truncated -- the real lag is that or more.  Silently reporting
    ## the ceiling as the answer would understate it.
    capped = [names[j] for j in range(len(names)) if med[j] >= args.max_lag]
    if capped:
        print(f"  [!] {len(capped)} joint(s) hit the --max-lag ceiling of "
              f"{args.max_lag} samples ({1000 * args.max_lag / fps:.0f} ms): "
              f"{', '.join(capped[:4])}{'...' if len(capped) > 4 else ''}")
        print(f"      The true lag is AT LEAST that. Re-run with "
              f"--max-lag {2 * args.max_lag} to find it.")
    ## A joint that never moves correlates with nothing; its 0 is "no answer",
    ## not "no lag".
    flat = [names[j] for j in range(len(names)) if med[j] == 0]
    if flat:
        print(f"  ({len(flat)} joint(s) report 0 -- too little motion to "
              f"measure: {', '.join(flat[:4])}{'...' if len(flat) > 4 else ''})")
    if args.per_joint:
        for j, nm in enumerate(names):
            print(f"    {nm:24s} {med[j]:4.0f} samples "
                  f"({1000 * med[j] / fps:5.0f} ms)")
    print()

    # ---- tracking, raw and lag-corrected --------------------------------
    print("TRACKING ERROR (measured - commanded, within a run)")
    for ep, (meas, cmd) in enumerate(runs):
        e = meas[:n] - cmd[:n]
        ## Each joint has its OWN lag, so shifting them independently yields
        ## columns of different lengths.  Trim every column to the same window
        ## -- the largest lag decides it -- or the stack is ragged and numpy
        ## refuses.  Comparing joints over different spans would be wrong
        ## anyway: the rms would mix different parts of the trajectory.
        span = n - int(lags[ep].max())
        if span > 10:
            shifted = np.stack([meas[lags[ep, j]:lags[ep, j] + span, j]
                                - cmd[:span, j]
                                for j in range(len(names))], axis=1)
        else:
            shifted = e
        print(f"  run {ep}: rms {fmt(float(np.sqrt(np.mean(e ** 2))))} | "
              f"max {fmt(float(np.abs(e).max()))} | "
              f"lag-corrected rms {fmt(float(np.sqrt(np.mean(shifted ** 2))))}")
        r = rms(e)
        groups = by_arm(names)
        print("    per arm: " + "  ".join(
            f"{arm} {np.degrees(float(np.sqrt(np.mean(r[idx] ** 2)))):.1f} deg"
            for arm, idx in groups.items()))
        if args.per_joint:
            for j, nm in enumerate(names):
                print(f"    {nm:24s} {fmt(float(r[j]))}")
    print()

    # ---- repeatability --------------------------------------------------
    print("REPEATABILITY (measured vs measured, across runs)  <- the floor")
    if n_runs < 2:
        print("  only one run; re-run with --repeat 2 or more.")
    else:
        pair_rms, worst = [], (None, -1.0)
        for i in range(n_runs):
            for k in range(i + 1, n_runs):
                d = runs[i][0][:n] - runs[k][0][:n]
                v = float(np.sqrt(np.mean(d ** 2)))
                pair_rms.append(v)
                if v > worst[1]:
                    worst = ((i, k), v)
        print(f"  mean over {len(pair_rms)} pair(s): {fmt(float(np.mean(pair_rms)))}")
        print(f"  worst pair {worst[0]}:            {fmt(worst[1])}")

        allm = np.stack([r[0][:n] for r in runs])       # [runs, steps, joints]
        spread = allm.std(axis=0)                        # per step, per joint
        print(f"  per-step spread: mean {fmt(float(spread.mean()))}, "
              f"max {fmt(float(spread.max()))} "
              f"(at step {int(spread.max(axis=1).argmax())} of {n})")
        if args.per_joint:
            pj = spread.mean(axis=0)
            for j, nm in enumerate(names):
                print(f"    {nm:24s} {fmt(float(pj[j]))}")

        ## Monotone growth here means the rig is not stationary -- servos
        ## warming, something shifting -- and early runs are not comparable
        ## with late ones.
        drift = [float(np.sqrt(np.mean((runs[0][0][:n] - runs[k][0][:n]) ** 2)))
                 for k in range(1, n_runs)]
        print("  vs run 0: " + ", ".join(
            f"run {k + 1} {np.degrees(d):.2f} deg" for k, d in enumerate(drift)))
    print()

    # ---- against the original recording ---------------------------------
    if not args.no_source and prov.get("source_root"):
        src_root = Path(prov["source_root"])
        if not src_root.is_dir():
            print(f"[source] {src_root} not found -- skipping")
            return
        try:
            src = open_local_dataset(src_root)
            c = columns(src, int(prov["source_episode"]),
                        ["observation.state", "action"])
        except Exception as exc:
            print(f"[source] could not read the source episode ({exc})")
            return
        s_meas = c["observation.state"]
        m = min(n, len(s_meas))
        print("VS THE ORIGINAL RECORDING (measured vs measured)")
        print("  teleop is in the loop there and not here, so this includes "
              "everything\n  the replay cannot reproduce -- read it as an upper "
              "bound, not as error.")
        wrapped = 0
        for ep, (meas, _) in enumerate(runs):
            raw = meas[:m] - s_meas[:m]
            d = wrap_pi(raw)
            wrapped += int((np.abs(raw - d) > 1e-6).any(axis=0).sum())
            r = rms(d)
            groups = by_arm(names)
            print(f"  run {ep}: rms {fmt(float(np.sqrt(np.mean(d ** 2))))} | "
                  f"max {fmt(float(np.abs(d).max()))}")
            print("    per arm: " + "  ".join(
                f"{arm} {np.degrees(float(np.sqrt(np.mean(r[idx] ** 2)))):.1f} deg"
                for arm, idx in groups.items()))
        if wrapped:
            print("  (differences wrapped into [-pi, pi]: a multiturn joint "
                  "sat in a different 2 pi frame)")


if __name__ == "__main__":
    main()
