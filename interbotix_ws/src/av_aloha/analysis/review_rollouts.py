"""One local HTML page per review: every scored rollout with its video.

    python review_rollouts.py                        # everything
    python review_rollouts.py --since 2026-09-10     # today's
    python review_rollouts.py --match cube --since 2026-09-10 --out rollouts/review_cube.html

Writes next to the videos (rollouts/ by default) so the <video> tags use
relative paths and the whole folder can be copied or shared as a unit.

Per condition (checkpoint @ step, n_action_steps, target): a summary row with
Wilson 95% intervals per stage, then one card per episode -- stages, flags,
automatic measurements, your note, and the mp4 inline.  Sorting is by
condition then episode, so the same scene under two settings sits together.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import os
from collections import defaultdict
from pathlib import Path


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def condition(rec):
    parts = Path(rec["checkpoint"]).parts
    job = step = "?"
    if "train" in parts:
        i = parts.index("train")
        if i + 1 < len(parts):
            job = parts[i + 1]
    if "checkpoints" in parts:
        i = parts.index("checkpoints")
        if i + 1 < len(parts):
            step = parts[i + 1]
    tgt = f"  target={rec['target']}" if rec.get("target") else ""
    return f"{job} @ {step}   n_action_steps={rec.get('n_action_steps')}{tgt}"


CSS = """
body{font:14px/1.45 system-ui,sans-serif;margin:0;padding:24px;background:#fafafa;color:#222}
h1{font-size:20px;margin:0 0 4px}  .sub{color:#666;margin-bottom:24px}
h2{font-size:15px;margin:32px 0 8px;padding-bottom:6px;border-bottom:2px solid #ddd}
table.sum{border-collapse:collapse;margin:8px 0 16px;font-variant-numeric:tabular-nums}
table.sum th,table.sum td{padding:4px 12px;text-align:right;border-bottom:1px solid #eee}
table.sum th{font-weight:600;color:#555;font-size:12px}  table.sum td:first-child,table.sum th:first-child{text-align:left}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:14px}
.card{background:#fff;border:1px solid #e3e3e3;border-radius:8px;padding:12px}
.card video{width:100%;border-radius:4px;background:#000}
.hdr{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px}
.ep{font-weight:600}  .meta{color:#777;font-size:12px}
.stages{margin:6px 0;font-size:13px}
.st{display:inline-block;padding:1px 7px;margin-right:4px;border-radius:10px;background:#eee;color:#888}
.st.ok{background:#d9f2df;color:#1c6b32}
.flag{display:inline-block;padding:1px 7px;margin-right:4px;border-radius:10px;background:#fde8c8;color:#8a4b00;font-size:12px}
.auto{color:#555;font-size:12px;margin:4px 0}  .note{margin-top:6px;font-style:italic}
.ended{color:#a33;font-size:12px}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores", default=None)
    ap.add_argument("--since", default=None, help="YYYY-MM-DD")
    ap.add_argument("--match", default=None, help="substring of the checkpoint path")
    ap.add_argument("--out", default=None, help="default rollouts/review_<date>.html")
    args = ap.parse_args()

    here = Path(__file__).parent
    scores = Path(args.scores or here / "rollouts" / "rollout_scores.jsonl")
    if not scores.exists():
        raise SystemExit(f"{scores} not found")
    cutoff = (dt.datetime.strptime(args.since, "%Y-%m-%d").timestamp()
              if args.since else None)

    recs = []
    for line in scores.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if cutoff and r.get("wall_time", 0) < cutoff:
            continue
        if args.match and args.match not in r.get("checkpoint", ""):
            continue
        recs.append(r)
    if not recs:
        raise SystemExit("no records matched")

    out = Path(args.out or scores.parent / f"review_{dt.date.today():%Y%m%d}.html")
    out.parent.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for r in recs:
        groups[condition(r)].append(r)

    parts = [f"<style>{CSS}</style>",
             f"<h1>Rollout review</h1>",
             f"<div class=sub>{len(recs)} episodes across {len(groups)} conditions"
             + (f" &middot; since {args.since}" if args.since else "")
             + f" &middot; generated {dt.datetime.now():%Y-%m-%d %H:%M}</div>"]

    for cond in sorted(groups):
        rs = sorted(groups[cond], key=lambda r: r.get("wall_time", 0))
        stages = list(rs[0]["stages"].keys())
        n = len(rs)
        parts.append(f"<h2>{html.escape(cond)}</h2>")
        # summary
        row = "".join(
            f"<td>{sum(1 for r in rs if r['stages'].get(s))}/{n}<br>"
            f"<span class=meta>{wilson(sum(1 for r in rs if r['stages'].get(s)), n)[0]:.0%}-"
            f"{wilson(sum(1 for r in rs if r['stages'].get(s)), n)[1]:.0%}</span></td>"
            for s in stages)
        wrong = sum(1 for r in rs if r.get("wrong_piece"))
        corr = sum(1 for r in rs if r.get("corrected"))
        parts.append(
            "<table class=sum><tr><th>n</th>" + "".join(f"<th>{s}</th>" for s in stages)
            + "<th>corrected</th><th>wrong piece</th></tr>"
            f"<tr><td>{n}</td>{row}<td>{corr}</td><td>{wrong}</td></tr></table>")
        # cards
        parts.append("<div class=cards>")
        for r in rs:
            vid = r.get("video")
            rel = os.path.relpath(vid, out.parent) if vid else None
            st = "".join(
                f"<span class='st{' ok' if r['stages'].get(s) else ''}'>{s}</span>"
                for s in stages)
            flags = ("<span class=flag>corrected</span>" if r.get("corrected") else "") + \
                    ("<span class=flag>wrong piece</span>" if r.get("wrong_piece") else "")
            a = r.get("auto") or {}
            bits = []
            for k in sorted(a):
                if k.startswith("final_dist_") and k.endswith("_px"):
                    bits.append(f"{k[11:-3]} {a[k]:.0f} px")
            if a.get("obj_displacement_px") is not None:
                bits.append(f"moved {a['obj_displacement_px']:.0f} px")
            if a.get("t_first_close") is not None:
                bits.append(f"closed @ {a['t_first_close']:.1f}s")
            if r.get("achieved_hz"):
                bits.append(f"{r['achieved_hz']:.0f} Hz")
            ended = r.get("ended")
            parts.append(
                "<div class=card>"
                f"<div class=hdr><span class=ep>episode {r['episode']}</span>"
                f"<span class=meta>{r.get('seconds', 0):.0f}s &middot; {r.get('ticks', '?')} ticks"
                + (f" &middot; <span class=ended>ended: {ended}</span>" if ended and ended != 'time' else "")
                + "</span></div>"
                + (f"<video controls preload=metadata src='{html.escape(rel)}'></video>" if rel else
                   "<div class=meta>(no video)</div>")
                + f"<div class=stages>{st}{flags}</div>"
                + (f"<div class=auto>{' &middot; '.join(bits)}</div>" if bits else "")
                + (f"<div class=note>{html.escape(r['note'])}</div>" if r.get("note") else "")
                + "</div>")
        parts.append("</div>")

    out.write_text("\n".join(parts))
    print(f"wrote {out}   ({len(recs)} episodes)")
    print(f"open with:  xdg-open {out}")


if __name__ == "__main__":
    main()
