"""GTA VI launch-day Twitch CCU forecaster (gta6-twitch-ccu-forecast-01).

Forecasts, for each of hours 2..24 of release day (UTC), five quantiles of the
Twitch "Grand Theft Auto VI" category's concurrent viewers, scored by mean
pinball loss on the log scale.

Method (fitted and validated on the kit's four practice launches, where it
scores a mean pinball loss of ~0.18 against the house baseline's ~0.39):

  For every past big launch in the case's `comparables`, take its release-day
  hourly curve and form the log-ratio of each hour's viewers to that launch's
  hour-1 print, r_h = ln(v_h / v_1).  Across launches, r_h is strongly
  related to the size of hour 1: launches that open huge (a global or
  midnight-UTC unlock already live in hour 1) barely rise later, while
  launches that open small (rolling midnight, US still asleep in hour 1)
  climb several-fold into the US evening.  So for each hour h we regress
  r_h on ln(v_1) across the comparables and take the target launch's
  predicted centre from that regression, with the spread from the
  regression's residuals.

  Two pools are fitted and blended 50/50 in log space:
    * all comparables (37 in a practice case, 41 in the final), empirical
      residual quantiles, spread x0.8;
    * console-only single-player launches (PS5/Xbox/Switch story games,
      the closest structural match to a PS5/Xbox single-player GTA VI),
      normal residual spread, x0.85.
  Blending the broad pool (stable) with the narrow pool (closest match)
  beat either alone on every practice case.

The program runs offline on the standard library only. It reads every
`<id>.in` in the input directory and writes `<id>.out` per case:
{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}.

Usage:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a helper
# module shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Blend weight on the broad (all-comparables) pool; the rest goes to the
# console-only pool. Chosen on the practice cases (flat region 0.4-0.5).
W_ALL = 0.5
# Spread multipliers on each pool's residual quantiles.
SPREAD_ALL = 0.8
SPREAD_CONSOLE = 0.85

# Names that mark a launch as multiplayer / live-service rather than a
# single-player story game. Only used inside the console-only pool.
MULTIPLAYER_MARKERS = (
    "Battlefield", "Call of Duty", "Marathon", "Marvel Rivals",
    "HELLDIVERS", "ARC Raiders", "Halo Infinite", "Monster Hunter",
    "Nightreign", "Diablo IV", "Forza Horizon",
)


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def round2(x):
    """Round to 2 decimals, half up (the starter's convention)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def quantile(sorted_vals, q):
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo)


def ols_1d(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, residuals)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx > 0 else 0.0
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    return a, b, resid


def day_hours(comp):
    """{hour: viewers} for the comparable's release day, hours 1..24 UTC."""
    ds = parse_ts(comp["day_start"])
    out = {}
    for row in comp.get("hourly", []):
        v = row.get("viewers")
        if not v or v <= 0:
            continue
        h = int((parse_ts(row["from"]) - ds).total_seconds() / 3600) + 1
        if 1 <= h <= 24:
            out[h] = v
    return out


def is_console_single(comp):
    """Console-only single-player story launch (no PC, no multiplayer)."""
    if "PC" in comp.get("platforms", ""):
        return False
    return not any(m in comp.get("name", "") for m in MULTIPLAYER_MARKERS)


def pool_ratios(case, console_only):
    """[(log v1, {hour: log-ratio})] for comparables in the pool."""
    pool = []
    for comp in sorted(case.get("comparables", []), key=lambda c: c.get("name", "")):
        if console_only and not is_console_single(comp):
            continue
        dh = day_hours(comp)
        if 1 not in dh:
            continue
        v1 = dh[1]
        ratios = {h: math.log(v / v1) for h, v in dh.items() if h >= 2}
        if ratios:
            pool.append((math.log(v1), ratios))
    return pool


def pool_quantiles(pool, log_p1, hour, kind, spread_mult):
    """Five log-ratio quantiles for `hour` from one pool, or None."""
    pts = [(x, r[hour]) for x, r in pool if hour in r]
    if len(pts) < 3:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    a, b, resid = ols_1d(xs, ys)
    centre = a + b * log_p1
    if kind == "emp":
        qs = [centre + spread_mult * quantile(sorted(resid), q) for q in LEVELS]
    else:  # normal spread from the residuals
        mu = sum(resid) / len(resid)
        var = sum((r - mu) ** 2 for r in resid) / max(len(resid) - 1, 1)
        sd = math.sqrt(var)
        qs = [centre + mu + spread_mult * z * sd for z in Z]
    return qs


def baseline_quantiles(p1, hour):
    """House-baseline fallback: persistence with a widening log spread."""
    return [p1 * math.exp(z * 0.15 * math.sqrt(hour - 1)) for z in Z]


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    log_p1 = math.log(p1)

    pool_all = pool_ratios(case, console_only=False)
    pool_console = pool_ratios(case, console_only=True)

    hours = []
    seen = set()
    for row in sorted(case["forecast"], key=lambda r: r["hour"]):
        h = row["hour"]
        if h in seen or h < 2 or h > 24:
            continue
        seen.add(h)
        qa = pool_quantiles(pool_all, log_p1, h, "emp", SPREAD_ALL)
        qb = pool_quantiles(pool_console, log_p1, h, "normal", SPREAD_CONSOLE)
        if qa is None and qb is None:
            qs = baseline_quantiles(p1, h)
        elif qa is None:
            qs = [p1 * math.exp(q) for q in qb]
        elif qb is None:
            qs = [p1 * math.exp(q) for q in qa]
        else:
            qs = [
                p1 * math.exp(W_ALL * qa[i] + (1.0 - W_ALL) * qb[i])
                for i in range(5)
            ]
        # Enforce non-decreasing quantiles and the contest's bounds.
        qs = [min(max(q, 1.0), 1_000_000_000.0) for q in qs]
        for i in range(1, 5):
            qs[i] = max(qs[i], qs[i - 1])
        qs = [round2(q) for q in qs]
        hours.append({"hour": h, "q": qs})
    hours.sort(key=lambda r: r["hour"])
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
