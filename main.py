"""gta6-trophy-rarity-forecast-01: forecast each GTA VI PS5 trophy's Exophase
earn rate at the case's target, as five quantiles (5/25/50/75/95%).

Method, in short
---------------
For every trophy, work on the logit scale, where a rate moves roughly
linearly against log-time while players work through the game.  The
house curve baseline fits one straight line per trophy over the last
24 hours; this program improves on it in four ways:

1. Recency-weighted fit.  Every read contributes, weighted by
   exp(-age / 18h), so the launch-night regime (tiny tracked counts,
   hardcore early players, Exophase refresh artefacts) fades and the
   recent regime — the one that will carry into the target week —
   dominates.  The fit is logit(rate) ~ a + b * ln(1 + hours since
   2026-11-19T00:00Z), the same clock the house curve uses.
2. Slope shrinkage.  A trophy whose own series is flat or noisy gets
   its slope pulled toward the median slope of its tier (or of all
   trophies when the tier has too few reliable fits), precision
   weighted, so a two-point wobble cannot launch the forecast.
3. Honest spread.  The 90% band comes from the fit itself: residual
   scatter times a prediction-interval factor that grows as the
   target moves past the data in log-time, plus a random-walk term
   that grows with the horizon.  Jumpier history (Exophase refresh
   steps) means a wider band; a clean trend means a narrower one.
   The house baseline's fixed 0.5*sqrt(days) spread is the fallback
   when a trophy has too little history to fit.
4. Physical guards.  Earners only grow, so a forecast whose implied
   earner count falls below the last read's is lifted to it.  The
   platinum's earners are a subset of every other trophy's, so its
   quantiles are capped by the other trophies' forecasts.  A trophy
   stuck at 0% gets a small positive median and a wide upward band
   (it can only rise); a trophy stuck at 100% gets a tight upward
   band and a wider downward one (it can only fall as new players
   join).  No median moves more than 5 logit units from its last
   rate, which only binds on degenerate fits.

Everything is deterministic: fixed arithmetic, reads processed oldest
first, no randomness, no network, standard library only.  The same
input always writes the same bytes.

Run it like the grader does (python:3.13-slim, `python3 -P -s /bot/main.py`,
PYTHONHASHSEED=0), or locally with:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so an `import`
# of a module shipped beside this file would find it (none is shipped now,
# but the line keeps the layout honest).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release midnight: the clock the house curve baseline uses, and the origin
# of the log-time axis.  Reads before it (launch night starts 23:50 the
# evening before) get a small negative time; ln(1 + t) stays defined.
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Recency weight: exp(-age_hours / TAU).  18 h means a read from 36 h ago
# counts about a seventh of a fresh one; launch night has faded by the
# deadline while the last day and a half dominates.
TAU_HOURS = 18.0
# Random-walk spread, logit units per square-root day of horizon.  The
# house baseline uses 0.5; the trend explains part of the move, so 0.30.
SIGMA_RW_PER_SQRT_DAY = 0.30
# Floors and caps on the total spread (logit units).  The floor keeps a
# flat series from claiming false certainty over a multi-day horizon;
# the cap keeps a wild fit from spraying quantiles across the scale.
SIGMA_FLOOR = 0.35
SIGMA_CAP = 1.70
# A floor on the residual scatter itself, for the same reason.
RESID_FLOOR = 0.20
# No median may move further than this, in logit units, from its last rate.
LOGIT_MOVE_CAP = 5.0
# A trophy needs at least this many usable reads to fit a line at all.
MIN_READS_TO_FIT = 3


def clamp_pct(x: float) -> float:
    """A rate in percent, kept inside the range the logit is defined on."""
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    """Logit back to percent, guarded against overflow either way."""
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it.  A whole number comes back
    as an int, so json.dumps writes 100 where JavaScript writes 100."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_epoch(at: str) -> float:
    return (parse_time(at) - EPOCH).total_seconds() / 3600.0


def x_of_time(at: str) -> float:
    """Log-time axis value for a read time (or the target)."""
    return math.log1p(max(hours_since_epoch(at), -0.9))


def weighted_fit(xs, ys, ws):
    """Weighted least squares of y ~ a + b*x.

    Returns (y_bar, x_bar, slope, se_slope, resid_std, n_eff, sxx) or
    None when the fit is degenerate (no spread in x, or too few points).
    """
    sw = sum(ws)
    if sw <= 0:
        return None
    x_bar = sum(w * x for w, x in zip(ws, xs)) / sw
    y_bar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - x_bar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - x_bar) * (y - y_bar) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    n_eff = sw * sw / sum(w * w for w in ws)
    resid_var = 0.0
    for w, x, y in zip(ws, xs, ys):
        r = y - (y_bar + slope * (x - x_bar))
        resid_var += w * r * r
    dof = max(n_eff - 2.0, 1.0)
    resid_var = resid_var / (sw * dof / max(n_eff, 1.0)) if n_eff > 0 else 0.0
    # Weighted residual variance, unbiased-ish for the effective sample.
    resid_std = math.sqrt(max(resid_var, 0.0))
    se_slope = resid_std / math.sqrt(sxx) if sxx > 0 else float("inf")
    return (y_bar, x_bar, slope, se_slope, resid_std, n_eff, sxx)


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def mad(xs, centre):
    return median([abs(v - centre) for v in xs])


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    history = case["history"]
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = x_of_time(case["target"])
    at_target = case["target"]

    if not history:
        # No reads at all: nothing to anchor on.  Emit a valid, wide,
        # neutral forecast rather than crashing (a crash is an invalid
        # case, which forfeits any prize).
        rows = []
        for trophy in trophies:
            lg = logit(1.0)
            rows.append({"id": trophy["id"], "q": quantiles(lg, 1.2)})
        return {"trophies": rows}

    last = history[-1]
    t_last = hours_since_epoch(last["at"])

    # Extrapolated players-tracked at the target, for the earners guard:
    # a recency-weighted log-time fit of ln(players).  Only its direction
    # matters (it must not fall), so a rough fit is enough.
    n_target = None
    n_pts = [(x_of_time(r["at"]), math.log(max(r["players_tracked"], 1)))
             for r in history if r.get("players_tracked")]
    if len(n_pts) >= 2:
        n_ws = [math.exp(-(t_last - hours_since_epoch(r["at"])) / TAU_HOURS)
                for r in history if r.get("players_tracked")]
        n_fit = weighted_fit([p[0] for p in n_pts], [p[1] for p in n_pts], n_ws)
        if n_fit is not None:
            y_bar, x_bar, slope, _se, _rs, _ne, _sx = n_fit
            n_target = math.exp(y_bar + max(slope, 0.0) * (x_target - x_bar))
            n_target = max(n_target, last["players_tracked"])
    n_last = last.get("players_tracked") or 0

    # First pass: per-trophy fits, collecting slopes for shrinkage.
    fits = []
    for i, trophy in enumerate(trophies):
        series = []
        for read in history:
            rate = read["rates"][i] if i < len(read["rates"]) else None
            if rate is None:
                continue
            t = hours_since_epoch(read["at"])
            w = math.exp(-(t_last - t) / TAU_HOURS)
            series.append((x_of_time(read["at"]), logit(rate), w))
        fits.append({"trophy": trophy, "series": series})

    # Reliable slopes per tier, for the shrinkage prior.
    tier_slopes = {}
    for fit in fits:
        tier = fit["trophy"]["tier"]
        tier_slopes.setdefault(tier, [])
    reliable = []
    for fit in fits:
        series = fit["series"]
        if len(series) < MIN_READS_TO_FIT:
            continue
        xs = [p[0] for p in series]
        ys = [p[1] for p in series]
        ws = [p[2] for p in series]
        wfit = weighted_fit(xs, ys, ws)
        fit["wfit"] = wfit
        if wfit is None:
            continue
        _yb, _xb, slope, se, resid, _ne, _sx = wfit
        if resid < 1.0 and se < 2.0:
            tier_slopes[fit["trophy"]["tier"]].append(slope)
            reliable.append(slope)
    tier_prior = {}
    for tier, slopes in tier_slopes.items():
        use = slopes if len(slopes) >= 4 else reliable
        if not use:
            continue
        centre = median(use)
        spread = max(1.4826 * mad(use, centre), 0.15)
        tier_prior[tier] = (centre, min(spread, 0.8))
    global_prior = None
    if reliable:
        centre = median(reliable)
        spread = max(1.4826 * mad(reliable, centre), 0.15)
        global_prior = (centre, min(spread, 0.8))

    # Second pass: medians and spreads.
    out = []
    for fit in fits:
        trophy = fit["trophy"]
        series = fit["series"]
        # Last non-null rate anchors the level (trophies are in case
        # order, so the trophy's index is its position in each read).
        idx = trophies.index(trophy)
        last_rate = None
        for read in reversed(history):
            rate = read["rates"][idx] if idx < len(read["rates"]) else None
            if rate is not None:
                last_rate = rate
                break
        if last_rate is None:
            last_rate = 0.0
        last_logit = logit(last_rate)

        wfit = fit.get("wfit")
        if wfit is None or len(series) < MIN_READS_TO_FIT:
            # Too little history to fit: persistence with the baseline's
            # spread (the house baseline's answer, honestly widened).
            med = last_logit
            sigma = max(0.5 * math.sqrt(horizon_days), SIGMA_FLOOR)
        else:
            y_bar, x_bar, slope, se_slope, resid_std, n_eff, sxx = wfit
            # Shrink the slope toward the tier (or global) prior.
            prior = tier_prior.get(trophy["tier"], global_prior)
            if prior is not None and math.isfinite(se_slope):
                p_slope, p_se = prior
                w_own = 1.0 / max(se_slope * se_slope, 1e-9)
                w_prior = 1.0 / max(p_se * p_se, 1e-9)
                slope = (slope * w_own + p_slope * w_prior) / (w_own + w_prior)
            # Prediction at the target, pivoting on the weighted mean.
            med = y_bar + slope * (x_target - x_bar)
            # Spread: residual scatter times the prediction-interval
            # factor (grows as the target moves past the data), plus a
            # random-walk term that grows with the horizon.
            resid = max(resid_std, RESID_FLOOR)
            lever = 1.0 + 1.0 / max(n_eff, 1.0) + (x_target - x_bar) ** 2 / sxx
            sigma_fit = resid * math.sqrt(lever)
            sigma_rw = SIGMA_RW_PER_SQRT_DAY * math.sqrt(horizon_days)
            sigma = math.sqrt(sigma_fit * sigma_fit + sigma_rw * sigma_rw)

        # Guards.
        med = min(max(med, last_logit - LOGIT_MOVE_CAP),
                  last_logit + LOGIT_MOVE_CAP)
        # Earners only grow: the forecast cannot imply fewer earners
        # than the last read already counted.
        if n_target is not None and n_last:
            e_last = last_rate / 100.0 * n_last
            floor_rate = 100.0 * e_last / n_target
            med = max(med, logit(floor_rate))
        sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)

        out.append({"trophy": trophy, "median": med, "sigma": sigma,
                    "last_rate": last_rate})

    # Platinum cap: its earners are a subset of every other trophy's,
    # so its rate cannot exceed any other trophy's rate.
    plat = [o for o in out if o["trophy"]["tier"] == "platinum"]
    others = [o for o in out if o["trophy"]["tier"] != "platinum"]
    if plat and others:
        cap_med = min(o["median"] for o in others)
        cap_q95 = min(o["median"] + Z[4] * o["sigma"] for o in others)
        for o in plat:
            o["median"] = min(o["median"], cap_med)
            o["cap_q95"] = cap_q95

    # Zero-rate trophies can only rise: lift the median a touch and
    # widen the band upward.  A trophy at 100% can only fall as new
    # players join: keep the median, widen the band downward.
    rows = []
    for o in out:
        med = o["median"]
        sigma = o["sigma"]
        sig_down = sigma
        sig_up = sigma
        if o["last_rate"] <= 0.0:
            med = max(med, logit(0.02))
            sig_up = max(sig_up, 0.9)
        if o["last_rate"] >= 100.0:
            sig_down = max(sig_down, 1.5)
            sig_up = min(sig_up, 0.15)
        qs = []
        for k, z in enumerate(Z):
            s = sig_down if z < 0 else sig_up
            qs.append(inv_logit(med + z * s))
        # Platinum cap on the top quantile.
        if "cap_q95" in o:
            qs[4] = min(qs[4], inv_logit(o["cap_q95"]))
        # Round, clamp, and enforce non-decreasing order.
        qs = [round4(min(max(q, 0.01), 99.99)) for q in qs]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": o["trophy"]["id"], "q": qs})
    return {"trophies": rows}


def quantiles(median_logit: float, sigma: float):
    """Five quantiles around a median logit with a symmetric spread."""
    qs = [round4(min(max(inv_logit(median_logit + z * sigma), 0.01), 99.99))
          for z in Z]
    for k in range(1, 5):
        if qs[k] < qs[k - 1]:
            qs[k] = qs[k - 1]
    return qs


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
