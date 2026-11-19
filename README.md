# GTA VI Trophy Rarity Forecast — week one

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/biuosa09-create/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/biuosa09-create/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for make.fun bounty [`gta6-trophy-rarity-forecast-01`](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
a program that forecasts, for each of the 55 PlayStation 5 trophies of
Grand Theft Auto VI, the earn rate (percent of tracked players who earned
it) at 2026-11-26T00:00:00Z — a week after launch — as five quantiles
(5%, 25%, 50%, 75%, 95%). It is scored by pinball loss on the logit
scale against the first Exophase read in the target hour.

New to make.fun? Sign up with my referral link: https://make.fun/r/shw861zs (code `shw861zs`).

## What it does

`main.py` (Python 3, standard library only, no network) reads every
`*.in` case in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one
`<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`). Each
output is exactly `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`
with one row per trophy, quantiles in percent, non-decreasing, rounded
to 4 decimals half-up.

Run it:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It is deterministic: the same input always writes byte-identical output
(fixed arithmetic, reads processed oldest first, no randomness).

## Method

The data is Exophase's trophy page for GTA VI on PS5, read on a fixed
clock from launch night on: each read lists `players_tracked` and the
earn rate of every trophy. The program works on the **logit scale**,
because a miss from 1% to 2% costs about what a miss from 33% to 50%
does under the contest's scoring, and because earn rates move roughly
linearly against log-time while players work through the game.

Per trophy, it fits a **recency-weighted least-squares line** of
`logit(rate)` against `ln(1 + hours since 2026-11-19T00:00:00Z)` — the
same clock the house curve baseline uses — with weights
`exp(-age / 18h)`, so the launch-night regime (tiny tracked counts,
hardcore early players, Exophase refresh artefacts) fades and the recent
regime dominates. The line is read at the case's target time.

Three things keep the fit honest:

- **Slope shrinkage.** A trophy whose own series is flat or noisy gets
  its slope pulled toward the median slope of its tier (or of all
  trophies when the tier has too few reliable fits), precision weighted,
  so a two-point wobble cannot launch the forecast. The platinum, which
  has no tier peers, falls back to the global median slope.
- **Honest spread.** The 90% band comes from the fit itself: residual
  scatter times a prediction-interval factor that grows as the target
  moves past the data in log-time, plus a random-walk term of
  0.30 logit units per square-root day of horizon. Jumpier history
  (Exophase refresh steps) means a wider band; a clean trend means a
  narrower one. Total spread is floored at 0.35 and capped at 1.70
  logit units. Where a trophy has too little history to fit, it falls
  back to persistence (median = last rate) with the house baseline's
  spread.
- **Physical guards.** Earners only grow, so a forecast whose implied
  earner count falls below the last read's is lifted to it (using a
  log-time fit of `players_tracked` extrapolated to the target). The
  platinum's earners are a subset of every other trophy's, so its
  quantiles are capped by the other trophies' forecasts. No median
  moves more than 5 logit units from its last rate.

### Edge cases

- **Zero earns so far:** median 0.02% with a wide upward band — it can
  only rise.
- **Already at 100%:** the median stays near 100% with a tight upward
  band and a wider downward one — it can only fall as new players join
  (the practice data supports this: the 100% trophy stayed at 100%
  through a sixfold dilution, 638 → 3,868 tracked players).
- **Flat series / missing reads / too few reads:** persistence with the
  baseline spread; null rates are skipped, history is sorted by time.
- **Empty history:** a valid, wide, neutral forecast instead of a
  crash — a crash is an invalid case, which forfeits any prize.

## Testing

- **Practice cases** (`p00`, `p01` from the kit): valid output, byte-identical
  across two runs, 0.09 s for both cases. Medians equal the house
  baseline's on the flat pre-refresh history, as they should; the bands
  are wider, because the practice answer is the Exophase refresh that
  neither the baseline nor this program can see coming.
- **Synthetic final-like cases** (mechanistic world: players arrive on a
  saturating curve, each player earns a trophy after a trophy-specific
  delay, rates are sampled only at Exophase refreshes; 60 h of history,
  target 4.5 days out): mean pinball loss **0.17–0.28** vs the house
  baseline's **0.81–1.33** across four scenarios (base, fast dilution,
  slow arrivals, frequent refreshes), with 90% coverage 1.00 throughout.
  A concave-quadratic variant was tried and rejected: it over-corrected
  and lost in three of the four scenarios.
- **Edge cases:** empty history, a single read, two reads, all-null rates
  for a trophy, a trophy at 0%, a trophy already at 40% (forecasts above
  a trophy at 2%, as it should), and a truncated practice series — all
  produce valid output; the 100% trophy stays above 90% and the platinum
  cap (platinum ≤ every other trophy) holds.

## Sources

- Trophy earn rates: [Exophase — Grand Theft Auto VI trophies (PSN)](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
- Bounty: [gta6-trophy-rarity-forecast-01 on make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
- The trophy list (55 trophies: 1 Platinum, 2 Gold, 6 Silver, 46 Bronze;
  28 secret) is the one published in the bounty case files.
