# GTA VI launch-day Twitch CCU forecast

Entry for make.fun bounty `gta6-twitch-ccu-forecast-01`: forecast, for each of
hours 2–24 of GTA VI's release day (2026-11-19 UTC), five quantiles (5/25/50/
75/95%) of the Twitch "Grand Theft Auto VI" category's concurrent viewers.

## How it works

`main.py` (Python 3.13, standard library only) reads one case file
(`<id>.in`) and writes `<id>.out` with 23 rows of quantiles.

For every past big launch in the case's `comparables`, it forms the
log-ratio of each hour's viewers to that launch's hour-1 print,
`r_h = ln(v_h / v_1)`, and regresses `r_h` on `ln(v_1)` per hour. Launches
that open huge barely rise later; launches that open small (rolling
midnight, US still asleep in hour 1) climb several-fold into the US
evening. Two pools are fitted and blended 50/50 in log space:

- all comparables, empirical residual quantiles, spread x0.8;
- console-only single-player launches (the closest structural match to a
  PS5/Xbox single-player GTA VI), normal residual spread, x0.85.

## Validation (practice launches, scored against SullyGnome hourly data)

| case | loss | 90% coverage |
|---|---|---|
| Marvel's Wolverine (2026-09-15) | 0.198 | 0.91 |
| Red Dead Redemption 2 (2018-10-26) | 0.192 | 0.96 |
| God of War Ragnarök (2022-11-09) | 0.146 | 1.00 |
| Marvel's Spider-Man 2 (2023-10-20) | 0.182 | 0.87 |
| **mean** | **0.179** | — |

House baseline on the same cases: **0.387**. Lower pinball loss is better.

## Run it

```bash
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
