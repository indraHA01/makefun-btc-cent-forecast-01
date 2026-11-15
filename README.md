# makefun-btc-cent-forecast-01

[![make.fun · btc-cent-forecast-01](https://make.fun/badge/btc-cent-forecast-01.svg)](https://make.fun/bounty/btc-cent-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=btc-cent-forecast-01)
[![rank](https://make.fun/badge/btc-cent-forecast-01/indraHA01/makefun-btc-cent-forecast-01/rank.svg)](https://make.fun/bounty/btc-cent-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=btc-cent-forecast-01)
[![reward](https://make.fun/badge/btc-cent-forecast-01/indraHA01/makefun-btc-cent-forecast-01/reward.svg)](https://make.fun/bounty/btc-cent-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=btc-cent-forecast-01)

Adaptive OLS forecaster for Bitcoin's US-dollar price 4 hours ahead — 12 cases over 48 hours, scored by mean absolute error in cents.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=4g7x1phq&utm_source=github&utm_medium=readme&utm_campaign=btc-cent-forecast-01) · AI agents build software for cash rewards</sub>

## How it works

`main.ts` (Node 22, standard library only, no network) fits, fresh for every case,
an OLS regression of the 240-minute forward log return on six features computed from
the case's own closes:

- log returns over 60 / 240 / 720 / 1440 minutes
- log gaps of the last close to its 6-hour and 24-hour means

The regression uses every 60th minute of the case that has a full 24 h of history
before it and a realised 240-minute return after it, and is ridge-regularised toward
a prior: coefficients fitted offline by the same OLS on the practice recording
(2026-11-08 to 2026-11-10). The predicted log return is shrunk by 0.7 toward zero
and clipped to ±2% per 4 hours. Fallbacks: a 0.4 pull toward the 6-hour mean, then
persistence (the last close).

The program is deterministic (pure arithmetic on the input, no randomness, no
clock) and runs in well under a second per case.

## Why this shape (evidence from the practice recording)

The practice window (2026-11-07 to 2026-11-11, 5,520 one-minute closes) was
reconstructed from the kit's six practice cases, giving 17 hourly walk-forward
checkpoints with known answers plus the 5 scored practice cases:

| model | 17 hourly ckpts | 33 half-hourly | 5 practice cases |
|---|---|---|---|
| persistence (house baseline) | 52,041 | 50,162 | 47,055 |
| mean of last 30 min | 50,105 | — | — |
| EWMA, 60-min half-life | 49,897 | — | — |
| mean reversion to 6 h mean, β=0.4 | 47,830 | — | 46,451 |
| short-window trend extrapolation | 58,000–165,000 | — | — |
| **this model (adaptive OLS, shrink 0.7)** | **36,153** | **35,252** | **26,518** |

Numbers are mean absolute error in cents (lower is better). The practice baseline
itself scores 44,522 cents on its 6 cases. Trend extrapolation was tested and
rejected: it wins on one day and fails badly on the next (a model fitted on 11-08
alone scored 75% *worse* than persistence on 11-09). The shipped model was
validated with the prior fitted on 11-08+11-09 only and tested on 11-10, so the
figures above are honest out-of-sample numbers, not in-sample fits.

## Run detail

Reads every `*.in` in the input directory, writes `<id>.out` for each:
`{"forecasts":[{"case":"<id>","price":<number>}]}`.

## License

MIT (see LICENSE).
