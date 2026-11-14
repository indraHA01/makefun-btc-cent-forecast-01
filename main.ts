// btc-cent-forecast-01: adaptive OLS forecaster for Bitcoin's price 240 minutes ahead.
//
// Reads one case (4,320 one-minute closes of BTC/USDC, 72 hours up to a checkpoint)
// and writes one guess: the price expected 240 minutes after the checkpoint. The score
// is the mean absolute error in cents, so the best guess is the honest conditional
// median of the price 4 hours out.
//
// Method:
//   1. Features, all from the case's own closes: log returns over 60 / 240 / 720 /
//      1440 minutes, and log gaps of the last close to its 6-hour and 24-hour means.
//      Short-window trend extrapolation hurts (tested: it blows up the error), but
//      the 12-hour return carries momentum and the gap to the 6-hour mean carries
//      mean reversion, and together they predict the next 4 hours.
//   2. An OLS regression of the 240-minute forward log return on those six features,
//      fitted on the case's own history: every 60th minute from minute 1440 to
//      minute 4079, each with its realised 240-minute return. The fit is redone for
//      every case, so it adapts to the regime the case itself shows.
//   3. The fit is ridge-regularised toward a prior: coefficients fitted offline by
//      the same OLS on the practice recording (2026-11-08 to 2026-11-10, 10-minute
//      samples). The prior keeps the in-case fit stable when the case's own data is
//      noisy; the case's data still moves the coefficients where it has a clear say.
//   4. The predicted log return is shrunk by 0.7 toward zero and clipped to +/-2%
//      per 4 hours. Shrinkage is what makes the model generalise: full-strength
//      predictions win on one day and fail badly on the next; shrunk ones win on
//      every day tested.
//   5. Fallbacks, in order: the ridge fit; a pull of 0.4 toward the 6-hour mean;
//      the last close (persistence). The program never crashes on a well-formed
//      case and always writes a valid output file.
//
// Honest local validation (walk-forward on the practice recording, checkpoints the
// model never saw while fitting; the prior for validation was fitted on 11-08+11-09
// only, the shipped prior on 11-08..11-10):
//   persistence baseline:  52,041 cents MAE (17 hourly checkpoints, 11-10)
//   this model:             36,153 cents MAE (-31%) on the same checkpoints
//                           35,252 cents on 33 half-hourly checkpoints (-30%)
//                           26,518 cents on the 5 scored practice cases (-44%)
//   The practice baseline itself scores 44,522 cents on its 6 cases, so this model
//   beats it by a wide margin on the practice. The final is scored on unseen prices,
//   so no practice number guarantees the final; the margins above are the evidence.
//
// The program is deterministic: pure arithmetic on the input file, no randomness, no
// clock, no network. Two runs on the same case write the same bytes. It runs in well
// under a second per case (limit 8 s), standard library only.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The smallest positive price on the grid: one unit of its 5 decimals. */
const MIN_PRICE = 0.00001;
/** The largest grid price below the scorer's exclusive 10,000,000-dollar cap. */
const MAX_PRICE = 9_999_999.99999;

/**
 * Prior coefficients [const, r60, r240, r720, r1440, gap360, gap1440], fitted offline
 * by ordinary least squares on the practice recording (2026-11-08T06:00Z to
 * 2026-11-10T22:00Z, samples every 10 minutes): each row is one minute's six features
 * and its realised 240-minute log return. The constant is the window's average drift
 * (-0.46% per 4 h, a down-trending window); the 12-hour return's +0.54 is momentum and
 * the 6-hour gap's -0.61 is mean reversion toward the recent mean.
 */
const PRIOR: number[] = [-0.004636, 0.149122, 0.002969, 0.535995, -0.408426, -0.60516, -0.358288];

/** Ridge strength toward the prior, in units of the mean diagonal of X'X. */
const KAPPA = 2.0;
/** Shrinkage of the predicted 240-minute log return toward zero. */
const SHRINK = 0.7;
/** Largest |predicted log return| allowed per 4 hours. */
const CLIP = 0.02;
/** Minutes between regression samples inside the case. */
const STEP = 60;
/** Minutes of history a regression sample needs before it (24 h, the longest feature). */
const NEED_BEFORE = 1440;
/** Minutes a regression sample needs after it (the 240-minute target). */
const NEED_AFTER = 240;
/** Pull toward the 6-hour mean in the fallback forecast. */
const MR_BETA = 0.4;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  markets: CaseMarket[];
}

/** x kept on the grid's 5 decimals and inside the scorer's (0, 10,000,000). */
const price5 = (x: number): number => Math.round(Math.min(MAX_PRICE, Math.max(MIN_PRICE, x)) * 1e5) / 1e5;

/** Solve A x = v by Gauss elimination with partial pivoting; null when singular. */
function solve(A: number[][], v: number[]): number[] | null {
  const k = v.length;
  const m: number[][] = A.map((row, r) => [...row, v[r] as number]);
  for (let c = 0; c < k; c++) {
    let p = c;
    for (let r = c + 1; r < k; r++) {
      if (Math.abs(m[r]?.[c] as number) > Math.abs(m[p]?.[c] as number)) p = r;
    }
    if (Math.abs(m[p]?.[c] as number) < 1e-14) return null;
    if (p !== c) {
      const tmp = m[c] as number[];
      m[c] = m[p] as number[];
      m[p] = tmp;
    }
    for (let r = 0; r < k; r++) {
      if (r === c) continue;
      const f = (m[r]?.[c] as number) / (m[c]?.[c] as number);
      for (let cc = c; cc <= k; cc++) {
        (m[r] as number[])[cc] = (m[r]?.[cc] as number) - f * (m[c]?.[cc] as number);
      }
    }
  }
  const out: number[] = [];
  for (let r = 0; r < k; r++) {
    const d = m[r]?.[r] as number;
    if (!Number.isFinite(d) || Math.abs(d) < 1e-14) return null;
    out.push((m[r]?.[k] as number) / d);
  }
  return out;
}

/**
 * Ridge regression of y on X, pulled toward `prior` with strength `kappa` (in units of
 * the mean diagonal of X'X, constant column excluded). Returns null when singular.
 */
function fitRidge(X: number[][], y: number[], prior: number[], kappa: number): number[] | null {
  const k = X[0]?.length as number;
  const a: number[][] = Array.from({ length: k }, () => new Array<number>(k).fill(0));
  const v: number[] = new Array<number>(k).fill(0);
  for (let s = 0; s < X.length; s++) {
    const row = X[s] as number[];
    for (let i = 0; i < k; i++) {
      v[i] = (v[i] as number) + (row[i] as number) * (y[s] as number);
      for (let j = 0; j < k; j++) {
        (a[i] as number[])[j] = ((a[i] as number[])[j] as number) + (row[i] as number) * (row[j] as number);
      }
    }
  }
  if (kappa > 0) {
    let scale = 0;
    for (let i = 1; i < k; i++) scale += (a[i]?.[i] as number);
    scale /= k - 1;
    const lam = kappa * scale;
    for (let i = 0; i < k; i++) {
      (a[i] as number[])[i] = ((a[i] as number[])[i] as number) + lam;
      v[i] = (v[i] as number) + lam * (prior[i] as number);
    }
  }
  return solve(a, v);
}

/** The 240-minute-ahead forecast for one case's closes. */
function forecast(prices: number[]): number {
  const n = prices.length;
  if (n === 0) throw new Error('market has no prices');
  // A bad tick (zero, negative, or non-finite price) would poison every log
  // return that touches it. Replace any such tick with the nearest good price;
  // on well-formed data this changes nothing.
  let firstGood = NaN;
  for (let i = 0; i < n; i++) {
    const p = prices[i] as number;
    if (typeof p === 'number' && Number.isFinite(p) && p > 0) { firstGood = p; break; }
  }
  if (!Number.isFinite(firstGood)) throw new Error('market has no usable prices');
  let lastGood = firstGood;
  for (let i = 0; i < n; i++) {
    const p = prices[i] as number;
    if (typeof p === 'number' && Number.isFinite(p) && p > 0) {
      lastGood = p;
    } else {
      prices[i] = lastGood;
    }
  }
  const last = prices[n - 1] as number;
  if (typeof last !== 'number' || !Number.isFinite(last) || last <= 0) throw new Error("market's last price is not a finite positive number");

  // Too little history to fit: fall back to the 6-hour mean pull, then persistence.
  const meanOf = (from: number, to: number): number => {
    let s = 0;
    for (let i = from; i <= to; i++) s += prices[i] as number;
    return s / (to - from + 1);
  };
  if (n < NEED_BEFORE + NEED_AFTER + 1) {
    if (n >= 360) return price5(last + MR_BETA * (meanOf(n - 360, n - 1) - last));
    return price5(last);
  }

  const logP: number[] = prices.map((p) => Math.log(p));
  const prefix: number[] = [0];
  for (const p of prices) prefix.push((prefix[prefix.length - 1] as number) + p);

  /** Six features at minute j: log returns and log gaps to the 6 h / 24 h means. */
  const features = (j: number): number[] => {
    const ma360 = ((prefix[j + 1] as number) - (prefix[j - 359] as number)) / 360;
    const ma1440 = ((prefix[j + 1] as number) - (prefix[j - 1439] as number)) / 1440;
    const pj = prices[j] as number;
    return [
      (logP[j] as number) - (logP[j - 60] as number),
      (logP[j] as number) - (logP[j - 240] as number),
      (logP[j] as number) - (logP[j - 720] as number),
      (logP[j] as number) - (logP[j - 1440] as number),
      Math.log(pj / ma360),
      Math.log(pj / ma1440),
    ];
  };

  // Regression samples: every STEP-th minute with a full 24 h of history before it
  // and a realised 240-minute return after it, all inside this case only.
  const x: number[][] = [];
  const y: number[] = [];
  for (let j = NEED_BEFORE; j <= n - 1 - NEED_AFTER; j += STEP) {
    x.push([1, ...features(j)]);
    y.push((logP[j + NEED_AFTER] as number) - (logP[j] as number));
  }
  const beta = x.length >= 8 ? fitRidge(x, y, PRIOR, KAPPA) : null;
  if (beta === null || beta.some((b) => !Number.isFinite(b))) {
    if (n >= 360) return price5(last + MR_BETA * (meanOf(n - 360, n - 1) - last));
    return price5(last);
  }
  const f = features(n - 1);
  let lr = beta[0] as number;
  for (let k = 0; k < 6; k++) lr += (beta[k + 1] as number) * (f[k] as number);
  lr = Math.max(-CLIP, Math.min(CLIP, SHRINK * lr));
  const guess = price5(last * Math.exp(lr));
  // A non-finite guess (bad input data) must never reach the output file: fall back.
  if (!Number.isFinite(guess)) {
    if (n >= 360) return price5(last + MR_BETA * (meanOf(n - 360, n - 1) - last));
    return price5(last);
  }
  return guess;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (!Array.isArray(c.markets) || c.markets.length !== 1) throw new Error(`${file} does not hold exactly one market`);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts: [{ case: id, price: forecast((c.markets[0] as CaseMarket).prices) }] })}\n`);
}
