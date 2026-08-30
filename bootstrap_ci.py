"""Paired stationary-bootstrap confidence intervals for tactical(20%) minus DCA.

The manuscript's Table 1 reports the headline comparison as point estimates only:
no standard errors, no confidence intervals, no test that the difference differs
from zero. The Reality Check tests the best combination in the 7,500-cell grid and
the Deflated Sharpe Ratio tests the best of five allocation-ratio trials, so neither
speaks to the fixed-convention 20% configuration the paper actually recommends.

Method. Weekly log returns are resampled with the Politis-Romano (1994) stationary
bootstrap, using the same mean block length the paper's Reality Check uses
(L = round(sqrt(T))). A price path is rebuilt from each resampled return sequence,
and BOTH strategies are re-simulated on that same path, so each replication yields a
paired difference and path-level noise cancels. The cash-rate series is resampled
with the identical index array, keeping each week's rate matched to its return.

Reported per metric: the observed difference, the 95% percentile interval, and the
share of replications whose difference carries the observed sign (a two-sided
bootstrap p-value is 2 * min(share, 1 - share)).
"""
import math
import os
import sys

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
os.chdir(D)

import backtest as bt
from reality_check import stationary_bootstrap_index

MARKETS = ["Taiwan", "United States", "Australia"]
FIXED = bt.FIXED_CONVENTION
DCA = dict(threshold=0.15, lookback=52, cap=0.20, deploy=1.00, alloc=0.0)
B = 2000
SEED = 20260830

# Sign we expect for a genuine improvement, per metric.
METRICS = [
    ("cagr",    "CAGR",              -1),   # tactical is expected to cost a little
    ("sortino", "Sortino ratio",     +1),
    ("mdd",     "Max drawdown",      +1),   # less negative => higher => +1
    ("p5",      "5th pct wkly ret",  +1),
]


def run_pair(prices, contrib, rate):
    """Simulate DCA and tactical(20%) on one path; return the metric dict of each."""
    hi_t = bt.rolling_high(prices, FIXED["lookback"])
    hi_d = bt.rolling_high(prices, DCA["lookback"])
    tac = bt.simulate(prices, contrib, rate, FIXED["threshold"], FIXED["lookback"],
                      FIXED["cap"], FIXED["deploy"], FIXED["alloc"], high=hi_t)
    dca = bt.simulate(prices, contrib, rate, DCA["threshold"], DCA["lookback"],
                      DCA["cap"], DCA["deploy"], DCA["alloc"], high=hi_d)
    return tac, dca


def main():
    markets, _, _ = bt.load_markets(MARKETS)
    rng_master = np.random.default_rng(SEED)

    print("Paired stationary-bootstrap CIs, tactical(20%%) - DCA, full sample, B=%d\n" % B)

    for name in MARKETS:
        cfg = markets[name]
        prices = np.asarray(cfg["prices"], dtype=float)
        rate = np.asarray(cfg["cash_rate"], dtype=float)
        contrib = cfg["weekly_contrib"]
        T = len(prices)

        tac0, dca0 = run_pair(prices, contrib, rate)
        obs = {k: tac0[k] - dca0[k] for k, _, _ in METRICS}

        logret = np.log(prices[1:] / prices[:-1])
        L = max(2, int(round(math.sqrt(T - 1))))
        rng = np.random.default_rng(rng_master.integers(0, 2 ** 32 - 1))

        draws = {k: np.empty(B) for k, _, _ in METRICS}
        for b in range(B):
            idx = stationary_bootstrap_index(T - 1, L, rng)
            p = np.empty(T)
            p[0] = prices[0]
            p[1:] = prices[0] * np.exp(np.cumsum(logret[idx]))
            r = np.empty(T)
            r[0] = rate[0]
            r[1:] = rate[1:][idx]
            tac, dca = run_pair(p, contrib, r)
            for k, _, _ in METRICS:
                draws[k][b] = tac[k] - dca[k]

        print("%s   (T=%d weeks, mean block L=%d)" % (name, T, L))
        print("  %-18s %10s %10s  %-22s %8s %8s %8s"
              % ("metric", "observed", "boot mean", "95% CI", "P(sign)", "p-value", "pctl(obs)"))
        for k, label, want in METRICS:
            d = draws[k]
            lo, hi = np.percentile(d, [2.5, 97.5])
            share = float(np.mean(d > 0)) if want > 0 else float(np.mean(d < 0))
            pval = 2.0 * min(share, 1.0 - share)
            # Where the actually-observed difference falls in the resampled
            # distribution. Far out in the tail means block resampling has not
            # reproduced the historical path feature the strategy trades on.
            pctl = float(np.mean(d < obs[k])) * 100.0
            scale = 100.0 if k in ("cagr", "mdd", "p5") else 1.0
            unit = "pp" if scale == 100.0 else ""
            print("  %-18s %9.4f%-2s %9.4f%-2s [%8.4f, %8.4f]%-2s %7.1f%% %8.3f %8.1f%%"
                  % (label, obs[k] * scale, unit, d.mean() * scale, unit,
                     lo * scale, hi * scale, unit, share * 100.0, pval, pctl))
        print()


main()
