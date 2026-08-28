"""
White's (2000) Reality Check, applied to the FULL joint 5-parameter grid
(7,500 combinations: threshold x lookback x cap x deploy x alloc), to rule
out the concern that a favorable-looking combination -- this paper's own
fixed-convention combination or any other -- could look good purely because
a very large number of combinations were implicitly available to search,
even though this paper never actually selects its headline parameters by
searching this grid (Section 3.2 fixes them by documented convention).

Two separate Reality Check tests are run per market, because the paper's own
central claim (Table 2) is explicitly NOT about raw return -- it is that the
tactical strategy trades a small return cost for better downside protection:

  (a) Raw NAV-return differential vs. DCA, the classical White/Sullivan-
      Timmermann-White convention. This is a sanity check, not the test that
      matters for this paper's claim: the null (no combination beats DCA on
      raw return) is expected to hold given the strategy's own trade-off.
  (b) Downside-protection differential vs. DCA: the same squared-shortfall-
      below-cash-rate quantity that the Sortino ratio's downside deviation is
      built from, DCA's downside minus combination k's downside each week
      (positive = combination k protected better that week). This directly
      operationalizes the paper's actual claim and is the test that matters.

For each market and each test:
  1. Simulate DCA (the benchmark) and all 7,500 grid combinations over the
     design period (2009-2018), storing each combination's weekly NAV-return
     series (test (b) reuses the same simulated series, no re-simulation).
  2. Observed statistic: V = max_k [ sqrt(T) * mean_t(d_{k,t}) ].
  3. Approximate the null distribution of V via the stationary bootstrap
     (Politis & Romano, 1994): B replications, each resampling the *time
     index* (shared across all 7,500 combinations, preserving their
     cross-sectional correlation) with geometrically distributed block
     lengths (mean block length L = sqrt(T)); each replication's bootstrap
     statistic is centered on the ORIGINAL sample mean per White's (2000)
     procedure.
  4. Reality Check p-value = fraction of bootstrap replications whose
     statistic is >= the observed V.
"""
import time
import numpy as np

from backtest import (
    load_markets, slice_period, rolling_high, simulate,
    FIXED_CONVENTION, GRID_THRESHOLD, GRID_LOOKBACK, GRID_CAP, GRID_DEPLOY, GRID_ALLOC,
    DESIGN_START, DESIGN_END,
)

B_REPS = 1000
RNG_SEED = 12345


def full_grid_nav_ret_matrix(prices, weekly_contrib, cash_rate):
    """Returns (D, dca_ret, mar): D is an (N_combos, T) array of weekly NAV
    returns for every grid combination (design period only); dca_ret is the
    (T,) DCA weekly NAV-return series (alloc=0, same for every combo); mar
    is the (T,) weekly mean-annual-cash-rate/52 series (identical across
    every combo and DCA, since it depends only on the market's own cash-rate
    series, not on strategy parameters)."""
    dca_high = rolling_high(prices, FIXED_CONVENTION["lookback"])
    m_dca = simulate(prices, weekly_contrib, cash_rate, FIXED_CONVENTION["threshold"],
                      FIXED_CONVENTION["lookback"], FIXED_CONVENTION["cap"],
                      FIXED_CONVENTION["deploy"], 0.0, high=dca_high)
    dca_ret = m_dca["nav_ret"]
    T = len(dca_ret)
    mar = np.full(T, m_dca["mean_cash_rate"] / 52.0)

    rows = []
    high_cache = {}
    for lookback in GRID_LOOKBACK:
        high_cache[lookback] = rolling_high(prices, lookback)

    for threshold in GRID_THRESHOLD:
        for lookback in GRID_LOOKBACK:
            high = high_cache[lookback]
            for cap in GRID_CAP:
                for deploy in GRID_DEPLOY:
                    for alloc in GRID_ALLOC:
                        m = simulate(prices, weekly_contrib, cash_rate, threshold, lookback,
                                     cap, deploy, alloc, high=high)
                        rows.append(m["nav_ret"])

    D = np.asarray(rows, dtype=np.float64)  # (N, T)
    return D, dca_ret, mar


def stationary_bootstrap_index(T, mean_block_len, rng):
    """Politis & Romano (1994) stationary bootstrap: returns a length-T
    array of indices into [0, T) with geometrically distributed block
    lengths (mean = mean_block_len), wrapping circularly."""
    p = 1.0 / mean_block_len
    idx = np.empty(T, dtype=np.int64)
    cur = rng.integers(0, T)
    for t in range(T):
        if t > 0 and rng.random() < p:
            cur = rng.integers(0, T)
        idx[t] = cur
        cur = (cur + 1) % T
    return idx


def _run_rc(d, b_reps, rng_seed, label, market_name, verbose):
    n_combos, T = d.shape
    dbar = d.mean(axis=1)
    V_obs = np.sqrt(T) * dbar.max()
    best_idx = int(np.argmax(dbar))

    rng = np.random.default_rng(rng_seed)
    L = max(2, int(round(np.sqrt(T))))

    boot_stats = np.empty(b_reps)
    for b in range(b_reps):
        idx = stationary_bootstrap_index(T, L, rng)
        dbar_boot = d[:, idx].mean(axis=1)
        boot_stats[b] = np.sqrt(T) * (dbar_boot - dbar).max()

    pval = float((boot_stats >= V_obs).mean())
    if verbose:
        print(f"{market_name} [{label}]: N={n_combos} T={T} best_idx={best_idx} "
              f"mean_diff={dbar[best_idx]:.6f} V_obs={V_obs:.4f} RC p-value={pval:.3f}")
    return dict(n_combos=n_combos, T=T, best_idx=best_idx, dbar_best=float(dbar[best_idx]),
                V_obs=float(V_obs), pval=pval)


def reality_check_both(prices, weekly_contrib, cash_rate, market_name, b_reps=B_REPS, verbose=True):
    t0 = time.time()
    D, dca_ret, mar = full_grid_nav_ret_matrix(prices, weekly_contrib, cash_rate)

    # (a) raw return differential vs DCA
    d_return = D - dca_ret[None, :]
    res_return = _run_rc(d_return, b_reps, RNG_SEED, "return", market_name, verbose)

    # (b) downside-protection differential vs DCA (Sortino-consistent)
    diff_D = D - mar[None, :]
    downside_sq_D = np.where(diff_D < 0.0, diff_D ** 2, 0.0)
    diff_dca = dca_ret - mar
    downside_sq_dca = np.where(diff_dca < 0.0, diff_dca ** 2, 0.0)
    d_downside = downside_sq_dca[None, :] - downside_sq_D  # positive = combo protects better
    res_downside = _run_rc(d_downside, b_reps, RNG_SEED + 1, "downside", market_name, verbose)

    if verbose:
        print(f"{market_name}: total time {time.time()-t0:.1f}s\n")
    return dict(return_test=res_return, downside_test=res_downside)


if __name__ == "__main__":
    PAPER_MARKETS = ["Taiwan", "United States", "Australia"]
    markets, common_start, common_end = load_markets(PAPER_MARKETS)
    print(f"Common date range: {common_start.date()} to {common_end.date()}\n")

    results = {}
    for name, data in markets.items():
        prices = data["prices"]
        dates = data["dates"]
        cash_rate = data["cash_rate"]
        weekly_contrib = data["weekly_contrib"]
        design_p = slice_period(dates, prices, DESIGN_START, DESIGN_END)
        design_r = slice_period(dates, cash_rate, DESIGN_START, DESIGN_END)
        results[name] = reality_check_both(design_p, weekly_contrib, design_r, name)

    import json
    import os
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "reality_check_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved {out_path}")
