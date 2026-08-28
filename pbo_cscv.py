"""
Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric
Cross-Validation (CSCV), following Bailey, Borwein, Lopez de Prado & Zhu (2016).

Trials: N=5 cash-allocation ratios (10/20/30/40/50%), other parameters held at
FIXED_CONVENTION (15% threshold, 52-week lookback, 20% cap, full lump-sum
deployment). Weekly excess-return series (NAV return minus the market's own
weekly mean cash rate) over the design period (2009-2018), split into S=16
contiguous blocks. All C(16,8)=12,870 IS/OOS combinatorial splits are
enumerated; for each split the IS-Sharpe-maximizing trial is selected and its
OOS rank among the N trials is checked -- PBO is the fraction of splits where
that selected trial's OOS performance falls at or below the cross-sectional
median.
"""
import itertools
import math
import numpy as np

from backtest import (
    load_markets, slice_period, rolling_high, simulate,
    FIXED_CONVENTION, GRID_ALLOC, DESIGN_START, DESIGN_END,
)

S_BLOCKS = 16
IS_BLOCKS = S_BLOCKS // 2
N_TRIALS_ALLOC = [0.10, 0.20, 0.30, 0.40, 0.50]


def _design_excess_returns(prices, dates, weekly_contrib, cash_rate, alloc):
    """Run simulate() on the full series, then restrict the weekly excess
    return series (nav_ret - weekly mean cash rate) to the design period."""
    p = dict(FIXED_CONVENTION)
    p["alloc"] = alloc
    high = rolling_high(prices, p["lookback"])
    m = simulate(prices, weekly_contrib, cash_rate, p["threshold"], p["lookback"],
                 p["cap"], p["deploy"], p["alloc"], high=high)
    nav_ret = m["nav_ret"]
    weekly_mar = m["mean_cash_rate"] / 52.0
    excess = nav_ret - weekly_mar
    # nav_ret[i] corresponds to the transition into week i+1, i.e. dates[1:]
    ret_dates = dates[1:]
    mask = (ret_dates >= np.datetime64(DESIGN_START)) & (ret_dates <= np.datetime64(DESIGN_END))
    return excess[mask]


def _block_stats(returns, s_blocks=S_BLOCKS):
    """Split `returns` into s_blocks contiguous (nearly equal) blocks; return
    per-block (sum, sumsq, downside_sumsq, n) arrays. `returns` is already the
    excess-of-cash-rate series, so downside_sumsq is the squared shortfall
    below zero (matching the Sortino MAR convention used throughout)."""
    n = len(returns)
    edges = np.linspace(0, n, s_blocks + 1).astype(int)
    sums = np.empty(s_blocks)
    sumsqs = np.empty(s_blocks)
    downside_sumsqs = np.empty(s_blocks)
    counts = np.empty(s_blocks)
    for b in range(s_blocks):
        seg = returns[edges[b]:edges[b + 1]]
        sums[b] = seg.sum()
        sumsqs[b] = (seg ** 2).sum()
        downside_sumsqs[b] = (np.where(seg < 0.0, seg ** 2, 0.0)).sum()
        counts[b] = len(seg)
    return sums, sumsqs, downside_sumsqs, counts


def compute_pbo(market_data, market_name, criterion="sharpe", verbose=True, return_omegas=False):
    prices = market_data["prices"]
    dates = market_data["dates"]
    cash_rate = market_data["cash_rate"]
    weekly_contrib = market_data["weekly_contrib"]

    trial_returns = [
        _design_excess_returns(prices, dates, weekly_contrib, cash_rate, alloc)
        for alloc in N_TRIALS_ALLOC
    ]
    lengths = {len(r) for r in trial_returns}
    assert len(lengths) == 1, f"design-period length mismatch across trials: {lengths}"
    n_trials = len(N_TRIALS_ALLOC)

    # per-trial, per-block (sum, sumsq, downside_sumsq, n)
    block_sum = np.empty((n_trials, S_BLOCKS))
    block_sumsq = np.empty((n_trials, S_BLOCKS))
    block_downside_sumsq = np.empty((n_trials, S_BLOCKS))
    block_n = None
    for i, r in enumerate(trial_returns):
        s, sq, dsq, n = _block_stats(r)
        block_sum[i] = s
        block_sumsq[i] = sq
        block_downside_sumsq[i] = dsq
        if block_n is None:
            block_n = n

    def score(sum_, sumsq_, downside_sumsq_, n_):
        mean_ = sum_ / n_
        if criterion == "sortino":
            downside_var = np.maximum(downside_sumsq_ / n_, 0.0)
            denom = np.sqrt(downside_var)
        else:
            var_ = np.maximum(sumsq_ / n_ - mean_ ** 2, 0.0)
            denom = np.sqrt(var_)
        return np.where(denom > 0, mean_ / denom, -np.inf)

    all_blocks = set(range(S_BLOCKS))
    total_splits = 0
    overfit_splits = 0
    omegas = []

    for is_blocks in itertools.combinations(range(S_BLOCKS), IS_BLOCKS):
        is_idx = list(is_blocks)
        oos_idx = list(all_blocks - set(is_blocks))

        is_n = block_n[is_idx].sum()
        is_sum = block_sum[:, is_idx].sum(axis=1)
        is_sumsq = block_sumsq[:, is_idx].sum(axis=1)
        is_dsq = block_downside_sumsq[:, is_idx].sum(axis=1)
        is_score = score(is_sum, is_sumsq, is_dsq, is_n)

        oos_n = block_n[oos_idx].sum()
        oos_sum = block_sum[:, oos_idx].sum(axis=1)
        oos_sumsq = block_sumsq[:, oos_idx].sum(axis=1)
        oos_dsq = block_downside_sumsq[:, oos_idx].sum(axis=1)
        oos_score = score(oos_sum, oos_sumsq, oos_dsq, oos_n)

        n_star = int(np.argmax(is_score))

        # rank of the IS-winner's OOS score among all N trials (1 = worst)
        rank = int((oos_score <= oos_score[n_star]).sum())
        omega = rank / (n_trials + 1)

        total_splits += 1
        if omega <= 0.5:
            overfit_splits += 1
        if return_omegas:
            omegas.append(omega)

    pbo = overfit_splits / total_splits
    if verbose:
        print(f"{market_name} [{criterion}]: PBO = {pbo:.1%}  ({overfit_splits}/{total_splits} splits, "
              f"design-period weeks={len(trial_returns[0])})")
    if return_omegas:
        return pbo, omegas
    return pbo


if __name__ == "__main__":
    markets, common_start, common_end = load_markets()
    print(f"Common date range: {common_start.date()} to {common_end.date()}\n")

    for crit in ("sharpe", "sortino"):
        print(f"--- criterion = {crit} ---")
        results = {}
        for name, data in markets.items():
            results[name] = compute_pbo(data, name, criterion=crit)
        print(f"  Taiwan vs published 47.9%; United States vs published 94.9%\n")
