"""
Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric
Cross-Validation (CSCV), computed for EACH of the strategy's five parameters
in turn (not just the cash-allocation ratio), following Bailey, Borwein,
Lopez de Prado & Zhu (2016). For each parameter, the N trials are that
parameter's own grid values (Table 1 / GRID_* in backtest.py), with the other
four parameters held at FIXED_CONVENTION -- exactly the same one-at-a-time
convention already used by the parameter-leverage diagnostic, so the two
diagnostics (leverage swing and PBO) are computed on directly comparable
trial sets and can be cross-tabulated.
"""
import itertools
import numpy as np

from backtest import (
    load_markets, rolling_high, simulate, slice_period,
    FIXED_CONVENTION, GRID_THRESHOLD, GRID_LOOKBACK, GRID_CAP, GRID_DEPLOY, GRID_ALLOC,
    DESIGN_START, DESIGN_END,
)

S_BLOCKS = 16
IS_BLOCKS = S_BLOCKS // 2

PARAM_GRIDS = {
    "threshold": GRID_THRESHOLD,
    "lookback": GRID_LOOKBACK,
    "cap": GRID_CAP,
    "deploy": GRID_DEPLOY,
    "alloc": GRID_ALLOC,
}


def _design_excess_returns_for_param(prices, dates, weekly_contrib, cash_rate, param_name, value):
    p = dict(FIXED_CONVENTION)
    p[param_name] = value
    high = rolling_high(prices, p["lookback"])
    m = simulate(prices, weekly_contrib, cash_rate, p["threshold"], p["lookback"],
                 p["cap"], p["deploy"], p["alloc"], high=high)
    nav_ret = m["nav_ret"]
    weekly_mar = m["mean_cash_rate"] / 52.0
    excess = nav_ret - weekly_mar
    ret_dates = dates[1:]
    mask = (ret_dates >= np.datetime64(DESIGN_START)) & (ret_dates <= np.datetime64(DESIGN_END))
    return excess[mask]


def _block_stats(returns, s_blocks=S_BLOCKS):
    n = len(returns)
    edges = np.linspace(0, n, s_blocks + 1).astype(int)
    sums = np.empty(s_blocks)
    sumsqs = np.empty(s_blocks)
    counts = np.empty(s_blocks)
    for b in range(s_blocks):
        seg = returns[edges[b]:edges[b + 1]]
        sums[b] = seg.sum()
        sumsqs[b] = (seg ** 2).sum()
        counts[b] = len(seg)
    return sums, sumsqs, counts


def compute_pbo_for_param(market_data, market_name, param_name, criterion="sharpe", verbose=True):
    prices = market_data["prices"]
    dates = market_data["dates"]
    cash_rate = market_data["cash_rate"]
    weekly_contrib = market_data["weekly_contrib"]

    grid = PARAM_GRIDS[param_name]
    trial_returns = [
        _design_excess_returns_for_param(prices, dates, weekly_contrib, cash_rate, param_name, v)
        for v in grid
    ]
    lengths = {len(r) for r in trial_returns}
    assert len(lengths) == 1, f"design-period length mismatch across trials: {lengths}"
    n_trials = len(grid)

    block_sum = np.empty((n_trials, S_BLOCKS))
    block_sumsq = np.empty((n_trials, S_BLOCKS))
    block_n = None
    for i, r in enumerate(trial_returns):
        s, sq, n = _block_stats(r)
        block_sum[i] = s
        block_sumsq[i] = sq
        if block_n is None:
            block_n = n

    def score(sum_, sumsq_, n_):
        mean_ = sum_ / n_
        var_ = np.maximum(sumsq_ / n_ - mean_ ** 2, 0.0)
        denom = np.sqrt(var_)
        return np.where(denom > 0, mean_ / denom, -np.inf)

    all_blocks = set(range(S_BLOCKS))
    total_splits = 0
    overfit_splits = 0

    for is_blocks in itertools.combinations(range(S_BLOCKS), IS_BLOCKS):
        is_idx = list(is_blocks)
        oos_idx = list(all_blocks - set(is_blocks))

        is_n = block_n[is_idx].sum()
        is_sum = block_sum[:, is_idx].sum(axis=1)
        is_sumsq = block_sumsq[:, is_idx].sum(axis=1)
        is_score = score(is_sum, is_sumsq, is_n)

        oos_n = block_n[oos_idx].sum()
        oos_sum = block_sum[:, oos_idx].sum(axis=1)
        oos_sumsq = block_sumsq[:, oos_idx].sum(axis=1)
        oos_score = score(oos_sum, oos_sumsq, oos_n)

        n_star = int(np.argmax(is_score))
        rank = int((oos_score <= oos_score[n_star]).sum())
        omega = rank / (n_trials + 1)

        total_splits += 1
        if omega <= 0.5:
            overfit_splits += 1

    pbo = overfit_splits / total_splits
    if verbose:
        print(f"{market_name:15s} {param_name:10s} N={n_trials:2d}  PBO={pbo:.1%}")
    return pbo


if __name__ == "__main__":
    import json
    import os

    PAPER_MARKETS = ["Taiwan", "United States", "Australia"]
    markets, common_start, common_end = load_markets(PAPER_MARKETS)
    print(f"Common date range: {common_start.date()} to {common_end.date()}\n")

    results = {}
    for name, data in markets.items():
        results[name] = {}
        for pname in PARAM_GRIDS:
            results[name][pname] = compute_pbo_for_param(data, name, pname)
        print()

    print("=== Summary: PBO by parameter, all markets ===")
    for name in results:
        ranked = sorted(results[name].items(), key=lambda kv: -kv[1])
        print(f"{name}: " + ", ".join(f"{p}={v:.1%}" for p, v in ranked))

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pbo_by_param.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nSaved {out_path}")
