"""
Deflated Sharpe Ratio (DSR), a second and complementary overfitting check to
the PBO/CSCV procedure in pbo_cscv.py, following Bailey and Lopez de Prado
(2014), "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
Overfitting, and Non-Normality", Journal of Portfolio Management, 40(5), 94-107.

Where PBO/CSCV asks "is the ranking of trials stable across resamples of the
data?", DSR asks a different question: "given that N=5 allocation-ratio
trials were tried, is the best-observed Sharpe ratio among them still
distinguishable from what pure variance across N noisy trials would produce
by chance, once the number of trials and the non-normality (skew, kurtosis)
of the winning trial's own return distribution are accounted for?" A low DSR
means the best-of-N result cannot be trusted as genuine skill.

Procedure, for each market, on the full 2009-2026 sample:
  1. Compute the weekly excess-return series (NAV return net of the market's
     own weekly mean cash rate) for each of the N=5 cash-allocation-ratio
     trials (10/20/30/40/50%, other parameters at FIXED_CONVENTION).
  2. Compute each trial's (non-annualized, weekly) Sharpe ratio; select the
     trial with the highest Sharpe ratio (SR_hat) -- not necessarily the 20%
     convention used elsewhere in the paper.
  3. Estimate the expected maximum Sharpe ratio across N independent trials
     with the observed cross-sectional variance of the N trial Sharpe ratios
     (the benchmark SR* against which SR_hat is judged):
        E[max SR] = sqrt(V[SR_n]) * ((1-gamma)*Phi^-1(1-1/N) + gamma*Phi^-1(1-1/(N*e)))
     where gamma is the Euler-Mascheroni constant.
  4. Compute the Probabilistic Sharpe Ratio of SR_hat relative to SR*, using
     the winning trial's own sample skewness and kurtosis to correct for
     non-normality:
        DSR = Phi( (SR_hat - SR*) * sqrt(T-1) / sqrt(1 - skew*SR_hat + ((kurt-1)/4)*SR_hat^2) )
"""
import math
import numpy as np
from scipy.stats import norm

import backtest as bt

EULER_GAMMA = 0.5772156649015329
N_TRIALS_ALLOC = [0.10, 0.20, 0.30, 0.40, 0.50]


def _trial_excess_returns(prices, dates, weekly_contrib, cash_rate, alloc):
    p = dict(bt.FIXED_CONVENTION)
    p["alloc"] = alloc
    high = bt.rolling_high(prices, p["lookback"])
    m = bt.simulate(prices, weekly_contrib, cash_rate, p["threshold"], p["lookback"],
                     p["cap"], p["deploy"], p["alloc"], high=high)
    nav_ret = m["nav_ret"]
    weekly_mar = m["mean_cash_rate"] / 52.0
    return nav_ret - weekly_mar


def expected_max_sharpe(sharpe_values):
    """E[max SR] across N i.i.d.-approximated trials, Bailey & Lopez de Prado (2014) eq. 6-8."""
    n = len(sharpe_values)
    v = np.var(sharpe_values, ddof=1)
    if v <= 0 or n < 2:
        return max(sharpe_values)
    term1 = (1 - EULER_GAMMA) * norm.ppf(1 - 1.0 / n)
    term2 = EULER_GAMMA * norm.ppf(1 - 1.0 / (n * math.e))
    return math.sqrt(v) * (term1 + term2)


def probabilistic_sharpe_ratio(sr_hat, sr_benchmark, t, skew, kurt):
    """PSR(SR*): probability the true Sharpe ratio exceeds sr_benchmark, given
    an observed SR of sr_hat over T observations with sample skew/kurtosis
    (kurt is RAW, not excess -- normal returns have kurt=3)."""
    denom = math.sqrt(max(1e-12, 1 - skew * sr_hat + ((kurt - 1) / 4.0) * sr_hat ** 2))
    z = (sr_hat - sr_benchmark) * math.sqrt(t - 1) / denom
    return norm.cdf(z)


def compute_dsr(market_data, market_name, verbose=True):
    prices = market_data["prices"]
    dates = market_data["dates"]
    cash_rate = market_data["cash_rate"]
    weekly_contrib = market_data["weekly_contrib"]

    full_p = bt.slice_period(dates, prices, bt.DESIGN_START, bt.OOS_END)
    full_r = bt.slice_period(dates, cash_rate, bt.DESIGN_START, bt.OOS_END)
    full_contrib_dates = bt.slice_period(dates, dates, bt.DESIGN_START, bt.OOS_END)

    trial_returns = [
        _trial_excess_returns(full_p, full_contrib_dates, weekly_contrib, full_r, a)
        for a in N_TRIALS_ALLOC
    ]
    sharpe_values = np.array([r.mean() / r.std(ddof=1) for r in trial_returns])

    best_idx = int(np.argmax(sharpe_values))
    best_alloc = N_TRIALS_ALLOC[best_idx]
    sr_hat = sharpe_values[best_idx]
    best_returns = trial_returns[best_idx]
    t = len(best_returns)

    mean_ = best_returns.mean()
    std_ = best_returns.std(ddof=1)
    skew = ((best_returns - mean_) ** 3).mean() / std_ ** 3
    kurt = ((best_returns - mean_) ** 4).mean() / std_ ** 4  # raw kurtosis

    sr_benchmark = expected_max_sharpe(sharpe_values)
    dsr = probabilistic_sharpe_ratio(sr_hat, sr_benchmark, t, skew, kurt)

    if verbose:
        print(f"{market_name}: best trial={best_alloc*100:.0f}% alloc, weekly SR={sr_hat:.4f} "
              f"(annualized {sr_hat*math.sqrt(52):.3f}), E[max SR|N=5]={sr_benchmark:.4f} "
              f"(annualized {sr_benchmark*math.sqrt(52):.3f}), T={t}, skew={skew:.3f}, kurt={kurt:.3f} "
              f"-> DSR={dsr:.1%}")

    return dict(best_alloc=best_alloc, sr_hat=sr_hat, sr_hat_annualized=sr_hat * math.sqrt(52),
                sr_benchmark=sr_benchmark, sr_benchmark_annualized=sr_benchmark * math.sqrt(52),
                t=t, skew=skew, kurt=kurt, dsr=dsr,
                all_sharpe_weekly=sharpe_values.tolist())


if __name__ == "__main__":
    import json
    import os

    PAPER_MARKETS = ["Taiwan", "United States", "Australia"]
    markets, common_start, common_end = bt.load_markets(PAPER_MARKETS)
    print(f"Common date range: {common_start.date()} to {common_end.date()}\n")
    results = {}
    for name, data in markets.items():
        results[name] = compute_dsr(data, name)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dsr_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nSaved {out_path}")
