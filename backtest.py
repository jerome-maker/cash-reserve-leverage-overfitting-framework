"""
Grid-search backtest of the tactical cash-reserve strategy on Taiwan (0050.TW)
and the United States (SPY), extending the base study's fixed-parameter design
to a full five-parameter grid searched independently per market.

Data: real daily total-return price series and real historical cash-reserve
rate series (Bank of Taiwan NTD savings deposit rate; FDIC/FRED USD national
savings deposit rate), reused from the base study's own data cache
(`real_data/`) rather than the constant-rate proxy used in an earlier draft
of this analysis. Both markets are trimmed to the common date range shared by
both price series, matching the base study's own comparability convention.

Remaining simplification (documented, not hidden):
  - Contribution cadence: modeled as a level weekly contribution equal to
    the base study's monthly nominal amount converted to a weekly-equivalent
    (monthly x 12 / 52), since the trigger/reserve mechanics are evaluated
    weekly.
"""
import os, math
import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
REAL_DATA_DIR = os.path.join(D, "real_data")

TXN_COST = 0.0015      # 0.15%, identical to base study

WEEKLY_CONTRIB_TW = 10000.0 * 12 / 52   # NT$10,000/month equivalent
WEEKLY_CONTRIB_US = 1000.0 * 12 / 52    # US$1,000/month equivalent
WEEKLY_CONTRIB_AU = 1000.0 * 12 / 52    # A$1,000/month equivalent (AUD is order-of-magnitude close to USD)
WEEKLY_CONTRIB_NZ = 1000.0 * 12 / 52    # NZ$1,000/month equivalent (NZD is order-of-magnitude close to USD)

# Market registry: name -> (price csv, rate csv, weekly contribution)
MARKET_SPECS = {
    "Taiwan": ("0050_total_return_daily.csv", "twd_savings_rate.csv", WEEKLY_CONTRIB_TW),
    "United States": ("spy_total_return_daily.csv", "usd_savings_rate.csv", WEEKLY_CONTRIB_US),
    "Australia": ("stw_total_return_daily.csv", "aud_cash_rate.csv", WEEKLY_CONTRIB_AU),
    "New Zealand": ("fnz_total_return_daily.csv", "nzd_deposit_rate.csv", WEEKLY_CONTRIB_NZ),
}

DESIGN_START, DESIGN_END = "2009-01-01", "2018-12-31"
OOS_START, OOS_END = "2019-01-01", "2026-12-31"

GRID_THRESHOLD = [0.05, 0.10, 0.15, 0.20, 0.25]
GRID_LOOKBACK  = [13, 26, 52, 78, 104]
GRID_CAP       = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
GRID_DEPLOY    = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
GRID_ALLOC     = [0.10, 0.20, 0.30, 0.40, 0.50]

FIXED_CONVENTION = dict(threshold=0.15, lookback=52, cap=0.20, deploy=1.00, alloc=0.20)


def _clean_price_series(price, jump_threshold=0.30):
    """Same isolated-jump correction used in the base study's own exploratory
    grid-search notebook: an isolated single-day return beyond jump_threshold
    (e.g. an unregistered unit split not reflected in adjusted-close metadata)
    is treated as a data artifact and the pre-event segment is backward-scaled
    by the observed jump ratio."""
    price = price.copy()
    ret = price.pct_change()
    jump_dates = ret[ret.abs() > jump_threshold].index
    n_fixed = 0
    for d in jump_dates:
        loc = price.index.get_loc(d)
        ratio = price.iloc[loc] / price.iloc[loc - 1]
        price.iloc[:loc] = price.iloc[:loc] * ratio
        n_fixed += 1
    return price, n_fixed


def _load_price_daily(csv_name):
    price = pd.read_csv(os.path.join(REAL_DATA_DIR, csv_name), index_col=0, parse_dates=True)["price"]
    price, n_fixed = _clean_price_series(price)
    if n_fixed:
        print(f"[load] {csv_name}: corrected {n_fixed} anomalous single-day price jump(s)")
    return price


def load_markets(markets=None):
    """Load each market's real daily price + real cash-rate series, trim ALL of
    them to the date range common across every market requested (matching the
    base study's comparability convention, now generalized from a pairwise trim
    to an N-way trim), and resample to weekly (Friday) frequency. Returns a
    dict: market -> dict(dates, prices, cash_rate) as aligned numpy arrays,
    plus weekly_contrib.

    `markets`: optional list of market names to load (default: all four in
    MARKET_SPECS, i.e. Taiwan, United States, Australia, New Zealand)."""
    names = markets if markets is not None else list(MARKET_SPECS.keys())

    price_daily = {}
    for name in names:
        price_csv, _, _ = MARKET_SPECS[name]
        price_daily[name] = _load_price_daily(price_csv)

    common_start = max(p.index.min() for p in price_daily.values())
    common_end = min(p.index.max() for p in price_daily.values())

    def build_weekly(price_d, rate_csv):
        price_weekly = price_d.loc[common_start:common_end].resample("W-FRI").last().dropna()
        rate_df = pd.read_csv(os.path.join(REAL_DATA_DIR, rate_csv), parse_dates=["date"]).set_index("date").sort_index()
        cash_rate_weekly = rate_df["annual_rate"].reindex(price_weekly.index, method="ffill").bfill()
        return price_weekly, cash_rate_weekly

    out = {}
    for name in names:
        _, rate_csv, weekly_contrib = MARKET_SPECS[name]
        pw, r = build_weekly(price_daily[name], rate_csv)
        out[name] = dict(dates=pw.index.values.astype("datetime64[D]"),
                          prices=pw.values.astype(float),
                          cash_rate=r.values.astype(float),
                          weekly_contrib=weekly_contrib)

    return out, common_start, common_end


def slice_period(dates, arr, start, end):
    mask = (dates >= np.datetime64(start)) & (dates <= np.datetime64(end))
    return arr[mask]


def rolling_high(prices, window):
    n = len(prices)
    out = np.empty(n)
    for i in range(n):
        lo = max(0, i - window + 1)
        out[i] = prices[lo:i + 1].max()
    return out


def simulate(prices, weekly_contrib, cash_rate, threshold, lookback, cap, deploy, alloc, high=None):
    """Simulate one parameter combination over the given price series.
    `cash_rate` may be a scalar (constant annual rate) or a numpy array the
    same length as `prices` (a real time-varying weekly annual rate series).
    Returns dict of metrics. alloc=0 reproduces the 100% DCA control.
    `high` may be a precomputed rolling_high(prices, lookback) array to avoid
    recomputing it on every grid cell that shares the same lookback."""
    n = len(prices)
    if high is None:
        high = rolling_high(prices, lookback)
    rate_arr = np.full(n, cash_rate, dtype=float) if np.isscalar(cash_rate) else np.asarray(cash_rate, dtype=float)
    weekly_cash_rate_arr = rate_arr / 52.0
    mean_cash_rate = rate_arr.mean()

    shares = 0.0
    reserve = 0.0
    cum_contrib = 0.0
    armed = True
    pv = np.empty(n)
    cum_contrib_series = np.empty(n)
    cash_weight = np.empty(n)

    for i in range(n):
        price = prices[i]

        # 1) reserve accrues interest (this week's rate)
        reserve *= (1.0 + weekly_cash_rate_arr[i])

        # 2) split contribution
        c = weekly_contrib
        cum_contrib += c
        target_reserve_add = alloc * c
        cap_level = cap * cum_contrib
        room = max(0.0, cap_level - reserve)
        reserve_add = min(target_reserve_add, room)
        invest_now = c - reserve_add
        reserve += reserve_add

        # 3) invest immediate portion
        if invest_now > 0:
            shares += invest_now * (1.0 - TXN_COST) / price

        # 4) evaluate trigger (using post-update high; high[i] already includes price[i])
        band = high[i] * (1.0 - threshold)
        if price <= band and armed and reserve > 0:
            deploy_amt = reserve * deploy
            shares += deploy_amt * (1.0 - TXN_COST) / price
            reserve -= deploy_amt
            armed = False
        elif price > band:
            armed = True

        cur_pv = shares * price + reserve
        pv[i] = cur_pv
        cum_contrib_series[i] = cum_contrib
        cash_weight[i] = reserve / cur_pv if cur_pv > 0 else 0.0

    # ---- metrics ----
    # CAGR via XIRR on weekly cash flows: -c each week, +final PV at last week
    cashflows = -np.full(n, weekly_contrib)
    cashflows[-1] += pv[-1]
    times = np.arange(n) / 52.0

    def npv(rate):
        return np.sum(cashflows / (1.0 + rate) ** times)

    lo, hi = -0.99, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    cagr = (lo + hi) / 2

    # NAV = portfolio value / cumulative contributions -- normalizes out contribution
    # timing/size, matching the base study's own metric definitions exactly.
    nav = pv / cum_contrib_series

    # MDD on the NAV series
    running_max = np.maximum.accumulate(nav)
    dd = nav / running_max - 1.0
    mdd = dd.min()

    # weekly NAV returns (drop the first, undefined, element)
    nav_ret = nav[1:] / nav[:-1] - 1.0

    # Sortino: downside deviation is the RMS of (return - MAR) over *all* periods
    # (zero-filled where the period was not downside), matching the base study's
    # own Sortino implementation, not a std-dev of the downside subset alone.
    mar_weekly = mean_cash_rate / 52.0
    diff = nav_ret - mar_weekly
    downside_sq = np.where(diff < 0.0, diff ** 2, 0.0)
    downside_dev = math.sqrt(downside_sq.mean()) * math.sqrt(52.0) if len(nav_ret) else float("nan")
    ann_return = nav_ret.mean() * 52.0 if len(nav_ret) else float("nan")
    sortino = (ann_return - mean_cash_rate) / downside_dev if downside_dev > 0 else float("nan")

    # cash drag: avg cash weight * (equity ann. return - cash ann. return)
    equity_ann_return = (prices[-1] / prices[0]) ** (52.0 / n) - 1.0
    cash_drag = cash_weight.mean() * (equity_ann_return - mean_cash_rate)

    # 5th percentile of weekly NAV returns
    p5 = np.percentile(nav_ret, 5) if len(nav_ret) else float("nan")

    return dict(cagr=cagr, mdd=mdd, sortino=sortino, cash_drag=cash_drag,
                p5=p5, avg_cash_weight=cash_weight.mean(), terminal_value=pv[-1],
                nav_ret=nav_ret, mean_cash_rate=mean_cash_rate)


def grid_search(prices, weekly_contrib, cash_rate, criterion="sortino"):
    best = None
    best_params = None
    best_metrics = None
    n_evaluated = 0
    for lookback in GRID_LOOKBACK:
        high = rolling_high(prices, lookback)
        for threshold in GRID_THRESHOLD:
            for cap in GRID_CAP:
                for deploy in GRID_DEPLOY:
                    for alloc in GRID_ALLOC:
                        m = simulate(prices, weekly_contrib, cash_rate, threshold, lookback, cap, deploy, alloc,
                                      high=high)
                        n_evaluated += 1
                        score = m[criterion]
                        if best is None or (score is not None and not math.isnan(score) and score > best):
                            best = score
                            best_params = dict(threshold=threshold, lookback=lookback, cap=cap,
                                                deploy=deploy, alloc=alloc)
                            best_metrics = m
    return best_params, best_metrics, n_evaluated
