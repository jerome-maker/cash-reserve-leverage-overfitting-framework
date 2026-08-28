"""
Full analysis suite for the base (fixed-convention) tactical cash-reserve
paper, three markets (Taiwan, United States, Australia). Produces every
table the manuscript needs (excluding the per-parameter PBO of Table 4,
computed separately by pbo_all_params.py, and the Deflated Sharpe Ratio and
Reality Check tables, computed by dsr.py and reality_check.py):

  t2_main_comparison       DCA vs Tactical(20%) at FIXED_CONVENTION, full sample
  t3_sensitivity_sweep     allocation ratio 10-50%, other params fixed, full sample
  t4_pbo                   Probability of Backtest Overfitting for the allocation
                           ratio specifically (CSCV, S=16, N=5) -- reproduced
                           exactly by the "alloc" row of pbo_all_params.py's output
  t5_design_vs_oos         DCA vs Tactical(20%), design period vs out-of-sample
  t6_threshold_robustness  alternate 10% correction threshold, full sample
  t8_leverage              one-at-a-time parameter-leverage diagnostic (design period)

All results are written to results/base_paper_results.json.
"""
import json
import os
import backtest as bt
from pbo_cscv import compute_pbo

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "base_paper_results.json")

FIXED = bt.FIXED_CONVENTION  # threshold=0.15, lookback=52, cap=0.20, deploy=1.00, alloc=0.20
DCA = dict(threshold=0.15, lookback=52, cap=0.20, deploy=1.00, alloc=0.0)
ALT_THRESHOLD = dict(FIXED, threshold=0.10)
ALT_THRESHOLD_DCA = dict(DCA, threshold=0.10)

SWEEP_ALLOC = [0.10, 0.20, 0.30, 0.40, 0.50]
GRIDS_FOR_LEVERAGE = {
    "threshold": bt.GRID_THRESHOLD,
    "lookback": bt.GRID_LOOKBACK,
    "cap": bt.GRID_CAP,
    "deploy": bt.GRID_DEPLOY,
    "alloc": bt.GRID_ALLOC,
}


def ev(params, prices, contrib, rate):
    high = bt.rolling_high(prices, params["lookback"])
    return bt.simulate(prices, contrib, rate, params["threshold"], params["lookback"],
                        params["cap"], params["deploy"], params["alloc"], high=high)


def strip(m):
    """Drop the large nav_ret array before JSON-serializing a metrics dict."""
    return {k: v for k, v in m.items() if k != "nav_ret"}


PAPER_MARKETS = ["Taiwan", "United States", "Australia"]  # New Zealand excluded per user decision


def main():
    markets, common_start, common_end = bt.load_markets(PAPER_MARKETS)
    print(f"Common date range: {common_start.date()} to {common_end.date()}")

    results = {"common_start": str(common_start.date()), "common_end": str(common_end.date()),
               "markets": {}}

    for name, cfg in markets.items():
        dates, prices, cash_rate, contrib = cfg["dates"], cfg["prices"], cfg["cash_rate"], cfg["weekly_contrib"]

        full_p = bt.slice_period(dates, prices, bt.DESIGN_START, bt.OOS_END)
        full_r = bt.slice_period(dates, cash_rate, bt.DESIGN_START, bt.OOS_END)
        design_p = bt.slice_period(dates, prices, bt.DESIGN_START, bt.DESIGN_END)
        design_r = bt.slice_period(dates, cash_rate, bt.DESIGN_START, bt.DESIGN_END)
        oos_p = bt.slice_period(dates, prices, bt.OOS_START, bt.OOS_END)
        oos_r = bt.slice_period(dates, cash_rate, bt.OOS_START, bt.OOS_END)

        print(f"\n=== {name} === full={len(full_p)}wk design={len(design_p)}wk oos={len(oos_p)}wk "
              f"mean_rate={full_r.mean()*100:.3f}%")

        # T2: main comparison (full sample)
        dca_full = ev(DCA, full_p, contrib, full_r)
        tac_full = ev(FIXED, full_p, contrib, full_r)
        print(f"  T2 DCA:      CAGR={dca_full['cagr']*100:6.2f}% MDD={dca_full['mdd']*100:6.2f}% "
              f"Sortino={dca_full['sortino']:.3f} P5={dca_full['p5']*100:.2f}%")
        print(f"  T2 Tactical: CAGR={tac_full['cagr']*100:6.2f}% MDD={tac_full['mdd']*100:6.2f}% "
              f"Sortino={tac_full['sortino']:.3f} P5={tac_full['p5']*100:.2f}%")

        # T3: sensitivity sweep (full sample)
        sweep = {}
        for a in SWEEP_ALLOC:
            p = dict(FIXED, alloc=a)
            sweep[a] = strip(ev(p, full_p, contrib, full_r))
        print("  T3 sweep Sortino by alloc:", {a: round(sweep[a]["sortino"], 3) for a in SWEEP_ALLOC})

        # T4: PBO / CSCV
        pbo = compute_pbo(cfg, name, criterion="sharpe", verbose=True)

        # T5: design vs OOS
        dca_design = ev(DCA, design_p, contrib, design_r)
        tac_design = ev(FIXED, design_p, contrib, design_r)
        dca_oos = ev(DCA, oos_p, contrib, oos_r)
        tac_oos = ev(FIXED, oos_p, contrib, oos_r)

        # T6: alternate threshold robustness (full sample)
        dca_alt = ev(ALT_THRESHOLD_DCA, full_p, contrib, full_r)
        tac_alt = ev(ALT_THRESHOLD, full_p, contrib, full_r)
        print(f"  T6 alt-threshold(10%) Tactical: CAGR={tac_alt['cagr']*100:6.2f}% "
              f"MDD={tac_alt['mdd']*100:6.2f}% Sortino={tac_alt['sortino']:.3f}")

        # T8: parameter-leverage diagnostic (design period, anchored at FIXED_CONVENTION)
        leverage = {}
        for pname, grid in GRIDS_FOR_LEVERAGE.items():
            sortinos = []
            for val in grid:
                params = dict(FIXED)
                params[pname] = val
                m = ev(params, design_p, contrib, design_r)
                sortinos.append((val, m["sortino"]))
            vals_only = [s for _, s in sortinos]
            swing = max(vals_only) - min(vals_only)
            leverage[pname] = dict(sweep=sortinos, swing=swing)
        ranked = sorted(leverage.items(), key=lambda kv: -kv[1]["swing"])
        print("  T8 leverage ranking:", [(p, round(d["swing"], 4)) for p, d in ranked])

        results["markets"][name] = dict(
            weekly_contrib=contrib,
            mean_cash_rate_full=float(full_r.mean()),
            n_weeks_full=len(full_p), n_weeks_design=len(design_p), n_weeks_oos=len(oos_p),
            t2_main_comparison=dict(dca=strip(dca_full), tactical=strip(tac_full)),
            t3_sensitivity_sweep=sweep,
            t4_pbo=pbo,
            t5_design_vs_oos=dict(
                design=dict(dca=strip(dca_design), tactical=strip(tac_design)),
                oos=dict(dca=strip(dca_oos), tactical=strip(tac_oos)),
            ),
            t6_threshold_robustness=dict(dca=strip(dca_alt), tactical=strip(tac_alt)),
            t8_leverage=leverage,
        )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nSaved to", OUT)


if __name__ == "__main__":
    main()
