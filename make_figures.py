"""
Generate all 7 figures for the base paper, extended from 2 to 3 market panels
(Taiwan, United States, Australia; New Zealand excluded per user decision).
"""
import json
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import backtest as bt
from pbo_cscv import compute_pbo

D = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(D, "figures")
os.makedirs(FIGDIR, exist_ok=True)

with open(os.path.join(D, "results", "base_paper_results.json")) as f:
    RES = json.load(f)

MARKETS = ["Taiwan", "United States", "Australia"]
FIXED = bt.FIXED_CONVENTION
DCA = dict(threshold=0.15, lookback=52, cap=0.20, deploy=1.00, alloc=0.0)

markets, common_start, common_end = bt.load_markets(MARKETS)


def run(name, params, period="full"):
    cfg = markets[name]
    dates, prices, cash_rate, contrib = cfg["dates"], cfg["prices"], cfg["cash_rate"], cfg["weekly_contrib"]
    if period == "design":
        p = bt.slice_period(dates, prices, bt.DESIGN_START, bt.DESIGN_END)
        r = bt.slice_period(dates, cash_rate, bt.DESIGN_START, bt.DESIGN_END)
        d = bt.slice_period(dates, dates, bt.DESIGN_START, bt.DESIGN_END)
    else:
        p = bt.slice_period(dates, prices, bt.DESIGN_START, bt.OOS_END)
        r = bt.slice_period(dates, cash_rate, bt.DESIGN_START, bt.OOS_END)
        d = bt.slice_period(dates, dates, bt.DESIGN_START, bt.OOS_END)
    high = bt.rolling_high(p, params["lookback"])
    m = bt.simulate(p, contrib, r, params["threshold"], params["lookback"], params["cap"],
                     params["deploy"], params["alloc"], high=high)
    return d, p, high, m


# ---------------------------------------------------------------------------
# Figure 1: mechanism demo (Taiwan, design period) -- price vs trailing high
# band, trigger events marked.
# ---------------------------------------------------------------------------
def fig1_mechanism():
    name = "Taiwan"
    dates, prices, high, m = run(name, FIXED, period="design")
    band = high * (1.0 - FIXED["threshold"])

    fig, ax = plt.subplots(figsize=(9, 3.6))
    t = dates.astype("datetime64[D]")
    ax.plot(t, prices, color="#1f77b4", lw=1.1, label="0050.TW weekly price")
    ax.plot(t, high, color="#888888", lw=0.8, ls="--", label=f"{FIXED['lookback']}-week trailing high")
    ax.plot(t, band, color="#d62728", lw=0.8, ls=":",
            label=f"trigger band (\u2212{FIXED['threshold']*100:.0f}% of high)")
    ax.fill_between(t, prices, band, where=(prices <= band), color="#d62728", alpha=0.15,
                     label="price at/below trigger band")
    ax.set_title("Figure 1. Tactical cash-reserve trigger mechanism (Taiwan, 2009\u20132018 design period)",
                 fontsize=10)
    ax.set_ylabel("Price index (NT$)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig1_mechanism.pdf"), dpi=200)
    plt.close(fig)
    print("fig1 done")


# ---------------------------------------------------------------------------
# Figure 2: equity-curve (NAV) growth, DCA vs Tactical, 2x2 market panels
# ---------------------------------------------------------------------------
def fig2_equity_curves():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, name in zip(axes.flat, MARKETS):
        dates_d, prices_d, _, m_dca = run(name, DCA, period="full")
        dates_t, prices_t, _, m_tac = run(name, FIXED, period="full")
        # reconstruct nav series by re-simulating with a small local copy of simulate's nav calc
        cfg = markets[name]
        d, p, r, c = cfg["dates"], cfg["prices"], cfg["cash_rate"], cfg["weekly_contrib"]
        fp = bt.slice_period(d, p, bt.DESIGN_START, bt.OOS_END)
        fr = bt.slice_period(d, r, bt.DESIGN_START, bt.OOS_END)
        fd = bt.slice_period(d, d, bt.DESIGN_START, bt.OOS_END)

        def nav_series(params):
            n = len(fp)
            high = bt.rolling_high(fp, params["lookback"])
            rate_arr = np.asarray(fr, dtype=float)
            weekly_cash = rate_arr / 52.0
            shares = 0.0; reserve = 0.0; cum = 0.0; armed = True
            pv = np.empty(n); cum_c = np.empty(n)
            for i in range(n):
                price = fp[i]
                reserve *= (1.0 + weekly_cash[i])
                cc = c
                cum += cc
                target = params["alloc"] * cc
                cap_level = params["cap"] * cum
                room = max(0.0, cap_level - reserve)
                radd = min(target, room)
                invest_now = cc - radd
                reserve += radd
                if invest_now > 0:
                    shares += invest_now * (1.0 - bt.TXN_COST) / price
                band = high[i] * (1.0 - params["threshold"])
                if price <= band and armed and reserve > 0:
                    dep = reserve * params["deploy"]
                    shares += dep * (1.0 - bt.TXN_COST) / price
                    reserve -= dep
                    armed = False
                elif price > band:
                    armed = True
                pv[i] = shares * price + reserve
                cum_c[i] = cum
            return pv / cum_c

        nav_dca = nav_series(DCA)
        nav_tac = nav_series(FIXED)
        t = fd.astype("datetime64[D]")
        ax.plot(t, nav_dca, color="#1f77b4", lw=1.1, label="100% DCA")
        ax.plot(t, nav_tac, color="#d62728", lw=1.1, label="Tactical (20% alloc.)")
        ax.axvline(np.datetime64(bt.OOS_START), color="#888888", lw=0.7, ls="--")
        ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=7)
        if ax is axes.flat[0]:
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Figure 2. NAV growth (portfolio value / cumulative contributions), "
                  "DCA vs. tactical cash-reserve strategy", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGDIR, "fig2_equity_curves.pdf"), dpi=200)
    plt.close(fig)
    print("fig2 done")


# ---------------------------------------------------------------------------
# Figure 3: drawdown / underwater chart, DCA vs Tactical, 2x2 panels
# ---------------------------------------------------------------------------
def fig3_drawdown():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, name in zip(axes.flat, MARKETS):
        cfg = markets[name]
        d, p, r, c = cfg["dates"], cfg["prices"], cfg["cash_rate"], cfg["weekly_contrib"]
        fp = bt.slice_period(d, p, bt.DESIGN_START, bt.OOS_END)
        fr = bt.slice_period(d, r, bt.DESIGN_START, bt.OOS_END)
        fd = bt.slice_period(d, d, bt.DESIGN_START, bt.OOS_END)

        def dd_series(params):
            n = len(fp)
            high = bt.rolling_high(fp, params["lookback"])
            rate_arr = np.asarray(fr, dtype=float)
            weekly_cash = rate_arr / 52.0
            shares = 0.0; reserve = 0.0; cum = 0.0; armed = True
            pv = np.empty(n); cum_c = np.empty(n)
            for i in range(n):
                price = fp[i]
                reserve *= (1.0 + weekly_cash[i])
                cc = c
                cum += cc
                target = params["alloc"] * cc
                cap_level = params["cap"] * cum
                room = max(0.0, cap_level - reserve)
                radd = min(target, room)
                invest_now = cc - radd
                reserve += radd
                if invest_now > 0:
                    shares += invest_now * (1.0 - bt.TXN_COST) / price
                band = high[i] * (1.0 - params["threshold"])
                if price <= band and armed and reserve > 0:
                    dep = reserve * params["deploy"]
                    shares += dep * (1.0 - bt.TXN_COST) / price
                    reserve -= dep
                    armed = False
                elif price > band:
                    armed = True
                pv[i] = shares * price + reserve
                cum_c[i] = cum
            nav = pv / cum_c
            running_max = np.maximum.accumulate(nav)
            return nav / running_max - 1.0

        dd_dca = dd_series(DCA)
        dd_tac = dd_series(FIXED)
        t = fd.astype("datetime64[D]")
        ax.fill_between(t, dd_dca * 100, 0, color="#1f77b4", alpha=0.35, label="100% DCA")
        ax.fill_between(t, dd_tac * 100, 0, color="#d62728", alpha=0.35, label="Tactical (20% alloc.)")
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("Drawdown (%)", fontsize=8)
        ax.tick_params(labelsize=7)
        if ax is axes.flat[0]:
            ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("Figure 3. NAV drawdown (underwater chart), DCA vs. tactical cash-reserve strategy",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGDIR, "fig3_drawdown.pdf"), dpi=200)
    plt.close(fig)
    print("fig3 done")


# ---------------------------------------------------------------------------
# Figure 4: sensitivity curves -- Sortino vs allocation ratio, one line per market
# ---------------------------------------------------------------------------
def fig4_sensitivity():
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = {"Taiwan": "#1f77b4", "United States": "#d62728", "Australia": "#2ca02c", "New Zealand": "#9467bd"}
    for name in MARKETS:
        sweep = RES["markets"][name]["t3_sensitivity_sweep"]
        allocs = sorted(float(a) for a in sweep.keys())
        sortinos = [sweep[[k for k in sweep.keys() if abs(float(k) - a) < 1e-9][0]]["sortino"] for a in allocs]
        ax.plot([a * 100 for a in allocs], sortinos, marker="o", color=colors[name], label=name)
    ax.set_xlabel("Cash-allocation ratio (%)", fontsize=9)
    ax.set_ylabel("Full-sample Sortino ratio", fontsize=9)
    ax.set_title("Figure 4. Sensitivity of the Sortino ratio to the cash-allocation ratio", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig4_sensitivity.pdf"), dpi=200)
    plt.close(fig)
    print("fig4 done")


# ---------------------------------------------------------------------------
# Figure 5: parameter-leverage tornado, one subplot per market
# ---------------------------------------------------------------------------
def fig5_tornado():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, name in zip(axes.flat, MARKETS):
        lev = RES["markets"][name]["t8_leverage"]
        items = sorted(lev.items(), key=lambda kv: kv[1]["swing"])
        labels = [k for k, _ in items]
        swings = [v["swing"] for _, v in items]
        ax.barh(labels, swings, color="#4c72b0")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Design-period Sortino swing (max \u2212 min)", fontsize=8)
        ax.tick_params(labelsize=8)
    fig.suptitle("Figure 5. One-at-a-time parameter-leverage diagnostic "
                  "(swing in design-period Sortino when varying one parameter, "
                  "others held at the fixed convention)", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIGDIR, "fig5_tornado.pdf"), dpi=200)
    plt.close(fig)
    print("fig5 done")


# ---------------------------------------------------------------------------
# Figure 6: cross-market comparison -- grouped bars, CAGR and Sortino
# ---------------------------------------------------------------------------
def fig6_cross_market():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(MARKETS))
    w = 0.35

    cagr_dca = [RES["markets"][m]["t2_main_comparison"]["dca"]["cagr"] * 100 for m in MARKETS]
    cagr_tac = [RES["markets"][m]["t2_main_comparison"]["tactical"]["cagr"] * 100 for m in MARKETS]
    sortino_dca = [RES["markets"][m]["t2_main_comparison"]["dca"]["sortino"] for m in MARKETS]
    sortino_tac = [RES["markets"][m]["t2_main_comparison"]["tactical"]["sortino"] for m in MARKETS]

    ax = axes[0]
    ax.bar(x - w / 2, cagr_dca, w, label="100% DCA", color="#1f77b4")
    ax.bar(x + w / 2, cagr_tac, w, label="Tactical (20%)", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(MARKETS, fontsize=8, rotation=15)
    ax.set_ylabel("CAGR (%)", fontsize=9)
    ax.set_title("CAGR", fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.bar(x - w / 2, sortino_dca, w, label="100% DCA", color="#1f77b4")
    ax.bar(x + w / 2, sortino_tac, w, label="Tactical (20%)", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(MARKETS, fontsize=8, rotation=15)
    ax.set_ylabel("Sortino ratio", fontsize=9)
    ax.set_title("Sortino ratio", fontsize=10)
    ax.legend(fontsize=7)

    fig.suptitle("Figure 6. Cross-market comparison, DCA vs. tactical cash-reserve strategy", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(FIGDIR, "fig6_cross_market.pdf"), dpi=200)
    plt.close(fig)
    print("fig6 done")


# ---------------------------------------------------------------------------
# Figure 7: PBO -- distribution of the CSCV logit statistic, one subplot per market
# ---------------------------------------------------------------------------
def fig7_pbo():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, name in zip(axes.flat, MARKETS):
        cfg = markets[name]
        pbo, omegas = compute_pbo(cfg, name, criterion="sharpe", verbose=False, return_omegas=True)
        omegas = np.array(omegas)
        eps = 1e-6
        omegas_c = np.clip(omegas, eps, 1 - eps)
        logits = np.log(omegas_c / (1 - omegas_c))
        ax.hist(logits, bins=40, color="#4c72b0", alpha=0.85)
        ax.axvline(0, color="#d62728", lw=1.2, ls="--")
        ax.set_title(f"{name}  (PBO = {pbo:.1%})", fontsize=10)
        ax.set_xlabel("logit(\u03c9)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Figure 7. Distribution of the CSCV logit statistic across all "
                 "C(16,8)=12,870 IS/OOS splits (mass left of the dashed line = overfit splits)",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIGDIR, "fig7_pbo.pdf"), dpi=200)
    plt.close(fig)
    print("fig7 done")


if __name__ == "__main__":
    fig1_mechanism()
    fig2_equity_curves()
    fig3_drawdown()
    fig4_sensitivity()
    fig5_tornado()
    fig6_cross_market()
    fig7_pbo()
    print("All figures saved to", FIGDIR)
