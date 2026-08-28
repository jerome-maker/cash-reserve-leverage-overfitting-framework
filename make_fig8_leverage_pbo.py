"""
Figure 8: leverage (design-period Sortino swing) vs. PBO (%) for all five
strategy parameters, one panel per market. Horizontal line at PBO=50% (the
no-overfitting benchmark: below this line, the IS winner outperforms OOS more
often than not); vertical line at the market's own median leverage across
its five parameters, splitting them into a higher- and lower-leverage half.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(D, "results")
FIGDIR = os.path.join(D, "figures")
os.makedirs(FIGDIR, exist_ok=True)

with open(os.path.join(RESULTS_DIR, "base_paper_results.json")) as f:
    RES = json.load(f)
with open(os.path.join(RESULTS_DIR, "pbo_by_param.json")) as f:
    PBO = json.load(f)

MARKETS = ["Taiwan", "United States", "Australia"]
PARAM_LABEL = {"threshold": "threshold", "lookback": "lookback", "cap": "cap",
               "deploy": "deploy", "alloc": "alloc"}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

for ax, name in zip(axes, MARKETS):
    lev = {k: v["swing"] for k, v in RES["markets"][name]["t8_leverage"].items()}
    pbo = PBO[name]
    params = list(lev.keys())
    xs = np.array([lev[p] for p in params])
    ys = np.array([pbo[p] * 100 for p in params])
    med_x = np.median(xs)

    ax.axhline(50, color="#888888", lw=0.9, ls="--", zorder=1)
    ax.axvline(med_x, color="#888888", lw=0.9, ls="--", zorder=1)
    ax.scatter(xs, ys, s=90, color="#c0392b", zorder=3, edgecolor="white", linewidth=0.8)
    for p, x, y in zip(params, xs, ys):
        offset = (6, 6)
        ax.annotate(PARAM_LABEL[p], (x, y), textcoords="offset points", xytext=offset, fontsize=9)

    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Leverage (design-period Sortino swing)", fontsize=9)
    if ax is axes[0]:
        ax.set_ylabel("PBO (%)", fontsize=9)
    ax.set_ylim(-5, 105)
    ax.tick_params(labelsize=8)
    xpad = max(xs) * 0.15 if max(xs) > 0 else 0.001
    ax.set_xlim(-xpad * 0.3, max(xs) + xpad)

fig.suptitle("Figure 8. Leverage vs. PBO by parameter, all three markets "
              "(dashed lines: PBO = 50% and each market's median leverage)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(FIGDIR, "fig8_leverage_pbo.pdf"), dpi=200)
plt.close(fig)
print("fig8 done")
