"""How much does the 0.15% transaction-cost assumption actually matter?

Both strategies are buy-only and both eventually invest every contributed dollar,
so the cost base is nearly identical across the two arms and most of the charge
cancels in the difference. This quantifies that: the full comparison is re-run
at cost levels from zero to 0.50% per purchase, and what is reported is the
tactical-minus-DCA difference in each metric at each level.
"""
import os
import sys

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
os.chdir(D)

import backtest as bt

MARKETS = ['Taiwan', 'United States', 'Australia']
FIXED = bt.FIXED_CONVENTION
DCA = dict(threshold=0.15, lookback=52, cap=0.20, deploy=1.00, alloc=0.0)
LEVELS = [0.0000, 0.0005, 0.0015, 0.0030, 0.0050]

markets, _, _ = bt.load_markets(MARKETS)


def run(cfg, params):
    hi = bt.rolling_high(cfg['prices'], params['lookback'])
    return bt.simulate(cfg['prices'], cfg['weekly_contrib'], cfg['cash_rate'],
                       params['threshold'], params['lookback'], params['cap'],
                       params['deploy'], params['alloc'], high=hi)


print('Tactical(20%) minus DCA, full sample, by assumed cost per purchase\n')
for name in MARKETS:
    cfg = markets[name]
    print('%s' % name)
    print('  %-9s %10s %10s %10s %10s   | %10s %10s'
          % ('cost', 'd CAGR', 'd MDD', 'd Sortino', 'd p5', 'DCA CAGR', 'Tac CAGR'))
    for c in LEVELS:
        bt.TXN_COST = c
        t = run(cfg, FIXED)
        d = run(cfg, DCA)
        print('  %-9s %9.4fpp %9.4fpp %10.4f %9.4fpp   | %9.2f%% %9.2f%%'
              % ('%.2f%%' % (c * 100),
                 (t['cagr'] - d['cagr']) * 100,
                 (t['mdd'] - d['mdd']) * 100,
                 t['sortino'] - d['sortino'],
                 (t['p5'] - d['p5']) * 100,
                 d['cagr'] * 100, t['cagr'] * 100))
    print()

bt.TXN_COST = 0.0015
