# Which Backtested Parameters Can You Trust? A Leverage-Overfitting Framework for Cash-Reserve Strategies

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22136986.svg)](https://doi.org/10.5281/zenodo.22136986)

Data and code supporting the paper by Jerome Chih-Lung Chou (Graduate Institute of
Finance, National Taiwan University of Science and Technology).

The paper backtests a rule-based tactical cash-reserve strategy against 100% dollar-cost
averaging (DCA) in three equity markets (Taiwan, the United States, and Australia), and
introduces a framework that crosses a parameter-leverage diagnostic against three
independent overfitting checks — the Probability of Backtest Overfitting (PBO) via
Combinatorially Symmetric Cross-Validation, the Deflated Sharpe Ratio (DSR), and White's
Reality Check across the full joint five-parameter grid — to classify each of the
strategy's five parameters by how much it matters and how much its backtested optimum
can be trusted.

This repository contains everything needed to reproduce every table and figure in the
paper from the original real-market data.

## Repository contents

```
real_data/     Daily price series and cash-reserve interest-rate series (see Data sources below)
backtest.py            Core simulation engine (strategy mechanics, CAGR/MDD/Sortino, market loader)
pbo_cscv.py             CSCV/PBO for the cash-allocation ratio specifically (Table 4's "Allocation ratio" row; Figure 5's logit distributions)
pbo_all_params.py       CSCV/PBO computed separately for all five parameters (Table 4)
dsr.py                  Deflated Sharpe Ratio, best-of-five allocation-ratio trial (Table 5)
reality_check.py        White's Reality Check across the full 7,500-combination grid (Table 6)
bootstrap_ci.py         Paired stationary-bootstrap confidence intervals for the
                        headline tactical-vs-DCA comparison (Supporting Information Table S6)
cost_sensitivity.py     Tactical-vs-DCA differences at assumed transaction costs from
                        0% to 0.50% per purchase (Supporting Information Table S7)
base_paper_4market.py   Main comparison, sensitivity sweep, design/OOS split, threshold robustness, parameter-leverage diagnostic (Tables 2, 3, 7, 8)
make_figures.py         Figures 1-7
make_fig8_leverage_pbo.py  Figure 8 (leverage-vs-PBO scatter)
figures/                Pre-generated output figures (PDF)
results/                Pre-generated output tables (JSON), so the paper's numbers can be checked without a full rerun
requirements.txt
LICENSE
```

## Reproducing the paper's results

```bash
pip install -r requirements.txt
python base_paper_4market.py      # Tables 2, 3, 7, 8 -> results/base_paper_results.json
python pbo_all_params.py          # Table 4                -> results/pbo_by_param.json
python dsr.py                     # Table 5                -> results/dsr_results.json
python bootstrap_ci.py            # SI Table S6            -> printed to stdout (~3 minutes, 2,000 paired replications per market)
python cost_sensitivity.py        # SI Table S7            -> printed to stdout (~1 minute)
python reality_check.py           # Table 6                -> results/reality_check_results.json (slow: ~10 minutes, simulates 7,500 combinations x 3 markets and runs a 1,000-replication stationary bootstrap on each)
python make_figures.py            # Figures 1-7            -> figures/
python make_fig8_leverage_pbo.py  # Figure 8                -> figures/
```

`base_paper_4market.py` must be run before `make_figures.py` and `make_fig8_leverage_pbo.py`,
since both read `results/base_paper_results.json`. All scripts use paths relative to their
own location, so the repository can be run from anywhere without editing any paths.

## Data sources

All price and interest-rate data are real, historical, and publicly available. No
synthetic or simulated data are used anywhere in the paper.

| File | Series | Source |
|---|---|---|
| `real_data/0050_total_return_daily.csv` | 0050.TW (Yuanta Taiwan Top 50 ETF), daily dividend-adjusted closing price | Yahoo Finance |
| `real_data/spy_total_return_daily.csv` | SPY (SPDR S&P 500 ETF Trust), daily dividend-adjusted closing price | Yahoo Finance |
| `real_data/stw_total_return_daily.csv` | STW.AX (SPDR S&P/ASX 200 ETF), daily dividend-adjusted closing price | Yahoo Finance |
| `real_data/twd_savings_rate.csv` | Bank of Taiwan NTD savings deposit rate | Bank of Taiwan |
| `real_data/usd_savings_rate.csv` | FDIC National Rate on Savings Deposits | FDIC / FRED (Federal Reserve Bank of St. Louis) |
| `real_data/aud_cash_rate.csv` | RBA Table F4.1 (Retail Deposit and Investment Rates), online savings accounts, $10,000 balance tier | Reserve Bank of Australia, mirrored via [DBnomics](https://db.nomics.world/RBA/F4) (series `FRDIRSAO10K`) |

Each price series is a real, historical daily series; the 0050.TW series contains one
isolated single-day price artifact (2 January 2014, an uncaptured corporate action) that
is corrected in `backtest.py` (`_clean_price_series`), as documented in the paper's
Section 3.1. No other artifacts were found or corrected.

## License

Code and data are released under the [Creative Commons Attribution 4.0 International
License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). See `LICENSE`.

## Citation

If you use this repository, please cite both the paper and the archive:

> Chou, J. C.-L. (2026). *Which Backtested Parameters Can You Trust? A Leverage-Overfitting
> Framework for Cash-Reserve Strategies*. Working paper, under review.

> Chou, J. C.-L. (2026). *Data and code for "Which Backtested Parameters Can You Trust? A
> Leverage-Overfitting Framework for Cash-Reserve Strategies"* [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.22136986
