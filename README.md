# Heterogeneous-Panel Estimators in Small Regional Macro Panels

Replication package for the article *Heterogeneous-Panel Estimators in Small Regional Macro Panels: Monte-Carlo Evidence and an Application to Digital Adoption and Income in Latin America* (H. Hernández-Lasso).

The package regenerates every number, table and figure of the article from the raw data and the simulation draws. The manuscript itself is not included in the repository.

## Quick start

```bash
pip install -r requirements.txt          # Python 3.9+ ; a LaTeX distribution is only needed to compile the manuscript
python code/run_all.py                   # data -> panel -> diagnostics -> estimation -> robustness -> Monte-Carlo summaries -> validation (about 5 minutes, uses the cached draws)
python code/run_all.py --download        # also refresh the raw data from the World Bank and UNDP
python code/run_all.py --montecarlo      # also re-run the Monte-Carlo study (2,000 replications per cell, 11 scenarios) and the cointegration size/power study (several hours)
```

## Layout

| Folder | Content |
|---|---|
| `data/raw/` | Raw files exactly as downloaded: World Development Indicators (World Bank API) and the UNDP Human Development Report 2025 time series; `download_metadata.json` documents the vintage. |
| `data/processed/` | Analysis panels: 17 countries × 24 years (2000–2023) without imputation, an extended panel (adds Nicaragua) and the full wide file. |
| `code/` | `01_download_data.py` … `09_accounting.py` (empirical pipeline), `10_montecarlo.py` and `mc_summary.py` (Monte-Carlo study and its summaries), `11_cce_anatomy.py` (retention ratio and reliability of the CCE projection), `lib/econ.py` (all estimators and tests), `run_all.py`. |
| `output/mc/` | Cached Monte-Carlo draws: `mc2_<scenario>_T<T>.npy` (nine estimators) and `mcx2_<scenario>_T<T>.npy` (four dynamic augmentations); array shape = replications × estimator × {estimate, standard error} × slope. |
| `output/tables/`, `output/figures/` | LaTeX tables, `macros.tex` (numbers quoted in the text) and vector figures. |
| `tests/` | `validate_code.py` (internal validation), `export_validation.py`, `power_cointegration.py` (size and power of the Kao, Pedroni-type and Westerlund tests). |
| `validation/` | External cross-check against R (`plm`, `cointReg`) and Stata (`xtpmg`, `xtdcce2`, `xtwest`); see `validation/README.md`. |

## What the code does

* **Empirical case.** Digital adoption (share of the population using the Internet) and GDP per capita in 17 Latin American countries, 2000–2023: cross-sectional dependence (CD), CIPS unit-root tests with simulated critical values, slope homogeneity, Kao / Pedroni-type / Westerlund cointegration tests, static benchmarks, FMOLS, DOLS, PMG/MG, CCE-MG/P, CS-DL, first-difference models, local projections (LP-IV multiplier), Dumitrescu–Hurlin causality, robustness and an instrumental-variable exercise. Inference with few clusters uses wild-cluster bootstrap (Webb weights) and t(G−1) p-values.
* **Monte-Carlo study.** N = 17, T ∈ {20, 24, 30, 50, 100}, 2,000 replications per cell, bounded logistic digital regressor. Scenarios: S1N (exogenous regressors), S1 (endogenous), S1U (unbounded regressor), S2 (persistent errors), S3 (no cointegration), S1H/S2H (heterogeneous slopes), S1C (slopes correlated with the regressors), S1F (loadings tied to adoption), S1M (measurement error), S4 (combination).
* **Validation.** Independent re-implementations, correctly specified designs and an external cross-check in R (CCE-MG, CCE-P, TWFE, DOLS-GM (0,0) and PMG coincide to four decimals; FMOLS-GM within 0.009).

## Reproducibility notes

* All random draws use `numpy.random.SeedSequence` keyed by scenario, sample size and replication, so results do not depend on the number of processes.
* The PMG likelihood is maximised from several starting values and the global maximum is reported; the number of local optima is part of the output.
* The Stata cross-check scripts are provided but were not run (no valid licence at the time).

## Data sources

World Bank, World Development Indicators (accessed through the Indicators API); UNDP, Human Development Report 2025, composite indices time series (mean years of schooling).
