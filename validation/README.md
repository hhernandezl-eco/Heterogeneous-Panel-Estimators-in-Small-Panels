# External cross-check (R and Stata)

The estimators are implemented in Python (`code/lib/econ.py`). This folder contains the material used to cross-check them against established software.

## Procedure
1. `python tests/export_validation.py` writes `panel_for_external_check.csv` (408 observations plus year-demeaned copies `*_dm`) and `python_results.csv`.
2. **R** (`plm`, `cointReg`): `Rscript check_r.R` writes `external_results_r.csv`. It also contains an independent implementation of the PMG concentrated likelihood.
3. **Stata** (`xtpmg`, `xtdcce2`, `xtwest`): run `check_stata.do` and copy the coefficients and standard errors into `external_results_stata.csv` (columns `estimator, regressor, coef, se`, same labels as `python_results.csv`).
4. `python validation/compare_external.py` prints the maximum absolute differences and writes `output/tables/tab_validation_external.tex`.

## Result (R)
CCE-MG, CCE-P, TWFE (Stata-type small-sample correction), DOLS-GM (0,0) and PMG coincide with the R results to four decimals in coefficients and standard errors. The group-mean FMOLS differs by at most 0.009 because `cointReg` uses its own long-run variance conventions. The Stata cross-check was not run because no valid Stata licence was available; the scripts are provided for anyone who can run them. Note that `xtpmg` returns a single local maximum of the PMG likelihood, which may differ from the global maximum reported here (the likelihood has several local optima).
