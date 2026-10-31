* External cross-check of the Python estimators against established Stata packages.
* Install once:  ssc install xtpmg  |  ssc install xtdcce2  |  ssc install xtwest  |  ssc install xtcips
* Run from this folder.  Results are written to external_results_stata.csv (estimator, regressor, coef, se).
* NOTE: option names differ slightly across package versions; adapt if Stata complains.  The Python code
*       removes year means (variables *_dm) before PMG/MG/FMOLS/DOLS, and uses the raw variables for CCE.

clear all
set more off
import delimited "panel_for_external_check.csv", clear
encode iso3, gen(id)
xtset id year

* ---------- CCE mean group and pooled (Pesaran 2006): xtdcce2 with cross-section averages of all variables ----------
xtdcce2 ln_gdppc ln_gfcf ln_hc inet_s ln_trade, cr(ln_gdppc ln_gfcf ln_hc inet_s ln_trade) cr_lags(0) nocross
* mean-group coefficients are in e(b); the pooled version:
xtdcce2 ln_gdppc ln_gfcf ln_hc inet_s ln_trade, cr(ln_gdppc ln_gfcf ln_hc inet_s ln_trade) cr_lags(0) pooled(ln_gfcf ln_hc inet_s ln_trade) nocross

* ---------- PMG / MG (Pesaran, Shin, Smith 1999) on year-demeaned data, ARDL(1,1,1,1,1) ----------
* Differences, lagged levels
foreach v in ln_gdppc ln_gfcf ln_hc inet_s ln_trade {
    gen d_`v' = D.`v'_dm
    gen l_`v' = L.`v'_dm
}
xtpmg d_ln_gdppc d_ln_gfcf d_ln_hc d_inet_s d_ln_trade, lr(l_ln_gdppc ln_gfcf_dm ln_hc_dm inet_s_dm ln_trade_dm) ec(ec) pmg replace
* NOTE: xtpmg uses a single (Newton-Raphson) maximisation from default starting values; the Python code evaluates the likelihood from
* several starting values and reports the global maximum.  Compare the xtpmg result with the *local optima* listed in Table 'pmgmodes'
* of the paper if the point estimates differ.
xtpmg d_ln_gdppc d_ln_gfcf d_ln_hc d_inet_s d_ln_trade, lr(l_ln_gdppc ln_gfcf_dm ln_hc_dm inet_s_dm ln_trade_dm) ec(ec) mg replace

* ---------- Westerlund (2007) error-correction tests, bootstrap p-values ----------
xtwest ln_gdppc_dm ln_gfcf_dm ln_hc_dm inet_s_dm ln_trade_dm, lags(0) leads(0) constant bootstrap(499)
