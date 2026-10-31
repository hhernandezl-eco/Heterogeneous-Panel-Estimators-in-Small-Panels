"""Export the 408-observation panel and the Python estimates for the external cross-check in validation/.

Writes validation/panel_for_external_check.csv and validation/python_results.csv (estimator, regressor, coef, se).
Run `check_stata.do` and `check_r.R` (see validation/README.md) and then `python validation/compare_external.py`.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from config import *
import lib.econ as E

VAL = ROOT / "validation"
VAL.mkdir(exist_ok=True)
df = pd.read_csv(PROC / "panel_main.csv")
cols = ["iso3", "year", Y_VAR] + X_BASE
out = df[cols].copy()
for v in [Y_VAR] + X_BASE:                                  # year-demeaned copies (used by the PMG/MG/FMOLS/DOLS runs)
    out[v + "_dm"] = out[v] - out.groupby("year")[v].transform("mean")
out.to_csv(VAL / "panel_for_external_check.csv", index=False)

ids, Y, X = E.to_arrays(df, Y_VAR, X_BASE)
Yd, Xd = E.csd_demean(Y), E.csd_demean(X)
N, T, _K = X.shape
tw = E.twfe_cluster(Y, X)
n_obs, kk = N * T, len(X_BASE) + N + T - 2
tw_stata = dict(tw)                                           # Stata/plm 'sss' convention: (n-1)/(n-K) instead of (n-1)/(n-K-N-T+2)
tw_stata["se"] = tw["se"] * np.sqrt((n_obs - kk) / (n_obs - len(X_BASE)))
res = {"PMG (global ML, year means removed)": E.pmg(Yd, Xd), "MG (year means removed)": E.mg_ecm(Yd, Xd),
       "CCE-MG": E.cce_mg(Y, X), "CCE-P": E.cce_pooled(Y, X), "FMOLS-GM (year means removed)": E.fmols_gm(Y, X, M=2),
       "DOLS-GM (0,0) (year means removed)": E.dols_gm(Y, X, 0, 0),
       "DOLS pooled (1,1)": E.dols_pooled(Y, X, 1, 1), "TWFE (CR1, Stata convention)": tw_stata}
rows = []
for k, r in res.items():
    for j, v in enumerate(X_BASE):
        rows.append([k, v, r["b"][j], r["se"][j]])
pd.DataFrame(rows, columns=["estimator", "regressor", "coef", "se"]).to_csv(VAL / "python_results.csv", index=False)
print("exported", len(rows), "rows to", VAL)
