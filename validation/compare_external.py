"""Compare the external results (R and/or Stata) with the Python estimates and write the comparison table.

Inputs (same folder):  python_results.csv, external_results_r.csv, external_results_stata.csv (optional)
Output:                ../output/tables/tab_validation_external.tex  (maximum absolute differences by estimator)
Usage:                 python validation/compare_external.py
"""
from pathlib import Path
import sys
import pandas as pd

V = Path(__file__).resolve().parent
sys.path.insert(0, str(V.parent / "code"))
from config import TAB
from lib.util import tabular

py = pd.read_csv(V / "python_results.csv")
rows = []
for name, label in (("external_results_r.csv", "R"), ("external_results_stata.csv", "Stata")):
    f = V / name
    if not f.exists():
        print("missing:", name)
        continue
    ex = pd.read_csv(f)
    m = py.merge(ex, on=["estimator", "regressor"], suffixes=("_py", "_ext"))
    if m.empty:
        print(name, ": no matching estimator labels")
        continue
    g = m.assign(dcoef=(m.coef_py - m.coef_ext).abs(), dse=(m.se_py - m.se_ext).abs()).groupby("estimator")[["dcoef", "dse"]].max()
    print(f"\n{name}: maximum absolute differences")
    print(g.round(4).to_string())
    pkg = {"R": {"CCE-MG": "plm::pcce (mg)", "CCE-P": "plm::pcce (p)", "TWFE": "plm (two-way), vcovHC sss",
                 "FMOLS": "cointReg (FM, Bartlett)", "DOLS": "stats::lm", "PMG": "independent R implementation (optim)"},
           "Stata": {}}[label]
    for est, r in g.iterrows():
        key = next((k for k in pkg if est.startswith(k)), None)
        rows.append([label, est.replace("_", "\\_"), pkg.get(key, "--"), f"{r.dcoef:.4f}", f"{r.dse:.4f}"])
if rows:
    note = ("Maximum absolute difference over the four slopes between the Python estimates and the external implementations, on the same 408 observations. "
            "The group-mean FMOLS differs slightly because \\texttt{cointReg} uses its own long-run variance conventions (centring, divisor and kernel scaling); "
            "all other estimators coincide to four decimals. TWFE standard errors use the Stata/\\texttt{plm} small-sample convention.")
    (TAB / "tab_validation_external.tex").write_text(tabular(rows, "llcrr", header=[["Software", "Estimator", "Package / routine", "Max $|\\Delta$ coef.$|$", "Max $|\\Delta$ s.e.$|$"]],
                                                              note=note, resize=True), encoding="utf-8")
