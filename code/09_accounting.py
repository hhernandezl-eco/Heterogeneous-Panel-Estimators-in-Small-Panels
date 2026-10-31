"""Step 9 - Back-of-the-envelope accounting: what would the estimated semi-elasticities imply for the
observed rise in Internet use and in GDP per capita, 2000-2023?

Outputs: macros only (see output/tables/macros.tex) and tab_accounting.tex
"""
import numpy as np, pandas as pd
from config import *
from lib.util import set_macros, tabular

df = pd.read_csv(PROC / "panel_main.csv")
g = df.sort_values(["iso3", "year"]).groupby("iso3")
d_inet = (g["inet_s"].last() - g["inet_s"].first())
d_lny = (g["ln_gdppc"].last() - g["ln_gdppc"].first())
_all = pd.read_csv(TAB / "estimates_inet.csv").set_index("estimator")
est = _all["b"]
_lr = _all[_all.family == "long-run"]
union_lo, union_hi = (_lr.b - 1.96 * _lr.se).min(), (_lr.b + 1.96 * _lr.se).max()   # union of the 95% intervals
rows = []
for nm in ["CCE-MG", "FMOLS-GM", "PMG", "DOLS pooled"]:
    contrib = est[nm] * d_inet                         # log points attributable, by country
    rows.append([nm, f"{est[nm]:.3f}", f"{100 * contrib.mean():.1f}", f"{100 * contrib.mean() / d_lny.mean():.0f}"])
note = (f"Mean over the {df.iso3.nunique()} countries of the change between {YEAR0} and {YEAR1}: Internet-use share +{d_inet.mean():.2f}; ln GDP per capita "
        f"+{d_lny.mean():.2f} (about {100 * (np.exp(d_lny.mean()) - 1):.0f}\\%). The implied contribution is the estimated semi-elasticity times the observed rise in the share; "
        "it is an accounting illustration that treats the estimates as causal, which the evidence in this paper does not establish.")
(TAB / "tab_accounting.tex").write_text(tabular(rows, "lrrr", header=[["Estimator", "Semi-elasticity", "Implied rise in ln GDPpc (log pts $\\times 100$)", "Share of observed rise (\\%)"]], note=note), encoding="utf-8")
set_macros({"unionLo": f"{union_lo:.2f}", "unionHi": f"{union_hi:.2f}", "dInet": f"{d_inet.mean():.2f}", "dInetPp": f"{100 * d_inet.mean():.0f}", "dLnY": f"{d_lny.mean():.2f}",
            "dYPct": f"{100 * (np.exp(d_lny.mean()) - 1):.0f}",
            "meanInetStart": f"{100 * df[df.year == YEAR0].inet_s.mean():.0f}", "meanInetEnd": f"{100 * df[df.year == YEAR1].inet_s.mean():.0f}",
            "contribLowPct": rows[0][3], "contribHighPct": rows[2][3]})
print(pd.DataFrame(rows))
