"""Step 5 - Panel cointegration tests (H0: no cointegration).

Kao (1999)                     homogeneous slopes, analytical N(0,1)
Pedroni-type residual ADF      heterogeneous slopes, common time effects removed, Monte-Carlo p-values
Westerlund (2007) ECM tests    bootstrap p-values robust to cross-sectional dependence

Reported for the baseline model and for the parsimonious bivariate model (GDPpc, Internet).
"""
import numpy as np, pandas as pd
from config import *
import lib.econ as E
from lib.util import tabular, set_macros

R_MC, B_BOOT = 2000, 499
df = pd.read_csv(PROC / "panel_main.csv")

models = {"Baseline (GFCF, schooling, Internet, trade)": X_BASE,
          "Bivariate (Internet only)": ["inet_s"],
          "Neoclassical core (GFCF, schooling, Internet)": ["ln_gfcf", "ln_hc", "inet_s"]}
rows = []
res_all = {}
for name, xs in models.items():
    ids, Y, X = E.to_arrays(df, Y_VAR, xs)
    k = kao_p = E.kao_adf(Y, X, p=1)
    pe0 = E.pedroni_adf(Y, X, p=0, R=R_MC)
    pe1 = E.pedroni_adf(Y, X, p=1, R=R_MC)
    # Westerlund on data net of year means (common time effects), lag order 0
    we = E.westerlund(E.csd_demean(Y), E.csd_demean(X), p=0, B=B_BOOT)
    res_all[name] = dict(kao=k, pe0=pe0, pe1=pe1, we=we)
    print(name)
    print("  Kao ADF", round(k["ADF"], 3), "p", round(k["p"], 4))
    print("  Pedroni p=0", {a: round(b, 3) for a, b in pe0.items()})
    print("  Pedroni p=1", {a: round(b, 3) for a, b in pe1.items()})
    print("  Westerlund", {a: round(b, 3) for a, b in we.items()})
    rows.append([f"\\textit{{{name}}}"] + [""] * 3)
    rows.append(["Kao ADF", f"{k['ADF']:.2f}", f"{k['p']:.3f}", "analytical"])
    rows.append(["Pedroni group-ADF, $p=0$", f"{pe0['group_adf']:.2f}", f"{pe0['p_group']:.3f}", "simulated"])
    rows.append(["Pedroni panel-ADF, $p=0$", f"{pe0['panel_adf']:.2f}", f"{pe0['p_panel']:.3f}", "simulated"])
    rows.append(["Pedroni group-ADF, $p=1$", f"{pe1['group_adf']:.2f}", f"{pe1['p_group']:.3f}", "simulated"])
    rows.append(["Pedroni panel-ADF, $p=1$", f"{pe1['panel_adf']:.2f}", f"{pe1['p_panel']:.3f}", "simulated"])
    for s in ("Gt", "Ga", "Pt", "Pa"):
        rows.append([f"Westerlund ${s[0]}_{s[1]}$", f"{we[s]:.2f}", f"{we['p_' + s]:.3f}", "bootstrap"])

note = (f"H$_0$: no cointegration (left-tail tests). Kao: pooled ADF on fixed-effects residuals with one lag, bandwidth-2 Bartlett long-run variances. "
        f"Pedroni-type: residual ADF statistics from country-specific cointegrating regressions on data net of year means; $p$ = ADF lag order; "
        f"p-values from {R_MC:,} Monte-Carlo replications under independent random walks with identical $N$, $T$ and number of regressors. "
        f"Westerlund (2007): error-correction tests with a constant and contemporaneous differences of the regressors, data net of year means; "
        f"p-values from {B_BOOT} bootstrap replications that resample the restricted-model innovations jointly across countries "
        f"(robust to cross-sectional dependence). $G$ = group-mean, $P$ = panel statistics.")
(TAB / "tab_cointegration.tex").write_text(
    tabular(rows, "lrrl", header=[["Test", "Statistic", "p-value", "p-value source"]],
            midrules=[], note=note), encoding="utf-8")

b = res_all["Baseline (GFCF, schooling, Internet, trade)"]
set_macros({"kaoStat": f"{b['kao']['ADF']:.2f}", "kaoP": f"{b['kao']['p']:.3f}",
            "pedGroupP": f"{b['pe0']['p_group']:.3f}", "pedPanelP": f"{b['pe0']['p_panel']:.3f}",
            "westGtP": f"{b['we']['p_Gt']:.3f}", "westGaP": f"{b['we']['p_Ga']:.3f}",
            "westPtP": f"{b['we']['p_Pt']:.3f}", "westPaP": f"{b['we']['p_Pa']:.3f}"})
print("cointegration done")
