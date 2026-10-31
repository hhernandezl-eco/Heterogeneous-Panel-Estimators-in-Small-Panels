"""Step 7 - Dumitrescu-Hurlin (2012) Granger non-causality tests in heterogeneous panels.

Variables are first differences (both series are I(0) after differencing: see CIPS in Table 2) net of
year means (to attenuate cross-sectional dependence).  Bootstrap p-values (joint time resampling of
the restricted-model residuals) are reported next to the analytical Z-tilde p-values.

Outputs: tab_causality.tex, fig_dh_units.pdf
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from config import *
import lib.econ as E
from lib import plotstyle as ps
from lib.util import tabular, set_macros

ps.setup()
B = 499
df = pd.read_csv(PROC / "panel_main.csv").sort_values(["iso3", "year"])
ids = list(df.iso3.unique())
N = len(ids)
dy = df.groupby("iso3")["ln_gdppc"].diff()
dx = df.groupby("iso3")["inet_s"].diff()
dfm = pd.DataFrame({"iso3": df.iso3, "year": df.year, "dy": dy, "dx": dx}).dropna()
DY = dfm.pivot(index="iso3", columns="year", values="dy").loc[ids].to_numpy()
DX = dfm.pivot(index="iso3", columns="year", values="dx").loc[ids].to_numpy()
DYc, DXc = E.csd_demean(DY), E.csd_demean(DX)       # remove year means
T = DY.shape[1]
print("N,T of differenced data:", N, T)

rows, store = [], {}
for K in (1, 2, 3):
    a = E.dh_test(DYc, DXc, K=K, B=B, seed=10 + K)      # H0: Internet does not Granger-cause GDP growth
    b = E.dh_test(DXc, DYc, K=K, B=B, seed=20 + K)      # H0: GDP growth does not Granger-cause Internet
    store[K] = (a, b)
    rows.append([f"{K}", f"{a['Wbar']:.2f}", f"{a['Ztilde']:.2f}", f"{a['p_Ztilde']:.3f}", f"{a['p_boot']:.3f}",
                 f"{b['Wbar']:.2f}", f"{b['Ztilde']:.2f}", f"{b['p_Ztilde']:.3f}", f"{b['p_boot']:.3f}"])
    print(K, {k: round(v, 3) for k, v in a.items() if k != "W_units"}, "||", {k: round(v, 3) for k, v in b.items() if k != "W_units"})
note = (f"Dumitrescu and Hurlin (2012) tests on first differences net of year means ($N={N}$, $T={T}$). $K$ = number of lags. $\\bar W$ = average of the individual Wald statistics; "
        f"$\\tilde Z$ = standardised statistic for fixed $T$ (analytical, two-sided p-value); Boot.\\ $p$ = bootstrap p-value of $\\bar W$ ({B} replications, "
        "residuals resampled jointly across countries to preserve cross-sectional dependence). H$_0$: no country exhibits Granger causality.")
(TAB / "tab_causality.tex").write_text(tabular(
    rows, "lcccccccc",
    header=[["", "\\multicolumn{4}{c}{Internet $\\not\\rightarrow$ GDP growth}", "\\multicolumn{4}{c}{GDP growth $\\not\\rightarrow$ Internet}"],
            ["$K$", "$\\bar W$", "$\\tilde Z$", "$p$", "Boot.\\ $p$", "$\\bar W$", "$\\tilde Z$", "$p$", "Boot.\\ $p$"]], note=note), encoding="utf-8")

a1, b1 = store[1]; a2, b2 = store[2]
set_macros({"dhFwdPOne": f"{a1['p_boot']:.3f}", "dhRevPOne": f"{b1['p_boot']:.3f}",
            "dhFwdPTwo": f"{a2['p_boot']:.3f}", "dhRevPTwo": f"{b2['p_boot']:.3f}",
            "dhFwdZOne": f"{a1['Ztilde']:.2f}", "dhRevZOne": f"{b1['Ztilde']:.2f}",
            "dhFwdZTwo": f"{a2['Ztilde']:.2f}", "dhRevZTwo": f"{b2['Ztilde']:.2f}"})

# unit-level Wald statistics (K=1): where does the (non-)causality come from?
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True)
for ax, (nm, st) in zip(axs, [("Internet $\\to$ GDP growth", a1), ("GDP growth $\\to$ Internet", b1)]):
    w = st["W_units"]; o = np.argsort(w)
    ax.barh(range(N), w[o], color=ps.BLUE, height=0.7)
    ax.axvline(3.84, color=ps.ORANGE, lw=1.2, label="5% critical value ($\\chi^2_1$)")
    ax.set_yticks(range(N)); ax.set_yticklabels([ids[i] for i in o], fontsize=7)
    ax.set_xlabel("Individual Wald statistic ($K=1$)")
    ax.set_title(nm)
    ax.legend(loc="lower right")
ps.save(fig, FIG / "fig_dh_units.pdf")
print("causality done")
