"""Step 4 - Panel diagnostics: cross-sectional dependence, unit roots, slope heterogeneity.

Outputs: tab_cd_unitroot.tex, tab_slope_hom.tex, fig_corr_heatmap.pdf
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from config import *
import lib.econ as E
from lib import plotstyle as ps
from lib.util import tabular, set_macros

ps.setup()
R_SIM = 4000
df = pd.read_csv(PROC / "panel_main.csv")
ids, Y, X = E.to_arrays(df, Y_VAR, X_BASE)
allv = [Y_VAR] + X_BASE
Z = {v: df.pivot(index="iso3", columns="year", values=v).loc[ids].to_numpy() for v in allv}

rows, mac = [], {}
for v in allv:
    z = Z[v]; dz = np.diff(z, axis=1)
    cd_l, cd_d = E.cd_test(z), E.cd_test(dz)
    # CIPS with intercept; lag orders 0 and 1.  Levels: also with trend (series are trending).
    c0 = E.cips_test(z, 0, False, R_SIM)
    c1 = E.cips_test(z, 1, False, R_SIM)
    ct0 = E.cips_test(z, 0, True, R_SIM)
    d0 = E.cips_test(dz, 0, False, R_SIM)
    d1 = E.cips_test(dz, 1, False, R_SIM)
    fi = E.adf_fisher(E.csd_demean(z[:, :, None])[:, :, 0], lags=1)
    fid = E.adf_fisher(E.csd_demean(dz[:, :, None])[:, :, 0], lags=1)

    def cell(r):
        s = "***" if r["p"] < .01 else "**" if r["p"] < .05 else "*" if r["p"] < .10 else ""
        return f"{r['CIPS']:.2f}{s}"
    lab = X_LABELS[v]
    rows.append([lab, f"{cd_l['CD']:.1f}", f"{cd_l['mean_abs_rho']:.2f}", cell(c0), cell(c1), cell(ct0),
                 f"{cd_d['CD']:.1f}", cell(d0), cell(d1)])
    key = {"ln_gdppc": "Gdp", "ln_gfcf": "Gfcf", "ln_hc": "Hc", "inet_s": "Inet", "ln_trade": "Trade"}[v]
    mac = {f"cdLevel{key}": f"{cd_l['CD']:.1f}", f"cipsLevel{key}": f"{c0['CIPS']:.2f}",
           f"cipsLevelP{key}": f"{c0['p']:.3f}", f"cipsDiff{key}": f"{d0['CIPS']:.2f}"}
    set_macros(mac)
    print(f"{v:10s} CD={cd_l['CD']:.1f}  CIPS0={c0['CIPS']:.2f} (p={c0['p']:.3f}) CIPS1={c1['CIPS']:.2f} (p={c1['p']:.3f}) "
          f"CIPS0-trend={ct0['CIPS']:.2f} (p={ct0['p']:.3f}) | D: CD={cd_d['CD']:.1f} CIPS0={d0['CIPS']:.2f} (p={d0['p']:.3f}) "
          f"CIPS1={d1['CIPS']:.2f} (p={d1['p']:.3f}) | Fisher-ADF(csd-demeaned) p={fi['p_P']:.3f}/{fid['p_P']:.3f}")
cv = E.cips_test(Z["ln_gdppc"], 0, False, R_SIM)
cvd = E.cips_test(np.diff(Z["ln_gdppc"], axis=1), 0, False, R_SIM)
tex = tabular(rows, "lrrrrrrrr",
              header=[["", "\\multicolumn{5}{c}{Levels}", "\\multicolumn{3}{c}{First differences}"],
                      ["Variable", "CD", "$\\overline{|\\rho|}$", "CIPS(0)", "CIPS(1)", "CIPS(0)$_{\\tau}$", "CD", "CIPS(0)", "CIPS(1)"]],
              note=f"CD = Pesaran (2004) cross-sectional dependence statistic (N(0,1) under cross-sectional independence; $|CD|>2.58$ rejects at 1\\%); "
                   f"$\\overline{{|\\rho|}}$ = mean absolute pairwise correlation. CIPS($p$) = Pesaran (2007) statistic with $p$ lags of the augmentation terms; "
                   f"subscript $\\tau$ = intercept and linear trend. H$_0$: unit root in every country. Critical values are simulated for the exact sample "
                   f"dimensions ($N={X.shape[0]}$, $T={X.shape[1]}$ in levels; $T={X.shape[1]-1}$ in differences; {R_SIM:,} replications): "
                   f"levels 10\\%/5\\%/1\\% = {cv['cv10']:.2f}/{cv['cv05']:.2f}/{cv['cv01']:.2f}; differences = {cvd['cv10']:.2f}/{cvd['cv05']:.2f}/{cvd['cv01']:.2f}. "
                   "*, **, *** denote rejection at 10\\%, 5\\%, 1\\%.")
(TAB / "tab_cd_unitroot.tex").write_text(tex, encoding="utf-8")

# ---- slope homogeneity (static FE-type regression, cross-section demeaned) ----
Yd, Xd = E.csd_demean(Y), E.csd_demean(X)
sh = E.slope_homogeneity(Yd, Xd)
print("slope homogeneity:", {k: round(v, 3) for k, v in sh.items()})
set_macros({"shDelta": f"{sh['delta']:.2f}", "shDeltaAdj": f"{sh['delta_adj']:.2f}",
            "shP": f"{sh['p_delta']:.3f}", "shPAdj": f"{sh['p_delta_adj']:.3f}"})
rows = [["$\\tilde\\Delta$", f"{sh['delta']:.2f}", f"{sh['p_delta']:.4f}"],
        ["$\\tilde\\Delta_{adj}$", f"{sh['delta_adj']:.2f}", f"{sh['p_delta_adj']:.4f}"]]
(TAB / "tab_slope_hom.tex").write_text(tabular(rows, "lrr", header=[["Test", "Statistic", "p-value"]],
    note="Pesaran and Yamagata (2008) slope-homogeneity tests on the static long-run regression (variables net of year means). H$_0$: common slopes."),
    encoding="utf-8")

# ---- Fig: correlation heatmap of GDP growth across countries ----
g = np.diff(Z["ln_gdppc"], axis=1)
C = np.corrcoef(g)
order = np.argsort(-C.mean(axis=1))
fig, ax = plt.subplots(figsize=(4.4, 3.8))
im = ax.imshow(C[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(ids))); ax.set_yticks(range(len(ids)))
ax.set_xticklabels([ids[i] for i in order], rotation=90, fontsize=6.5)
ax.set_yticklabels([ids[i] for i in order], fontsize=6.5)
ax.grid(False)
cb = fig.colorbar(im, fraction=0.046, pad=0.03); cb.set_label("Correlation of GDPpc growth")
ps.save(fig, FIG / "fig_corr_heatmap.pdf")
iu = np.triu_indices(len(ids), 1)
set_macros({"meanGrowthCorr": f"{C[iu].mean():.2f}"})
print("diagnostics done")
