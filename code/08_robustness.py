"""Step 8 - Robustness and identification checks.

(a) alternative specifications / measures / samples (4 long-run estimators each);
(b) leave-one-country-out;
(c) instrumental-variable evidence (legacy fixed-telephone network x frontier diffusion).

Effects are reported as the change in ln GDPpc associated with a one-standard-deviation increase in the digital variable
(sd computed on the sample used), which makes rows with different digital measures comparable.

Outputs: tab_robustness.tex, tab_iv.tex, fig_loo.pdf, robustness_raw.csv
"""
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from config import *
import lib.econ as E
from lib import plotstyle as ps
from lib.util import tabular, set_macros, stars

ps.setup()
main = pd.read_csv(PROC / "panel_main.csv")
ext = pd.read_csv(PROC / "panel_ext.csv")
SOUTH = ["ARG", "BOL", "BRA", "CHL", "COL", "ECU", "PRY", "PER", "URY"]
CENTRAL = ["CRI", "DOM", "SLV", "GTM", "HND", "HTI", "MEX", "PAN"]


def fit_all(df, y, xs, dj):
    """Run the four estimators; return {name: (b_dj, se_dj)} for digital variable index dj."""
    ids, Y, X = E.to_arrays(df, y, xs)
    Yd, Xd = E.csd_demean(Y), E.csd_demean(X)
    out = {}
    for nm, f in [("FMOLS-GM", lambda: E.fmols_gm(Y, X)), ("DOLS pooled", lambda: E.dols_pooled(Y, X, 1, 1)),
                  ("PMG", lambda: E.pmg(Yd, Xd)), ("CCE-MG", lambda: E.cce_mg(Y, X))]:
        try:
            r = f()
            out[nm] = (r["b"][dj], r["se"][dj])
        except Exception as ex:                                    # e.g. singular design in tiny subsamples
            out[nm] = (np.nan, np.nan)
    return out, X[:, :, dj].std(ddof=1)


def balanced(df, vars_, y0=None, y1=None):
    d = df.copy()
    if y0: d = d[d.year >= y0]
    if y1: d = d[d.year <= y1]
    d = d.dropna(subset=vars_)
    T = d.year.nunique()
    keep = d.groupby("iso3").size()
    return d[d.iso3.isin(keep[keep == T].index)]


B = ["ln_gfcf", "ln_hc", "inet_s", "ln_trade"]
variants = []
variants.append(("Baseline", main, Y_VAR, B, 2))
variants.append(("Log-log: ln Internet users", main, Y_VAR, ["ln_gfcf", "ln_hc", "ln_inet", "ln_trade"], 2))
variants.append(("Mobile subscriptions instead of Internet", balanced(main, ["mobile_s"]), Y_VAR, ["ln_gfcf", "ln_hc", "mobile_s", "ln_trade"], 2))
fb = balanced(main, ["fbb_s"], 2005, 2023)
variants.append((f"Fixed broadband, 2005--2023 (N={fb.iso3.nunique()})", fb, Y_VAR, ["ln_gfcf", "ln_hc", "fbb_s", "ln_trade"], 2))
variants.append(("GDP per capita in PPP", main, "ln_gdppc_ppp", B, 2))
variants.append(("Sample ends in 2019 (pre-COVID)", main[main.year <= 2019], Y_VAR, B, 2))
variants.append(("Sample starts in 2005", main[main.year >= 2005], Y_VAR, B, 2))
variants.append((f"Extended sample, GCF (N={ext.iso3.nunique()})", ext, Y_VAR, B, 2))
variants.append(("Add inflation (GDP deflator)", main, Y_VAR, B + ["infl_l"], 2))
variants.append(("Drop trade openness", main, Y_VAR, ["ln_gfcf", "ln_hc", "inet_s"], 2))
variants.append(("Drop schooling", main, Y_VAR, ["ln_gfcf", "inet_s", "ln_trade"], 1))
variants.append(("Drop GFCF and schooling", main, Y_VAR, ["inet_s", "ln_trade"], 0))
variants.append((f"South America only (N={len(SOUTH)})", main[main.iso3.isin(SOUTH)], Y_VAR, B, 2))
variants.append((f"Mexico, Central America, Caribbean (N={len(CENTRAL)})", main[main.iso3.isin(CENTRAL)], Y_VAR, B, 2))

rows, raw = [], []
for name, d, y, xs, dj in variants:
    out, sd = fit_all(d, y, xs, dj)
    cells = []
    for nm in ["FMOLS-GM", "DOLS pooled", "PMG", "CCE-MG"]:
        b, se = out[nm]
        p = 2 * (1 - stats.norm.cdf(abs(b / se))) if np.isfinite(b) and se > 0 else 1
        cells.append(f"{b * sd:.3f}{stars(p)} ({se * sd:.3f})" if np.isfinite(b) else "--")
        raw.append({"variant": name, "estimator": nm, "b": b, "se": se, "sd_x": sd})
    rows.append([name, f"{sd:.3f}"] + cells)
    print(name, "| sd", round(sd, 3), "|", cells, flush=True)
note = ("Entries: change in ln GDP per capita associated with a one-standard-deviation increase in the digital variable (standard error in parentheses); "
        "the s.d.\\ of the digital variable in each sample is shown in the second column. Baseline specification and estimators as in Table~\\ref{tab:longrun}. "
        "``Extended sample'' replaces GFCF by gross capital formation to include Nicaragua. *, **, *** denote significance at 10\\%, 5\\%, 1\\%.")
(TAB / "tab_robustness.tex").write_text(tabular(rows, "lrcccc",
    header=[["Variant", "s.d.", "FMOLS--GM", "DOLS pooled", "PMG", "CCE--MG"]], midrules=[0], note=note, resize=True), encoding="utf-8")
pd.DataFrame(raw).to_csv(TAB / "robustness_raw.csv", index=False)

# ---------------- Leave-one-country-out ----------------
isos = list(main.iso3.unique())
loo = {nm: [] for nm in ["PMG", "FMOLS-GM", "CCE-MG"]}
for iso in isos:
    out, sd = fit_all(main[main.iso3 != iso], Y_VAR, B, 2)
    for nm in loo:
        loo[nm].append(out[nm][0])
full, _ = fit_all(main, Y_VAR, B, 2)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.9), sharey=True)
for ax, nm in zip(axs, loo):
    v = np.array(loo[nm]); o = np.argsort(v)
    ax.scatter(v[o], range(len(isos)), color=ps.BLUE, s=16, zorder=3)
    ax.axvline(full[nm][0], color=ps.ORANGE, lw=1.4, label="Full sample")
    ax.axvline(0, color=ps.DARK, lw=0.7)
    ax.set_yticks(range(len(isos))); ax.set_yticklabels([f"w/o {isos[i]}" for i in o], fontsize=6.5)
    ax.set_title(nm.replace("-", "--"))
    ax.set_xlabel("Internet coefficient")
axs[0].legend(loc="lower right")
fig.tight_layout()
ps.save(fig, FIG / "fig_loo.pdf")
rng_ = {nm: (np.nanmin(v), np.nanmax(v)) for nm, v in loo.items()}
_pm = np.array(loo["PMG"])
set_macros({"looPmgMinCountry": LATAM[isos[int(np.nanargmin(_pm))]], "looPmgMaxCountry": LATAM[isos[int(np.nanargmax(_pm))]]})
print("LOO ranges:", rng_)
set_macros({"looPmgMin": f"{rng_['PMG'][0]:.2f}", "looPmgMax": f"{rng_['PMG'][1]:.2f}",
            "looCceMin": f"{rng_['CCE-MG'][0]:.2f}", "looCceMax": f"{rng_['CCE-MG'][1]:.2f}",
            "looFmolsMin": f"{rng_['FMOLS-GM'][0]:.2f}", "looFmolsMax": f"{rng_['FMOLS-GM'][1]:.2f}"})

# ---------------- IV: legacy telephone network x frontier diffusion ----------------
def twoway_demean(d, cols):
    out = d.copy()
    for c in cols:
        out[c] = d[c] - d.groupby("iso3")[c].transform("mean") - d.groupby("year")[c].transform("mean") + d[c].mean()
    return out


def iv_2sls(d, y, endog, exog, z, extra_ctrl, label):
    """2SLS after two-way demeaning (balanced panel), country-clustered variance, plus Anderson-Rubin p-value."""
    cols = [y, endog] + exog + [z] + extra_ctrl
    dd = twoway_demean(d.dropna(subset=cols), cols)
    cl = pd.factorize(dd["iso3"])[0]
    W = dd[exog + extra_ctrl].to_numpy()
    yv, xv, zv = dd[y].to_numpy(), dd[endog].to_numpy(), dd[z].to_numpy()
    # first stage
    F1 = sm.OLS(xv, np.column_stack([zv, W])).fit(cov_type="cluster", cov_kwds={"groups": cl})
    fs_t = F1.params[0] / F1.bse[0]
    xhat = np.column_stack([zv, W]) @ F1.params
    S = sm.OLS(yv, np.column_stack([xhat, W])).fit()
    b = S.params[0]
    # proper 2SLS residuals use actual x
    res = yv - np.column_stack([xv, W]) @ S.params
    Zr = np.column_stack([xhat, W])
    bread = np.linalg.inv(Zr.T @ np.column_stack([xv, W]))
    meat = np.zeros((Zr.shape[1],) * 2)
    for g in np.unique(cl):
        s = Zr[cl == g].T @ res[cl == g]
        meat += np.outer(s, s)
    n, k, G = len(yv), Zr.shape[1], len(np.unique(cl))
    V = (G / (G - 1)) * ((n - 1) / (n - k)) * bread @ meat @ bread.T
    se = np.sqrt(V[0, 0])
    # Anderson-Rubin test of beta = 0 (robust to weak instruments): reduced-form coefficient on z
    RF = sm.OLS(yv, np.column_stack([zv, W])).fit(cov_type="cluster", cov_kwds={"groups": cl})
    return {"label": label, "b": b, "se": se, "fs_t": fs_t, "fs_F": fs_t ** 2, "ar_p": RF.pvalues[0], "rf": RF.params[0], "n": n, "G": G}


iv = main.copy().sort_values(["iso3", "year"])
iv["z"] = iv["ln_tel95"] * iv["ln_inet_hic"]
iv["c_inc"] = iv["ln_gdppc95"] * iv["ln_inet_hic"]
res_iv = []
res_iv.append(iv_2sls(iv, "ln_gdppc", "inet_s", ["ln_gfcf", "ln_hc", "ln_trade"], "z", ["c_inc"], "Levels, TWFE"))
res_iv.append(iv_2sls(iv, "ln_gdppc", "inet_s", ["ln_gfcf", "ln_hc", "ln_trade"], "z", [], "Levels, TWFE, no income-trend control"))
dv = iv.copy()
for v in ["ln_gdppc", "inet_s", "ln_gfcf", "ln_hc", "ln_trade", "ln_inet_hic"]:
    dv["d_" + v] = dv.groupby("iso3")[v].diff()
dv["z"] = dv["ln_tel95"] * dv["d_ln_inet_hic"]
dv["c_inc"] = dv["ln_gdppc95"] * dv["d_ln_inet_hic"]
dv = dv.dropna(subset=["d_ln_gdppc"])
res_iv.append(iv_2sls(dv, "d_ln_gdppc", "d_inet_s", ["d_ln_gfcf", "d_ln_hc", "d_ln_trade"], "z", ["c_inc"], "First differences, TWFE"))
rows = []
for r in res_iv:
    p = 2 * (1 - stats.norm.cdf(abs(r["b"] / r["se"])))
    rows.append([r["label"], f"{r['b']:.3f}{stars(p)}", f"({r['se']:.3f})", f"{r['fs_F']:.1f}", f"{r['ar_p']:.3f}", str(r["n"])])
    print(r)
note = ("2SLS with country and year effects (two-way demeaning), standard errors clustered by country ($G=" + str(res_iv[0]["G"]) + "$). Instrument: 1995 fixed-telephone lines per 100 "
        "inhabitants (log) interacted with the log of Internet use in high-income countries (frontier diffusion); the interaction is the only source of identifying variation "
        "because country and year effects absorb the main terms. All specifications except the second control for $\\ln \\mathrm{GDPpc}_{1995}\\times$ frontier diffusion, "
        "which absorbs income-dependent exposure to global trends. $F$ = squared cluster-robust $t$-statistic of the instrument in the first stage; "
        "AR $p$ = Anderson--Rubin (weak-instrument-robust) $p$-value for H$_0$: no effect. The exclusion restriction is plausible but untestable; with 17 clusters "
        "inference is indicative only.")
(TAB / "tab_iv.tex").write_text(tabular(rows, "lcccrr",
    header=[["Specification", "Internet coef.", "s.e.", "First-stage $F$", "AR $p$", "Obs."]], note=note), encoding="utf-8")
r0 = res_iv[0]
set_macros({"ivCoef": f"{r0['b']:.3f}", "ivSe": f"{r0['se']:.3f}", "ivF": f"{r0['fs_F']:.1f}", "ivAR": f"{r0['ar_p']:.3f}"})
print("robustness done")
