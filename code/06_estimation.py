"""Step 6 - Main estimation.

Part A  Static benchmarks in levels (pooled OLS, RE, FE, TWFE) - the specification of the original term paper.
Part B  Long-run estimators for non-stationary heterogeneous panels with cross-sectional dependence:
        FMOLS, DOLS, PMG, MG, CCE-MG, CCE-P, CS-DL.
Part C  Growth-rate (first-difference) models and local projections - valid whether or not the levels cointegrate.

Outputs: tab_static.tex, tab_longrun.tex, tab_pmg_detail.tex, tab_growth.tex,
         fig_forest_inet.pdf, fig_country_lr.pdf, fig_irf.pdf, estimates_inet.csv
"""
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
from config import *
import lib.econ as E
from lib import plotstyle as ps
from lib.util import tabular, set_macros, stars

ps.setup()
df = pd.read_csv(PROC / "panel_main.csv")
ids, Y, X = E.to_arrays(df, Y_VAR, X_BASE)
N, T, K = X.shape
Yd, Xd = E.csd_demean(Y), E.csd_demean(X)      # remove common year effects (CSD mitigation)
lab = {v: X_LABELS[v] for v in X_BASE}


def fp(p):
    """Format a p-value (bootstrap p-values can be exactly zero)."""
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def cell(b, se, d=3):
    p = 2 * (1 - stats.norm.cdf(abs(b / se))) if se > 0 else 1
    return f"{b:.{d}f}{stars(p)}", f"({se:.{d}f})"


# =============================== Part A: static benchmarks ===============================
pdf = df.set_index(["iso3", "year"])
exog = sm.add_constant(pdf[X_BASE])
res = {}
res["Pooled OLS"] = PooledOLS(pdf[Y_VAR], exog).fit(cov_type="clustered", cluster_entity=True)
res["Random effects"] = RandomEffects(pdf[Y_VAR], exog).fit(cov_type="clustered", cluster_entity=True)
res["Fixed effects"] = PanelOLS(pdf[Y_VAR], pdf[X_BASE], entity_effects=True).fit(cov_type="clustered", cluster_entity=True)
res["Two-way FE"] = PanelOLS(pdf[Y_VAR], pdf[X_BASE], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
res["Two-way FE (DK)"] = PanelOLS(pdf[Y_VAR], pdf[X_BASE], entity_effects=True, time_effects=True).fit(cov_type="kernel", kernel="bartlett", bandwidth=3)

# Mundlak / Wooldridge cluster-robust version of the Hausman test (the classical contrast matrix is not
# positive definite here): add country means of the regressors to the RE-type regression and test them jointly.
mm = df.copy()
for v in X_BASE:
    mm["m_" + v] = mm.groupby("iso3")[v].transform("mean")
Dm = sm.add_constant(mm[X_BASE + ["m_" + v for v in X_BASE]])
mfit = sm.OLS(mm[Y_VAR], Dm).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(mm["iso3"])[0]})
Rm = np.zeros((K, len(mfit.params)))
for k, v in enumerate(X_BASE):
    Rm[k, list(mfit.params.index).index("m_" + v)] = 1
wt = mfit.wald_test(Rm, use_f=False, scalar=True)
H, pH = float(wt.statistic), float(wt.pvalue)
print("Mundlak-Hausman:", H, pH)

# residual CSD in the static models
def resid_cd(r):
    e = r.resids.to_numpy().reshape(N, T)
    return E.cd_test(e)["CD"]
cd_fe, cd_tw = resid_cd(res["Fixed effects"]), resid_cd(res["Two-way FE"])

# Inference with G = 17 clusters: wild-cluster restricted bootstrap-t (Webb weights, 999 draws) and t(G-1) p-values for the Internet slope
B_WCB = 999
u_id, t_id = np.repeat(np.arange(N), T), np.tile(np.arange(T), N)
Yu, Xu = Y - Y.mean(1, keepdims=True), X - X.mean(1, keepdims=True)
Ytw = Y - Y.mean(0, keepdims=True) - Y.mean(1, keepdims=True) + Y.mean()
Xtw = X - X.mean(0, keepdims=True) - X.mean(1, keepdims=True) + X.mean(axis=(0, 1), keepdims=True)
wcb = {"Pooled OLS": E.wild_cluster_p(Y.reshape(-1), np.column_stack([np.ones(N * T), X.reshape(-1, K)]), u_id, 1 + J_INET, B=B_WCB, seed=1),
       "Fixed effects": E.wild_cluster_p(Yu.reshape(-1), Xu.reshape(-1, K), u_id, J_INET, B=B_WCB, seed=2, absorb=[u_id]),
       "Two-way FE": E.wild_cluster_p(Ytw.reshape(-1), Xtw.reshape(-1, K), u_id, J_INET, B=B_WCB, seed=3, absorb=[u_id, t_id])}
print("WCB p (Internet):", {k: round(v[0], 3) for k, v in wcb.items()}, "t(G-1) p:", {k: round(v[2], 3) for k, v in wcb.items()})
set_macros({"wcbTwfeInet": f"{wcb['Two-way FE'][0]:.2f}", "wcbFeInet": f"{wcb['Fixed effects'][0]:.2f}", "wcbPooledInet": f"{wcb['Pooled OLS'][0]:.2f}",
            "tgTwfeInet": f"{wcb['Two-way FE'][2]:.2f}", "tgFeInet": f"{fp(wcb['Fixed effects'][2])}"})

cols = ["Pooled OLS", "Random effects", "Fixed effects", "Two-way FE", "Two-way FE (DK)"]
rows = []
for v in X_BASE:
    r1, r2 = [lab[v]], [""]
    for c in cols:
        r = res[c]
        b, se = float(r.params[v]), float(r.std_errors[v])
        a, s = cell(b, se)
        r1.append(a); r2.append(s)
    rows += [r1, r2]
rows.append(["Country effects"] + ["No", "RE", "Yes", "Yes", "Yes"])
rows.append(["Year effects"] + ["No", "No", "No", "Yes", "Yes"])
rows.append(["Observations"] + [str(N * T)] * 5)
rows.append(["$R^2$"] + [f"{res[c].rsquared_within:.3f}" if hasattr(res[c], 'rsquared_within') else f"{res[c].rsquared:.3f}" for c in cols])
rows.append(["Residual CD statistic"] + ["--", "--", f"{cd_fe:.1f}", f"{cd_tw:.1f}", f"{cd_tw:.1f}"])
rows.append(["$t(G-1)$ $p$-value, Internet"] + [f"{fp(wcb['Pooled OLS'][2])}", "--", f"{fp(wcb['Fixed effects'][2])}", f"{fp(wcb['Two-way FE'][2])}", "--"])
rows.append(["Wild-cluster bootstrap $p$-value, Internet"] + [f"{fp(wcb['Pooled OLS'][0])}", "--", f"{fp(wcb['Fixed effects'][0])}", f"{fp(wcb['Two-way FE'][0])}", "--"])
note = (f"Dependent variable: ln GDP per capita. Standard errors clustered by country in parentheses (Driscoll--Kraay with a Bartlett kernel of bandwidth 3 in the last column). "
        f"Because there are only $G={N}$ clusters, the last two rows report, for the Internet coefficient, the $p$-value from a $t(G-1)$ distribution and from a wild-cluster restricted bootstrap-$t$ with Webb weights ({B_WCB} draws), which are more reliable than the stars based on the normal approximation. "
        f"Mundlak (cluster-robust Hausman) test of RE vs.\\ FE: $\\chi^2({K})={H:.1f}$, $p={pH:.3f}$. Residual CD: Pesaran statistic of the model residuals; the country-effects-only residuals are strongly cross-sectionally dependent, "
        f"so those estimates (columns 1--3) are unreliable, whereas year effects absorb most of the common variation. *, **, *** denote significance at 10\\%, 5\\%, 1\\%.")
(TAB / "tab_static.tex").write_text(tabular(rows, "l" + "c" * len(cols), header=[[""] + [f"({i+1})" for i in range(len(cols))], [""] + cols],
                                            midrules=[2 * K - 1], note=note), encoding="utf-8")
set_macros({"hausmanP": f"{pH:.3f}", "cdResidFE": f"{cd_fe:.1f}", "cdResidTWFE": f"{cd_tw:.1f}"})

# =============================== Part B: long-run estimators ===============================
est = {}
est["FMOLS-GM"] = E.fmols_gm(Y, X, M=2)
est["DOLS pooled"] = E.dols_pooled(Y, X, 1, 1)
est["DOLS-GM"] = E.dols_gm(Y, X, 0, 0)
est["PMG"] = E.pmg(Yd, Xd)
est["MG"] = E.mg_ecm(Yd, Xd)
est["CCE-MG"] = E.cce_mg(Y, X)
est["CCE-P"] = E.cce_pooled(Y, X)
est["CS-DL"] = E.csdl_mg(Y, X, 1)
yD, ZD, gD = E.dols_pooled(Y, X, 1, 1, return_design=True)
wcb_dols = E.wild_cluster_p(yD, ZD, gD, J_INET, B=B_WCB, seed=4, absorb=[gD])
print("pooled DOLS: WCB p", wcb_dols[0], " t(G-1) p", wcb_dols[2])
set_macros({"wcbDolsInet": f"{wcb_dols[0]:.2f}", "tgDolsInet": f"{wcb_dols[2]:.2f}"})
h_all = E.hausman_mg_pmg(est["MG"], est["PMG"])
h_inet = E.hausman_mg_pmg(est["MG"], est["PMG"], J_INET)
print("Hausman MG vs PMG: joint", h_all, "internet", h_inet)

order = ["FMOLS-GM", "DOLS pooled", "DOLS-GM", "PMG", "MG", "CCE-MG", "CCE-P", "CS-DL"]
rows = []
for j, v in enumerate(X_BASE):
    r1, r2 = [lab[v]], [""]
    for c in order:
        a, s = cell(est[c]["b"][j], est[c]["se"][j])
        r1.append(a); r2.append(s)
    rows += [r1, r2]
rows.append(["Slope heterogeneity"] + ["Het.", "Hom.", "Het.", "Hom.", "Het.", "Het.", "Hom.", "Het."])
rows.append(["Common factors"] + ["Year means", "Year means", "Year means", "Year means", "Year means", "CCE", "CCE", "CCE"])
rows.append(["Speed of adjustment $\\bar\\phi$"] + ["--", "--", "--",
            f"{est['PMG']['phi']:.3f} ({est['PMG']['phi_se']:.3f})", f"{est['MG']['phi']:.3f} ({est['MG']['phi_se']:.3f})", "--", "--", "--"])
rows.append(["Wild-cluster bootstrap $p$, Internet"] + ["--", f"{fp(wcb_dols[0])}"] + ["--"] * (len(order) - 2))
rows.append(["Observations"] + [str(N * T)] * len(order))
note = ("Dependent variable: ln GDP per capita; $N=" + str(N) + "$ countries, $T=" + str(T) + "$ years. FMOLS: group-mean fully modified OLS (Pedroni 2000), Bartlett bandwidth 2. "
        "DOLS pooled: Kao--Chiang with one lead and one lag of the differenced regressors, country-clustered s.e. DOLS-GM (0,0): group-mean DOLS with contemporaneous differences only (no leads or lags; see Table~\\ref{tab:specs} for the versions with leads and lags). "
        "For pooled DOLS the last-but-one row gives the wild-cluster bootstrap $p$-value (Webb weights, 999 draws) of the Internet slope. "
        "PMG/MG: ARDL(1,1,1,1,1) in error-correction form (long-run coefficients $\\theta$; PMG imposes common $\\theta$, MG averages country estimates); the PMG standard errors "
        "come from the concentrated likelihood. CCE-MG/CCE-P: common correlated effects (Pesaran 2006) in levels; CS-DL: cross-sectionally augmented distributed lag "
        "(Chudik et al.\\ 2016). Group-mean s.e.\\ are nonparametric. Estimators marked ``Year means'' use data net of cross-sectional means. "
        f"Hausman MG vs.\\ PMG: joint $\\chi^2({K})={h_all['h']:.1f}$ ($p={h_all['p']:.3f}$). *, **, *** denote significance at 10\\%, 5\\%, 1\\%.")
(TAB / "tab_longrun.tex").write_text(tabular(rows, "l" + "c" * len(order),
    header=[[""] + [f"({i+1})" for i in range(len(order))], [""] + [("DOLS--GM (0,0)" if c == "DOLS-GM" else c.replace("-", "--")) for c in order]],
    midrules=[2 * K - 1], note=note, resize=True), encoding="utf-8")

# store internet coefficient across estimators
rec = []
for c, r in res.items():
    rec.append({"estimator": c, "family": "static", "b": float(r.params["inet_s"]), "se": float(r.std_errors["inet_s"])})
for c in order:
    rec.append({"estimator": c, "family": "long-run", "b": float(est[c]["b"][J_INET]), "se": float(est[c]["se"][J_INET])})
inet_df = pd.DataFrame(rec)
inet_df.to_csv(TAB / "estimates_inet.csv", index=False)
print(inet_df.round(3).to_string())

set_macros({"pmgInet": f"{est['PMG']['b'][J_INET]:.3f}", "pmgInetSe": f"{est['PMG']['se'][J_INET]:.3f}",
            "pmgPhi": f"{est['PMG']['phi']:.3f}", "pmgPhiSe": f"{est['PMG']['phi_se']:.3f}",
            "hausmanMgPmgP": f"{h_all['p']:.3f}", "hausmanMgPmgInetP": f"{h_inet['p']:.3f}",
            "fmolsInet": f"{est['FMOLS-GM']['b'][J_INET]:.3f}", "fmolsInetSe": f"{est['FMOLS-GM']['se'][J_INET]:.3f}",
            "dolsInet": f"{est['DOLS pooled']['b'][J_INET]:.3f}", "dolsInetSe": f"{est['DOLS pooled']['se'][J_INET]:.3f}",
            "cceInet": f"{est['CCE-MG']['b'][J_INET]:.3f}", "cceInetSe": f"{est['CCE-MG']['se'][J_INET]:.3f}",
            "ccepInet": f"{est['CCE-P']['b'][J_INET]:.3f}", "ccepInetSe": f"{est['CCE-P']['se'][J_INET]:.3f}",
            "twfeInet": f"{float(res['Two-way FE'].params['inet_s']):.3f}",
            "feInet": f"{float(res['Fixed effects'].params['inet_s']):.3f}",
            "csdlInet": f"{est['CS-DL']['b'][J_INET]:.3f}", "csdlInetSe": f"{est['CS-DL']['se'][J_INET]:.3f}"})

# ---- PMG detail table: country-level adjustment speed ----
phi = est["PMG"]["phi_units"]
rows = [[LATAM[i], f"{p:.3f}"] for i, p in zip(ids, phi)]
(TAB / "tab_pmg_detail.tex").write_text(tabular(rows, "lr", header=[["Country", "$\\hat\\phi_i$"]]), encoding="utf-8")
set_macros({"nPhiNeg": int((phi < 0).sum())})

# ---- Figure: forest plot of the Internet coefficient ----
fd = inet_df.copy()
fd["lo"], fd["hi"] = fd.b - 1.96 * fd.se, fd.b + 1.96 * fd.se
fig, ax = plt.subplots(figsize=(6.2, 4.3))
yy = np.arange(len(fd))[::-1]
colors = [ps.GREY if f == "static" else ps.BLUE for f in fd.family]
for y_, (_, r), c in zip(yy, fd.iterrows(), colors):
    ax.plot([r.lo, r.hi], [y_, y_], color=c, lw=1.6)
    ax.plot(r.b, y_, "o", color=c, ms=4.5)
ax.axvline(0, color=ps.DARK, lw=0.8)
ax.set_yticks(yy); ax.set_yticklabels([("DOLS-GM (0,0)" if e_ == "DOLS-GM" else e_) for e_ in fd.estimator])
ax.set_xlabel("Effect on ln GDP per capita of moving Internet use from 0% to 100% (95% CI)")
ax.plot([], [], "o-", color=ps.GREY, label="Static benchmarks (not CSD-robust)")
ax.plot([], [], "o-", color=ps.BLUE, label="Long-run / dynamic estimators")
ax.legend(loc="lower right")
ps.save(fig, FIG / "fig_forest_inet.pdf")

# ---- Figure: country-level long-run Internet coefficients (FMOLS, CCE-MG) ----
fig, axs = plt.subplots(1, 2, figsize=(7.0, 4.0), sharey=True)
for ax, nm in zip(axs, ["FMOLS-GM", "CCE-MG"]):
    u = est[nm]["units"][:, J_INET]
    o = np.argsort(u)
    ax.scatter(u[o], range(N), color=ps.BLUE, s=16, zorder=3)
    ax.axvline(0, color=ps.DARK, lw=0.7)
    ax.axvline(est[nm]["b"][J_INET], color=ps.ORANGE, lw=1.4, label=f"Group mean = {est[nm]['b'][J_INET]:.2f}")
    ax.set_yticks(range(N)); ax.set_yticklabels([ids[i] for i in o], fontsize=7)
    ax.set_title(nm.replace("-", "--"))
    ax.set_xlabel("Country-specific long-run coefficient")
    ax.legend(loc="lower right")
ps.save(fig, FIG / "fig_country_lr.pdf")

# =============================== Part C: growth models ===============================
g = df.sort_values(["iso3", "year"]).copy()
for v in ["ln_gdppc", "ln_gfcf", "ln_hc", "ln_trade", "inet_s"]:
    g["d_" + v] = g.groupby("iso3")[v].diff()
for l in (1, 2, 3):
    g[f"d_inet_s_l{l}"] = g.groupby("iso3")["d_inet_s"].shift(l)
    g[f"d_ln_gdppc_l{l}"] = g.groupby("iso3")["d_ln_gdppc"].shift(l)
g = g.dropna(subset=["d_ln_gdppc", "d_inet_s"])


def fe_cluster(data, y, xs, fe=("iso3", "year"), cluster="iso3"):
    d = data.dropna(subset=[y] + xs).copy()
    parts = [d[xs]]
    for f in fe:
        parts.append(pd.get_dummies(d[f], prefix=f, drop_first=True, dtype=float))
    D = sm.add_constant(pd.concat(parts, axis=1))
    m = sm.OLS(d[y], D).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d[cluster])[0]})
    return m, d


specs = {
    "(1)": ["d_inet_s"],
    "(2)": ["d_inet_s", "d_ln_gfcf", "d_ln_hc", "d_ln_trade"],
    "(3)": ["d_inet_s", "d_inet_s_l1", "d_inet_s_l2", "d_ln_gfcf", "d_ln_hc", "d_ln_trade"],
    "(4)": ["d_inet_s", "d_inet_s_l1", "d_inet_s_l2", "d_ln_gdppc_l1", "d_ln_gdppc_l2", "d_ln_gfcf", "d_ln_hc", "d_ln_trade"],
}
gm = {k: fe_cluster(g, "d_ln_gdppc", v) for k, v in specs.items()}


def design_cl(data, y, xs):
    d = data.dropna(subset=[y] + xs).copy()
    D = sm.add_constant(pd.concat([d[xs]] + [pd.get_dummies(d[f], prefix=f, drop_first=True, dtype=float) for f in ("iso3", "year")], axis=1)).to_numpy(float)
    return d[y].to_numpy(float), D, pd.factorize(d["iso3"])[0]


gwcb = {k: E.wild_cluster_p(*design_cl(g, "d_ln_gdppc", v)[:2], design_cl(g, "d_ln_gdppc", v)[2], 1, B=B_WCB, seed=10 + i)
        for i, (k, v) in enumerate(specs.items())}
gl = {"d_inet_s": "$\\Delta$ Internet users (share)", "d_inet_s_l1": "$\\Delta$ Internet users$_{t-1}$",
      "d_inet_s_l2": "$\\Delta$ Internet users$_{t-2}$", "d_ln_gdppc_l1": "$\\Delta$ ln GDPpc$_{t-1}$",
      "d_ln_gdppc_l2": "$\\Delta$ ln GDPpc$_{t-2}$", "d_ln_gfcf": "$\\Delta$ ln GFCF", "d_ln_hc": "$\\Delta$ ln schooling",
      "d_ln_trade": "$\\Delta$ ln trade"}
rows = []
for v in gl:
    r1, r2 = [gl[v]], [""]
    for k, (m, d) in gm.items():
        if v in m.params.index:
            a, s = cell(m.params[v], m.bse[v]); r1.append(a); r2.append(s)
        else:
            r1.append(""); r2.append("")
    rows += [r1, r2]
cum = []
for k, (m, d) in gm.items():
    names = [n for n in ("d_inet_s", "d_inet_s_l1", "d_inet_s_l2") if n in m.params.index]
    if len(names) > 1:
        w = np.zeros(len(m.params));
        for n in names: w[list(m.params.index).index(n)] = 1
        tt = m.t_test(w)
        b, se = float(tt.effect[0]), float(tt.sd[0][0])
        cum.append("%.3f (%.3f)" % (b, se))
    else:
        cum.append("--")
rows.append(["Sum of Internet coefficients"] + cum)
rows.append(["Country and year effects"] + ["Yes"] * 4)
rows.append(["$t(G-1)$ $p$, impact coefficient"] + [f"{fp(gwcb[k][2])}" for k in specs])
rows.append(["Wild-cluster bootstrap $p$, impact coefficient"] + [f"{fp(gwcb[k][0])}" for k in specs])
rows.append(["Observations"] + [str(int(m.nobs)) for m, d in gm.values()])
note = ("Dependent variable: annual log growth of GDP per capita ($\\Delta$ ln GDPpc). Country and year fixed effects included; standard errors clustered by country "
        "in parentheses. The Internet variable is the annual change in the share of the population using the Internet, so a coefficient of 0.1 means that a 10-point "
        "increase in the share raises growth by 1 percentage point. *, **, *** denote significance at 10\\%, 5\\%, 1\\% (normal approximation; with 17 clusters "
        "these levels are indicative); the last two rows before the number of observations give the $p$-value of the impact coefficient from a $t(G-1)$ distribution and from a wild-cluster "
        "restricted bootstrap-$t$ (Webb weights, 999 draws).")
(TAB / "tab_growth.tex").write_text(tabular(rows, "l" + "c" * len(specs), header=[[""] + list(specs)], midrules=[2 * len(gl) - 1], note=note), encoding="utf-8")
m2 = gm["(2)"][0]
set_macros({"growthCoef": f"{m2.params['d_inet_s']:.3f}", "growthSe": f"{m2.bse['d_inet_s']:.3f}",
            "growthP": f"{m2.pvalues['d_inet_s']:.3f}", "wcbGrowthTwo": f"{gwcb['(2)'][0]:.2f}", "wcbGrowthThree": f"{gwcb['(3)'][0]:.2f}"})
print("growth spec (2):", m2.params["d_inet_s"], m2.bse["d_inet_s"], m2.pvalues["d_inet_s"])

# ---- Local projections: cumulative response of GDPpc to a change in Internet adoption ----
H_MAX = 6
lp = df.sort_values(["iso3", "year"]).copy()
for v in ["ln_gfcf", "ln_hc", "ln_trade", "inet_s", "ln_gdppc"]:
    lp["d_" + v] = lp.groupby("iso3")[v].diff()
for l in (1, 2):
    lp[f"d_inet_s_l{l}"] = lp.groupby("iso3")["d_inet_s"].shift(l)
    lp[f"d_ln_gdppc_l{l}"] = lp.groupby("iso3")["d_ln_gdppc"].shift(l)
    lp[f"d_ln_gfcf_l{l}"] = lp.groupby("iso3")["d_ln_gfcf"].shift(l)
ctrl = ["d_inet_s_l1", "d_inet_s_l2", "d_ln_gdppc_l1", "d_ln_gdppc_l2", "d_ln_gfcf", "d_ln_hc", "d_ln_trade"]


def iv_lp(data, y, x, z, ctrl):
    """Local-projection IV (Ramey & Zubairy 2018): cumulative y response per unit of cumulative x response, instrument = the
    impact change z.  Country and year effects and controls partialled out; cluster-robust (country) variance computed
    for the ratio directly (so the confidence band of the multiplier is a joint one, not a ratio of two bands).
    Returns (beta, se, first-stage F)."""
    d = data.dropna(subset=[y, x, z] + ctrl).copy()
    FE = pd.concat([pd.get_dummies(d[f], prefix=f, drop_first=True, dtype=float) for f in ("iso3", "year")], axis=1)
    W = np.column_stack([np.ones(len(d)), d[ctrl].to_numpy(), FE.to_numpy()])
    res = lambda v: v - W @ np.linalg.lstsq(W, v, rcond=None)[0]
    yt, xt, zt = res(d[y].to_numpy()), res(d[x].to_numpy()), res(d[z].to_numpy())
    beta = (zt @ yt) / (zt @ xt)
    u = yt - beta * xt
    cl = pd.factorize(d["iso3"])[0]
    G, n, k = len(set(cl)), len(d), np.linalg.matrix_rank(W) + 1
    meat = sum((zt[cl == g_] @ u[cl == g_]) ** 2 for g_ in range(G))
    V = meat / (zt @ xt) ** 2 * G / (G - 1) * (n - 1) / (n - k)
    pi = (zt @ xt) / (zt @ zt)
    vres = xt - pi * zt
    Vpi = sum((zt[cl == g_] @ vres[cl == g_]) ** 2 for g_ in range(G)) / (zt @ zt) ** 2 * G / (G - 1) * (n - 1) / (n - k)
    return beta, np.sqrt(V), pi ** 2 / Vpi


irf = []
for h in range(0, H_MAX + 1):
    lp["y_h"] = lp.groupby("iso3")["ln_gdppc"].shift(-h) - lp.groupby("iso3")["ln_gdppc"].shift(1)
    lp["x_h"] = lp.groupby("iso3")["inet_s"].shift(-h) - lp.groupby("iso3")["inet_s"].shift(1)
    m, d = fe_cluster(lp, "y_h", ["d_inet_s"] + ctrl)
    mx, dx_ = fe_cluster(lp, "x_h", ["d_inet_s"] + ctrl)
    mult, mult_se, fsF = iv_lp(lp, "y_h", "x_h", "d_inet_s", ctrl)
    yh_, Dh_, gh_ = design_cl(lp, "y_h", ["d_inet_s"] + ctrl)
    pw_, t_rf, pt_ = E.wild_cluster_p(yh_, Dh_, gh_, 1, B=B_WCB, seed=30 + h)
    irf.append({"h": h, "b": m.params["d_inet_s"], "se": m.bse["d_inet_s"], "n": int(m.nobs),
                "bx": mx.params["d_inet_s"], "sex": mx.bse["d_inet_s"], "mult": mult, "mult_se": mult_se, "fsF": fsF, "p_wcb": pw_, "p_tg": pt_})
irf = pd.DataFrame(irf)
irf.to_csv(TAB / "irf_lp.csv", index=False)
TC = float(stats.t.ppf(0.975, 16))                 # t(G-1) critical value, G = 17 clusters
lp_rows = [[str(int(r_.h)), f"{r_.b:.3f} ({r_.se:.3f})", f"{r_.bx:.3f} ({r_.sex:.3f})", f"{r_.mult:.3f} ({r_.mult_se:.3f})", f"{fp(r_.p_tg)}", f"{fp(r_.p_wcb)}"]
           for r_ in irf.itertuples()]
lp_note = ("Local projections with country and year effects and the controls of equation~\\eqref{eq:lp}; standard errors clustered by country. "
           "Multiplier = cumulative response of ln GDPpc per unit of cumulative response of the Internet share, estimated by LP-IV with the impact change in adoption as the instrument "
           "(joint standard error). The last two columns give $p$-values of H$_0$: multiplier $=0$ (equivalently, no response of GDPpc to the impact change) from a $t(G-1)$ distribution "
           f"and from a wild-cluster restricted bootstrap-$t$ with Webb weights ({B_WCB} draws). Confidence bands in the figure use $t_{{16}}$ critical values ({TC:.2f}).")
(TAB / "tab_lp.tex").write_text(tabular(lp_rows, "lccccc", header=[["$h$", "$\\ln$ GDPpc response", "Internet response", "Multiplier (LP-IV)", "$t(G-1)$ $p$", "WCB $p$"]], note=lp_note, resize=True), encoding="utf-8")
print(irf.round(3).to_string())
fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.0))
ax = axs[0]
ax.fill_between(irf.h, irf.b - TC * irf.se, irf.b + TC * irf.se, color=ps.BLUE, alpha=0.18)
ax.plot(irf.h, irf.b, "o-", color=ps.BLUE, ms=4)
ax.axhline(0, color=ps.DARK, lw=0.8)
ax.set_xlabel("Years after the shock ($h$)")
ax.set_ylabel("Cumulative response of ln GDPpc")
ax.set_title("(a) GDP per capita", fontsize=8.5)
ax = axs[1]
ax.fill_between(irf.h, irf.bx - TC * irf.sex, irf.bx + TC * irf.sex, color=ps.ORANGE, alpha=0.18)
ax.plot(irf.h, irf.bx, "o-", color=ps.ORANGE, ms=4)
ax.axhline(0, color=ps.DARK, lw=0.8)
ax.set_xlabel("Years after the shock ($h$)")
ax.set_ylabel("Cumulative response of Internet share")
ax.set_title("(b) Internet-use share", fontsize=8.5)
ax = axs[2]
ax.fill_between(irf.h, irf.mult - TC * irf.mult_se, irf.mult + TC * irf.mult_se, color=ps.GREEN, alpha=0.18)
ax.plot(irf.h, irf.mult, "o-", color=ps.GREEN, ms=4)
ax.axhline(0, color=ps.DARK, lw=0.8)
ax.set_xlabel("Horizon $h$")
ax.set_ylabel("Implied semi-elasticity (LP-IV)")
ax.set_title("(c) Multiplier with joint 95% band", fontsize=8.5)
fig.tight_layout()
ps.save(fig, FIG / "fig_irf.pdf")
def at(h, c): return irf.loc[irf.h == h, c].iloc[0]
set_macros({"irfFour": f"{at(4, 'b'):.3f}", "irfFourSe": f"{at(4, 'se'):.3f}", "irfSix": f"{at(6, 'b'):.3f}",
            "irfOne": f"{at(1, 'b'):.3f}", "irfOneSe": f"{at(1, 'se'):.3f}",
            "irfXFour": f"{at(4, 'bx'):.3f}", "irfXSix": f"{at(6, 'bx'):.3f}",
            "irfMultFour": f"{at(4, 'mult'):.2f}", "irfMultFourSe": f"{at(4, 'mult_se'):.2f}",
            "irfMultSix": f"{at(6, 'mult'):.2f}", "irfMultSixSe": f"{at(6, 'mult_se'):.2f}",
            "lpWcbTwo": f"{at(2, 'p_wcb'):.2f}", "lpWcbFour": f"{at(4, 'p_wcb'):.2f}", "lpWcbSix": f"{at(6, 'p_wcb'):.2f}",
            "lpTgFour": f"{at(4, 'p_tg'):.3f}", "tCritG": f"{TC:.2f}", "irfMultTwo": f"{at(2, 'mult'):.2f}", "irfMultTwoSe": f"{at(2, 'mult_se'):.2f}"})

# =============================== Multiple optima of the PMG likelihood ===============================
rows = []
for mi, (f_, x_) in enumerate(est["PMG"]["modes"]):
    rows.append([("Global maximum" if mi == 0 else f"Local optimum {mi}"), f"{-f_:.1f}"] + [f"{v:.3f}" for v in x_])
note = ("Distinct optima of the concentrated PMG log-likelihood found from six starting values (median and mean of the unit MG estimates, zero, static pooled OLS and random perturbations), "
        "ranked by the value of the log-likelihood (data net of year means). The global maximum is the PMG estimate reported in Table~\\ref{tab:longrun}; "
        "a single run of a standard routine started from the median of the unit MG estimates ends at the first local optimum.")
(TAB / "tab_pmg_modes.tex").write_text(tabular(rows, "lrrrrr", header=[["", "Log-lik.", "ln GFCF", "ln schooling", "Internet", "ln trade"]], note=note), encoding="utf-8")
set_macros({"pmgModes": est["PMG"]["n_modes"], "pmgLocalInet": f"{est['PMG']['modes'][1][1][J_INET]:.3f}" if est["PMG"]["n_modes"] > 1 else "NA",
            "pmgGlobalHc": f"{est['PMG']['b'][1]:.2f}", "pmgGlobalGfcf": f"{est['PMG']['b'][0]:.2f}"})

# =============================== Specification sensitivity of the dynamic augmentations ===============================
def dols_df(lead, lag):
    return (T - lag - 1 - lead) - (1 + K + K * (lead + lag + 1))

specs_rows = []
def add_spec(label, r, dfv):
    j = J_INET
    a_, s_ = cell(r["b"][j], r["se"][j])
    specs_rows.append([label, a_ + " " + s_, dfv if isinstance(dfv, str) else str(dfv)])
add_spec("CCE-MG, static averages (baseline)", E.cce_mg(Y, X, 0), E.cce_mg(Y, X, 0)["df"])
add_spec("CCE-MG, 1 lag of the averages", E.cce_mg(Y, X, 1), E.cce_mg(Y, X, 1)["df"])
r2_ = E.cce_mg(Y, X, 2)
add_spec("CCE-MG, $p_T=\\lfloor T^{1/3}\\rfloor=2$ lags of the averages", r2_, r2_["df"])
add_spec("CS-DL, $p_\\Delta=1$, contemporaneous averages (baseline)", E.csdl_mg(Y, X, 1, 0), E.csdl_mg(Y, X, 1, 0)["df"])
r_ = E.csdl_mg(Y, X, 1, 1)
add_spec("CS-DL, $p_\\Delta=1$, 1 lag of the averages", r_, r_["df"])
specs_rows.append(["CS-DL, $p_\\Delta=1$, $p_T=2$ lags of the averages", "infeasible", str(E.csdl_mg(Y, X, 1, 2)["df"])])
for (ld, lg, lab) in [(0, 0, "DOLS-GM, contemporaneous $\\Delta x$ only (baseline)"), (1, 0, "DOLS-GM, one lead"), (1, 1, "DOLS-GM, one lead and one lag")]:
    add_spec(lab, E.dols_gm(Y, X, ld, lg), dols_df(ld, lg))
for (ld, lg, lab) in [(0, 0, "Pooled DOLS, contemporaneous $\\Delta x$ only"), (1, 1, "Pooled DOLS, one lead and one lag (baseline)"), (2, 2, "Pooled DOLS, two leads and two lags")]:
    r_ = E.dols_pooled(Y, X, ld, lg)
    specs_rows.append([lab, "%.3f%s (%.3f)" % (r_["b"][J_INET], stars(r_["p"][J_INET]), r_["se"][J_INET]), "pooled"])
for M_ in (1, 2, 3, 4):
    r_ = E.fmols_gm(Y, X, M=M_)
    add_spec(f"FMOLS-GM, Bartlett bandwidth {M_}" + (" (baseline)" if M_ == 2 else ""), r_, "--")
note = ("Slope on the Internet-use share. Residual degrees of freedom of the unit regressions in the last column ($T=24$, $K=4$). Adding lags of the cross-sectional averages (CCE, CS-DL) "
        "or leads and lags of the differenced regressors (DOLS-GM) leaves only 2--9 residual degrees of freedom per country. Group-mean standard errors are nonparametric; "
        "pooled DOLS standard errors are clustered by country.")
(TAB / "tab_specs.tex").write_text(tabular(specs_rows, "lcc", header=[["Specification", "Internet slope", "d.f. per unit"]], note=note), encoding="utf-8")
set_macros({"cceLoneDf": E.cce_mg(Y, X, 1)["df"], "cceLoneInet": f"{E.cce_mg(Y, X, 1)['b'][J_INET]:.2f}", "cceLtwoInet": f"{r2_['b'][J_INET]:.2f}", "cceLtwoDf": r2_["df"],
            "dolsgmFullInet": f"{E.dols_gm(Y, X, 1, 1)['b'][J_INET]:.2f}", "dolsgmFullDf": dols_df(1, 1)})

print("estimation done")
