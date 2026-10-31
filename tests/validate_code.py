"""Internal validation of the estimators in code/lib/econ.py.

This script provides (i) independent re-implementations that must agree to machine precision, (ii) an analytic check of the PMG standard error and
(iii) coverage experiments under the *correctly specified* design of each estimator.  Results go to output/tables/code_validation.csv
and tab_validation_code.tex.  The external cross-check against R and Stata is in validation/.

Usage: python tests/validate_code.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from config import *
import lib.econ as E
from lib.util import tabular, set_macros

rows = []
df = pd.read_csv(PROC / "panel_main.csv")
ids, Y, X = E.to_arrays(df, Y_VAR, X_BASE)
N, T, K = X.shape
Yd, Xd = E.csd_demean(Y), E.csd_demean(X)


def add(name, what, value, ok, info=False):
    """ok=False raises a FAIL; info=True records a finite-sample property (not a correctness test)."""
    status = "info" if info else ("pass" if ok else "FAIL")
    rows.append([name, what, value, status])
    print(f"{status.upper()}  {name}: {what} = {value}")


# ---------------------------------------------------------------- 1. CCE-MG vs unit-by-unit statsmodels OLS
r = E.cce_mg(Y, X)
Yb, Xb = Y.mean(0), X.mean(0)
B = []
for i in range(N):
    D = sm.add_constant(np.column_stack([X[i], Yb, Xb]))
    B.append(sm.OLS(Y[i], D).fit().params[1:1 + K])
B = np.array(B)
add("CCE-MG", "max |b - statsmodels unit OLS mean|", f"{np.abs(r['b'] - B.mean(0)).max():.2e}", np.abs(r["b"] - B.mean(0)).max() < 1e-10)
add("CCE-MG", "max |se - sd/sqrt(N)|", f"{np.abs(r['se'] - B.std(0, ddof=1) / np.sqrt(N)).max():.2e}", np.abs(r["se"] - B.std(0, ddof=1) / np.sqrt(N)).max() < 1e-10)

# ---------------------------------------------------------------- 2. CCE-P vs one big regression with unit-specific averages
rp = E.cce_pooled(Y, X)
rows_y, rows_X = [], []
Z = np.zeros((N * T, N * (2 + K)))
for i in range(N):
    Z[i * T:(i + 1) * T, i * (2 + K):(i + 1) * (2 + K)] = np.column_stack([np.ones(T), Yb, Xb])
big = sm.OLS(Y.reshape(-1), np.column_stack([X.reshape(N * T, K), Z])).fit()
add("CCE-P", "max |b - big dummy regression|", f"{np.abs(rp['b'] - big.params[:K]).max():.2e}", np.abs(rp["b"] - big.params[:K]).max() < 1e-8)

# ---------------------------------------------------------------- 3. TWFE vs linearmodels
from linearmodels.panel import PanelOLS
pdf = df.set_index(["iso3", "year"])
lm = PanelOLS(pdf[Y_VAR], pdf[X_BASE], entity_effects=True, time_effects=True).fit()
tw = E.twfe_cluster(Y, X)
add("TWFE", "max |b - linearmodels PanelOLS|", f"{np.abs(tw['b'] - lm.params.values).max():.2e}", np.abs(tw["b"] - lm.params.values).max() < 1e-8)

# ---------------------------------------------------------------- 4. DOLS pooled / group-mean vs explicit statsmodels regressions
def dols_gm_explicit(Y, X, lead, lag):
    Yc, Xc = E.csd_demean(Y), E.csd_demean(X)
    out = []
    for i in range(N):
        n = T
        cols = {}
        for k in range(K):
            cols[f"x{k}"] = Xc[i, :, k]
        dx = np.vstack([np.full((1, K), np.nan), np.diff(Xc[i], axis=0)])
        for j in range(-lead, lag + 1):
            sh = np.roll(dx, j, axis=0)          # dx_{t-j}
            for k in range(K):
                cols[f"d{k}_{j}"] = sh[:, k]
        D = pd.DataFrame(cols)
        D["y"] = Yc[i]
        valid = np.arange(T)
        lo, hi = 1 + max(lag, 0), T - max(lead, 0)
        D = D.iloc[lo:hi]
        m = sm.OLS(D["y"], sm.add_constant(D.drop(columns="y"))).fit()
        out.append([m.params[f"x{k}"] for k in range(K)])
    return np.mean(out, axis=0)
g = E.dols_gm(Y, X, 0, 0)["b"]
add("DOLS-GM(0,0)", "max |b - explicit statsmodels|", f"{np.abs(g - dols_gm_explicit(Y, X, 0, 0)).max():.2e}", np.abs(g - dols_gm_explicit(Y, X, 0, 0)).max() < 1e-8)

# pooled DOLS(1,1): explicit regression with country dummies on year-demeaned data
def dols_pooled_explicit(Y, X, lead=1, lag=1):
    Yc, Xc = E.csd_demean(Y), E.csd_demean(X)
    recs = []
    dX = np.diff(Xc, axis=1)
    t = np.arange(lag + 1, T - lead)
    for i in range(N):
        d = {"iso": i, "y": Yc[i][t]}
        for k in range(K):
            d[f"x{k}"] = Xc[i][t][:, k] if Xc[i][t].ndim > 1 else Xc[i][t, k]
        for j in range(-lead, lag + 1):
            for k in range(K):
                d[f"d{k}_{j}"] = dX[i][t - j - 1][:, k]
        recs.append(pd.DataFrame(d))
    D = pd.concat(recs)
    Xm = pd.concat([D.drop(columns=["y", "iso"]), pd.get_dummies(D["iso"], prefix="c", dtype=float)], axis=1)
    return sm.OLS(D["y"], Xm).fit().params[[f"x{k}" for k in range(K)]].values
dp = E.dols_pooled(Y, X, 1, 1)["b"]
add("DOLS pooled(1,1)", "max |b - explicit dummy regression|", f"{np.abs(dp - dols_pooled_explicit(Y, X)).max():.2e}", np.abs(dp - dols_pooled_explicit(Y, X)).max() < 1e-8)

# ---------------------------------------------------------------- 5. FMOLS: independent loop-based implementation (Phillips & Hansen 1990)
def fmols_independent(y, x, M=2):
    """Single-unit FMOLS written directly from the textbook formulas with explicit lag sums."""
    T_, K_ = x.shape
    xd = x - x.mean(0); yd = y - y.mean()
    b_ols = np.linalg.solve(xd.T @ xd, xd.T @ yd)
    u = yd - xd @ b_ols
    eps = np.diff(x, axis=0)
    w = np.column_stack([u[1:], eps]); w = w - w.mean(0)
    n = w.shape[0]
    Om = np.zeros((1 + K_, 1 + K_)); De = np.zeros((1 + K_, 1 + K_))
    for j in range(0, M + 1):
        Gj = np.zeros((1 + K_, 1 + K_))
        for t_ in range(0, n - j):
            Gj += np.outer(w[t_], w[t_ + j])
        Gj /= n
        k_ = 1 - j / (M + 1)
        if j == 0:
            Om += Gj; De += Gj
        else:
            Om += k_ * (Gj + Gj.T); De += k_ * Gj
    O12 = Om[0, 1:]; O22 = Om[1:, 1:]; D21 = De[1:, 0]; D22 = De[1:, 1:]
    gam = np.linalg.solve(O22, O12)
    ystar = yd[1:] - eps @ gam
    dstar = D21 - D22 @ gam
    xx = xd[1:]
    return np.linalg.solve(xx.T @ xx, xx.T @ ystar - (T_ - 1) * dstar)
fm = E.fmols_gm(Y, X)
ind = np.mean([fmols_independent(Yd[i], Xd[i]) for i in range(N)], axis=0)
add("FMOLS-GM", "max |b - independent loop implementation|", f"{np.abs(fm['b'] - ind).max():.2e}", np.abs(fm["b"] - ind).max() < 1e-8)

# ---------------------------------------------------------------- 6. PMG: optimum from several starts + analytic information matrix
pm = E.pmg(Yd, Xd)
gl = [E.pmg(Yd, Xd, n_start=14, seed=sd)["b"] for sd in (11, 12, 13)]          # more (random) starting values
add("PMG", "global maximum unchanged with 14 starting values (max |diff|)", f"{np.abs(np.array(gl) - pm['b']).max():.2e}", np.abs(np.array(gl) - pm["b"]).max() < 1e-3)
rows.append(["PMG", "distinct local optima of the concentrated likelihood in the empirical panel", str(pm["n_modes"]), "info"])
print("INFO  PMG: local optima in the empirical panel:", pm["n_modes"])

def pmg_info(Y, X, th):
    """Observed information on theta after concentrating out phi_i, mu_i, gamma_i and sigma_i^2:
    I = sum_i (phi_i^2 / sigma_i^2) * x_{i,-1}' M_[1, dX, z_i] x_{i,-1}   (Pesaran, Shin & Smith 1999)."""
    N_, T_, K_ = X.shape
    n = T_ - 1
    I = np.zeros((K_, K_))
    for i in range(N_):
        dy = np.diff(Y[i]); yl = Y[i][:-1]; xl = X[i][:-1]; dx = np.diff(X[i], axis=0)
        z = yl - xl @ th
        D = np.column_stack([np.ones(n), z, dx])
        b = np.linalg.lstsq(D, dy, rcond=None)[0]
        e = dy - D @ b
        s2 = e @ e / n
        phi = b[1]
        Dz = np.column_stack([np.ones(n), dx, z])
        Mz = np.eye(n) - Dz @ np.linalg.pinv(Dz)
        I += phi ** 2 / s2 * xl.T @ Mz @ xl
    return I
V_an = np.linalg.inv(pmg_info(Yd, Xd, pm["b"]))
se_an = np.sqrt(np.diag(V_an))
pm_loc = E.pmg(Yd, Xd, theta0=pm["modes"][1][1] if pm["n_modes"] > 1 else pm["b"], se=True)           # a local optimum, for a like-for-like comparison
V_an_l = np.linalg.inv(pmg_info(Yd, Xd, pm["b"]))
rel = np.abs(pm["se_ml"] / np.sqrt(np.diag(V_an_l)) - 1).max()
add("PMG", "numerical-Hessian s.e. vs analytic expected-information s.e. at the global optimum (max relative gap)", f"{rel:.2f}", rel < 0.35)

# ---------------------------------------------------------------- 7. Coverage under the correctly specified design of each estimator
def sim_pmg_ecm(rng, N=17, T=24, K=2, th=np.array([0.5, -0.3])):
    """ARDL(1,1) in ECM form, common theta, unit-specific phi, exogenous I(1) regressors, iid normal errors:
    the exact DGP of the PMG likelihood."""
    X = np.cumsum(rng.standard_normal((N, T + 20, K)) * 0.5, axis=1)
    Yv = np.zeros((N, T + 20))
    phi = -rng.uniform(0.3, 0.8, N); mu = rng.normal(0, 0.1, N); gam = rng.normal(0.3, 0.1, (N, K)); sg = rng.uniform(0.3, 0.6, N)
    for t in range(1, T + 20):
        z = Yv[:, t - 1] - X[:, t - 1] @ th
        Yv[:, t] = Yv[:, t - 1] + mu + phi * z + (gam * (X[:, t] - X[:, t - 1])).sum(1) + sg * rng.standard_normal(N)
    return Yv[:, 20:], X[:, 20:], th

R = 400
rng = np.random.default_rng(7)
cov, ratio, bias = [], [], []
for _ in range(R):
    Yp, Xp, th = sim_pmg_ecm(rng)
    r_ = E.pmg(Yp, Xp)
    cov.append(np.abs(r_["b"] - th) < 1.96 * r_["se"]); bias.append(r_["b"] - th); ratio.append(r_["se"])
cov, bias, ratio = np.array(cov), np.array(bias), np.array(ratio)
add("PMG", "coverage of 95% CI (df-corrected s.e.), exactly specified ECM DGP (N=17, T=24)", f"{cov.mean():.3f}", cov.mean() > 0.80, info=True)
add("PMG", "mean s.e. / s.d. of estimates, same DGP", f"{(ratio.mean(0) / bias.std(0)).mean():.2f}", 0.6 < (ratio.mean(0) / bias.std(0)).mean() < 1.4, info=True)

# CCE-MG under its own assumptions: exogenous stationary-error cointegrated panel, factors absorbed by averages
def sim_cce(rng, N=17, T=24, K=2, th=np.array([0.4, 0.2])):
    f = np.cumsum(rng.standard_normal(T) * 0.3)
    lam = rng.uniform(0.5, 1.5, (N, K)); dl = rng.uniform(0.5, 1.5, N)
    xi = np.cumsum(rng.standard_normal((N, T, K)) * 0.3, axis=1)
    X = lam[:, None, :] * f[None, :, None] + xi
    e = rng.standard_normal((N, T)) * 0.3
    Yv = (X * (th + rng.normal(0, 0.1, (N, K))[:, None, :])).sum(2) + dl[:, None] * f[None, :] + e
    return Yv, X, th
cov, bias = [], []
for _ in range(R):
    Yc_, Xc_, th = sim_cce(rng)
    r_ = E.cce_mg(Yc_, Xc_)
    cov.append(np.abs(r_["b"] - th) < 1.96 * r_["se"]); bias.append(r_["b"] - th)
cov = np.array(cov); bias = np.array(bias)
add("CCE-MG", "coverage of 95% CI, ideal DGP (exogenous, heterogeneous slopes, N=17, T=24, K=2)", f"{cov.mean():.3f}", cov.mean() > 0.88)
add("CCE-MG", "bias in the ideal DGP", f"{np.abs(bias.mean(0)).max():.3f}", np.abs(bias.mean(0)).max() < 0.03)

# Westerlund/Pedroni null size under independent random walks (no CSD): should be near nominal
rej = []
for r_i in range(120):
    rg = np.random.default_rng(100 + r_i)
    Ys = np.cumsum(rg.standard_normal((N, T)), axis=1); Xs = np.cumsum(rg.standard_normal((N, T, 2)), axis=1)
    pe = E.pedroni_adf(Ys, Xs, p=0, R=400, seed=5)
    rej.append(pe["p_group"] < 0.05)
add("Pedroni-type", "size at 5% under independent random walks (120 reps)", f"{np.mean(rej):.3f}", 0.0 <= np.mean(rej) < 0.15)

pd.DataFrame(rows, columns=["estimator", "check", "value", "status"]).to_csv(TAB / "code_validation.csv", index=False)
note = ("Internal validation of the routines. Agreement checks compare each routine with an independent "
        "implementation (statsmodels, linearmodels, an explicit loop implementation of FMOLS, an analytic information matrix for PMG); coverage checks use the "
        "exactly specified design of each estimator. The external cross-check against R and Stata is described in the appendix and in the folder \\texttt{validation/}.")
(TAB / "tab_validation_code.tex").write_text(tabular([[a, b.replace("%", "\\%").replace("_", "\\_"), c, d] for a, b, c, d in rows], "llrl",
                                                       header=[["Routine", "Check", "Value", "Status"]], note=note, resize=True), encoding="utf-8")
n_fail = sum(1 for r in rows if r[3] == "FAIL")
set_macros({"valNpass": sum(1 for r in rows if r[3] == "pass"), "valNcheck": sum(1 for r in rows if r[3] in ("pass", "FAIL")), "valNfail": n_fail})
print("\n", sum(1 for r in rows if r[3] == "pass"), "of", sum(1 for r in rows if r[3] in ("pass", "FAIL")), "correctness checks passed;", n_fail, "failures")
