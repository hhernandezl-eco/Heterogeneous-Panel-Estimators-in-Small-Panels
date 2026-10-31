"""Step 11 - Anatomy of the CCE regression in small panels.

CCE (Pesaran 2006) projects every unit regression on H_t = (1, ybar_t, xbar_t')' (K+2 columns).  Two finite-sample
objects determine its behaviour when N and T are small:

  (i)  degrees of freedom of a unit regression: T - (2K + 2)  (versus T - K - 1 for a plain unit regression);
  (ii) the retention ratio  rho_ik = x_ik' Mbar x_ik / x_ik' M_iota x_ik,  the share of the within-unit
       variation of regressor k that survives the projection on H.  The unit-level variance of the CCE slope
       is proportional to 1/(x' Mbar x), so 1/rho is the variance inflation relative to the plain regression.

We compute rho by simulation for K = 1..6 regressors (I(1) with a global trend and a stationary factor), N = 17
and T in {20, 24, 30, 50, 100}, and its counterpart for the empirical panel.

Outputs: tab_cce_retention.tex, fig_cce_retention.pdf, macros.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from config import *
import lib.econ as E
from lib import plotstyle as ps
from lib.util import tabular, set_macros

ps.setup()
N = 17
TS = [20, 24, 30, 50, 100]
KS = [1, 2, 3, 4, 5, 6]
R = 300
rng = np.random.default_rng(SEED + 11)


def retention(Y, X):
    """Mean over units and regressors of x'Mbar x / x'M_iota x (Y: N,T ; X: N,T,K)."""
    N_, T_, K_ = X.shape
    H = np.column_stack([np.ones(T_), Y.mean(0), X.mean(0)])
    Mb = np.eye(T_) - H @ np.linalg.pinv(H)
    Mi = np.eye(T_) - np.ones((T_, T_)) / T_
    r = []
    for i in range(N_):
        for k in range(K_):
            x = X[i, :, k]
            r.append((x @ Mb @ x) / (x @ Mi @ x))
    return np.mean(r)


def sim(T, K):
    f = np.zeros(T); g = np.zeros(T)
    for s in range(1, T):
        f[s] = 0.7 * f[s - 1] + 0.03 * rng.standard_normal()
        g[s] = g[s - 1] + 0.02 + 0.03 * rng.standard_normal()
    lam, gam, dl = rng.uniform(.5, 1.5, (N, K)), rng.uniform(.5, 1.5, (N, K)), rng.uniform(.5, 1.5, N)
    xi = np.cumsum(rng.standard_normal((N, T, K)) * 0.05 + 0.005, axis=1)
    X = gam[:, None, :] * g[None, :, None] + xi + lam[:, None, :] * f[None, :, None]
    e = np.zeros((N, T))
    v = rng.standard_normal((N, T)) * 0.03
    for s in range(1, T):
        e[:, s] = 0.5 * e[:, s - 1] + v[:, s]
    Y = X.sum(axis=2) * 0.25 + dl[:, None] * f[None, :] + e
    return Y, X


res = pd.DataFrame(index=KS, columns=TS, dtype=float)
for K in KS:
    for T in TS:
        res.loc[K, T] = np.mean([retention(*sim(T, K)) for _ in range(R)])
print(res.round(3))

# empirical panel
df = pd.read_csv(PROC / "panel_main.csv")
ids, Y, X = E.to_arrays(df, Y_VAR, X_BASE)
rho_emp = retention(Y, X)
# --- reliability of the digital regressor under classical measurement error (attenuation illustration) ---
dd = X[:, :, J_INET]
tw = dd - dd.mean(0, keepdims=True) - dd.mean(1, keepdims=True) + dd.mean()          # two-way demeaned
v_tw = tw.var()
H_ = np.column_stack([np.ones(Y.shape[1]), Y.mean(0), X.mean(0)])
Mb_ = np.eye(Y.shape[1]) - H_ @ np.linalg.pinv(H_)
v_cce = np.mean([(dd[i] @ Mb_ @ dd[i]) / Y.shape[1] for i in range(N)])             # variation left after the CCE projection
rel = {k: (v_tw / (v_tw + sm ** 2), v_cce / (v_cce + sm ** 2)) for k, sm in (("A", 0.03), ("B", 0.05), ("C", 0.10))}
set_macros({"ratioAtt": f"{rel['B'][1] / rel['B'][0]:.2f}", "sdTw": f"{np.sqrt(v_tw):.3f}", "sdCce": f"{np.sqrt(v_cce):.3f}",
            **{f"relTw{k}": f"{v[0]:.2f}" for k, v in rel.items()}, **{f"relCce{k}": f"{v[1]:.2f}" for k, v in rel.items()}})
print("reliability (two-way, CCE):", {k: tuple(np.round(v, 2)) for k, v in rel.items()}, "sd", np.sqrt(v_tw), np.sqrt(v_cce))
rho_emp_nohc = retention(Y, X[:, :, [0, 2, 3]])      # without the (interpolated) schooling regressor
rho_emp_dig = np.mean([retention(Y, X[:, :, [2]]), ])
set_macros({"retEmpNoHc": f"{rho_emp_nohc:.2f}", "retEmp": f"{rho_emp:.2f}", "retKfourTtwentyfour": f"{res.loc[4, 24]:.2f}", "retKfourThundred": f"{res.loc[4, 100]:.2f}",
            "retKoneTtwentyfour": f"{res.loc[1, 24]:.2f}", "retKsixTtwentyfour": f"{res.loc[6, 24]:.2f}",
            "retKfourTtwenty": f"{res.loc[4, 20]:.2f}", "retKfourTfifty": f"{res.loc[4, 50]:.2f}",
            "inflKfourTtwentyfour": f"{1 / res.loc[4, 24]:.1f}", "dfCceKfour": f"{24 - 2 * 4 - 2}", "dfPlainKfour": f"{24 - 4 - 1}"})

rows = []
for K in KS:
    rows.append([f"$K={K}$"] + [f"{res.loc[K, T]:.2f}" for T in TS] + [str(24 - 2 * K - 2), str(24 - K - 1)])
note = (f"Mean retention ratio $\\rho$ = share of within-unit variation of a regressor left after projecting on $(1,\\bar y_t,\\bar x_t)$, $N=17$, {R} simulated panels per cell "
        f"(I(1) regressors with a global trend and a stationary factor). The last two columns give the residual degrees of freedom of a unit regression with $T=24$ under CCE ($T-2K-2$) "
        f"and under a plain unit regression ($T-K-1$). In the empirical panel ($K=4$, $T=24$) the retention ratio is {rho_emp:.2f}.")
(TAB / "tab_cce_retention.tex").write_text(tabular(rows, "lrrrrrrr", header=[["", *[f"$T={T}$" for T in TS], "df CCE ($T$=24)", "df plain ($T$=24)"]], note=note), encoding="utf-8")

fig, ax = plt.subplots(figsize=(4.8, 3.1))
cm = plt.get_cmap("viridis")
for i, K in enumerate(KS):
    ax.plot(range(len(TS)), res.loc[K].values, "-o", ms=3, color=cm(i / (len(KS) - 1)), label=f"$K={K}$")
ax.axhline(rho_emp, color=ps.RED, lw=1.0, ls="--")
ax.annotate(f"empirical ($K=4$, $T=24$): {rho_emp:.2f}", (0.05, rho_emp + 0.015), fontsize=7, color=ps.RED)
ax.set_xticks(range(len(TS))); ax.set_xticklabels(TS); ax.set_xlabel("$T$"); ax.set_ylabel("Retention ratio $\\rho$")
ax.legend(fontsize=7, ncol=2)
ps.save(fig, FIG / "fig_cce_retention.pdf")
print("empirical retention", round(rho_emp, 3))
