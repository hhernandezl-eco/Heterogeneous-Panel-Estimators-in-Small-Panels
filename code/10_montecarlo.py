"""Step 10 - Monte-Carlo study of long-run panel estimators in small regional macro panels.

Design (N = 17 fixed, as in the application; K = 4 regressors; the third regressor is a bounded
logistic "digital adoption" share):

  Baseline scenarios
    S1  genuine cointegration, moderately persistent AR(1) errors (rho = 0.5), endogenous regressors
    S1N as S1 but with strictly exogenous regressors
    S1U as S1 but the digital regressor is an unbounded I(1) process instead of a bounded logistic share
    S2  as S1 with highly persistent errors (rho = 0.85)
    S3  NO cointegration: y is I(1) with a common stochastic trend and I(1) idiosyncratic errors; true slopes 0
  Scenarios on heterogeneity, loadings and measurement error
    S1H  as S1 with heterogeneous slopes  theta_ik = theta_k + sigma_k * nu_ik   (sigma calibrated to the cross-country
         dispersion of unit-level estimates in the data)
    S2H  as S2 with heterogeneous slopes
    S1F  as S1 with a direct global-trend loading of y that is correlated with the country's adoption timing
         (heterogeneous common-factor loadings coupled with the regressor)
    S1M  as S1 with classical measurement error in the observed adoption share (s.d. 0.05)
    S4   S1H + S1F + S1M together (the 'realistic combination')
    S1C  as S1H but the slope deviations are correlated (rho = 0.7) with the scale of each regressor / the timing of adoption,
         so that pooled estimators converge to a variance-weighted average that differs from the simple mean
  Time dimension  T in {20, 24, 30, 50, 100} (baseline) and {20, 24, 30, 50} (added scenarios)
  Replications    M = 2000 per cell (configurable)

Two blocks of estimators are stored separately (so that adding estimators never changes earlier results):
  base  (EST)   TWFE, FMOLS-GM, DOLS pooled, DOLS-GM, PMG, MG, CCE-MG, CCE-P, CS-DL          -> mc2_<scen>_T<T>.npy
  extra (EST_X) CCE-MG with 1 and with p_T=2 lags of the cross-section averages, CS-DL with 1 lag of the averages,
                DOLS-GM with one lead and one lag                                                -> mcx2_<scen>_T<T>.npy
Each array has shape (replications, estimator, {estimate, s.e.}, slope).

Usage:  python code/10_montecarlo.py run [M]        (parallel, cached in output/mc/)
        python code/10_montecarlo.py summarize      (tables and figures from the cache)
"""
import sys, time, os
import numpy as np, pandas as pd
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import *
import lib.econ as E

MC = ROOT / "output" / "mc"
MC.mkdir(parents=True, exist_ok=True)
N, K = 17, 4
J = 2                                       # index of the digital (bounded) regressor
THETA = np.array([0.20, 0.10, 0.30, 0.05])
SIG_H = np.array([0.12, 0.15, 0.20, 0.13])  # s.d. of country-specific slopes (calibrated to unit-level CCE-MG/CS-DL dispersion)
SIG_ME = 0.05                               # s.d. of measurement error in the adoption share
KAPPA_F = 0.5                               # strength of the factor-loading / adoption-timing correlation in S1F
TS = [20, 24, 30, 50, 100]
SCEN = {"S1": 0.5, "S1N": 0.5, "S2": 0.85, "S3": None, "S1U": 0.5,
        "S1H": 0.5, "S2H": 0.85, "S1F": 0.5, "S1M": 0.5, "S4": 0.5, "S1C": 0.5}
ENDO = {"S1": 1.0, "S1N": 0.0, "S2": 1.0, "S1U": 1.0, "S1H": 1.0, "S2H": 1.0, "S1F": 1.0, "S1M": 1.0, "S4": 1.0, "S1C": 1.0}
HET = {"S1H", "S2H", "S4"}
FAC = {"S1F", "S4"}
ME = {"S1M", "S4"}
CORRH = {"S1C"}                              # heterogeneous slopes correlated with the regressors (scale of x_k, timing of adoption)
RHO_C = 0.7
TS_BY = {"S1": TS, "S1N": [20, 24, 30, 50], "S2": [20, 24, 30, 50], "S3": TS, "S1U": TS,
         "S1H": [20, 24, 30, 50], "S2H": [20, 24, 30, 50], "S1F": [20, 24, 30, 50], "S1M": [20, 24, 30, 50], "S4": [20, 24, 30, 50], "S1C": [20, 24, 30, 50]}
EST = ["TWFE", "FMOLS-GM", "DOLS pooled", "DOLS-GM", "PMG", "MG", "CCE-MG", "CCE-P", "CS-DL"]
EST_X = ["CCE-MG L1", "CCE-MG L2", "CS-DL L1", "DOLS-GM (1,1)"]


def simulate(rng, T, scen, N=N):
    """One panel (Y: N x T, X: N x T x K).  Random draws of the baseline design come first and in the original
    order; draws needed only by the added scenarios come last, so baseline results are unchanged."""
    t = np.arange(T, dtype=float)
    f = np.zeros(T); g = np.zeros(T)                         # stationary common factor, I(1) global trend
    ef, eg = rng.standard_normal(T), rng.standard_normal(T)
    for s in range(1, T):
        f[s] = 0.7 * f[s - 1] + 0.03 * ef[s]
        g[s] = g[s - 1] + 0.02 + 0.03 * eg[s]
    lam = rng.uniform(0.5, 1.5, (N, K)); gam = rng.uniform(0.5, 1.5, (N, K)); dlt = rng.uniform(0.5, 1.5, N)
    sd_xi = np.array([0.08, 0.015, 0.0, 0.06]); drift = np.array([0.0, 0.01, 0.0, 0.005])
    eta = rng.standard_normal((N, T, K)) * sd_xi             # idiosyncratic I(1) innovations (column 2 unused)
    xi = np.cumsum(eta + drift, axis=1)
    X = gam[:, None, :] * g[None, :, None] + xi + lam[:, None, :] * f[None, :, None]
    # bounded logistic adoption share with a common wave, country-specific timing/speed and AR(1) noise
    mid = 12.0 + rng.normal(0, 3.0, N); sp = np.clip(rng.normal(0.28, 0.05, N), 0.1, None)
    path = 1.0 / (1.0 + np.exp(-sp[:, None] * (t[None, :] - mid[:, None])))
    u = np.zeros((N, T)); un = rng.standard_normal((N, T)) * 0.03
    for s in range(1, T):
        u[:, s] = 0.6 * u[:, s - 1] + un[:, s]
    d = np.clip(path + u + 0.5 * lam[:, 2][:, None] * f[None, :], 0.002, 0.998)
    if scen == "S1U":                                        # unbounded I(1) digital regressor (benchmark design)
        eta[:, :, J] = rng.standard_normal((N, T)) * 0.05
        xiu = np.cumsum(eta[:, :, J] + 0.012, axis=1)
        d = gam[:, J][:, None] * g[None, :] + xiu + lam[:, J][:, None] * f[None, :]
    X[:, :, J] = d
    # error / outcome
    v = rng.standard_normal((N, T)) * 0.03
    e = np.zeros((N, T))
    if SCEN[scen] is not None:
        endo = 0.15 * eta[:, :, 0] + 0.15 * eta[:, :, 3] + 0.15 * (d - np.roll(d, 1, axis=1)) * (np.arange(T) > 0)
        v = v + ENDO[scen] * endo
        rho = SCEN[scen]
        for s in range(1, T):
            e[:, s] = rho * e[:, s - 1] + v[:, s]
        Y = (X * THETA).sum(axis=2) + dlt[:, None] * f[None, :] + e + 0.2 * np.arange(N)[:, None]
    else:                                                    # no cointegration: independent I(1) errors, shared trend
        for s in range(1, T):
            e[:, s] = e[:, s - 1] + v[:, s]
        Y = 1.5 * g[None, :] * dlt[:, None] + dlt[:, None] * f[None, :] + e + 0.2 * np.arange(N)[:, None]
    # ---- added scenarios (extra draws last) ----
    if scen in HET:                                          # heterogeneous slopes theta_ik = theta_k + sigma_k nu_ik
        nu = rng.standard_normal((N, K)) * SIG_H
        Y = Y + (X * nu[:, None, :]).sum(axis=2)
    if scen in CORRH:                                        # slopes correlated with the regressors: nu_ik = sigma_k (rho z_ik + sqrt(1-rho^2) e_ik)
        nu_ind = rng.standard_normal((N, K))
        z_std = (gam - 1.0) / 0.2887                         # standardised loading on the global trend = scale of x_k
        z_std[:, J] = -(mid - 12.0) / 3.0                    # digital regressor: standardised earliness of adoption
        nu = SIG_H * (RHO_C * z_std + np.sqrt(1 - RHO_C ** 2) * nu_ind)
        Y = Y + (X * nu[:, None, :]).sum(axis=2)
    if scen in FAC:                                          # global-trend loading of y correlated with adoption timing
        z = -(mid - 12.0) / 3.0                              # standardised earliness of adoption
        Y = Y + KAPPA_F * z[:, None] * g[None, :]
    if scen in ME:                                           # classical measurement error in the observed share
        X = X.copy()
        X[:, :, J] = X[:, :, J] + rng.standard_normal((N, T)) * SIG_ME
    return Y, X


def _wrap(f, K=K, min_df=None):
    try:
        r = f()
        if min_df is not None and r.get("df", 99) < min_df:
            raise ValueError("too few degrees of freedom")
        return np.asarray(r["b"], float), np.asarray(r["se"], float)
    except Exception:
        return np.full(K, np.nan), np.full(K, np.nan)


def estimate(Y, X):
    """Base block: dict estimator -> (b (K,), se (K,))."""
    Yd, Xd = E.csd_demean(Y), E.csd_demean(X)
    fns = {
        "TWFE": lambda: E.twfe_cluster(Y, X),
        "FMOLS-GM": lambda: E.fmols_gm(Y, X),
        "DOLS pooled": lambda: E.dols_pooled(Y, X, 1, 1),
        "DOLS-GM": lambda: E.dols_gm(Y, X, 0, 0),
        "PMG": lambda: E.pmg(Yd, Xd),
        "MG": lambda: E.mg_ecm(Yd, Xd),
        "CCE-MG": lambda: E.cce_mg(Y, X),
        "CCE-P": lambda: E.cce_pooled(Y, X),
        "CS-DL": lambda: E.csdl_mg(Y, X, 1),
    }
    return {k: _wrap(f) for k, f in fns.items()}


def estimate_extra(Y, X):
    """Extra block (dynamic augmentations); estimators with fewer than 3 residual d.f. return NaN."""
    T = X.shape[1]
    Yd, Xd = E.csd_demean(Y), E.csd_demean(X)
    out = {
        "CCE-MG L1": _wrap(lambda: E.cce_mg(Y, X, 1), min_df=3),
        "CCE-MG L2": _wrap(lambda: E.cce_mg(Y, X, 2), min_df=3),
        "CS-DL L1": _wrap(lambda: E.csdl_mg(Y, X, 1, 1), min_df=3),
    }
    if T - 2 - (1 + K + 3 * K) >= 3:                         # DOLS-GM(1,1): T-2 obs, 1+K+3K parameters
        out["DOLS-GM (1,1)"] = _wrap(lambda: E.dols_gm(Y, X, 1, 1))
    else:
        out["DOLS-GM (1,1)"] = (np.full(K, np.nan), np.full(K, np.nan))
    return out


def _worker(args):
    T, scen, seed_key, reps, blocks = args
    res = {b: np.full((len(reps), len(EST if b == "base" else EST_X), 2, K), np.nan) for b in blocks}
    for ii, r in enumerate(reps):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, seed_key, r]))
        Y, X = simulate(rng, T, scen)
        if "base" in blocks:
            o = estimate(Y, X)
            for jj, k in enumerate(EST):
                res["base"][ii, jj, 0], res["base"][ii, jj, 1] = o[k]
        if "extra" in blocks:
            o = estimate_extra(Y, X)
            for jj, k in enumerate(EST_X):
                res["extra"][ii, jj, 0], res["extra"][ii, jj, 1] = o[k]
    return res


def _have(f, M):
    return f.exists() and np.load(f, mmap_mode="r").shape[0] >= M


def run(M):
    t0 = time.time()
    only_T = os.environ.get("MC_T", "").split(",") if os.environ.get("MC_T") else None
    only_S = os.environ.get("MC_S", "").split(",") if os.environ.get("MC_S") else None
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for si, scen in enumerate(SCEN):
            if only_S and scen not in only_S:
                continue
            for T in TS_BY[scen]:
                if only_T and str(T) not in only_T:
                    continue
                fb, fx = MC / f"mc2_{scen}_T{T}.npy", MC / f"mcx2_{scen}_T{T}.npy"
                blocks = [b for b, f in (("base", fb), ("extra", fx)) if not _have(f, M)]
                if not blocks:
                    continue
                chunks = [list(range(a, min(a + 25, M))) for a in range(0, M, 25)]
                jobs = [(T, scen, si * 1000 + T, c, blocks) for c in chunks]
                outs = pool.map(_worker, jobs)
                for b in blocks:
                    np.save(fb if b == "base" else fx, np.concatenate([o[b] for o in outs], axis=0))
                print(f"{scen} T={T}: {M} reps done for {blocks} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
    else:
        import importlib
        importlib.import_module("mc_summary").main()
