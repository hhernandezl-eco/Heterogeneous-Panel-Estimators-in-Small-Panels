"""Panel time-series econometrics in Python (numpy/scipy/statsmodels).

All routines work on balanced panels stored as arrays
    Y : (N, T)        dependent variable
    X : (N, T, K)     regressors
and are written to be transparent and replicable rather than fast.  They are
validated in tests/validate_code.py and in the Monte-Carlo study (code/10_montecarlo.py).

Contents
  Diagnostics     : cd_test, slope_homogeneity
  Unit roots      : adf_fisher, cips_stat, cips_test (Monte-Carlo critical values)
  Cointegration   : kao_adf, pedroni_adf (MC p-values), westerlund (bootstrap p-values)
  Long-run coefs  : fmols_gm, dols_pooled, dols_gm, mg_ecm, pmg, cce_mg, cce_pooled, csdl_mg
  Causality       : dh_test (Dumitrescu-Hurlin, analytical + bootstrap p-values)
"""
import numpy as np
from scipy import stats, optimize
from statsmodels.tsa.stattools import adfuller

# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def to_arrays(df, y, xs, id="iso3", t="year"):
    d = df.sort_values([id, t])
    ids = list(d[id].unique())
    N = len(ids)
    T = d.groupby(id).size().iloc[0]
    assert (d.groupby(id).size() == T).all(), "panel must be balanced"
    Y = d[y].to_numpy(float).reshape(N, T)
    X = d[list(xs)].to_numpy(float).reshape(N, T, len(xs))
    return ids, Y, X


def ols(y, X):
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    return b, e, XtXi


def ols_t(y, X, j):
    """OLS returning coefficient j, its classical se and residuals."""
    b, e, XtXi = ols(y, X)
    s2 = e @ e / (len(y) - X.shape[1])
    return b, np.sqrt(s2 * np.diag(XtXi)), e


def csd_demean(a):
    """Remove cross-sectional means period by period (common time effects). a: (N,T,...)"""
    return a - a.mean(axis=0, keepdims=True)


def within(a):
    return a - a.mean(axis=1, keepdims=True)


def nw_lrv(W, M, one_sided=False):
    """Bartlett long-run covariance of a (T,p) demeaned array.  Returns (Omega, Delta, Sigma)."""
    T = W.shape[0]
    G0 = W.T @ W / T
    Om, De = G0.copy(), G0.copy()
    for j in range(1, M + 1):
        w = 1 - j / (M + 1)
        Gj = W[:-j].T @ W[j:] / T              # E(w_t w_{t+j}')
        Om = Om + w * (Gj + Gj.T)
        De = De + w * Gj
    return Om, De, G0


# --------------------------------------------------------------------------------------
# Cross-sectional dependence and slope heterogeneity
# --------------------------------------------------------------------------------------

def cd_test(E):
    """Pesaran (2004) CD test on an (N,T) matrix of (residual-like) series."""
    N, T = E.shape
    E = within(E)
    C = np.corrcoef(E)
    iu = np.triu_indices(N, 1)
    r = C[iu]
    cd = np.sqrt(2.0 * T / (N * (N - 1))) * r.sum()
    return {"CD": cd, "p": 2 * (1 - stats.norm.cdf(abs(cd))), "mean_rho": r.mean(), "mean_abs_rho": np.abs(r).mean()}


def slope_homogeneity(Y, X):
    """Pesaran & Yamagata (2008) Delta-tilde and adjusted Delta-tilde tests (H0: equal slopes)."""
    N, T, K = X.shape
    Yd, Xd = within(Y), within(X)
    XtX = sum(Xd[i].T @ Xd[i] for i in range(N))
    Xty = sum(Xd[i].T @ Yd[i] for i in range(N))
    bfe = np.linalg.solve(XtX, Xty)
    s2 = np.array([((Yd[i] - Xd[i] @ bfe) @ (Yd[i] - Xd[i] @ bfe)) / (T - 1) for i in range(N)])
    A = [Xd[i].T @ Xd[i] / s2[i] for i in range(N)]
    bw = np.linalg.solve(sum(A), sum(Xd[i].T @ Yd[i] / s2[i] for i in range(N)))
    S = 0.0
    for i in range(N):
        bi = np.linalg.solve(Xd[i].T @ Xd[i], Xd[i].T @ Yd[i])
        d = bi - bw
        S += d @ A[i] @ d
    delta = np.sqrt(N) * (S / N - K) / np.sqrt(2 * K)
    delta_adj = np.sqrt(N) * (S / N - K) / np.sqrt(2 * K * (T - K - 1) / (T + 1))
    return {"delta": delta, "p_delta": 1 - stats.norm.cdf(delta),
            "delta_adj": delta_adj, "p_delta_adj": 1 - stats.norm.cdf(delta_adj)}


# --------------------------------------------------------------------------------------
# Unit-root tests
# --------------------------------------------------------------------------------------

def adf_fisher(Z, lags=1, trend=False):
    """Maddala-Wu (1999) Fisher and Choi (2001) inverse-normal panel unit-root tests built on
    unit-specific ADF p-values (MacKinnon).  Z : (N,T).  H0: all units have a unit root."""
    reg = "ct" if trend else "c"
    pv = np.array([adfuller(z, maxlag=lags, regression=reg, autolag=None)[1] for z in Z])
    pv = np.clip(pv, 1e-10, 1 - 1e-10)
    N = len(pv)
    P = -2 * np.log(pv).sum()
    Zc = stats.norm.ppf(pv).sum() / np.sqrt(N)
    return {"P": P, "p_P": 1 - stats.chi2.cdf(P, 2 * N), "Z": Zc, "p_Z": stats.norm.cdf(Zc)}


def cadf_tstats(Y, p=0, trend=False):
    """Cross-sectionally augmented ADF t-stats (Pesaran 2007).  Y: (R,N,T) -> (R,N)."""
    R, N, T = Y.shape
    ybar = Y.mean(axis=1)                         # (R,T)
    dY = np.diff(Y, axis=2)                       # (R,N,T-1)  dY[...,s] = y_{s+1}-y_s
    dYb = np.diff(ybar, axis=1)                   # (R,T-1)
    t = np.arange(p + 1, T)                       # periods used (0-indexed)
    n = len(t)
    cols = [np.ones((R, N, n))]
    if trend:
        cols.append(np.broadcast_to(t.astype(float), (R, N, n)))
    iy = len(cols)
    cols.append(Y[:, :, t - 1])
    cols.append(np.broadcast_to(ybar[:, None, t - 1], (R, N, n)))
    cols.append(np.broadcast_to(dYb[:, None, t - 1], (R, N, n)))
    for j in range(1, p + 1):
        cols.append(dY[:, :, t - j - 1])
        cols.append(np.broadcast_to(dYb[:, None, t - j - 1], (R, N, n)))
    X = np.stack(cols, axis=-1)                   # (R,N,n,k)
    y = dY[:, :, t - 1]                           # (R,N,n)
    XtX = np.einsum("rnti,rntj->rnij", X, X)
    Xty = np.einsum("rnti,rnt->rni", X, y)
    XtXi = np.linalg.inv(XtX)
    b = np.einsum("rnij,rnj->rni", XtXi, Xty)
    e = y - np.einsum("rnti,rni->rnt", X, b)
    k = X.shape[-1]
    s2 = (e ** 2).sum(-1) / (n - k)
    se = np.sqrt(s2 * XtXi[..., iy, iy])
    return b[..., iy] / se


_CIPS_CACHE = {}


def cips_sim(N, T, p, trend, R=3000, seed=1):
    """Monte-Carlo null distribution of CIPS = mean_i CADF_i for the given (N,T,p,trend)."""
    key = (N, T, p, trend, R, seed)
    if key in _CIPS_CACHE:
        return _CIPS_CACHE[key]
    rng = np.random.default_rng(seed)
    out = np.empty(R)
    chunk = 500
    for s in range(0, R, chunk):
        r = min(chunk, R - s)
        f = rng.standard_normal((r, 1, T))
        lam = rng.normal(1.0, 0.3, (r, N, 1))
        eps = rng.standard_normal((r, N, T))
        dy = lam * f + eps                       # common factor + idiosyncratic shocks
        Y = np.cumsum(dy, axis=2)
        out[s:s + r] = cadf_tstats(Y, p, trend).mean(axis=1)
    _CIPS_CACHE[key] = out
    return out


def cips_test(Z, p=0, trend=False, R=3000, seed=1):
    """CIPS statistic with finite-sample critical values simulated for exactly this (N,T,p,trend)."""
    N, T = Z.shape
    stat = cadf_tstats(Z[None], p, trend).mean()
    sim = cips_sim(N, T, p, trend, R, seed)
    return {"CIPS": stat, "p": (sim <= stat).mean(), "cv10": np.quantile(sim, .10),
            "cv05": np.quantile(sim, .05), "cv01": np.quantile(sim, .01)}


# --------------------------------------------------------------------------------------
# Panel cointegration tests (H0: no cointegration)
# --------------------------------------------------------------------------------------

def _adf_resid(E, p):
    """ADF on residual matrix E (N,T) without deterministics; returns per-unit t-stats,
    pooled t-stat and pooled/individual details."""
    N, T = E.shape
    tis, num, den, res2, nobs = [], 0.0, 0.0, 0.0, 0
    for i in range(N):
        e = E[i]
        de = np.diff(e)
        t = np.arange(p + 1, T)
        y = de[t - 1]
        Z = [e[t - 1]] + [de[t - j - 1] for j in range(1, p + 1)]
        Z = np.column_stack(Z)
        b, se, u = ols_t(y, Z, 0)
        tis.append(b[0] / se[0])
        # pooled version: partial out lags unit by unit
        if p > 0:
            L = Z[:, 1:]
            yt = y - L @ np.linalg.lstsq(L, y, rcond=None)[0]
            xt = Z[:, 0] - L @ np.linalg.lstsq(L, Z[:, 0], rcond=None)[0]
        else:
            yt, xt = y, Z[:, 0]
        num += xt @ yt
        den += xt @ xt
        g = xt @ yt / (xt @ xt)
        res2 += ((yt - g * xt) ** 2).sum()
        nobs += len(yt)
    g = num / den
    s2 = res2 / nobs
    return np.array(tis), g / np.sqrt(s2 / den)


def _pedroni_stats(Y, X, p, demean_time=True):
    if demean_time:
        Y, X = csd_demean(Y), csd_demean(X)
    N, T, K = X.shape
    E = np.empty((N, T))
    for i in range(N):
        D = np.column_stack([np.ones(T), X[i]])
        b, e, _ = ols(Y[i], D)
        E[i] = e
    tis, tpool = _adf_resid(E, p)
    return tis.sum() / np.sqrt(N), tpool


_PED_CACHE = {}


def _pedroni_null(N, T, K, p, R, seed):
    key = (N, T, K, p, R, seed)
    if key not in _PED_CACHE:
        rng = np.random.default_rng(seed)
        sim = np.empty((R, 2))
        for r in range(R):
            Ys = np.cumsum(rng.standard_normal((N, T)), axis=1)
            Xs = np.cumsum(rng.standard_normal((N, T, K)), axis=1)
            sim[r] = _pedroni_stats(Ys, Xs, p)
        _PED_CACHE[key] = sim
    return _PED_CACHE[key]


def pedroni_adf(Y, X, p=0, R=2000, seed=2):
    """Residual-based panel cointegration tests in the spirit of Pedroni (1999, 2004):
    group-mean ADF (t-bar) and panel (pooled) ADF.  Heterogeneous cointegrating vectors and
    common time effects removed.  Finite-sample p-values from a Monte-Carlo of the null of no
    cointegration (independent I(1) series) with identical N, T, K and lag order."""
    N, T, K = X.shape
    g, pn = _pedroni_stats(Y, X, p)
    sim = _pedroni_null(N, T, K, p, R, seed)
    return {"group_adf": g, "p_group": (sim[:, 0] <= g).mean(),
            "panel_adf": pn, "p_panel": (sim[:, 1] <= pn).mean()}


def kao_adf(Y, X, p=1, M=2):
    """Kao (1999) ADF residual-based panel cointegration test (homogeneous slopes, FE)."""
    N, T, K = X.shape
    Yd, Xd = within(Y), within(X)
    b = np.linalg.lstsq(Xd.reshape(N * T, K), Yd.reshape(-1), rcond=None)[0]
    E = Yd - Xd @ b
    rows_y, rows_X, idx = [], [], []
    for i in range(N):
        de = np.diff(E[i])
        t = np.arange(p + 1, T)
        rows_y.append(de[t - 1])
        rows_X.append(np.column_stack([E[i][t - 1]] + [de[t - j - 1] for j in range(1, p + 1)]))
    y = np.concatenate(rows_y)
    Z = np.vstack(rows_X)
    bb, e, ZtZi = ols(y, Z)
    s2 = e @ e / (len(y) - Z.shape[1])
    t_adf = bb[0] / np.sqrt(s2 * ZtZi[0, 0])
    # long-run variances of (u_it = d e_it, eps_it = d x_it)
    W = np.concatenate([np.diff(E, axis=1)[:, :, None], np.diff(X, axis=1)], axis=2)  # (N,T-1,1+K)
    W = W - W.mean(axis=1, keepdims=True)
    Sig = sum(W[i].T @ W[i] for i in range(N)) / (N * (T - 1))
    Om = sum(nw_lrv(W[i], M)[0] for i in range(N)) / N
    def cond(S):
        return S[0, 0] - S[0, 1:] @ np.linalg.solve(S[1:, 1:], S[1:, 0])
    sv2, s0v2 = cond(Sig), cond(Om)
    sv, s0v = np.sqrt(sv2), np.sqrt(s0v2)
    adf = (t_adf + np.sqrt(6 * N) * sv / (2 * s0v)) / np.sqrt(s0v2 / (2 * sv2) + 3 * sv2 / (10 * s0v2))
    return {"ADF": adf, "p": stats.norm.cdf(adf), "t_raw": t_adf}


def _westerlund_stats(Y, X, p):
    """Westerlund (2007) error-correction tests Gt, Ga, Pt, Pa.  ECM with constant,
    p lags of dy, contemporaneous dx and p lags of dx."""
    N, T, K = X.shape
    Gt, Ga, ytil_sum, num, den, s2s = [], [], 0.0, 0.0, 0.0, []
    for i in range(N):
        y, x = Y[i], X[i]
        dy = np.diff(y)
        dx = np.diff(x, axis=0)
        t = np.arange(p + 1, T)
        yy = dy[t - 1]
        ylag = y[t - 1]
        others = [np.ones(len(t)), *[x[t - 1][:, k] for k in range(K)]]
        others += [dy[t - j - 1] for j in range(1, p + 1)]
        others += [dx[t - j - 1][:, k] for j in range(0, p + 1) for k in range(K)]
        O = np.column_stack(others)
        D = np.column_stack([ylag, O])
        b, e, DtDi = ols(yy, D)
        n, k = D.shape
        s2 = e @ e / (n - k)
        a, se = b[0], np.sqrt(s2 * DtDi[0, 0])
        a1 = 1 - b[1 + 1 + K:1 + 1 + K + p].sum()
        Gt.append(a / se)
        Ga.append(T * a / a1)
        # partial out the other regressors for the pooled statistics
        Mo = lambda v: v - O @ np.linalg.lstsq(O, v, rcond=None)[0]
        yt_, dyt_ = Mo(ylag), Mo(yy)
        num += yt_ @ dyt_ / a1
        den += yt_ @ yt_
        s2s.append(s2)
    alpha = num / den
    se = np.sqrt(np.mean(s2s) / den)
    return np.array([np.mean(Gt), np.mean(Ga), alpha / se, (T) * alpha])


def westerlund(Y, X, p=0, B=499, seed=3):
    """Westerlund (2007) ECM panel cointegration tests with bootstrap p-values that are robust to
    cross-sectional dependence (joint time resampling of the restricted-model innovations)."""
    N, T, K = X.shape
    obs = _westerlund_stats(Y, X, p)
    rng = np.random.default_rng(seed)
    # restricted model under H0 (no error correction): dy_i on const, p lags of dy and dx_t..dx_{t-p}
    dY, dX = np.diff(Y, axis=1), np.diff(X, axis=1)
    t = np.arange(p + 1, T)
    coef, resid, DXs = [], np.zeros((N, len(t))), []
    for i in range(N):
        cols = [np.ones(len(t))] + [dY[i][t - j - 1] for j in range(1, p + 1)]
        cols += [dX[i][t - j - 1][:, k] for j in range(0, p + 1) for k in range(K)]
        Z = np.column_stack(cols)
        b, e, _ = ols(dY[i][t - 1], Z)
        coef.append(b)
        resid[i] = e
    # innovations: stacked (resid, dx) for t in sample; recentre dx
    dXs = dX[:, t - 1, :]                                # (N, n, K)
    dXs = dXs - dXs.mean(axis=1, keepdims=True)
    resid = resid - resid.mean(axis=1, keepdims=True)
    n = len(t)
    boot = np.empty((B, 4))
    for bidx in range(B):
        draw = rng.integers(0, n, n + p + 1 + 50)         # extra draws: burn-in not needed for iid resampling
        draw = draw[:T - 1]
        e_s = resid[:, draw]
        dx_s = dXs[:, draw, :]
        dyb = np.zeros((N, T - 1))
        for i in range(N):
            b = coef[i]
            # p=0: dy = c + gamma dx_t + e (generalised below for p>0 by recursion)
            for s in range(T - 1):
                val = b[0] + e_s[i, s]
                pos = 1
                for j in range(1, p + 1):
                    val += b[pos] * (dyb[i, s - j] if s - j >= 0 else 0.0); pos += 1
                for j in range(0, p + 1):
                    for k in range(K):
                        val += b[pos] * (dx_s[i, s - j, k] if s - j >= 0 else 0.0); pos += 1
                dyb[i, s] = val
        Yb = np.concatenate([np.zeros((N, 1)), np.cumsum(dyb, axis=1)], axis=1)
        Xb = np.concatenate([np.zeros((N, 1, K)), np.cumsum(dx_s, axis=1)], axis=1)
        boot[bidx] = _westerlund_stats(Yb, Xb, p)
    pv = (boot <= obs).mean(axis=0)
    return {"Gt": obs[0], "Ga": obs[1], "Pt": obs[2], "Pa": obs[3],
            "p_Gt": pv[0], "p_Ga": pv[1], "p_Pt": pv[2], "p_Pa": pv[3]}


# --------------------------------------------------------------------------------------
# Long-run estimators
# --------------------------------------------------------------------------------------

def _nonpar_se(B):
    """Mean-group nonparametric standard errors from unit estimates B (N,K)."""
    N = B.shape[0]
    m = B.mean(axis=0)
    V = (B - m).T @ (B - m) / (N * (N - 1))
    return m, V


def _summ(m, V):
    se = np.sqrt(np.diag(V))
    return {"b": m, "se": se, "t": m / se, "p": 2 * (1 - stats.norm.cdf(np.abs(m / se))), "V": V}


def fmols_gm(Y, X, M=2, demean_time=True):
    """Group-mean FMOLS (Pedroni 2000; Phillips & Hansen 1990).  Returns summary + unit estimates."""
    if demean_time:
        Y, X = csd_demean(Y), csd_demean(X)
    N, T, K = X.shape
    B = np.empty((N, K))
    for i in range(N):
        x, y = X[i], Y[i]
        xd, yd = x - x.mean(0), y - y.mean()
        b0 = np.linalg.solve(xd.T @ xd, xd.T @ yd)
        u = yd - xd @ b0
        eps = np.diff(x, axis=0)
        W = np.column_stack([u[1:], eps])
        W = W - W.mean(0)
        Om, De, _ = nw_lrv(W, M)
        O12, O22 = Om[0, 1:], Om[1:, 1:]
        D21 = De[1:, 0]; D22 = De[1:, 1:]
        gam = np.linalg.solve(O22, O12)
        ystar = yd[1:] - eps @ gam
        dstar = D21 - D22 @ gam
        xx = xd[1:]
        B[i] = np.linalg.solve(xx.T @ xx, xx.T @ ystar - (T - 1) * dstar)
    m, V = _nonpar_se(B)
    out = _summ(m, V)
    out["units"] = B
    return out


def dols_pooled(Y, X, lead=1, lag=1, demean_time=True, return_design=False):
    """Pooled DOLS (Kao & Chiang 2000): common slope, country effects (within), common lead/lag
    coefficients on the first differences of the regressors.  Cluster-robust (country) s.e."""
    if demean_time:
        Y, X = csd_demean(Y), csd_demean(X)
    N, T, K = X.shape
    dX = np.diff(X, axis=1)                                   # dX[:, s] = x_{s+1}-x_s  <-> period s+1
    t = np.arange(lag + 1, T - lead)                          # need dx_{t-lag}..dx_{t+lead}
    ys, Zs, Ids = [], [], []
    for i in range(N):
        cols = [X[i][t]]
        for j in range(-lead, lag + 1):                       # dx_{t-j}, j=-lead..lag
            cols.append(dX[i][t - j - 1])
        Z = np.hstack(cols)
        ys.append(Y[i][t] - Y[i][t].mean())
        Zs.append(Z - Z.mean(0))
        Ids.append(np.full(len(t), i))
    y = np.concatenate(ys); Z = np.vstack(Zs); ids = np.concatenate(Ids)
    if return_design:
        return y, Z, ids
    b, e, ZtZi = ols(y, Z)
    meat = np.zeros((Z.shape[1],) * 2)
    for i in range(N):
        s = Z[ids == i].T @ e[ids == i]
        meat += np.outer(s, s)
    n, k = Z.shape
    corr = N / (N - 1) * (n - 1) / (n - k - N)
    V = corr * ZtZi @ meat @ ZtZi
    return _summ(b[:K], V[:K, :K])


def dols_gm(Y, X, lead=0, lag=0, demean_time=True):
    """Group-mean DOLS (Pedroni 2001; Mark & Sul 2003)."""
    if demean_time:
        Y, X = csd_demean(Y), csd_demean(X)
    N, T, K = X.shape
    dX = np.diff(X, axis=1)
    t = np.arange(lag + 1, T - lead)
    B = np.empty((N, K))
    for i in range(N):
        cols = [np.ones(len(t)), X[i][t]]
        for j in range(-lead, lag + 1):
            cols.append(dX[i][t - j - 1])
        Z = np.column_stack(cols)
        b = np.linalg.lstsq(Z, Y[i][t], rcond=None)[0]
        B[i] = b[1:1 + K]
    m, V = _nonpar_se(B)
    out = _summ(m, V)
    out["units"] = B
    return out


def _ecm_design(Y, X, i):
    """Return dy, y_{t-1}, x_{t-1}, dx_t for unit i (first period lost)."""
    dy = np.diff(Y[i])
    dx = np.diff(X[i], axis=0)
    return dy, Y[i][:-1], X[i][:-1], dx


def mg_ecm(Y, X):
    """Mean-group (Pesaran & Smith 1995) on the ARDL(1,1,..,1) error-correction form:
    dy = mu + phi*y_{t-1} + pi'x_{t-1} + g'dx_t + e;  theta_i = -pi/phi."""
    N, T, K = X.shape
    TH, PHI = np.empty((N, K)), np.empty(N)
    for i in range(N):
        dy, yl, xl, dx = _ecm_design(Y, X, i)
        D = np.column_stack([np.ones(T - 1), yl, xl, dx])
        b = np.linalg.lstsq(D, dy, rcond=None)[0]
        PHI[i] = b[1]
        TH[i] = -b[2:2 + K] / b[1]
    m, V = _nonpar_se(TH)
    out = _summ(m, V)
    out["units"] = TH
    out["phi_units"] = PHI
    pm, pV = _nonpar_se(PHI[:, None])
    out["phi"] = pm[0]
    out["phi_se"] = np.sqrt(pV[0, 0])
    return out


def pmg(Y, X, theta0=None, se=True, n_start=6, dfc=True, seed=0):
    """Pooled mean group (Pesaran, Shin & Smith 1999) by direct maximisation of the concentrated
    likelihood: common long-run theta, unit-specific adjustment speed, intercept, short-run
    coefficients and variance.  Unit regressions are solved in batch (vectorised over countries).

    The concentrated likelihood can be multimodal in small panels (see the paper), so the maximiser is run from
    n_start starting values (median MG, mean MG, zero, static pooled OLS, and random perturbations) and the
    global maximum is returned; out['n_modes'] counts the distinct local optima found and out['modes'] lists them.
    Standard errors come from the curvature of the concentrated log-likelihood; with dfc=True they are multiplied by
    sqrt(n/(n-p)), p = K+2 parameters per unit, the usual degrees-of-freedom correction for the downward-biased ML variance
    (out['se_ml'] keeps the uncorrected ones)."""
    N, T, K = X.shape
    n = T - 1
    dy = np.diff(Y, axis=1)                       # (N,n)
    yl = Y[:, :-1]
    xl = X[:, :-1, :]
    dx = np.diff(X, axis=1)
    ones = np.ones((N, n, 1))

    def fit(th):
        z = yl - xl @ th
        D = np.concatenate([ones, z[:, :, None], dx], axis=2)          # (N,n,2+K)
        DtD = np.einsum("nti,ntj->nij", D, D)
        Dty = np.einsum("nti,nt->ni", D, dy)
        b = np.linalg.solve(DtD, Dty[:, :, None])[:, :, 0]
        e = dy - np.einsum("nti,ni->nt", D, b)
        return b, (e ** 2).sum(axis=1)

    def negll(th):
        v = fit(th)[1]
        if np.any(v <= 0) or not np.all(np.isfinite(v)):
            return 1e12
        return 0.5 * n * np.log(v / n).sum()

    if theta0 is None:
        mg = mg_ecm(Y, X)["units"]
        pooled = np.linalg.lstsq(X.reshape(N * T, K), Y.reshape(-1), rcond=None)[0]
        rng = np.random.default_rng(seed)
        med = np.clip(np.median(mg, axis=0), -3, 3)
        starts = [med, np.clip(mg.mean(0), -3, 3), np.zeros(K), np.clip(pooled, -3, 3)]
        starts += [med + rng.normal(0, 0.3, K) for _ in range(max(0, n_start - len(starts)))]
        starts = starts[:max(n_start, 1)]
    else:
        starts = [np.asarray(theta0, float)]
    sols = []
    for s0 in starts:
        try:
            r = optimize.minimize(negll, s0, method="BFGS")
            if np.isfinite(r.fun) and r.fun < 1e11:
                sols.append((r.fun, r.x))
        except Exception:
            pass
    if not sols:
        raise RuntimeError("PMG maximisation failed from all starting values")
    sols.sort(key=lambda a_: a_[0])
    fbest, th = sols[0]
    modes = []
    for f_, x_ in sols:
        if not any(np.max(np.abs(x_ - m_[1])) < 0.02 for m_ in modes):
            modes.append((f_, x_))
    phi = fit(th)[0][:, 1]
    pm_, pV = _nonpar_se(phi[:, None])
    if se:
        h = 1e-4
        H = np.empty((K, K))
        for a_ in range(K):
            for c in range(a_, K):
                ea, ec = np.zeros(K), np.zeros(K)
                ea[a_], ec[c] = h, h
                H[a_, c] = H[c, a_] = (negll(th + ea + ec) - negll(th + ea - ec)
                                       - negll(th - ea + ec) + negll(th - ea - ec)) / (4 * h * h)
        V_ml = np.linalg.inv(H)
    else:
        V_ml = np.full((K, K), np.nan)
    corr = n / (n - (K + 2)) if dfc and n > K + 2 else 1.0
    out = _summ(th, V_ml * corr)
    out["se_ml"] = np.sqrt(np.diag(V_ml))
    out["phi_units"] = phi
    out["phi"] = pm_[0]
    out["phi_se"] = np.sqrt(pV[0, 0])
    out["negll"] = fbest
    out["n_modes"] = len(modes)
    out["modes"] = [(float(f_), np.asarray(x_)) for f_, x_ in modes]
    return out


def hausman_mg_pmg(mg, pm, j=None):
    """Hausman h-test of long-run slope homogeneity: MG (consistent) vs PMG (efficient under H0)."""
    d = mg["b"] - pm["b"]
    if j is not None:
        v = mg["V"][j, j] - pm["V"][j, j]
        h = d[j] ** 2 / v if v > 0 else np.nan
        return {"h": h, "df": 1, "p": 1 - stats.chi2.cdf(h, 1) if np.isfinite(h) else np.nan}
    Vd = mg["V"] - pm["V"]
    h = float(d @ np.linalg.pinv(Vd) @ d)
    return {"h": h, "df": len(d), "p": 1 - stats.chi2.cdf(h, len(d))}


def cce_mg(Y, X, p_bar=0):
    """Common correlated effects mean group (Pesaran 2006) in levels.  With p_bar > 0 the cross-sectional averages
    enter with p_bar lags (dynamic CCE augmentation of Chudik & Pesaran 2015); the first p_bar periods are lost."""
    N, T, K = X.shape
    Yb, Xb = Y.mean(0), X.mean(0)
    t = np.arange(p_bar, T)
    B = np.empty((N, K))
    for i in range(N):
        cols = [np.ones(len(t)), X[i][t]]
        for l in range(p_bar + 1):
            cols += [Yb[t - l][:, None], Xb[t - l]]
        D = np.column_stack(cols)
        b = np.linalg.lstsq(D, Y[i][t], rcond=None)[0]
        B[i] = b[1:1 + K]
    m, V = _nonpar_se(B)
    out = _summ(m, V)
    out["units"] = B
    out["df"] = len(t) - D.shape[1]
    return out


def cce_pooled(Y, X):
    """CCE pooled estimator (Pesaran 2006) with the robust variance of eq. (67)."""
    N, T, K = X.shape
    H = np.column_stack([np.ones(T), Y.mean(0), X.mean(0)])
    Mh = np.eye(T) - H @ np.linalg.solve(H.T @ H, H.T)
    Psi_i = [X[i].T @ Mh @ X[i] / T for i in range(N)]
    Bi = np.array([np.linalg.solve(X[i].T @ Mh @ X[i], X[i].T @ Mh @ Y[i]) for i in range(N)])
    Psi = sum(Psi_i) / N
    bP = np.linalg.solve(sum(X[i].T @ Mh @ X[i] for i in range(N)), sum(X[i].T @ Mh @ Y[i] for i in range(N)))
    bmg = Bi.mean(0)
    R = sum(Psi_i[i] @ np.outer(Bi[i] - bmg, Bi[i] - bmg) @ Psi_i[i].T for i in range(N)) / (N - 1)
    Pi = np.linalg.inv(Psi)
    V = Pi @ R @ Pi / N
    return _summ(bP, V)


def csdl_mg(Y, X, p_delta=1, p_bar=0):
    """Cross-sectionally augmented distributed lag, mean group (Chudik, Mohaddes, Pesaran & Raissi
    2016): y_it = a_i + theta_i'x_it + sum_{l<p_delta} d_il' dx_{i,t-l} + sum_{l<=p_bar} w_il'(ybar_{t-l}, xbar_{t-l}) + e.
    p_delta = 1 means contemporaneous differences only; p_bar = lags of the cross-sectional averages."""
    N, T, K = X.shape
    Yb, Xb = Y.mean(0), X.mean(0)
    dX = np.diff(X, axis=1)
    t = np.arange(max(p_delta, p_bar), T)
    B = np.empty((N, K))
    for i in range(N):
        cols = [np.ones(len(t)), X[i][t]]
        for l in range(p_delta):
            cols.append(dX[i][t - l - 1])
        for l in range(p_bar + 1):
            cols += [Yb[t - l][:, None], Xb[t - l]]
        D = np.column_stack(cols)
        b = np.linalg.lstsq(D, Y[i][t], rcond=None)[0]
        B[i] = b[1:1 + K]
    m, V = _nonpar_se(B)
    out = _summ(m, V)
    out["units"] = B
    out["df"] = len(t) - D.shape[1]
    return out


# --------------------------------------------------------------------------------------
# Granger non-causality in heterogeneous panels (Dumitrescu & Hurlin 2012)
# --------------------------------------------------------------------------------------

def _dh_W(Y, X, K):
    N, T = Y.shape
    W = np.empty(N)
    for i in range(N):
        t = np.arange(K, T)
        yl = np.column_stack([Y[i][t - k] for k in range(1, K + 1)])
        xl = np.column_stack([X[i][t - k] for k in range(1, K + 1)])
        one = np.ones(len(t))
        Du = np.column_stack([one, yl, xl])
        Dr = np.column_stack([one, yl])
        eu = Y[i][t] - Du @ np.linalg.lstsq(Du, Y[i][t], rcond=None)[0]
        er = Y[i][t] - Dr @ np.linalg.lstsq(Dr, Y[i][t], rcond=None)[0]
        dfu = len(t) - 2 * K - 1
        W[i] = (er @ er - eu @ eu) / (eu @ eu / dfu)
    return W


def dh_test(Y, X, K=1, B=499, seed=4):
    """Dumitrescu-Hurlin test of H0: X does not Granger-cause Y for any unit.
    Reports W-bar, Z-bar, Z-tilde (analytical p-values) and a bootstrap p-value for W-bar that is
    robust to cross-sectional dependence."""
    N, T = Y.shape
    W = _dh_W(Y, X, K)
    Wb = W.mean()
    Zbar = np.sqrt(N / (2 * K)) * (Wb - K)
    Zt = np.sqrt(N / (2 * K) * (T - 2 * K - 5) / (T - K - 3)) * ((T - 2 * K - 3) / (T - 2 * K - 1) * Wb - K)
    out = {"Wbar": Wb, "Zbar": Zbar, "p_Zbar": 2 * (1 - stats.norm.cdf(abs(Zbar))),
           "Ztilde": Zt, "p_Ztilde": 2 * (1 - stats.norm.cdf(abs(Zt))), "W_units": W}
    if B:
        rng = np.random.default_rng(seed)
        t = np.arange(K, T)
        # restricted model: y on own lags -> residuals, jointly resampled over time
        coefs, res = [], np.zeros((N, len(t)))
        for i in range(N):
            Dr = np.column_stack([np.ones(len(t))] + [Y[i][t - k] for k in range(1, K + 1)])
            b = np.linalg.lstsq(Dr, Y[i][t], rcond=None)[0]
            coefs.append(b)
            res[i] = Y[i][t] - Dr @ b
        res = res - res.mean(1, keepdims=True)
        boot = np.empty(B)
        for bi in range(B):
            draw = rng.integers(0, len(t), T)
            Yb = np.zeros((N, T))
            Yb[:, :K] = Y[:, :K]
            for s in range(K, T):
                for i in range(N):
                    b = coefs[i]
                    Yb[i, s] = b[0] + sum(b[k] * Yb[i, s - k] for k in range(1, K + 1)) + res[i, draw[s]]
            boot[bi] = _dh_W(Yb, X, K).mean()
        out["p_boot"] = (boot >= Wb).mean()
    return out


def twfe_cluster(Y, X):
    """Two-way fixed effects (country effects + year effects by double demeaning) with
    country-clustered standard errors (CR1)."""
    N, T, K = X.shape
    Yd = Y - Y.mean(0, keepdims=True) - Y.mean(1, keepdims=True) + Y.mean()
    Xd = X - X.mean(0, keepdims=True) - X.mean(1, keepdims=True) + X.mean(axis=(0, 1), keepdims=True)
    Z, y = Xd.reshape(N * T, K), Yd.reshape(-1)
    b, e, ZtZi = ols(y, Z)
    meat = np.zeros((K, K))
    for i in range(N):
        s = Z[i * T:(i + 1) * T].T @ e[i * T:(i + 1) * T]
        meat += np.outer(s, s)
    n, kk = N * T, K + N + T - 2
    V = (N / (N - 1)) * ((n - 1) / (n - kk)) * ZtZi @ meat @ ZtZi
    return _summ(b, V)


# --------------------------------------------------------------------------------------
# Inference with few clusters
# --------------------------------------------------------------------------------------

_WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def _absorb(v, groups):
    """Sweep out group means (successively; exact for balanced panels)."""
    v = v.copy()
    for g_ in groups:
        for _ in range(2 if len(groups) > 1 else 1):
            for gg in groups:
                s = np.bincount(gg, weights=v) / np.bincount(gg)
                v = v - s[gg]
    return v


def cluster_t(y, X, cl, j):
    """Cluster-robust (CR1) t-statistic of coefficient j and the number of clusters."""
    n, k = X.shape
    b, e, XtXi = ols(y, X)
    ug = np.unique(cl)
    meat = np.zeros((k, k))
    for g_ in ug:
        s = X[cl == g_].T @ e[cl == g_]
        meat += np.outer(s, s)
    G = len(ug)
    V = G / (G - 1) * (n - 1) / (n - k) * XtXi @ meat @ XtXi
    return b[j] / np.sqrt(V[j, j]), G


def wild_cluster_p(y, X, cl, j, B=999, seed=0, absorb=None):
    """Wild cluster restricted bootstrap-t p-value for H0: beta_j = 0 (Cameron, Gelbach & Miller 2008) with the six-point
    weights of Webb (2014), which behave better than Rademacher weights with few clusters (G < 20).
    y, X are the variables as used in the regression (fixed effects either included in X or swept out beforehand);
    if `absorb` (list of integer group-id arrays) is given, the bootstrap outcome is re-swept of those effects in every draw.
    Returns (p-value, t statistic, t(G-1) p-value)."""
    rng = np.random.default_rng(seed)
    t0, G = cluster_t(y, X, cl, j)
    Xr = np.delete(X, j, axis=1)
    br, er, _ = ols(y, Xr)
    fit = y - er
    ug, inv = np.unique(cl, return_inverse=True)
    ts = np.empty(B)
    for b in range(B):
        w = _WEBB[rng.integers(0, 6, len(ug))][inv]
        ys = fit + w * er
        if absorb is not None:
            ys = _absorb(ys, absorb)
        ts[b] = cluster_t(ys, X, cl, j)[0]
    p_boot = float((np.abs(ts) >= abs(t0)).mean())
    p_t = float(2 * (1 - stats.t.cdf(abs(t0), G - 1)))
    return p_boot, float(t0), p_t
