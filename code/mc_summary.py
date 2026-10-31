"""Summaries of the Monte-Carlo study (tables, figures, macros).  Called by `10_montecarlo.py summarize`."""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config import *
from lib import plotstyle as ps
from lib.util import tabular, set_macros

MC = ROOT / "output" / "mc"
N, K, J = 17, 4, 2
THETA_D = 0.30
TS = [20, 24, 30, 50, 100]
TS_BY = {"S1": TS, "S1N": [20, 24, 30, 50], "S2": [20, 24, 30, 50], "S3": TS, "S1U": TS,
         "S1H": [20, 24, 30, 50], "S2H": [20, 24, 30, 50], "S1F": [20, 24, 30, 50], "S1M": [20, 24, 30, 50], "S4": [20, 24, 30, 50], "S1C": [20, 24, 30, 50]}
SCEN = ["S1N", "S1", "S1U", "S2", "S1H", "S2H", "S1C", "S1F", "S1M", "S4", "S3"]
SC_LAB = {"S1N": "S1N: cointegrated, exogenous regressors", "S1": "S1: cointegrated, endogenous regressors",
          "S1U": "S1U: as S1, unbounded regressor", "S2": "S2: as S1, persistent errors", "S3": "S3: no cointegration",
          "S1H": "S1H: as S1, heterogeneous slopes", "S2H": "S2H: as S2, heterogeneous slopes",
          "S1F": "S1F: as S1, factor loadings tied to adoption", "S1M": "S1M: as S1, measurement error in adoption",
          "S4": "S4: S1H + S1F + S1M", "S1C": "S1C: slopes correlated with the regressors"}
MN = {"S1": "Sa", "S2": "Sb", "S1U": "Su", "S3": "Sc", "S1N": "Sn", "S1H": "Sh", "S2H": "Sk", "S1F": "Sf", "S1M": "Sm", "S4": "Sq", "S1C": "Sr",
      24: "Tsmall", 100: "Tlarge", 50: "Tmid"}   # letters-only macro tags
EST = ["TWFE", "FMOLS-GM", "DOLS pooled", "DOLS-GM", "PMG", "MG", "CCE-MG", "CCE-P", "CS-DL"]
EST_X = ["CCE-MG L1", "CCE-MG L2", "CS-DL L1", "DOLS-GM (1,1)"]
EMP8 = ["FMOLS-GM", "DOLS pooled", "DOLS-GM", "PMG", "MG", "CCE-MG", "CCE-P", "CS-DL"]
COL = {"TWFE": "#444444", "FMOLS-GM": "#8c564b", "DOLS pooled": "#2ca02c", "DOLS-GM": "#98df8a", "PMG": "#1f77b4",
       "MG": "#aec7e8", "CCE-MG": "#d62728", "CCE-P": "#ff9896", "CS-DL": "#9467bd"}


def dn(e):
    """Display name used in tables and figures; the group-mean DOLS is always shown with its lead/lag specification."""
    return "DOLS--GM (0,0)" if e == "DOLS-GM" else e.replace("-", "--")


def load(scen, T):
    return np.load(MC / f"mc2_{scen}_T{T}.npy")        # (M, est, 2, K)


def load_x(scen, T):
    return np.load(MC / f"mcx2_{scen}_T{T}.npy")       # extra estimators


def truth(scen):
    return 0.0 if scen == "S3" else THETA_D


def stats_cell(scen, T, extra=False):
    a = load_x(scen, T) if extra else load(scen, T)
    th = truth(scen)
    rows = {}
    for j, e in enumerate(EST_X if extra else EST):
        b, se = a[:, j, 0, J], a[:, j, 1, J]
        ok = np.isfinite(b) & np.isfinite(se) & (se > 0)
        b, se = b[ok], se[ok]
        if len(b) < 20:
            rows[e] = {k_: np.nan for k_ in ("n", "mean", "median", "bias", "medbias", "sd", "iqrsd", "rmse", "medrmse", "cover", "reject0", "rej_pos", "meanse", "ratio")}
            rows[e]["n"] = len(b)
            continue
        t0 = b / se
        q1, q3 = np.percentile(b, [25, 75])
        rows[e] = {"n": ok.sum(), "mean": b.mean(), "median": np.median(b), "bias": b.mean() - th, "medbias": np.median(b) - th,
                   "sd": b.std(ddof=1), "iqrsd": (q3 - q1) / 1.349, "rmse": np.sqrt(((b - th) ** 2).mean()),
                   "medrmse": np.sqrt(np.median((b - th) ** 2)),
                   "cover": (np.abs(b - th) < 1.96 * se).mean(), "reject0": (np.abs(t0) > 1.96).mean(),
                   "rej_pos": (t0 > 1.96).mean(), "meanse": se.mean(), "ratio": np.median(se) / ((q3 - q1) / 1.349)}
    return pd.DataFrame(rows).T


def cells(fmt_s, d, M, scens, T=24):
    r = []
    for s in scens:
        x = M[s][T].loc[d]
        if s == "S3":
            r += [f"{x['median']:+.3f}", f"{x['iqrsd']:.3f}", f"{100 * x['reject0']:.0f}"]
        else:
            r += [f"{x['bias']:+.3f}", f"{x['rmse']:.3f}", f"{100 * x['cover']:.0f}"]
    return r


def main():
    M = {s: {T: stats_cell(s, T) for T in TS_BY[s]} for s in SCEN}
    nrep = load("S1", 24).shape[0]
    set_macros({"mcReps": f"{nrep:,}".replace(",", "\\,")})

    note_base = (f"$N=17$, $T=24$, {nrep:,} replications per scenario. Statistic: slope on the bounded digital-adoption regressor "
                 "(true value 0.30 except in S3). Bias = mean estimate minus truth; RMSE around the truth; Cov.\\% = coverage of the nominal 95\\% interval "
                 "constructed with the estimator's own standard error.")
    # ---------- Table A: cointegrated scenarios ----------
    sc = ["S1N", "S1", "S1U"]
    rows = [[dn(e)] + cells(None, e, M, sc) for e in EST]
    hdr1 = [""] + ["\\multicolumn{3}{c}{%s}" % t for t in ("S1N: exogenous regressors", "S1: endogenous regressors", "S1U: S1 with unbounded regressor")]
    hdr2 = ["Estimator"] + ["Bias", "RMSE", "Cov.\\%"] * 3
    (TAB / "tab_mc_main.tex").write_text(tabular(rows, "l" + "rrr" * 3, header=[hdr1, hdr2], note=note_base, resize=True), encoding="utf-8")
    # ---------- Table B: persistent errors and no cointegration ----------
    sc = ["S2", "S3"]
    rows = [[dn(e)] + cells(None, e, M, sc) for e in EST]
    hdr1 = [""] + ["\\multicolumn{3}{c}{S2: persistent errors ($\\rho=0.85$)}", "\\multicolumn{3}{c}{S3: no cointegration (truth = 0)}"]
    hdr2 = ["Estimator", "Bias", "RMSE", "Cov.\\%", "Median", "IQR-s.d.", "Rej.\\%"]
    note = (note_base + " In S3 the table reports the median estimate, a robust dispersion measure (IQR/1.349, because PMG and MG have fat tails when there is no "
            "error-correction relation) and the rejection rate (\\%) of H$_0$: slope $=0$ at the nominal 5\\% level, i.e.\\ the false-positive rate of a spurious relation.")
    (TAB / "tab_mc_s3.tex").write_text(tabular(rows, "lrrrrrr", header=[hdr1, hdr2], note=note, resize=True), encoding="utf-8")

    # ---------- Table: by T ----------
    def by_T(s, col, fmtf):
        return [[dn(e)] + [fmtf(M[s][T].loc[e, col]) if T in M[s] else "--" for T in TS] for e in EST]
    rows = [["\\textit{S1N: bias (exogenous regressors)}"] + [""] * len(TS)] + by_T("S1N", "bias", lambda v: f"{v:+.3f}")
    rows += [["\\textit{S1: bias (endogenous regressors)}"] + [""] * len(TS)] + by_T("S1", "bias", lambda v: f"{v:+.3f}")
    rows += [["\\textit{S3: false-positive rate (\\%), nominal 5\\%}"] + [""] * len(TS)] + by_T("S3", "reject0", lambda v: f"{100 * v:.0f}")
    note = f"Slope on the digital regressor, $N=17$, {nrep:,} replications; true value 0.30 (S1N, S1) and 0 (S3). ``--'' = not simulated."
    (TAB / "tab_mc_T.tex").write_text(tabular(rows, "l" + "r" * len(TS), header=[["Estimator"] + [f"$T={T}$" for T in TS]],
                                              midrules=[len(EST), 2 * len(EST) + 1], note=note), encoding="utf-8")

    # ---------- Tables: heterogeneity, factor loadings, measurement error ----------
    def grp(scens, names, fname, cap_note):
        rows = [[dn(e)] + cells(None, e, M, scens) for e in EST]
        h1 = [""] + ["\\multicolumn{3}{c}{%s}" % t for t in names]
        h2 = ["Estimator"] + ["Bias", "RMSE", "Cov.\\%"] * len(scens)
        (TAB / fname).write_text(tabular(rows, "l" + "rrr" * len(scens), header=[h1, h2], note=cap_note, resize=True), encoding="utf-8")
    note_h = (note_base.replace("true value 0.30 except in S3", "true value 0.30 = mean of the country slopes") +
              " S1H/S2H: country slopes $\\theta_{ik}=\\theta_k+\\sigma_k\\nu_{ik}$ with $\\sigma=(0.12,0.15,0.20,0.13)$, calibrated to the dispersion of the unit-level CCE-MG and CS-DL estimates; "
              "S1C: as S1H but the slope deviations are correlated ($\\rho=0.7$) with the scale of each regressor and with the timing of adoption; S1F: the global-trend loading of $y$ is correlated with the timing of adoption; S1M: classical measurement error (s.d.\\ 0.05) in the observed adoption share; "
              "S4: all three together.")
    grp(["S1H", "S1C", "S1F"], ["S1H: heterogeneous slopes", "S1C: slopes correlated with regressors", "S1F: loadings tied to adoption"], "tab_mc_hetero.tex", note_h)
    grp(["S1M", "S2H", "S4"], ["S1M: measurement error", "S2H: persistent errors, heterog. slopes", "S4: combined"], "tab_mc_hetero2.tex", note_h)

    # extra estimators (dynamic augmentations) at T = 30 so that all of them are feasible
    MX = {s_: stats_cell(s_, 30, extra=True) for s_ in ["S1N", "S1", "S4"]}
    rows = []
    for e in EST_X:
        r = [dn(e)]
        for s_ in ["S1N", "S1", "S4"]:
            x = MX[s_].loc[e]
            r += [f"{x['bias']:+.3f}", f"{x['rmse']:.3f}", f"{100 * x['cover']:.0f}"]
        rows.append(r)
    for e in ["CCE-MG", "CS-DL", "DOLS-GM"]:                                           # baseline counterparts at the same T
        r = [dn(e) + " (baseline)"]
        for s_ in ["S1N", "S1", "S4"]:
            x = M[s_][30].loc[e]
            r += [f"{x['bias']:+.3f}", f"{x['rmse']:.3f}", f"{100 * x['cover']:.0f}"]
        rows.append(r)
    note = (f"$N=17$, $T=30$ (at $T=24$ the CCE-MG with $p_T=2$ lags has 2 residual d.f.\\ and is not computed), {nrep:,} replications; true value 0.30. "
            "L1/L2 = one / two lags of the cross-sectional averages; DOLS--GM (1,1) = one lead and one lag of the differenced regressors. "
            "Baselines: static CCE-MG, CS-DL with contemporaneous averages, DOLS--GM with contemporaneous differences.")
    (TAB / "tab_mc_extra.tex").write_text(tabular(rows, "lrrrrrrrrr", header=[[""] + ["\\multicolumn{3}{c}{S1N}", "\\multicolumn{3}{c}{S1}", "\\multicolumn{3}{c}{S4}"],
                                                                         ["Estimator"] + ["Bias", "RMSE", "Cov.\\%"] * 3], midrules=[3], note=note, resize=True), encoding="utf-8")
    for e, key in [("CCE-MG L1", "CceLone"), ("CCE-MG L2", "CceLtwo"), ("DOLS-GM (1,1)", "DolsgmFull"), ("CS-DL L1", "CsdlLone")]:
        for s_, tag in [("S1", "Sa"), ("S4", "Sq")]:
            x = MX[s_].loc[e]
            set_macros({f"mcx{tag}{key}Bias": f"{x['bias']:+.2f}", f"mcx{tag}{key}Rmse": f"{x['rmse']:.2f}", f"mcx{tag}{key}Cov": f"{100 * x['cover']:.0f}"})

    # ---------- Dispersion across estimators within a replication ----------
    emp = pd.read_csv(TAB / "estimates_inet.csv").set_index("estimator")["b"]
    emp_range = emp[EMP8].max() - emp[EMP8].min()
    emp_sd = emp[EMP8].std(ddof=1)
    rows, disp = [], {}
    idx = [EST.index(e) for e in EMP8]
    for s in SCEN:
        a = load(s, 24)
        b, se = a[:, idx, 0, J], a[:, idx, 1, J]
        ok = np.isfinite(b).all(1)
        b, se, a9 = b[ok], se[ok], a[ok]
        rng_ = b.max(1) - b.min(1)
        sig_pos = (b / se > 1.96)
        mixed = sig_pos.any(1) & (~sig_pos.all(1))
        signdis = ((b > 0).any(1) & (b < 0).any(1))
        allt = np.concatenate([a9[:, [EST.index("TWFE")], 0, J], b], 1) / np.concatenate([a9[:, [EST.index("TWFE")], 1, J], se], 1)
        pick = (allt > 1.96).any(1)
        disp[s] = rng_
        joint = (rng_ >= emp_range) & (b > 0).all(1)
        allpos = (b > 0).all(1)
        # the empirical ordering: every CCE-type/CS-DL estimate below every pooled/error-correction estimate
        cce_fam = a9[:, [EST.index("CCE-MG"), EST.index("CCE-P"), EST.index("CS-DL")], 0, J]
        pool_fam = a9[:, [EST.index("TWFE"), EST.index("DOLS pooled"), EST.index("PMG")], 0, J]
        order_emp = cce_fam.max(1) < pool_fam.min(1)
        order_rev = cce_fam.min(1) > pool_fam.max(1)
        full = order_emp & (rng_ >= emp_range) & (b > 0).all(1)
        rows.append([SC_LAB[s], f"{np.median(rng_):.2f}", f"{(rng_ >= emp_range).mean() * 100:.0f}", f"{100 * mixed.mean():.0f}", f"{100 * signdis.mean():.0f}",
                     f"{100 * order_emp.mean():.0f}", f"{100 * full.mean():.1f}", f"{100 * pick.mean():.0f}"])
        set_macros({f"dispMed{MN[s]}": f"{np.median(rng_):.2f}", f"dispP{MN[s]}": f"{(rng_ >= emp_range).mean() * 100:.0f}",
                    f"mixedP{MN[s]}": f"{100 * mixed.mean():.0f}", f"pickP{MN[s]}": f"{100 * pick.mean():.0f}",
                    f"signP{MN[s]}": f"{100 * signdis.mean():.0f}", f"jointP{MN[s]}": f"{100 * joint.mean():.1f}",
                    f"allposP{MN[s]}": f"{100 * allpos.mean():.0f}",
                    f"fullP{MN[s]}": f"{100 * full.mean():.1f}", f"orderP{MN[s]}": f"{100 * order_emp.mean():.0f}", f"orderRevP{MN[s]}": f"{100 * order_rev.mean():.0f}"})
    note = (f"$N=17$, $T=24$, {nrep:,} replications. All entries except the first are percentages of replications. Range = maximum minus minimum of the digital slope across the eight "
            f"long-run estimators of the empirical table; the empirical range is {emp_range:.2f}. P(range) = share with a range at least as large as the empirical one. "
            "Mixed = at least one estimator is significantly positive (5\\%) and at least one is not. Sign = estimates of both signs. "
            "Order = the empirical ordering holds, i.e.\\ the three CCE-type estimates (CCE--MG, CCE--P, CS--DL) are all below the three pooled or error-correction estimates "
            "(TWFE, pooled DOLS, PMG). Full = range at least as large as the empirical one, all estimates positive and the empirical ordering together. "
            "Best-of-9 = at least one of the nine estimators (including TWFE) is significantly positive, i.e.\\ the success rate of an analyst who reports the most favourable estimator "
            "(the nominal rate for a single estimator in S3 is 5\\%).")
    (TAB / "tab_mc_dispersion.tex").write_text(tabular(
        rows, "p{5.4cm}rrrrrrr", header=[["Scenario", "Median range", "P(range)", "Mixed", "Sign", "Order", "Full", "Best-of-9"]],
        note=note, resize=False), encoding="utf-8")
    set_macros({"empRange": f"{emp_range:.2f}", "empSd": f"{emp_sd:.2f}", "empMax": f"{emp[EMP8].max():.2f}", "empMin": f"{emp[EMP8].min():.3f}",
                "empMaxPct": f"{100 * 0.1 * emp[EMP8].max():.1f}"})

    # ---------- macros ----------
    keys = [("CCE-MG", "Cce"), ("CCE-P", "Ccep"), ("PMG", "Pmg"), ("DOLS pooled", "Dols"), ("TWFE", "Twfe"), ("FMOLS-GM", "Fmols"),
            ("MG", "Mg"), ("CS-DL", "Csdl"), ("DOLS-GM", "Dolsgm")]
    for s in SCEN:
        for e, key in keys:
            for T in (24, 100):
                if T not in M[s]:
                    continue
                d = M[s][T].loc[e]
                set_macros({f"mc{MN[s]}{key}Bias{MN[T]}": f"{d['bias']:+.2f}", f"mc{MN[s]}{key}Med{MN[T]}": f"{d['medbias']:+.2f}",
                            f"mc{MN[s]}{key}Cov{MN[T]}": f"{100 * d['cover']:.0f}", f"mc{MN[s]}{key}Rej{MN[T]}": f"{100 * d['reject0']:.0f}",
                            f"mc{MN[s]}{key}Rmse{MN[T]}": f"{d['rmse']:.2f}", f"mc{MN[s]}{key}Ratio{MN[T]}": f"{d['ratio']:.2f}"})

    # ---------- Figures ----------
    ps.setup()
    fig, axs = plt.subplots(1, 4, figsize=(7.6, 2.9), sharey=True)
    for ax, s in zip(axs, ["S1N", "S1", "S1U", "S2"]):
        for e in EST:
            Ts = TS_BY[s]
            ax.plot([TS.index(T) for T in Ts], [M[s][T].loc[e, "bias"] for T in Ts], "-o", ms=2.5, lw=1.1, color=COL[e], label=e if s == "S1N" else None)
        ax.axhline(0, color=ps.DARK, lw=0.8)
        ax.set_xticks(range(len(TS))); ax.set_xticklabels(TS, fontsize=7)
        ax.set_xlabel("$T$"); ax.set_title(s, fontsize=8.5); ax.set_ylim(-0.08, 0.20)
    axs[0].set_ylabel("Bias of the digital slope")
    fig.legend(loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.14), fontsize=7)
    fig.tight_layout()
    ps.save(fig, FIG / "fig_mc_bias_T.pdf")

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for e in EST:
        axs[0].plot(range(len(TS)), [100 * M["S3"][T].loc[e, "reject0"] for T in TS], "-o", ms=3, lw=1.2, color=COL[e], label=e)
        axs[1].plot(range(len(TS)), [100 * M["S1"][T].loc[e, "cover"] for T in TS], "-o", ms=3, lw=1.2, color=COL[e])
    axs[0].axhline(5, color=ps.DARK, lw=0.8, ls="--"); axs[1].axhline(95, color=ps.DARK, lw=0.8, ls="--")
    for ax, ttl, yl in [(axs[0], "(a) S3, no cointegration: false-positive rate", "Rejection rate of H$_0$: slope = 0 (%)"),
                        (axs[1], "(b) S1, endogenous regressors: 95% coverage", "Coverage (%)")]:
        ax.set_xticks(range(len(TS))); ax.set_xticklabels(TS); ax.set_xlabel("$T$"); ax.set_title(ttl, fontsize=8.5); ax.set_ylabel(yl)
    axs[0].legend(fontsize=6.5, ncol=1, loc="upper left")
    fig.tight_layout()
    ps.save(fig, FIG / "fig_mc_size_cover.pdf")

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, s in zip(axs, ["S1", "S3"]):
        a = load(s, 24)
        data = [a[:, EST.index(e), 0, J] for e in EST]
        data = [d[np.isfinite(d)] for d in data]
        data = [np.clip(d, -1.5, 1.5) for d in data]
        bp = ax.boxplot(data, vert=False, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.6)
        for patch, e in zip(bp["boxes"], EST):
            patch.set_facecolor(COL[e]); patch.set_alpha(0.6)
        ax.axvline(truth(s), color=ps.DARK, lw=1.0)
        ax.set_yticks(range(1, len(EST) + 1)); ax.set_yticklabels(EST, fontsize=7)
        ax.set_title(SC_LAB[s], fontsize=8); ax.set_xlabel("Estimated slope (clipped at $\\pm1.5$)")
    ps.save(fig, FIG / "fig_mc_dist_T24.pdf")

    sc_plot = ["S1", "S1H", "S1F", "S1M", "S4"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for ie, e in enumerate(EST):
        ax.plot([M[s_][24].loc[e, "bias"] for s_ in sc_plot], range(len(sc_plot)), "o", color=COL[e], ms=5, label=e)
    ax.axvline(0, color=ps.DARK, lw=0.8)
    short = {"S1": "S1: endogenous regressors", "S1H": "S1H: heterogeneous slopes", "S1F": "S1F: loadings tied to adoption",
             "S1M": "S1M: measurement error", "S4": "S4: all three combined"}
    ax.set_yticks(range(len(sc_plot))); ax.set_yticklabels([short[s_] for s_ in sc_plot], fontsize=7.5)
    ax.set_xlim(-0.25, 0.30)
    ax.set_xlabel("Bias of the digital slope at $T=24$ (true mean slope 0.30)\n(MG, with mean bias above 0.25 in several scenarios, is outside the axis)")
    ax.legend(fontsize=6.5, ncol=3, loc="lower right"); ax.invert_yaxis()
    ps.save(fig, FIG / "fig_mc_hetero.pdf")

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    cols = {"S1N": ps.GREEN, "S1": ps.BLUE, "S3": ps.ORANGE}
    for s in ["S1N", "S1", "S3"]:
        ax.hist(np.clip(disp[s], 0, 2.0), bins=np.linspace(0, 2.0, 41), alpha=0.45, color=cols[s], label=SC_LAB[s].split(":")[0] + (" (no cointegration)" if s == "S3" else ""), density=True)
    ax.axvline(emp_range, color=ps.RED, lw=1.6, label=f"Empirical range ({emp_range:.2f})")
    ax.set_xlabel("Range of the digital slope across eight estimators"); ax.set_ylabel("Density"); ax.legend(fontsize=7)
    ps.save(fig, FIG / "fig_mc_dispersion.pdf")
    print("summary done; empirical range", round(emp_range, 3))
    print(pd.concat({s: M[s][24][["bias", "medbias", "rmse", "cover", "reject0"]] for s in SCEN}, axis=1).round(3).to_string())
