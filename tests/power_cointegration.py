"""Size and power of the panel cointegration tests at the dimensions of the application (N=17, T=24),
using the same data-generating processes as the main Monte-Carlo study (code/10_montecarlo.py):
bounded logistic digital regressor, common I(1) trend and stationary factor, endogenous regressors.

  S1  cointegration, rho=0.5   S2  cointegration, rho=0.85   S1U  as S1, unbounded regressor   S3  no cointegration

Usage: python tests/power_cointegration.py [R] [B]    (R replications per scenario, B bootstrap draws for Westerlund)
Output: output/tables/tab_power.tex, power_cointegration.csv
"""
import sys, os, time, importlib
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from lib import econ as E
from lib.util import tabular
from config import TAB, SEED

mc = importlib.import_module("10_montecarlo")
T = 24
NAMES = ["Kao", "Pedroni group-ADF", "Pedroni panel-ADF", "Westerlund Gt", "Westerlund Pt"]
SCEN = [("S3", "No cointegration (size)"), ("S1", "Cointegration, $\\rho=0.5$"), ("S1U", "Cointegration, unbounded regressor"),
        ("S2", "Cointegration, $\\rho=0.85$"), ("S1H", "Cointegration, heterogeneous slopes"),
        ("S2H", "Cointegration, $\\rho=0.85$, heterogeneous slopes"), ("S4", "Heterogeneity + loadings + measurement error")]


def one(args):
    scen, r, B = args
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 99, r, {"S1": 1, "S2": 2, "S3": 3, "S1U": 4, "S1H": 5, "S2H": 6, "S4": 7}[scen]]))
    Y, X = mc.simulate(rng, T, scen)
    Yd, Xd = E.csd_demean(Y), E.csd_demean(X)
    k = E.kao_adf(Y, X, p=1)["p"]
    pe = E.pedroni_adf(Y, X, p=0, R=1000)
    we = E.westerlund(Yd, Xd, p=0, B=B, seed=r)
    return [k < .05, pe["p_group"] < .05, pe["p_panel"] < .05, we["p_Gt"] < .05, we["p_Pt"] < .05]


def main(R=500, B=99):
    out = []
    t0 = time.time()
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for scen, lab in SCEN:
            res = np.array(pool.map(one, [(scen, r, B) for r in range(R)]), float).mean(0)
            print(scen, dict(zip(NAMES, res.round(2))), f"({time.time()-t0:.0f}s)", flush=True)
            out.append([lab] + list(res))
    pd.DataFrame(out, columns=["Scenario"] + NAMES).to_csv(TAB / "power_cointegration.csv", index=False)
    rows = [[r[0]] + [f"{v:.2f}" for v in r[1:]] for r in out]
    note = (f"Rejection frequencies at the nominal 5\\% level over {R} Monte-Carlo replications, $N=17$, $T={T}$, $K=4$. Data-generating processes as in the main Monte-Carlo "
            f"study (Section~\\ref{{sec:mc}}). Westerlund p-values use {B} bootstrap draws per replication; Pedroni-type p-values use 1,000 simulated null draws.")
    (TAB / "tab_power.tex").write_text(tabular(rows, "lccccc", header=[["Scenario"] + NAMES], midrules=[0], note=note, resize=True), encoding="utf-8")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500, int(sys.argv[2]) if len(sys.argv) > 2 else 99)
