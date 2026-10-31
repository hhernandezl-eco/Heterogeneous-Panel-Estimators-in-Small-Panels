"""Step 2 - Build the analysis panels from the raw files.

Outputs
  data/processed/panel_main.csv   balanced panel, N countries x 24 years, no imputation
  data/processed/panel_ext.csv    extended panel (adds Nicaragua; GCF instead of GFCF; no imputation)
  data/processed/panel_wide_all.csv   everything, incl. incomplete countries (for the availability table)
  output/tables/tab_availability.tex  data-availability table (sample-selection rule)
"""
import numpy as np, pandas as pd
from config import *
from lib.util import set_macros, tabular

wdi = pd.read_csv(RAW / "wdi_raw.csv")
wdi["name"] = wdi["indicator"].map(WDI)
wide = wdi.pivot_table(index=["iso3", "year"], columns="name", values="value").reset_index()

# --- UNDP mean years of schooling (population 25+) -> long ---
hdr = pd.read_csv(HDR_FILE, encoding="latin1")
cols = [f"mys_{y}" for y in range(1990, 2024)]
h = hdr[hdr["iso3"].isin(list(LATAM))][["iso3"] + cols].melt("iso3", var_name="v", value_name="mys")
h["year"] = h["v"].str[4:].astype(int)
h = h.drop(columns="v")
wide = wide.merge(h, on=["iso3", "year"], how="left")
wide["country"] = wide["iso3"].map({**LATAM, **AGGREGATES})

# --- Instrument ingredients (time-invariant 1995 baselines, frontier diffusion) ---
base = wide[wide.year == 1995].set_index("iso3")
wide["tel95"] = wide["iso3"].map(base["fixedtel"])
wide["gdppc95"] = wide["iso3"].map(base["gdppc"])
# Frontier diffusion: mean Internet use of high-income economies outside Latin America with complete 2000-2023 data
allnet = pd.read_csv(RAW / "wdi_internet_all.csv", keep_default_na=False, na_values=[""])
cm = pd.read_csv(RAW / "wb_countries.csv", keep_default_na=False)
hic_ids = cm[(cm.income == "HIC") & (cm.region != "LCN") & (cm.region != "NA")]["iso3"]
fr = allnet[allnet.iso3.isin(hic_ids) & allnet.year.between(YEAR0, YEAR1)].dropna(subset=["value"])
cnt = fr.groupby("iso3").size()
fr = fr[fr.iso3.isin(cnt[cnt == YEAR1 - YEAR0 + 1].index)]
N_FRONTIER = fr.iso3.nunique()
frontier = fr.groupby("year")["value"].mean()
print("frontier economies:", N_FRONTIER)
wide["inet_hic"] = wide["year"].map(frontier)

win = wide[(wide.year >= YEAR0) & (wide.year <= YEAR1) & wide.iso3.isin(list(LATAM))].copy()
# WDI reports 0.0 (instead of null) for some Venezuelan national-accounts ratios 2000-2011: treat as missing
for v in ("gdppc", "gdppc_ppp", "gfcf", "gcf", "trade", "mys"):
    win.loc[win[v] <= 0, v] = np.nan
win.to_csv(PROC / "panel_wide_all.csv", index=False)

# ---------- Availability table / sample-selection rule ----------
T = YEAR1 - YEAR0 + 1
need = ["gdppc", "gfcf", "mys", "inet", "trade"]
avail = win.groupby("iso3")[need + ["fbb", "mobile", "infl", "sec_gross"]].count()
complete = avail[need].eq(T).all(axis=1)
main_iso = [c for c in LATAM if complete.get(c, False) and c != "CUB"]
print("complete on baseline variables:", [c for c in LATAM if complete.get(c, False)])
print("main sample:", len(main_iso), main_iso)

# Cuba is excluded by design (non-market statistical system -> WDI GDP not comparable; internet access
# heavily restricted).  Venezuela (12 years of missing GFCF/trade, 2 of internet use) and Nicaragua (GFCF missing
# 2000-05) fail the completeness rule; Nicaragua re-enters in the extended sample through GCF.
rows = []
for c in LATAM:
    a = avail.loc[c]
    status = "Main" if c in main_iso else ("Robustness" if c == "NIC" else "Excluded")
    rows.append([LATAM[c]] + [str(int(a[v])) for v in ["gdppc", "gfcf", "mys", "inet", "trade", "sec_gross", "fbb"]] + [status])
tex = tabular(rows, "lrrrrrrrl",
              header=[["Country", "GDPpc", "GFCF", "Schooling", "Internet", "Trade", "Sec. enrol.", "Fixed BB", "Use"]],
              note=f"Number of non-missing annual observations in {YEAR0}--{YEAR1} ({T} possible). "
                   "Schooling = UNDP mean years of schooling (25+); Sec. enrol. = WDI gross secondary enrolment "
                   f"(shown to justify not using it: it is incomplete for {int((avail['sec_gross'] < T).sum())} of the {len(LATAM)} countries). "
                   "Main sample = countries with complete data on the five baseline variables, excluding Cuba "
                   "(non-comparable national accounts). Nicaragua enters only the extended sample (gross capital "
                   "formation replaces GFCF). Venezuela is dropped (WDI reports no GFCF or trade data for 2000--2011).")
(TAB / "tab_availability.tex").write_text(tex, encoding="utf-8")


# ---------- Transformations ----------
def transform(df):
    d = df.copy()
    for v in ("fbb", "mobile"):          # zeros in the first years -> log undefined -> NaN
        d.loc[d[v] <= 0, v] = np.nan
    d["ln_gdppc"] = np.log(d["gdppc"])
    d["ln_gdppc_ppp"] = np.log(d["gdppc_ppp"])
    d["ln_gfcf"] = np.log(d["gfcf"])
    d["ln_gcf"] = np.log(d["gcf"])
    d["ln_hc"] = np.log(d["mys"])
    d["ln_inet"] = np.log(d["inet"])
    d["inet_s"] = d["inet"] / 100.0          # share of population using the Internet
    d["mobile_s"] = d["mobile"] / 100.0      # subscriptions per inhabitant
    d["fbb_s"] = d["fbb"] / 100.0
    d["ln_trade"] = np.log(d["trade"])
    d["ln_mobile"] = np.log(d["mobile"])
    d["ln_fbb"] = np.log(d["fbb"])
    d["infl_l"] = np.log1p(d["infl"] / 100.0)
    d["ln_pop"] = np.log(d["pop"])
    d["ln_inet_hic"] = np.log(d["inet_hic"])
    d["ln_tel95"] = np.log(d["tel95"])
    d["ln_gdppc95"] = np.log(d["gdppc95"])
    return d.sort_values(["iso3", "year"]).reset_index(drop=True)


main = transform(win[win.iso3.isin(main_iso)])
assert main[["ln_gdppc", "ln_gfcf", "ln_hc", "ln_inet", "ln_trade"]].notna().all().all()
assert len(main) == len(main_iso) * T
main.to_csv(PROC / "panel_main.csv", index=False)

# Extended sample: GCF (instead of GFCF) as capital proxy so that Nicaragua can be added; no imputation
ext_iso = main_iso + ["NIC"]
e = transform(win[win.iso3.isin(ext_iso)])
e["ln_gfcf"] = e["ln_gcf"]          # capital proxy in the extended sample is GCF
need_ext = ["ln_gdppc", "ln_gfcf", "ln_hc", "ln_inet", "ln_trade"]
assert e[need_ext].notna().all().all(), e[need_ext].isna().sum()
e.to_csv(PROC / "panel_ext.csv", index=False)

set_macros({"nFrontier": N_FRONTIER, "nCountries": len(main_iso), "nTime": T, "nObs": len(main), "yearStart": YEAR0, "yearEnd": YEAR1,
            "nCountriesExt": len(ext_iso), "nUniverse": len(LATAM)})
print("ext sample:", len(ext_iso))
print(main.groupby("iso3")[["tel95"]].first().T.round(2).to_string())
print(avail.to_string())
