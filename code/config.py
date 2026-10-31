"""Global configuration for the replication package.

Every script imports this module so that paths, the country list, the sample
window and the random seed are defined in exactly one place.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "output" / "figures"
TAB = ROOT / "output" / "tables"
LOG = ROOT / "output" / "logs"
for _p in (RAW, PROC, FIG, TAB, LOG):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 20261028

# --- Universe: the 20 sovereign states of Latin America (UN geoscheme + Haiti) ---
LATAM = {
    "ARG": "Argentina", "BOL": "Bolivia", "BRA": "Brazil", "CHL": "Chile",
    "COL": "Colombia", "CRI": "Costa Rica", "CUB": "Cuba",
    "DOM": "Dominican Republic", "ECU": "Ecuador", "SLV": "El Salvador",
    "GTM": "Guatemala", "HTI": "Haiti", "HND": "Honduras", "MEX": "Mexico",
    "NIC": "Nicaragua", "PAN": "Panama", "PRY": "Paraguay", "PER": "Peru",
    "URY": "Uruguay", "VEN": "Venezuela",
}
AGGREGATES = {"WLD": "World"}

# --- Sample window ---
YEAR0, YEAR1 = 2000, 2023

# --- World Bank WDI indicators (code -> short name) ---
WDI = {
    "NY.GDP.PCAP.KD":    "gdppc",       # GDP per capita, constant 2015 US$
    "NY.GDP.PCAP.PP.KD": "gdppc_ppp",   # GDP per capita, PPP, constant 2021 intl $
    "IT.NET.USER.ZS":    "inet",        # Individuals using the Internet (% pop.)
    "IT.CEL.SETS.P2":    "mobile",      # Mobile cellular subscriptions (per 100)
    "IT.NET.BBND.P2":    "fbb",         # Fixed broadband subscriptions (per 100)
    "IT.MLT.MAIN.P2":    "fixedtel",    # Fixed telephone subscriptions (per 100)
    "NE.GDI.FTOT.ZS":    "gfcf",        # Gross fixed capital formation (% GDP)
    "NE.GDI.TOTL.ZS":    "gcf",         # Gross capital formation (% GDP)
    "NE.TRD.GNFS.ZS":    "trade",       # Trade (% GDP)
    "NY.GDP.DEFL.KD.ZG": "infl",        # Inflation, GDP deflator (annual %)
    "SP.POP.TOTL":       "pop",         # Population
    "SE.SEC.ENRR":       "sec_gross",   # Gross secondary enrolment (%), only for the availability table
}

HDR_URL = ("https://hdr.undp.org/sites/default/files/2025_HDR/"
           "HDR25_Composite_indices_complete_time_series.csv")
HDR_FILE = RAW / "HDR25_Composite_indices_complete_time_series.csv"

# Baseline long-run vector: log-linear in GDP per capita, GFCF, schooling and trade; the digital variable
# enters as a share of population in [0,1] (semi-log, as in the broadband-growth literature).
Y_VAR = "ln_gdppc"
X_BASE = ["ln_gfcf", "ln_hc", "inet_s", "ln_trade"]
J_INET = X_BASE.index("inet_s")
X_LABELS = {
    "ln_gfcf": "ln GFCF (\% GDP)",
    "ln_hc": "ln mean years of schooling",
    "inet_s": "Internet users (share of pop.)",
    "ln_inet": "ln Internet users (\% pop.)",
    "ln_trade": "ln Trade openness (\% GDP)",
    "ln_gdppc": "ln GDP per capita",
    "mobile_s": "Mobile subscriptions per capita",
    "fbb_s": "Fixed broadband per capita",
}
