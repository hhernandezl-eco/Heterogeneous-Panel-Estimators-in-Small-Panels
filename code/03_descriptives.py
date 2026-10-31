"""Step 3 - Descriptive statistics and descriptive figures.

Outputs: tab_descriptives.tex, tab_country_summary.tex, fig_internet_paths.pdf,
         fig_scatter_levels_within.pdf, fig_gdp_internet_by_country.pdf
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from config import *
from lib import plotstyle as ps
from lib.util import tabular, set_macros
import lib.econ as E

ps.setup()
df = pd.read_csv(PROC / "panel_main.csv")

# ---------------- Table: descriptive statistics ----------------
spec = [("GDP per capita (constant 2015 US\\$)", "gdppc", 0), ("ln GDP per capita", "ln_gdppc", 2),
        ("GFCF (\\% of GDP)", "gfcf", 1), ("Mean years of schooling (25+)", "mys", 1),
        ("Internet users (\\% of population)", "inet", 1), ("Trade openness (\\% of GDP)", "trade", 1),
        ("Mobile subscriptions (per 100)", "mobile", 1), ("Fixed broadband (per 100)", "fbb", 1),
        ("Inflation, GDP deflator (\\%)", "infl", 1)]
rows = []
for lab, v, d in spec:
    x = df[v].dropna()
    # between / within standard deviations
    g = df.groupby("iso3")[v]
    between = g.mean().std()
    within = (df[v] - g.transform("mean")).std()
    rows.append([lab, str(len(x)), f"{x.mean():.{d}f}", f"{x.std():.{d}f}", f"{between:.{d}f}", f"{within:.{d}f}",
                 f"{x.min():.{d}f}", f"{x.max():.{d}f}"])
tex = tabular(rows, "lrrrrrrr",
              header=[["Variable", "Obs.", "Mean", "Std. dev.", "Between", "Within", "Min", "Max"]],
              note=f"Balanced panel of {df.iso3.nunique()} Latin American countries, {YEAR0}--{YEAR1}. "
                   "Between = s.d. of country means; within = s.d. of deviations from country means. "
                   "Sources: World Bank WDI; UNDP Human Development Report time series (schooling). "
                   "Fixed broadband and mobile subscriptions have fewer observations because early values are zero.")
(TAB / "tab_descriptives.tex").write_text(tex, encoding="utf-8")

# ---------------- Table: country summary ----------------
rows = []
for iso, g in df.groupby("iso3"):
    g = g.sort_values("year")
    cagr = (np.log(g.gdppc.iloc[-1]) - np.log(g.gdppc.iloc[0])) / (len(g) - 1) * 100
    rows.append([LATAM[iso], f"{g.gdppc.iloc[0]:,.0f}".replace(",", "\\,"), f"{g.gdppc.iloc[-1]:,.0f}".replace(",", "\\,"),
                 f"{cagr:.2f}", f"{g.inet.iloc[0]:.1f}", f"{g.inet.iloc[-1]:.1f}", f"{g.mys.iloc[-1]:.1f}"])
tex = tabular(rows, "lrrrrrr", resize=True,
              header=[["Country", f"GDPpc {YEAR0}", f"GDPpc {YEAR1}", "Growth (\\% p.a.)", f"Internet {YEAR0}", f"Internet {YEAR1}", f"Schooling {YEAR1}"]],
              note="GDP per capita in constant 2015 US dollars; growth = average annual log growth; Internet = share (\%) of the population using the Internet; "
                   "schooling = mean years of schooling of adults.")
(TAB / "tab_country_summary.tex").write_text(tex, encoding="utf-8")

# ---------------- Fig 1: internet trajectories ----------------
wide = df.pivot(index="year", columns="iso3", values="inet")
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for iso in wide:
    ax.plot(wide.index, wide[iso], color=ps.GREY, lw=0.8, alpha=0.7)
med = wide.median(axis=1)
ax.plot(wide.index, med, color=ps.BLUE, lw=2.2, label="Cross-country median")
q1, q3 = wide.quantile(.25, axis=1), wide.quantile(.75, axis=1)
ax.fill_between(wide.index, q1, q3, color=ps.BLUE, alpha=0.12, label="Interquartile range")
for iso in ("CHL", "HTI", "ARG", "URY"):
    ax.annotate(LATAM[iso], (wide.index[-1], wide[iso].iloc[-1]), xytext=(4, 0), textcoords="offset points",
                fontsize=7, va="center", color=ps.DARK)
ax.set_xlim(YEAR0, YEAR1 + 3.2)
ax.set_ylabel("Internet users (% of population)")
ax.legend(loc="upper left")
ps.save(fig, FIG / "fig_internet_paths.pdf")

# ---------------- Fig 2: pooled vs. within relation ----------------
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.3))
ax = axs[0]
cmap = plt.get_cmap("viridis")
sc = ax.scatter(df["inet_s"] * 100, df["ln_gdppc"], c=df["year"], cmap=cmap, s=8, alpha=0.85, linewidths=0)
ax.set_xlabel("Internet users (% of population)")
ax.set_ylabel("ln GDP per capita")
ax.set_title("(a) Levels (all country-years)")
cb = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.05)
cb.set_label("Year")
# (b) two-way demeaned
d2 = df.copy()
for v in ("ln_gdppc", "inet_s"):
    d2[v + "_w"] = (d2[v] - d2.groupby("iso3")[v].transform("mean") - d2.groupby("year")[v].transform("mean") + d2[v].mean())
ax = axs[1]
ax.scatter(d2["inet_s_w"] * 100, d2["ln_gdppc_w"], color=ps.ORANGE, s=8, alpha=0.6, linewidths=0)
b = np.polyfit(d2["inet_s_w"] * 100, d2["ln_gdppc_w"], 1)
xx = np.linspace(d2["inet_s_w"].min() * 100, d2["inet_s_w"].max() * 100, 50)
ax.plot(xx, np.polyval(b, xx), color=ps.DARK, lw=1.4)
ax.set_xlabel("Internet users, net of country and year effects (pp)")
ax.set_ylabel("ln GDP per capita, net of effects")
ax.set_title("(b) Two-way demeaned")
set_macros({"twfeBivariateSlope": f"{b[0] * 10:.3f}"})
ps.save(fig, FIG / "fig_scatter_levels_within.pdf")

# ---------------- Fig 3: small multiples GDP vs internet ----------------
isos = list(df.iso3.unique())
nc = 5
nr = int(np.ceil(len(isos) / nc))
fig, axs = plt.subplots(nr, nc, figsize=(7.2, 1.45 * nr), sharex=True)
for a, iso in zip(axs.ravel(), isos):
    g = df[df.iso3 == iso]
    a.plot(g.year, g.gdppc / g.gdppc.iloc[0] * 100, color=ps.BLUE, lw=1.2)
    a.set_title(LATAM[iso].replace("Dominican Republic", "Dominican Rep."), fontsize=7.5, pad=2)
    a.tick_params(labelsize=6.5)
    a2 = a.twinx()
    a2.plot(g.year, g.inet, color=ps.ORANGE, lw=1.2)
    a2.set_ylim(0, 100)
    a2.tick_params(labelsize=6.5, colors=ps.ORANGE)
    a2.spines["right"].set_visible(True); a2.spines["right"].set_color(ps.ORANGE)
    a2.grid(False)
for a in axs.ravel()[len(isos):]:
    a.axis("off")
axs[0, 0].plot([], [], color=ps.BLUE, label="GDP per capita (2000 = 100, left)")
axs[0, 0].plot([], [], color=ps.ORANGE, label="Internet users (%, right)")
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout()
ps.save(fig, FIG / "fig_gdp_internet_by_country.pdf")
print("descriptives done")
