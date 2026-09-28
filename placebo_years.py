"""
placebo_years.py - placebo-in-time test using 11 years of daily PGCB data.
For each year Y (2016-2026) estimate the same difference-in-differences as Table 1:
Y vs Y-1, same calendar day, change in 1 Mar-20 Jul (and 21 Jul-31 Aug) relative to Jan-Feb.
2015-2024: PGCB hourly data (Mendeley vpk8spw2mm) aggregated to days; 2025-2026: BPDB/PGCB panel.
In 2025 the two sources correlate at 0.99 for gas generation and 0.998 for unserved energy.
"""
import sys, os, numpy as np, pandas as pd, statsmodels.formula.api as smf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HOURLY, PANEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
h = pd.read_csv(HOURLY, parse_dates=["date"])
h = pd.DataFrame({"date": h.date, "gas": h.gas_mw * .024, "coal": h.coal_mw * .024, "oil": h.oil_mw * .024,
                  "unserved": h.ls_mean_mw * .024})
s = pd.read_csv(PANEL, parse_dates=["data_date"])
s = pd.DataFrame({"date": s.data_date, "gas": s.pgcb_gen_gas_mkwh, "coal": s.pgcb_gen_coal_mkwh,
                  "oil": s.pgcb_gen_hfo_mkwh.fillna(0) + s.pgcb_gen_hsd_mkwh.fillna(0), "unserved": s.energy_unserved_mkwh})
d = pd.concat([h[h.date < "2025-01-01"], s[s.date >= "2025-01-01"]]).sort_values("date")
d["year"] = d.date.dt.year; d["doy"] = d.date.dt.dayofyear
d = d[d.doy <= 264]                                   # up to 21 Sep, the last 2026 day
rows = []
for Y in range(2016, 2027):
    x = d[d.year.isin([Y - 1, Y])].copy()
    x["yY"] = (x.year == Y).astype(int)
    md = x.date.dt.strftime("%m-%d")
    x["w1"] = x.yY * ((md >= "03-01") & (md <= "07-20")).astype(int)
    x["w2"] = x.yY * ((md >= "07-21") & (md <= "08-31")).astype(int)
    x["w3"] = x.yY * (md >= "09-01").astype(int)
    for y in ["unserved", "gas", "coal", "oil"]:
        z = x.dropna(subset=[y])
        if z.yY.nunique() < 2 or z[z.yY == 1].w1.sum() < 30: continue
        m = smf.ols(f"{y} ~ C(doy) + yY + w1 + w2 + w3", z).fit(cov_type="HAC", cov_kwds={"maxlags": 14})
        for t, lab in [("w1", "Mar-20 Jul"), ("w2", "21 Jul-31 Aug")]:
            rows.append(dict(year=Y, outcome=y, window=lab, coef=m.params[t], se=m.bse[t], p=m.pvalues[t], n=int(m.nobs)))
r = pd.DataFrame(rows); r.to_csv(os.path.join(OUT, "table6_placebo_years.csv"), index=False)
lab = {"unserved": "Energy not served (MkWh/day)", "gas": "Gas-fired generation (MkWh/day)",
       "coal": "Coal-fired generation (MkWh/day)", "oil": "Oil-fired generation (MkWh/day)"}
fig, axs = plt.subplots(2, 2, figsize=(10, 7)); axs = axs.ravel()
for a, y in zip(axs, lab):
    z = r[(r.outcome == y)]
    for k, (w, mk) in enumerate([("Mar-20 Jul", "o"), ("21 Jul-31 Aug", "s")]):
        zz = z[z.window == w]
        col = ["C3" if yr == 2026 else ("C1" if yr == 2022 else "0.4") for yr in zz.year]
        a.errorbar(zz.year + (k - .5) * .25, zz.coef, yerr=1.96 * zz.se, fmt="none", ecolor="0.75", lw=1)
        a.scatter(zz.year + (k - .5) * .25, zz.coef, c=col, marker=mk, s=28, zorder=3, label=w)
    a.axhline(0, color="k", lw=.7); a.set_title(lab[y], loc="left", fontsize=9); a.grid(alpha=.3)
    a.set_xticks(range(2016, 2027)); a.tick_params(axis="x", labelsize=7, rotation=45)
axs[0].legend(fontsize=7, title="Window (vs Jan-Feb)", title_fontsize=7)
fig.suptitle("Placebo years: same estimate for every year vs the previous year (red = 2026 war year; orange = 2022 LNG crisis)\n"
             "Circles: 1 Mar-20 Jul; squares: 21 Jul-31 Aug. 95% CI, HAC. Day-of-year FE.", x=.02, ha="left", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(os.path.join(OUT, "fig4_placebo_years.png"), dpi=200)
pd.set_option("display.width", 200)
print(r.assign(c=r.coef.round(2).astype(str) + np.where(r.p < .01, "***", np.where(r.p < .05, "**", np.where(r.p < .1, "*", ""))))
      .pivot_table(index="year", columns=["outcome", "window"], values="c", aggfunc="first").to_string())
