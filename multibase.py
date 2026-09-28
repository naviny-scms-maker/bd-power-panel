"""multibase.py - 2026 compared with a multi-year baseline (several control years with year FE), not 2025 alone."""
import sys, os, numpy as np, pandas as pd, statsmodels.formula.api as smf
HOURLY, PANEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
h = pd.read_csv(HOURLY, parse_dates=["date"])
h = pd.DataFrame({"date": h.date, "gas": h.gas_mw * .024, "unserved": h.ls_mean_mw * .024, "coal": h.coal_mw * .024, "oil": h.oil_mw * .024})
s = pd.read_csv(PANEL, parse_dates=["data_date"])
s = pd.DataFrame({"date": s.data_date, "gas": s.pgcb_gen_gas_mkwh, "unserved": s.energy_unserved_mkwh,
                  "coal": s.pgcb_gen_coal_mkwh, "oil": s.pgcb_gen_hfo_mkwh.fillna(0) + s.pgcb_gen_hsd_mkwh.fillna(0)})
d = pd.concat([h[h.date < "2025-01-01"], s[s.date >= "2025-01-01"]]).sort_values("date")
d["year"] = d.date.dt.year; d["doy"] = d.date.dt.dayofyear; d = d[d.doy <= 264]
md = d.date.dt.strftime("%m-%d")
rows = []
for base in [(2025,), (2024, 2025), (2023, 2024, 2025), (2022, 2023, 2024, 2025), (2019, 2021, 2023, 2024, 2025)]:
    x = d[d.year.isin(base + (2026,))].copy(); mdx = x.date.dt.strftime("%m-%d")
    t = (x.year == 2026).astype(int)
    x["war"] = t * ((mdx >= "03-01") & (mdx <= "07-20")).astype(int)
    x["fsru"] = t * ((mdx >= "07-21") & (mdx <= "08-31")).astype(int)
    x["late"] = t * (mdx >= "09-01").astype(int)
    for y in ["unserved", "gas", "coal", "oil"]:
        m = smf.ols(f"{y} ~ C(doy) + C(year) + war + fsru + late", x.dropna(subset=[y])).fit(cov_type="HAC", cov_kwds={"maxlags": 14})
        for k in ("war", "fsru"):
            rows.append(dict(baseline="+".join(map(str, base)), outcome=y, window=k, coef=m.params[k], se=m.bse[k], p=m.pvalues[k]))
r = pd.DataFrame(rows); r.to_csv(os.path.join(OUT, "table7_multiyear_baseline.csv"), index=False)
r["c"] = r.coef.round(2).astype(str) + " (" + r.se.round(2).astype(str) + ")" + np.where(r.p < .01, "***", np.where(r.p < .05, "**", np.where(r.p < .1, "*", "")))
pd.set_option("display.width", 220)
print(r.pivot_table(index="baseline", columns=["outcome", "window"], values="c", aggfunc="first").to_string())
