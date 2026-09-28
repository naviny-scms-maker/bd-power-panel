"""
robustness.py - robustness checks for Table 1 (system-level 2026 vs 2025 difference-in-differences).
"""
import sys, os, numpy as np, pandas as pd, statsmodels.formula.api as smf, pyfixest as pf
IN, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
WAR, FSRU, LATE, FERT = map(pd.Timestamp, ["2026-03-01", "2026-07-21", "2026-09-01", "2026-05-01"])
RAMADAN = {2025: ("2025-03-02", "2025-03-30"), 2026: ("2026-02-19", "2026-03-20")}
EIDS = ["2025-03-31", "2025-06-07", "2026-03-21", "2026-05-28"]

s = pd.read_csv(os.path.join(IN, "system_day_panel.csv"), parse_dates=["data_date"])
p = pd.read_csv(os.path.join(IN, "plant_day_panel.csv"), parse_dates=["data_date"], low_memory=False)
s = s[s.data_date >= "2025-01-01"].copy()
# coal generation excluding Patuakhali (second unit commissioned during the war window)
pat = p.pgcb_name.eq("Patuakhali 1320 MW (RNPL)")
coal = p[p.fuel_group.eq("coal")]
s = s.merge(coal[~coal.pgcb_name.eq("Patuakhali 1320 MW (RNPL)")].groupby("data_date").energy_kwh.sum().div(1e6)
            .rename("coal_gen_ex_patuakhali").reset_index(), on="data_date", how="left")
s["temp"] = s.bpdb_max_temp_c.fillna(s.max_temp_c)
s["unserved_pct"] = 100 * s.unserved_share
s["gas_gen"] = s.pgcb_gen_gas_mkwh; s["coal_gen"] = s.pgcb_gen_coal_mkwh
s["import_gen"] = s.pgcb_gen_import_mkwh
s = s[s.doy <= s[s.year == 2026].doy.max()]
win = lambda d, a, b: ((d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b))).astype(int)
s["ramadan"] = win(s.data_date, *RAMADAN[2025]) | win(s.data_date, *RAMADAN[2026])
s["eid"] = 0
for e in EIDS: s["eid"] |= win(s.data_date, pd.Timestamp(e) - pd.Timedelta(days=3), pd.Timestamp(e) + pd.Timedelta(days=5))
s["y26"] = (s.year == 2026).astype(int)
s["war"] = s.y26 * ((s.data_date >= WAR) & (s.data_date < FSRU)).astype(int)
s["fsru"] = s.y26 * ((s.data_date >= FSRU) & (s.data_date < LATE)).astype(int)
s["late"] = s.y26 * (s.data_date >= LATE).astype(int)
s["t26"] = s.y26 * (s.doy / 30.4)          # 2026-specific linear trend, per month

OUTC = {"unserved_pct": "Energy not served (pp)", "gas_supplied_mmcfd": "Gas to power (MMCFD)",
        "gas_gen": "Gas generation (MkWh/day)", "oil_shortage_plants": "Plants citing oil shortage",
        "coal_gen": "Coal generation (MkWh/day)", "coal_gen_ex_patuakhali": "Coal generation excl. Patuakhali (MkWh/day)",
        "import_gen": "Imported electricity (MkWh/day) [placebo outcome]"}
BASE = "C(doy) + y26 + war + fsru + late + temp + I(temp**2) + ramadan + eid"

def run(d, y, rhs=BASE, lags=14, terms=("war", "fsru")):
    d = d.dropna(subset=[y, "temp"])
    m = smf.ols(f"{y} ~ {rhs}", d).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {t: (m.params.get(t, np.nan), m.bse.get(t, np.nan), m.pvalues.get(t, np.nan)) for t in terms} | {"n": int(m.nobs)}

specs = {
 "1. Baseline (Table 1)": lambda y: run(s, y),
 "2. Pre-war reference = February only (drop January)": lambda y: run(s[s.data_date.dt.month != 1], y),
 "3. 2026-specific linear trend": lambda y: run(s, y, BASE + " + t26"),
 "4. Drop Ramadan and Eid windows": lambda y: run(s[(s.ramadan == 0) & (s.eid == 0)], y, BASE.replace(" + ramadan + eid", "")),
 "5. No temperature control": lambda y: run(s, y, BASE.replace(" + temp + I(temp**2)", "")),
 "6. HAC 7 lags": lambda y: run(s, y, lags=7),
 "7. HAC 28 lags": lambda y: run(s, y, lags=28),
 "8. Weekly averages (removes day-level noise)": None,
}
# weekly aggregation spec
wk = s.assign(week=((s.doy - 1) // 7)).groupby(["year", "week"]).mean(numeric_only=True).reset_index()
wk["y26"] = (wk.year == 2026).astype(int)
for c in ["war", "fsru", "late"]: wk[c] = (wk[c] > 0.5).astype(int)
specs["8. Weekly averages (removes day-level noise)"] = lambda y: run(wk, y, BASE.replace("C(doy)", "C(week)"), lags=3)

rows = []
for name, f in specs.items():
    for y, lab in OUTC.items():
        r = f(y)
        for t in ("war", "fsru"):
            c, se, pv = r[t]; rows.append(dict(spec=name, outcome=lab, term=t, coef=c, se=se, p=pv, n=r["n"]))
rob = pd.DataFrame(rows); rob.to_csv(os.path.join(OUT, "table3_robustness.csv"), index=False)

# placebo in time: pretend the war started on 1 Feb 2026, using only Jan-Feb data
pre = s[s.data_date.dt.month <= 2].copy()
pre["fake"] = pre.y26 * (pre.data_date.dt.month == 2).astype(int)
plac = []
for y, lab in OUTC.items():
    d = pre.dropna(subset=[y, "temp"])
    m = smf.ols(f"{y} ~ C(doy) + y26 + fake + temp + I(temp**2) + ramadan + eid", d).fit(cov_type="HAC", cov_kwds={"maxlags": 14})
    plac.append(dict(outcome=lab, fake_onset_1Feb=m.params["fake"], se=m.bse["fake"], p=m.pvalues["fake"], n=int(m.nobs)))
plac = pd.DataFrame(plac); plac.to_csv(os.path.join(OUT, "table4_placebo.csv"), index=False)

# placebo distribution: fake onsets on every day 15 Jan-15 Feb (pre-war only) for the key outcomes
dist = []
for y in ["unserved_pct", "gas_supplied_mmcfd", "oil_shortage_plants"]:
    for fd in pd.date_range("2026-01-15", "2026-02-15"):
        d = pre.dropna(subset=[y, "temp"]).copy()
        d["fake"] = d.y26 * (d.doy >= fd.dayofyear).astype(int)
        m = smf.ols(f"{y} ~ C(doy) + y26 + fake + temp + I(temp**2) + ramadan + eid", d).fit()
        dist.append(dict(outcome=y, fake_date=fd.date(), coef=m.params["fake"]))
dist = pd.DataFrame(dist); dist.to_csv(os.path.join(OUT, "placebo_distribution.csv"), index=False)

# plant level excluding Patuakhali
q = p[(p.data_date >= "2025-01-01") & p.fuel_group.isin(["gas", "oil_hfo", "coal"]) & ~pat].copy()
q.loc[q.cf > 1.2, "cf"] = np.nan; q["cf"] = 100 * q.cf
q["period"] = np.select([q.data_date < WAR, q.data_date < FSRU, q.data_date < LATE], ["pre", "war", "fsru"], "late")
q["fuel_moy"] = q.fuel_group + "_" + q.data_date.dt.month.astype(str)
terms = []
for per in ("war", "fsru", "late"):
    for g in ("gas", "oil_hfo"):
        c = f"{per}_x_{g}"; q[c] = ((q.period == per) & (q.fuel_group == g)).astype(int); terms.append(c)
fit = pf.feols(f"cf ~ {' + '.join(terms)} | plant_id + data_date + fuel_moy", data=q.dropna(subset=["cf"]), vcov={"CRV1": "plant_id"})
pt = fit.tidy().reset_index(); pt.to_csv(os.path.join(OUT, "table5_plant_ex_patuakhali.csv"), index=False)

pd.set_option("display.width", 250)
rob["cell"] = rob.apply(lambda r: f"{r.coef:.2f} ({r.se:.2f}){'***' if r.p < .01 else '**' if r.p < .05 else '*' if r.p < .1 else ''}", axis=1)
print(rob[rob.term == "war"].pivot(index="spec", columns="outcome", values="cell").to_string())
print(rob[rob.term == "fsru"].pivot(index="spec", columns="outcome", values="cell").to_string())
print(plac.round(3).to_string())
print(dist.groupby("outcome").coef.describe().round(2))
print(pt.round(3).to_string())

# ---- pre-period-only trend: estimate the 2026-vs-2025 slope on Jan-Feb only, extrapolate, then re-estimate
tr = []
for y, lab in OUTC.items():
    d = s.dropna(subset=[y, "temp"]).copy()
    pre_d = d[d.data_date.dt.month <= 2]
    b = smf.ols(f"{y} ~ C(doy) + y26 + t26 + temp + I(temp**2) + ramadan + eid", pre_d).fit().params["t26"]
    d[y + "_dt"] = d[y] - b * d.t26
    r = run(d, y + "_dt")
    for t in ("war", "fsru"):
        c, se, pv = r[t]; tr.append(dict(outcome=lab, pre_slope_per_month=b, term=t, coef=c, se=se, p=pv))
tr = pd.DataFrame(tr); tr.to_csv(os.path.join(OUT, "table3b_preperiod_trend.csv"), index=False)
print(tr.round(3).to_string())
