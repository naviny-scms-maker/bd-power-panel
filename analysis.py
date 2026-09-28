"""
analysis.py - first empirical results: system-level year-over-year difference-in-differences,
weekly event study, and plant-level fuel-channel regressions.
Input: system_day_panel.csv, plant_day_panel.csv (from build_panel.py)
"""
import sys, os, numpy as np, pandas as pd, statsmodels.formula.api as smf, pyfixest as pf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
IN, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
WAR, FSRU, LATE = pd.Timestamp("2026-03-01"), pd.Timestamp("2026-07-21"), pd.Timestamp("2026-09-01")
# religious calendar: Bangladesh National Moon Sighting Committee decisions (BSS)
RAMADAN = {2025: ("2025-03-02", "2025-03-30"), 2026: ("2026-02-19", "2026-03-20")}
EID_FITR = {2025: "2025-03-31", 2026: "2026-03-21"}
EID_ADHA = {2025: "2025-06-07", 2026: "2026-05-28"}
# policy phases: 5 Mar 2026 fertilizer plants shut and power gas cut 50 mmcfd (TBS);
# 1 May 2026 gas restored to Ashuganj fertilizer "although the decision will hamper power generation" (BSS, 23 Apr)
FERT_RESTORED = pd.Timestamp("2026-05-01")

# ---------------------------------------------------------------- system level: 2026 vs 2025, same calendar day
s = pd.read_csv(os.path.join(IN, "system_day_panel.csv"), parse_dates=["data_date"])
s = s[(s.data_date >= "2025-01-01")].copy()
s["temp"] = s.bpdb_max_temp_c.fillna(s.max_temp_c)
s["unserved_pct"] = 100 * s.unserved_share
s["gas_gen"] = s.pgcb_gen_gas_mkwh; s["coal_gen"] = s.pgcb_gen_coal_mkwh
s["oil_gen"] = s.pgcb_gen_hfo_mkwh.fillna(0) + s.pgcb_gen_hsd_mkwh.fillna(0)
s["import_gen"] = s.pgcb_gen_import_mkwh
s["gas_idle_gw"] = s.gas_shortage_idle_mw / 1000
last_doy = s[s.year == 2026].doy.max()
s = s[s.doy <= last_doy]
def window(d, a, b): return ((d >= pd.Timestamp(a)) & (d <= pd.Timestamp(b))).astype(int)
s["ramadan"] = 0; s["eid"] = 0
for y in (2025, 2026):
    s["ramadan"] |= window(s.data_date, *RAMADAN[y])
    for e in (EID_FITR[y], EID_ADHA[y]):
        s["eid"] |= window(s.data_date, pd.Timestamp(e) - pd.Timedelta(days=3), pd.Timestamp(e) + pd.Timedelta(days=5))
y26 = (s.year == 2026).astype(int)
s["y26"] = y26
s["war"] = y26 * ((s.data_date >= WAR) & (s.data_date < FSRU)).astype(int)
s["war1"] = y26 * ((s.data_date >= WAR) & (s.data_date < FERT_RESTORED)).astype(int)
s["war2"] = y26 * ((s.data_date >= FERT_RESTORED) & (s.data_date < FSRU)).astype(int)
s["fsru"] = y26 * ((s.data_date >= FSRU) & (s.data_date < LATE)).astype(int)
s["late"] = y26 * (s.data_date >= LATE).astype(int)

outcomes = {"unserved_pct": "Energy not served (% of demand)", "gas_supplied_mmcfd": "Gas supplied to power (MMCFD)",
            "gas_gen": "Gas-fired generation (MkWh/day)", "gas_idle_gw": "Gas plants idle citing gas shortage (GW)",
            "oil_shortage_plants": "Plants citing liquid-fuel shortage (count)", "oil_gen": "Oil-fired generation (MkWh/day)",
            "coal_gen": "Coal-fired generation (MkWh/day)", "import_gen": "Imported electricity (MkWh/day)",
            "cost_per_kwh_tk": "Generation cost (Tk/kWh)", "energy_demand_mkwh": "Energy demand (MkWh/day)"}
rows = []
for y, lab in outcomes.items():
    d = s.dropna(subset=[y, "temp"])
    m = smf.ols(f"{y} ~ C(doy) + y26 + war + fsru + late + temp + I(temp**2) + ramadan + eid", d).fit(
        cov_type="HAC", cov_kwds={"maxlags": 14})
    base = d[(d.year == 2025) & (d.data_date >= "2025-03-01") & (d.data_date < "2025-07-21")][y].mean()
    for t in ("y26", "war", "fsru", "late"):
        rows.append(dict(outcome=lab, term=t, coef=m.params[t], se=m.bse[t], p=m.pvalues[t], base_2025_mar_jul=base, n=int(m.nobs)))
    m2 = smf.ols(f"{y} ~ C(doy) + y26 + war1 + war2 + fsru + late + temp + I(temp**2) + ramadan + eid", d).fit(
        cov_type="HAC", cov_kwds={"maxlags": 14})
    for t in ("war1", "war2"):
        rows.append(dict(outcome=lab, term=t, coef=m2.params[t], se=m2.bse[t], p=m2.pvalues[t], base_2025_mar_jul=base, n=int(m2.nobs)))
    diff = m2.t_test("war2 - war1 = 0")
    rows.append(dict(outcome=lab, term="war2_minus_war1", coef=float(diff.effect.ravel()[0]), se=float(diff.sd.ravel()[0]), p=float(diff.pvalue), base_2025_mar_jul=base, n=int(m2.nobs)))
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(OUT, "table1_system_did.csv"), index=False)

# ---------------------------------------------------------------- weekly event study (2026 x week relative to 1 March)
s["relweek"] = np.floor((s.data_date - s.data_date.dt.year.map(lambda y: pd.Timestamp(f"{y}-03-01"))).dt.days / 7).astype(int)
wk = sorted(s.relweek.unique()); ref = -1
for w in wk:
    if w != ref: s[f"e{w+100}"] = y26 * (s.relweek == w).astype(int)
evterms = [f"e{w+100}" for w in wk if w != ref]
es = {}
for y in ["unserved_pct", "gas_supplied_mmcfd", "gas_idle_gw", "oil_shortage_plants", "coal_gen", "gas_gen"]:
    d = s.dropna(subset=[y, "temp"])
    m = smf.ols(f"{y} ~ C(doy) + y26 + " + " + ".join(evterms) + " + temp + I(temp**2) + ramadan + eid", d).fit(
        cov_type="HAC", cov_kwds={"maxlags": 14})
    es[y] = pd.DataFrame({"week": [w for w in wk if w != ref],
                          "coef": [m.params.get(t, np.nan) for t in evterms],
                          "se": [m.bse.get(t, np.nan) for t in evterms]})
    es[y].to_csv(os.path.join(OUT, f"eventstudy_{y}.csv"), index=False)
fig, axs = plt.subplots(3, 2, figsize=(10, 9)); axs = axs.ravel()
for a, y in zip(axs, es):
    e = es[y]
    a.axhline(0, color="k", lw=.7); a.axvline(-0.5, color="C3", ls=":")
    a.axvline((FSRU - WAR).days / 7, color="C3", ls="--")
    a.fill_between(e.week, e.coef - 1.96 * e.se, e.coef + 1.96 * e.se, color="C0", alpha=.2)
    a.plot(e.week, e.coef, "o-", ms=2.5, color="C0"); a.set_title(outcomes[y], loc="left", fontsize=9)
    a.set_xlabel("Weeks relative to 1 March (2026 vs 2025)", fontsize=8); a.grid(alpha=.3)
fig.suptitle("Event study: 2026 minus 2025, same calendar week (ref. = last week of Feb); 95% CI, HAC\n"
             "Dotted = war onset; dashed = FSRU fault (21 Jul). Controls: day-of-year FE, temperature, Ramadan/Eid",
             x=0.02, ha="left", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(os.path.join(OUT, "fig3_event_study.png"), dpi=200)

# ---------------------------------------------------------------- plant level: fuel channels
p = pd.read_csv(os.path.join(IN, "plant_day_panel.csv"), parse_dates=["data_date"], low_memory=False)
p = p[(p.data_date >= "2025-01-01") & p.fuel_group.isin(["gas", "oil_hfo", "coal"])].copy()
p.loc[p.cf > 1.2, "cf"] = np.nan
p["cf"] = 100 * p.cf
own = {"gas": "gas_shortage", "oil_hfo": "oil_shortage", "coal": "coal_shortage"}
p["own_fuel_short"] = 100 * (p.shortage_cause == p.fuel_group.map(own)).astype(int)
p["period"] = np.select([p.data_date < WAR, p.data_date < FSRU, p.data_date < LATE], ["pre", "war", "fsru"], "late")
p["moy"] = p.data_date.dt.month
p["fuel_moy"] = p.fuel_group + "_" + p.moy.astype(str)
p["week"] = p.data_date.dt.to_period("W").astype(str)
res = []
def interact(df, groups, ref_group, col):
    terms = []
    for per in ("war", "fsru", "late"):
        for gname in groups:
            if gname == ref_group: continue
            c = f"{per}_x_{gname}"; df[c] = ((df.period == per) & (df[col] == gname)).astype(int); terms.append(c)
    return terms
terms = interact(p, ["gas", "oil_hfo", "coal"], "coal", "fuel_group")
for y, lab in [("cf", "Capacity factor (%)"), ("own_fuel_short", "Plant reports shortage of its own fuel (% of days)")]:
    fit = pf.feols(f"{y} ~ {' + '.join(terms)} | plant_id + data_date + fuel_moy",
                   data=p.dropna(subset=[y]), vcov={"CRV1": "plant_id"})
    t = fit.tidy().reset_index(); t["outcome"] = lab; t["n"] = fit._N; res.append(t)
# within gas plants: network position (reference = field-adjacent Sylhet plants)
g = p[(p.fuel_group == "gas") & p.exposure_group.isin(["rlng_landing", "grid_mixed", "field_adjacent", "bhola_local_field"])].copy()
g["exp_moy"] = g.exposure_group + "_" + g.moy.astype(str)
gterms = interact(g, ["rlng_landing", "grid_mixed", "field_adjacent", "bhola_local_field"], "field_adjacent", "exposure_group")
for y, lab in [("cf", "Gas plants: capacity factor (%)"), ("own_fuel_short", "Gas plants: gas shortage reported (% of days)")]:
    fit = pf.feols(f"{y} ~ {' + '.join(gterms)} | plant_id + data_date + exp_moy",
                   data=g.dropna(subset=[y]), vcov={"CRV1": "plant_id"})
    t = fit.tidy().reset_index(); t["outcome"] = lab; t["n"] = fit._N; res.append(t)
plant_tab = pd.concat(res)
plant_tab.to_csv(os.path.join(OUT, "table2_plant_fuel_channels.csv"), index=False)
base = p[(p.data_date >= "2025-03-01") & (p.data_date < "2025-07-21")].groupby("fuel_group")[["cf", "own_fuel_short"]].mean()
base.to_csv(os.path.join(OUT, "table2_baseline_2025.csv"))
print(tab.round(3).to_string()); print(plant_tab.round(3).to_string()); print(base.round(1))
print("plants:", p.groupby("fuel_group").plant_id.nunique().to_dict())
