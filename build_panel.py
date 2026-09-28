"""
build_panel.py - clean the parsed BPDB/PGCB tables and build the analysis panel.
Inputs : parsed/*.csv (from bpdb_parse.py), plant_master_v3.csv, plant_aliases.csv
Outputs: plant_day_panel.csv, system_day_panel.csv, zone_day_panel.csv, data_quality.txt
"""
import re, sys, difflib, os
import numpy as np, pandas as pd

IN, OUT, MASTER = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
key = lambda n: re.sub(r"[^a-z0-9]", "", str(n).lower())
log = []
def note(s): print(s); log.append(s)

pm = pd.read_csv(os.path.join(MASTER, "plant_master_v3.csv"))
al = pd.read_csv(os.path.join(MASTER, "plant_aliases.csv"))
nid = lambda n: pm.loc[pm.pgcb_name == n, "plant_id"].iloc[0]

# ---------------------------------------------------------------- extend master with plants seen in the full run
extra_plants = [  # name, area, zone, fuel, fuel_group, installed, note
 ("Feni 11 MW PP (Doreen)", "Cumilla Area", "Cumilla", "Gas", "gas", 11, "full run: data Jan-Jun 2025"),
 ("Narshingdi 22 MW PP (Doreen)", "Dhaka Area", "Dhaka", "Gas", "gas", 22, "full run: data Jan-Jun 2025"),
 ("Katpotti 52 MW PP (Sinha)", "Dhaka Area", "Dhaka", "HFO", "oil_hfo", 52, "long-term out/contract expired"),
 ("Jamalpur 95 MW PP(Powerpac)", "Mymensingh Area", "Mymensing", "HFO", "oil_hfo", 95, "long-term out/contract expired"),
 ("Bosila 108MW PP(CLC)", "Dhaka Area", "Dhaka", "HFO", "oil_hfo", 108, "long-term out/contract expired"),
 ("Rupsha 800 MW CCPP", "Khulna Area", "Khulna", "Gas/HSD", "gas", 800, "commissioning from Dec 2025"),
 ("Ashuganj TPP Unit- 3, 4", "Cumilla Area", "Cumilla", "Gas", "gas", None, "one day only (Oct 2025)"),
]
z2d = {'Dhaka':'Titas','Mymensing':'Titas','Cumilla':'Bakhrabad (BGDCL)','Chattogram':'Karnaphuli (KGDCL)','Sylhet':'Jalalabad (JGTDSL)',
       'Rajshahi':'Pashchimanchal (PGCL)','Rangpur':'Pashchimanchal (PGCL)','Khulna':'Sundarban (SGCL)','Barishal':'Sundarban (SGCL)'}
egm = {'Karnaphuli (KGDCL)':'rlng_landing','Jalalabad (JGTDSL)':'field_adjacent'}
rows = []
for n, a, z, f, fg, cap, nt in extra_plants:
    if n in set(pm.pgcb_name): continue
    g = fg == "gas"; d = z2d[z] if g else "n/a"
    rows.append(dict(plant_id="P%03d" % (len(pm) + len(rows) + 1), plant_key=key(n), pgcb_name=n, bpdb_name=n,
        match_status="manual_new_plant", area=a, zone=z, fuel=f, fuel_group=fg,
        fuel_capability="dual (gas + HSD)" if "HSD" in f else "single", primary_fuel="natural gas" if g else "HFO",
        backup_fuel="HSD" if "HSD" in f else "none", power_grid="grid-connected",
        gas_network="national transmission grid" if g else "n/a", installed_mw=cap, gas_distributor=d,
        exposure_group=egm.get(d, "grid_mixed") if g else "n/a", verify="VERIFY" if g else "", notes=nt, source_citation="", days=0))
pm = pd.concat([pm, pd.DataFrame(rows)], ignore_index=True)
# capacity corrections: plants whose second unit came online during the period
pm.loc[pm.pgcb_name == "Patuakhali 1320 MW (RNPL)", "installed_mw"] = 1244
alias_extra = [  # spelling seen -> master name
 ("Rangpur 113 MW PP (Confidence) IPP", "Rangpur 113 MW PP (Confidence)"),
 ("Kushiara163MWCPC(KPP)", "Kushiara 163 MW CCPP (KP)"),
 ("Energypac Power Venture Thakurgaon", "Energypac Power Venture Thakurgaon Ltd."),
 ("Moulvibazar 10 MW Solar Power Planat", "Moulvibazar 10 MW Solar Power Plant"),
 ("Kanchan Purbachal Power Generation", "Kanchan Purbachal Power Generation Ltd."),
 ("Unique Meghnaghat Power Limited (U", "Unique Meghnaghat Power Limited (UMPL)"),
 ("Unique Meghnaghat Power Limited (", "Unique Meghnaghat Power Limited (UMPL)"),
 ("Unique Meghnaghat Power Limited (UM", "Unique Meghnaghat Power Limited (UMPL)"),
 ("Sonagazi 75 MW (AC) Solar Power P", "Sonagazi 75 MW (AC) Solar Power Plant"),
 ("Sikalbaha 105 MW PP (Baraka Sikalb", "Sikalbaha 105 MW PP (Baraka Sikalbaha)"),
 ("Saidpur 150 MW Simple Cycle Power P", "Saidpur 150 MW Simple Cycle Power Plant"),
 ("Saidpur 150 MW Simple Cycle Power", "Saidpur 150 MW Simple Cycle Power Plant"),
 ("GFehnocrahsuaglo Rnje CpoCwPePr ePdh CasCeP-1P Unit-3", "Fenchugonj CCPP Phase-1"),
 ("GFehnocrahsuaglo Rnje CpoCwPePr ePdh CasCeP-2P Unit-3", "Fenchugonj CCPP Phase-2"),
 ("GBhhaoirraosba 5l 4R.e5p MowWered CCPP Unit-3", "Bhairob 54.5 MW"),
 ("Confidence Power Bogura Unit-1 113", "Bagura 113 MW PP (Confidence)-1"),
]
amap = dict(zip(al.plant_key, al.plant_id))
for r in rows: amap[r["plant_key"]] = r["plant_id"]
for sp, target in alias_extra: amap[key(sp)] = nid(target)

# ---------------------------------------------------------------- PGCB plant-day
p = pd.read_csv(os.path.join(IN, "plant_day_pgcb.csv"), low_memory=False)
p["area"] = p.area.astype(str).str.extract(r"(\w+ (?:Area|Grid))$")[0]
p = p[~p.is_total & ~p.plant_raw.astype(str).str.startswith("Unit No")].copy()
p["plant_id"] = p.plant_key.map(amap)
# garbled overprinted names not in the list: match by the non-ghost part if possible
unm = p.plant_id.isna()
for k in p.loc[unm, "plant_key"].unique():
    m = difflib.get_close_matches(k, list(amap), n=1, cutoff=0.85)
    if m: amap[k] = amap[m[0]]
p["plant_id"] = p.plant_key.map(amap)
note(f"PGCB rows: {len(p)}; unmatched to a plant_id: {p.plant_id.isna().sum()} (dropped)")
p = p[p.plant_id.notna()]
dup = p.duplicated(["data_date", "plant_id"], keep=False)
note(f"PGCB duplicate plant-days (two spellings same day): {dup.sum()} rows; keeping the row with more energy")
p = p.sort_values("energy_kwh", na_position="first").drop_duplicates(["data_date", "plant_id"], keep="last")

# ---------------------------------------------------------------- BPDB plant-day
b = pd.read_csv(os.path.join(IN, "plant_day_bpdb.csv"), low_memory=False)
FUELTOK = r"(gas/?hsd|hsd/?gas|gas|hfo|coal|solar|hydro|hsd|wind|india|nepal|import)?(pdb|ipp|apscl|egcb|nwpgcl|bcpcl|cpgcl|bifpcl|rpcl|nenp|sippreb|cippreb|brpgen|brpowergen)?\d*$"
bmap = {key(n): pid for n, pid in zip(pm.bpdb_name, pm.plant_id) if isinstance(n, str)}
bmap.update({key("Meghnaghat 589 MW CCPP(Summit)"): nid("Meghnaghat CCPP(Summit)-2"),
             key("Sreepur 150 MW"): nid("Sreepur 150 MW PP BRPL"),
             key("Bheramara GTPP Unit-"): nid("Bheramara GTPP Unit-3"),
             key("Malancha, Ctg.EPZ (United)"): nid("Malancha, Ctg. EPZ (United)")})
def bmatch(k):
    if k in bmap: return bmap[k]
    k2 = re.sub(FUELTOK, "", k)
    if k2 in bmap: return bmap[k2]
    m = difflib.get_close_matches(k2, list(bmap), n=1, cutoff=0.8)
    return bmap[m[0]] if m else None
bk = {k: bmatch(k) for k in b.plant_key.unique()}
b["plant_id"] = b.plant_key.map(bk)
note(f"BPDB rows: {len(b)}; unmatched: {b.plant_id.isna().sum()} (dropped)")
b = b[b.plant_id.notna() & b.data_date.notna()]
b = b.sort_values("actual_eve_mw", na_position="first").drop_duplicates(["data_date", "plant_id"], keep="last")
b = b[["data_date", "plant_id", "installed_mw", "derated_mw", "actual_day_mw", "actual_eve_mw", "fuel_limitation_mw",
       "shutdown_mw", "remarks", "remark_cat"]].rename(columns={"remarks": "bpdb_remarks", "remark_cat": "bpdb_remark_cat",
       "installed_mw": "bpdb_installed_mw"})

# ---------------------------------------------------------------- merge + master attributes
panel = p.merge(b, on=["data_date", "plant_id"], how="outer")
attrs = pm.set_index("plant_id")[["pgcb_name", "fuel_group", "zone", "gas_distributor", "exposure_group", "fuel_capability", "installed_mw"]]
panel = panel.join(attrs, on="plant_id")
# area: use the plant's usual area (a missing area-total row can push a block of plants into the next area)
modal_area = p.groupby("plant_id").area.agg(lambda s: s.mode().iat[0] if s.notna().any() else None)
panel["area_reported"] = panel.area
panel["area"] = panel.plant_id.map(modal_area).fillna(panel.plant_id.map(pm.set_index("plant_id").area))
panel["data_date"] = pd.to_datetime(panel.data_date)
# physical plausibility: energy <= capacity x 24h (+10%), peak <= capacity (+30%).
# capacity = same-day BPDB installed MW (tracks units added, e.g. Patuakhali unit 2), else master installed MW,
# else the plant's median present capacity among sane values
cap_m = panel.plant_id.map(pm.set_index("plant_id").installed_mw)
med = panel[panel.present_capacity_mw < 3000].groupby("plant_id").present_capacity_mw.median()
base = cap_m.where(cap_m > 0).fillna(panel.plant_id.map(med))
# capacity reference = the largest of: master installed MW, same-day BPDB installed MW, same-day PGCB present capacity;
# the daily values only count if within 2x the master figure (guards against mis-parsed digits, while
# still tracking capacity added during the period, e.g. Patuakhali's second 660 MW unit)
cands = pd.concat([base,
                   panel.bpdb_installed_mw.where(panel.bpdb_installed_mw <= 2.5 * base),
                   panel.present_capacity_mw.where(panel.present_capacity_mw <= 2.5 * base)], axis=1)
panel["capacity_ref_mw"] = cands.max(axis=1)
imports = panel.fuel_group.eq("import")
bad_e = (panel.energy_kwh > panel.capacity_ref_mw.clip(lower=1) * 24_000 * 1.10) & ~imports
bad_p = (panel.peak_gen_mw > panel.capacity_ref_mw.clip(lower=1) * 1.30) & ~imports
bad_c = panel.present_capacity_mw > panel.capacity_ref_mw.clip(lower=1) * 1.5
note(f"Values above physical maximum set to missing: energy {bad_e.sum()}, peak {bad_p.sum()}, present capacity {bad_c.sum()}")
panel["energy_flag"] = np.where(bad_e, "implausible_set_missing", "")
panel.loc[bad_e, "energy_kwh"] = np.nan
panel.loc[bad_p, "peak_gen_mw"] = np.nan
panel.loc[bad_c, "present_capacity_mw"] = np.nan
gas = panel.fuel_group.eq("gas")
# generic 'fuel shortage' at a gas plant is a gas shortage
generic = panel.remarks.astype(str).str.contains(r"^\s*fuel short", case=False, regex=True)
panel.loc[gas & generic & panel.remark_cat.eq("oil_shortage"), "remark_cat"] = "gas_shortage"
# combined shortage reason: PGCB remark, else BPDB remark
def cause(r):
    for c in (r.remark_cat, r.bpdb_remark_cat):
        if c in ("gas_shortage", "oil_shortage", "coal_shortage"): return c
    return None
panel["shortage_cause"] = [cause(r) for r in panel[["remark_cat", "bpdb_remark_cat"]].itertuples()]
panel["idle_mw"] = (panel.present_capacity_mw - panel.peak_gen_mw).clip(lower=0)
panel["cf"] = panel.energy_kwh / (panel.capacity_ref_mw.clip(lower=1) * 24_000)
d = panel.data_date
panel["period"] = np.select([d < "2026-03-01", d < "2026-07-01", d < "2026-09-01"],
                            ["pre_war", "war_clean", "war_fsru_confounded"], "war_late")
panel["year"] = d.dt.year; panel["month"] = d.dt.month; panel["doy"] = d.dt.dayofyear
panel = panel.sort_values(["plant_id", "data_date"]).reset_index(drop=True)
gas = panel.fuel_group.eq("gas")
panel.to_csv(os.path.join(OUT, "plant_day_panel.csv"), index=False)
note(f"plant_day_panel: {len(panel)} plant-days, {panel.plant_id.nunique()} plants, {panel.data_date.nunique()} days")

# ---------------------------------------------------------------- QA after cleaning
tot = pd.read_csv(os.path.join(IN, "plant_day_pgcb.csv"), low_memory=False)
tot = tot[tot.is_total & tot.plant_raw.astype(str).str.contains("Area")].copy()
tot["area"] = tot.area.astype(str).str.extract(r"(\w+ Area)$")[0]
tot = tot.groupby(["data_date", "area"]).energy_kwh.last()
sm = panel.assign(data_date=panel.data_date.dt.strftime("%Y-%m-%d")).groupby(["data_date", "area"]).energy_kwh.sum()
qa = pd.concat([tot, sm], axis=1, keys=["report", "plants"]).dropna(subset=["report"])
qa["gap_pct"] = 100 * (qa.report - qa.plants) / qa.report
bad = qa[qa.gap_pct.abs() > 1]
note(f"QA after cleaning: {len(bad)} of {len(qa)} day-areas differ from the report total by >1% "
     f"(median gap among them {bad.gap_pct.median():.1f}%); positive gap = rows missing, energy is left missing not zero")
bad.reset_index().to_csv(os.path.join(OUT, "qa_after_cleaning.csv"), index=False)

# ---------------------------------------------------------------- system-day
s = pd.read_csv(os.path.join(IN, "system_day.csv"))
sb = pd.read_csv(os.path.join(IN, "system_day_bpdb.csv"))
miss = sb.data_date.isna()
if miss.any():   # report date unread: take it from the archive file name (archive date = data date + 1)
    fd = pd.to_datetime(sb.loc[miss, "file"].str.extract(r"_(\d{8})_")[0], format="%Y%m%d")
    sb.loc[miss, "data_date"] = (fd - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    note(f"BPDB reports with unread date recovered from file name: {miss.sum()}")
sb = sb.drop_duplicates("data_date", keep="last")
# some early-2025 BPDB reports print energy in a unit 1000x larger (e.g. 0.22 instead of 220 MkWh): rescale
ecols = [c for c in sb.columns if c.startswith("energy_") and c.endswith("_mkwh")]
small = sb.energy_total_mkwh < 5
sb.loc[small, ecols] = sb.loc[small, ecols] * 1000
note(f"BPDB reports with energy printed 1000x smaller, rescaled: {small.sum()}")
sysd = s.drop(columns=["file"]).merge(sb.drop(columns=["file"]).add_prefix("bpdb_").rename(columns={"bpdb_data_date": "data_date"}),
                                     on="data_date", how="outer")
zg = pd.read_csv(os.path.join(IN, "zone_day.csv"))
zg = zg[zg.zone == "Total"].drop_duplicates("data_date", keep="last")
zg = zg[["data_date"] + [c for c in zg.columns if c.startswith("gen_")]].rename(columns=lambda c: c if c == "data_date" else "pgcb_" + c)
sysd = sysd.merge(zg, on="data_date", how="left")
sysd["data_date"] = pd.to_datetime(sysd.data_date)
sysd["unserved_share"] = sysd.energy_unserved_mkwh / sysd.energy_demand_mkwh
gs = panel[gas & panel.shortage_cause.eq("gas_shortage")].groupby("data_date").idle_mw.sum()
os_ = panel[panel.shortage_cause.eq("oil_shortage")].groupby("data_date").plant_id.nunique()
cs = panel[panel.shortage_cause.eq("coal_shortage")].groupby("data_date").plant_id.nunique()
gl = panel[gas].groupby("data_date").fuel_limitation_mw.sum(min_count=1)
sysd = sysd.set_index("data_date").join(gs.rename("gas_shortage_idle_mw")).join(os_.rename("oil_shortage_plants")) \
           .join(cs.rename("coal_shortage_plants")).join(gl.rename("gas_plant_fuel_limitation_mw")).reset_index()
for c in ["gas_shortage_idle_mw", "oil_shortage_plants", "coal_shortage_plants"]:
    sysd[c] = sysd[c].fillna(0)
d = sysd.data_date
sysd["period"] = np.select([d < "2026-03-01", d < "2026-07-01", d < "2026-09-01"], ["pre_war", "war_clean", "war_fsru_confounded"], "war_late")
sysd["year"] = d.dt.year; sysd["month"] = d.dt.month; sysd["doy"] = d.dt.dayofyear
sysd = sysd.sort_values("data_date")
sysd.to_csv(os.path.join(OUT, "system_day_panel.csv"), index=False)
note(f"system_day_panel: {len(sysd)} days, {sysd.data_date.min().date()} to {sysd.data_date.max().date()}")

z = pd.read_csv(os.path.join(IN, "zone_day.csv"))
z = z[z.zone != "Total"].drop(columns=["file"])
z.to_csv(os.path.join(OUT, "zone_day_panel.csv"), index=False)
pm.to_csv(os.path.join(OUT, "plant_master_v4.csv"), index=False)
pd.DataFrame(sorted(amap.items()), columns=["plant_key", "plant_id"]).to_csv(os.path.join(OUT, "plant_aliases_v4.csv"), index=False)
open(os.path.join(OUT, "data_quality.txt"), "w").write("\n".join(log))
