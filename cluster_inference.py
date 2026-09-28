"""
cluster_inference.py - inference checks for the plant-level regressions (Eq. 2).
Treatment varies at the fuel-group level, so clustering by plant may understate uncertainty.
Two checks:
  (a) wild cluster bootstrap by plant, Rademacher weights, null imposed (9,999 replications);
  (b) randomisation inference: fuel labels permuted across plants within capacity strata (5,000 draws),
      which respects the fact that only three fuel groups generate the treatment.
"""
import sys, gc, numpy as np, pandas as pd, pyfixest as pf
IN, OUT = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(42)
p = pd.read_csv(f"{IN}/plant_day_panel.csv", parse_dates=["data_date"], low_memory=False)
p = p[(p.data_date >= "2025-01-01") & p.fuel_group.isin(["gas", "oil_hfo", "coal"])].copy()
p.loc[p.cf > 1.2, "cf"] = np.nan
p["cf"] = 100 * p.cf
p["period"] = np.select([p.data_date < "2026-03-01", p.data_date < "2026-07-21", p.data_date < "2026-09-01"],
                        ["pre", "war", "fsru"], "late")
def build(df):
    df = df.copy()
    df["fuel_moy"] = df.fuel_group + "_" + df.data_date.dt.month.astype(str)
    terms = []
    for per in ("war", "fsru", "late"):
        for g in ("gas", "oil_hfo"):
            c = f"{per}_x_{g}"; df[c] = ((df.period == per) & (df.fuel_group == g)).astype(int); terms.append(c)
    return df, terms
FML = "cf ~ {} | plant_id + data_date + fuel_moy"
d, terms = build(p.dropna(subset=["cf"]))
base = pf.feols(FML.format(" + ".join(terms)), data=d, vcov={"CRV1": "plant_id"})
tidy = base.tidy()
print(tidy.round(3).to_string())

# ---------------- (a) wild cluster bootstrap by plant, null imposed
def wild_boot(param, reps=9999):
    others = [t for t in terms if t != param]
    r = pf.feols(FML.format(" + ".join(others)), data=d, vcov={"CRV1": "plant_id"})   # restricted model
    fitted = np.asarray(r.predict(newdata=d), dtype=float)
    y = d["cf"].to_numpy(dtype=float)
    ok = ~np.isnan(fitted)
    fitted = np.where(ok, fitted, y)          # rows dropped as singletons keep their own value
    res = y - fitted
    t_obs = tidy.loc[param, "Estimate"] / tidy.loc[param, "Std. Error"]
    clusters = d.plant_id.to_numpy(); uniq = np.unique(clusters)
    idx = {c: np.where(clusters == c)[0] for c in uniq}
    t_star = np.empty(reps)
    dd = d.copy()
    for b in range(reps):
        w = rng.choice([-1.0, 1.0], size=len(uniq))
        mult = np.empty(len(clusters))
        for k, c in enumerate(uniq): mult[idx[c]] = w[k]
        dd["cf"] = fitted + res * mult
        f = pf.feols(FML.format(" + ".join(terms)), data=dd, vcov={"CRV1": "plant_id"}, lean=True, copy_data=False, store_data=False)
        t = f.tidy()
        t_star[b] = t.loc[param, "Estimate"] / t.loc[param, "Std. Error"]
        del f, t
        if b % 50 == 0: gc.collect()
    return t_obs, float(np.mean(np.abs(t_star) >= abs(t_obs)))

# ---------------- (b) randomisation inference: permute fuel labels across plants
plants = p.groupby("plant_id").agg(fuel_group=("fuel_group", "first"), cap=("capacity_ref_mw", "median")).reset_index()
plants["stratum"] = pd.qcut(plants.cap.rank(method="first"), 4, labels=False)
def ri(params, draws=5000):
    obs = {k: tidy.loc[k, "Estimate"] for k in params}
    count = {k: 0 for k in params}
    for b in range(draws):
        perm = plants.copy()
        perm["fuel_group"] = perm.groupby("stratum").fuel_group.transform(lambda s: rng.permutation(s.values))
        q = p.drop(columns="fuel_group").merge(perm[["plant_id", "fuel_group"]], on="plant_id")
        q, _ = build(q.dropna(subset=["cf"]))
        f = pf.feols(FML.format(" + ".join(terms)), data=q, vcov="iid", lean=True, copy_data=False, store_data=False)
        t = f.tidy()
        for k in params:
            if abs(t.loc[k, "Estimate"]) >= abs(obs[k]): count[k] += 1
        del f, t, q
        if b % 25 == 0: gc.collect()
    return {k: (obs[k], count[k] / draws) for k in params}

if __name__ == "__main__":
    rows = []
    for param in ["war_x_gas", "fsru_x_gas", "war_x_oil_hfo", "fsru_x_oil_hfo"]:
        t_obs, p_boot = wild_boot(param, reps=int(sys.argv[3]) if len(sys.argv) > 3 else 999)
        rows.append(dict(term=param, estimate=tidy.loc[param, "Estimate"], se_cluster=tidy.loc[param, "Std. Error"],
                         p_cluster=tidy.loc[param, "Pr(>|t|)"], t_stat=t_obs, p_wild_bootstrap=p_boot))
        print(rows[-1])
    r = ri(["war_x_gas", "fsru_x_gas"], draws=int(sys.argv[4]) if len(sys.argv) > 4 else 500)
    for k, (est, pv) in r.items():
        for row in rows:
            if row["term"] == k: row["p_randomisation"] = pv
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/table9_cluster_inference.csv", index=False)
    print(out.round(4).to_string())
