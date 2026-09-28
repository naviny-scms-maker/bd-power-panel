"""fsru_compare.py - compare two FSRU outages: Cyclone Remal (27 May 2024, peacetime) and the July 2026 fault (after the war).
Each event: mean of the 28 days after minus the 28 days before, minus the same calendar change one year earlier
(removes seasonality). Outcomes in MkWh/day. 'Pass-through' = rise in unserved energy per MkWh of gas generation lost."""
import sys, os, numpy as np, pandas as pd
HOURLY, PANEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
h = pd.read_csv(HOURLY, parse_dates=["date"])
h = pd.DataFrame({"date": h.date, "gas": h.gas_mw * .024, "coal": h.coal_mw * .024, "oil": h.oil_mw * .024,
                  "imports": h.imports_mw * .024, "unserved": h.ls_mean_mw * .024})
s = pd.read_csv(PANEL, parse_dates=["data_date"])
s = pd.DataFrame({"date": s.data_date, "gas": s.pgcb_gen_gas_mkwh, "coal": s.pgcb_gen_coal_mkwh,
                  "oil": s.pgcb_gen_hfo_mkwh.fillna(0) + s.pgcb_gen_hsd_mkwh.fillna(0), "imports": s.pgcb_gen_import_mkwh,
                  "unserved": s.energy_unserved_mkwh})
d = pd.concat([h[h.date < "2025-01-01"], s[s.date >= "2025-01-01"]]).set_index("date").sort_index()
def change(t0, days=28):
    t0 = pd.Timestamp(t0)
    after = d.loc[t0: t0 + pd.Timedelta(days=days - 1)].mean()
    before = d.loc[t0 - pd.Timedelta(days=days): t0 - pd.Timedelta(days=1)].mean()
    return after - before
rows = []
for name, t0 in [("Cyclone Remal, Summit FSRU damaged (27 May 2024)", "2024-05-28"),
                 ("FSRU technical fault (21 Jul 2026)", "2026-07-21"),
                 ("Cyclone Mocha, both FSRUs disconnected (May 2023)", "2023-05-13")]:
    ev = change(t0); prior = change(pd.Timestamp(t0) - pd.DateOffset(years=1))
    net = ev - prior
    rows.append(dict(event=name, **{f"d_{k}": round(v, 2) for k, v in net.items()},
                     passthrough=round(-net["unserved"] / net["gas"], 2) if net["gas"] < 0 else np.nan))
r = pd.DataFrame(rows); r.to_csv(os.path.join(OUT, "table8_fsru_events.csv"), index=False)
print(r.to_string())
