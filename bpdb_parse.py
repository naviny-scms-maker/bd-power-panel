"""
bpdb_parse.py
Parses the BPDB Daily Generation Archive PDFs into analysis tables.
File type is detected from the PDF CONTENT, not the filename, so it works on the
folders created by bpdb_archive_download.py (or any folder of these PDFs).

  Archive column | Document                                   | Output table(s)
  ---------------+--------------------------------------------+-----------------------------------
  Page 1         | PGCB QF-LDC-08 Sheet-1 System Summary      | system_day.csv, zone_day.csv
  Page 2         | PGCB QF-LDC-08 Sheet-2 plant energy        | plant_day_pgcb.csv   (MAIN PANEL)
  Page 3         | PGCB QF-LDC-09 sub-station max load        | substation_day.csv
  Summary        | BPDB Daily Electricity Generation Report   | plant_day_bpdb.csv, system_day_bpdb.csv

DATES: every output uses data_date = the day the numbers describe. PGCB sheets are dated
with the data day; the BPDB report is dated the day AFTER (its "Actual" columns are yesterday),
so the parser subtracts one day. (The BPDB header year misprints e.g. "14-Apr-25"; ignored.)

Install: pip install pdfplumber pandas
Run:     python bpdb_parse.py --in bpdb_pdfs --out parsed
"""
import argparse, os, re, sys, functools, difflib
from datetime import datetime, timedelta
import pdfplumber, pandas as pd

sys.setrecursionlimit(20000)

# ------------------------------------------------------------------ helpers
def undouble(x):
    """Bold cells are sometimes extracted with every glyph doubled ('66**88..77' -> '6*8.7')."""
    if x is None: return None
    s = str(x)
    t = s.replace(" ", "")
    # only when a non-digit glyph is doubled too, so genuine numbers like 554400 are left alone
    if len(t) >= 4 and len(t) % 2 == 0 and t[0::2] == t[1::2] and any(not ch.isdigit() for ch in t[0::2]):
        return t[0::2]
    return s

def _pairs(tok):
    return len(tok) >= 2 and len(tok) % 2 == 0 and tok[0::2] == tok[1::2]

def undouble_row(r):
    """If a row was printed in bold (non-digit glyphs doubled in any cell), collapse every
    cell of that row token by token, including pure numbers like '110000' -> '100'."""
    cells = ["" if x is None else str(x) for x in r]
    # judge on the plant-name cell: bold rows have (nearly) every name token doubled,
    # whereas normal names only occasionally contain a pair such as 'PP' or 'CCPP'
    toks = [t for t in (cells[1] if len(cells) > 1 else "").split() if len(t) >= 2]
    flagged = len(toks) >= 2 and sum(_pairs(t) for t in toks) / len(toks) >= 0.8
    if not flagged: return list(r), False
    out = [None if x is None else " ".join(t[0::2] if _pairs(t) else t for t in str(x).split()) for x in r]
    return out, True

def num(x, last=False):
    if x is None: return None
    s = undouble(x).replace(",", "").strip()
    ms = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not ms: return None
    return float(ms[-1] if last else ms[0])

def first_date(text):
    m = re.search(r"(\d{2})-(\d{2})-(\d{4})", text)
    return datetime.strptime(m.group(), "%d-%m-%Y").date() if m else None

def key(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())

REMARK_RULES = [  # (category, regex) - first match wins
    ("gas_shortage",   r"gas\s*short|low gas|gas\s*restr|gas pressure|gas limitation"),
    ("oil_shortage",   r"liquid fuel|fuel short|hfo short|oil short"),   # generic 'fuel shortage' at a GAS plant is re-coded to gas when joined to the plant master
    ("coal_shortage",  r"coal short|coal supply|wet coal"),
    ("payment_arrears",r"late pay|bill delay|neo cut|payment"),
    ("maintenance",    r"maint|overhaul|hgpi|s/d|shutdown|shut down|outage|inspection"),
    ("machine_problem",r"machine|engine|problem|trip|exciter|flame|availability less|availability is less"),
    ("dispatch_standby",r"standby|nldc instruction|as per nldc|to be dispatched|default availability|condenser mode"),
    ("grid_constraint",r"grid voltage|mvar"),
    ("contract_expired",r"contract expired"),
    ("project_test",   r"project|test run"),
    ("running",        r"fgmo"),
]
def remark_flags(r):
    r = str(r or "").lower()
    return (bool(re.search(r"late pay|bill delay|payment", r)), bool(re.search(r"usd|exchange rate|dollar|lc ", r)))

def remark_cat(r):
    r = str(r or "").lower().strip()
    if r in ("", "-", "none", "nan"): return "none"
    for cat, rx in REMARK_RULES:
        if re.search(rx, r): return cat
    return "other"

def unoverlay(garbled, ghost):
    """First row on page 2 of Sheet-2 is printed on top of the repeated header row.
    Recover the real name by removing the ghost string as an interleaved subsequence."""
    G, H = garbled.replace(" ", ""), ghost.replace(" ", "")
    n, m = len(G), len(H)
    if m == 0 or n <= m: return garbled
    @functools.lru_cache(None)
    def ok(i, j):
        if i == n: return j == m
        if j < m and G[i] == H[j] and ok(i + 1, j + 1): return True
        return (n - i - 1) >= (m - j) and ok(i + 1, j)
    if not ok(0, 0): return garbled
    out, i, j = [], 0, 0
    while i < n:
        if j < m and G[i] == H[j] and ok(i + 1, j + 1): i += 1; j += 1
        else: out.append(G[i]); i += 1
    return "".join(out)

def detect(pdf):
    t = (pdf.pages[0].extract_text() or "")[:600]
    if "DAILY ELECTRICITY GENERATION REPORT" in t: return "bpdb_report"
    if "QF-LDC-09" in t: return "substation"
    if "Sheet-2" in t: return "plant_pgcb"
    if "Sheet-1" in t: return "system_pgcb"
    return None

# ------------------------------------------------------------------ Page 2: plant energy (PGCB Sheet-2)
def parse_plant_pgcb(pdf, fname):
    text = pdf.pages[0].extract_text() or ""
    d = first_date(text)
    rows, ghost, pending = [], None, []
    for pg in pdf.pages:
        for tb in pg.extract_tables():
            for r in tb:
                if not r or len(r) < 8: continue
                r, bold = undouble_row(r)
                sl, name, prod, unitcap, pcap, peak, energy, rem = (r + [None]*8)[:8]
                name = (name or "").replace("\n", " ").strip()
                # long names wrap into the Producer cell ("Sikalbaha ... (Baraka Sikalbah" + "a) IPP"): move the spill back
                prod = (prod or "").replace("\n", " ").strip()
                if " " in prod:
                    spill, prod = prod.rsplit(" ", 1)
                    name = (name + spill if not name.endswith(" ") and not spill.startswith(("(", ")")) and name[-1:].isalnum()
                            and spill[:1].islower() else name + ("" if spill.startswith(")") else " ") + spill).strip()
                if sl in ("Sl.",) or (sl == "" and name == "" and prod == ""): continue
                label = str(sl or "")
                if label.endswith("Area Total") or label.endswith("Grid Total"):
                    area = label.replace(" Total", "").strip()
                    for p in pending: p["area"] = area
                    pending = []
                    rows.append(dict(data_date=d, sl=None, plant_raw=label, is_total=True, area=area,
                                     present_capacity_mw=num(pcap), peak_gen_mw=num(peak),
                                     energy_kwh=num(energy), remarks=None, file=fname))
                    continue
                if not name: continue
                if ghost is None: ghost = name
                fixed = False
                if pg.page_number > 1 and ghost and len(name.replace(" ","")) > len(ghost.replace(" ","")) + 3:
                    rec = unoverlay(name, ghost)
                    if rec != name: name, fixed = rec, True
                row = dict(data_date=d, sl=sl or None, plant_raw=name, is_total=False, area=None,
                           producer=(prod or "").strip(), unit_capacity=(unitcap or "").strip(),
                           present_capacity_mw=num(pcap), peak_gen_mw=num(peak), energy_kwh=num(energy),
                           remarks=(rem or "").replace("\n", " ").strip(), name_recovered=fixed,
                           bold_row_fixed=bold, file=fname)
                rows.append(row); pending.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["remark_cat"] = df.remarks.map(remark_cat)
        df["payment_flag"] = df.remarks.map(lambda r: remark_flags(r)[0])
        df["fx_flag"] = df.remarks.map(lambda r: remark_flags(r)[1])
        df["plant_key"] = df.plant_raw.map(key)   # spacing/punctuation-free id: use this to link days
    return df

# ------------------------------------------------------------------ Summary: BPDB report (plant limitation MW + block C)
FUEL_RX = re.compile(r"\s(\(?Gas/HSD\)?|HSD/\s?Gas|Gas|HFO|Coal|\(?Solar\)?|Hydro|HSD|\(?Wind\)?|India|Nepal)\s*\(?([A-Za-z ,\-]*)\)?\s*$")
def parse_bpdb_report(pdf, fname):
    full = "\n".join((p.extract_text() or "") for p in pdf.pages)
    rdate = None
    m = re.search(r"Date\s*:\s*(\d{2})[-.](\d{2})[-.](\d{2,4})", full)   # 15-01-2026 or 01.01.25
    if m:
        y = int(m.group(3)); y = y + 2000 if y < 100 else y
        rdate = datetime(y, int(m.group(2)), int(m.group(1))).date()
    ddate = rdate - timedelta(days=1) if rdate else None
    plants, pending = [], []
    for pg in pdf.pages:
        for tb in pg.extract_tables():
            for r in tb:
                if not r or len(r) < 13: continue
                joined = " ".join(str(x) for x in r if x)
                if "Zone Total" in joined:
                    zone = re.search(r"(\w+) Zone Total", joined).group(1)
                    for p in pending: p["zone"] = zone
                    pending = []; continue
                c = list(r)
                if len(c) == 14: c = c[:2] + c[3:]          # page 2+: name cell spans two columns
                c = [undouble(x) if x is not None else "" for x in c[:13]]
                sl, name = c[0].strip(), c[1].replace("\r", "").strip()
                if not name or name.startswith(("Name of", "Sub-total", "Sub-Total", "Available", "(A)", "(B)", "By ", "Total ")): continue
                if "Gross Total" in sl + name or not re.search(r"\d", c[2] + c[3] + c[4]): continue
                name = name.split("\n")[0].strip()          # merged rows: keep the first plant
                name = re.sub(r"\s+\d+$", "", name)         # wrapped unit-capacity digit
                f = FUEL_RX.search(name)
                fuel = f.group(1).strip("()") if f else None
                owner = f.group(2).strip() if f else None
                pname = name[:f.start()].strip() if f else name
                if fuel is None:
                    low = name.lower()
                    fuel = next((k for k, w in [("Import","import"),("Solar","solar"),("Wind","wind")] if w in low), None)
                pname = re.sub(r"^[ab]\)\s*", "", pname)
                row = dict(data_date=ddate, report_date=rdate, sl=sl or None, plant_raw=pname, fuel=fuel, owner=owner,
                           unit_capacity=c[2], installed_mw=num(c[3], last=True), derated_mw=num(c[4], last=True),
                           actual_day_mw=num(c[5], last=True), actual_eve_mw=num(c[6], last=True),
                           prob_day_mw_next=num(c[7], last=True), prob_eve_mw_next=num(c[8], last=True),
                           fuel_limitation_mw=num(c[9]), shutdown_mw=num(c[10]),
                           remarks=c[11].replace("\n", " ").strip(), zone=None, file=fname)
                plants.append(row); pending.append(row)
    pdf_df = pd.DataFrame(plants)
    if not pdf_df.empty:
        pdf_df["remark_cat"] = pdf_df.remarks.map(remark_cat)
        pdf_df["plant_key"] = pdf_df.plant_raw.map(key)
    g = lambda rx: (lambda mm: float(mm.group(1).replace(",", "")) if mm else None)(re.search(rx, full, re.S))
    sysrow = dict(data_date=ddate, report_date=rdate, file=fname,
        max_demand_eve_mw=g(r"Max\. Demand at eve\. peak \(Generation end\)\s*:\s*([\d.]+)"),
        eve_peak_gen_mw=g(r"Evening-peak Generation \(Generation end\)\s*:\s*([\d.]+)"),
        min_gen_mw=g(r"Minimum Generation \(Generation end\)\s*:\s*([\d.]+)"),
        eve_loadshed_mw=g(r"Evening Peak Load-shed \(Sub-station end\)\s*:\s*([\d.]+)"),
        gas_lf_limitation_mw=g(r"Gas/LF limitation\s*:\s*([\d.]+)"),
        coal_limitation_mw=g(r"Coal supply Limitation\s*:\s*([\d.]+)"),
        kaptai_limitation_mw=g(r"Low water level in Kaptai lake\s*:\s*([\d.]+)"),
        shutdown_maint_mw=g(r"Plants under shut down/ maintenance\s*:\s*([\d.]+)"),
        energy_total_mkwh=g(r"Total Energy \(Generation \+ Import\)\s*:\s*([\d.]+)"),
        energy_gas_mkwh=g(r"By Gas\s*=\s*([\d.]+)"), energy_coal_mkwh=g(r"By Coal\s*=\s*([\d.]+)"),
        energy_oil_mkwh=g(r"By Oil\s*=\s*([\d.]+)"), energy_solar_mkwh=g(r"By Solar\s*=\s*([\d.]+)"),
        energy_import_mkwh=g(r"Imported\s*=\s*([\d.]+)"),
        gas_supplied_mmcfd=g(r"Total Gas Supplied\s*:\s*([\d.]+)"),
        fuelcost_gas_tk=g(r"\(a\) Gas\s*=\s*([\d.]+)"), fuelcost_oil_tk=g(r"\(b\) Oil\s*=\s*([\d.]+)"),
        fuelcost_coal_tk=g(r"\(c\) Coal\s*=\s*([\d.]+)"), fuelcost_import_tk=g(r"\(e\)\s*Import\s*=\s*([\d.]+)"),
        max_temp_c=g(r"Maximum Temperature:\s*([\d.]+)"),
        next_prob_loadshed_mw=g(r"Probable Load Shed:\s*([\d.]+)"),
        next_prob_eve_demand_mw=g(r"Probable Maximum Demand at Evening Peak:\s*([\d.]+)"))
    return pdf_df, pd.DataFrame([sysrow])

# ------------------------------------------------------------------ Page 1: PGCB System Summary
def parse_system_pgcb(pdf, fname):
    text = pdf.pages[0].extract_text() or ""
    d = first_date(text)
    s = dict(data_date=d, file=fname); zones = []
    lab = {"Day Peak Generation":"day_peak_gen_mw","Day Peak Demand":"day_peak_demand_mw",
           "Evening Peak Generation":"eve_peak_gen_mw","Evening Peak Demand":"eve_peak_demand_mw",
           "Minimum Generation of the Day":"min_gen_mw","Maximum Generation of the Day":"max_gen_mw",
           "Energy Generated":"energy_generated_mkwh","Energy Unserverd":"energy_unserved_mkwh",
           "Energy Unserved":"energy_unserved_mkwh","Energy Demand":"energy_demand_mkwh",
           "Maximum Temperature":"max_temp_c","Total Gas Supplied":"gas_supplied_mmcfd",
           "Production Cost per KWHr":"cost_per_kwh_tk"}
    gen, shed = {}, {}
    page_text = pdf.pages[0].extract_text() or ""
    zone_in_mwh = "Generation Summary (MWHr" in page_text          # 2024-25 format reports MWh, later MkWh
    for tb in pdf.pages[0].extract_tables():
        tb = [undouble_row(r)[0] if r and r[0] and _pairs(str(r[0]).replace(" ", "")) else r for r in tb]
        hdr = [str(x or "").strip() for x in tb[0]]
        for r in tb:
            r = [str(x or "").strip() for x in r]
            for i in range(0, len(r)):
                if r[i] in lab and i + 1 < len(r):
                    s[lab[r[i]]] = num(r[i + 1])
                else:                                               # 'Energy Generated 219337.66 MWHr' in one cell
                    for L, k in lab.items():
                        if r[i].startswith(L + " ") and k not in s:
                            v = num(r[i][len(L):])
                            if v is not None and "MWHr" in r[i] and k.endswith("_mkwh"): v = v / 1000
                            s[k] = v
        if hdr[:3] == ["", "Gas", "Coal"]:
            for r in tb[1:]:
                vals = [num(x) for x in r[1:]]
                if zone_in_mwh: vals = [None if v is None else v / 1000 for v in vals]
                gen[str(r[0]).replace(" Zone", "").strip()] = dict(zip(hdr[1:], vals))
        if hdr[:3] == ["Zone", "Load-Shed", "Demand"]:
            for r in tb[1:]:
                r2, _ = undouble_row(r)
                zname = " ".join(t[0::2] if _pairs(t) else t for t in str(r[0]).split())
                shed[zname.strip()] = (num(r2[1]), num(r2[2]))
        if hdr[0] == "Fuel" and len(hdr) >= 4:
            for r in tb[1:]:
                s[f"cost_{str(r[0]).lower()}_tk"] = num(r[1]); s[f"cost_{str(r[2]).lower()}_tk"] = num(r[3])
        for r in tb:
            r = [str(x or "") for x in r]
            if len(r) >= 6 and "Import through" in r[3]:
                k = re.sub(r"Import through | C/B Interconnector:", "", r[3]).strip().lower().replace(" ", "_").replace("total", "total")
                s[f"import_{k.replace('/', '')}_mkwh"] = num(r[5])
    for z in sorted(set(gen) | set(shed)):
        row = dict(data_date=d, zone=z, file=fname)
        for f, v in gen.get(z, {}).items(): row[f"gen_{f.lower()}_mkwh"] = v
        if z in shed: row["eve_loadshed_mw"], row["eve_demand_mw"] = shed[z]
        zones.append(row)
    return pd.DataFrame([s]), pd.DataFrame(zones)

# ------------------------------------------------------------------ Page 3: sub-station max load
def parse_substation(pdf, fname):
    d = first_date(pdf.pages[0].extract_text() or ""); rows = []
    for tb in pdf.pages[0].extract_tables():
        if not tb or "Sub-station" not in [str(x) for x in tb[0]]: continue
        for r in tb[1:]:
            for k in range(0, len(r) - 3, 4):
                name = (r[k + 1] or "").strip()
                if name: rows.append(dict(data_date=d, substation=name, max_load_mw=num(r[k + 2]),
                                          time=(r[k + 3] or "").strip(), file=fname))
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="bpdb_pdfs")
    ap.add_argument("--out", default="parsed")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    out = {k: [] for k in ["plant_day_pgcb","plant_day_bpdb","system_day","system_day_bpdb","zone_day","substation_day"]}
    log = []
    files = [os.path.join(r, f) for r, _, fs in os.walk(a.inp) for f in fs if f.lower().endswith(".pdf")]
    for i, path in enumerate(sorted(files), 1):
        fname = os.path.relpath(path, a.inp)
        try:
            with pdfplumber.open(path) as pdf:
                kind = detect(pdf)
                if kind == "plant_pgcb": out["plant_day_pgcb"].append(parse_plant_pgcb(pdf, fname))
                elif kind == "bpdb_report":
                    p, s = parse_bpdb_report(pdf, fname); out["plant_day_bpdb"].append(p); out["system_day_bpdb"].append(s)
                elif kind == "system_pgcb":
                    s, z = parse_system_pgcb(pdf, fname); out["system_day"].append(s); out["zone_day"].append(z)
                elif kind == "substation": out["substation_day"].append(parse_substation(pdf, fname))
                log.append(dict(file=fname, kind=kind, ok=kind is not None, error=None))
        except Exception as e:
            log.append(dict(file=fname, kind=None, ok=False, error=repr(e)))
        if i % 100 == 0: print(f"{i}/{len(files)} files")
    for k, parts in out.items():
        parts = [p for p in parts if p is not None and not p.empty]
        if not parts: continue
        df = pd.concat(parts, ignore_index=True)
        # re-uploaded reports: keep the last file per day (per plant / zone / sub-station)
        ids = {"plant_day_pgcb":["data_date","plant_raw","is_total"],"plant_day_bpdb":["data_date","plant_raw"],
               "system_day":["data_date"],"system_day_bpdb":["data_date"],"zone_day":["data_date","zone"],
               "substation_day":["data_date","substation"]}[k]
        df = df.sort_values("file").drop_duplicates(ids, keep="last").sort_values(ids)
        df.to_csv(os.path.join(a.out, f"{k}.csv"), index=False)
        print(f"{k}: {len(df)} rows, {df.data_date.nunique()} days")
    # QA: plant energy must add up to the report's own area totals
    fp = os.path.join(a.out, "plant_day_pgcb.csv")
    if os.path.exists(fp):
        p = pd.read_csv(fp)
        tot = p[p.is_total & p.plant_raw.str.contains("Area")].set_index(["data_date", "area"]).energy_kwh
        sm = p[~p.is_total].groupby(["data_date", "area"]).energy_kwh.sum()
        qa = pd.concat([tot, sm], axis=1, keys=["report_total_kwh", "sum_of_plants_kwh"]).reset_index()
        qa["gap_kwh"] = qa.report_total_kwh - qa.sum_of_plants_kwh
        qa.to_csv(os.path.join(a.out, "qa_area_totals.csv"), index=False)
        badqa = qa[qa.gap_kwh.abs() > 1000]
        print(f"QA: {len(badqa)} of {len(qa)} day-area totals do not reconcile (see qa_area_totals.csv).")
    # Draft plant master: link PGCB plant names (energy panel) to BPDB names (fuel type, zone).
    fb = os.path.join(a.out, "plant_day_bpdb.csv")
    if os.path.exists(fp) and os.path.exists(fb):
        p = pd.read_csv(fp); b = pd.read_csv(fb)
        pg = p[~p.is_total].groupby("plant_key").agg(plant_raw=("plant_raw", "last"), area=("area", "last"), producer=("producer", "last"),
                                                     unit_capacity=("unit_capacity", "last"), days=("data_date", "nunique")).reset_index()
        bb = b.groupby("plant_raw").agg(fuel=("fuel", "last"), owner=("owner", "last"), zone=("zone", "last"),
                                        installed_mw=("installed_mw", "last")).reset_index()
        bkeys = {key(n): n for n in bb.plant_raw}
        def best(n):
            m = difflib.get_close_matches(key(n), list(bkeys), n=1, cutoff=0.0)
            if not m: return None, 0.0
            return bkeys[m[0]], round(difflib.SequenceMatcher(None, key(n), m[0]).ratio(), 3)
        pg[["bpdb_name", "match_score"]] = pg.plant_raw.apply(lambda n: pd.Series(best(n)))
        pm = pg.merge(bb, left_on="bpdb_name", right_on="plant_raw", how="left", suffixes=("", "_bpdb")).drop(columns="plant_raw_bpdb")
        pm = pm.rename(columns={"plant_raw": "pgcb_name"})
        pm["check_match"] = pm.match_score < 0.75
        for c in ["gas_source", "gas_distributor", "source_citation"]: pm[c] = ""   # to be filled by hand
        pm.sort_values(["area", "pgcb_name"]).to_csv(os.path.join(a.out, "plant_master_draft.csv"), index=False)
        print(f"plant_master_draft.csv: {len(pm)} plants, {int(pm.check_match.sum())} low-confidence name matches to check.")
    pd.DataFrame(log).to_csv(os.path.join(a.out, "parse_log.csv"), index=False)
    bad = [l for l in log if not l["ok"]]
    print(f"Parsed {len(log)} files; {len(bad)} not recognised or failed (see parse_log.csv).")

if __name__ == "__main__":
    main()
