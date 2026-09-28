# Bangladesh power system: plant-day panel, 2024-2026

A daily, plant-level panel of Bangladesh's electricity system, constructed from the public archive of daily generation reports published by the Bangladesh Power Development Board (BPDB) and the National Load Dispatch Center of Power Grid Bangladesh (PGCB).

The data files are deposited separately at Mendeley Data; this repository holds the code that builds them.

## Coverage

- **Period:** 31 December 2024 to 21 September 2026 (629 days).
- **Units:** 163 power plants, 97,925 plant-days.
- **Geography:** national, with nine dispatch areas and BPDB zones.
- **Source documents:** 2,512 PDF reports from https://misc.bpdb.gov.bd/daily-generation-archive

## Files (deposited at Mendeley Data)

| File | Rows | Content |
| --- | --- | --- |
| `plant_day_panel.csv` | 97,925 | Plant-day panel: capacity, evening-peak output, daily energy, stated reason for any shortfall, and plant attributes |
| `system_day_panel.csv` | 629 | System-day series: demand, unserved energy, gas supplied to power, generation by fuel, production cost, temperature |
| `zone_day_panel.csv` | 5,616 | Zone-day series: generation by fuel, evening-peak demand and load shed |
| `plant_master_v4.csv` | 163 | Plant classification: fuel, capability, zone, gas distributor, network position |
| `plant_aliases_v4.csv` | 185 | Crosswalk from every observed name spelling to a stable plant identifier |
| `codebook.csv` | 150 | Definition, units and source for every variable in every file |
| `qa_after_cleaning.csv` | — | Day-area reconciliation against the subtotals printed in the source reports |
| `data_quality.txt` | — | Log of the construction run |

## Construction

Scripts run in this order:

1. `bpdb_archive_download.py` — downloads the daily reports from the BPDB archive.
2. `bpdb_parse.py` — identifies each document by content and extracts its tables.
3. `build_panel.py` — cleans, links plant identities, applies plausibility screens and builds the panels.
4. `analysis.py`, `robustness.py`, `placebo_years.py`, `multibase.py`, `fsru_compare.py`, `cluster_inference.py` — reproduce the estimates in the accompanying paper.

Requirements: Python 3.11+, `requests`, `beautifulsoup4`, `pdfplumber`, `pandas`, `numpy`, `statsmodels`, `pyfixest`, `matplotlib`.

## Validation

- **Internal reconciliation.** Plant-level energy is compared with the area subtotals printed in each report across 5,585 area-days; after cleaning, 48 (0.9 per cent) differ by more than one per cent and are recorded in `qa_after_cleaning.csv`. Energy for those cases is coded missing, not zero.
- **Plausibility screens.** Energy above capacity x 24 hours (+10 per cent) and peak output above capacity (+30 per cent) are set to missing: 27 energy, 15 peak and 18 capacity values.
- **External check.** The panel reproduces an independently announced policy action: following the 50 mmcfd reduction in gas to power announced on 5 March 2026, estimated deliveries are about 72 mmcfd below their year-earlier level in March and April.
- **Cross-source check.** Over 2025 the series correlate at 0.99 (gas generation) and 0.998 (unserved energy) with the independent hourly PGCB dataset at doi 10.17632/vpk8spw2mm.1.

## Known limitations

- A small number of days are missing from the archive, and one substation document could not be read.
- The wording of plant remarks changes over the period: reports in early 2025 frequently give no reason for idling, so shortage attribution should use the BPDB fuel-limitation fields, which are consistent throughout.
- The `gas_distributor` and `exposure_group` classifications are provisional and are not official; rows still to be confirmed are flagged in `plant_master_v4.csv`.
- Plant-level gas deliveries are not published by Petrobangla and are therefore not in this panel.
- Reported demand includes estimated unserved demand, so shortages may be understated when demand is suppressed.

## Provenance and licence

The source documents are public records published by BPDB and PGCB. This repository contains the code that produces the derived data, not bulk copies of the source PDFs, which remain available from the BPDB archive. Code in this repository is released under the MIT licence; the derived data deposited at Mendeley Data are released under CC BY 4.0.

## Citation

Navin, Y. (2026). *Bangladesh power system: plant-day panel, 2024-2026* [Data set]. Mendeley Data. doi: [to be assigned]

Accompanying article: Navin, Y. *Prices or quantities? Administered energy prices, inventories and the 2026 Hormuz shock in Bangladesh.*

ORCID: https://orcid.org/0009-0003-4608-7236

## Contact

Y. Navin, navinkumer14388@gmail.com
