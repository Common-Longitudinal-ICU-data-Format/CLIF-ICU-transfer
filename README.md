# CLIF ICU Transfer Analysis

## CLIF version

2.1.0

## Overview

Identifies hospitalizations coded as outside-hospital (OSH) admissions that are actually transfers linked to a prior encounter at the same site (discharged ≤6 hours before the OSH admission), plus the subset of those transferred directly to the ICU. Compares ICU length of stay, hospital length of stay, ICU/hospital type, mortality, and discharge category before vs. after the transfer.

## How to run with uv

```bash
uv sync
uv run python 00_icu_transfer_analysis.py
```

`config/config.json` must be filled in first.

## Outputs

Written to `output_directory` (from `config/config.json`, or overridden with `--output-dir`). Every file name ends in `_<site_name>` (the lowercased `site_name` from config).

| File | Description |
|------|-------------|
| `cohort_summary_<site>.csv` | OSH cohort counts: total OSH, in-system OSH (linked prior encounter), direct-to-ICU transfers |
| `icu_los_summary_<site>.csv` | ICU length of stay (first-stay and cumulative), long format, by cohort/period |
| `hospital_los_summary_<site>.csv` | Hospital length of stay by cohort/period |
| `icu_type_summary_in_system_osh_<site>.csv` | ICU type counts/percentages, in-system OSH cohort, before vs. after transfer |
| `icu_type_summary_direct_icu_<site>.csv` | ICU type counts/percentages, direct-to-ICU transfer cohort |
| `hospital_type_summary_in_system_osh_cohort_<site>.csv` | Hospital type counts/percentages, in-system OSH cohort |
| `hospital_type_summary_direct_icu_<site>.csv` | Hospital type counts/percentages, direct-to-ICU transfer cohort |
| `mortality_summary_<site>.csv` | Mortality rate by cohort |
| `discharge_summary_overall_osh_<site>.csv` | Discharge category, overall OSH cohort |
| `discharge_summary_in_system_osh_cohort_<site>.csv` | Discharge category, in-system OSH cohort |
| `discharge_summary_direct_icu_<site>.csv` | Discharge category, direct-to-ICU transfer cohort |
| `sankey_icu_type_in_system_osh_cohort_<site>.png` | ICU type before → after transfer, in-system OSH cohort |
| `sankey_icu_type_direct_icu_<site>.png` | ICU type before → after transfer, direct-to-ICU transfer cohort |
| `sankey_hospital_type_in_system_osh_cohort_<site>.png` | Hospital type before → after transfer, in-system OSH cohort |
| `sankey_hospital_type_direct_icu_<site>.png` | Hospital type before → after transfer, direct-to-ICU transfer cohort |
| `sankey_discharge_category_in_system_osh_cohort_<site>.png` | Discharge category before → after transfer, in-system OSH cohort |
| `sankey_discharge_category_direct_icu_<site>.png` | Discharge category before → after transfer, direct-to-ICU transfer cohort |
