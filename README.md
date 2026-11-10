# Occupational injuries among foreign workers in Japan, 2023-2025

Industry- and age-standardized incidence ratios (SIRs) by residence status, computed from public aggregate statistics only.

## Run
    pip install pandas scipy matplotlib statsmodels
    python analysis.py
Outputs are written to `output/` (results.json, tables as CSV, Figure1.png, Figure2.png). `table_observed_expected_pooled.csv` gives the observed and expected numbers (Supplementary Table S3) and `table_status_industry_pooled.csv` the industry-specific SIRs including the residual industries (Supplementary Table S4); the combined sensitivity scenarios are in `results.json` under `sensitivity_revision1.all_combined`.

## Data (`data/`)
| File | Content | Source |
|---|---|---|
| ref_injuries.csv | Injuries of all workers by year, industry (6 strata), sex, age. 2023-2024 tabulated from the MHLW open individual-record data; 2025 from the final statistics (no sex) | MHLW, Anzen-info site |
| ref_employees.csv | Employees (persons) by year, industry, sex (T/M/F), age | Labour Force Survey, Table II-2-2 |
| official_rates.csv | Published annual injury rates per 1,000 workers by industry | MHLW |
| ref_injuries_industry8.csv | Injuries of all workers in eight industries | MHLW final statistics |
| foreign_all_by_industry8.csv | Injuries and notified workers of all foreign workers in eight industries; agriculture, fishery and staffing-agency workers | MHLW |
| foreign_workers_by_status_industry.csv | Notified foreign workers by residence status and industry (end of October) | MHLW, Appendix Table 6 |
| foreign_injuries_by_status_industry.csv | Injuries of foreign workers by residence status and industry | MHLW |
| residents_by_status_sex_age.csv | Foreign residents aged 15+ by status group, sex and age (Dec 2023, Jun 2024, Jun 2025) | Immigration Services Agency, e-Stat |
| isa_midyear_residents.csv | Residents with TITP / SSW status at the end of June | Immigration Services Agency |
| employee_population_ratio_2025.csv | Employees / population by sex and age | Labour Force Survey, Table I-B-1 |
| ref_health_welfare_sub.csv | Injuries and employees of all workers in medical services vs. social insurance/welfare/care services, by sex and age | MHLW; Labour Force Survey |
| foreign_workers_health_welfare.csv | Notified foreign workers in the two sub-industries (all statuses) | MHLW, Appendix Table 4 |
| accident_type_2025.csv | Injuries by accident type, foreign workers by status and all workers | MHLW |

Status codes: titp = technical intern training, ssw = specified skilled worker, status_based = permanent residents / spouses / long-term residents,
engineer_humanities = engineer/specialist in humanities/international services, other_professional, designated_activities, outside_status (students and dependents).
Industry codes: mfg, con, com (wholesale/retail), hlt (medical/welfare), hos (accommodation/food and personal services), oth; trn = transport, agf = agriculture/forestry/fishery.

## Notes
- The figures were transcribed from the official tables; totals were checked against published totals. Re-download and re-check before publication.
- The December 2023 immigration table is used for 2023 because the detailed June 2023 table is not published.

## Licence
Code: MIT licence. Data: transcribed from statistics published by the Government of Japan (Ministry of Health, Labour and Welfare; Statistics Bureau; Immigration Services Agency); please cite the original sources.

## Contact
Mestate LLC, Kyoto, Japan (office@mestate.jp)

## Additional sensitivity analyses (revision 1, November 2026)

`output/results.json` → `sensitivity_revision1`:

- `age_by_industry`: SIRs in construction and manufacturing for TITP and SSW when the age structure is allowed to differ between industries (8 or 12 percentage points moved between ages 20–24 and 30–34).
- `all_denominator`: all foreign workers with the mean of adjacent end-of-October numbers as the denominator, and with the notified numbers ±10%.
- `all_six_strata`: all foreign workers from the sum of status-specific expected numbers (six industries).
- `all_age_factor`: all foreign workers under alternative age factors.

## Supplementary tables

`supplement/Supplementary_Tables.pdf` (source: `supplement/Supplementary_Tables.md`) contains Tables S1–S5 cited in the article: further sensitivity analyses (S1), results for each year (S2), observed and expected numbers (S3), industry-specific SIRs (S4) and the age distributions used for standardization (S5; all years in `output/table_age_structure.csv`).
