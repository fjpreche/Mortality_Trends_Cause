# Age- and cause-specific mortality trends in high-income countries, 2012–2019

Code and results for manuscript Francisco J. Pérez-Reche, Jennifer Summers, Gareth T. Jones, Gary J. Macfarlane, *Diverging Pre-Pandemic Mortality Trends: Age-Specific and Cause-Specific
Patterns Across High-Income Countries*.

Everything behind the figures and tables of the paper is here: the estimation code, one output table
per cause level for the combined-sex and the sex-specific analyses, the notebook that draws every
figure, and the workbook of Supplementary File 2. **Cells that could not be estimated are kept in
the output tables, with the reason recorded**, so the denominator of any summary can be
reconstructed from the files themselves.

## Contents

```
mortality_estimates.py          estimation: model, classification, output tables
check_segments.py               segment slopes for the series that reject a straight line
Figures.ipynb                   every figure of the paper and Supplementary File 1, and the workbook
estimates/
  estimates_all_cause.csv               224 rows   country x age group
  estimates_chapter.csv               3,716 rows   country x age group x ICD-10 chapter
  estimates_code.csv.gz             128,788 rows   country x age group x ICD-10 code
  estimates_sex_all_cause.csv           224 rows   as above, sex-by-year interaction
  estimates_sex_chapter.csv           3,716 rows
  estimates_sex_code.csv.gz         128,787 rows
  linearity_segments.csv                 34 rows   series rejecting a single straight line
reference/
  McCartney_breakpoints_our_countries.csv          published breakpoints, for Supplementary File 1, Section 2.5
figures/                        output of Figures.ipynb (main-text figures also as .tif)
workbooks/
  Supplementary_estimates.xlsx                     Supplementary File 2
```

The two code-level tables are gzip-compressed; pandas reads them directly.

One row of an output table is one fitted model: a country × age group × cause cell, over the seven
age groups 0, 1–14, 15–29, 30–44, 45–59, 60–74 and 75+. The all-cause tables cover the 32 countries
of the analysis; the cause-level tables additionally carry rows for Greece, New Zealand and Norway,
whose cause-coded data support no estimate at any level, so that the files record why those
countries are absent from the results.

## Figures

| file in `figures/` | in the paper |
|---|---|
| `Figure1_AllCause_APC_heatmap` | Figure 1 |
| `Figure2_Chapter_APC_heatmap` | Figure 2 |
| `Figure3_TrendCategory_shares` | Figure 3 |
| `FigureS1_Linearity` | Supplementary File 1, Figure S1 |
| `FigureS2_Estimator_comparison` | Figure S2 |
| `FigureS3_Level_relative_to_median` | Figure S3 |
| `FigureS4_AllCause_forest_all_ages` | Figure S4 |
| `FigureS5_Detectability` | Figure S5 |
| `FigureS6_Precision_funnel` | Figure S6 |
| `FigureS7_Sex_APC_heatmaps` | Figure S7 |
| `FigureS8_Sex_dumbbell` | Figure S8 |
| `FigureS9_Sex_resolvability` | Figure S9 |
| `FigureS10_Sex_chapter_differences` | Figure S10 |

## Data sources

The analysis uses two public sources and redistributes neither.

- **All-cause deaths and exposures**: [Human Mortality Database](https://www.mortality.org), single
  years of age.
- **Cause-specific deaths**: [WHO Mortality
  Database](https://www.who.int/data/data-collection-tools/who-mortality-database), at its reporting
  age bands, coded to ICD-10.

Both require registration, so the raw inputs are not included. **TODO:** state the extract dates.

## Columns of the output tables

### Identifiers and inputs

| column | meaning |
|---|---|
| `country`, `age_group`, `cause_level`, `cause`, `cause_label` | the cell |
| `year_min`, `year_max`, `n_years` | the window actually used |
| `total_deaths`, `person_years`, `mean_annual_deaths` | the data behind the fit |
| `n_years_with_deaths` | sparsity of the series |
| `share_of_group_deaths_pct` | the cause's share of deaths in that country and age group |

### The estimate and its uncertainty

| column | meaning |
|---|---|
| `apc`, `apc_lo`, `apc_hi` | annual percentage change and its 95% confidence interval |
| `beta`, `se` | the log-linear trend per year and its standard error |
| `p`, `p_bh` | the p-value and its Benjamini–Hochberg adjustment |
| `dispersion`, `dispersion_used`, `df_resid` | over-dispersion and residual degrees of freedom |
| `mde80` | smallest annual change the series could detect with 80% power |
| `category` | one of the five classes below |
| `test_family`, `n_tests_in_family` | the family within which the false discovery rate is controlled |

### Classification

A materiality threshold of **1% per year** is applied alongside significance. With `q` the adjusted
p-value and `(apc_lo, apc_hi)` the interval:

| `category` | condition |
|---|---|
| `increasing` | `q < 0.05` and `apc >= 1` |
| `decreasing` | `q < 0.05` and `apc <= -1` |
| `significant but small` | `q < 0.05` and `|apc| < 1` |
| `no material change` | `q >= 0.05` and the interval lies inside ±1 |
| `inconclusive` | `q >= 0.05` and the interval admits a change of 1 or more |

*No material change* excludes a change of material size; *inconclusive* is an absence of evidence.
The paper reports *significant but small* and *no material change* together as **stagnation**.
Figures 1 and 2 mark the classes as follows: a filled dot for a material change, an open circle for
*significant but small*, cross-hatching for *no material change*, and no mark for *inconclusive*.

### Estimability

`status` takes the values `ok`, `sparse`, `not estimable`, `unstable` and `failed`. A cell counts as
estimable when `status` is `ok` or `sparse`; the other values carry no effect size, and
`status_detail` gives the reason.

### Comparison estimates

| column | meaning |
|---|---|
| `crude_` prefix | the same model with the age term suppressed |
| `apc_minus_crude_apc` | the effect of adjusting for age composition, in percentage points per year |
| `dsr_esp2013_first`, `dsr_esp2013_last` | directly standardised rate (ESP2013) in the first and last year |
| `dsr_apc` | slope of the log directly standardised rate |

### Linearity test

`lin_knot_year`, `lin_dev_drop_knot`, `lin_F_knot` and `lin_p_knot` give the fixed-knot test of
Supplementary File 1, Section 2.5. `linearity_segments.csv` holds one row for each all-cause series
in which a single straight line is rejected, with the slope over the whole window and the slopes
before and after the knot, with their intervals; it is written by `check_segments.py`.

Other columns in the tables are diagnostics written by the estimation code and not used in the paper.

### The sex-specific tables

One fitted interaction model per cell, so quantities are suffixed by the sex they describe:
`apc_male`, `apc_female`, their common component `apc_sexavg`, and the difference `apc_diff`, the
additional annual percentage change in men. Each carries its own interval, p-value, adjusted
p-value, minimum detectable effect and category.

## Reproducing the analysis

Tested with Python 3.12; package versions are in `requirements.txt`.

**Figures and workbook from the tables provided.** From the repository root, run `Figures.ipynb`
top to bottom. It reads `estimates/` and `reference/` and writes `figures/` and `workbooks/`.

**The tables from the raw data.** Place the HMD and WHO extracts under one folder, laid out as in
`CONFIG` in `mortality_estimates.py`, and set `MORTALITY_BASE` to that folder. Then:

```
P="python mortality_estimates.py --age-groups revised --linearity-test"
$P --causes    --level chapter --out estimates_chapter.csv
$P --all-cause --countries-from estimates_chapter.csv --out estimates_all_cause.csv
$P --causes    --level code    --out estimates_code.csv
$P --sex-model --level all-cause --countries-from estimates_chapter.csv --out estimates_sex_all_cause.csv
$P --sex-model --level chapter --out estimates_sex_chapter.csv
$P --sex-model --level code    --out estimates_sex_code.csv
python check_segments.py estimates_all_cause.csv
```

The code-level runs take of the order of twenty minutes each. `python mortality_estimates.py
--selftest` runs the simulation checks of the estimator.

## Citation

**TODO:** add the citation of the paper and the archive DOI.

## Licence

Code: MIT (see `LICENSE`). Output tables, figures and workbook: CC BY 4.0. Neither input database may
be redistributed under these terms; see their own conditions of use.
