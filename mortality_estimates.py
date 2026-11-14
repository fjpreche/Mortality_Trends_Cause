"""
mortality_estimates.py
======================

Builds the estimates tables accompanying "Diverging Pre-Pandemic Mortality
Trends: Age-Specific and Cause-Specific Patterns Across High-Income Countries":
one row per (country x age group x cause), carrying effect magnitude,
uncertainty, multiplicity-adjusted p-values, minimum detectable effect and the
trend class.

Every figure and supplementary table is generated from these tables, so that
classification happens at presentation time and never inside the estimation.

Method (specified in full in Supplementary File 1)
------
  * counts are modelled directly (quasi-Poisson with a log link and log
    person-years as offset), so the variance scales with the expected number
    of deaths;
  * age composition within each broad band is adjusted for by a factor for
    each age unit (single years of age for all-cause mortality, WHO reporting
    bands for causes), i.e. internal age standardisation;
  * the annual percentage change (APC) and its 95% CI are the primary estimate;
  * two comparison estimates are returned alongside it: the same model without
    the age term, and the slope of the log directly standardised rate
    (ESP2013); an OLS slope of the crude rate, in deaths per 100,000 per year,
    is kept as a descriptive column;
  * Benjamini-Hochberg FDR control is applied within an explicitly declared
    family of tests;
  * the minimum detectable APC at 80% power is reported for every cell.

Dependencies
------------
numpy and pandas only. scipy and statsmodels are used if importable (exact
t-distribution p-values, negative-binomial cross-check) but are not required.

Usage
-----
The tables of the paper were built with:

    P="python mortality_estimates.py --age-groups revised --linearity-test"
    $P --causes    --level chapter --out estimates_chapter.csv
    $P --all-cause --countries-from estimates_chapter.csv --out estimates_all_cause.csv
    $P --causes    --level code    --out estimates_code.csv
    $P --sex-model --level all-cause --countries-from estimates_chapter.csv \
        --out estimates_sex_all_cause.csv
    $P --sex-model --level chapter --out estimates_sex_chapter.csv
    $P --sex-model --level code    --out estimates_sex_code.csv

and the simulation checks run with ``python mortality_estimates.py --selftest``.

or from a notebook:

    import mortality_estimates as me
    est = me.build_all_cause_estimates()
    est = me.finalise(est)                     # adds p_bh, mde80, category
    est.to_csv("estimates_all_cause.csv", index=False)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from dataclasses import dataclass, asdict
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # optional, only for exact p-values
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover
    _scipy_stats = None


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# Data root: the folder holding the HMD extract (OurWorld_in_data/...) and the
# WHO Mortality Database files (WHO/...), laid out as in CONFIG below. Set the
# MORTALITY_BASE environment variable to point at it; by default it is ./data.
# Importing the module without the data is allowed (the figures notebook uses
# its helper functions); building a table without it is not.
BASE = Path(os.environ.get("MORTALITY_BASE") or "data")
if not BASE.exists():
    warnings.warn(f"data root does not exist: {BASE}; "
                  f"set MORTALITY_BASE to the folder holding OurWorld_in_data/ and WHO/")

CONFIG = {
    # HMD, single year of age
    "hmd_exposures_dir": BASE / "OurWorld_in_data/Data/Period_data/exposures/Exposures_1x1",
    "hmd_deaths_lexis_dir": BASE / "OurWorld_in_data/Data/Period_data/deaths/Deaths_lexis",
    "hmd_mx_dir": BASE / "OurWorld_in_data/Data/Period_data/death_rates/Mx_5x1",
    # WHO mortality database (ICD-10), the four part files
    "who_parts": [
        BASE / "WHO/Data/morticd10_part3_2008-2012/Morticd10_part3",
        BASE / "WHO/Data/morticd10_part4_2013-2016/Morticd10_part4",
        BASE / "WHO/Data/morticd10_part5_2017-2020/Morticd10_part5",
        BASE / "WHO/Data/morticd10_part6_2021-/Morticd10_part6",
    ],
    "who_country_codes": BASE / "WHO/Data/mort_country_codes/country_codes",
    # analysis period
    "yearmin": 2012,
    "yearmax": 2019,
    # a country must have this many usable years to be analysed
    "min_years": 8,
}

# Infancy is separated from childhood because its causes of death are distinct
# and the numbers support it. Two cautions when using the "0" band:
#   * the denominator here is person-years of exposure, not live births, so the
#     LEVEL is not the conventional infant mortality rate per 1000 live births;
#     the TREND (the annual percentage change) is nonetheless well defined and
#     is what this analysis reports;
#   * "0-14" is the union of "0" and "1-14", so including all three in one run
#     puts overlapping tests in the same false discovery rate family. Run the
#     split bands for the analysis and "0-14" separately, as a comparison with
#     the analysis with infancy and childhood pooled.
AGE_BINS: Dict[str, Tuple[int, int]] = {
    "0": (0, 0),          # infants, reported separately
    "1-14": (1, 14),
    "0-14": (0, 14),      # infancy and childhood pooled
    "15-29": (15, 29),
    "30-44": (30, 44),
    "45-59": (45, 59),
    "60-74": (60, 74),
    "75+": (75, 200),
}

#: age groups with infancy and childhood pooled
PAPER_AGE_GROUPS = ["0-14", "15-29", "30-44", "45-59", "60-74", "75+"]

#: age groups of the analysis reported in the paper, with infancy separated
#: from childhood. Note that "0-14" is deliberately absent - see AGE_BINS.
REVISED_AGE_GROUPS = ["0", "1-14", "15-29", "30-44", "45-59", "60-74", "75+"]

#: HMD populations not analysed: subpopulations, and countries whose coverage
#: of 2012-2019 is incomplete in one of the two sources (Supplementary File 1,
#: Section 1).
EXCLUDED_COUNTRIES = [
    "Belarus",
    "Israel",
    "Russia",
    "Ukraine",
    "Slovakia",
    "New Zealand -- Maori",
    "New Zealand -- Non-Maori",
]

#: WHO Morticd10 death columns -> closed age interval (years).
#: Deaths1 is all ages and Deaths26 is "age unspecified"; both are excluded from
#: the band model but Deaths26 is reported as a data-quality diagnostic.
WHO_BANDS: Dict[str, Tuple[int, int]] = {
    "Deaths2": (0, 0),
    "Deaths3": (1, 1),
    "Deaths4": (2, 2),
    "Deaths5": (3, 3),
    "Deaths6": (4, 4),
    "Deaths7": (5, 9),
    "Deaths8": (10, 14),
    "Deaths9": (15, 19),
    "Deaths10": (20, 24),
    "Deaths11": (25, 29),
    "Deaths12": (30, 34),
    "Deaths13": (35, 39),
    "Deaths14": (40, 44),
    "Deaths15": (45, 49),
    "Deaths16": (50, 54),
    "Deaths17": (55, 59),
    "Deaths18": (60, 64),
    "Deaths19": (65, 69),
    "Deaths20": (70, 74),
    "Deaths21": (75, 79),
    "Deaths22": (80, 84),
    "Deaths23": (85, 89),
    "Deaths24": (90, 94),
    "Deaths25": (95, 200),
}

#: European Standard Population 2013, per 100 000, by 5-year group start age.
ESP2013: Dict[Tuple[int, int], float] = {
    (0, 0): 1000,
    (1, 4): 4000,
    (5, 9): 5500,
    (10, 14): 5500,
    (15, 19): 5500,
    (20, 24): 6000,
    (25, 29): 6000,
    (30, 34): 6500,
    (35, 39): 7000,
    (40, 44): 7000,
    (45, 49): 7000,
    (50, 54): 7000,
    (55, 59): 6500,
    (60, 64): 6000,
    (65, 69): 5500,
    (70, 74): 5000,
    (75, 79): 4000,
    (80, 84): 2500,
    (85, 89): 1500,
    (90, 94): 800,
    (95, 200): 200,
}

#: ICD-10 chapters as (roman numeral, label, (first letter, first number),
#: (last letter, last number)).
ICD10_CHAPTERS: List[Tuple[str, str, Tuple[str, int], Tuple[str, int]]] = [
    ("I", "Infectious and parasitic diseases", ("A", 0), ("B", 99)),
    ("II", "Neoplasms", ("C", 0), ("D", 48)),
    ("III", "Blood and immune system", ("D", 50), ("D", 89)),
    ("IV", "Endocrine, nutritional and metabolic", ("E", 0), ("E", 90)),
    ("V", "Mental and behavioural disorders", ("F", 0), ("F", 99)),
    ("VI", "Nervous system", ("G", 0), ("G", 99)),
    ("VII", "Eye and adnexa", ("H", 0), ("H", 59)),
    ("VIII", "Ear and mastoid process", ("H", 60), ("H", 95)),
    ("IX", "Circulatory system", ("I", 0), ("I", 99)),
    ("X", "Respiratory system", ("J", 0), ("J", 99)),
    ("XI", "Digestive system", ("K", 0), ("K", 93)),
    ("XII", "Skin and subcutaneous tissue", ("L", 0), ("L", 99)),
    ("XIII", "Musculoskeletal and connective tissue", ("M", 0), ("M", 99)),
    ("XIV", "Genitourinary system", ("N", 0), ("N", 99)),
    ("XV", "Pregnancy, childbirth and the puerperium", ("O", 0), ("O", 99)),
    ("XVI", "Perinatal conditions", ("P", 0), ("P", 96)),
    ("XVII", "Congenital malformations", ("Q", 0), ("Q", 99)),
    ("XVIII", "Symptoms and ill-defined conditions", ("R", 0), ("R", 99)),
    ("XIX", "Injury and poisoning (nature of injury)", ("S", 0), ("T", 98)),
    ("XX", "External causes of morbidity and mortality", ("V", 1), ("Y", 98)),
    ("XXI", "Factors influencing health status", ("Z", 0), ("Z", 99)),
    ("XXII", "Codes for special purposes", ("U", 0), ("U", 99)),
]


# ----------------------------------------------------------------------------
# Small statistical utilities (numpy only)
# ----------------------------------------------------------------------------

#: z_{0.975}; and z_{0.975} + z_{0.80}, the multiplier for the minimum
#: detectable effect at 80% power and a two-sided 5% test.
Z975 = 1.959963984540054
MDE_MULT = 2.801585552618561



# ----------------------------------------------------------------------------
# Distribution functions (no scipy required)
# ----------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float, itmax: int = 300, eps: float = 3e-14) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """Exact two-sided p-value from Student's t, via the incomplete beta."""
    t = abs(float(t))
    if not np.isfinite(t) or df <= 0:
        return np.nan
    return float(betainc(df / 2.0, 0.5, df / (df + t * t)))


def f_sf(f: float, df1: float, df2: float) -> float:
    """Upper tail of the F distribution, via the incomplete beta."""
    f = float(f)
    if not np.isfinite(f) or f <= 0:
        return 1.0
    return float(betainc(df2 / 2.0, df1 / 2.0, df2 / (df2 + df1 * f)))


def _norm_sf(z: float) -> float:
    """Upper tail of the standard normal, without scipy."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def two_sided_p(t: float, df: Optional[int] = None) -> float:
    """Two-sided p-value from a t or z statistic.

    Uses the t distribution when scipy is available and ``df`` is given
    (important here: the series are 8 years long), otherwise the normal
    approximation, which is anti-conservative for such short series.
    """
    t = abs(float(t))
    if not np.isfinite(t):
        return np.nan
    if df is not None and df > 0:
        return t_two_sided_p(t, df)      # exact, and independent of scipy
    return float(2.0 * _norm_sf(t))


def t_quantile(df: Optional[int], q: float = 0.975) -> float:
    """Two-sided critical value; falls back to the normal quantile."""
    if df is not None and _scipy_stats is not None:
        return float(_scipy_stats.t.ppf(q, df))
    return Z975


def benjamini_hochberg(p: Sequence[float]) -> np.ndarray:
    """BH step-up adjusted p-values (q-values), NaNs preserved.

    Valid under independence and under positive regression dependence. Trends
    for related ICD codes within a country are positively dependent in
    practice, but see ``benjamini_yekutieli`` for a dependence-agnostic check.
    """
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.clip(ranked, 0, 1)
    out[ok] = adj
    return out


def benjamini_yekutieli(p: Sequence[float]) -> np.ndarray:
    """BY adjusted p-values: BH inflated by the harmonic number, valid under
    arbitrary dependence."""
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return np.full(p.shape, np.nan)
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    return np.clip(benjamini_hochberg(p) * c_m, 0, 1)


def pct_change_from_log(x: float) -> float:
    """Convert a log rate ratio to a percentage change, without overflowing.

    Sparse cause-of-death series (many zero years) can produce enormous or
    near-separable slope estimates; ``math.exp`` overflows above ~709. Such
    cells are meaningless as effect sizes and are flagged by
    :func:`fit_poisson_trend`, but the arithmetic must not raise.
    """
    if x is None or not np.isfinite(x):
        return np.nan
    if x > 700.0:
        return np.inf
    if x < -700.0:
        return -100.0
    return 100.0 * math.expm1(x)


#: |beta| or SE above this marks a fit as unstable (|APC| > ~1900 %/year).
UNSTABLE_BETA = 3.0

#: Lower bound applied to the estimated Pearson dispersion before it scales the
#: covariance. phi_hat comes from n - p residual degrees of freedom (as few as
#: 20 at chapter level), so values below 1 are usually sampling noise; using
#: them would make the intervals narrower than Poisson. Set to None to use the
#: raw estimate.
DISPERSION_FLOOR_DEFAULT = 1.0


@dataclass
class PoissonFit:
    """Result of one log-linear trend fit."""

    beta: float          # log rate ratio per year
    se: float            # standard error of beta (quasi-Poisson by default)
    dispersion: float    # Pearson chi2 / residual df
    df_resid: int
    n_obs: int
    converged: bool
    lag1_resid_acf: float   # lag-1 autocorrelation of Pearson residuals
    deviance: float
    n_years_with_deaths: int = 0
    note: str = ""          # non-empty when the fit should not be interpreted
    dispersion_used: float = np.nan   # after the floor; this scales the covariance
    nb_alpha: float = np.nan          # negative-binomial dispersion, when fitted

    @property
    def apc(self) -> float:
        """Annual percentage change, %/year."""
        return pct_change_from_log(self.beta)

    def ci(self, use_t: bool = True) -> Tuple[float, float]:
        if not np.isfinite(self.beta) or not np.isfinite(self.se):
            return np.nan, np.nan
        crit = t_quantile(self.df_resid if use_t else None)
        return (
            pct_change_from_log(self.beta - crit * self.se),
            pct_change_from_log(self.beta + crit * self.se),
        )

    @property
    def p_value(self) -> float:
        if not np.isfinite(self.se) or self.se <= 0:
            return np.nan
        return two_sided_p(self.beta / self.se, self.df_resid)

    @property
    def mde80(self) -> float:
        """Minimum APC detectable with 80% power at a two-sided 5% level."""
        if not np.isfinite(self.se):
            return np.nan
        return 100.0 * (math.exp(MDE_MULT * self.se) - 1.0)


def fit_poisson_trend(
    deaths: np.ndarray,
    person_years: np.ndarray,
    year: np.ndarray,
    age_index: Optional[np.ndarray] = None,
    dispersion: str = "quasi",
    max_iter: int = 100,
    tol: float = 1e-10,
    dispersion_floor: Optional[float] = DISPERSION_FLOOR_DEFAULT,
) -> PoissonFit:
    """Fit ``log E[deaths] = a_age + beta * (year - mean) + log(person_years)``.

    Parameters
    ----------
    deaths, person_years, year
        Equal-length arrays. Rows are (age unit x year) cells when
        ``age_index`` is given, otherwise one row per year.
    age_index
        Integer codes for the age units (single years of age for all-cause,
        WHO 5-year bands for cause-specific). Including this factor is what
        removes residual confounding by age composition within a broad band:
        ``beta`` is then an internally age-standardised trend.
    dispersion
        ``"quasi"`` scales the covariance by the Pearson dispersion (the
        default: annual counts of causes of death are routinely overdispersed
        relative to Poisson), ``"poisson"`` leaves it unscaled, ``"sandwich"``
        uses an HC1 robust covariance.

    Notes
    -----
    Iteratively reweighted least squares, implemented directly so that the
    module needs only numpy. ``beta`` is estimated on centred years, which does
    not change the slope but improves conditioning.
    """
    y = np.asarray(deaths, dtype=float)
    E = np.asarray(person_years, dtype=float)
    t = np.asarray(year, dtype=float)
    if y.shape != E.shape or y.shape != t.shape:
        raise ValueError("deaths, person_years and year must have equal length")

    good = np.isfinite(y) & np.isfinite(E) & np.isfinite(t) & (E > 0)
    y, E, t = y[good], E[good], t[good]
    if age_index is not None:
        age_index = np.asarray(age_index)[good]

    if y.size == 0:
        return PoissonFit(np.nan, np.nan, np.nan, 0, 0, False, np.nan, np.nan,
                          0, "no usable observations")

    # A trend is not identifiable from a handful of non-zero years: the slope
    # either diverges or is driven entirely by one year. Refuse to fit rather
    # than return a number that would be read as an effect size.
    years_with_deaths = int(pd.Series(y).groupby(pd.Series(t)).sum().gt(0).sum())
    n_distinct_years = int(np.unique(t).size)
    if years_with_deaths < 3:
        return PoissonFit(np.nan, np.nan, np.nan, 0, int(y.size), False, np.nan, np.nan,
                          years_with_deaths,
                          f"only {years_with_deaths} of {n_distinct_years} years have deaths")

    tc = t - t.mean()

    # design matrix: age dummies (or a single intercept) plus centred year
    if age_index is not None and len(np.unique(age_index)) > 1:
        levels = np.unique(age_index)
        D = np.column_stack([(age_index == lv).astype(float) for lv in levels])
    else:
        D = np.ones((y.size, 1))
    X = np.column_stack([D, tc])
    n, p = X.shape
    if n <= p:
        return PoissonFit(np.nan, np.nan, np.nan, 0, n, False, np.nan, np.nan,
                          years_with_deaths, "fewer observations than parameters")

    if dispersion == "nb":
        res = _fit_nb(X, y, E, max_iter=max_iter, tol=tol)
    else:
        res = _irls_poisson(X, y, E, dispersion=dispersion, max_iter=max_iter,
                            tol=tol, dispersion_floor=dispersion_floor)
    if res is None:
        return PoissonFit(np.nan, np.nan, np.nan, 0, n, False, np.nan, np.nan,
                          years_with_deaths, "IRLS failed")

    beta_hat = res["theta"]
    cov = res["cov"]
    phi = res["dispersion"]
    deviance = res["deviance"]
    converged = res["converged"]
    df_resid = res["df_resid"]
    pearson = res["pearson"]

    se = float(np.sqrt(max(cov[-1, -1], 0.0)))

    # lag-1 autocorrelation of residuals, averaged over age units: a positive
    # value means the CI above is still optimistic (serially correlated years)
    acf = _mean_lag1_acf(pearson, t, age_index)

    beta = float(beta_hat[-1])
    note = ""
    if not converged:
        note = "IRLS did not converge"
    elif abs(beta) > UNSTABLE_BETA or (np.isfinite(se) and se > UNSTABLE_BETA):
        # near-separable: e.g. deaths only in the last few years of the period
        note = "unstable fit (near-separable sparse series)"

    return PoissonFit(
        beta=beta,
        se=se,
        dispersion=phi,
        df_resid=int(df_resid),
        n_obs=int(n),
        converged=bool(converged),
        lag1_resid_acf=acf,
        deviance=deviance,
        n_years_with_deaths=years_with_deaths,
        note=note,
        dispersion_used=res.get("dispersion_used", res["dispersion"]),
        nb_alpha=res.get("alpha", np.nan),
    )


def _nb_loglik(y: np.ndarray, mu: np.ndarray, alpha: float) -> float:
    """Negative-binomial (NB2) log-likelihood, Var(y) = mu + alpha*mu^2."""
    if alpha <= 0:
        return float(np.sum(y * np.log(np.maximum(mu, 1e-12)) - mu
                            - np.array([math.lgamma(v + 1) for v in y])))
    r = 1.0 / alpha
    out = 0.0
    for yi, mi in zip(y, mu):
        out += (math.lgamma(yi + r) - math.lgamma(r) - math.lgamma(yi + 1)
                + yi * math.log(alpha * mi) - (yi + r) * math.log1p(alpha * mi))
    return float(out)


def _irls_poisson(
    X: np.ndarray,
    y: np.ndarray,
    E: np.ndarray,
    dispersion: str = "quasi",
    max_iter: int = 100,
    tol: float = 1e-10,
    alpha: float = 0.0,
    dispersion_floor: Optional[float] = None,
) -> Optional[dict]:
    """Fit ``log E[y] = X theta + log(E)`` by iteratively reweighted least squares.

    Shared by the age-only model (:func:`fit_poisson_trend`) and the
    age-by-sex model (:func:`fit_poisson_sex_trend`) so that both use exactly
    the same estimator, dispersion handling and covariance.

    Returns a dict with the coefficient vector, its covariance, the Pearson
    dispersion, residual deviance, residual degrees of freedom, Pearson
    residuals and a convergence flag; or ``None`` if the fit failed.
    """
    n, p = X.shape
    offset = np.log(E)
    eta = np.log(np.maximum(y, 0.5)) - offset  # start from the observed rates
    theta = None
    converged = False
    XtWX_inv = None

    for _ in range(max_iter):
        mu = np.clip(np.exp(eta + offset), 1e-12, None)
        # log link: W = mu^2 / V(mu). Poisson V = mu gives W = mu; negative
        # binomial V = mu(1 + alpha*mu) gives W = mu/(1 + alpha*mu), which
        # downweights the large counts where the extra variance sits.
        W = mu if alpha <= 0 else mu / (1.0 + alpha * mu)
        z = eta + (y - mu) / mu
        XtW = X.T * W
        try:
            XtWX_inv = np.linalg.pinv(XtW @ X)
        except np.linalg.LinAlgError:  # pragma: no cover
            return None
        new_theta = XtWX_inv @ (XtW @ z)
        new_eta = X @ new_theta
        if theta is not None and np.max(np.abs(new_theta - theta)) < tol:
            theta, eta = new_theta, new_eta
            converged = True
            break
        theta, eta = new_theta, new_eta

    if theta is None or XtWX_inv is None:
        return None

    mu = np.clip(np.exp(eta + offset), 1e-12, None)
    df_resid = n - p
    var_mu = mu if alpha <= 0 else mu * (1.0 + alpha * mu)
    pearson = (y - mu) / np.sqrt(var_mu)
    phi = float(np.sum(pearson ** 2) / df_resid) if df_resid > 0 else np.nan
    phi_used = phi
    if dispersion_floor is not None and np.isfinite(phi):
        # phi_hat is itself estimated from n - p residual degrees of freedom, so
        # a value below 1 is usually sampling noise rather than real
        # underdispersion; using it would narrow the interval below Poisson.
        phi_used = max(phi, dispersion_floor)

    if dispersion == "sandwich":
        meat = (X.T * ((y - mu) ** 2)) @ X
        cov = XtWX_inv @ meat @ XtWX_inv * (n / max(df_resid, 1))
    elif dispersion == "quasi":
        cov = XtWX_inv * (phi_used if np.isfinite(phi_used) and phi_used > 0 else 1.0)
    elif dispersion in ("poisson", "nb"):
        # for "nb" the extra variance is already in the weights
        cov = XtWX_inv
    else:
        raise ValueError(f"unknown dispersion: {dispersion!r}")

    with np.errstate(invalid="ignore"):
        dev_terms = np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu), 0.0) - (y - mu)

    return {
        "theta": theta,
        "cov": cov,
        "dispersion": phi,
        "dispersion_used": phi_used,
        "alpha": float(alpha),
        "deviance": float(2.0 * np.sum(dev_terms)),
        "df_resid": int(df_resid),
        "pearson": pearson,
        "mu": mu,
        "converged": bool(converged),
    }


def _fit_nb(X: np.ndarray, y: np.ndarray, E: np.ndarray,
            max_iter: int = 100, tol: float = 1e-10,
            alpha_grid: Optional[np.ndarray] = None) -> Optional[dict]:
    """Negative-binomial fit, with the dispersion alpha profiled out.

    For each candidate alpha the mean model is a GLM with variance
    mu(1 + alpha*mu); the profile log-likelihood selects alpha. With only eight
    years per series alpha is weakly identified, so the grid is coarse and
    bounded, and alpha = 0 (Poisson) is always a candidate.
    """
    if alpha_grid is None:
        alpha_grid = np.concatenate([[0.0], np.logspace(-6, 0.5, 40)])
    best = None
    for a in alpha_grid:
        res = _irls_poisson(X, y, E, dispersion="nb", max_iter=max_iter,
                            tol=tol, alpha=float(a))
        if res is None:
            continue
        ll = _nb_loglik(y, res["mu"], float(a))
        if best is None or ll > best[0]:
            best = (ll, float(a), res)
    if best is None:
        return None
    ll, a, res = best
    res["alpha"] = a
    res["loglik"] = ll
    return res


def _mean_lag1_acf(resid: np.ndarray, year: np.ndarray, age_index) -> float:
    """Mean lag-1 autocorrelation of residuals within age unit, over time."""
    try:
        if age_index is None:
            groups = [np.arange(resid.size)]
        else:
            groups = [np.where(age_index == lv)[0] for lv in np.unique(age_index)]
        vals = []
        for idx in groups:
            if idx.size < 4:
                continue
            order = np.argsort(year[idx])
            r = resid[idx][order]
            r = r - r.mean()
            denom = np.sum(r ** 2)
            if denom <= 0:
                continue
            vals.append(float(np.sum(r[:-1] * r[1:]) / denom))
        return float(np.mean(vals)) if vals else np.nan
    except Exception:  # pragma: no cover
        return np.nan



# ----------------------------------------------------------------------------
# Is a straight line adequate over the window?
#
# The study period is chosen to lie after the trend breaks already identified in
# the literature, so the question is not WHERE a break occurs - eight annual
# points cannot locate one reliably, still less at ICD chapter or code level -
# but whether a single log-linear slope DESCRIBES the window adequately. That is
# testable: fit one extra degree of freedom and compare.
#
#   H0:  log mu = alpha_x + beta (t - tbar)                       (the model used)
#   H1k: log mu = alpha_x + beta (t - tbar) + delta (t - tau)_+    (one knot)
#   H1q: log mu = alpha_x + beta (t - tbar) + delta (t - tbar)^2   (curvature)
#
# tau is FIXED in advance (the midpoint of the window by default), which keeps
# the test a standard one-degree-of-freedom comparison. Under quasi-likelihood
# the statistic is F = (D0 - D1) / (phi * 1) on (1, n - p1) degrees of freedom.
#
# The best-fitting knot is also reported, but WITHOUT a p-value: maximising over
# tau makes the null distribution non-standard (Davies' problem), so that
# quantity is descriptive only.
# ----------------------------------------------------------------------------

def linearity_test(
    deaths: np.ndarray,
    person_years: np.ndarray,
    year: np.ndarray,
    age_index: Optional[np.ndarray] = None,
    dispersion: str = "quasi",
    dispersion_floor: Optional[float] = DISPERSION_FLOOR_DEFAULT,
    knot: Optional[float] = None,
    scan_knots: bool = True,
) -> dict:
    """Test whether one extra degree of freedom in time improves the fit.

    Returns a dict with, for the fixed-knot and quadratic alternatives, the
    deviance drop, the quasi-F statistic and its p-value; plus the best-fitting
    knot and its deviance drop as a descriptive summary.
    """
    y = np.asarray(deaths, dtype=float)
    E = np.asarray(person_years, dtype=float)
    t = np.asarray(year, dtype=float)
    good = np.isfinite(y) & np.isfinite(E) & np.isfinite(t) & (E > 0)
    y, E, t = y[good], E[good], t[good]
    ages = np.asarray(age_index)[good] if age_index is not None else np.zeros(y.size)

    out = {"lin_knot_year": np.nan, "lin_dev_drop_knot": np.nan, "lin_F_knot": np.nan,
           "lin_p_knot": np.nan, "lin_dev_drop_quad": np.nan, "lin_F_quad": np.nan,
           "lin_p_quad": np.nan, "lin_bestknot_year": np.nan,
           "lin_bestknot_dev_drop": np.nan, "lin_note": ""}

    years = np.unique(t)
    if y.size == 0 or years.size < 4:
        out["lin_note"] = "too few years to test"
        return out
    if int(pd.Series(y).groupby(pd.Series(t)).sum().gt(0).sum()) < 4:
        out["lin_note"] = "too few years with deaths to test"
        return out

    tc = t - t.mean()
    if ages is not None and len(np.unique(ages)) > 1:
        levels = np.unique(ages)
        D = np.column_stack([(ages == lv).astype(float) for lv in levels])
    else:
        D = np.ones((y.size, 1))

    def fit(X):
        return _irls_poisson(X, y, E, dispersion=dispersion,
                             dispersion_floor=dispersion_floor)

    X0 = np.column_stack([D, tc])
    base = fit(X0)
    if base is None:
        out["lin_note"] = "base model failed"
        return out
    d0 = base["deviance"]

    def compare(extra, label, tau=np.nan):
        X1 = np.column_stack([D, tc, extra])
        n, p1 = X1.shape
        if n <= p1 + 1:
            return
        alt = fit(X1)
        if alt is None:
            return
        drop = d0 - alt["deviance"]
        phi = alt.get("dispersion_used", alt["dispersion"])
        if not np.isfinite(phi) or phi <= 0:
            phi = 1.0
        df2 = n - p1
        F = (drop / 1.0) / phi
        out[f"lin_dev_drop_{label}"] = float(drop)
        out[f"lin_F_{label}"] = float(F)
        out[f"lin_p_{label}"] = f_sf(F, 1, df2) if df2 > 0 else np.nan
        if label == "knot":
            out["lin_knot_year"] = float(tau)

    tau = float(np.median(years)) if knot is None else float(knot)
    compare(np.maximum(t - tau, 0.0), "knot", tau)
    compare(tc ** 2, "quad")

    # descriptive only: the knot that fits best, with no p-value attached
    if scan_knots and years.size >= 5:
        best, best_tau = -np.inf, np.nan
        for cand in years[1:-1]:
            X1 = np.column_stack([D, tc, np.maximum(t - cand, 0.0)])
            if X1.shape[0] <= X1.shape[1] + 1:
                continue
            alt = fit(X1)
            if alt is None:
                continue
            drop = d0 - alt["deviance"]
            if drop > best:
                best, best_tau = drop, float(cand)
        if np.isfinite(best):
            out["lin_bestknot_year"] = best_tau
            out["lin_bestknot_dev_drop"] = float(best)
    return out


def linearity_summary(est: pd.DataFrame,
                      family_cols: Sequence[str] = ("age_group", "cause_level"),
                      alpha: float = 0.05) -> pd.DataFrame:
    """Per family: how often a straight line is rejected, after FDR control.

    This is the quantity to report: the proportion of series for which one
    additional degree of freedom in time improves the fit beyond chance.
    """
    if "lin_p_knot" not in est.columns:
        raise ValueError("no linearity columns; build with linearity_test=True")
    d = est[est.status.isin(ESTIMABLE_STATUSES) & est.lin_p_knot.notna()].copy()
    family_cols = [c for c in family_cols if c in d.columns]
    grp = d.groupby(family_cols, observed=True) if family_cols else [((), d)]
    rows = []
    for key, g in grp:
        q_knot = benjamini_hochberg(g.lin_p_knot.values)
        q_quad = benjamini_hochberg(g.lin_p_quad.values)
        rows.append(dict(zip(family_cols, key if isinstance(key, tuple) else (key,))) | {
            "n_cells": len(g),
            "pct_knot_rejected": 100.0 * np.nanmean(q_knot < alpha),
            "pct_quadratic_rejected": 100.0 * np.nanmean(q_quad < alpha),
            "pct_either_rejected": 100.0 * np.nanmean(
                (np.nan_to_num(q_knot, nan=1.0) < alpha)
                | (np.nan_to_num(q_quad, nan=1.0) < alpha)),
        })
    return pd.DataFrame(rows)


def ols_slope_on_rate(rate: np.ndarray, year: np.ndarray) -> Tuple[float, float, float]:
    """OLS slope of a rate on year.

    Returned as (slope per year, se, p). Applied to the crude rate it gives the
    descriptive slope in deaths per 100 000 per year; applied to the log
    directly standardised rate it gives the comparison estimate dsr_apc.
    """
    r = np.asarray(rate, dtype=float)
    t = np.asarray(year, dtype=float)
    ok = np.isfinite(r) & np.isfinite(t)
    r, t = r[ok], t[ok]
    n = r.size
    if n < 3:
        return np.nan, np.nan, np.nan
    tc = t - t.mean()
    sxx = float(np.sum(tc ** 2))
    if sxx <= 0:
        return np.nan, np.nan, np.nan
    slope = float(np.sum(tc * (r - r.mean())) / sxx)
    resid = r - (r.mean() + slope * tc)
    s2 = float(np.sum(resid ** 2) / (n - 2))
    se = math.sqrt(s2 / sxx) if s2 > 0 else 0.0
    p = two_sided_p(slope / se, n - 2) if se > 0 else np.nan
    return slope, se, p


# ----------------------------------------------------------------------------
# Age standardisation
# ----------------------------------------------------------------------------

def esp2013_weights(agemin: int, agemax: int) -> Dict[Tuple[int, int], float]:
    """ESP2013 weights restricted to an age range and renormalised to 1.

    Groups overlapping the requested range are apportioned pro rata by the
    number of single years of age included, which is the usual approximation
    when the requested band does not align with the 5-year standard.
    """
    w: Dict[Tuple[int, int], float] = {}
    for (lo, hi), weight in ESP2013.items():
        hi_eff = min(hi, 200)
        overlap_lo, overlap_hi = max(lo, agemin), min(hi_eff, agemax)
        if overlap_hi < overlap_lo:
            continue
        width = hi_eff - lo + 1
        share = (overlap_hi - overlap_lo + 1) / width
        w[(overlap_lo, overlap_hi)] = weight * share
    total = sum(w.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in w.items()}


def directly_standardised_rate(
    deaths_by_age: pd.Series,
    exposure_by_age: pd.Series,
    weights: Dict[Tuple[int, int], float],
    per: float = 100_000.0,
) -> float:
    """Directly standardised rate for one year.

    ``deaths_by_age`` and ``exposure_by_age`` are indexed by single year of
    age; ``weights`` maps closed age intervals to weights summing to 1.
    """
    num = 0.0
    used = 0.0
    for (lo, hi), w in weights.items():
        ages = [a for a in range(lo, min(hi, int(exposure_by_age.index.max())) + 1)]
        if not ages:
            continue
        d = float(deaths_by_age.reindex(ages).fillna(0).sum())
        e = float(exposure_by_age.reindex(ages).fillna(0).sum())
        if e <= 0:
            continue
        num += w * d / e
        used += w
    if used <= 0:
        return np.nan
    return per * num / used


# ----------------------------------------------------------------------------
# HMD readers (all-cause)
# ----------------------------------------------------------------------------

def read_hmd_file(path: Path) -> pd.DataFrame:
    """Read an HMD flat file, tolerating the multi-line preamble."""
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("Year"):
            data = "".join(lines[i:])
            break
    else:
        raise ValueError(f"no header row found in {path}")
    df = pd.read_csv(StringIO(data), sep=r"\s+")
    df.columns = df.columns.str.strip()
    # '110+' -> 110; 'TOT' / 'UNK' rows become NaN and are dropped, so that
    # aggregate rows can never be double counted with the single ages.
    age = df["Age"].astype(str).str.replace(r"\D+", "", regex=True)
    df["Age"] = pd.to_numeric(age.replace("", np.nan), errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df.Age.notna() & df.Year.notna()].copy()
    df["Age"] = df["Age"].astype(int)
    df["Year"] = df["Year"].astype(int)
    for col in ("Female", "Male", "Total"):
        if col in df.columns:
            # HMD writes '.' for unavailable values
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def map_codes_to_countries(mx_dir: Path) -> Dict[str, str]:
    """HMD country code -> country name, from the Mx_5x1 file headers."""
    out: Dict[str, str] = {}
    for path in sorted(Path(mx_dir).glob("*.Mx_5x1.txt")):
        code = path.name.split(".")[0]
        with open(path, "r", encoding="utf-8") as fh:
            out[code] = fh.readline().split(",")[0].strip()
    return out


def _sex_column(sex: str) -> str:
    return {"total": "Total", "male": "Male", "female": "Female"}[sex]


def load_hmd_country(
    code: str,
    yearmin: int,
    yearmax: int,
    sex: str = "total",
    exposures_dir: Optional[Path] = None,
    deaths_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Single-year-of-age deaths and exposures for one country.

    Returns a frame with columns ``Year, Age, deaths, person_years``. Accepts a
    list of HMD codes (summed), for populations such as New Zealand that are
    split into sub-populations in the HMD.
    """
    exposures_dir = Path(exposures_dir or CONFIG["hmd_exposures_dir"])
    deaths_dir = Path(deaths_dir or CONFIG["hmd_deaths_lexis_dir"])
    codes = code if isinstance(code, (list, tuple)) else [code]
    col = _sex_column(sex)

    frames = []
    for c in codes:
        exp_path = exposures_dir / f"{c}.Exposures_1x1.txt"
        dth_path = deaths_dir / f"{c}.Deaths_lexis.txt"
        if not exp_path.exists():
            raise FileNotFoundError(exp_path)
        if not dth_path.exists():
            raise FileNotFoundError(dth_path)
        exp = read_hmd_file(exp_path)
        dth = read_hmd_file(dth_path)
        exp = exp[(exp.Year >= yearmin) & (exp.Year <= yearmax)]
        dth = dth[(dth.Year >= yearmin) & (dth.Year <= yearmax)]
        # Deaths_lexis holds two Lexis triangles per Year x Age; summing over
        # them within Year x Age gives period deaths.
        dth = dth.groupby(["Year", "Age"], as_index=False)[col].sum()
        exp = exp.groupby(["Year", "Age"], as_index=False)[col].sum()
        merged = exp.merge(dth, on=["Year", "Age"], how="left", suffixes=("_exp", "_dth"))
        merged.columns = ["Year", "Age", "person_years", "deaths"]
        frames.append(merged)

    out = pd.concat(frames, ignore_index=True)
    out = out.groupby(["Year", "Age"], as_index=False)[["person_years", "deaths"]].sum()
    return out


# ----------------------------------------------------------------------------
# All-cause estimates table
# ----------------------------------------------------------------------------

def build_all_cause_estimates(
    countries: Optional[Iterable[str]] = None,
    age_groups: Sequence[str] = tuple(PAPER_AGE_GROUPS),
    yearmin: Optional[int] = None,
    yearmax: Optional[int] = None,
    sexes: Sequence[str] = ("total",),
    dispersion: str = "quasi",
    linearity: bool = False,
    exclude: Sequence[str] = tuple(EXCLUDED_COUNTRIES),
    verbose: bool = True,
) -> pd.DataFrame:
    """Estimates table for all-cause mortality, by country, age group and sex.

    One row per (country, sex, age group), with the age-adjusted APC as the
    primary estimate and the crude and ESP2013-standardised alternatives
    alongside it.
    """
    yearmin = CONFIG["yearmin"] if yearmin is None else yearmin
    yearmax = CONFIG["yearmax"] if yearmax is None else yearmax
    code_to_country = map_codes_to_countries(CONFIG["hmd_mx_dir"])

    # group HMD codes by country name so sub-populations are summed
    by_country: Dict[str, List[str]] = {}
    for code, name in code_to_country.items():
        by_country.setdefault(name, []).append(code)
    if "New Zealand" not in by_country:
        nz = [c for name, codes in by_country.items() if "New Zealand" in name for c in codes]
        if nz:
            by_country["New Zealand"] = nz

    wanted = set(countries) if countries is not None else set(by_country) - set(exclude)

    rows: List[dict] = []
    for country in sorted(wanted):
        codes = by_country.get(country)
        if not codes:
            rows.append(_failed_row(country, reason="country not found in HMD Mx_5x1"))
            continue
        for sex in sexes:
            try:
                data = load_hmd_country(codes, yearmin, yearmax, sex=sex)
            except Exception as exc:
                rows.append(_failed_row(country, sex=sex, reason=f"load failed: {exc}"))
                continue
            for age_group in age_groups:
                agemin, agemax = AGE_BINS[age_group]
                rows.append(
                    _all_cause_row(
                        country, sex, age_group, agemin, agemax, data,
                        yearmin, yearmax, dispersion, linearity,
                    )
                )
        if verbose:
            print(f"  {country}: done", flush=True)

    return pd.DataFrame(rows)


def _failed_row(country: str, sex: str = "total", reason: str = "", **extra) -> dict:
    """A row that records why a cell could not be estimated.

    Failures are recorded, never silently turned into a 'neutral' trend.
    """
    row = {
        "country": country,
        "sex": sex,
        "age_group": np.nan,
        "cause_level": np.nan,
        "cause": np.nan,
        "cause_label": np.nan,
        "status": "failed",
        "status_detail": reason,
    }
    row.update(extra)
    return row


def _all_cause_row(
    country: str,
    sex: str,
    age_group: str,
    agemin: int,
    agemax: int,
    data: pd.DataFrame,
    yearmin: int,
    yearmax: int,
    dispersion: str,
    linearity: bool = False,
) -> dict:
    sub = data[(data.Age >= agemin) & (data.Age <= agemax)].copy()
    years = sorted(sub.Year.unique())
    n_years = len(years)

    row = {
        "country": country,
        "sex": sex,
        "age_group": age_group,
        "cause_level": "all-cause",
        "cause": "all",
        "cause_label": "All causes",
        "year_min": int(min(years)) if years else np.nan,
        "year_max": int(max(years)) if years else np.nan,
        "n_years": n_years,
        "status": "ok",
        "status_detail": "",
    }

    if n_years < CONFIG["min_years"]:
        row.update(status="failed", status_detail=f"only {n_years} usable years")
        return row

    totals = sub.groupby("Year")[["deaths", "person_years"]].sum()
    row["total_deaths"] = float(totals.deaths.sum())
    row["person_years"] = float(totals.person_years.sum())
    row["mean_annual_deaths"] = float(totals.deaths.mean())

    crude = 1e5 * totals.deaths / totals.person_years
    row["crude_rate_first"] = float(crude.iloc[0])
    row["crude_rate_last"] = float(crude.iloc[-1])

    # ---- primary: age-adjusted log-linear Poisson trend
    fit = fit_poisson_trend(
        sub.deaths.values, sub.person_years.values, sub.Year.values,
        age_index=sub.Age.values, dispersion=dispersion,
    )
    _write_fit(row, fit, prefix="")

    if linearity:
        row.update(linearity_test(sub.deaths.values, sub.person_years.values,
                                  sub.Year.values, age_index=sub.Age.values,
                                  dispersion=dispersion))

    # ---- crude (no age term), to show the size of the age-composition effect
    crude_fit = fit_poisson_trend(
        totals.deaths.values, totals.person_years.values,
        np.asarray(totals.index, dtype=float), age_index=None, dispersion=dispersion,
    )
    _write_fit(row, crude_fit, prefix="crude_")
    if np.isfinite(row.get("apc", np.nan)) and np.isfinite(row.get("crude_apc", np.nan)):
        row["apc_minus_crude_apc"] = row["apc"] - row["crude_apc"]

    # ---- directly standardised rates (ESP2013) and their log-linear trend
    weights = esp2013_weights(agemin, agemax)
    dsr = {}
    for year in years:
        yr = sub[sub.Year == year]
        dsr[year] = directly_standardised_rate(
            yr.set_index("Age").deaths, yr.set_index("Age").person_years, weights
        )
    dsr_series = pd.Series(dsr).sort_index()
    row["dsr_esp2013_first"] = float(dsr_series.iloc[0])
    row["dsr_esp2013_last"] = float(dsr_series.iloc[-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        log_dsr = np.log(dsr_series.values)
    dsr_slope, dsr_se, dsr_p = ols_slope_on_rate(log_dsr, dsr_series.index.values)
    row["dsr_apc"] = 100.0 * (math.exp(dsr_slope) - 1.0) if np.isfinite(dsr_slope) else np.nan
    row["dsr_apc_p"] = dsr_p

    # ---- OLS slope of the crude rate, deaths per 100 000 per year (descriptive)
    legacy_slope, legacy_se, legacy_p = ols_slope_on_rate(crude.values, crude.index.values)
    row["legacy_ols_slope_per_100k"] = legacy_slope
    row["legacy_ols_se"] = legacy_se
    row["legacy_ols_p"] = legacy_p

    return row


def _write_fit(row: dict, fit: PoissonFit, prefix: str = "") -> None:
    row[f"{prefix}n_years_with_deaths"] = fit.n_years_with_deaths
    row[f"{prefix}fit_note"] = fit.note
    if not np.isfinite(fit.beta):
        # only the primary fit decides the row's status; the secondary crude fit
        # failing is recorded but does not invalidate the age-adjusted estimate
        if prefix == "":
            row["status"] = "not estimable"
            row["status_detail"] = (
                (row.get("status_detail") or "") + (fit.note or "fit failed") + "; "
            )
        else:
            row["status_detail"] = (
                (row.get("status_detail") or "") + f"{prefix}fit failed; "
            )
        return
    if fit.note and prefix == "":
        row["status"] = "unstable"
        row["status_detail"] = (row.get("status_detail") or "") + fit.note + "; "
    lo, hi = fit.ci()
    row[f"{prefix}apc"] = fit.apc
    row[f"{prefix}apc_lo"] = lo
    row[f"{prefix}apc_hi"] = hi
    row[f"{prefix}beta"] = fit.beta
    row[f"{prefix}se"] = fit.se
    row[f"{prefix}p"] = fit.p_value
    row[f"{prefix}mde80"] = fit.mde80
    row[f"{prefix}dispersion"] = fit.dispersion
    row[f"{prefix}dispersion_used"] = fit.dispersion_used
    row[f"{prefix}nb_alpha"] = fit.nb_alpha
    row[f"{prefix}df_resid"] = fit.df_resid
    row[f"{prefix}n_obs"] = fit.n_obs
    row[f"{prefix}lag1_resid_acf"] = fit.lag1_resid_acf
    row[f"{prefix}converged"] = fit.converged


# ----------------------------------------------------------------------------
# WHO cause-specific estimates table
# ----------------------------------------------------------------------------

def icd10_chapter(code: str) -> Tuple[Optional[str], Optional[str]]:
    """Map an ICD-10 code (e.g. ``'X420'``) to (chapter numeral, label)."""
    if not isinstance(code, str) or len(code) < 3:
        return None, None
    letter = code[0].upper()
    try:
        number = int(code[1:3])
    except ValueError:
        return None, None
    for numeral, label, (lo_l, lo_n), (hi_l, hi_n) in ICD10_CHAPTERS:
        if lo_l == hi_l:
            if letter == lo_l and lo_n <= number <= hi_n:
                return numeral, label
        else:
            if letter == lo_l and number >= lo_n:
                return numeral, label
            if letter == hi_l and number <= hi_n:
                return numeral, label
            if lo_l < letter < hi_l:
                return numeral, label
    return None, None


#: Default ICD-10 detail for code-level analysis. Three characters is the
#: standard category level and is strongly recommended for trend work: several
#: countries migrate between 4-character and 3-character coding of the same
#: cause within a study period (for example X70.9 -> X70 for intentional
#: self-harm by hanging, X42.9 -> X42 for drug poisoning). Analysed separately,
#: the two halves produce a spurious rise of roughly +60% per year in one code
#: mirrored by a fall of about -35% per year in the other; aggregated to the
#: 3-character category, the artefact cancels.
CODE_DIGITS_DEFAULT = 3


def truncate_icd(code: str, digits: int = CODE_DIGITS_DEFAULT) -> str:
    """Reduce an ICD-10 code to its first ``digits`` characters."""
    c = str(code).strip()
    return c[:digits] if digits and len(c) > digits else c


def load_who_country_codes(path: Optional[Path] = None) -> pd.DataFrame:
    df = pd.read_csv(path or CONFIG["who_country_codes"])
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_who_deaths(
    yearmin: Optional[int] = None,
    yearmax: Optional[int] = None,
    parts: Optional[Sequence[Path]] = None,
    sex: str = "total",
    who_country_codes: Optional[Sequence[int]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Long WHO deaths table: ``who_country, Year, Cause, band, deaths``.

    Only national, all-year records are kept: rows with a sub-national
    ``Admin1`` or a ``SubDiv`` marker are dropped rather than double-counted,
    and the 'age unspecified' column is carried separately as a data-quality
    diagnostic.
    """
    yearmin = CONFIG["yearmin"] if yearmin is None else yearmin
    yearmax = CONFIG["yearmax"] if yearmax is None else yearmax
    parts = parts or CONFIG["who_parts"]

    keep_cols = list(WHO_BANDS) + ["Deaths1", "Deaths26"]
    frames = []
    for path in parts:
        path = Path(path)
        if not path.exists():
            warnings.warn(f"WHO part not found, skipped: {path}")
            continue
        if verbose:
            print(f"reading {path.name} ...", flush=True)
        df = pd.read_csv(path, low_memory=False)
        df = df[(df.Year >= yearmin) & (df.Year <= yearmax)]
        if df.empty:
            continue
        if "Admin1" in df.columns:
            df = df[df.Admin1.isna() | (df.Admin1.astype(str).str.strip() == "")]
        if "SubDiv" in df.columns:
            df = df[df.SubDiv.isna() | (df.SubDiv.astype(str).str.strip() == "")]
        if sex != "total" and "Sex" in df.columns:
            df = df[df.Sex == (1 if sex == "male" else 2)]
        df = df[df.Cause.astype(str) != "AAA"]
        if who_country_codes is not None:
            df = df[df.Country.isin(list(who_country_codes))]
        present = [c for c in keep_cols if c in df.columns]
        df = df[["Country", "Year", "Cause", "Frmat"] + present] if "Frmat" in df.columns \
            else df[["Country", "Year", "Cause"] + present]
        frames.append(df)

    if not frames:
        raise FileNotFoundError("no WHO part files could be read")

    raw = pd.concat(frames, ignore_index=True)
    for col in keep_cols:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0.0)

    diag = (
        raw.groupby(["Country", "Year"])[[c for c in ("Deaths1", "Deaths26") if c in raw.columns]]
        .sum()
        .reset_index()
    )

    band_cols = [c for c in WHO_BANDS if c in raw.columns]
    long = raw.melt(
        id_vars=["Country", "Year", "Cause"],
        value_vars=band_cols,
        var_name="band",
        value_name="deaths",
    )
    long = long[long.deaths > 0]
    long = long.groupby(["Country", "Year", "Cause", "band"], as_index=False).deaths.sum()
    long.rename(columns={"Country": "who_country"}, inplace=True)
    long.attrs["diagnostics"] = diag
    return long


def hmd_exposure_by_who_band(
    hmd_single_year: pd.DataFrame, bands: Sequence[str]
) -> pd.DataFrame:
    """Aggregate single-year HMD exposures into the WHO age bands."""
    out = []
    for band in bands:
        lo, hi = WHO_BANDS[band]
        sub = hmd_single_year[(hmd_single_year.Age >= lo) & (hmd_single_year.Age <= hi)]
        agg = sub.groupby("Year", as_index=False).person_years.sum()
        agg["band"] = band
        out.append(agg)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["Year", "person_years", "band"]
    )


def bands_for_age_group(age_group: str) -> List[str]:
    """WHO band columns fully contained in an age group."""
    agemin, agemax = AGE_BINS[age_group]
    return [b for b, (lo, hi) in WHO_BANDS.items() if lo >= agemin and hi <= agemax]


def build_cause_estimates(
    level: str = "chapter",
    countries: Optional[Iterable[str]] = None,
    age_groups: Sequence[str] = tuple(PAPER_AGE_GROUPS),
    yearmin: Optional[int] = None,
    yearmax: Optional[int] = None,
    sex: str = "total",
    dispersion: str = "quasi",
    min_deaths: int = 50,
    top_n_codes: Optional[int] = None,
    code_digits: int = CODE_DIGITS_DEFAULT,
    linearity: bool = False,
    exclude: Sequence[str] = tuple(EXCLUDED_COUNTRIES),
    who_long: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Estimates table for cause-specific mortality.

    Parameters
    ----------
    level
        ``"chapter"`` for ICD-10 chapters, ``"code"`` for individual codes.
    min_deaths
        Series with fewer cumulative deaths over the period are still returned
        but marked ``status="sparse"``; they are reported as *inconclusive*
        rather than dropped, which is what lets small countries stay in the
        figures honestly.
    top_n_codes
        If set, restrict code-level analysis to the N codes with most deaths in
        each country x age group.
    """
    yearmin = CONFIG["yearmin"] if yearmin is None else yearmin
    yearmax = CONFIG["yearmax"] if yearmax is None else yearmax

    code_to_country = map_codes_to_countries(CONFIG["hmd_mx_dir"])
    hmd_by_country: Dict[str, List[str]] = {}
    for code, name in code_to_country.items():
        hmd_by_country.setdefault(name, []).append(code)
    nz = [c for name, codes in hmd_by_country.items() if "New Zealand" in name for c in codes]
    if nz:
        hmd_by_country["New Zealand"] = nz

    who_codes = load_who_country_codes()
    name_equiv = {
        "The United States of America": "United States",
        "United States of America": "United States",
        "Russian Federation": "Russia",
        # HMD and WHO spell these differently.
        "Czech Republic": "Czechia",
        "Hong Kong SAR": "Hong Kong",
    }
    who_name_to_code = {
        name_equiv.get(str(r["name"]).strip(), str(r["name"]).strip()): r["country"]
        for _, r in who_codes.iterrows()
    }

    wanted = set(countries) if countries is not None else (
        {name_equiv.get(n, n) for n in hmd_by_country} & set(who_name_to_code)
    ) - set(exclude)

    if who_long is None:
        who_long = load_who_deaths(
            yearmin, yearmax, sex=sex,
            who_country_codes=[who_name_to_code[c] for c in wanted if c in who_name_to_code],
            verbose=verbose,
        )

    if level == "chapter":
        chapters = who_long.Cause.astype(str).map(lambda c: icd10_chapter(c)[0])
        who_long = who_long.assign(cause_key=chapters)
        who_long = who_long[who_long.cause_key.notna()]
        labels = {numeral: label for numeral, label, _, _ in ICD10_CHAPTERS}
    elif level == "code":
        # aggregate to the 3-character category by default; see CODE_DIGITS_DEFAULT
        who_long = who_long.assign(
            cause_key=who_long.Cause.astype(str).map(lambda c: truncate_icd(c, code_digits)))
        who_long = who_long.groupby(
            ["who_country", "Year", "band", "cause_key"], as_index=False).deaths.sum()
        if verbose:
            print(f"codes aggregated to {code_digits} characters: "
                  f"{who_long.cause_key.nunique()} categories")
        labels = {}
    else:
        raise ValueError("level must be 'chapter' or 'code'")

    rows: List[dict] = []
    for country in sorted(wanted):
        who_code = who_name_to_code.get(country)
        hmd_codes = hmd_by_country.get(country) or hmd_by_country.get(
            next((n for n in hmd_by_country if name_equiv.get(n, n) == country), ""), None
        )
        if who_code is None or not hmd_codes:
            rows.append(_failed_row(country, reason="no HMD/WHO correspondence"))
            continue
        try:
            hmd = load_hmd_country(hmd_codes, yearmin, yearmax, sex=sex)
        except Exception as exc:
            rows.append(_failed_row(country, reason=f"HMD load failed: {exc}"))
            continue

        cdeaths = who_long[who_long.who_country == who_code]
        if cdeaths.empty:
            rows.append(_failed_row(country, reason="no WHO records in period"))
            continue

        who_years = set(cdeaths.Year.unique())
        hmd_years = set(hmd.Year.unique())
        avail = sorted(who_years & hmd_years)
        if len(avail) < CONFIG["min_years"]:
            rows.append(
                _failed_row(country, reason=f"only {len(avail)} years in both sources")
            )
            continue

        for age_group in age_groups:
            bands = bands_for_age_group(age_group)
            if not bands:
                continue
            exposure = hmd_exposure_by_who_band(hmd, bands)
            exposure = exposure[exposure.Year.isin(avail)]

            sub = cdeaths[cdeaths.band.isin(bands) & cdeaths.Year.isin(avail)]
            if sub.empty:
                continue

            totals = sub.groupby("cause_key").deaths.sum().sort_values(ascending=False)
            keys = list(totals.index)
            if top_n_codes is not None and level == "code":
                keys = keys[:top_n_codes]

            all_deaths_in_group = float(totals.sum())

            for key in keys:
                rows.append(
                    _cause_row(
                        country, sex, age_group, level, key, labels.get(key, key),
                        sub[sub.cause_key == key], exposure, avail, dispersion,
                        min_deaths, all_deaths_in_group, linearity,
                    )
                )
        if verbose:
            print(f"  {country}: {len(rows)} rows so far", flush=True)

    return pd.DataFrame(rows)


def _cause_row(
    country: str,
    sex: str,
    age_group: str,
    level: str,
    key: str,
    label: str,
    deaths_sub: pd.DataFrame,
    exposure: pd.DataFrame,
    years: Sequence[int],
    dispersion: str,
    min_deaths: int,
    all_deaths_in_group: float,
    linearity: bool = False,
) -> dict:
    """One estimates row for a cause within a country and age group.

    Cells absent from the WHO extract are genuine zero-death cells (the extract
    lists only causes with deaths), so they are filled with 0 deaths against
    the observed person-years rather than being dropped - dropping them biases
    the trend upwards in small countries.
    """
    grid = (
        pd.MultiIndex.from_product(
            [sorted(years), sorted(deaths_sub.band.unique())], names=["Year", "band"]
        )
        .to_frame(index=False)
    )
    d = grid.merge(
        deaths_sub.groupby(["Year", "band"], as_index=False).deaths.sum(),
        on=["Year", "band"], how="left",
    )
    d["deaths"] = d.deaths.fillna(0.0)
    d = d.merge(exposure, on=["Year", "band"], how="left")
    d = d[d.person_years.notna() & (d.person_years > 0)]

    row = {
        "country": country,
        "sex": sex,
        "age_group": age_group,
        "cause_level": level,
        "cause": key,
        "cause_label": label,
        "year_min": int(min(years)),
        "year_max": int(max(years)),
        "n_years": int(len(years)),
        "status": "ok",
        "status_detail": "",
    }
    if d.empty:
        row.update(status="failed", status_detail="no exposure matched")
        return row

    row["total_deaths"] = float(d.deaths.sum())
    row["person_years"] = float(d.person_years.sum())
    row["mean_annual_deaths"] = float(d.groupby("Year").deaths.sum().mean())
    row["n_zero_years"] = int((d.groupby("Year").deaths.sum() == 0).sum())
    row["share_of_group_deaths_pct"] = (
        100.0 * row["total_deaths"] / all_deaths_in_group if all_deaths_in_group > 0 else np.nan
    )

    totals = d.groupby("Year")[["deaths", "person_years"]].sum()
    crude = 1e5 * totals.deaths / totals.person_years
    row["crude_rate_first"] = float(crude.iloc[0])
    row["crude_rate_last"] = float(crude.iloc[-1])
    row["abs_change_per_100k_per_year"] = ols_slope_on_rate(crude.values, crude.index.values)[0]

    if row["total_deaths"] < min_deaths:
        row.update(status="sparse",
                   status_detail=f"{row['total_deaths']:.0f} deaths < min_deaths={min_deaths}")

    band_codes = pd.Categorical(d.band).codes
    fit = fit_poisson_trend(
        d.deaths.values, d.person_years.values, d.Year.values,
        age_index=band_codes, dispersion=dispersion,
    )
    _write_fit(row, fit, prefix="")

    if linearity:
        row.update(linearity_test(d.deaths.values, d.person_years.values, d.Year.values,
                                  age_index=band_codes, dispersion=dispersion))

    legacy_slope, legacy_se, legacy_p = ols_slope_on_rate(crude.values, crude.index.values)
    row["legacy_ols_slope_per_100k"] = legacy_slope
    row["legacy_ols_se"] = legacy_se
    row["legacy_ols_p"] = legacy_p
    return row


# ----------------------------------------------------------------------------
# Finalisation: multiplicity, power, classification
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Sex-specific trends: a single model with a sex x year interaction
#
# This model is fitted separately from the combined-sex model above, and each
# has its own output tables.
#
#     log mu_{x,s,t} = alpha_{x,s} + beta (t - tbar) + gamma s (t - tbar)
#                      + log E_{x,s,t},          s = 1 for males, 0 for females
#
# The intercepts are fully crossed in age and sex, so the two sexes may have
# completely different age patterns; only the trends are compared. Estimands:
#
#     female trend          beta
#     male trend            beta + gamma
#     sex-averaged trend    beta + gamma/2   (a 50:50 sex-standardised trend)
#     difference            gamma            (log ratio of annual rate ratios)
#
# Testing gamma directly is what avoids the common error of declaring a sex
# difference because one sex reached significance and the other did not.
# ----------------------------------------------------------------------------

@dataclass
class SexPoissonFit:
    """Result of one age-by-sex trend fit."""

    beta: float              # female (reference) log trend per year
    gamma: float             # male minus female log trend per year
    cov: Optional[np.ndarray]  # covariance of (beta, gamma)
    dispersion: float
    df_resid: int
    n_obs: int
    converged: bool
    deviance: float
    lag1_resid_acf: float
    dispersion_used: float = np.nan
    nb_alpha: float = np.nan
    deaths_male: float = 0.0
    deaths_female: float = 0.0
    years_with_deaths_male: int = 0
    years_with_deaths_female: int = 0
    note: str = ""

    # -- contrasts ----------------------------------------------------------
    def _contrast(self, w_beta: float, w_gamma: float) -> Tuple[float, float]:
        """Point estimate and standard error of w_beta*beta + w_gamma*gamma."""
        if self.cov is None or not np.isfinite(self.beta):
            return np.nan, np.nan
        c = np.array([w_beta, w_gamma], dtype=float)
        est = float(c @ np.array([self.beta, self.gamma]))
        var = float(c @ self.cov @ c)
        return est, float(np.sqrt(max(var, 0.0)))

    def trend(self, which: str) -> Tuple[float, float, float, float, float]:
        """(APC, lo, hi, p, MDE80) for 'female', 'male', 'average' or 'difference'.

        For 'difference' the quantity returned in place of an APC is the ratio
        of annual rate ratios expressed as a percentage,
        100*(exp(gamma) - 1): the extra annual percentage change in men
        relative to women.
        """
        weights = {"female": (1.0, 0.0), "male": (1.0, 1.0),
                   "average": (1.0, 0.5), "difference": (0.0, 1.0)}
        if which not in weights:
            raise ValueError(f"unknown contrast: {which!r}")
        est, se = self._contrast(*weights[which])
        if not np.isfinite(est) or not np.isfinite(se):
            return (np.nan,) * 5
        crit = t_quantile(self.df_resid)
        return (
            pct_change_from_log(est),
            pct_change_from_log(est - crit * se),
            pct_change_from_log(est + crit * se),
            two_sided_p(est / se, self.df_resid) if se > 0 else np.nan,
            pct_change_from_log(MDE_MULT * se),
        )


def fit_poisson_sex_trend(
    deaths: np.ndarray,
    person_years: np.ndarray,
    year: np.ndarray,
    sex: np.ndarray,
    age_index: Optional[np.ndarray] = None,
    dispersion: str = "quasi",
    max_iter: int = 100,
    tol: float = 1e-10,
    dispersion_floor: Optional[float] = DISPERSION_FLOOR_DEFAULT,
) -> SexPoissonFit:
    """Fit the age-by-sex trend model with a sex x year interaction.

    ``sex`` holds 1 for males and 0 for females. Rows are
    (age unit x sex x year) cells.
    """
    y = np.asarray(deaths, dtype=float)
    E = np.asarray(person_years, dtype=float)
    t = np.asarray(year, dtype=float)
    s = np.asarray(sex, dtype=float)
    if not (y.shape == E.shape == t.shape == s.shape):
        raise ValueError("deaths, person_years, year and sex must have equal length")

    good = np.isfinite(y) & np.isfinite(E) & np.isfinite(t) & np.isfinite(s) & (E > 0)
    y, E, t, s = y[good], E[good], t[good], s[good]
    ages = np.asarray(age_index)[good] if age_index is not None else np.zeros(y.size)

    def _blank(note, **kw):
        return SexPoissonFit(np.nan, np.nan, None, np.nan, 0, int(y.size), False,
                             np.nan, np.nan, note=note, **kw)

    if y.size == 0:
        return _blank("no usable observations")

    dm, df_ = float(y[s == 1].sum()), float(y[s == 0].sum())
    yrs_m = int(pd.Series(y[s == 1]).groupby(pd.Series(t[s == 1])).sum().gt(0).sum())
    yrs_f = int(pd.Series(y[s == 0]).groupby(pd.Series(t[s == 0])).sum().gt(0).sum())
    common = dict(deaths_male=dm, deaths_female=df_,
                  years_with_deaths_male=yrs_m, years_with_deaths_female=yrs_f)

    # the interaction is not identifiable unless both sexes have a trend to fit
    if min(yrs_m, yrs_f) < 3:
        return _blank(f"only {yrs_m} male and {yrs_f} female years have deaths", **common)
    if (s == 1).sum() == 0 or (s == 0).sum() == 0:
        return _blank("only one sex present", **common)

    tc = t - t.mean()

    # fully crossed age-by-sex cell means
    keys = np.array([f"{int(a)}|{int(v)}" for a, v in zip(ages, s)])
    levels = np.unique(keys)
    D = np.column_stack([(keys == lv).astype(float) for lv in levels])
    X = np.column_stack([D, tc, s * tc])
    n, p = X.shape
    if n <= p:
        return _blank("fewer observations than parameters", **common)

    if dispersion == "nb":
        res = _fit_nb(X, y, E, max_iter=max_iter, tol=tol)
    else:
        res = _irls_poisson(X, y, E, dispersion=dispersion, max_iter=max_iter,
                            tol=tol, dispersion_floor=dispersion_floor)
    if res is None:
        return _blank("IRLS failed", **common)

    theta, cov_full = res["theta"], res["cov"]
    phi_used = res.get("dispersion_used", res["dispersion"])
    nb_alpha = res.get("alpha", np.nan)
    beta, gamma = float(theta[-2]), float(theta[-1])
    cov = np.asarray(cov_full)[-2:, -2:]

    note = ""
    if not res["converged"]:
        note = "IRLS did not converge"
    elif (abs(beta) > UNSTABLE_BETA or abs(gamma) > UNSTABLE_BETA
          or np.sqrt(max(cov[0, 0], 0)) > UNSTABLE_BETA
          or np.sqrt(max(cov[1, 1], 0)) > UNSTABLE_BETA):
        note = "unstable fit (near-separable sparse series)"

    return SexPoissonFit(
        beta=beta, gamma=gamma, cov=cov,
        dispersion=res["dispersion"], df_resid=res["df_resid"], n_obs=int(n),
        converged=res["converged"], deviance=res["deviance"],
        lag1_resid_acf=_mean_lag1_acf(res["pearson"], t, keys),
        dispersion_used=phi_used, nb_alpha=nb_alpha,
        note=note, **common,
    )


def _write_sex_fit(row: dict, fit: SexPoissonFit) -> None:
    """Write every contrast of a sex fit into an output row."""
    row["deaths_male"] = fit.deaths_male
    row["deaths_female"] = fit.deaths_female
    row["n_years_with_deaths_male"] = fit.years_with_deaths_male
    row["n_years_with_deaths_female"] = fit.years_with_deaths_female
    row["fit_note"] = fit.note

    if not np.isfinite(fit.beta):
        row["status"] = "not estimable"
        row["status_detail"] = (row.get("status_detail") or "") + (fit.note or "fit failed") + "; "
        return
    if fit.note:
        row["status"] = "unstable"
        row["status_detail"] = (row.get("status_detail") or "") + fit.note + "; "

    for which, prefix in (("female", "apc_female"), ("male", "apc_male"),
                          ("average", "apc_sexavg"), ("difference", "apc_diff")):
        apc, lo, hi, p, mde = fit.trend(which)
        row[prefix] = apc
        row[f"{prefix}_lo"] = lo
        row[f"{prefix}_hi"] = hi
        row[f"{prefix}_p"] = p
        row[f"{prefix}_mde80"] = mde

    row["gamma"] = fit.gamma
    row["ratio_of_rate_ratios"] = float(np.exp(fit.gamma)) if abs(fit.gamma) < 700 else np.inf
    row["p"] = row["apc_diff_p"]          # the family tested by finalise_sex
    row["dispersion"] = fit.dispersion
    row["dispersion_used"] = fit.dispersion_used
    row["nb_alpha"] = fit.nb_alpha
    row["df_resid"] = fit.df_resid
    row["n_obs"] = fit.n_obs
    row["lag1_resid_acf"] = fit.lag1_resid_acf
    row["converged"] = fit.converged


def load_who_deaths_by_sex(
    yearmin: Optional[int] = None,
    yearmax: Optional[int] = None,
    parts: Optional[Sequence[Path]] = None,
    who_country_codes: Optional[Sequence[int]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Long WHO deaths table retaining sex: adds a ``sex`` column (1 male, 0 female).

    Records coded sex 9 (unspecified) cannot be allocated and are dropped; the
    number dropped is returned in ``.attrs['n_sex_unspecified']`` so that it can
    be reported.
    """
    yearmin = CONFIG["yearmin"] if yearmin is None else yearmin
    yearmax = CONFIG["yearmax"] if yearmax is None else yearmax
    parts = parts or CONFIG["who_parts"]
    band_cols = list(WHO_BANDS)

    frames, n_unspecified = [], 0
    for path in parts:
        path = Path(path)
        if not path.exists():
            warnings.warn(f"WHO part not found, skipped: {path}")
            continue
        if verbose:
            print(f"reading {path.name} ...", flush=True)
        df = pd.read_csv(path, low_memory=False)
        df = df[(df.Year >= yearmin) & (df.Year <= yearmax)]
        if df.empty:
            continue
        if "Admin1" in df.columns:
            df = df[df.Admin1.isna() | (df.Admin1.astype(str).str.strip() == "")]
        if "SubDiv" in df.columns:
            df = df[df.SubDiv.isna() | (df.SubDiv.astype(str).str.strip() == "")]
        df = df[df.Cause.astype(str) != "AAA"]
        if who_country_codes is not None:
            df = df[df.Country.isin(list(who_country_codes))]
        if "Sex" not in df.columns:
            raise ValueError(f"{path.name} has no Sex column")
        present = [c for c in band_cols if c in df.columns]
        for c in present:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        n_unspecified += float(df.loc[~df.Sex.isin([1, 2]), present].sum().sum())
        df = df[df.Sex.isin([1, 2])]
        frames.append(df[["Country", "Year", "Cause", "Sex"] + present])

    if not frames:
        raise FileNotFoundError("no WHO part files could be read")

    raw = pd.concat(frames, ignore_index=True)
    long = raw.melt(id_vars=["Country", "Year", "Cause", "Sex"],
                    value_vars=[c for c in band_cols if c in raw.columns],
                    var_name="band", value_name="deaths")
    long = long[long.deaths > 0]
    long["sex"] = (long.Sex == 1).astype(int)       # 1 = male, 0 = female
    long = long.groupby(["Country", "Year", "Cause", "band", "sex"],
                        as_index=False).deaths.sum()
    long.rename(columns={"Country": "who_country"}, inplace=True)
    long.attrs["n_sex_unspecified"] = n_unspecified
    if verbose and n_unspecified:
        print(f"  {n_unspecified:,.0f} deaths of unspecified sex dropped")
    return long


def _hmd_by_sex(codes, yearmin: int, yearmax: int) -> pd.DataFrame:
    """Single-year-of-age deaths and exposures for both sexes, long format."""
    parts = []
    for sex_name, s in (("male", 1), ("female", 0)):
        d = load_hmd_country(codes, yearmin, yearmax, sex=sex_name)
        d["sex"] = s
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def build_sex_estimates(
    level: str = "all-cause",
    countries: Optional[Iterable[str]] = None,
    age_groups: Sequence[str] = tuple(PAPER_AGE_GROUPS),
    yearmin: Optional[int] = None,
    yearmax: Optional[int] = None,
    dispersion: str = "quasi",
    min_deaths: int = 50,
    top_n_codes: Optional[int] = None,
    code_digits: int = CODE_DIGITS_DEFAULT,
    exclude: Sequence[str] = tuple(EXCLUDED_COUNTRIES),
    verbose: bool = True,
) -> pd.DataFrame:
    """Estimates table from the age-by-sex model, one row per analysed cell.

    Parameters
    ----------
    level
        ``"all-cause"``, ``"chapter"`` or ``"code"``.
    top_n_codes
        At code level, restrict to the N causes with most deaths in each
        country x age group. Strongly recommended: the sex interaction needs
        roughly four times as many deaths as a main-effect trend of the same
        size, so most individual codes cannot support it.

    Returns
    -------
    DataFrame with, for each cell, the female, male and sex-averaged annual
    percentage changes with 95% intervals, and the male-minus-female difference
    (``apc_diff``, the ratio of annual rate ratios expressed as a percentage)
    with its interval, p-value and minimum detectable effect. The ``p`` column
    carries the interaction p-value, so :func:`finalise_sex` controls the false
    discovery rate over the tests of difference.
    """
    yearmin = CONFIG["yearmin"] if yearmin is None else yearmin
    yearmax = CONFIG["yearmax"] if yearmax is None else yearmax
    if level not in ("all-cause", "chapter", "code"):
        raise ValueError("level must be 'all-cause', 'chapter' or 'code'")

    code_to_country = map_codes_to_countries(CONFIG["hmd_mx_dir"])
    hmd_by_country: Dict[str, List[str]] = {}
    for code, name in code_to_country.items():
        hmd_by_country.setdefault(name, []).append(code)
    nz = [c for name, codes in hmd_by_country.items() if "New Zealand" in name for c in codes]
    if nz:
        hmd_by_country["New Zealand"] = nz

    name_equiv = {"The United States of America": "United States",
                  "United States of America": "United States",
                  "Russian Federation": "Russia",
                  "Czech Republic": "Czechia",
                  "Hong Kong SAR": "Hong Kong"}

    if level == "all-cause":
        wanted = set(countries) if countries is not None else set(hmd_by_country) - set(exclude)
        who_name_to_code, who_long = {}, None
    else:
        who_codes = load_who_country_codes()
        who_name_to_code = {
            name_equiv.get(str(r["name"]).strip(), str(r["name"]).strip()): r["country"]
            for _, r in who_codes.iterrows()
        }
        wanted = set(countries) if countries is not None else (
            {name_equiv.get(n, n) for n in hmd_by_country} & set(who_name_to_code)
        ) - set(exclude)
        who_long = load_who_deaths_by_sex(
            yearmin, yearmax,
            who_country_codes=[who_name_to_code[c] for c in wanted if c in who_name_to_code],
            verbose=verbose,
        )
        if level == "chapter":
            who_long = who_long.assign(
                cause_key=who_long.Cause.astype(str).map(lambda c: icd10_chapter(c)[0]))
            who_long = who_long[who_long.cause_key.notna()]
            labels = {numeral: label for numeral, label, _, _ in ICD10_CHAPTERS}
        else:
            who_long = who_long.assign(
                cause_key=who_long.Cause.astype(str).map(
                    lambda c: truncate_icd(c, code_digits)))
            who_long = who_long.groupby(
                ["who_country", "Year", "band", "sex", "cause_key"],
                as_index=False).deaths.sum()
            labels = {}

    rows: List[dict] = []
    for country in sorted(wanted):
        codes = hmd_by_country.get(country) or hmd_by_country.get(
            next((n for n in hmd_by_country if name_equiv.get(n, n) == country), ""), None)
        if not codes:
            rows.append(_failed_row(country, reason="country not found in HMD"))
            continue
        try:
            hmd = _hmd_by_sex(codes, yearmin, yearmax)
        except Exception as exc:
            rows.append(_failed_row(country, reason=f"HMD load failed: {exc}"))
            continue

        if level == "all-cause":
            for age_group in age_groups:
                rows.append(_sex_all_cause_row(country, age_group, hmd, yearmin,
                                               yearmax, dispersion))
        else:
            who_code = who_name_to_code.get(country)
            cdeaths = who_long[who_long.who_country == who_code] if who_code else None
            if cdeaths is None or cdeaths.empty:
                rows.append(_failed_row(country, reason="no WHO records in period"))
                continue
            avail = sorted(set(cdeaths.Year.unique()) & set(hmd.Year.unique()))
            if len(avail) < CONFIG["min_years"]:
                rows.append(_failed_row(country,
                                        reason=f"only {len(avail)} years in both sources"))
                continue
            for age_group in age_groups:
                bands = bands_for_age_group(age_group)
                if not bands:
                    continue
                exposure = pd.concat([
                    hmd_exposure_by_who_band(hmd[hmd.sex == s], bands).assign(sex=s)
                    for s in (0, 1)
                ], ignore_index=True)
                exposure = exposure[exposure.Year.isin(avail)]
                sub = cdeaths[cdeaths.band.isin(bands) & cdeaths.Year.isin(avail)]
                if sub.empty:
                    continue
                totals = sub.groupby("cause_key").deaths.sum().sort_values(ascending=False)
                keys = list(totals.index)
                if top_n_codes is not None and level == "code":
                    keys = keys[:top_n_codes]
                all_deaths = float(totals.sum())
                for key in keys:
                    rows.append(_sex_cause_row(
                        country, age_group, level, key, labels.get(key, key),
                        sub[sub.cause_key == key], exposure, avail, dispersion,
                        min_deaths, all_deaths))
        if verbose:
            print(f"  {country}: {len(rows)} rows so far", flush=True)

    return pd.DataFrame(rows)


def _sex_all_cause_row(country, age_group, hmd, yearmin, yearmax, dispersion) -> dict:
    agemin, agemax = AGE_BINS[age_group]
    sub = hmd[(hmd.Age >= agemin) & (hmd.Age <= agemax)]
    years = sorted(sub.Year.unique())
    row = {
        "country": country, "model": "age-by-sex", "age_group": age_group,
        "cause_level": "all-cause", "cause": "all", "cause_label": "All causes",
        "year_min": int(min(years)) if years else np.nan,
        "year_max": int(max(years)) if years else np.nan,
        "n_years": len(years), "status": "ok", "status_detail": "",
    }
    if len(years) < CONFIG["min_years"]:
        row.update(status="failed", status_detail=f"only {len(years)} usable years")
        return row
    row["total_deaths"] = float(sub.deaths.sum())
    row["person_years"] = float(sub.person_years.sum())
    fit = fit_poisson_sex_trend(sub.deaths.values, sub.person_years.values,
                                sub.Year.values, sub.sex.values,
                                age_index=sub.Age.values, dispersion=dispersion)
    _write_sex_fit(row, fit)
    return row


def _sex_cause_row(country, age_group, level, key, label, deaths_sub, exposure,
                   years, dispersion, min_deaths, all_deaths_in_group) -> dict:
    grid = pd.MultiIndex.from_product(
        [sorted(years), sorted(deaths_sub.band.unique()), (0, 1)],
        names=["Year", "band", "sex"]).to_frame(index=False)
    d = grid.merge(
        deaths_sub.groupby(["Year", "band", "sex"], as_index=False).deaths.sum(),
        on=["Year", "band", "sex"], how="left")
    d["deaths"] = d.deaths.fillna(0.0)          # absent cells are true zeros
    d = d.merge(exposure, on=["Year", "band", "sex"], how="left")
    d = d[d.person_years.notna() & (d.person_years > 0)]

    row = {
        "country": country, "model": "age-by-sex", "age_group": age_group,
        "cause_level": level, "cause": key, "cause_label": label,
        "year_min": int(min(years)), "year_max": int(max(years)),
        "n_years": int(len(years)), "status": "ok", "status_detail": "",
    }
    if d.empty:
        row.update(status="failed", status_detail="no exposure matched")
        return row

    row["total_deaths"] = float(d.deaths.sum())
    row["person_years"] = float(d.person_years.sum())
    row["share_of_group_deaths_pct"] = (
        100.0 * row["total_deaths"] / all_deaths_in_group if all_deaths_in_group > 0 else np.nan)
    if row["total_deaths"] < min_deaths:
        row.update(status="sparse",
                   status_detail=f"{row['total_deaths']:.0f} deaths < min_deaths={min_deaths}")

    band_codes = pd.Categorical(d.band).codes
    fit = fit_poisson_sex_trend(d.deaths.values, d.person_years.values, d.Year.values,
                                d.sex.values, age_index=band_codes, dispersion=dispersion)
    _write_sex_fit(row, fit)
    return row


def finalise_sex(
    est: pd.DataFrame,
    family_cols: Sequence[str] = ("age_group", "cause_level"),
    apc_threshold: float = 1.0,
    diff_threshold: Optional[float] = None,
    use_adjusted: bool = True,
) -> pd.DataFrame:
    """Add FDR-adjusted p-values and classifications to a sex estimates table.

    Three classifications are produced: ``category_female`` and
    ``category_male`` apply the rules of :func:`finalise` to each sex, and
    ``category_sexdiff`` classifies the interaction:

    ``male worse`` / ``female worse``
        the difference is significant after FDR control and at least
        ``diff_threshold`` percentage points per year;
    ``similar trends``
        not significant, and the interval excludes a difference of that size;
    ``inconclusive``
        not significant, but a difference of that size remains possible.

    The false discovery rate is controlled separately within each family for
    each of the three test sets, so the interaction tests are not diluted by the
    (far more numerous) significant main effects.
    """
    out = est.copy()
    diff_threshold = apc_threshold if diff_threshold is None else diff_threshold
    family_cols = [c for c in family_cols if c in out.columns]
    out["test_family"] = (out[family_cols].astype(str).agg(" | ".join, axis=1)
                          if family_cols else "all")

    estimable = out.status.isin(ESTIMABLE_STATUSES)

    for prefix in ("apc_female", "apc_male", "apc_sexavg", "apc_diff"):
        pcol, qcol = f"{prefix}_p", f"{prefix}_p_bh"
        if pcol not in out.columns:
            continue
        masked = out[pcol].where(estimable)
        out[qcol] = masked.groupby(out.test_family).transform(
            lambda s: benjamini_hochberg(s.values))

    def classify(prefix, threshold):
        q = out[f"{prefix}_p_bh"] if use_adjusted else out[f"{prefix}_p"]
        apc, lo, hi = out[prefix], out[f"{prefix}_lo"], out[f"{prefix}_hi"]
        sig = q.lt(0.05)
        material = apc.abs().ge(threshold)
        precise_null = lo.gt(-threshold) & hi.lt(threshold)
        cat = pd.Series("inconclusive", index=out.index, dtype=object)
        cat[sig & material & apc.gt(0)] = "increasing"
        cat[sig & material & apc.lt(0)] = "decreasing"
        cat[sig & ~material] = "significant but small"
        cat[~sig & precise_null] = "no material change"
        cat[~estimable | apc.isna()] = "not estimable"
        return cat

    out["category_female"] = classify("apc_female", apc_threshold)
    out["category_male"] = classify("apc_male", apc_threshold)
    out["category_sexavg"] = classify("apc_sexavg", apc_threshold)

    q = out["apc_diff_p_bh"] if use_adjusted else out["apc_diff_p"]
    diff, lo, hi = out["apc_diff"], out["apc_diff_lo"], out["apc_diff_hi"]
    sig = q.lt(0.05)
    material = diff.abs().ge(diff_threshold)
    cat = pd.Series("inconclusive", index=out.index, dtype=object)
    cat[sig & material & diff.gt(0)] = "male worse"
    cat[sig & material & diff.lt(0)] = "female worse"
    cat[sig & ~material] = "significant but small"
    cat[~sig & lo.gt(-diff_threshold) & hi.lt(diff_threshold)] = "similar trends"
    cat[~estimable | diff.isna()] = "not estimable"
    out["category_sexdiff"] = cat

    out["underpowered_for_sexdiff"] = out.get("apc_diff_mde80") > diff_threshold
    return out


def sex_power_summary(est: pd.DataFrame, diff_threshold: float = 1.0) -> pd.DataFrame:
    """Per age group: how often the data could resolve a sex difference at all.

    Use this before interpreting an absence of sex differences. The interaction
    carries roughly twice the standard error of a pooled trend, so a high
    ``pct_underpowered`` means the analysis could not have detected a difference
    of ``diff_threshold``, not that none exists.
    """
    d = est[est.status.isin(ESTIMABLE_STATUSES) & est.apc_diff_mde80.notna()]
    return (
        d.groupby(["cause_level", "age_group"], observed=True)
        .apply(lambda g: pd.Series({
            "n_cells": len(g),
            "median_mde80_difference": g.apc_diff_mde80.median(),
            "pct_underpowered": 100.0 * (g.apc_diff_mde80 > diff_threshold).mean(),
            "n_differences_found": int(g.category_sexdiff.isin(
                ["male worse", "female worse"]).sum()) if "category_sexdiff" in g else np.nan,
        }), include_groups=False)
        .reset_index()
    )


#: row statuses whose estimates may be interpreted as effect sizes
ESTIMABLE_STATUSES = ("ok", "sparse")


def finalise(
    est: pd.DataFrame,
    family_cols: Sequence[str] = ("sex", "age_group", "cause_level"),
    apc_threshold: float = 1.0,
    use_adjusted: bool = True,
    add_by: bool = True,
) -> pd.DataFrame:
    """Add FDR-adjusted p-values and the four-way classification.

    Parameters
    ----------
    family_cols
        Declares the family of tests over which the false discovery rate is
        controlled. The default treats all country x cause tests within one sex,
        age group and cause level as one family. The adjusted p-values depend
        on this choice.
    apc_threshold
        The APC, in % per year, below which a change is regarded as not
        substantive. Used both for the "no material change" verdict and for the
        power screen.
    use_adjusted
        Classify on the BH-adjusted p-value (recommended) rather than the raw
        one.

    Categories
    ----------
    ``increasing`` / ``decreasing``
        Significant after FDR control **and** |APC| >= ``apc_threshold``.
    ``significant but small``
        Significant but below the threshold: precise, not important.
    ``no material change``
        Not significant, and the CI excludes changes of +/- threshold, so the
        data do rule out a material trend.
    ``inconclusive``
        Not significant and the CI (or the minimum detectable effect) admits a
        material trend - i.e. too few deaths to tell. This is the category that
        keeps small countries in the analysis without over-claiming.
    """
    out = est.copy()
    if "p" not in out.columns:
        raise ValueError("no 'p' column; run a build_* function first")

    family_cols = [c for c in family_cols if c in out.columns]
    out["test_family"] = (
        out[family_cols].astype(str).agg(" | ".join, axis=1) if family_cols else "all"
    )
    out["n_tests_in_family"] = out.groupby("test_family")["p"].transform(
        lambda s: np.isfinite(s).sum()
    )
    out["p_bh"] = out.groupby("test_family")["p"].transform(
        lambda s: benjamini_hochberg(s.values)
    )
    if add_by:
        out["p_by"] = out.groupby("test_family")["p"].transform(
            lambda s: benjamini_yekutieli(s.values)
        )

    p_use = out["p_bh"] if use_adjusted else out["p"]
    sig = p_use < 0.05
    apc, lo, hi = out.get("apc"), out.get("apc_lo"), out.get("apc_hi")
    material = apc.abs() >= apc_threshold
    # CI rules out a material change in either direction
    precise_null = (lo > -apc_threshold) & (hi < apc_threshold)

    category = pd.Series("inconclusive", index=out.index, dtype=object)
    category[sig & material & (apc > 0)] = "increasing"
    category[sig & material & (apc < 0)] = "decreasing"
    category[sig & ~material] = "significant but small"
    category[~sig & precise_null] = "no material change"
    # cells that cannot support an effect-size claim are never given a verdict
    category[out["status"].isin(["failed", "not estimable", "unstable"])] = "not estimable"
    category[out["apc"].isna()] = "not estimable"
    category[~np.isfinite(out["apc"].astype(float))] = "not estimable"
    out["category"] = category

    # three-way classification from the sign of a significant crude OLS slope
    if "legacy_ols_p" in out.columns:
        legacy = pd.Series("neutral", index=out.index, dtype=object)
        legacy[(out.legacy_ols_p < 0.05) & (out.legacy_ols_slope_per_100k > 0)] = "increasing"
        legacy[(out.legacy_ols_p < 0.05) & (out.legacy_ols_slope_per_100k < 0)] = "decreasing"
        out["legacy_category"] = legacy
        out["category_changed_vs_legacy"] = (
            out.legacy_category.replace({"neutral": "no trend"})
            != out.category.replace(
                {
                    "no material change": "no trend",
                    "inconclusive": "no trend",
                    "significant but small": "no trend",
                }
            )
        )

    out["underpowered_for_threshold"] = out.get("mde80") > apc_threshold
    return out


def comparison_table(est: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate the classification against the one from the crude OLS slope."""
    if "legacy_category" not in est.columns:
        raise ValueError("run finalise() first")
    return pd.crosstab(est.legacy_category, est.category, dropna=False)


def power_summary(est: pd.DataFrame, apc_threshold: float = 1.0) -> pd.DataFrame:
    """Per-country summary of detectability: median MDE and share of cells that
    cannot resolve a change of ``apc_threshold``.

    This is the objective answer to 'is the geography of your findings really
    the geography of statistical power?'.
    """
    g = est[est.status.isin(ESTIMABLE_STATUSES)].groupby(["country", "age_group"])
    return (
        g.apply(
            lambda d: pd.Series(
                {
                    "n_cells": len(d),
                    "total_deaths": d.total_deaths.sum(),
                    "median_mde80": d.mde80.median(),
                    "pct_underpowered": 100.0 * (d.mde80 > apc_threshold).mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )


# ----------------------------------------------------------------------------
# Self-test on simulated data
# ----------------------------------------------------------------------------

def _selftest(seed: int = 20260929) -> None:
    """Check the estimator on data with a known answer.

    1. Poisson data with a known APC: the estimate should recover it and the
       nominal 95% CI should cover it about 95% of the time.
    2. A population that ages within a fixed broad band, with age-specific
       rates held perfectly constant: the crude trend must show a spurious
       increase and the age-adjusted trend must not: confounding by age
       composition within the band, reproduced deliberately.
    3. Small vs large country with the same true APC: same point estimate on
       average, much wider CI and larger MDE in the small one.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2012, 2020)
    print("=" * 72)
    print("1. recovery of a known APC (-2%/year), 400 replicates")
    true_apc = -2.0
    beta = math.log(1 + true_apc / 100)
    ages = np.arange(45, 60)
    est, cover = [], 0
    for _ in range(400):
        rows = []
        for a in ages:
            base = 300e-5 * math.exp(0.08 * (a - 45))
            for t in years:
                E = 200_000.0
                mu = E * base * math.exp(beta * (t - years.mean()))
                rows.append((a, t, rng.poisson(mu), E))
        d = pd.DataFrame(rows, columns=["Age", "Year", "deaths", "E"])
        fit = fit_poisson_trend(d.deaths.values, d.E.values, d.Year.values,
                                age_index=d.Age.values, dispersion="quasi")
        est.append(fit.apc)
        lo, hi = fit.ci()
        cover += int(lo <= true_apc <= hi)
    print(f"   mean APC estimate : {np.mean(est):+.3f} %/yr (true {true_apc:+.1f})")
    print(f"   95% CI coverage   : {100 * cover / 400:.1f} %  (expect ~95)")
    assert abs(np.mean(est) - true_apc) < 0.15, "estimator is biased"
    assert 90 <= 100 * cover / 400 <= 99, "coverage is off"

    print("=" * 72)
    print("2. ageing population, age-specific rates exactly constant")
    rows = []
    for t in years:
        shift = (t - years[0]) * 0.02  # population shifts toward older ages
        for a in ages:
            w = 1.0 + shift * (a - ages.mean()) / np.ptp(ages)
            E = 200_000.0 * w
            rate = 300e-5 * math.exp(0.09 * (a - 45))
            rows.append((a, t, E * rate, E))
    d = pd.DataFrame(rows, columns=["Age", "Year", "deaths", "E"])
    tot = d.groupby("Year")[["deaths", "E"]].sum()
    crude_fit = fit_poisson_trend(tot.deaths.values, tot.E.values,
                                  np.asarray(tot.index, float), None, dispersion="poisson")
    adj_fit = fit_poisson_trend(d.deaths.values, d.E.values, d.Year.values,
                                age_index=d.Age.values, dispersion="poisson")
    print(f"   crude APC        : {crude_fit.apc:+.3f} %/yr  (spurious)")
    print(f"   age-adjusted APC : {adj_fit.apc:+.3f} %/yr  (should be ~0)")
    assert crude_fit.apc > 0.1, "the simulated confounding did not materialise"
    assert abs(adj_fit.apc) < 0.02, "age adjustment failed to remove it"

    print("=" * 72)
    print("3. precision vs population size, same true APC (+1.5%/yr)")
    beta = math.log(1.015)
    for label, E in (("small country", 30_000.0), ("large country", 3_000_000.0)):
        apcs, ses = [], []
        for _ in range(200):
            rows = []
            for a in ages:
                for t in years:
                    mu = E * 300e-5 * math.exp(beta * (t - years.mean()))
                    rows.append((a, t, rng.poisson(mu), E))
            d = pd.DataFrame(rows, columns=["Age", "Year", "deaths", "E"])
            f = fit_poisson_trend(d.deaths.values, d.E.values, d.Year.values,
                                  age_index=d.Age.values, dispersion="quasi")
            apcs.append(f.apc)
            ses.append(f.mde80)
        print(f"   {label:14s}: mean APC {np.mean(apcs):+.2f} %/yr, "
              f"median MDE80 {np.median(ses):.2f} %/yr")

    print("=" * 72)
    print("4. Benjamini-Hochberg sanity check")
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.5, np.nan])
    q = benjamini_hochberg(p)
    print("   raw p :", np.round(p, 3))
    print("   BH q  :", np.round(q, 3))
    assert np.isnan(q[-1])
    assert np.all(np.diff(q[:-1][np.isfinite(q[:-1])]) >= -1e-12), "q-values not monotone"

    print("=" * 72)
    print("5. classification and the four-way scheme")
    demo = pd.DataFrame(
        {
            "country": ["Large", "Small", "Large", "Small"],
            "sex": "total",
            "age_group": "45-59",
            "cause_level": "chapter",
            "cause": ["IX", "IX", "X", "X"],
            "status": "ok",
            "apc": [2.5, 2.5, 0.1, 0.1],
            "apc_lo": [2.0, -1.5, -0.1, -6.0],
            "apc_hi": [3.0, 6.5, 0.3, 6.2],
            "p": [1e-6, 0.30, 0.3, 0.9],
            "mde80": [0.4, 5.0, 0.4, 5.0],
            "legacy_ols_p": [1e-6, 0.30, 0.3, 0.9],
            "legacy_ols_slope_per_100k": [3.0, 2.0, 0.1, 0.1],
        }
    )
    got = finalise(demo, apc_threshold=1.0)[["country", "cause", "category"]]
    print(got.to_string(index=False))
    expect = ["increasing", "inconclusive", "no material change", "inconclusive"]
    assert list(got.category) == expect, f"got {list(got.category)}"

    print("=" * 72)
    print("6. ICD-10 chapter mapping")
    for code, numeral in [("X420", "XX"), ("R99", "XVIII"), ("C259", "II"),
                          ("I250", "IX"), ("K703", "XI"), ("F102", "V"),
                          ("D500", "III"), ("T659", "XIX"), ("U071", "XXII")]:
        got_numeral = icd10_chapter(code)[0]
        print(f"   {code:5s} -> {got_numeral}")
        assert got_numeral == numeral, f"{code}: expected {numeral}, got {got_numeral}"

    print("=" * 72)
    print("7. ESP2013 weights")
    w = esp2013_weights(45, 59)
    print("   45-59 weights:", {k: round(v, 4) for k, v in w.items()})
    assert abs(sum(w.values()) - 1.0) < 1e-12
    w2 = esp2013_weights(0, 14)
    assert abs(sum(w2.values()) - 1.0) < 1e-12

    print("=" * 72)
    print("8. degenerate sparse series must be flagged, never overflow")
    years8 = np.arange(2012, 2020)
    cases = {
        "all zero": np.zeros(8),
        "one non-zero year": np.array([0, 0, 0, 0, 0, 0, 0, 3.0]),
        "two non-zero years": np.array([0, 0, 0, 0, 0, 0, 2.0, 3.0]),
        "zeros then deaths": np.array([0, 0, 0, 0, 1.0, 2.0, 4.0, 8.0]),
        "deaths then zeros": np.array([8.0, 4.0, 2.0, 1.0, 0, 0, 0, 0]),
    }
    for label, counts in cases.items():
        f = fit_poisson_trend(counts, np.full(8, 50_000.0), years8, None, dispersion="quasi")
        lo, hi = f.ci()  # must not raise
        print(f"   {label:20s} APC={f.apc!s:>10.10s}  CI=({lo!s:.8s},{hi!s:.8s})  "
              f"note={f.note or '-'}")
        assert f.note != "" or np.isfinite(f.beta), label
        if f.n_years_with_deaths < 3:
            assert not np.isfinite(f.beta), f"{label}: should not have been fitted"

    demo = pd.DataFrame(
        {
            "country": ["A", "B", "C"],
            "sex": "total",
            "age_group": "30-44",
            "cause_level": "code",
            "cause": ["X001", "X002", "X003"],
            "status": ["ok", "unstable", "not estimable"],
            "status_detail": ["", "unstable fit (near-separable sparse series)", "only 1 of 8 years"],
            "apc": [2.0, np.inf, np.nan],
            "apc_lo": [1.5, np.nan, np.nan],
            "apc_hi": [2.5, np.inf, np.nan],
            "p": [0.001, 0.4, np.nan],
            "mde80": [0.5, np.nan, np.nan],
        }
    )
    got = finalise(demo, apc_threshold=1.0)
    print("   ", list(got.category))
    assert list(got.category) == ["increasing", "not estimable", "not estimable"]

    print("=" * 72)
    print("9. age-by-sex model: recovery of known sex-specific trends")
    years8 = np.arange(2012, 2020)
    ages9 = np.arange(45, 60)
    true_f, true_m = -2.0, +1.0          # %/year
    bf, bm = math.log(1 + true_f / 100), math.log(1 + true_m / 100)
    est_f, est_m, est_d, cover_d = [], [], [], 0
    for _ in range(300):
        rows = []
        for a in ages9:
            for s, b in ((0, bf), (1, bm)):
                for t in years8:
                    E = 150_000.0
                    base = 300e-5 * math.exp(0.08 * (a - 45)) * (1.6 if s == 1 else 1.0)
                    mu = E * base * math.exp(b * (t - years8.mean()))
                    rows.append((a, s, t, rng.poisson(mu), E))
        d = pd.DataFrame(rows, columns=["Age", "sex", "Year", "deaths", "E"])
        f = fit_poisson_sex_trend(d.deaths.values, d.E.values, d.Year.values,
                                  d.sex.values, age_index=d.Age.values,
                                  dispersion="quasi")
        est_f.append(f.trend("female")[0])
        est_m.append(f.trend("male")[0])
        dd = f.trend("difference")
        est_d.append(dd[0])
        true_diff = 100 * (math.exp(bm - bf) - 1)
        cover_d += int(dd[1] <= true_diff <= dd[2])
    true_diff = 100 * (math.exp(bm - bf) - 1)
    print(f"   female APC : {np.mean(est_f):+.3f} (true {true_f:+.1f})")
    print(f"   male   APC : {np.mean(est_m):+.3f} (true {true_m:+.1f})")
    print(f"   difference : {np.mean(est_d):+.3f} (true {true_diff:+.3f})")
    print(f"   95% CI coverage of the difference: {100 * cover_d / 300:.1f}%")
    assert abs(np.mean(est_f) - true_f) < 0.15
    assert abs(np.mean(est_m) - true_m) < 0.15
    assert abs(np.mean(est_d) - true_diff) < 0.2
    assert 90 <= 100 * cover_d / 300 <= 99

    print("=" * 72)
    print("10. variance relations claimed in the methods note")
    rows = []
    for a in ages9:
        for s in (0, 1):
            for t in years8:
                E = 150_000.0
                mu = E * 300e-5 * math.exp(0.08 * (a - 45))
                rows.append((a, s, t, rng.poisson(mu), E))
    d = pd.DataFrame(rows, columns=["Age", "sex", "Year", "deaths", "E"])
    fs = fit_poisson_sex_trend(d.deaths.values, d.E.values, d.Year.values, d.sex.values,
                               age_index=d.Age.values, dispersion="poisson")
    pooled = fit_poisson_trend(d.groupby(["Age", "Year"]).deaths.sum().values,
                               d.groupby(["Age", "Year"]).E.sum().values,
                               np.array([y for a, y in d.groupby(["Age", "Year"]).groups]),
                               age_index=np.array([a for a, y in d.groupby(["Age", "Year"]).groups]),
                               dispersion="poisson")
    se_f = fs._contrast(1.0, 0.0)[1]
    se_d = fs._contrast(0.0, 1.0)[1]
    se_avg = fs._contrast(1.0, 0.5)[1]
    print(f"   se(one sex)    / se(pooled) = {se_f / pooled.se:.3f}   (expect ~1.414)")
    print(f"   se(difference) / se(pooled) = {se_d / pooled.se:.3f}   (expect ~2.000)")
    print(f"   se(sex-average)/ se(pooled) = {se_avg / pooled.se:.3f}   (expect ~1.000)")
    assert 1.30 < se_f / pooled.se < 1.55
    assert 1.85 < se_d / pooled.se < 2.15
    assert 0.92 < se_avg / pooled.se < 1.08

    print("=" * 72)
    print("11. age-by-sex model: degenerate cells are refused, not fitted")
    base = [(a, s, t, 5.0, 50_000.0) for a in (45, 50) for s in (0, 1) for t in years8]
    d = pd.DataFrame(base, columns=["Age", "sex", "Year", "deaths", "E"])
    d.loc[d.sex == 1, "deaths"] = 0.0                       # no male deaths at all
    d.loc[(d.sex == 1) & (d.Year == 2019), "deaths"] = 3.0  # except one year
    f = fit_poisson_sex_trend(d.deaths.values, d.E.values, d.Year.values, d.sex.values,
                              age_index=d.Age.values)
    print(f"   note = {f.note}")
    assert not np.isfinite(f.beta) and "male" in f.note
    row = {"status": "ok", "status_detail": ""}
    _write_sex_fit(row, f)
    assert row["status"] == "not estimable"

    print("=" * 72)
    print("12. finalise_sex classification")
    demo = pd.DataFrame({
        "country": ["Large", "Large", "Small"],
        "age_group": "30-44", "cause_level": "all-cause",
        "status": "ok", "status_detail": "",
        "apc_female": [-1.0, 0.1, 2.0], "apc_female_lo": [-1.3, -0.1, -3.0],
        "apc_female_hi": [-0.7, 0.3, 7.0], "apc_female_p": [1e-8, 0.3, 0.5],
        "apc_male": [3.0, 0.1, 2.0], "apc_male_lo": [2.6, -0.1, -3.0],
        "apc_male_hi": [3.4, 0.3, 7.0], "apc_male_p": [1e-9, 0.3, 0.5],
        "apc_sexavg": [1.0, 0.1, 2.0], "apc_sexavg_lo": [0.7, -0.1, -3.0],
        "apc_sexavg_hi": [1.3, 0.3, 7.0], "apc_sexavg_p": [1e-6, 0.3, 0.5],
        "apc_diff": [4.0, 0.0, 0.5], "apc_diff_lo": [3.2, -0.2, -9.0],
        "apc_diff_hi": [4.8, 0.2, 10.0], "apc_diff_p": [1e-7, 0.9, 0.9],
        "apc_diff_mde80": [0.6, 0.3, 9.0],
    })
    got = finalise_sex(demo, apc_threshold=1.0)
    print(got[["country", "category_female", "category_male",
               "category_sexdiff"]].to_string(index=False))
    assert list(got.category_sexdiff) == ["male worse", "similar trends", "inconclusive"]
    assert list(got.category_female) == ["decreasing", "no material change", "inconclusive"]
    assert list(got.underpowered_for_sexdiff) == [False, False, True]

    print("=" * 72)
    print("13. coverage under overdispersion: Poisson vs quasi vs NB")
    years8 = np.arange(2012, 2020)
    ages13 = np.arange(45, 60)
    true_apc = -1.5
    b = math.log(1 + true_apc / 100)
    # negative-binomial counts: Var = mu + alpha*mu^2, so phi = 1 + alpha*mu
    # grows with the cell size, which is the pattern seen in the real data
    alpha_true = 1.5e-3   # gives phi ~ 3, as in the largest real cells
    cov = {k: 0 for k in ("poisson", "quasi", "nb")}
    phis = []
    REPS = 150
    for _ in range(REPS):
        rows = []
        for a in ages13:
            for t in years8:
                E = 400_000.0
                mu = E * 300e-5 * math.exp(0.08 * (a - 45)) * math.exp(b * (t - years8.mean()))
                shape = 1.0 / alpha_true
                lam = rng.gamma(shape, mu / shape)      # Poisson-gamma = NB2
                rows.append((a, t, rng.poisson(lam), E))
        d = pd.DataFrame(rows, columns=["Age", "Year", "deaths", "E"])
        for kind in ("poisson", "quasi", "nb"):
            f = fit_poisson_trend(d.deaths.values, d.E.values, d.Year.values,
                                  age_index=d.Age.values, dispersion=kind)
            lo, hi = f.ci()
            cov[kind] += int(lo <= true_apc <= hi)
            if kind == "quasi":
                phis.append(f.dispersion)
    print(f"   simulated phi (median): {np.median(phis):.2f}")
    for kind in ("poisson", "quasi", "nb"):
        print(f"   {kind:8s} 95% CI coverage: {100 * cov[kind] / REPS:5.1f}%")
    assert cov["poisson"] / REPS < 0.90, "Poisson should under-cover here"
    assert 0.88 <= cov["quasi"] / REPS <= 1.0, "quasi-Poisson should recover coverage"
    assert 0.85 <= cov["nb"] / REPS <= 1.0, "NB should recover coverage"

    print("=" * 72)
    print("14. dispersion floor")
    rows = []
    for a in (45, 50, 55):
        for t in years8:
            E = 100_000.0
            rows.append((a, t, E * 1e-3 * math.exp(b * (t - years8.mean())), E))
    d = pd.DataFrame(rows, columns=["Age", "Year", "deaths", "E"])  # no noise at all
    f_floor = fit_poisson_trend(d.deaths.values, d.E.values, d.Year.values,
                                age_index=d.Age.values, dispersion="quasi",
                                dispersion_floor=1.0)
    f_raw = fit_poisson_trend(d.deaths.values, d.E.values, d.Year.values,
                              age_index=d.Age.values, dispersion="quasi",
                              dispersion_floor=None)
    print(f"   phi_hat = {f_raw.dispersion:.4f} (underdispersed by construction)")
    print(f"   se with floor {f_floor.se:.5f} vs without {f_raw.se:.5f}")
    assert f_raw.dispersion < 1
    assert f_floor.se > f_raw.se, "the floor must not narrow the interval"
    assert abs(f_floor.beta - f_raw.beta) < 1e-12, "the floor must not move the estimate"

    print("=" * 72)
    print("15. the dispersion choice never moves the point estimate")
    betas = {k: fit_poisson_trend(d.deaths.values, d.E.values, d.Year.values,
                                  age_index=d.Age.values, dispersion=k).beta
             for k in ("poisson", "quasi", "sandwich")}
    print("   ", {k: round(v, 10) for k, v in betas.items()})
    assert max(betas.values()) - min(betas.values()) < 1e-12

    print("=" * 72)
    print("16. exact t and F agree with the reference implementation")
    if _scipy_stats is not None:
        for tv, df in [(2.1, 20), (0.5, 6), (4.0, 104)]:
            assert abs(t_two_sided_p(tv, df) - 2 * _scipy_stats.t.sf(tv, df)) < 1e-10
        for fv, d1, d2 in [(4.0, 1, 20), (0.8, 1, 6), (9.1, 2, 100)]:
            assert abs(f_sf(fv, d1, d2) - _scipy_stats.f.sf(fv, d1, d2)) < 1e-10
        print("   t and F match scipy to 1e-10")
    else:
        print("   scipy absent; skipped (the module no longer needs it)")

    print("=" * 72)
    print("17. linearity test: size under a true straight line, power against a break")
    years8 = np.arange(2012, 2020)
    ages17 = np.arange(45, 60)
    b = math.log(1 - 0.015)

    def simulate(break_at=None, jump=0.0, E=250_000.0):
        rows = []
        for a in ages17:
            for t in years8:
                base = 300e-5 * math.exp(0.08 * (a - 45))
                eta = b * (t - years8.mean())
                if break_at is not None:
                    eta += jump * max(t - break_at, 0)
                rows.append((a, t, rng.poisson(E * base * math.exp(eta)), E))
        return pd.DataFrame(rows, columns=["Age", "Year", "deaths", "E"])

    REPS = 200
    rej_null = 0
    for _ in range(REPS):
        d = simulate()
        r = linearity_test(d.deaths.values, d.E.values, d.Year.values,
                           age_index=d.Age.values, scan_knots=False)
        rej_null += int(r["lin_p_knot"] < 0.05)
    size = 100 * rej_null / REPS
    print(f"   truly linear  : rejected in {size:.1f}% of replicates (nominal 5%)")
    assert 1.0 <= size <= 12.0, "test size is off"

    rej_alt, caught = 0, 0
    for _ in range(REPS):
        d = simulate(break_at=2015, jump=math.log(1.04))   # +4%/yr extra after 2015
        r = linearity_test(d.deaths.values, d.E.values, d.Year.values,
                           age_index=d.Age.values, scan_knots=True)
        rej_alt += int(r["lin_p_knot"] < 0.05)
        caught += int(abs(r["lin_bestknot_year"] - 2015) <= 1)
    print(f"   real break    : rejected in {100 * rej_alt / REPS:.1f}% of replicates")
    print(f"   best knot within one year of the truth: {100 * caught / REPS:.0f}%")
    assert rej_alt / REPS > 0.5, "no power against a clear break"

    print("=" * 72)
    print("18. linearity_summary")
    demo = pd.DataFrame({
        "age_group": ["30-44"] * 6, "cause_level": ["all-cause"] * 6,
        "status": "ok",
        "lin_p_knot": [0.001, 0.2, 0.4, 0.6, 0.8, 0.9],
        "lin_p_quad": [0.002, 0.3, 0.5, 0.7, 0.8, 0.95],
    })
    out = linearity_summary(demo)
    print(out.round(1).to_string(index=False))
    assert out.n_cells.iloc[0] == 6

    print("=" * 72)
    print("ALL SELF-TESTS PASSED")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--selftest", action="store_true", help="run simulation checks and exit")
    ap.add_argument("--all-cause", action="store_true", help="build the all-cause table")
    ap.add_argument("--causes", action="store_true", help="build the cause-specific table")
    ap.add_argument("--sex-model", action="store_true",
                    help="build the age-by-sex table (separate model, separate output)")
    ap.add_argument("--level", choices=["all-cause", "chapter", "code"], default="chapter")
    ap.add_argument("--age-groups", nargs="+", default=None, metavar="BAND",
                    help="age bands to analyse (default: 0-14 pooled). Use "
                         "'--age-groups revised' for the bands of the paper, with "
                         "infancy separated from childhood, or name them "
                         "explicitly, e.g. --age-groups 0 1-14 15-29 30-44")
    ap.add_argument("--sex", choices=["total", "male", "female"], default="total")
    ap.add_argument("--yearmin", type=int, default=CONFIG["yearmin"])
    ap.add_argument("--yearmax", type=int, default=CONFIG["yearmax"])
    ap.add_argument("--min-deaths", type=int, default=50)
    ap.add_argument("--top-n-codes", type=int, default=None)
    ap.add_argument("--code-digits", type=int, default=CODE_DIGITS_DEFAULT,
                    choices=[3, 4],
                    help="ICD detail for --level code: 3 (category, default and "
                         "recommended) or 4. At 4 characters, countries that "
                         "migrate between X70.9 and X70 style coding mid-period "
                         "produce spurious trends in both halves.")
    ap.add_argument("--apc-threshold", type=float, default=1.0,
                    help="%%/year regarded as a material change")
    ap.add_argument("--dispersion", choices=["quasi", "poisson", "sandwich", "nb"],
                    default="quasi",
                    help="variance model: quasi-Poisson (default), Poisson, "
                         "heteroskedasticity-robust sandwich, or negative binomial")
    ap.add_argument("--linearity-test", action="store_true",
                    help="also test whether a single log-linear slope is adequate "
                         "over the window (fixed-knot and quadratic alternatives)")
    ap.add_argument("--dispersion-floor", type=float, default=DISPERSION_FLOOR_DEFAULT,
                    help="lower bound on the estimated dispersion before it scales "
                         "the covariance (default 1.0; pass 0 to use the raw estimate)")
    ap.add_argument("--countries", nargs="*", default=None)
    ap.add_argument("--countries-from", default=None, metavar="CSV",
                    help="restrict to the countries that have at least one usable "
                         "estimate in an existing table, so that the all-cause and "
                         "cause-specific analyses cover the same countries and the "
                         "false discovery rate families are built on that set")
    ap.add_argument("--out", default=None, help="output CSV path")
    args = ap.parse_args(argv)

    if args.selftest:
        _selftest()
        return 0

    if args.countries_from:
        if args.countries:
            ap.error("give --countries or --countries-from, not both")
        ref = pd.read_csv(args.countries_from, usecols=["country", "status"],
                          low_memory=False)
        args.countries = sorted(
            ref.loc[ref.status.isin(ESTIMABLE_STATUSES), "country"].unique())
        print(f"restricted to the {len(args.countries)} countries with usable "
              f"estimates in {Path(args.countries_from).name}")

    if args.age_groups is None:
        age_groups = tuple(PAPER_AGE_GROUPS)
    elif len(args.age_groups) == 1 and args.age_groups[0].lower() == "revised":
        age_groups = tuple(REVISED_AGE_GROUPS)
    else:
        age_groups = tuple(args.age_groups)
    unknown = [a for a in age_groups if a not in AGE_BINS]
    if unknown:
        ap.error(f"unknown age band(s): {unknown}; known: {list(AGE_BINS)}")
    if "0-14" in age_groups and ("0" in age_groups or "1-14" in age_groups):
        warnings.warn("'0-14' overlaps '0'/'1-14'; the FDR family will contain "
                      "overlapping tests. Run them separately.")
    print(f"age bands: {list(age_groups)}")

    if not (args.all_cause or args.causes or args.sex_model):
        ap.error("choose --all-cause, --causes, --sex-model or --selftest")

    if args.sex_model:
        est = build_sex_estimates(
            level=args.level, countries=args.countries, age_groups=age_groups,
            yearmin=args.yearmin, yearmax=args.yearmax,
            dispersion=args.dispersion, min_deaths=args.min_deaths,
            top_n_codes=args.top_n_codes, code_digits=args.code_digits,
        )
        est = finalise_sex(est, apc_threshold=args.apc_threshold)
        out = args.out or f"estimates_sex_{args.level}_{args.yearmin}-{args.yearmax}.csv"
        est.to_csv(out, index=False)
        print(f"\nwrote {len(est)} rows to {out}")
        ok = est[est.status.isin(ESTIMABLE_STATUSES)]
        print("\nsex-difference categories:")
        print(est.category_sexdiff.value_counts().to_string())
        print("\nfemale / male trend categories:")
        print(pd.crosstab(ok.category_female, ok.category_male).to_string())
        print("\npower to resolve a sex difference:")
        print(sex_power_summary(ok, diff_threshold=args.apc_threshold)
              .round(2).to_string(index=False))
        return 0

    if args.all_cause:
        est = build_all_cause_estimates(
            countries=args.countries, age_groups=age_groups,
            yearmin=args.yearmin, yearmax=args.yearmax,
            sexes=(args.sex,), dispersion=args.dispersion,
            linearity=args.linearity_test,
        )
        out = args.out or f"estimates_all_cause_{args.yearmin}-{args.yearmax}_{args.sex}.csv"
    else:
        est = build_cause_estimates(
            level=args.level, countries=args.countries, age_groups=age_groups,
            yearmin=args.yearmin, yearmax=args.yearmax, sex=args.sex,
            dispersion=args.dispersion, min_deaths=args.min_deaths,
            top_n_codes=args.top_n_codes, code_digits=args.code_digits,
            linearity=args.linearity_test,
        )
        out = args.out or (
            f"estimates_{args.level}_{args.yearmin}-{args.yearmax}_{args.sex}.csv"
        )

    est = finalise(est, apc_threshold=args.apc_threshold)
    est.to_csv(out, index=False)
    print(f"\nwrote {len(est)} rows to {out}")

    ok = est[est.status.isin(ESTIMABLE_STATUSES)]
    print("\ncategory counts:")
    print(est.category.value_counts().to_string())
    if "legacy_category" in ok.columns:
        print("\nclassification vs crude OLS classification (estimable cells only):")
        print(comparison_table(ok).to_string())

    if args.linearity_test and "lin_p_knot" in est.columns:
        print("\nis a straight line adequate over the window?")
        print(linearity_summary(est).round(1).to_string(index=False))

    bad = est[~est.status.isin(ESTIMABLE_STATUSES)]
    if len(bad):
        print(f"\n{len(bad)} cells not estimable, by reason:")
        print(bad.groupby(["status", "status_detail"]).size().to_string())
        print("\n(first 20)")
        print(
            bad[["country", "age_group", "cause", "total_deaths", "status", "status_detail"]]
            .head(20)
            .to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
