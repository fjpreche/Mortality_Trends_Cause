"""Segment slopes for the series that reject a single straight line.

For every all-cause cell whose fixed-knot linearity test survives false
discovery rate control (Supplementary File 1, Section 2.5), refit the trend
allowing the slope to change at the knot used by the test, the midpoint of the
window, and report the annual percentage change either side of it with
confidence intervals, beside the single-slope estimate.

Intervals and the test for a change in slope use the same quasi-Poisson
covariance and t reference distribution as the main analysis, so they are
comparable with the estimates table.

Needs the HMD data (see mortality_estimates.py) and the all-cause estimates
table. Run from the repository root:

    python check_segments.py estimates/estimates_all_cause.csv

Writes linearity_segments.csv to the working directory and prints the table.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mortality_estimates as me

DROP = {"England and Wales", "Scotland", "Northern Ireland",
        "East Germany", "West Germany"}
EST = ("ok", "sparse")
#: a change smaller than this is not treated as material in the paper
APC_THRESHOLD = 1.0


def apc(beta):
    """Annual percentage change from a log-linear slope."""
    return 100.0 * (np.exp(beta) - 1.0)


def apc_ci(beta, se, df):
    """Interval for the APC, built on the log scale and then transformed."""
    if not np.isfinite(se) or se <= 0:
        return np.nan, np.nan
    crit = me.t_quantile(df)
    return apc(beta - crit * se), apc(beta + crit * se)


def main(path="estimates_all_cause.csv", alpha=0.05):
    est = pd.read_csv(path, low_memory=False)
    est = est[est.status.isin(EST) & est.lin_p_knot.notna()
              & ~est.country.isin(DROP)].copy()
    # One family per age group, as in Supplementary File 1 (all country x cause
    # tests sharing a sex, age group and cause level).
    est["q_knot"] = est.groupby("age_group", observed=True).lin_p_knot.transform(
        lambda s: pd.Series(me.benjamini_hochberg(s.values), index=s.index))
    sel = est[(est.q_knot < alpha) & est.lin_knot_year.notna()]
    print(f"{len(sel)} of {len(est)} cells reject linearity after FDR control\n")

    code_to_country = me.map_codes_to_countries(me.CONFIG["hmd_mx_dir"])
    by_country = {}
    for code, name in code_to_country.items():
        by_country.setdefault(name, []).append(code)

    rows, cache = [], {}
    for _, r in sel.iterrows():
        codes = by_country.get(r.country)
        if not codes:
            print(f"  skipped {r.country}: not found in HMD")
            continue
        if r.country not in cache:
            cache[r.country] = me.load_hmd_country(
                codes, me.CONFIG["yearmin"], me.CONFIG["yearmax"])
        amin, amax = me.AGE_BINS[r.age_group]
        d = cache[r.country]
        d = d[(d.Age >= amin) & (d.Age <= amax)]
        y = d.deaths.values.astype(float)
        E = d.person_years.values.astype(float)
        t = d.Year.values.astype(float)
        ages = d.Age.values
        good = np.isfinite(y) & np.isfinite(E) & (E > 0)
        y, E, t, ages = y[good], E[good], t[good], ages[good]
        D = np.column_stack([(ages == lv).astype(float) for lv in np.unique(ages)])

        def segments(tau):
            """Slope either side of a change at tau, with intervals.

            The design is [age factor, centred year, (year - tau)+], so the
            coefficient on the hinge is the CHANGE in slope: the second-segment
            slope is beta + gamma and its variance needs the covariance term.
            """
            X = np.column_stack([D, t - t.mean(), np.maximum(t - tau, 0.0)])
            f = me._irls_poisson(X, y, E, dispersion="quasi",
                                 dispersion_floor=me.DISPERSION_FLOOR_DEFAULT)
            if f is None or not f["converged"]:
                return None
            b, g = f["theta"][-2], f["theta"][-1]
            V, df = f["cov"], f["df_resid"]
            se_b = float(np.sqrt(max(V[-2, -2], 0.0)))
            se_g = float(np.sqrt(max(V[-1, -1], 0.0)))
            se_after = float(np.sqrt(max(V[-2, -2] + V[-1, -1] + 2 * V[-2, -1], 0.0)))
            lo1, hi1 = apc_ci(b, se_b, df)
            lo2, hi2 = apc_ci(b + g, se_after, df)
            # the change in slope, as a ratio of annual rates of change
            lo_d, hi_d = apc_ci(g, se_g, df)
            p_change = (me.two_sided_p(g / se_g, df) if se_g > 0 else np.nan)
            yrs = np.unique(t)
            arm = min(int((yrs <= tau).sum()), int((yrs >= tau).sum()))
            return dict(before=apc(b), before_lo=lo1, before_hi=hi1,
                        after=apc(b + g), after_lo=lo2, after_hi=hi2,
                        change=apc(g), change_lo=lo_d, change_hi=hi_d,
                        p_change=p_change, min_arm_years=arm,
                        dispersion=f["dispersion"], df_resid=df)

        fixed = segments(float(r.lin_knot_year))
        if fixed is None:
            print(f"  skipped {r.country} {r.age_group}: segmented fit failed")
            continue

        rec = dict(country=r.country, age_group=r.age_group,
                   category=r.category,
                   linear_apc=float(r.apc), linear_lo=float(r.apc_lo),
                   linear_hi=float(r.apc_hi),
                   p_knot=float(r.lin_p_knot), q_knot=float(r.q_knot),
                   material=bool(abs(float(r.apc)) >= APC_THRESHOLD),
                   fixed_knot=float(r.lin_knot_year))
        for pre, s in (("fixed", fixed),):
            for k, v in s.items():
                rec[f"{pre}_{k}"] = v
            rec[f"{pre}_same_direction"] = bool(
                np.sign(s["before"]) == np.sign(s["after"]))
            # do the two segment intervals overlap? non-overlap is a strong
            # statement, overlap a weak one; the formal test is p_change.
            rec[f"{pre}_ci_overlap"] = bool(
                s["before_lo"] <= s["after_hi"] and s["after_lo"] <= s["before_hi"])
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        print("nothing to report")
        return out

    def fmt(v, lo, hi):
        return f"{v:6.2f} ({lo:6.2f}, {hi:6.2f})"

    print("Slopes either side of the knot at the midpoint of the window\n")
    show = pd.DataFrame({
        "country": out.country, "age": out.age_group,
        "category": out.category,
        "whole window": [fmt(a, b, c) for a, b, c
                         in zip(out.linear_apc, out.linear_lo, out.linear_hi)],
        "first half": [fmt(a, b, c) for a, b, c
                       in zip(out.fixed_before, out.fixed_before_lo, out.fixed_before_hi)],
        "second half": [fmt(a, b, c) for a, b, c
                        in zip(out.fixed_after, out.fixed_after_lo, out.fixed_after_hi)],
        "p(change)": out.fixed_p_change.map(lambda v: f"{v:.4f}"),
        "same dir": out.fixed_same_direction,
        "material": out.material,
    })
    print(show.to_string(index=False))

    n = len(out)
    print(f"\n  same direction either side:      {int(out.fixed_same_direction.sum())} of {n}")
    print(f"  segment intervals overlap:       {int(out.fixed_ci_overlap.sum())} of {n}")
    print(f"  both segment intervals exclude 0:"
          f" {int(((out.fixed_before_lo * out.fixed_before_hi > 0) & (out.fixed_after_lo * out.fixed_after_hi > 0)).sum())} of {n}")
    print(f"  shortest arm 3 years or fewer:   {int((out.fixed_min_arm_years <= 3).sum())} of {n}")

    print("\n  by size of the whole-window change:")
    band = pd.cut(out.linear_apc.abs(), [0, 1, 2, np.inf],
                  labels=["< 1 %/yr", "1-2 %/yr", ">= 2 %/yr"], right=False)
    print(out.groupby(band, observed=True)
             .fixed_same_direction.agg(cells="size", agree="sum").to_string())

    rev = out[~out.fixed_same_direction]
    if len(rev):
        print(f"\n  reversing cells ({len(rev)}), of which "
              f"{int((~rev.material).sum())} are below the "
              f"{APC_THRESHOLD:g} %/yr materiality threshold:")
        print(rev[["country", "age_group", "category", "linear_apc",
                   "fixed_before", "fixed_after"]].round(2).to_string(index=False))

    out.to_csv("linearity_segments.csv", index=False)
    print("\nwrote linearity_segments.csv")
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
