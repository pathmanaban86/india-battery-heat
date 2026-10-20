"""Lifetime metrics used in the paper."""
import numpy as np


def time_to(q, t_days, threshold):
    """Years until remaining capacity first reaches `threshold`; NaN if not reached within the simulation."""
    i = np.where(q <= threshold)[0]
    return t_days[i[0]] / 365.25 if len(i) else np.nan


def q_at_years(q, t_days, years):
    """Remaining capacity after `years` (linear interpolation)."""
    return float(np.interp(years * 365.25, t_days, q))


def lifetime_summary(sim, eol=(0.80, 0.75, 0.70)):
    """t80/t75/t70, Q after 10 years and at the end of the simulation."""
    q, td = sim['q'], sim['t_days']
    out = {f'life{int(round(e * 100))}_yr': time_to(q, td, e) for e in eol}
    out['q_end'] = float(q[-1]); out['q_yr10'] = q_at_years(q, td, 10)
    return out


def calendar_share(sim, years=10):
    """Share (%) of calendar loss in the total capacity loss after `years`."""
    tot = 1 - q_at_years(sim['q'], sim['t_days'], years)
    cal = 1 - q_at_years(sim['q_cal'], sim['t_days'], years)
    return 100 * cal / tot if tot > 0 else np.nan


def out_of_range_share(T, soc, rng):
    """Share (%) of SOC throughput at pack temperatures outside the tested cycling range `rng` = (min, max) degC."""
    thr = np.abs(np.diff(soc, append=soc[-1]))
    tot = thr.sum()
    out = (T < rng[0]) | (T > rng[1])
    return 100 * thr[out].sum() / tot if tot > 0 else np.nan


def anomaly_pct(series):
    """Relative anomaly (%) of each value to the mean of the series."""
    return (series / series.mean() - 1) * 100
