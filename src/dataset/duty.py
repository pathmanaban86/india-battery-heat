"""Hourly duty profiles: SOC at the start of each hour and operating mode (0 parked, 1 driving, 2 charging/cycling).

Reference templates follow Table 3 of the paper; parametric templates are used in the Monte Carlo analysis.
"""
import numpy as np
from src.utils.config import load

D = load('duty_profiles')


def e2w_day(full, weekend):
    p = D['E2W']
    d = p['soc_per_trip_weekend'] if weekend else p['soc_per_trip_weekday']
    soc, mode = np.full(24, full), np.zeros(24, int)
    th = tuple(p['trips_weekend'] if weekend else p['trips_weekday'])
    soc[th[0] + 1:] -= d; soc[th[1] + 1:] -= d; mode[list(th)] = 1
    s = full - 2 * d
    for h in p['charge_hours']:
        mode[h] = 2; s = min(full, s + p['charge_soc_per_hour']); soc[h + 1:] = s
    return soc, mode


def e3w_day(full, rest_day):
    p = D['E3W']
    soc, mode = np.full(24, full), np.zeros(24, int)
    if rest_day:
        return soc, mode
    s = full
    for h, m in p['plan']:
        mode[h] = m
        s = s - p['soc_per_drive_hour'] if m == 1 else min(full, s + p['charge_soc_per_hour'])
        soc[h + 1:] = s
    return soc, mode


def bess_day():
    p = D['BESS']
    mode = np.zeros(24, int); mode[p['operating_hours']] = 2
    return np.array(p['soc'], float), mode


def car_day():
    p = D['CAR']
    soc = np.full(24, p['soc_limit']); mode = np.zeros(24, int)
    t0, t1 = p['trips']
    soc[t0 + 1:] -= p['soc_per_trip']; soc[t1 + 1:] -= p['soc_per_trip']; mode[[t0, t1]] = 1
    mode[p['charge_hour']] = 2; soc[p['charge_hour'] + 1:] = p['soc_limit']
    return soc, mode


def build_series(sc, dow):
    """SOC and mode series for a scenario, given the day of week of every hour (Monday = 0)."""
    tpl = {}
    for wd in range(7):
        if sc['sys'] == 'E2W': tpl[wd] = e2w_day(sc['cap'], wd >= 5)
        elif sc['sys'] == 'E3W': tpl[wd] = e3w_day(sc['cap'], wd == D['E3W']['rest_day'])
        elif sc['sys'] == 'BESS': tpl[wd] = bess_day()
        else: tpl[wd] = car_day()
    days = dow[::24]
    return np.concatenate([tpl[d][0] for d in days]), np.concatenate([tpl[d][1] for d in days])


# ------------------------------------------------------------------ parametric templates (Monte Carlo)
def e2w_param(full, weekend, km, wh, h0, pkw):
    cap = 3.0
    d = (km / 2) * wh / 1000 / cap
    if weekend:
        d *= 0.5
    soc, mode = np.full(24, full), np.zeros(24, int)
    th = (11, 17) if weekend else (8, 18)
    soc[th[0] + 1:] -= d; soc[th[1] + 1:] -= d; mode[list(th)] = 1
    s = full - 2 * d; h = h0
    while s < full - 1e-9 and h < 24:       # charger power and start chosen so charging ends before midnight
        mode[h] = 2; s = min(full, s + pkw / cap); soc[h + 1:] = s; h += 1
    return soc, mode


def e3w_param(full, rest_day, km, opp, pkw):
    soc, mode = np.full(24, full), np.zeros(24, int)
    if rest_day:
        return soc, mode
    dh = km * 0.060 / 8.0 / 10.0
    s = full
    plan = [(h, 1) for h in range(7, 12)] + ([(12, 2), (13, 2)] if opp else [(12, 0), (13, 0)]) + [(h, 1) for h in range(14, 19)] + [(h, 2) for h in range(19, 24)]
    for h, m in plan:
        if m == 1:
            s = max(0.05, s - dh); mode[h] = 1
        elif m == 2 and s < full - 1e-9:
            s = min(full, s + pkw / 8.0); mode[h] = 2
        soc[h + 1:] = s
    return soc, mode


def bess_day_crate(crate):
    """Daily BESS template for 0.4 C (reference) or 0.6 C (stress cases)."""
    if crate <= 0.45:
        return bess_day()
    soc = np.array([.1] * 11 + [.1, .7] + [.9] * 7 + [.9, .3] + [.1] * 2)[:24]
    mode = np.zeros(24, int); mode[[11, 12, 19, 20]] = 2
    return soc, mode


def erickshaw_protocol(dod, days=400):
    """Two 12-h cycles per day at constant temperature (Bharathamani et al., 2025, adapted)."""
    cyc = np.ones(12); d = dod / 3
    cyc[1:4] = 1 - d * np.arange(1, 4); cyc[4:6] = 1 - dod; cyc[6:9] = 1 - dod + d * np.arange(1, 4); cyc[9:] = 1.0
    return np.tile(cyc, 2 * days)
