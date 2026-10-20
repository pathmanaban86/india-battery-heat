"""Scenario sampling for the joint Monte Carlo analysis (the analogue of data augmentation in this study).

The order of random draws is fixed so that the published samples are reproduced exactly:
start year, tau, k_sun, k_shade, dT_drive, dT_charge, then the duty parameters.
"""
import numpy as np


def draw(rng, sysn, p0, mc):
    y0 = int(rng.integers(2001, 2017))
    lo, hi = mc['thermal_multipliers']['tau_log_uniform']
    tau = p0['tau'] * np.exp(rng.uniform(np.log(lo), np.log(hi)))
    klo, khi = mc['thermal_multipliers']['k_sol_uniform']
    k_sun = p0['k_sun'] * rng.uniform(klo, khi); k_shade = p0['k_shade'] * rng.uniform(klo, khi)
    dlo, dhi = mc['thermal_multipliers']['dT_self_uniform']
    dT_drive = p0['dT_drive'] * rng.uniform(dlo, dhi); dT_charge = p0['dT_charge'] * rng.uniform(dlo, dhi)
    th = dict(p0, tau=tau, tau_fan=tau * p0['tau_fan'] / p0['tau'], k_sun=k_sun, k_shade=k_shade, dT_drive=dT_drive, dT_charge=dT_charge)
    if sysn == 'E2W':
        c = mc['E2W']
        duty = dict(km=rng.uniform(*c['km']), wh=rng.uniform(*c['Wh_per_km']),
                    h0=int(rng.integers(min(c['charge_start_hours']), max(c['charge_start_hours']) + 1)), pkw=rng.uniform(*c['charger_kW']))
    else:
        c = mc['E3W']
        opp = bool(rng.integers(0, 2))
        km = rng.uniform(*c['km_with_midday_charge']) if opp else rng.uniform(*c['km_without_midday_charge'])
        duty = dict(km=km, opp=opp, pkw=rng.uniform(*c['charger_kW']))
    return y0, th, duty
