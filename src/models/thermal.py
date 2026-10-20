"""Lumped pack thermal model (Eqs. 2-3 of the paper).

tau * dTb/dt = Teq - Tb,  Teq = Ta + k_sol * G + dT_self
integrated with the exact discrete solution over hourly steps:
Tb[n+1] = Teq[n] + (Tb[n] - Teq[n]) * exp(-dt / tau)
"""
import numpy as np


def pack_temperature(sc, Ta, G, mode, p, cooling):
    """Hourly pack temperature for one scenario.

    sc      : dict with keys sys, cool, park
    Ta, G   : hourly ambient temperature (degC) and irradiance (W/m2)
    mode    : hourly operating mode (0 parked, 1 driving, 2 charging/cycling)
    p       : thermal parameters of the system (configs/thermal.yaml -> systems)
    cooling : cooling definitions (configs/thermal.yaml -> cooling)
    """
    if sc['park'] == 'mixed':
        k = 0.5 * (p['k_sun'] + p['k_shade'])
    else:
        k = p['k_sun'] if sc['park'] == 'sun' else p['k_shade']
    selfh = np.where(mode == 1, p['dT_drive'], np.where(mode == 2, p['dT_charge'], 0.0))
    fan_on = (mode > 0) & (sc['cool'] == 'fan')
    tau = np.where(fan_on, p['tau_fan'], p['tau'])
    # a fan lowers the thermal resistance: the same factor reduces the solar and self-heating rise
    r = np.where(fan_on, p['tau_fan'] / p['tau'], 1.0)
    Teq = Ta + r * (k * G + selfh)
    a = np.exp(-1.0 / tau)
    set_point = cooling.get(sc['cool'], {}).get('set_point')
    op_limit = cooling.get(sc['cool'], {}).get('limit_in_operation')
    T = np.empty_like(Ta)
    t = Ta[0]
    for i in range(len(Ta)):
        t = Teq[i] + (t - Teq[i]) * a[i]
        if set_point is not None:
            t = min(t, set_point)                 # HVAC holds the container at or below the set point
        if op_limit is not None and mode[i] > 0:
            t = min(t, op_limit)                  # liquid cooling active only while driving/charging
        T[i] = t
    return T


def bess_temperature(Ta, G, mode, tau, k, dT, hvac=None, cap=None, hotspot=0.0):
    """Container temperature for the BESS stress cases.

    hvac    : set point (degC) or None
    cap     : maximum cooling below ambient (K) for an undersized HVAC unit, or None
    hotspot : offset added for the warmest rack (K)
    """
    a = np.exp(-1.0 / tau)
    Teq = Ta + k * G + np.where(mode > 0, dT, 0.0)
    T = np.empty_like(Ta)
    t = Ta[0]
    for i in range(len(Ta)):
        t = Teq[i] + (t - Teq[i]) * a
        if hvac is not None:
            t = min(t, hvac if cap is None else max(hvac, Ta[i] - cap))
        T[i] = t
    return T + hotspot
