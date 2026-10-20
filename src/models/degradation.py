"""Wrapper around NREL BLAST-Lite semi-empirical degradation models.

Three compatibility measures are applied at import time:
1. BLAST-Lite imports `h5pyd` only for its optional NSRDB weather download, which is not used here;
   a stub module is registered so that the package imports without that dependency.
2. NumPy >= 2 removed `np.trapz`; it is aliased to `np.trapezoid`.
3. Numerical guard (paper, Section 2.5): when a cold period lowers the stress-dependent asymptote of a
   sigmoid state below the loss already accumulated, the inverse used by BLAST-Lite is undefined.
   Capacity loss cannot reverse, so the increment is set to zero.
"""
import sys
import types
import numpy as np

sys.modules.setdefault('h5pyd', types.ModuleType('h5pyd'))
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

from blast import models as _models  # noqa: E402
from blast.models.degradation_model import BatteryDegradationModel as _BDM  # noqa: E402

_orig_sigmoid = _BDM._update_sigmoid_state


def _guarded_sigmoid(y0, dx, y_inf, k, p):
    if y0 != 0 and dx != 0 and y0 >= y_inf:
        return 0.0
    v = _orig_sigmoid(y0, dx, y_inf, k, p)
    return 0.0 if not np.isfinite(v) else v


_BDM._update_sigmoid_state = staticmethod(_guarded_sigmoid)


def simulate(model_name, T, soc):
    """Run one BLAST-Lite model on hourly pack temperature T (degC) and SOC (0-1).

    Returns a dict with time in days `t_days`, remaining capacity `q`, the calendar-only capacity `q_cal`
    (cycling loss = (1 - q) - (1 - q_cal)) and the cell object.
    """
    cell = getattr(_models, model_name)()
    cell.simulate_battery_life({'Time_s': np.arange(len(T)) * 3600.0, 'SOC': soc, 'Temperature_C': T})
    o = cell.outputs
    q = np.asarray(o['q'])
    q_cal = np.asarray(o['q_t'] if 'q_t' in o else o['q_LLI_t'])
    return {'t_days': np.asarray(cell.stressors['t_days']), 'q': q, 'q_cal': q_cal, 'cell': cell}
