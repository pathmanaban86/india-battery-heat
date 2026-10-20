"""Statistical tests: year-level permutation tests (El Nino) and importance re-weighting (Monte Carlo priors)."""
import itertools
import numpy as np
from scipy import stats


def permutation_test(x, idx, rng, n_random=20000, exact_max=5):
    """One-sided test of mean(x[idx]) - mean(x[~idx]) > 0.

    Exact enumeration when the number of events <= exact_max, otherwise `n_random` random permutations drawn
    from the shared generator `rng` (shared across calls, as in the published analysis).
    """
    k = int(idx.sum()); obs = x[idx].mean() - x[~idx].mean()
    if k <= exact_max:
        d = np.array([x[list(c)].mean() - np.delete(x, list(c)).mean() for c in itertools.combinations(range(len(x)), k)])
    else:
        d = np.array([(lambda p: x[p[:k]].mean() - x[p[k:]].mean())(rng.permutation(len(x))) for _ in range(n_random)])
    return obs, (np.sum(d >= obs) + 1) / (len(d) + 1)


def weighted_quantile(x, w, q):
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w) / w.sum()
    return float(np.interp(q, c, x))


def prior_weights(d, sysn, prior, cfg, ref):
    """Importance weights to re-weight uniform Monte Carlo samples to an alternative prior.

    d   : DataFrame with columns km, tau, k_sun, dT_drive (one row per sample)
    cfg : configs/montecarlo.yaml
    ref : reference thermal parameters of the vehicle (configs/thermal.yaml)
    """
    w = np.ones(len(d))
    lo, hi = (cfg['E2W']['km'] if sysn == 'E2W' else cfg['E3W']['km_with_midday_charge'])
    ap = cfg['alternative_priors']
    if prior in ('lognormal distance', 'all peaked'):
        med = ap['lognormal_distance']['median_km'][sysn]
        w *= stats.lognorm(s=ap['lognormal_distance']['sigma_log'], scale=med).pdf(d.km) / (1 / (hi - lo))
    if prior == 'triangular distance':
        mode = ap['triangular_distance']['mode_km'][sysn]
        w *= stats.triang(c=(mode - lo) / (hi - lo), loc=lo, scale=hi - lo).pdf(d.km) / (1 / (hi - lo))
    if prior in ('peaked thermal', 'all peaked'):
        pt = ap['peaked_thermal']
        w *= stats.norm(0, pt['tau_lognormal_sigma']).pdf(np.log(d.tau / ref['tau'])) / (1 / (np.log(2) - np.log(.5)))
        w *= stats.triang(c=0.5, loc=0.5, scale=1.0).pdf(d.k_sun / ref['k_sun']) / 1.0
        w *= stats.triang(c=0.5, loc=0, scale=2).pdf(d.dT_drive / ref['dT_drive']) / 0.5
    return w
