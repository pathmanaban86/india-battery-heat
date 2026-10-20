#!/usr/bin/env python3
"""Replicate the results of the paper with one command.

    python run_reproducibility.py                 # all stages, all cities (several hours on 2 cores)
    python run_reproducibility.py --quick         # 2 cities, reduced Monte Carlo: smoke test (~10 min)
    python run_reproducibility.py --stages main enso --workers 8

Outputs are written to results/ (CSV per stage) and results/key_numbers.json, which is compared with
configs/expected_results.yaml (full run) or the 'quick' section of that file.
"""
import argparse
import json
import multiprocessing as mp
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent
STAGES = ['main', 'extra', 'ensemble', 'montecarlo', 'priors', 'bess', 'selfheat', 'erickshaw', 'enso']


def _pool_map(fn, items, workers):
    if workers <= 1:
        return [fn(x) for x in items]
    with mp.Pool(workers) as p:
        return p.map(fn, items)


def run(stages, quick, workers):
    from src import pipeline as P
    from src.dataset import climate
    from src.utils.config import load, path
    out = ROOT / 'results'; out.mkdir(exist_ok=True)
    cities = ['Bikaner', 'Bengaluru'] if quick else climate.cities()
    reps = ['Bikaner', 'Bengaluru'] if quick else load('cities')['representative']
    t0 = time.time()

    if 'main' in stages:
        res = _pool_map(P.main_city, cities, workers)
        pd.DataFrame([r for o in res for r in o[0]]).to_csv(out / 'main_lifetimes.csv', index=False)
        pd.DataFrame([r for o in res for r in o[1]]).to_csv(out / 'heat_year.csv', index=False)
        print(f'[main] {len(cities)} cities done ({time.time() - t0:.0f} s)')
    if 'extra' in stages:
        res = _pool_map(P.extra_city, cities, workers)
        for i, name in enumerate(['calendar_share', 'trajectories', 'temperature_representation', 'sensitivity_oat']):
            pd.DataFrame([r for o in res for r in o[i]]).to_csv(out / f'{name}.csv', index=False)
        print(f'[extra] done ({time.time() - t0:.0f} s)')
    if 'ensemble' in stages:
        res = _pool_map(P.ensemble_city, cities, workers)
        pd.DataFrame([r for o in res for r in o]).to_csv(out / 'ensemble.csv', index=False)
        print(f'[ensemble] done ({time.time() - t0:.0f} s)')
    if 'montecarlo' in stages:
        tasks = P.montecarlo_tasks(n_samples=4 if quick else None, cities=reps)
        res = _pool_map(P.montecarlo_task, tasks, workers)
        pd.DataFrame([r for o in res for r in o]).to_csv(out / 'montecarlo.csv', index=False)
        print(f'[montecarlo] done ({time.time() - t0:.0f} s)')
    if 'priors' in stages and (out / 'montecarlo.csv').exists():
        P.priors(pd.read_csv(out / 'montecarlo.csv')).to_csv(out / 'mc_priors.csv', index=False)
        print('[priors] done')
    if 'bess' in stages:
        res = _pool_map(P.bess_city, cities, workers)
        pd.DataFrame([r for o in res for r in o[0]]).to_csv(out / 'bess_stress.csv', index=False)
        pd.DataFrame([r for o in res for r in o[1]]).to_csv(out / 'bess_heat_year.csv', index=False)
        print(f'[bess] done ({time.time() - t0:.0f} s)')
    if 'selfheat' in stages:
        res = _pool_map(P.selfheat_city, reps, workers)
        pd.DataFrame([r for o in res for r in o]).to_csv(out / 'selfheat.csv', index=False)
        print(f'[selfheat] done ({time.time() - t0:.0f} s)')
    if 'erickshaw' in stages:
        P.erickshaw().to_csv(out / 'erickshaw.csv', index=False)
        print('[erickshaw] done')
    if 'enso' in stages and (out / 'heat_year.csv').exists():
        H = P.heat_year_anomalies(pd.read_csv(out / 'heat_year.csv'))
        H.to_csv(out / 'heat_year.csv', index=False)
        P.enso(H, path(load('data')['oni']['path'])).to_csv(out / 'enso.csv', index=False)
        print('[enso] done')
    return out


def key_numbers(out):
    """Key numbers of the paper, computed from the result files that exist."""
    K = {}
    sel = lambda d, **k: d.loc[np.logical_and.reduce([d[a] == b for a, b in k.items()])]
    med_t = lambda s: float(s.fillna(np.inf).median())
    f = out / 'main_lifetimes.csv'
    if f.exists():
        L = pd.read_csv(f)
        e2w = lambda chem, **k: sel(L, sys='E2W', chem=chem, cool=k.get('cool', 'passive'), park=k.get('park', 'sun'), cap=k.get('cap', 1.0))
        K['t80_E2W_NMC_sun_median'] = med_t(e2w('NMC').life80_yr)
        K['t80_E2W_LFP_sun_median'] = med_t(e2w('LFP').life80_yr)
        K['t80_E3W_NMC_sun_median'] = med_t(sel(L, sys='E3W', chem='NMC', cool='passive', park='sun').life80_yr)
        K['t80_E3W_LFP_sun_median'] = med_t(sel(L, sys='E3W', chem='LFPpris', cool='passive', park='sun').life80_yr)
        K['t80_BESS_passive_median'] = med_t(sel(L, sys='BESS', cool='passive').life80_yr)
        K['t80_E2W_NMC_sun_Bikaner'] = float(e2w('NMC').set_index('city').life80_yr.get('Bikaner', np.nan))
        K['Q10_E2W_NMC_sun_Bikaner'] = float(e2w('NMC').set_index('city').q_yr10.get('Bikaner', np.nan))
        K['Tb_p99_E2W_sun_Bikaner'] = float(e2w('NMC').set_index('city').Tb_p99.get('Bikaner', np.nan))
        base = e2w('NMC').set_index('city').q_yr10
        K['dQ10_E2W_NMC_cap80_median'] = float(((e2w('NMC', cap=0.8).set_index('city').q_yr10 - base) * 100).median())
        K['dQ10_E2W_NMC_shade_median'] = float(((e2w('NMC', park='shade').set_index('city').q_yr10 - base) * 100).median())
        K['dQ10_E2W_NMC_fan_median'] = float(((e2w('NMC', cool='fan').set_index('city').q_yr10 - base) * 100).median())
    f = out / 'calendar_share.csv'
    if f.exists():
        A = pd.read_csv(f); A['share'] = A.cal_loss10 / A.tot_loss10 * 100
        K['calendar_share_E2W_NMC_median'] = float(sel(A, sys='E2W', chem='NMC').share.median())
    f = out / 'heat_year.csv'
    if f.exists():
        H = pd.read_csv(f)
        if 'anom_pct' in H:
            g = sel(H, sys='E2W', chem='NMC').copy(); g['Ta_anom'] = g.groupby('city').Ta_mean.transform(lambda x: x - x.mean())
            K['heat_year_r_E2W_NMC'] = float(np.corrcoef(g.Ta_anom, g.anom_pct)[0, 1])
    f = out / 'enso.csv'
    if f.exists():
        E = pd.read_csv(f); r = sel(E, sys='E2W', chem='NMC', definition='DJF>=1.0').iloc[0]
        K['enso_strong_decaying_E2W_NMC_diff'] = float(r.diff_pp); K['enso_strong_decaying_E2W_NMC_p'] = float(r.p_perm)
        r = sel(E, sys='E2W', chem='NMC', definition='JJA>=0.5 (developing)').iloc[0]
        K['enso_developing_E2W_NMC_diff'] = float(r.diff_pp); K['enso_developing_E2W_NMC_p'] = float(r.p_perm)
    f = out / 'ensemble.csv'
    if f.exists():
        X = pd.read_csv(f); k = X[(X.climate != 'cold') & (X.sys == 'E2W') & (X.cool == 'passive') & (X.park == 'sun') & (X.cap == 1.0) & (X.chem == 'NMC')]
        m = k.groupby('model').t80.apply(lambda s: s.fillna(np.inf).median())
        K['ensemble_E2W_NMC_t80_min'] = float(m.min()); K['ensemble_E2W_NMC_t80_max'] = float(m.max())
    f = out / 'montecarlo.csv'
    if f.exists():
        Mc = pd.read_csv(f); m = Mc[(Mc.sys == 'E2W') & (Mc.chem == 'NMC')].pivot_table(index=['city', 'sample'], columns='variant', values='Q10')
        K['mc_E2W_NMC_cap80_median'] = float(((m.cap80 - m.base) * 100).median())
        K['mc_E2W_NMC_P_cap80_gt_shade'] = float((m.cap80 > m.shade).mean() * 100)
        K['mc_E2W_NMC_P_shade_gt_fan'] = float((m.shade > m.fan).mean() * 100)
    f = out / 'bess_stress.csv'
    if f.exists():
        B = pd.read_csv(f); nb = B[B.climate != 'cold']
        t = lambda case: float(nb[(nb.case == case) & (nb.model == 'Lfp_Gr_250AhPrismatic')].t80.fillna(np.inf).median())
        K['bess_reference_t80'] = t('reference passive (0.4C)'); K['bess_joint_0.6C_t80'] = t('poor airflow + undersized HVAC + hot spot (0.6C)')
    f = out / 'selfheat.csv'
    if f.exists():
        S = pd.read_csv(f); P = S[(S.variant == 'base') & (S.chem == 'NMC')].pivot_table(index='city', columns='selfheat', values='Q10')
        K['selfheat_8K_dQ10_E2W_NMC'] = float(((P['rise_8K_per_trip'] - P['reference']) * 100).median())
    f = out / 'erickshaw.csv'
    if f.exists():
        R = pd.read_csv(f); K['erickshaw_Q130_min_pct'] = float(R.Q_at_130FEC.min() * 100); K['erickshaw_Q130_max_pct'] = float(R.Q_at_130FEC.max() * 100)
    return K


def compare(K, quick):
    exp = yaml.safe_load(open(ROOT / 'configs' / 'expected_results.yaml'))
    ref = exp['quick'] if quick else exp['full']; tol = exp['tolerance']
    ok = True
    print('\n{:42s} {:>12s} {:>12s}  status'.format('quantity', 'expected', 'reproduced'))
    for k, v in ref.items():
        if k not in K:
            continue
        got = K[k]
        same = (np.isinf(v) and np.isinf(got)) or abs(got - v) <= max(tol['abs'], tol['rel'] * abs(v))
        ok &= bool(same)
        print('{:42s} {:>12.3f} {:>12.3f}  {}'.format(k, v, got, 'OK' if same else 'DIFF'))
    print('\nAll compared quantities reproduced.' if ok else '\nSome quantities differ; see above.')
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stages', nargs='+', default=STAGES, choices=STAGES)
    ap.add_argument('--quick', action='store_true', help='2 cities and 4 Monte Carlo samples (smoke test)')
    ap.add_argument('--workers', type=int, default=max(1, mp.cpu_count()))
    a = ap.parse_args()
    out = run(a.stages, a.quick, a.workers)
    K = key_numbers(out)
    json.dump(K, open(out / 'key_numbers.json', 'w'), indent=2)
    compare(K, a.quick)
