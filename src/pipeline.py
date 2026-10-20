"""Experiment stages of the paper. Each stage returns pandas DataFrames; run_reproducibility.py orchestrates them.

Stage                  Paper element
main                   868 lifetime runs + 5425 heat-year runs (Tables 11, 15; Figs. 3-9)
extra                  calendar share, capacity trajectories, temperature representation, one-at-a-time
                       thermal and cell-model sensitivity (Fig. 5, 6, 10b; Table 17; Supplementary S4)
ensemble               seven-model ensemble with validity-range shares (Table 12, Fig. 11a)
montecarlo             joint duty/thermal Monte Carlo (Table 13, Fig. 11b)
priors                 importance re-weighting to alternative priors (Table 14)
bess                   BESS stress cases incl. heat-year runs (Table 18)
selfheat               self-heating matched to measured module temperature rise (Table 9)
erickshaw              e-rickshaw life-cycle protocol with the LFP models (Table 10)
enso                   year-level El Nino permutation tests (Table 16)
"""
import itertools
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.config import load
from src.utils import metrics as M
from src.utils.stats import permutation_test, prior_weights, weighted_quantile
from src.dataset import climate, duty
from src.dataset.sampling import draw
from src.models.thermal import pack_temperature, bess_temperature
from src.models.degradation import simulate

TH = load('thermal'); DM = load('degradation_models'); SC = load('scenarios'); MCC = load('montecarlo')
BS = load('bess_stress'); EN = load('enso'); CITY = load('cities')
REF = DM['reference']


def scenarios():
    """Main scenario list in the published order (28 per city)."""
    out = []
    for s in ['E2W', 'E3W', 'BESS', 'CAR']:
        g = SC[s]
        for chem, cool, park, cap in itertools.product(g['chem'], g['cool'], g['park'], g['cap']):
            out.append(dict(sys=s, chem=chem, cool=cool, park=park, cap=cap))
    return out


def _temp(sc, c, mode, params=None):
    return pack_temperature(sc, c['Ta'], c['G'], mode, params or TH['systems'][sc['sys']], TH['cooling'])


# ------------------------------------------------------------------ main lifetime + heat-year runs
def main_city(city):
    c = climate.load_city(city)
    hy_keys = {tuple(k) for k in SC['heat_year']}
    rows, hy = [], []
    for sc in scenarios():
        soc, mode = duty.build_series(sc, c['dow'])
        T = _temp(sc, c, mode)
        r = dict(city=city, climate=c['climate'], **sc, Tb_mean=T.mean(), Tb_p99=np.percentile(T, 99),
                 Tb_hrs45_per_yr=(T > 45).sum() / 25, Ta_mean=c['Ta'].mean())
        r.update(M.lifetime_summary(simulate(REF[sc['chem']], T, soc), DM['end_of_life']))
        rows.append(r)
        if (sc['sys'], sc['chem'], sc['cool'], sc['park'], sc['cap']) in hy_keys:
            for y in sorted(set(c['year'])):
                m = c['year'] == y
                q = simulate(REF[sc['chem']], T[m], soc[m])['q']
                hy.append(dict(city=city, sys=sc['sys'], chem=sc['chem'], cool=sc['cool'], year=int(y), fade_1yr=1 - q[-1],
                               Tb_mean=T[m].mean(), Ta_mean=c['Ta'][m].mean(), Ta_hrs40=int((c['Ta'][m] > 40).sum())))
    return rows, hy


# ------------------------------------------------------------------ calendar share, trajectories, temperature representation, OAT
KEY = [dict(sys='E2W', chem='LFP', cool='passive', park='sun', cap=1.0), dict(sys='E2W', chem='NMC', cool='passive', park='sun', cap=1.0),
       dict(sys='E3W', chem='LFPpris', cool='passive', park='sun', cap=1.0), dict(sys='E3W', chem='NMC', cool='passive', park='sun', cap=1.0),
       dict(sys='BESS', chem='LFPpris', cool='passive', park='sun', cap=0.9), dict(sys='BESS', chem='LFPpris', cool='hvac30', park='sun', cap=0.9),
       dict(sys='CAR', chem='NMC', cool='liquid35', park='mixed', cap=0.9)]
ALT = {'NMC': ['Nca_Gr_Panasonic3Ah_Battery', 'NMC_Gr_75Ah_A'], 'LFP': ['Lfp_Gr_250AhPrismatic'], 'LFPpris': ['Lfp_Gr_SonyMurata3Ah_Battery']}


def extra_city(city):
    c = climate.load_city(city); share, traj, temprep, sens = [], [], [], []
    for sc in KEY:
        soc, mode = duty.build_series(sc, c['dow']); T = _temp(sc, c, mode)
        sim = simulate(REF[sc['chem']], T, soc)
        tot = 1 - M.q_at_years(sim['q'], sim['t_days'], 10); cal = 1 - M.q_at_years(sim['q_cal'], sim['t_days'], 10)
        share.append(dict(city=city, climate=c['climate'], sys=sc['sys'], chem=sc['chem'], cool=sc['cool'], cal_loss10=cal, tot_loss10=tot))
        for yr in range(26):
            traj.append(dict(city=city, sys=sc['sys'], chem=sc['chem'], cool=sc['cool'], year=yr, Q=M.q_at_years(sim['q'], sim['t_days'], yr)))
    if city in CITY['representative']:
        for sc in KEY[:4]:
            soc, mode = duty.build_series(sc, c['dow']); Tb = _temp(sc, c, mode)
            for lab, T in [('pack model (hourly)', Tb), ('ambient only (hourly)', c['Ta']), ('annual-mean ambient', np.full_like(c['Ta'], c['Ta'].mean()))]:
                s = simulate(REF[sc['chem']], T, soc)
                temprep.append(dict(city=city, sys=sc['sys'], chem=sc['chem'], method=lab, t80=M.time_to(s['q'], s['t_days'], 0.8), Q10=M.q_at_years(s['q'], s['t_days'], 10)))
            if sc['sys'] == 'E2W':
                p0 = TH['systems']['E2W']
                for lab, mod in [('k x0.5', dict(k_sun=p0['k_sun'] * .5, k_shade=p0['k_shade'] * .5)), ('k x1.5', dict(k_sun=p0['k_sun'] * 1.5, k_shade=p0['k_shade'] * 1.5)),
                                 ('tau x0.5', dict(tau=p0['tau'] * .5)), ('tau x2', dict(tau=p0['tau'] * 2)),
                                 ('self-heat x0', dict(dT_drive=0, dT_charge=0)), ('self-heat x2', dict(dT_drive=8, dT_charge=4))]:
                    T = _temp(sc, c, mode, {**p0, **mod}); s = simulate(REF[sc['chem']], T, soc)
                    sens.append(dict(city=city, sys='E2W', chem=sc['chem'], test='thermal', variant=lab, Q10=M.q_at_years(s['q'], s['t_days'], 10), t80=M.time_to(s['q'], s['t_days'], 0.8), Tb_mean=T.mean()))
            for m in ALT[sc['chem']]:
                s = simulate(m, Tb, soc)
                sens.append(dict(city=city, sys=sc['sys'], chem=sc['chem'], test='cell model', variant=m, Q10=M.q_at_years(s['q'], s['t_days'], 10), t80=M.time_to(s['q'], s['t_days'], 0.8), Tb_mean=Tb.mean()))
    return share, traj, temprep, sens


# ------------------------------------------------------------------ ensemble
def ensemble_city(city):
    c = climate.load_city(city); rows = []
    for s, cool, park, cap in SC['ensemble_scenarios']:
        sc = dict(sys=s, cool=cool, park=park, cap=cap)
        soc, mode = duty.build_series(sc, c['dow']); T = _temp(sc, c, mode)
        chems = [('LFP', DM['ensemble']['LFP'])] if s == 'BESS' else [('NMC', DM['ensemble']['NMC']), ('LFP', DM['ensemble']['LFP'])]
        for chem, models in chems:
            for m in models:
                sim = simulate(m, T, soc)
                rows.append(dict(city=city, climate=c['climate'], **sc, chem=chem, model=m, t80=M.time_to(sim['q'], sim['t_days'], 0.8),
                                 Q10=M.q_at_years(sim['q'], sim['t_days'], 10), Q25=float(sim['q'][-1]),
                                 env_out_pct=M.out_of_range_share(T, soc, DM['cycling_range_degC'][m]), Tb_min=T.min(), Tb_max=T.max()))
    return rows


# ------------------------------------------------------------------ Monte Carlo
def montecarlo_task(args):
    city, sysn, n_samples, seed = args
    c = climate.load_city(city); rng = np.random.default_rng(seed); rows = []
    p0 = TH['systems'][sysn]; H = MCC['horizon_years']
    for k in range(n_samples):
        y0, th, dp = draw(rng, sysn, p0, MCC)
        m10 = (c['year'] >= y0) & (c['year'] < y0 + H)
        Ta, G, dow = c['Ta'][m10], c['G'][m10], c['dow'][m10]
        variants = {'base': dict(cool='passive', park='sun', cap=1.0), 'shade': dict(cool='passive', park='shade', cap=1.0), 'fan': dict(cool='fan', park='sun', cap=1.0)}
        if sysn == 'E2W':
            variants['cap80'] = dict(cool='passive', park='sun', cap=0.8)
        for vn, v in variants.items():
            sc = dict(sys=sysn, **v); days = dow[::24]
            if sysn == 'E2W':
                tpl = {d: duty.e2w_param(v['cap'], d >= 5, dp['km'], dp['wh'], dp['h0'], dp['pkw']) for d in range(7)}
            else:
                tpl = {d: duty.e3w_param(1.0, d == 6, dp['km'], dp['opp'], dp['pkw']) for d in range(7)}
            soc = np.concatenate([tpl[d][0] for d in days]); mode = np.concatenate([tpl[d][1] for d in days])
            T = pack_temperature(sc, Ta, G, mode, th, TH['cooling'])
            for chem, m in MCC['models'][sysn].items():
                q = simulate(m, T, soc)['q']
                rows.append(dict(city=city, sys=sysn, sample=k, variant=vn, chem=chem, Q10=float(q[-1]), start=y0,
                                 tau=th['tau'], k_sun=th['k_sun'], dT_drive=th['dT_drive'], **dp))
    return rows


def montecarlo_tasks(n_samples=None, cities=None):
    n = n_samples or MCC['samples_per_city']
    full = CITY['representative']                 # seeds follow the position in the full list, also in quick mode
    reps = cities or full
    return [(c, s, n, MCC['seeds']['base'] + full.index(c) * 10 + j) for c in reps for j, s in enumerate(['E2W', 'E3W'])]


def priors(mc):
    rows = []
    for s, vars_ in [('E2W', ['cap80', 'shade', 'fan']), ('E3W', ['shade', 'fan'])]:
        m = mc[mc.sys == s]
        P = m.pivot_table(index=['city', 'sample', 'chem'], columns='variant', values='Q10')
        par = m[m.variant == 'base'].drop_duplicates(['city', 'sample', 'chem']).set_index(['city', 'sample', 'chem'])[['km', 'tau', 'k_sun', 'dT_drive']]
        P = P.join(par)
        for prior in ['uniform (reference)', 'lognormal distance', 'triangular distance', 'peaked thermal', 'all peaked']:
            for chem in ['NMC', 'LFP']:
                d = P.xs(chem, level='chem'); w = prior_weights(d, s, prior, MCC, TH['systems'][s])
                r = dict(sys=s, chem=chem, prior=prior, ESS=float(w.sum() ** 2 / (w ** 2).sum()))
                for cc in vars_:
                    r['d_' + cc] = weighted_quantile(((d[cc] - d['base']) * 100).to_numpy(), w, 0.5)
                r['P_shade_gt_fan'] = float(np.sum(w * (d.shade > d.fan).to_numpy()) / w.sum() * 100)
                if s == 'E2W':
                    r['P_cap_gt_shade'] = float(np.sum(w * (d.cap80 > d.shade).to_numpy()) / w.sum() * 100)
                rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ BESS stress cases
def bess_city(city):
    c = climate.load_city(city); rows, hy = [], []
    for name, p in BS['cases'].items():
        s1, m1 = duty.bess_day_crate(p['crate']); nd = len(c['Ta']) // 24
        soc, mode = np.tile(s1, nd), np.tile(m1, nd)
        T = bess_temperature(c['Ta'], c['G'], mode, p['tau'], p['k'], p['dT'], p['hvac'], p['cap'], p['hs'])
        for m in DM['ensemble']['LFP']:
            sim = simulate(m, T, soc)
            rows.append(dict(city=city, climate=c['climate'], case=name, model=m, t80=M.time_to(sim['q'], sim['t_days'], 0.8),
                             Q10=M.q_at_years(sim['q'], sim['t_days'], 10), Tb_mean=T.mean(), Tb_p99=np.percentile(T, 99),
                             env_out_pct=M.out_of_range_share(T, soc, DM['cycling_range_degC'][m])))
        if name in BS['heat_year_cases']:
            for y in sorted(set(c['year'])):
                mm = c['year'] == y; q = simulate('Lfp_Gr_250AhPrismatic', T[mm], soc[mm])['q']
                hy.append(dict(city=city, case=name, year=int(y), fade_1yr=1 - q[-1], Ta_mean=c['Ta'][mm].mean()))
    return rows, hy


# ------------------------------------------------------------------ self-heating check
def selfheat_city(city):
    c = climate.load_city(city); rows = []; p0 = TH['systems']['E2W']
    for lab, (dd, dc) in MCC['self_heating_checks'].items():
        params = dict(p0, dT_drive=dd, dT_charge=dc)
        for v, sc in {'base': dict(cool='passive', park='sun', cap=1.0), 'shade': dict(cool='passive', park='shade', cap=1.0),
                      'fan': dict(cool='fan', park='sun', cap=1.0), 'cap80': dict(cool='passive', park='sun', cap=0.8)}.items():
            sc = dict(sys='E2W', **sc); soc, mode = duty.build_series(sc, c['dow']); T = _temp(sc, c, mode, params)
            for chem, m in [('NMC', REF['NMC']), ('LFP', REF['LFP'])]:
                sim = simulate(m, T, soc)
                rows.append(dict(city=city, selfheat=lab, variant=v, chem=chem, Q10=M.q_at_years(sim['q'], sim['t_days'], 10),
                                 t80=M.time_to(sim['q'], sim['t_days'], 0.8), Tb_mean=T.mean(), Tb_p99=np.percentile(T, 99)))
    return rows


# ------------------------------------------------------------------ e-rickshaw protocol
def erickshaw():
    p = EN['erickshaw_protocol']; rows = []
    for dod in p['dods']:
        soc = duty.erickshaw_protocol(dod, p['days']); T = np.full(len(soc), 25.0); efc_per_day = p['cycles_per_day'] * dod
        for m in DM['ensemble']['LFP']:
            sim = simulate(m, T, soc); efc = sim['t_days'] * efc_per_day
            q130 = float(np.interp(p['fec_eol'], efc, sim['q']))
            rows.append(dict(model=m, dod=dod, Q_at_130FEC=q130, Q_at_end=float(sim['q'][-1]), FEC_end=float(efc[-1]),
                             loss_per_FEC_pct=(1 - q130) / p['fec_eol'] * 100))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ El Nino (year-level permutation tests)
def heat_year_anomalies(H):
    H = H.copy()
    H['anom_pct'] = H.groupby(['city', 'sys', 'chem', 'cool']).fade_1yr.transform(M.anomaly_pct)
    H['months_equiv'] = H.anom_pct / 100 * 12
    return H


def enso(H, oni_path):
    oni = pd.read_csv(oni_path, sep=r'\s+')
    years = np.arange(int(H.year.min()), int(H.year.max()) + 1)
    seas = {s: oni[oni.SEAS == s].set_index('YR').ANOM for s in oni.SEAS.unique()}
    ann = oni.groupby('YR').ANOM.mean()
    rng = np.random.default_rng(EN['seed']); rows = []
    for (s, ch, co), g in H.groupby(['sys', 'chem', 'cool']):
        y = g.groupby('year').anom_pct.mean().reindex(years).to_numpy()
        for name, d in EN['definitions'].items():
            series = ann if d['season'] == 'annual' else seas[d['season']]
            idx = (series >= d['threshold']).reindex(years).fillna(False).to_numpy()
            obs, p = permutation_test(y, idx, rng, EN['random_permutations'], EN['exact_enumeration_max_events'])
            rows.append(dict(sys=s, chem=ch, cool=co, definition=name, n_events=int(idx.sum()), events=','.join(map(str, years[idx])), diff_pp=obs, p_perm=p))
        rho, pr = stats.spearmanr(y, seas['JJA'].reindex(years))
        rows.append(dict(sys=s, chem=ch, cool=co, definition='Spearman vs JJA ONI', n_events=len(years), events='', diff_pp=rho, p_perm=pr))
    return pd.DataFrame(rows)
