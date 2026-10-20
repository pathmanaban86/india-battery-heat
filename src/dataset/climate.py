"""Hourly climate inputs from NASA POWER.

`build()` assembles the per-city CSV files downloaded by download_data.sh into one parquet file;
`load_city()` returns the arrays used by the models for one city.
Run as a module to build the processed file:  python -m src.dataset.climate
"""
import io
import pandas as pd
from src.utils.config import load, path

_CACHE = {}


def _read_power_csv(fp):
    text = open(fp).read()
    body = text.split('-END HEADER-', 1)[1]
    return pd.read_csv(io.StringIO(body.strip()))


def build():
    """Combine data/raw/power/<city>_<y0>_<y1>.csv into the processed parquet file."""
    dcfg, ccfg = load('data'), load('cities')
    frames = []
    for c in ccfg['cities']:
        parts = [_read_power_csv(path(dcfg['power']['raw_dir']) / f"{c['name']}_{a}_{b}.csv") for a, b in dcfg['power']['chunks']]
        df = pd.concat(parts, ignore_index=True)
        df.insert(0, 'city', c['name']); df.insert(1, 'lat', c['lat']); df.insert(2, 'lon', c['lon']); df.insert(3, 'climate', c['climate'])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).rename(columns={'YEAR': 'year', 'MO': 'month', 'DY': 'day', 'HR': 'hour', 'T2M': 'T2M_C',
                                                               'ALLSKY_SFC_SW_DWN': 'GHI_Wh_m2', 'WS2M': 'WS2M_m_s'})
    missing = int((df[['T2M_C', 'GHI_Wh_m2']] == -999).sum().sum())
    out = path(dcfg['processed']); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, compression='zstd')
    print(f'wrote {out}: {len(df)} rows, {df.city.nunique()} cities, missing values: {missing}')


def table():
    """Full processed climate table (cached)."""
    if 'all' not in _CACHE:
        _CACHE['all'] = pd.read_parquet(path(load('data')['processed']))
    return _CACHE['all']


def cities():
    """Cities used in the analysis (excluded cities removed), sorted by name."""
    ccfg = load('cities')
    return sorted(c['name'] for c in ccfg['cities'] if c['name'] not in ccfg['excluded'])


def load_city(city):
    """Arrays for one city: Ta (degC), G (W/m2, clipped at 0), day-of-week per hour, year per hour, climate group."""
    c = table()
    c = c[c.city == city].sort_values(['year', 'month', 'day', 'hour'])
    dates = pd.to_datetime(dict(year=c.year, month=c.month, day=c.day))
    return {'Ta': c.T2M_C.to_numpy(float), 'G': c.GHI_Wh_m2.to_numpy(float).clip(0),
            'dow': dates.dt.dayofweek.to_numpy(), 'year': c.year.to_numpy(), 'climate': c.climate.iloc[0]}


if __name__ == '__main__':
    build()
