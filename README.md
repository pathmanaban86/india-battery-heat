# Climate-resolved ageing of Li-ion batteries in Indian two- and three-wheelers

Code and configuration to reproduce the results of the manuscript:

> *Charging habits, parking or heat? A climate-resolved scenario analysis of lithium-ion battery ageing in Indian two- and three-wheelers across 31 cities, with grid storage as a contrast case* (submitted to the Journal of Energy Storage).

The study couples 25 years (2001–2025) of hourly NASA POWER climate data for 31 Indian cities with a lumped pack thermal model and NREL BLAST-Lite semi-empirical degradation models. All inputs are open data; no proprietary data are used.

## Repository structure

```
├── README.md               <- this file
├── requirements.txt        <- pinned dependencies (BLAST-Lite pinned to a git commit)
├── download_data.sh        <- downloads NASA POWER and NOAA ONI data, builds the processed climate file
├── run_reproducibility.py  <- single script that runs all experiments and checks the key numbers
├── configs/                <- YAML files with every parameter used in the paper
│   ├── cities.yaml             31 study cities (+ Shimla, excluded), coordinates, climate groups
│   ├── data.yaml               data URLs, local paths, download chunks
│   ├── duty_profiles.yaml      E2W, E3W, BESS and car duty profiles (Table 3)
│   ├── thermal.yaml            lumped thermal model parameters (Table 4)
│   ├── degradation_models.yaml reference and ensemble cell models, tested temperature ranges (Tables 5-6)
│   ├── scenarios.yaml          scenario matrix (Table 7), heat-year and ensemble subsets
│   ├── montecarlo.yaml         Monte Carlo ranges, seeds, alternative priors, self-heating checks (Tables 8, 9, 14)
│   ├── bess_stress.yaml        BESS stress cases (Table 18)
│   ├── enso.yaml               El Nino definitions and permutation settings; e-rickshaw protocol (Tables 10, 16)
│   └── expected_results.yaml   key numbers of the paper used for the automatic check
└── src/
    ├── models/
    │   ├── thermal.py          lumped pack thermal model (Eqs. 2-3) and BESS container model
    │   └── degradation.py      BLAST-Lite wrapper, compatibility shims and numerical guard
    ├── dataset/
    │   ├── climate.py          builds and loads the hourly climate data
    │   ├── duty.py             hourly SOC/mode templates (reference and parametric)
    │   └── sampling.py         Monte Carlo scenario sampling (fixed random-draw order)
    ├── utils/
    │   ├── config.py           YAML loading and paths
    │   ├── metrics.py          t80, Q10, calendar share, out-of-range share, anomalies
    │   └── stats.py            permutation tests, importance weights, weighted quantiles
    └── pipeline.py             experiment stages (one function per analysis)
```

This is a physics-based simulation study, so `src/models` contains the thermal and degradation models rather than a learned architecture, and the role of data augmentation is taken by the scenario sampling in `src/dataset/sampling.py`.

## 1. Setup

Tested with Python 3.13.16 on Linux (2 CPU cores, 8 GB RAM).

```bash
git clone https://github.com/pathmanaban86/india-battery-heat.git
cd india-battery-heat
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps "blast-lite @ git+https://github.com/NREL/BLAST-Lite.git@b093495b47dc40dd96dba865d91f553619501e94"
```

BLAST-Lite is installed without its dependencies because its optional NSRDB weather downloader needs `h5pyd`, which this study does not use. `src/models/degradation.py` registers a stub for it and adds `np.trapz` for NumPy 2.

## 2. Download the data

```bash
bash download_data.sh            # NASA POWER (32 cities x 5 requests) + NOAA ONI, then builds the parquet file
bash download_data.sh --with-wu  # additionally downloads Wu et al. (2026) data for the car comparison
```

The script is idempotent: files that are already complete are skipped. The processed file is written to `data/processed/india_hourly_2001_2025.parquet` (7,012,608 rows; 219,144 hours per city; no missing values) and its SHA-256 checksum is printed.

Expected checksum of the processed file (NASA POWER data retrieved on 20 October 2026):
`6025799818ba8b34e57f035f46c40f7835f91bd523b4f93b79704e97a6958531`.
NASA POWER and NOAA may revise past values. The ONI file used in the paper is provided in `snapshots/oni_2026-10-20.txt`; to use it instead of the downloaded file, run `cp snapshots/oni_2026-10-20.txt data/raw/oni.txt` before `run_reproducibility.py`.

Sources:
- NASA POWER hourly API, https://power.larc.nasa.gov (MERRA-2 temperature, CERES SYN1deg irradiance)
- NOAA CPC Oceanic Niño Index, https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
- Wu et al. (2026), Nature Climate Change 16, 460–467, data: https://doi.org/10.5281/zenodo.17173892

## 3. Reproduce the results

```bash
python run_reproducibility.py --quick        # smoke test: 2 cities, 4 Monte Carlo samples
python run_reproducibility.py                # full replication (all stages, all cities)
python run_reproducibility.py --stages main enso --workers 8   # selected stages
```

Approximate run times on 2 cores: `main` 30 min, `extra` 40 min, `ensemble` 80 min, `montecarlo` 35 min, `bess` 15 min, `selfheat` 5 min, `erickshaw` < 1 min, `enso` 2 min. Run time scales with the number of workers.

Each stage writes CSV files to `results/`. At the end, `results/key_numbers.json` is computed and compared with `configs/expected_results.yaml`; the script prints `OK` or `DIFF` for each quantity. In `--quick` mode only quantities that do not depend on the full set of cities are compared.

| Stage | Output file(s) | Paper element |
|---|---|---|
| `main` | `main_lifetimes.csv`, `heat_year.csv` | Tables 11, 15; Figs. 3–9 |
| `extra` | `calendar_share.csv`, `trajectories.csv`, `temperature_representation.csv`, `sensitivity_oat.csv` | Figs. 5, 6, 10b; Table 17; Suppl. S4 |
| `ensemble` | `ensemble.csv` | Table 12; Fig. 11a |
| `montecarlo`, `priors` | `montecarlo.csv`, `mc_priors.csv` | Tables 13, 14; Fig. 11b |
| `bess` | `bess_stress.csv`, `bess_heat_year.csv` | Table 18 |
| `selfheat` | `selfheat.csv` | Table 9 |
| `erickshaw` | `erickshaw.csv` | Table 10 |
| `enso` | `enso.csv` (+ anomalies added to `heat_year.csv`) | Table 16 |

## 4. Determinism

All simulations are deterministic. The Monte Carlo analysis uses `numpy.random.default_rng` with the seeds in `configs/montecarlo.yaml` (seed = 1000 + 10 × city index + vehicle index), and the random draws are taken in a fixed order (`src/dataset/sampling.py`). The El Niño permutation tests use one generator with seed 1, shared across tests in a fixed order, as in the published analysis. Results reproduce to the printed precision when the pinned versions are used; other NumPy/SciPy versions may change the last digits.

## 5. Notes and limitations

- The thermal parameters are assumed, not calibrated (Section 2.4 of the paper). A calibration protocol and fitting script are given in Supplementary Material S5.
- Some cycling occurs outside the tested temperature range of the cell models; `ensemble.csv` reports the share for every model, city and scenario (`env_out_pct`).
- Results for the cold cities (Srinagar, Leh) with LFP models are outside the tested cycling range and are not interpreted in the paper.
- NASA POWER values are grid-cell averages (about 50 km); urban heat islands are not represented.

## Citation and licence

Please cite the paper if you use this code. Code: MIT licence (see `LICENSE`). Data remain subject to the terms of their providers (NASA POWER, NOAA, Zenodo record of Wu et al.).
