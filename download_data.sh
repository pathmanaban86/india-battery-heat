#!/usr/bin/env bash
# Download all open data used in the paper and build the processed climate file.
#   NASA POWER hourly point data (T2M, ALLSKY_SFC_SW_DWN, WS2M), 2001-2025, 32 cities, local solar time
#   NOAA CPC Oceanic Nino Index (ONI)
#   Optional (--with-wu): data of Wu et al. (2026), Zenodo record 17173892, for the car comparison
# Usage: bash download_data.sh [--with-wu]
set -euo pipefail
cd "$(dirname "$0")"
RAW=data/raw; POWER=$RAW/power
mkdir -p "$POWER" data/processed

# city list and chunks from the YAML configs (single source of truth)
mapfile -t CITIES < <(python3 -c "
import yaml
for c in yaml.safe_load(open('configs/cities.yaml'))['cities']:
    print(c['name'], c['lat'], c['lon'])")
mapfile -t CHUNKS < <(python3 -c "
import yaml
for a, b in yaml.safe_load(open('configs/data.yaml'))['power']['chunks']:
    print(a, b)")

fetch () {  # url output  (retries with back-off; NASA POWER may rate-limit)
  local url="$1" out="$2" n=0
  until curl -fsS --max-time 300 -o "$out" "$url"; do
    n=$((n + 1)); [ $n -ge 5 ] && { echo "failed: $url" >&2; return 1; }
    sleep $((10 * n))
  done
}

for line in "${CITIES[@]}"; do
  read -r name lat lon <<< "$line"
  for ch in "${CHUNKS[@]}"; do
    read -r y0 y1 <<< "$ch"
    out="$POWER/${name}_${y0}_${y1}.csv"
    if [ -s "$out" ] && grep -q -- "-END HEADER-" "$out"; then continue; fi
    url="https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,ALLSKY_SFC_SW_DWN,WS2M&community=RE&latitude=${lat}&longitude=${lon}&start=${y0}0101&end=${y1}1231&time-standard=LST&format=CSV"
    echo "NASA POWER  ${name} ${y0}-${y1}"
    fetch "$url" "$out"
  done
done

echo "NOAA ONI"
fetch "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt" "$RAW/oni.txt"

if [ "${1:-}" = "--with-wu" ]; then
  echo "Wu et al. (2026) Zenodo data"
  fetch "https://zenodo.org/api/records/17173892/files/Code_Data_EV_Batt_Clim_Chang_main.zip/content" "$RAW/wu2026.zip"
  python3 -c "import zipfile; zipfile.ZipFile('$RAW/wu2026.zip').extractall('$RAW/wu2026')"
fi

echo "Building processed climate file"
python3 -m src.dataset.climate
python3 - <<'PY'
import hashlib
p = 'data/processed/india_hourly_2001_2025.parquet'
print('sha256', hashlib.sha256(open(p, 'rb').read()).hexdigest(), p)
PY
