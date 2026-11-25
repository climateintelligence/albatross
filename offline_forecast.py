#!/usr/bin/env python
"""
Offline forecast with an Albatross operational model, without the WPS service
and without the albatross package.

Only final_model_<phase>.npz is needed, from the model/ folder of the run
archive. The predictor field is downloaded from the same NOAA PSL files used by the
service (SST: ERSST v6; SLP and Z500: NCEP/NCAR Reanalysis 1) and stored in --data-dir
(default: $ALBATROSS_DATA_DIR or ~/data, the same folder used by the service). If the
file already exists but does not cover the requested months, it is downloaded again.

Example (Java, OND 2011, M=1):
    python offline_forecast.py --model model/final_model_allyears.npz \
        --predictor sst --target-months 10,11,12 --year 2011 --check forecast.csv

With more than one phase (M>1), pass the model folder, phase_thresholds.json and the
climate index file: the script averages the index over the predictor months, selects
the phase and loads the model of that phase:
    python offline_forecast.py --model-dir model \
        --thresholds model/phase_thresholds.json --index "NINO 3.4.txt" \
        --predictor sst --target-months 10,11,12 --year 2011

Dependencies: numpy, netCDF4 (the download uses only the standard library).
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

LAG = 3  # the predictor is the mean of the 3 months preceding the target season

# Same files used by the service: (file name, URL, NetCDF variable, pressure level)
SOURCES = {
    "sst": ("sst.mnmean.v6.nc",
            "https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v6/sst.mnmean.nc",
            "sst", None),
    "slp": ("slp.mon.mean.nc",
            "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/surface/slp.mon.mean.nc",
            "slp", None),
    "z500": ("hgt.mon.mean.nc",
             "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/pressure/hgt.mon.mean.nc",
             "hgt", 500.0),
}


# -----------------------------------------------------------------------------
# 1. Predictor months
# -----------------------------------------------------------------------------
def predictor_months(target_months, year):
    """(year, month index) of the LAG months preceding the first month of the target season.
    E.g. target 10,11,12 of 2011 -> 2011-07, 2011-08, 2011-09;
         target 1,2,3 of 2011    -> 2010-10, 2010-11, 2010-12."""
    first = year * 12 + (target_months[0] - 1)          # absolute month
    return [divmod(first - LAG + k, 12) for k in range(LAG)]


def _ym(ab):
    """(year, 0-based month) -> (year, 1-based month)."""
    y, m0 = ab
    return y, m0 + 1


# -----------------------------------------------------------------------------
# 2. Predictor field: download, 3-month mean, increasing latitude (as in the service)
# -----------------------------------------------------------------------------
def download(url, dest):
    """Download url to dest through a temporary file, and check that it is a valid NetCDF."""
    import tempfile
    import urllib.request
    import netCDF4
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    print(f"Downloading {url} -> {dest}")
    try:
        with os.fdopen(fd, "wb") as f, urllib.request.urlopen(url, timeout=120) as r:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 2**20:.0f} / {total / 2**20:.0f} MB", end="", flush=True)
        print()
        with netCDF4.Dataset(tmp):
            pass
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def last_month(path):
    """(year, month) of the last time step in a monthly NetCDF file."""
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        t = ds["time"]
        d = netCDF4.num2date(t[-1], t.units, getattr(t, "calendar", "standard"))
    return d.year, d.month


def ensure_field_file(predictor, months_abs, data_dir, path=None, url=None):
    """Path of the predictor NetCDF; downloads it if missing or if it does not cover the requested months."""
    name, default_url, _, _ = SOURCES[predictor]
    url = url or default_url
    path = Path(os.path.expanduser(path)) if path else Path(os.path.expanduser(data_dir)) / name
    need = _ym(months_abs[-1])
    if not path.is_file():
        download(url, path)
    elif last_month(path) < need:
        y, m = last_month(path)
        print(f"{path} ends in {y}-{m:02d}, data up to {need[0]}-{need[1]:02d} are needed: updating")
        download(url, path)
    if last_month(path) < need:
        y, m = last_month(path)
        msg = (f"Predictor {predictor} is not yet available for {need[0]}-{need[1]:02d} "
               f"(the NOAA file ends in {y}-{m:02d}).")
        if predictor in ("slp", "z500"):
            msg += " NCEP/NCAR Reanalysis 1 is no longer produced after March 2026."
        raise SystemExit(msg)
    return path


def load_field(path, var, months_abs, level=None):
    """Mean of the predictor field over the given months, with latitude in increasing order."""
    import netCDF4
    with netCDF4.Dataset(os.path.expanduser(path)) as ds:
        t = ds["time"]
        dates = netCDF4.num2date(t[:], t.units, getattr(t, "calendar", "standard"))
        index = {(d.year, d.month): k for k, d in enumerate(dates)}
        wanted = [_ym(a) for a in months_abs]
        missing = [w for w in wanted if w not in index]
        if missing:
            raise ValueError(f"{path}: months not available {missing} "
                             f"(the file ends in {dates[-1].year}-{dates[-1].month:02d})")
        v = ds[var]
        slabs = []
        for w in wanted:
            k = index[w]
            if level is not None:
                levname = next(n for n in v.dimensions if n in ("level", "lev", "plev", "isobaric"))
                lv = np.asarray(ds[levname][:], dtype=float)
                kl = int(np.argmin(np.abs(lv - level)))
                idx = [k if d == "time" else (kl if d == levname else slice(None)) for d in v.dimensions]
                a = v[tuple(idx)]
            else:
                a = v[k, ...]
            slabs.append(np.squeeze(np.ma.filled(a.astype(float), np.nan)))
        latname = next(n for n in ("lat", "latitude", "y") if n in ds.variables)
        lat = np.asarray(ds[latname][:], dtype=float)
    field = np.mean(slabs, axis=0)
    if lat[0] > lat[-1]:                 # NOAA files: decreasing latitude -> increasing
        field, lat = field[::-1, :], lat[::-1]
    return field, lat


# -----------------------------------------------------------------------------
# 3. Phase (M>1 only)
# -----------------------------------------------------------------------------
def index_value(path, months_abs):
    """Mean of the climate index over the predictor months. Same format as the target:
    line 1 description, line 2 first and last year, then 12 values per year."""
    with open(path) as f:
        f.readline()
        first_year = int(f.readline().split()[0])
    vals = np.loadtxt(path, skiprows=2).ravel()
    vals[vals <= -99.0] = np.nan
    w = []
    for a in months_abs:
        y, m = _ym(a)
        k = (y - first_year) * 12 + (m - 1)
        w.append(vals[k] if 0 <= k < vals.size else np.nan)
    if not np.all(np.isfinite(w)):
        raise ValueError(f"Climate index not available for all predictor months: {w}")
    return float(np.mean(w))


def pick_phase(x, meta):
    """First phase with x <= threshold, otherwise the last one (as in the service)."""
    for thr, lab in zip(meta["thresholds"], meta["labels"]):
        if x <= float(thr):
            return lab
    return meta["labels"][-1]


# -----------------------------------------------------------------------------
# 4. Model application: same steps as the service
# -----------------------------------------------------------------------------
def apply_model(npz_path, field, year):
    p = np.load(npz_path, allow_pickle=False)
    lat = p["lat"]
    if field.shape != (lat.size, p["lon"].size):
        raise ValueError(f"Field grid {field.shape} differs from the model grid "
                         f"{(lat.size, p['lon'].size)}")
    year = float(year)

    # a) anomaly with respect to the climatology of the calibration period
    x = field - p["clim_2d"]

    # b) trends: target trend extrapolated to the year, and trend of each predictor cell
    if bool(p["detrend"]):
        m, s, tb = p["trend_y"]
        trend = float(m + s * (year - tb))
        x = x - (p["trend_X_mean"] + p["trend_X_slope"] * (year - float(p["trend_X_tbar"])))
    else:
        trend = 0.0

    if not bool(p["has_model"]):         # no cell retained in calibration
        yf = float(p["y_fallback"])
        return dict(forecast=trend + yf, trend_component=trend,
                    teleconnection_component=yf, PC1=np.nan, n_pix=0, fallback=True)

    # c) cos(latitude) weighting, retained cells, NaN imputation, centering
    x = (x * np.cos(np.radians(lat))[:, None]).reshape(-1)[p["mask_idx"]]
    nan = np.isnan(x)
    if nan.any():
        fill = p["col_means"] if p["col_means"].size == x.size else p["x_mean"]
        x[nan] = fill[nan]
    pcs = (x - p["x_mean"]) @ p["eofs"]

    # d) regression on PC1
    tele = float(pcs @ p["coef"] + float(p["intercept"]))
    return dict(forecast=trend + tele, trend_component=trend, teleconnection_component=tele,
                PC1=float(p["pc1_sign"] * pcs[0]), n_pix=int(np.sum(p["mask_idx"])),
                fallback=False)


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="final_model_<phase>.npz (M=1, or phase already known)")
    g.add_argument("--model-dir", help="folder containing the final_model_*.npz files (M>1)")
    ap.add_argument("--thresholds", help="phase_thresholds.json (with --model-dir)")
    ap.add_argument("--index", help="climate index text file (with --model-dir)")
    ap.add_argument("--predictor", required=True, choices=sorted(SOURCES),
                    help="predictor field used in the run (training_config.json: glo_var_name)")
    ap.add_argument("--data-dir", default=os.environ.get("ALBATROSS_DATA_DIR", "~/data"),
                    help="folder where the NOAA file is stored (default: $ALBATROSS_DATA_DIR or ~/data)")
    ap.add_argument("--field", help="existing NetCDF file to use instead of the one in --data-dir")
    ap.add_argument("--url", help="alternative download URL (e.g. a mirror)")
    ap.add_argument("--target-months", required=True, help="e.g. 10,11,12 (up to 15 = March of the following year)")
    ap.add_argument("--year", type=int, required=True, help="year of the target season")
    ap.add_argument("--check", help="forecast.csv returned by the service, for comparison")
    a = ap.parse_args()

    months = [int(m) for m in a.target_months.split(",")]
    pm = predictor_months(months, a.year)
    print("Predictor months:", ", ".join(f"{y}-{m:02d}" for y, m in map(_ym, pm)))

    if a.model:
        npz = Path(a.model)
    else:
        if not (a.thresholds and a.index):
            ap.error("--model-dir requires --thresholds and --index")
        meta = json.loads(Path(a.thresholds).read_text())
        idx_val = index_value(a.index, pm)
        phase = pick_phase(idx_val, meta)
        npz = Path(a.model_dir) / f"final_model_{phase}.npz"
        print(f"Index over the predictor months: {idx_val:.3f} -> phase '{phase}'")

    path = ensure_field_file(a.predictor, pm, a.data_dir, path=a.field, url=a.url)
    _, _, var, level = SOURCES[a.predictor]
    field, _ = load_field(path, var, pm, level=level)
    out = apply_model(npz, field, a.year)

    print(f"Model: {npz}")
    print(f"Forecast {a.year} (months {a.target_months}): {out['forecast']:.3f}")
    print(f"  trend component:          {out['trend_component']:.3f}")
    print(f"  teleconnection component: {out['teleconnection_component']:.3f}")
    print(f"  PC1: {out['PC1']:.3f}   retained cells: {out['n_pix']}   fallback: {out['fallback']}")

    if a.check:
        check = Path(a.check)
        if not check.is_file():
            # by default forecast.csv is in the run folder, above model/
            alt = npz.resolve().parent.parent / check.name
            if not alt.is_file():
                raise SystemExit(f"{a.check} not found (also looked in {alt})")
            check = alt
        print(f"Comparing with {check}")
        with open(check) as f:
            row = next(csv.DictReader(f))
        ref = float(row["forecast"])
        ok = np.isclose(out["forecast"], ref, rtol=0, atol=1e-6)
        print(f"Service: {ref:.3f}  ->  {'MATCH' if ok else 'MISMATCH'} "
              f"(difference {out['forecast'] - ref:.2e})")


if __name__ == "__main__":
    main()