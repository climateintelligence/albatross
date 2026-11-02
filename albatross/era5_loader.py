#!/usr/bin/env python
"""
ERA5 mensile (CDS, reanalysis-era5-single-levels-monthly-means) come campo globale
predittore per Albatross. Stesso output di openDAPsst / openDAPslp:
seasonal_var(data=(n_yr, nlat, nlon), lat crescente, lon 0..360).

- Un blocco di n_mon mesi per anno, a partire da kwargs["months"][0]
  (stessa logica di _load_local_seasonal).
- Media mese per mese e coarsening a `ERA5_TARGET_RES` gradi PRIMA di caricare
  in memoria: a 0.25 deg il campo intero non sta in RAM e il gridCheck legacy
  (doppio loop Python) su ~1M celle per ogni fold LOO e' impraticabile.
- "tp" viene convertita da m/giorno a mm/giorno (solo leggibilita').
"""
import logging
import os
from collections import namedtuple
from functools import lru_cache

import numpy as np

LOGGER = logging.getLogger("PYWPS")
seasonal_var = namedtuple("seasonal_var", ("data", "lat", "lon"))

ERA5_TARGET_RES = float(os.environ.get("ALBATROSS_ERA5_RES", "2.0"))  # gradi, come ERSSTv5

# glo_var_name -> (variabile nel file, env var del path, path di default, fattore unita', box lat)
ERA5_VARS = {
    "tp":      ("tp",  "ALBATROSS_TP_NC",  "~/data/era5_tp_monthly.nc",  1000.0, None),
    "tp_trop": ("tp",  "ALBATROSS_TP_NC",  "~/data/era5_tp_monthly.nc",  1000.0, (-30.0, 30.0)),
    "t2m":     ("t2m", "ALBATROSS_T2M_NC", "~/data/era5_t2m_monthly.nc", 1.0,    None),
}


def _open_era5(path, var):
    import xarray as xr

    ds = xr.open_dataset(path)
    ren = {k: v for k, v in {"valid_time": "time", "latitude": "lat", "longitude": "lon"}.items()
           if k in ds.variables}
    ds = ds.rename(ren)
    da = ds[var]

    # File misti ERA5 + ERA5T (vecchio CDS): expver come dimensione
    if "expver" in da.dims:
        da = da.isel(expver=0).combine_first(da.isel(expver=1))
    if "number" in da.dims:
        da = da.isel(number=0)
    return da


def openERA5(var="tp", **kwargs):
    import pandas as pd

    if var not in ERA5_VARS:
        raise ValueError(f"ERA5 variable {var} not supported: {list(ERA5_VARS)}")
    fvar, env, default, scale, lat_box = ERA5_VARS[var]
    path = os.environ.get(env, os.path.expanduser(default))
    da = _open_era5(path, fvar)

    if lat_box is not None:
        lo, hi = lat_box
        da = da.where((da["lat"] >= lo) & (da["lat"] <= hi), drop=True)

    months, n_mon = kwargs["months"], kwargs["n_mon"]
    sy, ey = int(kwargs["startyr"]), int(kwargs["endyr"])
    m0 = (months[0] - 1) % 12 + 1
    m1 = (months[-1] - 1) % 12 + 1

    times = pd.DatetimeIndex(da["time"].values)
    hit = np.where((times.year == sy) & (times.month == m0))[0]
    if hit.size == 0:
        raise ValueError(f"ERA5 {fvar}: start {sy}-{m0:02d} not in file ({times[0]:%Y-%m} .. {times[-1]:%Y-%m})")
    i0 = int(hit[0])
    n_yr = ((ey * 12 + m1) - (sy * 12 + m0) + 1 - n_mon) // 12 + 1
    i1 = i0 + 12 * (n_yr - 1) + n_mon
    if i1 > len(times):
        raise ValueError(f"ERA5 {fvar}: file ends at {times[-1]:%Y-%m}, requested up to block {n_yr} from {sy}-{m0:02d}")

    dlat = float(abs(np.median(np.diff(da["lat"].values))))
    f = max(1, int(round(ERA5_TARGET_RES / dlat)))

    blocks, lat, lon = [], None, None
    for y in range(n_yr):
        s = i0 + 12 * y
        if times[s].month != m0:          # buchi nella serie mensile
            raise ValueError(f"ERA5 {fvar}: non-contiguous monthly series at {times[s]:%Y-%m}")
        blk = da.isel(time=slice(s, s + n_mon)).mean("time")
        if f > 1:
            blk = blk.coarsen(lat=f, lon=f, boundary="trim").mean()
        if lat is None:
            lat = np.asarray(blk["lat"].values, dtype=float)
            lon = np.asarray(blk["lon"].values, dtype=float)
        blocks.append(np.asarray(blk.values, dtype=float) * scale)

    grid = np.stack(blocks)
    if lat[0] > lat[-1]:                  # ERA5: lat decrescente -> crescente
        grid, lat = grid[:, ::-1, :], lat[::-1]

    LOGGER.info(f"✅  ERA5 {var} loaded from local file ({grid.shape}, ~{ERA5_TARGET_RES} deg).")
    return seasonal_var(grid, lat, lon)


@lru_cache(maxsize=20)
def _openERA5_cached_frozen(var, frozen_kwargs):
    return openERA5(var=var, **dict(frozen_kwargs))


def openERA5_cached(var="tp", **kwargs):
    from albatross.atmos_ocean_data import _freeze_kwargs
    return _openERA5_cached_frozen(var, _freeze_kwargs(kwargs))