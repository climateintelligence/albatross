#!/usr/bin/env python
"""
Module for loading atmospheric and oceanic data necessary to run NIPA
"""
import logging
import time

LOGGER = logging.getLogger("PYWPS")

"""
Function needed to download of OPeNDAP data and compute anomalies.
"""
def _load_local_seasonal(path, var, level=None, **kwargs):
    """Legge un NetCDF mensile PSL (ERSST, NCEP/NCAR R1) e restituisce un blocco
    medio di n_mon mesi per anno (= RANGEEDGES + boxAverage IRIDL).
    Lat crescente come IRIDL. `level` (es. 500.) seleziona un livello di pressione."""
    import numpy as np
    import netCDF4
    from collections import namedtuple

    months, n_mon = kwargs["months"], int(kwargs["n_mon"])
    sy, ey = int(kwargs["startyr"]), int(kwargs["endyr"])
    m0 = (months[0] - 1) % 12 + 1          # es. 0 -> Dic, -2 -> Ott
    m1 = (months[-1] - 1) % 12 + 1

    with netCDF4.Dataset(path) as ds:
        tvar = ds["time"]
        dates = netCDF4.num2date(tvar[:], tvar.units, getattr(tvar, "calendar", "standard"))
        i0 = next((k for k, d in enumerate(dates) if d.year == sy and d.month == m0), None)
        if i0 is None:
            raise ValueError(f"{path}: {sy}-{m0:02d} not available (file covers {dates[0]} .. {dates[-1]})")
        n_yr = ((ey * 12 + m1) - (sy * 12 + m0) + 1 - n_mon) // 12 + 1
        i1 = i0 + 12 * (n_yr - 1) + n_mon
        if i1 > len(dates):
            raise ValueError(f"{path}: file ends at {dates[-1]}, data needed up to {ey}-{m1:02d}")

        v = ds[var]
        if level is not None:
            levname = next(n for n in v.dimensions if n in ("level", "lev", "plev", "isobaric"))
            lv = np.asarray(ds[levname][:], dtype=float)
            k = int(np.argmin(np.abs(lv - level)))
            if abs(lv[k] - level) > 1e-3:
                raise ValueError(f"{path}: pressure level {level} not available ({lv.tolist()})")
            idx = [slice(i0, i1) if d == "time" else (k if d == levname else slice(None))
                   for d in v.dimensions]
            raw = np.ma.filled(v[tuple(idx)].astype(float), np.nan)
        else:
            raw = np.ma.filled(v[i0:i1, ...].astype(float), np.nan)

        # toglie dimensioni di lunghezza 1 (es. zlev), tenendo sempre il tempo
        keep = [0] + [k for k in range(1, raw.ndim) if raw.shape[k] != 1]
        arr = raw.reshape([raw.shape[k] for k in keep])
        if arr.ndim != 3:
            raise ValueError(f"{path}: '{var}' has dimensions {v.dimensions} {v.shape}, expected (time, lat, lon)")
        latname = next(n for n in ("lat", "latitude", "y") if n in ds.variables)
        lonname = next(n for n in ("lon", "longitude", "x") if n in ds.variables)
        lat = np.asarray(ds[latname][:], dtype=float)
        lon = np.asarray(ds[lonname][:], dtype=float)

    if lat[0] > lat[-1]:                   # PSL: lat decrescente -> crescente (come IRIDL)
        arr, lat = arr[:, ::-1, :], lat[::-1]

    grid = np.stack([arr[12 * y: 12 * y + n_mon].mean(0) for y in range(n_yr)])
    seasonal_var = namedtuple("seasonal_var", ("data", "lat", "lon"))
    return seasonal_var(grid, lat, lon)


def _local_path(env, default_name):
    """Path of the local copy: env variable for the single file, otherwise
    ALBATROSS_DATA_DIR/<name> (default ~/data/<name>)."""
    import os
    if os.environ.get(env):
        return os.environ[env]
    data_dir = os.environ.get("ALBATROSS_DATA_DIR", os.path.expanduser("~/data"))
    return os.path.join(data_dir, default_name)


# -----------------------------------------------------------------------------
# Download of the predictor fields from NOAA PSL (same files used locally)
# -----------------------------------------------------------------------------
PSL_SOURCES = {
    # key: (env variable for the file, file name, URL)
    "sst": ("ALBATROSS_SST_NC", "sst.mnmean.nc",
            "https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc"),
    "slp": ("ALBATROSS_SLP_NC", "slp.mon.mean.nc",
            "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/surface/slp.mon.mean.nc"),
    "hgt": ("ALBATROSS_HGT_NC", "hgt.mon.mean.nc",
            "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Monthlies/pressure/hgt.mon.mean.nc"),
    "orog": ("ALBATROSS_OROG_NC", "hgt.sfc.nc",
             "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/surface/hgt.sfc.nc"),
}
R1_KEYS = {"slp", "hgt"}


def _download_file(url, dest, timeout=120):
    """Stream url to dest atomically (temporary file + rename), checking that the
    result opens as NetCDF. Concurrent downloads cannot leave a truncated file."""
    import os
    import tempfile
    import requests
    import netCDF4

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest) or ".", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f, requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        with netCDF4.Dataset(tmp):
            pass
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _last_month(path):
    """(year, month) of the last time step in a monthly NetCDF file."""
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        t = ds["time"]
        d = netCDF4.num2date(t[-1], t.units, getattr(t, "calendar", "standard"))
    return int(d.year), int(d.month)


def ensure_local(key, need_until=None):
    """
    Return the path of the local copy of a PSL file, downloading it if missing.
    If need_until=(year, month) is not covered, the file is downloaded again,
    at most once every ALBATROSS_REFRESH_HOURS (default 24) per file.
    If the download fails, an existing copy is still returned.
    """
    import os
    import time as _time

    env, name, url = PSL_SOURCES[key]
    path = _local_path(env, name)
    exists = os.path.isfile(path)

    refresh = not exists
    if exists and need_until is not None:
        try:
            last = _last_month(path)
        except Exception as e:
            LOGGER.warning(f"{path} unreadable ({e}): downloading again")
            last, refresh = (0, 0), True
        if tuple(last) < tuple(need_until):
            max_age_h = float(os.environ.get("ALBATROSS_REFRESH_HOURS", "24"))
            age_h = (_time.time() - os.path.getmtime(path)) / 3600.0
            refresh = refresh or age_h >= max_age_h
            if not refresh:
                LOGGER.info(f"{name} ends at {last[0]}-{last[1]:02d} and was refreshed "
                            f"{age_h:.1f} h ago: not downloading again")

    if refresh:
        LOGGER.info(f"Downloading {url} -> {path}")
        try:
            with_retry(lambda: _download_file(url, path), max_attempts=3, delay=20,
                       log_prefix=f"[{name}] ", context=url)
            LOGGER.info(f"{name} downloaded")
        except Exception as e:
            if not exists:
                raise RuntimeError(f"Could not download {url}: {e}") from e
            LOGGER.warning(f"Could not refresh {name} ({e}); using the existing copy")
    return path


def _need_until(kwargs):
    """Last (year, month) needed for a block request (startyr..endyr, months)."""
    m1 = (kwargs["months"][-1] - 1) % 12 + 1
    return int(kwargs["endyr"]), int(m1)


def _load_psl(key, var, level=None, **kwargs):
    need = _need_until(kwargs)
    path = ensure_local(key, need_until=need)
    try:
        return _load_local_seasonal(path, var, level=level, **kwargs)
    except ValueError as e:
        if key in R1_KEYS and _last_month(path) < need:
            raise ValueError(
                f"{e}. NCEP/NCAR Reanalysis 1 production ended in March 2026 "
                f"(https://psl.noaa.gov/news/2026/r1datanotice.html): no {var} data "
                f"after that date.") from e
        raise


def openDAPsst(**kwargs):
    """
    SST ERSST v5 from the NOAA PSL file, downloaded and cached locally
    (ALBATROSS_SST_NC, or ALBATROSS_DATA_DIR/sst.mnmean.nc).
    IRIDL is used only if the PSL file cannot be obtained at all.
    """
    try:
        out = _load_psl("sst", "sst", **kwargs)
        LOGGER.info("SST read from local PSL copy")
        return out
    except RuntimeError as e:
        LOGGER.warning(f"{e}: falling back to IRIDL")
    return _openDAPsst_iridl(**kwargs)


def _openDAPsst_iridl(**kwargs):
    """
    Fetch ERSSTv5 global SST from IRIDL via OPeNDAP

    Parameters
    ----------
    workdir   : unused (kept for symmetry).
    kwargs    : expected keys: startyr, endyr, months (e.g. [12,1,2]), n_mon
    """
    import re, urllib.error
    from collections import namedtuple
    from pydap.client import open_url
    from numpy import arange
    from albatross.utils import int_to_month

    LOGGER.info("⏬  Downloading SST via OPeNDAP…")

    # ----------------------------
    # 1) Build IRIDL URL
    # ----------------------------
    var = "sst"
    base_url = (
        f"http://iridl.ldeo.columbia.edu"
        f"/SOURCES/.NOAA/.NCDC/.ERSST/.version5/.{var}"
        f"/T/(startmon%20startyr)(endmon%20endyr)RANGEEDGES"
        f"/T/nbox/0.0/boxAverage/dods"
    )

    i2m = int_to_month()
    subs = {
        "startmon": i2m [ kwargs [ "months" ] [ 0 ] ],
        "endmon": i2m [ kwargs [ "months" ] [ -1 ] ],
        "startyr": kwargs [ "startyr" ],
        "endyr": kwargs [ "endyr" ],
        "nbox": kwargs [ "n_mon" ],
    }
    for k, v in subs.items():
        base_url = re.sub(k, str(v), base_url)

    LOGGER.debug(f"SST URL → {base_url}")

    # ----------------------------
    # 2) Open dataset with retry
    # ----------------------------
    def open_url_retry():
        return open_url(base_url)

    ds = with_retry(
        open_url_retry,
        max_attempts=5,
        delay=20,
        allowed_exceptions=(urllib.error.URLError, TimeoutError, Exception),
        log_prefix="[SST] "
    )

    grid = ds [ var ].array [ :, :, :, : ].data.squeeze()  # (t , y , x)
    time = ds [ "T" ].data [ : ].squeeze()
    lat = ds [ "Y" ] [ : ]
    lon = ds [ "X" ] [ : ]

    # ----------------------------
    # 3) Collapse to seasons
    # ----------------------------
    nseas = 12 // kwargs [ "n_mon" ]
    season_idx = arange(0, len(time), nseas).astype(int)
    grid = grid [ season_idx ]  # (nyears , y , x)

    seasonal_var = namedtuple("seasonal_var", ("data", "lat", "lon"))
    LOGGER.info("✅  SST download & processing complete.")
    return seasonal_var(grid, lat, lon)

def openDAPslp(**kwargs):
    """
    SLP NCEP/NCAR Reanalysis 1 (hPa) from the NOAA PSL file, downloaded and cached
    locally (ALBATROSS_SLP_NC, or ALBATROSS_DATA_DIR/slp.mon.mean.nc).
    IRIDL is used only if the PSL file cannot be obtained at all.
    """
    try:
        out = _load_psl("slp", "slp", **kwargs)
        LOGGER.info("SLP read from local PSL copy")
        return out
    except RuntimeError as e:
        LOGGER.warning(f"{e}: falling back to IRIDL")
    return _openDAPslp_iridl(**kwargs)


def _openDAPslp_iridl(**kwargs):
    """
    Fetch CPC global sea-level pressure from IRIDL via OPeNDAP

    Parameters
    ----------

    workdir   : unused (kept for interface symmetry).
    kwargs    : created by create_kwgroups():
                startyr, endyr, months, n_mon
    """
    import re, urllib.error
    from collections import namedtuple
    from pathlib import Path
    import numpy as np
    from numpy import arange
    from pydap.client import open_url
    from albatross.utils import int_to_month

    LOGGER.info("⏬  Downloading SLP via OPeNDAP…")

    # ------------------------------------------------------------------ #
    # 1) Build IRIDL URL
    # ------------------------------------------------------------------ #
    base_url = (
        "http://iridl.ldeo.columbia.edu"
        "/SOURCES/.NOAA/.NCEP-NCAR/.CDAS-1"
        "/.MONTHLY/.Intrinsic/.MSL/.pressure"
        "/T/(startmon%20startyr)(endmon%20endyr)RANGEEDGES"
        "/T/nbox/0.0/boxAverage/dods"
    )

    i2m = int_to_month()
    subs = {
        "startmon": i2m[kwargs["months"][0]],
        "endmon":   i2m[kwargs["months"][-1]],
        "startyr":  kwargs["startyr"],
        "endyr":    kwargs["endyr"],
        "nbox":     kwargs["n_mon"],                 # NB: text “nbox” precedes the value
    }
    for k, v in subs.items():
        base_url = re.sub(k, str(v), base_url)

    LOGGER.debug(f"OPeNDAP URL → {base_url}")

    # ------------------------------------------------------------------ #
    # 2) Open dataset
    # ------------------------------------------------------------------ #
    def open_url_retry():
        return open_url(base_url)

    ds = with_retry(
        open_url_retry,
        max_attempts=5,
        delay=20,
        allowed_exceptions=(urllib.error.URLError, TimeoutError, Exception),
        log_prefix="[SLP] "
    )

    grid = ds["pressure"].array[:, :, :, :].data.squeeze()   # (t , y , x)
    time = ds["T"].data[:].squeeze()
    lat  = ds["Y"][:]
    lon  = ds["X"][:]

    # ------------------------------------------------------------------ #
    # 3) Collapse to seasons
    # ------------------------------------------------------------------ #
    nseas = 12 // kwargs["n_mon"]          # e.g. 4 for 3-month seasons
    season_idx = arange(0, len(time), nseas).astype(int)
    grid = grid[season_idx]                # (nyears , y , x)

    # ------------------------------------------------------------------ #
    # 5) Wrap up
    # ------------------------------------------------------------------ #
    seasonal_var = namedtuple("seasonal_var", ("data", "lat", "lon"))
    LOGGER.info("✅  SLP download & processing complete.")
    return seasonal_var(grid, lat, lon)

def openDAPz500(convert_to_height_m: bool = True, **kwargs):
    """
    Z500 NCEP/NCAR Reanalysis 1 from the NOAA PSL file, downloaded and cached
    locally (ALBATROSS_HGT_NC, or ALBATROSS_DATA_DIR/hgt.mon.mean.nc).
    The PSL file holds geopotential height in metres; with convert_to_height_m=False
    geopotential (m^2/s^2) is returned, as in the IRIDL version.
    IRIDL is used only if the PSL file cannot be obtained at all.
    """
    try:
        out = _load_psl("hgt", "hgt", level=500.0, **kwargs)
        if not convert_to_height_m:
            out = type(out)(out.data * 9.80665, out.lat, out.lon)
        LOGGER.info("Z500 read from local PSL copy")
        return out
    except RuntimeError as e:
        LOGGER.warning(f"{e}: falling back to IRIDL")
    return _openDAPz500_iridl(convert_to_height_m=convert_to_height_m, **kwargs)


def _openDAPz500_iridl(convert_to_height_m: bool = True, **kwargs):
    """
    Fetch NCEP/NCAR CDAS-1 monthly geopotential at 500 hPa from IRIDL via OPeNDAP

    Parameters
    ----------

    convert_to_height_m : bool, default True
        If True converts geopotential (phi, m^2/s^2) to geopotential height (m) by /g.
    kwargs : expected keys: startyr, endyr, months (e.g. [12,1,2]), n_mon
    """
    import re, urllib.error
    import numpy as np
    from collections import namedtuple
    from numpy import arange
    from pydap.client import open_url
    from albatross.utils import int_to_month

    LOGGER.info("⏬  Downloading Z500 via OPeNDAP…")

    # ----------------------------
    # 1) Build IRIDL URL
    # ----------------------------
    # CDAS-1 monthly, Intrinsic, PressureLevel, geopotential (phi)
    # Pressure selection is done via P/(500)VALUES
    base_url = (
        "http://iridl.ldeo.columbia.edu"
        "/SOURCES/.NOAA/.NCEP-NCAR/.CDAS-1"
        "/.MONTHLY/.Intrinsic/.PressureLevel/.phi"
        "/P/(500)VALUES"
        "/T/(startmon%20startyr)(endmon%20endyr)RANGEEDGES"
        "/T/nbox/0.0/boxAverage/dods"
    )

    i2m = int_to_month()
    subs = {
        "startmon": i2m[kwargs["months"][0]],
        "endmon":   i2m[kwargs["months"][-1]],
        "startyr":  kwargs["startyr"],
        "endyr":    kwargs["endyr"],
        "nbox":     kwargs["n_mon"],
    }
    for k, v in subs.items():
        base_url = re.sub(k, str(v), base_url)

    LOGGER.debug(f"Z500 URL → {base_url}")

    # ----------------------------
    # 2) Open dataset with retry
    # ----------------------------
    def open_url_retry():
        return open_url(base_url)

    ds = with_retry(
        open_url_retry,
        max_attempts=20,
        delay=20,
        allowed_exceptions=(urllib.error.URLError, TimeoutError, Exception),
        log_prefix="[Z500] ",
        context=base_url
    )

    # Variable name is "phi" at this path.
    grid = ds["phi"].array[:, :, :].data.squeeze()  # (t, y, x)
    time = ds["T"].data[:].squeeze()
    lat  = ds["Y"][:]
    lon  = ds["X"][:]

    # ----------------------------
    # 3) Collapse to seasons
    # ----------------------------
    nseas = 12 // kwargs["n_mon"]
    season_idx = arange(0, len(time), nseas).astype(int)
    grid = grid[season_idx]  # (nyears, y, x)

    # ----------------------------
    # 4) Convert geopotential to height (optional)
    # ----------------------------
    if convert_to_height_m:
        g = 9.80665
        grid = grid / g


    seasonal_var = namedtuple("seasonal_var", ("data", "lat", "lon"))
    LOGGER.info("✅  Z500 download & processing complete.")
    return seasonal_var(grid, lat, lon)

"""
Function needed to allow caching of OPeNDAP data. Using it, we can avoid downloading multiple time same glo_var
"""
from functools import lru_cache
def _freeze_kwargs(kwargs: dict):
    """
    Freeze keyword arguments for use as an lru_cache key.

    - Drops non–data-defining keys (e.g. workdir)
    - Converts lists to tuples
    """
    drop_keys = {"workdir"}

    items = []
    for k, v in kwargs.items():
        if k in drop_keys:
            continue
        if isinstance(v, list):
            v = tuple(v)
        items.append((k, v))

    return tuple(sorted(items))

@lru_cache(maxsize=20)
def openDAPsst_cached_frozen(frozen_kwargs):
    kwargs = dict(frozen_kwargs)
    return openDAPsst(**kwargs)

def openDAPsst_cached(**kwargs):
    frozen = _freeze_kwargs(kwargs)
    return openDAPsst_cached_frozen(frozen)

@lru_cache(maxsize=20)
def openDAPz500_cached_frozen(convert_to_height_m: bool, frozen_kwargs):
    kwargs = dict(frozen_kwargs)
    return openDAPz500(convert_to_height_m=convert_to_height_m, **kwargs)

def openDAPz500_cached(convert_to_height_m=True, **kwargs):
    frozen = _freeze_kwargs(kwargs)
    return openDAPz500_cached_frozen(convert_to_height_m, frozen)

@lru_cache(maxsize=20)
def openDAPslp_cached_frozen(frozen_kwargs):
    kwargs = dict(frozen_kwargs)
    return openDAPslp(**kwargs)

def openDAPslp_cached(**kwargs):
    frozen = _freeze_kwargs(kwargs)
    return openDAPslp_cached_frozen(frozen)

def with_retry(func, max_attempts=3, delay=5, allowed_exceptions=(Exception,), log_prefix="", context=""):
    """
    Run func() with retries. Prints attempt number and flushes immediately.
    `context` is an optional string (e.g., URL) to print once per attempt.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            msg = f"{log_prefix}Attempt {attempt}/{max_attempts}"
            if context:
                msg += f" | {context}"
            print(msg, flush=True)
            logging.warning(msg)
            return func()

        except allowed_exceptions as e:
            msg = f"{log_prefix}⚠️ Attempt {attempt}/{max_attempts} failed: {e}"
            print(msg, flush=True)
            logging.warning(msg)

            if attempt < max_attempts:
                msg2 = f"{log_prefix}⏳ Retrying in {delay} seconds..."
                print(msg2, flush=True)
                logging.warning(msg2)
                time.sleep(delay)
            else:
                raise RuntimeError(f"{log_prefix}❌ All {max_attempts} attempts failed.") from e

"""
Function needed to handle clim files used as target
"""
def load_clim_file(fp):
    import numpy as np
    import pandas as pd

    # First get the description and years from the first two lines
    with open(fp, "r") as f:
        description = f.readline().strip()
        years_line = f.readline().strip()

    startyr, _ = years_line[:4], years_line[5:9]  # just use first year

    # Now load actual data (from third line onward)
    data = np.loadtxt(fp, skiprows=2)
    data[data <= -99.0] = np.nan
    data = data.reshape(data.size)  # Make 1D

    timeargs = {
        'start': startyr + '-01',
        'periods': len(data),
        'freq': 'ME'
    }
    index = pd.date_range(**timeargs)
    clim_data = pd.Series(data=data, index=index)

    return clim_data

def load_climdata(**kwargs):
    """
    Load target/clim time series and aggregate selected months per year.

    Expected kwargs:
      fp: path to file
      months: list[int]  # e.g. [4,5,6]
      startyr: int
      n_year: int
      aggregation: str  # "mean" or "sum" (aliases accepted: "average", "cumulated")
    """
    import numpy as np
    from numpy import where
    from albatross.utils import lag_to_month

    data = load_clim_file(kwargs["fp"])

    # --- aggregation mode ---
    agg = str(kwargs.get("aggregation", "mean")).strip().lower()
    aliases = {"average": "mean", "avg": "mean", "cumulated": "sum", "cumulative": "sum"}
    agg = aliases.get(agg, agg)
    if agg not in {"mean", "sum"}:
        raise ValueError("aggregation must be one of: mean, sum (aliases: average, cumulated)")

    reducer = np.nanmean if agg == "mean" else np.nansum

    # --- locate first month index in the series ---
    tran = lag_to_month()
    startmon = int(tran[kwargs["months"][0]])
    startyr = int(kwargs["startyr"])

    idx0 = where((data.index.year == startyr) & (data.index.month == startmon))[0]
    if idx0.size == 0:
        raise ValueError(f"Start date not found in target series: {startyr}-{startmon:02d}")

    idx_start = int(idx0[0])

    # --- build year-by-year aggregation ---
    n_year = int(kwargs["n_year"])
    months = list(kwargs["months"])
    climdata = np.full((n_year,), np.nan, dtype=float)

    n_months = len(months)
    month_offsets = np.arange(n_months)
    n_dropped = 0

    for y in range(n_year):
        mons = idx_start + 12 * y + month_offsets

        # Stagione troncata dalla fine (o dall'inizio) della serie: NaN, non media parziale.
        # Con aggregation="sum" una finestra parziale darebbe un totale sottostimato.
        in_range = (mons >= 0) & (mons < len(data.values))
        if not in_range.all():
            n_dropped += 1
            LOGGER.warning(
                f"Incomplete season for year {startyr + y}: "
                f"{int(in_range.sum())}/{n_months} months available in the series. Year set to NaN."
            )
            continue

        vals = np.asarray(data.values)[mons].astype(float)

        # Buchi interni: np.nansum/np.nanmean su una finestra parzialmente NaN
        # restituiscono un valore parziale, non NaN, quindi va controllato qui.
        n_valid = int(np.isfinite(vals).sum())
        if n_valid < n_months:
            n_dropped += 1
            LOGGER.warning(
                f"Incomplete season for year {startyr + y}: "
                f"{n_valid}/{n_months} valid months (NaN in data). Year set to NaN."
            )
            continue

        climdata[y] = reducer(vals)

    if n_dropped:
        LOGGER.warning(
            f"load_climdata: {n_dropped}/{n_year} years set to NaN due to incomplete seasons "
            f"(months={months}, aggregation={agg})."
        )

    return climdata

"""
Function needed to partition years into phases based on index values
"""
def create_phase_index2(**kwargs):
    from copy import deepcopy
    import numpy as np
    index = load_clim_file(kwargs [ 'fp' ])
    from numpy import arange, where, zeros

    from albatross.utils import lag_to_month
    tran = lag_to_month()
    startmon = int(tran [ kwargs [ 'months' ] [ 0 ] ])
    startyr = kwargs [ 'startyr' ]
    idx_start = where((index.index.year == startyr) & (index.index.month == startmon))
    if idx_start[0].size == 0:
        raise ValueError(f"Index start date not found: {startyr}-{startmon:02d}")
    i0 = int(idx_start[0][0])

    # Sentinelle (-999, -99.99, -99.9) -> NaN
    vals = np.asarray(index.values, dtype=float).copy()
    vals[vals <= -99.0] = np.nan

    n_mon = len(kwargs['months'])
    index_avg = np.full(kwargs['n_year'], np.nan)
    for year in range(kwargs['n_year']):
        mons = i0 + 12 * year + arange(n_mon)
        if mons[-1] >= vals.size:
            continue                      # finestra oltre la fine del file -> NaN
        w = vals[mons]
        if np.isfinite(w).all():          # finestra incompleta -> NaN, non media parziale
            index_avg[year] = w.mean()

    # gli anni con indice NaN restano fuori da tutte le fasi
    finite_pos = np.where(np.isfinite(index_avg))[0]
    idx = finite_pos[np.argsort(index_avg[finite_pos])]
    nyrs = idx.size
    nphase = kwargs [ 'n_phases' ]
    phases_even = kwargs [ 'phases_even' ]
    p = np.zeros((len(index_avg)), dtype='bool')
    p1 = deepcopy(p)
    p2 = deepcopy(p)
    p3 = deepcopy(p)
    p4 = deepcopy(p)
    p5 = deepcopy(p)
    phaseind = {}

    # Ensure x and x2 are integers before slicing
    if nphase==1:
        p [ idx [ : ] ] = True
        phaseind [ 'allyears' ] = p
    if nphase==2:
        x = nyrs / nphase
        x = int(x) if isinstance(x, float) else x  # Make sure x is an integer
        p1 [ idx [ :int(x) ] ] = True
        phaseind [ 'neg' ] = p1
        p2 [ idx [ int(x): ] ] = True
        phaseind [ 'pos' ] = p2
    if nphase==3:
        if phases_even:
            x = nyrs / nphase
            x = int(x) if isinstance(x, float) else x  # Ensure x is an integer
            x2 = nyrs - x
            x2 = int(x2) if isinstance(x2, float) else x2  # Ensure x2 is an integer
        else:
            x = nphase / 4
            x2 = nyrs - x
            x = int(x) if isinstance(x, float) else x  # Ensure x is an integer
            x2 = int(x2) if isinstance(x2, float) else x2  # Ensure x2 is an integer

        p1 [ idx [ :x ] ] = True
        phaseind [ 'neg' ] = p1
        p2 [ idx [ x:x2 ] ] = True
        phaseind [ 'neutral' ] = p2
        p3 [ idx [ x2: ] ] = True
        phaseind [ 'pos' ] = p3

    if nphase==4:
        if phases_even:
            x = nyrs / nphase
            x = int(x) if isinstance(x, float) else x  # Ensure x is an integer
            x3 = nyrs - x
            x3 = int(x3) if isinstance(x3, float) else x3  # Ensure x3 is an integer
            xr = (x3 - x) / 2
            xr = int(xr) if isinstance(xr, float) else xr  # Ensure xr is an integer
            x2 = x + xr
        else:
            half = nyrs / 2
            half = int(half) if isinstance(half, float) else half  # Ensure half is an integer
            x = int(round(half * 0.34))
            x3 = nyrs - x
            xr = (x3 - x) / 2
            xr = int(xr) if isinstance(xr, float) else xr  # Ensure xr is an integer
            x2 = x + xr

        p1 [ idx [ :x ] ] = True
        phaseind [ 'neg' ] = p1
        p2 [ idx [ x:x2 ] ] = True
        phaseind [ 'neutneg' ] = p2
        p3 [ idx [ x2:x3 ] ] = True
        phaseind [ 'neutpos' ] = p3
        p4 [ idx [ x3: ] ] = True
        phaseind [ 'pos' ] = p4
    if nphase==5:
        if phases_even:
            x = nyrs / nphase
            x4 = nyrs - x
            x4 = int(x4) if isinstance(x4, float) else x4  # Ensure x4 is an integer
            xr = (x4 - x) / 3
            xr = int(xr) if isinstance(xr, float) else xr  # Ensure xr is an integer
            x2 = x + xr
            x3 = x4 - xr
        else:
            half = nyrs / 2
            half = int(half) if isinstance(half, float) else half  # Ensure half is an integer
            x = int(round(half * 0.3))
            x4 = nyrs - x
            xr = (x4 - x) / 3
            xr = int(xr) if isinstance(xr, float) else xr  # Ensure xr is an integer
            x2 = x + xr
            x3 = x4 - xr

        p1 [ idx [ :x ] ] = True
        phaseind [ 'neg' ] = p1
        p2 [ idx [ x:x2 ] ] = True
        phaseind [ 'neutneg' ] = p2
        p3 [ idx [ x2:x3 ] ] = True
        phaseind [ 'neutral' ] = p3
        p4 [ idx [ x3:x4 ] ] = True
        phaseind [ 'neutpos' ] = p4
        p5 [ idx [ x4: ] ] = True
        phaseind [ 'pos' ] = p5

    return index_avg, phaseind