#!/usr/bin/env python
"""
Module for loading atmospheric and oceanic data necessary to run NIPA
"""
import logging
import os
import tempfile
from pathlib import Path
import urllib.error
from pydap.exceptions import ServerError

LOGGER = logging.getLogger("PYWPS")

def openDAPsst(anomalies: bool = True, **kwargs):
    """
    Fetch ERSSTv5 global SST from IRIDL via OPeNDAP and return seasonal means.

    Parameters
    ----------
    anomalies : bool, default True
        If True the 1981-2010 seasonal climatology is removed.
    workdir   : unused (kept for symmetry).
    kwargs    : expected keys: startyr, endyr, months (e.g. [12,1,2]), n_mon
    """
    import re, urllib.error
    import numpy as np
    from pathlib import Path
    from collections import namedtuple
    from pydap.client import open_url
    from numpy import arange
    from albatross.utils import int_to_month

    LOGGER.info("⏬  Downloading SST via OPeNDAP…")

    # ----------------------------
    # 1) Build IRIDL URL
    # ----------------------------
    var = "anom" if anomalies else "sst"
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

    # ----------------------------
    # 4) Anomalies (optional)
    # ----------------------------
    if anomalies:
        if grid.size==0 or np.isnan(grid).all():
            raise ValueError("[SST] ❌ SST grid is empty or contains only NaNs — cannot compute anomalies.")
        clim = np.nanmean(grid, axis=0)
        grid = grid - clim

    seasonal_var = namedtuple("seasonal_var", ("data", "lat", "lon"))
    LOGGER.info("✅  SST download & processing complete.")
    return seasonal_var(grid, lat, lon)

def openDAPslp(anomalies: bool = True,  **kwargs):
    """
    Fetch CPC global sea-level pressure from IRIDL via OPeNDAP and return
    seasonal means (optionally as 1981-2010 anomalies).

    Parameters
    ----------
    anomalies : bool, default True
        If True the 1981-2010 seasonal climatology is removed.
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
    # 4) anomalies (optional)
    # ------------------------------------------------------------------ #
    if anomalies:
        if grid.size==0 or np.isnan(grid).all():
            raise ValueError("[SLP] ❌ SLP grid is empty or contains only NaNs — cannot compute anomalies.")
        clim = np.nanmean(grid, axis=0)  # (y, x)
        grid = grid - clim

    # ------------------------------------------------------------------ #
    # 5) Wrap up
    # ------------------------------------------------------------------ #
    seasonal_var = namedtuple("seasonal_var", ("data", "lat", "lon"))
    LOGGER.info("✅  SLP download & processing complete.")
    return seasonal_var(grid, lat, lon)

LOGGER = logging.getLogger("PYWPS")

def openDAPz500(anomalies: bool = True, convert_to_height_m: bool = True, **kwargs):
    """
    Fetch NCEP/NCAR CDAS-1 monthly geopotential at 500 hPa from IRIDL via OPeNDAP
    and return seasonal means (optionally as anomalies).

    Parameters
    ----------
    anomalies : bool, default True
        If True removes seasonal climatology (computed over returned period).
        (If you need a fixed 1981-2010 climatology, we can implement that explicitly.)
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
        max_attempts=20,          # Z500/phi can be slower; keep higher like your SST tweak
        delay=20,
        allowed_exceptions=(urllib.error.URLError, TimeoutError, Exception),
        log_prefix="[Z500] ",
        context=base_url
    )

    # In pydap, dimensions usually come back (T, Y, X) after P fixed to 500
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

    # ----------------------------
    # 5) Anomalies (optional)
    # ----------------------------
    if anomalies:
        if grid.size == 0 or np.isnan(grid).all():
            raise ValueError("[Z500] ❌ Z500 grid is empty or contains only NaNs — cannot compute anomalies.")
        clim = np.nanmean(grid, axis=0)
        grid = grid - clim

    seasonal_var = namedtuple("seasonal_var", ("data", "lat", "lon"))
    LOGGER.info("✅  Z500 download & processing complete.")
    return seasonal_var(grid, lat, lon)

"""Function needed to allow caching of OPeNDAP data. Using it, we can avoid downloading multiple time same glo_var"""

from functools import lru_cache
def _freeze_kwargs(kwargs: dict):
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in kwargs.items()))

@lru_cache(maxsize=20)
def openDAPsst_cached_frozen(anomalies: bool, frozen_kwargs):
    kwargs = dict(frozen_kwargs)
    return openDAPsst(anomalies=anomalies, **kwargs)

def openDAPsst_cached(anomalies=True, **kwargs):
    frozen = _freeze_kwargs(kwargs)
    return openDAPsst_cached_frozen(anomalies, frozen)

@lru_cache(maxsize=20)
def openDAPz500_cached_frozen(anomalies: bool, convert_to_height_m: bool, frozen_kwargs):
    kwargs = dict(frozen_kwargs)
    return openDAPz500(anomalies=anomalies, convert_to_height_m=convert_to_height_m, **kwargs)

def openDAPz500_cached(anomalies=True, convert_to_height_m=True, **kwargs):
    frozen = _freeze_kwargs(kwargs)
    return openDAPz500_cached_frozen(anomalies, convert_to_height_m, frozen)

@lru_cache(maxsize=20)
def openDAPslp_cached_frozen(anomalies: bool, frozen_kwargs):
    kwargs = dict(frozen_kwargs)
    return openDAPslp(anomalies=anomalies, **kwargs)

def openDAPslp_cached(anomalies=True, **kwargs):
    frozen = _freeze_kwargs(kwargs)
    return openDAPslp_cached_frozen(anomalies, frozen)

import time
import logging

import time
import logging
import sys

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
    data = data.reshape(data.size)  # Make 1D

    timeargs = {
        'start': startyr + '-01',
        'periods': len(data),
        'freq': 'M'
    }
    index = pd.date_range(**timeargs)
    clim_data = pd.Series(data=data, index=index)

    return clim_data

def create_phase_index2(**kwargs):
    from copy import deepcopy
    import numpy as np
    index = load_clim_file(kwargs [ 'fp' ])
    from numpy import arange, where, zeros

    from albatross.utils import lag_to_month
    tran = lag_to_month()
    startmon = int(tran [ kwargs [ 'months' ] [ 0 ] ])
    startyr = kwargs [ 'startyr' ]
    idx_start = where((index.index.year==startyr) & (index.index.month==startmon))
    idx = [ ]
    [ idx.extend(arange(len(kwargs [ 'months' ])) + idx_start + 12 * n) for n in range(kwargs [ 'n_year' ]) ]
    index_avg = zeros((kwargs [ 'n_year' ]))
    for year, mons in enumerate(idx):
        index_avg [ year ] = index.values [ mons ].mean()

    idx = np.argsort(index_avg)
    nyrs = kwargs [ 'n_year' ]
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
        phaseind [ 'netpos' ] = p3
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


def load_climdata(**kwargs):

    data = load_clim_file(kwargs['fp'])
    from numpy import arange, where, zeros

    from albatross.utils import lag_to_month
    tran = lag_to_month()
    startmon = int(tran[kwargs['months'][0]])
    startyr = kwargs['startyr']
    idx_start = where((data.index.year == startyr) & (data.index.month == startmon))
    idx = []
    [idx.extend(arange(len(kwargs['months'])) + idx_start + 12 * n) for n in range(kwargs['n_year'])]
    climdata = zeros((kwargs['n_year']))
    for year, mons in enumerate(idx):
        if data.values [ mons ].size > 0:
            climdata [ year ] = data.values [ mons ].mean()
        else:
            LOGGER.warning(f"No data available for year {year}, skipping mean calculation.")
    return climdata
