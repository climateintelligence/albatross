#!/usr/bin/env python
"""
Altezza geopotenziale a un livello qualsiasi da NCEP/NCAR Reanalysis 1 (file locale PSL),
per usare come predittore anche la stratosfera: --glo-var z50, z100, z30, z10, ...

Stesso file di Z500: variabile d'ambiente ALBATROSS_HGT_NC, altrimenti ~/data/hgt.mon.mean.nc
(il file mensile PSL contiene 17 livelli: 1000 ... 100, 70, 50, 30, 20, 10 hPa).
Stessa logica di lettura di _load_local_seasonal: un blocco medio di n_mon mesi per anno,
latitudini in ordine crescente, valori in metri.

z500 continua a usare il percorso originale (openDAPz500), quindi i risultati gia' ottenuti
con Z500 non cambiano.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from collections import namedtuple
from functools import lru_cache

import numpy as np

LOGGER = logging.getLogger("PYWPS")
seasonal_var = namedtuple("seasonal_var", ("data", "lat", "lon"))

_GLO_RE = re.compile(r"^z(\d+)$")


def glo_var_arg(s: str) -> str:
    """Tipo per argparse: sst, slp o z<livello hPa> (es. z500, z50)."""
    s = s.strip().lower()
    if s in ("sst", "slp") or _GLO_RE.match(s):
        return s
    raise argparse.ArgumentTypeError(f"--glo-var '{s}' non valido: usa sst, slp o z<hPa> (es. z500, z50)")


def level_of(glo_var_name: str):
    m = _GLO_RE.match(str(glo_var_name).lower())
    return float(m.group(1)) if m else None


def _hgt_path():
    return os.environ.get("ALBATROSS_HGT_NC", os.path.expanduser("~/data/hgt.mon.mean.nc"))


def load_hgt(level, **kwargs):
    """Blocco stagionale di altezza geopotenziale (m) al livello `level` hPa."""
    import netCDF4

    path = _hgt_path()
    months, n_mon = kwargs["months"], int(kwargs["n_mon"])
    sy, ey = int(kwargs["startyr"]), int(kwargs["endyr"])
    m0 = (months[0] - 1) % 12 + 1
    m1 = (months[-1] - 1) % 12 + 1

    with netCDF4.Dataset(path) as ds:
        tvar = ds["time"]
        dates = netCDF4.num2date(tvar[:], tvar.units, getattr(tvar, "calendar", "standard"))
        i0 = next((k for k, d in enumerate(dates) if d.year == sy and d.month == m0), None)
        if i0 is None:
            raise ValueError(f"{path}: {sy}-{m0:02d} non presente (file {dates[0]} .. {dates[-1]})")
        n_yr = ((ey * 12 + m1) - (sy * 12 + m0) + 1 - n_mon) // 12 + 1
        i1 = i0 + 12 * (n_yr - 1) + n_mon
        if i1 > len(dates):
            raise ValueError(f"{path}: il file finisce a {dates[-1]}, servono dati fino al blocco {n_yr}")

        v = ds["hgt"]
        levname = next(n for n in v.dimensions if n in ("level", "lev", "plev", "isobaric"))
        lv = np.asarray(ds[levname][:], dtype=float)
        k = int(np.argmin(np.abs(lv - level)))
        if abs(lv[k] - level) > 1e-3:
            raise ValueError(f"{path}: livello {level:g} hPa non presente. Livelli disponibili: {lv.tolist()}")
        idx = [slice(i0, i1) if d == "time" else (k if d == levname else slice(None)) for d in v.dimensions]
        arr = np.ma.filled(v[tuple(idx)].astype(float), np.nan)
        lat = np.asarray(ds["lat"][:], dtype=float)
        lon = np.asarray(ds["lon"][:], dtype=float)

    if lat[0] > lat[-1]:
        arr, lat = arr[:, ::-1, :], lat[::-1]
    grid = np.stack([arr[12 * y: 12 * y + n_mon].mean(0) for y in range(n_yr)])
    LOGGER.info(f"✅  Z{level:g} letto da file locale {path}")
    return seasonal_var(grid, lat, lon)


def _freeze(kwargs):
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v)
                        for k, v in kwargs.items() if k != "workdir"))


@lru_cache(maxsize=20)
def _load_hgt_frozen(level, frozen):
    return load_hgt(level, **dict(frozen))


def load_hgt_cached(level, **kwargs):
    return _load_hgt_frozen(float(level), _freeze(kwargs))