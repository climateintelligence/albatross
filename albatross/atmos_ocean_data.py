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

def openDAPsst(version='5', anomalies=True, workdir = None, **kwargs):
    '''
    # This function downloads data from the new ERSSTv5 on the IRI data library
    # kwargs should contain: startyr, endyr, startmon, endmon, nbox
    '''
    import pickle
    import re
    from collections import namedtuple

    from numpy import arange
    from pydap.client import open_url

    from albatross.utils import int_to_month

    # getting NOAA raw data
    SSTurl = 'http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCDC/.ERSST/.version' + version + '/' + \
        '.anom/T/%28startmon%20startyr%29%28endmon%20endyr%29RANGEEDGES/T/nbox/0.0/boxAverage/dods'

    if not anomalies:
        SSTurl = 'http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCDC/.ERSST/.version' + version + '/' + \
            '.sst/T/%28startmon%20startyr%29%28endmon%20endyr%29RANGEEDGES/T/nbox/0.0/boxAverage/dods'

    i2m = int_to_month()

    # Keyword arguments for setting up the download
    DLargs = {
        'startmon': i2m[kwargs['months'][0]],
        'endmon': i2m[kwargs['months'][-1]],
        'startyr': str(kwargs['startyr']),
        'endyr': str(kwargs['endyr']),
        'nbox': str(kwargs['n_mon'])
    }
    # Base path: directory of the current file
    # Use workdir if given, else default to temp path
    base_path = workdir if workdir is not None else Path(tempfile.gettempdir())
    fp = base_path / f"{DLargs [ 'startmon' ]}{DLargs [ 'startyr' ]}_{DLargs [ 'endmon' ]}{DLargs [ 'endyr' ]}_nbox_{DLargs [ 'nbox' ]}_version{version}"
    fp = fp.with_name(fp.name + ('_anoms.pkl' if anomalies else '_ssts.pkl'))
    fp.parent.mkdir(parents=True, exist_ok=True)

    os.makedirs(os.path.dirname(fp), exist_ok=True)

    seasonal_var = namedtuple('seasonal_var', ('data', 'lat', 'lon'))

    LOGGER.info('New SST field, will save to %s' % fp)

    for kw in DLargs:
        SSTurl = re.sub(kw, DLargs[kw], SSTurl)

    print(f'SSTurl: {SSTurl}')
    LOGGER.info('Starting download...')

    try:
        dataset = open_url(SSTurl)
    except (ServerError, urllib.error.URLError, Exception) as e:
        raise RuntimeError(f"❌ Failed to open SST dataset from {SSTurl}: {e}")

    arg = 'anom' if anomalies else 'sst'
    sst = dataset[arg]

    time = dataset['T']
    grid = sst.array[:, :, :, :].data.squeeze()  # MODIFIED ANDREA: inserted "data"
    t = time.data[:].squeeze()
    sstlat = dataset['Y'][:]
    sstlon = dataset['X'][:]
    LOGGER.info('Download finished.')

    # _Grid has shape (ntim, nlat, nlon)

    nseasons = 12 / kwargs['n_mon']

    ntime = len(t)

    idx = arange(0, ntime, nseasons).astype(int)

    sst = grid[idx]
    sstdata = {'grid': sst, 'lat': sstlat, 'lon': sstlon}
    var = seasonal_var(sst, sstlat, sstlon)

    f = open(fp, 'wb')
    pickle.dump(sstdata, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    return var


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


def load_climdata(**kwargs):

    data = load_clim_file(kwargs['fp'])
    from numpy import arange, where, zeros

    from albatross.utils import slp_tf
    tran = slp_tf()
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


def create_phase_index2(**kwargs):
    from copy import deepcopy

    import numpy as np
    index = load_clim_file(kwargs['fp'])
    from numpy import arange, where, zeros

    from albatross.utils import slp_tf
    tran = slp_tf()
    startmon = int(tran[kwargs['months'][0]])
    startyr = kwargs['startyr']
    idx_start = where((index.index.year == startyr) & (index.index.month == startmon))
    idx = []
    [idx.extend(arange(kwargs['n_mon']) + idx_start + 12 * n) for n in range(kwargs['n_year'])]
    index_avg = zeros((kwargs['n_year']))
    for year, mons in enumerate(idx):
        index_avg[year] = index.values[mons].mean()

    idx = np.argsort(index_avg)
    nyrs = kwargs['n_year']
    nphase = kwargs['n_phases']
    phases_even = kwargs['phases_even']
    p = np.zeros((len(index_avg)), dtype='bool')
    p1 = deepcopy(p)
    p2 = deepcopy(p)
    p3 = deepcopy(p)
    p4 = deepcopy(p)
    p5 = deepcopy(p)
    phaseind = {}
    if nphase == 1:
        p[idx[:]] = True
        phaseind['allyears'] = p
    if nphase == 2:
        x = nyrs / nphase
        p1[idx[:int(x)]] = True
        phaseind['neg'] = p1
        p2[idx[int(x):]] = True
        phaseind['pos'] = p2
    if nphase == 3:
        if phases_even:
            x = nyrs / nphase
            x2 = nyrs - x
        else:
            x = nphase / 4
            x2 = nyrs - x
        p1[idx[:x]] = True
        phaseind['neg'] = p1
        p2[idx[x:x2]] = True
        phaseind['neutral'] = p2
        p3[idx[x2:]] = True
        phaseind['pos'] = p3

    if nphase == 4:
        if phases_even:
            x = nyrs / nphase
            x3 = nyrs - x
            xr = (x3 - x) / 2
            x2 = x + xr
        else:
            half = nyrs / 2
            x = int(round(half * 0.34))
            x3 = nyrs - x
            xr = (x3 - x) / 2
            x2 = x + xr
        p1[idx[:x]] = True
        phaseind['neg'] = p1
        p2[idx[x:x2]] = True
        phaseind['neutneg'] = p2
        p3[idx[x2:x3]] = True
        phaseind['netpos'] = p3
        p4[idx[x3:]] = True
        phaseind['pos'] = p4
    if nphase == 5:
        if phases_even:
            x = nyrs / nphase
            x4 = nyrs - x
            xr = (x4 - x) / 3
            x2 = x + xr
            x3 = x4 - xr
        else:
            half = nyrs / 2
            x = int(round(half * 0.3))
            x4 = nyrs - x
            xr = (x4 - x) / 3
            x2 = x + xr
            x3 = x4 - xr
        p1[idx[:x]] = True
        phaseind['neg'] = p1
        p2[idx[x:x2]] = True
        phaseind['neutneg'] = p2
        p3[idx[x2:x3]] = True
        phaseind['neutral'] = p3
        p4[idx[x3:x4]] = True
        phaseind['neutpos'] = p4
        p5[idx[x4:]] = True
        phaseind['pos'] = p5
    return index_avg, phaseind
