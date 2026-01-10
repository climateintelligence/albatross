#!/usr/bin/env python
"""
Module for loading climate division data for running NIPA
"""

from albatross.atmos_ocean_data import create_phase_index2, load_climdata, openDAPsst, openDAPslp, openDAPsst_cached, openDAPslp_cached, openDAPz500, openDAPz500_cached
from albatross.utils import int_to_month

from functools import lru_cache
import json
import numpy as np

def freeze_dict(d):
    """
    Convert dict to a hashable representation for caching.
    Converts nested structures (e.g., tuples, lists) recursively.
    """
    return json.dumps(d, sort_keys=True, default=str)

def get_data(kwgroups, glo_var_name=None, workdir=None, use_cache=True, load_target=True):
    # Create a unique key from parameters
    clim_key = freeze_dict(kwgroups['climdata']) if load_target else "NO_TARGET"
    glo_key  = freeze_dict(kwgroups['glo_var'])
    index_key = freeze_dict(kwgroups['index'])
    cache_key = (glo_var_name, clim_key, glo_key, index_key, bool(use_cache), bool(load_target))

    if not hasattr(get_data, "_cache"):
        get_data._cache = {}

    if cache_key in get_data._cache:
        return get_data._cache[cache_key]

    # -------------------------
    # Target (optional)
    # -------------------------
    if load_target:
        clim_data = load_climdata(**kwgroups['climdata'])
    else:
        clim_data = None

    # -------------------------
    # Global field
    # -------------------------
    if glo_var_name == 'sst':
        glo_var_func = openDAPsst_cached if use_cache else openDAPsst
        glo_var = glo_var_func(anomalies=True, workdir=workdir, **kwgroups['glo_var'])
    elif glo_var_name == 'slp':
        glo_var_func = openDAPslp_cached if use_cache else openDAPslp
        glo_var = glo_var_func(anomalies=True, workdir=workdir, **kwgroups['glo_var'])
    elif glo_var_name=='z500':
        glo_var_func = openDAPz500_cached if use_cache else openDAPz500
        # choose whether you want height in meters (recommended)
        glo_var = glo_var_func(anomalies=True, convert_to_height_m=True, workdir=workdir, **kwgroups [ 'glo_var' ])
    elif glo_var_name=="olr":
        # Option A: keep resolution fixed in code (simplest)
        glo_var_func = openDAPolr_cached if use_cache else openDAPolr
        glo_var = glo_var_func(anomalies=True, workdir=workdir,**kwgroups [ "glo_var" ])

    else:
        raise ValueError(f"Global variable {glo_var_name} not supported.")

    assert not np.isnan(glo_var.data).all(), f"{glo_var_name.upper()} contains only NaNs!"
    assert glo_var.data.shape[0] > 0, f"{glo_var_name.upper()} has no time steps!"

    # -------------------------
    # Index + phase masks
    # -------------------------
    index, phaseind = create_phase_index2(**kwgroups['index'])

    result = (clim_data, glo_var_name, glo_var, index, phaseind)
    get_data._cache[cache_key] = result
    return result

def create_kwgroups(debug=False, climdata_startyr=1871, n_yrs=145,
    climdata_months=[1, 2, 3], n_mon_glo_var=3, glo_var_lag=3, n_mon_index=3, index_lag=3, n_phases=2, phases_even=True,
    index_fp=None, climdata_fp=None, glo_var_name=None, climdata_aggregation='mean'):

    """
    This function takes information about the seasons, years, and type of divisional
    data to look at, and creates appropriate kwgroups (parameters) to be input into
    data loading and openDap modules.
    """
    # _Check a few things

    assert climdata_months[0] >= 1, 'Divisonal data can only wrap to the following year'
    assert climdata_months[-1] <= 15, 'DJFM (i.e. [12, 13, 14, 15]) is the biggest wrap allowed'

    # _Following block sets the appropriate start month for the SST and SLP fields
    # _based on the input climdata_months and the specified lags

    glo_var_months = []
    index_months = []

    glo_var_start = climdata_months[0] - glo_var_lag
    glo_var_months.append(glo_var_start)

    index_start = climdata_months[0] - index_lag
    index_months.append(index_start)

    # _The for loops then populate the rest of the sst(slp)_months based n_mon_sst(slp)
    for i in range(1, n_mon_glo_var):
        glo_var_months.append(glo_var_start + i)

    for i in range(1, n_mon_index):
        index_months.append(index_start + i)

    assert glo_var_months[0] >= -8, 'glo_var_lag set too high, only goes to -8'
    assert index_months[0] >= -8, 'index_lag set too high, only goes to -8'

    # _Next block of code checks start years and end years and sets appropriately.
    # _So hacky..

    #########################################################
    #########################################################

    if climdata_months[-1] <= 12:
        climdata_endyr = climdata_startyr + n_yrs - 1
        if glo_var_months[0] < 1 and glo_var_months[-1] < 1:
            glo_var_startyr = climdata_startyr - 1
            glo_var_endyr = climdata_endyr - 1
        elif glo_var_months[0] < 1 and glo_var_months[-1] >= 1:
            glo_var_startyr = climdata_startyr - 1
            glo_var_endyr = climdata_endyr
        elif glo_var_months[0] >= 1 and glo_var_months[-1] >= 1:
            glo_var_startyr = climdata_startyr
            glo_var_endyr = climdata_endyr

    elif climdata_months[-1] > 12:
        climdata_endyr = climdata_startyr + n_yrs
        if glo_var_months[0] < 1 and glo_var_months[-1] < 1:
            glo_var_startyr = climdata_startyr - 1
            glo_var_endyr = climdata_endyr - 2
        elif glo_var_months[0] < 1 and glo_var_months[-1] >= 1:
            glo_var_startyr = climdata_startyr - 1
            glo_var_endyr = climdata_endyr - 1
        elif glo_var_months[0] >= 1 and 1 <= glo_var_months[-1] <= 12:
            glo_var_startyr = climdata_startyr
            glo_var_endyr = climdata_endyr - 1
        elif glo_var_months[0] >= 1 and glo_var_months[-1] > 12:
            glo_var_startyr = climdata_startyr
            glo_var_endyr = climdata_endyr

    if climdata_months[-1] <= 12:
        climdata_endyr = climdata_startyr + n_yrs - 1
        if index_months[0] < 1 and index_months[-1] < 1:
            index_startyr = climdata_startyr - 1
            index_endyr = climdata_endyr - 1
        elif index_months[0] < 1 and index_months[-1] >= 1:
            index_startyr = climdata_startyr - 1
            index_endyr = climdata_endyr
        elif index_months[0] >= 1 and index_months[-1] >= 1:
            index_startyr = climdata_startyr
            index_endyr = climdata_endyr

    elif climdata_months[-1] > 12:
        climdata_endyr = climdata_startyr + n_yrs
        if index_months[0] < 1 and index_months[-1] < 1:
            index_startyr = climdata_startyr - 1
            index_endyr = climdata_endyr - 2
        elif index_months[0] < 1 and index_months[-1] >= 1:
            index_startyr = climdata_startyr - 1
            index_endyr = climdata_endyr - 1
        elif index_months[0] >= 1 and 1 <= index_months[-1] <= 12:
            index_startyr = climdata_startyr
            index_endyr = climdata_endyr - 1
        elif index_months[0] >= 1 and index_months[-1] > 12:
            index_startyr = climdata_startyr
            index_endyr = climdata_endyr

    if debug:
        i2m = int_to_month()
        print('Target dataset starts in %s-%d, ends in %s-%d' %
              (i2m[climdata_months[0]], climdata_startyr, i2m[climdata_months[-1]], climdata_endyr))
        print('Glo var starts in %s-%d, ends in %s-%d' %
              (i2m[glo_var_months[0]], glo_var_startyr, i2m[glo_var_months[-1]], glo_var_endyr))
        print('INDEX starts in %s-%d, ends in %s-%d' %
              (i2m[index_months[0]], index_startyr, i2m[index_months[-1]], index_endyr))


        # _Create function output
    kwgroups = {
        'climdata': {'fp': climdata_fp,
                     'startyr': climdata_startyr,
                     'endyr': climdata_endyr,
                     'months': climdata_months,
                     'n_year': n_yrs,
                     'aggregation': climdata_aggregation
                     },

        'glo_var': {
            'glo_var_name': glo_var_name,
            'n_mon': n_mon_glo_var,
            'months': glo_var_months,
            'startyr': glo_var_startyr,
            'endyr': glo_var_endyr
        },

        'index': {'n_mon': n_mon_index,
                  'months': index_months,
                  'startyr': index_startyr,
                  'endyr': index_endyr,
                  'n_year': n_yrs,
                  'fp': index_fp,
                  'n_phases': n_phases,
                  'phases_even': phases_even
                  }
    }
    # Ensure all `months` fields are tuples (for hashing with @lru_cache)
    kwgroups [ 'climdata' ] [ 'months' ] = tuple(kwgroups [ 'climdata' ] [ 'months' ])
    kwgroups [ 'glo_var' ] [ 'months' ] = tuple(kwgroups [ 'glo_var' ] [ 'months' ])
    kwgroups [ 'index' ] [ 'months' ] = tuple(kwgroups [ 'index' ] [ 'months'])
    return kwgroups
