#!/usr/bin/env python
"""
New NIPA module
"""

from collections import namedtuple
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

seasonal_var = namedtuple("seasonal_var", ("data", "lat", "lon"))


class NIPAphase(object):
    """
    Class and methods for operations on phases as determined by the MEI.

    _INPUTS
    phaseind:    dictionary containing phase names as keys and corresponding booleans as index vectors
    clim_data:    n x 1 pandas time series of the climate data (predictands)
    sst:        dictionary containing the keys 'data', 'lat', and 'lon'
    slp:        dictionary containing the keys 'data', 'lat', and 'lon'
    mei:        n x 1 pandas time series containing averaged MEI values

    _ATTRIBUTES
    sstcorr_grid
    slpcorr_grid

    """

    def __init__(self, clim_data, sst, mei, phaseind, alt=False):
        if alt:
            self.clim_data = clim_data
        else:
            self.clim_data = clim_data[phaseind]
        self.sst = seasonal_var(sst.data[phaseind], sst.lat, sst.lon)
        self.mei = mei[phaseind]
        self.flags = {}
        self.lin_model = None
        # self.pc1 = None
        self.pcs = None
        self.correlation = None
        self.hindcast = None
        return

    def categorize(self, ncat=3, hindcast=False):
        from numpy import sort
        from pandas import Series

        if hindcast:
            data = self.hindcast.copy()
        else:
            data = self.clim_data.copy()
        x = sort(data)
        n = len(x)
        upper = x[((2 * n) / ncat) - 1]
        lower = x[(n / ncat) - 1]

        cat_dat = Series(index=data.index)

        for year in data.index:
            if data[year] <= lower:
                cat_dat[year] = "B"
            elif data[year] > upper:
                cat_dat[year] = "A"
            else:
                cat_dat[year] = "N"
        if hindcast:
            self.hcat = cat_dat
        else:
            self.cat = cat_dat
        return

    def bootcorr(
        self, ntim=1000, corrconf=0.95, bootconf=0.80, debug=False, quick=True
    ):
        from numpy import isnan, ma
        from albatross.utils import sig_test, vcorr

        corrlevel = 1 - corrconf

        fieldData = self.sst.data
        clim_data = self.clim_data

        corr_grid = vcorr(X=fieldData, y=clim_data)
        n_yrs = len(clim_data)
        p_value = sig_test(corr_grid, n_yrs)

        # Mask insignificant gridpoints
        corr_grid = ma.masked_array(corr_grid, ~(p_value < corrlevel))
        # Mask land
        corr_grid = ma.masked_array(corr_grid, isnan(corr_grid))
        # Mask northern/southern ocean
        corr_grid.mask[self.sst.lat > 60] = True
        corr_grid.mask[self.sst.lat < -60] = True
        nlat = len(self.sst.lat)
        nlon = len(self.sst.lon)

        if quick:
            self.corr_grid = corr_grid
            self.n_pre_grid = nlat * nlon - corr_grid.mask.sum()
            if self.n_pre_grid == 0:
                self.flags["noSST"] = True
            else:
                self.flags["noSST"] = False
            return

        # INITIALIZE A NEW CORR GRID
        count = np.zeros((nlat, nlon))
        dat = clim_data.copy()

        for boot in range(ntim):
            # SHUFFLE THE YEARS AND CREATE THE BOOT DATA###
            idx = np.random.randint(0, len(dat) - 1, len(dat))
            boot_fieldData = np.zeros((len(idx), nlat, nlon))
            boot_fieldData[:] = fieldData[idx]
            boot_climData = dat[idx]

            boot_corr_grid = vcorr(X=boot_fieldData, y=boot_climData)
            p_value = sig_test(boot_corr_grid, n_yrs)

            count[p_value <= corrlevel] += 1
            if debug:
                print("Count max is %i" % count.max())

        # CREATE MASKED ARRAY USING THE COUNT AND BOOTCONF ATTRIBUTES
        corr_grid = np.ma.masked_array(corr_grid, count < bootconf * ntim)

        self.corr_grid = corr_grid
        self.n_pre_grid = nlat * nlon - corr_grid.mask.sum()

        if self.n_pre_grid == 0:
            self.flags["noSST"] = True
        else:
            self.flags["noSST"] = False
        return

    def gridCheck(self, lim=5, ntim=2, debug=False):
        if self.n_pre_grid < 50:
            lim = 6
            ntim = 2
        for time in range(ntim):
            dat = self.corr_grid.mask
            count = 0
            for i in np.arange(1, dat.shape[0] - 1):
                for j in np.arange(1, dat.shape[1] - 1):
                    if not dat[i, j]:
                        check = np.zeros(8)
                        check[0] = dat[i + 1, j]
                        check[1] = dat[i + 1, j + 1]
                        check[2] = dat[i + 1, j - 1]
                        check[3] = dat[i, j + 1]
                        check[4] = dat[i, j - 1]
                        check[5] = dat[i - 1, j]
                        check[6] = dat[i - 1, j + 1]
                        check[7] = dat[i - 1, j - 1]
                        if check.sum() >= lim:
                            dat[i, j] = True
                            count += 1
            if debug:
                print("Deleted %i grids" % count)

            self.corr_grid.mask = dat
            self.n_post_grid = dat.size - dat.sum()

        return

    def crossvalpcr(self, xval=True, explained_variance_threshold=0.95):
        import numpy as np
        from scipy.stats import pearsonr as corr
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import KFold
        from albatross.utils import weightsst

        predictand = self.clim_data

        # Check for insufficient SST data
        if self.corr_grid.mask.sum() >= len(self.sst.lat) * len(self.sst.lon) - 4:
            self.flags [ "noSST" ] = True
            self.hindcast = None
            self.pcs = None
            self.lin_model = None
            self.correlation = None
            print("Insufficient SST data for PCA regression.")
            return

        self.flags [ "noSST" ] = False
        sstidx = ~self.corr_grid.mask
        raw_sst = weightsst(self.sst).data [ :, sstidx ]
        n_samples = len(predictand)

        if not xval:
            # Standard PCA regression (no CV)
            cov_matrix = np.cov(raw_sst.T)
            eigval, eigvec = np.linalg.eig(cov_matrix)
            eigval, eigvec = np.real(eigval), np.real(eigvec)

            sorted_idx = np.argsort(eigval) [ ::-1 ]
            eigvec = eigvec [ :, sorted_idx ]

            explained_ratio = eigval / eigval.sum()
            cumulative_var = np.cumsum(explained_ratio)
            n_pc = np.searchsorted(cumulative_var, explained_variance_threshold) + 1

            eofs = eigvec [ :, :n_pc ]
            pcs = raw_sst.dot(eofs)

            reg = LinearRegression().fit(pcs, predictand)
            yhat = reg.predict(pcs)

            self.pcs = pcs
            self.hindcast = yhat
            self.correlation = corr(predictand, yhat) [ 0 ]
            self.lin_model = {
                "eofs": eofs,
                "regression": reg,
                "n_pc": n_pc
            }
            return

        # Cross-validation PCA regression
        yhat = np.zeros(n_samples)
        pcs_all = np.zeros((n_samples, raw_sst.shape [ 1 ]))
        models = [ ]

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(raw_sst):
            X_train, X_test = raw_sst [ train_idx ], raw_sst [ test_idx ]
            y_train = predictand [ train_idx ]

            cov_matrix = np.cov(X_train.T)
            eigval, eigvec = np.linalg.eig(cov_matrix)
            eigval, eigvec = np.real(eigval), np.real(eigvec)

            sorted_idx = np.argsort(eigval) [ ::-1 ]
            eigvec = eigvec [ :, sorted_idx ]

            explained_ratio = eigval / eigval.sum()
            cumulative_var = np.cumsum(explained_ratio)
            n_pc = np.searchsorted(cumulative_var, explained_variance_threshold) + 1

            if n_pc==0 or np.isnan(eigval [ :n_pc ]).any():
                continue

            eofs = eigvec [ :, :n_pc ]
            pcs_train = X_train.dot(eofs)
            pcs_test = X_test.dot(eofs)

            if pcs_train.shape [ 0 ] < n_pc:
                continue

            reg = LinearRegression().fit(pcs_train, y_train)
            preds = reg.predict(pcs_test)

            yhat [ test_idx ] = preds
            pcs_all [ test_idx, :n_pc ] = pcs_test

            models.append({
                "eofs": eofs,
                "regression": reg,
                "n_pc": n_pc,
                "corr": corr(y_train, reg.predict(pcs_train)) [ 0 ]
            })

        if not models:
            self.hindcast = None
            self.pcs = None
            self.lin_model = None
            self.correlation = None
            self.flags [ "noSST" ] = True
            return

        # Store hindcast from CV
        self.hindcast = yhat
        self.correlation = corr(predictand, yhat) [ 0 ]

        # Select best number of PCs from CV and refit on all data
        best_model = max(models, key=lambda m: m [ "corr" ])
        n_pc_best = best_model [ "n_pc" ]

        # Refit on full dataset using best number of PCs
        cov_matrix = np.cov(raw_sst.T)
        eigval, eigvec = np.linalg.eig(cov_matrix)
        eigval, eigvec = np.real(eigval), np.real(eigvec)

        sorted_idx = np.argsort(eigval) [ ::-1 ]
        eigvec = eigvec [ :, sorted_idx ]

        eofs = eigvec [ :, :n_pc_best ]
        pcs_full = raw_sst.dot(eofs)
        reg_full = LinearRegression().fit(pcs_full, predictand)

        self.pcs = pcs_full
        self.lin_model = {
            "eofs": eofs,
            "regression": reg_full,
            "n_pc": n_pc_best
        }
