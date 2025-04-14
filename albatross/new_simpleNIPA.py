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
        self.pc1 = None
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
        corr_grid.mask[self.sst.lat < -30] = True
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
            # boot_climData = np.zeros((len(idx)))
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


    def crossvalpcr(self, xval=True, k=5, explained_variance_threshold=0.95):
        import numpy as np
        from scipy.stats import pearsonr as corr
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import KFold
        from albatross.utils import weightsst

        predictand = self.clim_data

        if self.corr_grid.mask.sum() >= len(self.sst.lat) * len(self.sst.lon) - 4:
            self.flags [ "noSST" ] = True
            return

        self.flags [ "noSST" ] = False
        sstidx = ~self.corr_grid.mask
        n = len(predictand)
        rawSSTdata = weightsst(self.sst).data [ :, sstidx ]

        yhat = np.zeros(n)
        models = [ ]
        pcs_all = np.zeros((n, rawSSTdata.shape [ 1 ]))  # max possible size, will truncate later

        if not xval:
            cvr = np.cov(rawSSTdata.T)
            eigval, eigvec = np.linalg.eig(cvr)
            eigval = np.real(eigval)
            eigvec = np.real(eigvec)

            # Sort eigenvalues and eigenvectors
            sorted_idx = np.argsort(eigval) [ ::-1 ]
            eigval = eigval [ sorted_idx ]
            eigvec = eigvec [ :, sorted_idx ]

            explained_ratio = eigval / eigval.sum()
            cumulative_variance = np.cumsum(explained_ratio)
            n_pc = np.searchsorted(cumulative_variance, explained_variance_threshold) + 1

            eofs = eigvec [ :, :n_pc ]
            pcs = rawSSTdata.dot(eofs)

            model = LinearRegression().fit(pcs, predictand)
            yhat = model.predict(pcs)

            self.pc1 = pcs [ :, 0 ]  # Save first PC only (for plotting)
            self.hindcast = yhat
            self.correlation = corr(predictand, yhat) [ 0 ]
            self.lin_model = {
                "eofs": eofs,
                "regression": model,
                "n_pc": n_pc
            }
            return

        # Cross-validation mode
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(rawSSTdata):
            X_train, X_test = rawSSTdata [ train_idx ], rawSSTdata [ test_idx ]
            y_train = predictand [ train_idx ]

            cvr = np.cov(X_train.T)
            eigval, eigvec = np.linalg.eig(cvr)
            eigval = np.real(eigval)
            eigvec = np.real(eigvec)

            sorted_idx = np.argsort(eigval) [ ::-1 ]
            eigval = eigval [ sorted_idx ]
            eigvec = eigvec [ :, sorted_idx ]

            explained_ratio = eigval / eigval.sum()
            cumulative_variance = np.cumsum(explained_ratio)
            n_pc = np.searchsorted(cumulative_variance, explained_variance_threshold) + 1

            eofs = eigvec [ :, :n_pc ]
            pcs_train = X_train.dot(eofs)
            pcs_test = X_test.dot(eofs)

            model = LinearRegression().fit(pcs_train, y_train)
            preds = model.predict(pcs_test)

            yhat [ test_idx ] = preds
            pcs_all [ test_idx, :n_pc ] = pcs_test  # fill variable-length PCs

            models.append({
                "eofs": eofs,
                "regression": model,
                "n_pc": n_pc,
                "corr": corr(y_train, model.predict(pcs_train)) [ 0 ]
            })

        r, _ = corr(predictand, yhat)

        self.correlation = round(r, 2)
        self.hindcast = yhat
        self.pc1 = pcs_all [ :, 0 ]  # First PC for all samples (approximate)
        self.lin_model = max(models, key=lambda m: m [ "corr" ])


    """def crossvalpcr(self, xval=True, k=5):
        import numpy as np
        from scipy.stats import linregress, pearsonr as corr
        from albatross.utils import weightsst

        predictand = self.clim_data

        if self.corr_grid.mask.sum() >= len(self.sst.lat) * len(self.sst.lon) - 4:
            self.flags [ "noSST" ] = True
            return

        self.flags [ "noSST" ] = False
        sstidx = ~self.corr_grid.mask
        n = len(predictand)
        rawSSTdata = weightsst(self.sst).data [ :, sstidx ]
        yhat = np.zeros(n)
        pc1_full = np.zeros(n)
        models = [ ]

        if not xval:
            cvr = np.cov(rawSSTdata.T)
            eigval, eigvec = np.linalg.eig(cvr)
            eof_1 = np.real(eigvec [ :, 0 ])
            pc_1 = eof_1.T.dot(rawSSTdata.T).squeeze()
            slope, intercept, r, p, err = linregress(pc_1, predictand)
            self.pc1 = pc_1
            self.correlation = r
            self.hindcast = slope * pc_1 + intercept
            self.lin_model = {
                "eof_1": eof_1,
                "slope": slope,
                "intercept": intercept,
                "corr": r
            }
            return

        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(rawSSTdata):
            X_train, X_test = rawSSTdata [ train_idx ], rawSSTdata [ test_idx ]
            y_train = predictand [ train_idx ]

            cvr = np.cov(X_train.T)
            eigval, eigvec = np.linalg.eig(cvr)
            eof_1 = np.real(eigvec [ :, 0 ])
            pc_1 = eof_1.T.dot(X_train.T).squeeze()

            slope, intercept, r_value, p_value, std_err = linregress(pc_1, y_train)

            # Predict test
            preds = X_test.dot(eof_1) * slope + intercept
            yhat [ test_idx ] = preds
            pc1_full [ test_idx ] = X_test.dot(eof_1)

            models.append({
                "eof_1": eof_1,
                "slope": slope,
                "intercept": intercept,
                "corr": r_value
            })

        r, _ = corr(predictand, yhat)

        self.correlation = round(r, 2)
        self.hindcast = yhat
        self.pc1 = pc1_full

        # Pick best model based on correlation
        best_idx = np.argmax([ m [ "corr" ] for m in models ])
        self.lin_model = models [ best_idx ]"""

