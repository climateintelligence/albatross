#!/usr/bin/env python
"""
New NIPA module
"""

from collections import namedtuple
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from pathlib import Path

seasonal_var = namedtuple("seasonal_var", ("data", "lat", "lon"))


class NIPAphase(object):

    def __init__(self, clim_data, glo_var_name, glo_var, indicator, phaseind, alt=False):
        if alt:
            self.clim_data = clim_data
        else:
            self.clim_data = clim_data[phaseind]
        self.glo_var = seasonal_var(glo_var.data[phaseind], glo_var.lat, glo_var.lon)
        self.glo_var_name = glo_var_name
        self.indicator = indicator[phaseind]
        self.flags = {}
        self.lin_model = None
        self.pcs = None
        self.correlation = None
        self.hindcast = None
        self.phase = None
        self.corr_grid_full = None
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

        fieldData = self.glo_var.data
        clim_data = self.clim_data

        corr_grid = vcorr(X=fieldData, y=clim_data)
        self.corr_grid_full = vcorr(X=fieldData, y=clim_data)  # unfiltered full correlation map
        n_yrs = len(clim_data)
        p_value = sig_test(corr_grid, n_yrs)

        # Mask insignificant gridpoints
        corr_grid = ma.masked_array(corr_grid, ~(p_value < corrlevel))
        # Mask land
        corr_grid = ma.masked_array(corr_grid, isnan(corr_grid))
        # Mask northern/southern ocean
        corr_grid.mask[self.glo_var.lat > 60] = True
        corr_grid.mask[self.glo_var.lat < -60] = True
        nlat = len(self.glo_var.lat)
        nlon = len(self.glo_var.lon)

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
                            dat [ i, j ] = True
                            count += 1
            if debug:
                print("Deleted %i grids" % count)

            self.corr_grid.mask = dat
            self.n_post_grid = dat.size - dat.sum()

        return


    def crossvalpcr(self, xval: bool = True, explained_variance_threshold: float = 0.75):
        """
        PCA/EOF + Linear Regression with optional 5-fold cross-validation.

        If xval=False: fit once on all data (in-sample hindcast).
        If xval=True : 5-fold CV to choose n_pc and produce CV hindcast, then refit on all data
                       with the chosen n_pc and store artifacts.

        Stores on self:
          - hindcast     : np.ndarray (length n_samples)
          - pcs          : PCs from the final refit (all data)
          - correlation  : Pearson r between predictand and hindcast
          - lin_model    : dict with:
                'eofs'      : (n_features_kept, n_pc)
                'regression': sklearn.linear_model.LinearRegression
                'n_pc'      : int
                'x_mean'    : (n_features_kept,)
                'mask_idx'  : boolean mask (nlat*nlon,) mapping kept features in original grid
        """
        import numpy as np
        from scipy.stats import pearsonr as corr
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import KFold
        from collections import defaultdict
        from albatross.utils import weight_glo_var

        # -----------------------------
        # 0) Predictand & feasibility
        # -----------------------------
        y = np.asarray(self.clim_data)  # (n_samples,)
        n_samples = y.shape [ 0 ]

        nlat, nlon = len(self.glo_var.lat), len(self.glo_var.lon)
        # Too few usable grid cells?
        if self.corr_grid.mask.sum() >= (nlat * nlon - 4):
            self.flags [ "noSST" ] = True
            self.hindcast = None
            self.pcs = None
            self.lin_model = None
            self.correlation = None
            print("Insufficient global-variable data for PCA regression.")
            return

        self.flags [ "noSST" ] = False

        # -----------------------------
        # 1) Build X (samples × features)
        # -----------------------------
        # mask_2d: True where we KEEP features
        mask_2d = ~self.corr_grid.mask  # (nlat, nlon)
        mask_1d = mask_2d.ravel()  # (nlat*nlon,)

        X_full = np.asarray(weight_glo_var(self.glo_var).data)  # (n_samples, nlat, nlon)

        # Apply 2-D mask across the last two axes → (n_samples, n_features_raw)
        X = X_full [ :, mask_2d ]

        # Sanity checks and cleaning
        if X.size==0:
            self.flags [ "noSST" ] = True
            self.hindcast = None
            self.pcs = None
            self.lin_model = None
            self.correlation = None
            print("Mask removed all features.")
            return

        # Drop columns with no finite data
        finite_any = np.isfinite(X).any(axis=0)
        X = X [ :, finite_any ]
        mask_1d = mask_1d.copy()
        # Update mask_1d to reflect dropped columns
        # Build an index map from original flattened grid → kept columns
        # We need to compress mask_1d so that only finite_any cols remain.
        # First, get indices of kept columns inside the masked subset:
        kept_within_mask = np.where(mask_1d) [ 0 ] [ finite_any ]
        # Now rebuild a boolean mask over the original flattened grid:
        mask_final = np.zeros(mask_1d.shape [ 0 ], dtype=bool)
        mask_final [ kept_within_mask ] = True

        if X.shape [ 1 ]==0:
            self.flags [ "noSST" ] = True
            self.hindcast = None
            self.pcs = None
            self.lin_model = None
            self.correlation = None
            print("All masked features lacked finite data.")
            return

        # Mean-impute remaining NaNs column-wise (keeps dimensionality)
        if np.isnan(X).any():
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X [ inds ] = np.take(col_means, inds [ 1 ])

        # Remove zero-variance columns (post-imputation)
        var = X.var(axis=0)
        keep_var = var > 0
        X = X [ :, keep_var ]

        if not keep_var.all():
            # reflect in mask_final
            kept_within_mask = kept_within_mask [ keep_var ]
            mask_final = np.zeros(nlat * nlon, dtype=bool)
            mask_final [ kept_within_mask ] = True

        n_features = X.shape [ 1 ]
        if n_features==0:
            self.flags [ "noSST" ] = True
            self.hindcast = None
            self.pcs = None
            self.lin_model = None
            self.correlation = None
            print("No features with non-zero variance.")
            return

        # -----------------------------
        # 2) Helpers: SVD-PCA & n_pc
        # -----------------------------
        def _svd_pca(X0):
            """
            Thin SVD on centered data X0: (n_samples, n_features).
            Returns (U, S, Vt, exp_var_ratio). EOFs = Vt.T (n_features, n_pc).
            """
            U, S, Vt = np.linalg.svd(X0, full_matrices=False)
            S2 = S ** 2
            total = S2.sum()
            if total <= 0 or not np.isfinite(total):
                return U, S, Vt, np.zeros_like(S2)
            evr = S2 / total
            return U, S, Vt, evr

        def _choose_npc(exp_var_ratio, thr):
            cum = np.cumsum(exp_var_ratio)
            k = int(np.searchsorted(cum, thr) + 1)  # +1 to convert index to count
            return max(1, min(k, exp_var_ratio.size))

        # -----------------------------
        # 3) NO-CV branch
        # -----------------------------
        if not xval:
            x_mean = X.mean(axis=0)
            X0 = X - x_mean

            U, S, Vt, evr = _svd_pca(X0)
            if evr.size==0 or not np.isfinite(evr).any():
                self.flags [ "noSST" ] = True
                self.hindcast = None
                self.pcs = None
                self.lin_model = None
                self.correlation = None
                print("PCA failed (no finite variance).")
                return

            n_pc = _choose_npc(evr, explained_variance_threshold)
            eofs = Vt.T [ :, :n_pc ]  # (n_features, n_pc)
            pcs = X0 @ eofs  # (n_samples, n_pc)

            reg = LinearRegression().fit(pcs, y)
            yhat = reg.predict(pcs)

            self.pcs = pcs
            self.hindcast = yhat
            self.correlation = corr(y, yhat) [ 0 ]
            self.lin_model = {
                "eofs": eofs,
                "regression": reg,
                "n_pc": n_pc,
                "x_mean": x_mean,
                "mask_idx": mask_final,  # boolean over original nlat*nlon
            }
            return

        # -----------------------------
        # 4) CV branch (5-fold)
        # -----------------------------
        yhat_cv = np.full(n_samples, np.nan)
        fold_models = [ ]

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for tr, te in kf.split(X):
            X_tr, X_te = X [ tr ], X [ te ]
            y_tr = y [ tr ]

            # Center on TRAIN mean
            x_mean_tr = X_tr.mean(axis=0)
            Xtr0 = X_tr - x_mean_tr
            Xte0 = X_te - x_mean_tr

            U, S, Vt, evr = _svd_pca(Xtr0)
            if evr.size==0 or not np.isfinite(evr).any():
                continue

            n_pc = _choose_npc(evr, explained_variance_threshold)
            pcs_tr = Xtr0 @ Vt.T [ :, :n_pc ]
            if pcs_tr.shape [ 0 ] < n_pc:
                # Not enough samples relative to the number of PCs
                continue

            reg = LinearRegression().fit(pcs_tr, y_tr)
            pcs_te = Xte0 @ Vt.T [ :, :n_pc ]
            y_pred = reg.predict(pcs_te)

            yhat_cv [ te ] = y_pred
            r_train = corr(y_tr, reg.predict(pcs_tr)) [ 0 ]
            fold_models.append({"n_pc": n_pc, "r_train": r_train})

        # If CV produced no valid predictions/models
        if not np.isfinite(yhat_cv).any() or not fold_models:
            self.flags [ "noSST" ] = True
            self.hindcast = None
            self.pcs = None
            self.lin_model = None
            self.correlation = None
            print("Cross-validation failed (no valid folds).")
            return

        valid = np.isfinite(yhat_cv)
        self.hindcast = np.where(valid, yhat_cv, np.nan)
        self.correlation = corr(y [ valid ], yhat_cv [ valid ]) [ 0 ] if valid.any() else np.nan

        # Select n_pc by median training r (robust to outliers)
        acc = defaultdict(list)
        for fm in fold_models:
            acc [ fm [ "n_pc" ] ].append(fm [ "r_train" ])
        n_pc_best = max(acc.items(), key=lambda kv: np.nanmedian(kv [ 1 ])) [ 0 ]

        # Final refit on ALL data with chosen n_pc
        x_mean_all = X.mean(axis=0)
        X0_all = X - x_mean_all
        U, S, Vt, evr_all = _svd_pca(X0_all)
        if evr_all.size==0 or not np.isfinite(evr_all).any():
            self.flags [ "noSST" ] = True
            self.pcs = None
            self.lin_model = None
            self.correlation = None
            print("Final PCA refit failed.")
            return

        rank = min(X0_all.shape)  # numerical rank upper bound
        n_pc_final = int(max(1, min(n_pc_best, rank)))
        eofs_all = Vt.T [ :, :n_pc_final ]
        pcs_full = X0_all @ eofs_all
        reg_full = LinearRegression().fit(pcs_full, y)

        self.pcs = pcs_full
        self.lin_model = {
            "eofs": eofs_all,
            "regression": reg_full,
            "n_pc": n_pc_final,
            "x_mean": x_mean_all,
            "mask_idx": mask_final,  # boolean mask over original flattened grid (nlat*nlon,)
        }

    def save_regressor(self, workdir):
        """
        Save EOF regression model artifacts for offline use.

        Produces:
          - pc_vs_hindcast/eofs_<phase>.csv         → EOF loading patterns (features × PCs)
          - pc_vs_hindcast/coefficients_<phase>.csv → linear regression coefficients
          - pc_vs_hindcast/glo_var_mask_<phase>.npy → 2-D mask of the global variable field
          - pc_vs_hindcast/pc_transform_<phase>.npz → compressed package with EOFs, mean, mask, and n_pc
        """
        import numpy as np
        import pandas as pd
        from pathlib import Path

        if not getattr(self, "lin_model", None):
            print("⚠️ No regression model found. Skipping save.")
            return

        lin_model = self.lin_model
        if "eofs" not in lin_model or "regression" not in lin_model:
            print("⚠️ Incomplete regression model (missing EOFs or regressor). Skipping save.")
            return

        eofs = lin_model [ "eofs" ]  # (n_features_kept, n_pc)
        reg = lin_model [ "regression" ]
        x_mean = lin_model.get("x_mean", None)
        mask_idx = lin_model.get("mask_idx", None)
        n_pc = lin_model.get("n_pc", eofs.shape [ 1 ])

        out_dir = Path(workdir) / "pc_vs_hindcast"
        out_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # 1) Save EOF loadings
        # -----------------------------
        eofs_path = out_dir / f"eofs_{self.phase}.csv"
        eofs_df = pd.DataFrame(eofs, columns=[ f"PC{i + 1}" for i in range(eofs.shape [ 1 ]) ])
        eofs_df.to_csv(eofs_path, index=False)

        # -----------------------------
        # 2) Save regression coefficients
        # -----------------------------
        coef_df = pd.DataFrame(
            {"Coefficient": np.r_ [ reg.coef_, reg.intercept_ ]},
            index=[ f"PC{i + 1}" for i in range(n_pc) ] + [ "Intercept" ]
        )
        coef_path = out_dir / f"coefficients_{self.phase}.csv"
        coef_df.to_csv(coef_path)

        # -----------------------------
        # 3) Save the 2-D correlation mask
        # -----------------------------
        mask_2d_path = out_dir / f"glo_var_mask_{self.phase}.npy"
        np.save(mask_2d_path, getattr(self.corr_grid, "mask", np.array([ ])))

        # -----------------------------
        # 4) Save unified NPZ for projection
        # -----------------------------
        npz_path = out_dir / f"pc_transform_{self.phase}.npz"
        np.savez_compressed(
            npz_path,
            eofs=eofs,
            x_mean=x_mean if x_mean is not None else np.array([ ], dtype=float),
            mask_idx=mask_idx if mask_idx is not None else np.array([ ], dtype=bool),
            n_pc=np.array([ n_pc ], dtype=int),
        )

        print(f"✅ Regressor saved to {out_dir}")
        print(f" - EOFs:           {eofs_path.name}")
        print(f" - Coefficients:   {coef_path.name}")
        print(f" - Mask:           {mask_2d_path.name}")
        print(f" - NPZ package:    {npz_path.name}")
