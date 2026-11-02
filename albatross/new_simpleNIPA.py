#!/usr/bin/env python
"""
New NIPA module
"""
from collections import namedtuple
import numpy as np
 
seasonal_var = namedtuple("seasonal_var", ("data", "lat", "lon"))


def linear_trend_fit(t, Z):
    """Trend lineare per colonna, NaN ignorati. t: (n,), Z: (n, ...).
    Restituisce (media, pendenza, t medio), da usare con linear_trend_eval."""
    t = np.asarray(t, dtype=float)
    Z = np.asarray(Z, dtype=float)
    tc = (t - t.mean()).reshape(-1, *[1] * (Z.ndim - 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        Zm = np.nanmean(Z, axis=0)
        slope = np.nansum(tc * (Z - Zm), axis=0) / np.sum(tc ** 2)
    return Zm, slope, float(t.mean())


def linear_trend_eval(fit, t):
    """Valore del trend agli istanti t (estrapolazione inclusa). Forma: (len(t), ...)."""
    Zm, slope, tbar = fit
    t = np.atleast_1d(np.asarray(t, dtype=float))
    Zm, slope = np.asarray(Zm), np.asarray(slope)
    return Zm[None] + slope[None] * (t - tbar).reshape(-1, *[1] * slope.ndim)


def linear_detrend(t, Z):
    """Residui rispetto al trend lineare stimato su tutta la serie (solo per diagnostica e mappe)."""
    return np.asarray(Z, dtype=float) - linear_trend_eval(linear_trend_fit(t, Z), t)
 
class NIPAphase(object):
 
    def __init__(self, clim_data, glo_var_name, glo_var, indicator, phaseind, alt=False):
        if alt or phaseind is None:
            self.clim_data = np.asarray(clim_data)
            self.glo_var = seasonal_var(np.asarray(glo_var.data), glo_var.lat, glo_var.lon)
            self.indicator = np.asarray(indicator)
        else:
            self.clim_data = np.asarray(clim_data) [ phaseind ]
            self.glo_var = seasonal_var(np.asarray(glo_var.data) [ phaseind ], glo_var.lat, glo_var.lon)
            self.indicator = np.asarray(indicator) [ phaseind ]
 
        self.glo_var_name = glo_var_name
        self.flags = {}
        self.lin_model = None
        self.pcs = None
        self.correlation = None
        self.hindcast = None
        self.phase = None
        self.corr_grid_full = None
        self.corr_grid_mean = None
        self.corr_grid = None
        self.n_pre_grid = 0
        self.n_post_grid = 0
        return
 
    def bootcorr(
        self, ntim=1000, corrconf=0.95, bootconf=0.80, debug=False, quick=True, seed = None,
        lat_limit=60.0, orog_max_m=20000.0,
    ):
        from numpy import isnan, ma
        from albatross.utils import sig_test, vcorr
        rng = np.random.default_rng(seed)
 
        corrlevel = 1 - corrconf
 
        fieldData = self.glo_var.data
        clim_data = self.clim_data
 
        corr_grid = vcorr(X=fieldData, y=clim_data)
        self.corr_grid_full = corr_grid.copy()
        n_yrs = len(clim_data)
        p_value = sig_test(corr_grid, n_yrs)
 
        # Mask insignificant gridpoints
        corr_grid = ma.masked_array(corr_grid, ~(p_value < corrlevel))
 
        # Mask NaNs (if any)
        corr_grid = ma.masked_array(corr_grid, isnan(corr_grid))
 
        lat_arr = np.asarray(self.glo_var.lat, dtype=float)
        # Alte latitudini mascherate per TUTTE le variabili (prima solo sst/tp/t2m)
        if lat_limit is not None:
            corr_grid.mask [ np.abs(lat_arr) > lat_limit ] = True

        # SLP sopra orografia alta = estrapolazione al livello del mare
        if self.glo_var_name == "slp" and orog_max_m is not None:
            from albatross.utils import high_orography_mask
            corr_grid.mask |= high_orography_mask(lat_arr, self.glo_var.lon, orog_max_m)
 
        nlat = len(self.glo_var.lat)
        nlon = len(self.glo_var.lon)
 
 
        if quick:
            self.corr_grid = corr_grid
            self.corr_grid_full = np.ma.masked_array(self.corr_grid_full, mask=corr_grid.mask)
            self.n_pre_grid = nlat * nlon - corr_grid.mask.sum()
            if self.n_pre_grid == 0:
                self.flags["no_glo_var"] = True
            else:
                self.flags["no_glo_var"] = False
            return
 
        # INITIALIZE A NEW CORR GRID
        count = np.zeros((nlat, nlon))
        dat = clim_data.copy()

        from scipy.stats import t as tdist
        # sig_test() calls scipy's generic t CDF on the whole grid, which dominates
        # the bootstrap cost. The test |t| > t_crit is equivalent and needs the
        # inverse CDF only once, since df and corrlevel are fixed here.
        df = n_yrs - 2
        t_crit = float(tdist.isf(corrlevel / 2.0, df))   # two-tailed, as sig_test
 
        for boot in range(ntim):
            idx = rng.integers(0, len(dat), size=len(dat))
 
            boot_fieldData = np.zeros((len(idx), nlat, nlon))
            boot_fieldData[:] = fieldData[idx]
            boot_climData = dat[idx]
 
            boot_corr_grid = vcorr(X=boot_fieldData, y=boot_climData)
            with np.errstate(divide="ignore", invalid="ignore"):
                t_stat = np.abs(boot_corr_grid) * np.sqrt(
                    df / np.maximum(1.0 - boot_corr_grid ** 2, 1e-300)
                )
            count[t_stat >= t_crit] += 1
            if debug:
                print("Count max is %i" % count.max())
 
        # CREATE MASKED ARRAY USING THE COUNT AND BOOTCONF ATTRIBUTES
        corr_grid = np.ma.masked_array(corr_grid, count < bootconf * ntim)
        self.corr_grid_full = np.ma.masked_array(self.corr_grid_full, mask=corr_grid.mask)
 
        self.corr_grid = corr_grid
        self.n_pre_grid = nlat * nlon - corr_grid.mask.sum()
 
        if self.n_pre_grid == 0:
            self.flags["no_glo_var"] = True
        else:
            self.flags["no_glo_var"] = False
        return
 
    def gridCheck(self, lim=5, ntim=2, debug=False, edges="legacy"):
        """
        Rimuove pixel isolati: un pixel tenuto viene scartato se almeno `lim` degli
        8 vicini sono mascherati.

        edges="legacy": comportamento originale. Prima/ultima riga (poli) e
            prima/ultima colonna (lon 0 e lon max) NON vengono mai controllate,
            quindi pixel isolati sui bordi sopravvivono sempre.
        edges="fixed": controlla TUTTE le celle; longitudine periodica se la
            griglia e' globale (lon max e lon 0 sono vicine); oltre i poli i
            vicini contano come mascherati.
        """
        if self.n_pre_grid < 50:
            lim = 6
            ntim = 2
        if edges == "fixed":
            return self._gridCheck_fixed(lim=lim, ntim=ntim, debug=debug)
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

            # gridCheck mutates corr_grid.mask, so corr_grid_full must follow or the
            # binary map and the correlation map will show different retained pixels.
            if self.corr_grid_full is not None:
                self.corr_grid_full = np.ma.masked_array(
                np.ma.getdata(self.corr_grid_full), mask=self.corr_grid.mask
            )
 
        return
 
    def _gridCheck_fixed(self, lim=5, ntim=2, debug=False):
        dat = np.array(np.ma.getmaskarray(self.corr_grid), dtype=bool)
        nlat, nlon = dat.shape
        lon = np.asarray(self.glo_var.lon, dtype=float)
        dlon = float(np.median(np.diff(lon))) if lon.size > 1 else 0.0
        periodic = lon.size > 1 and abs((lon[-1] - lon[0] + dlon) - 360.0) < 0.5 * abs(dlon)

        for _ in range(ntim):
            count = 0
            # stessa scansione sequenziale in-place dell'originale
            for i in range(nlat):
                for j in range(nlon):
                    if dat[i, j]:
                        continue
                    n_masked = 0
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            if di == 0 and dj == 0:
                                continue
                            ii, jj = i + di, j + dj
                            if ii < 0 or ii >= nlat:
                                n_masked += 1           # oltre il polo
                                continue
                            if jj < 0 or jj >= nlon:
                                if not periodic:
                                    n_masked += 1       # bordo di una griglia regionale
                                    continue
                                jj %= nlon
                            n_masked += int(dat[ii, jj])
                    if n_masked >= lim:
                        dat[i, j] = True
                        count += 1
            if debug:
                print("Deleted %i grids" % count)

        self.corr_grid.mask = dat
        self.n_post_grid = int(dat.size - dat.sum())
        if self.corr_grid_full is not None:
            self.corr_grid_full = np.ma.masked_array(
                np.ma.getdata(self.corr_grid_full), mask=self.corr_grid.mask
            )
        return

    def crossvalpcr(self, xval: bool = True, explained_variance_threshold: float = 0.75,
                    n_pc_fixed=None, regressor: str = "ridge"):
        """
        PCR with optional CV, hardened against overfitting.
 
        Changes vs your version:
          - Ridge regression (shrinkage) instead of OLS
          - n_pc is selected by validation RMSE (not correlation)
          - explicit hard/soft caps on n_pc based on N_train
          - grid-search over (n_pc, alpha) inside each fold (true model selection)
          - stores col_means to allow consistent application to held-out data later
 
        Notes:
          - This still assumes corr_grid (bootcorr+gridCheck) is already computed on the data
            you intend to model. If you want zero leakage, corr_grid must be recomputed per fold.
          - Uses 5-fold CV by default; with very small N you should prefer LOOCV.
        """
        import numpy as np
        from scipy.stats import pearsonr as corr
        from sklearn.linear_model import Ridge, LinearRegression
        from sklearn.model_selection import KFold

        # regressor="ols"  -> regressione lineare classica (minimi quadrati), in TUTTI i rami
        # regressor="ridge"-> comportamento precedente (Ridge alpha fisso)
        def _new_reg(a):
            return LinearRegression() if regressor == "ols" else Ridge(alpha=a)
        from collections import defaultdict
        from albatross.utils import weight_glo_var
 
        # -----------------------------
        # 0) Predictand & feasibility
        # -----------------------------
        y = np.asarray(self.clim_data).ravel()
        n_samples = y.shape [ 0 ]
 
        nlat, nlon = len(self.glo_var.lat), len(self.glo_var.lon)
        if self.corr_grid.mask.sum() >= (nlat * nlon - 4):
            self.flags [ "no_glo_var" ] = True
            self.hindcast = None
            self.pcs = None
            self.lin_model = None
            self.correlation = None
            print("Insufficient global-variable data for PCA regression.")
            return
        self.flags [ "no_glo_var" ] = False
 
        # -----------------------------
        # 1) Build X (samples × features)
        # -----------------------------
        mask_2d = ~self.corr_grid.mask
        mask_1d = mask_2d.ravel()
 
        X_full = np.asarray(weight_glo_var(self.glo_var).data)  # (n_samples, nlat, nlon)
        X = X_full [ :, mask_2d ]  # (n_samples, n_features_raw)
 
        if X.size==0:
            self.flags [ "no_glo_var" ] = True
            self.hindcast = None
            self.pcs = None
            self.lin_model = None
            self.correlation = None
            print("Mask removed all features.")
            return
 
        # Drop columns with no finite data (global)
        finite_any = np.isfinite(X).any(axis=0)
        X = X [ :, finite_any ]
 
        kept_within_mask = np.where(mask_1d) [ 0 ] [ finite_any ]
        mask_final = np.zeros(mask_1d.shape [ 0 ], dtype=bool)
        mask_final [ kept_within_mask ] = True
 
        if X.shape [ 1 ]==0:
            self.flags [ "no_glo_var" ] = True
            self.hindcast = None
            self.pcs = None
            self.lin_model = None
            self.correlation = None
            print("All masked features lacked finite data.")
            return
 
        # Mean-impute NaNs (global) and STORE col_means for later application
        col_means = None
        if np.isnan(X).any():
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X [ inds ] = np.take(col_means, inds [ 1 ])
 
        # Remove zero-variance columns
        var = X.var(axis=0)
        keep_var = var > 0
        X = X [ :, keep_var ]
        if col_means is not None:
            col_means = col_means [ keep_var ]
 
        if not keep_var.all():
            kept_within_mask = kept_within_mask [ keep_var ]
            mask_final = np.zeros(nlat * nlon, dtype=bool)
            mask_final [ kept_within_mask ] = True
 
        n_features = X.shape [ 1 ]
        if n_features==0:
            self.flags [ "no_glo_var" ] = True
            self.hindcast = None
            self.pcs = None
            self.lin_model = None
            self.correlation = None
            print("No features with non-zero variance.")
            return
 
        # -----------------------------
        # 2) Helpers: SVD-PCA
        # -----------------------------
        def _svd_pca(X0):
            U, S, Vt = np.linalg.svd(X0, full_matrices=False)
            S2 = S ** 2
            tot = S2.sum()
            evr = (S2 / tot) if (tot > 0 and np.isfinite(tot)) else np.zeros_like(S2)
            return U, S, Vt, evr
 
        def _choose_npc_from_evr(evr, thr):
            cum = np.cumsum(evr)
            k = int(np.searchsorted(cum, thr) + 1)
            return max(1, min(k, evr.size))
 
        def _rmse(a, b):
            a = np.asarray(a).ravel()
            b = np.asarray(b).ravel()
            return float(np.sqrt(np.mean((a - b) ** 2)))
 
        # ------------------------------------------------------
        # 3) No-CV branch (in-sample) — still apply hard caps
        # ------------------------------------------------------
        if not xval:
            x_mean = X.mean(axis=0)
            X0 = X - x_mean
 
            U, S, Vt, evr = _svd_pca(X0)
            if evr.size==0 or not np.isfinite(evr).any():
                self.flags [ "no_glo_var" ] = True
                self.hindcast = None
                self.pcs = None
                self.lin_model = None
                self.correlation = None
                print("PCA failed (no finite variance).")
                return
 
            if n_pc_fixed is None:
                n_pc = _choose_npc_from_evr(evr, explained_variance_threshold)
            else:
                n_pc = int(n_pc_fixed)
 
            # Hard/soft caps (strong anti-overfit)
            soft_cap = max(1, n_samples // 3)  # use //4 if very small phases
            hard_cap = max(1, n_samples - 2)
            rank_cap = min(X0.shape [ 0 ] - 1, X0.shape [ 1 ])
            n_pc = int(min(n_pc, soft_cap, hard_cap, rank_cap))
 
            eofs = Vt.T [ :, :n_pc ]
            pcs = X0 @ eofs
 
            if regressor == "ols":
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression().fit(pcs, y)
            else:
                reg = Ridge(alpha=10.0).fit(pcs, y)

            yhat = reg.predict(pcs)
 
            self.pcs = pcs
            self.hindcast = yhat
            self.correlation = corr(y, yhat) [ 0 ]
            self.lin_model = {
                "eofs": eofs,
                "regression": reg,
                "n_pc": n_pc,
                "x_mean": x_mean,
                "mask_idx": mask_final,
                "col_means": col_means,
            }
            return
 
        # -----------------------------
        # 4) CV branch (5-fold) with (n_pc, alpha) grid search
        # -----------------------------
        # If N is tiny, 5-fold can be unstable; at least ensure feasible splits.
        n_splits = 5
        if n_samples < 10:
            n_splits = max(2, min(5, n_samples))  # degrade gracefully (still not LOOCV)
 
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
 
        #yhat_cv = np.full(n_samples, np.nan)
        #fold_choices = [ ]  # store chosen (k, alpha, rmse) per fold
        yhat_cv = np.full(n_samples, np.nan)
        acc = defaultdict(list)   # {(k, alpha): [rmse per fold]}
        fold_store = [ ]          # [(te, {(k,alpha): pred})]
        n_folds_done = 0
 
        # Candidate alphas (shrinkage strength)
        alpha_grid = [10.0]
 
        for tr, te in kf.split(X):
            X_tr, X_te = X [ tr ], X [ te ]
            y_tr, y_te = y [ tr ], y [ te ]
 
            # Center with TRAIN mean
            x_mean_tr = X_tr.mean(axis=0)
            Xtr0 = X_tr - x_mean_tr
            Xte0 = X_te - x_mean_tr
 
            U, S, Vt, evr = _svd_pca(Xtr0)
            if evr.size==0 or not np.isfinite(evr).any():
                continue
 
            # Starting point from EVR, but then we SEARCH around it under caps
            k0 = _choose_npc_from_evr(evr, explained_variance_threshold)
 
            Ntr = X_tr.shape [ 0 ]
            rank_cap = min(Xtr0.shape [ 0 ] - 1, Xtr0.shape [ 1 ])
            hard_cap = max(1, Ntr - 2)
            soft_cap = max(1, Ntr // 3)  # or //4 for even stricter
            k_max = int(min(rank_cap, hard_cap, soft_cap))
 
            if k_max < 1:
                continue
 
            # Candidate k grid: 1..k_max (keep it small)
            # If you want faster: test only up to min(k_max, 8)
            k_grid = range(1, min(k_max, 8) + 1)
 
            """best = None
            for k in k_grid:
                eofs_k = Vt.T [ :, :k ]
                pcs_tr = Xtr0 @ eofs_k
                pcs_te = Xte0 @ eofs_k
 
                for a in alpha_grid:
                    reg = _new_reg(a).fit(pcs_tr, y_tr)
                    pred = reg.predict(pcs_te)
                    rmse = _rmse(y_te, pred)
 
                    if (best is None) or (rmse < best [ "rmse" ]):
                        best = {"k": k, "alpha": a, "rmse": rmse, "pred": pred}
 
            if best is None:
                continue
 
            yhat_cv [ te ] = best [ "pred" ]
            fold_choices.append(best)"""

            fold_preds = {}
            for k in k_grid:
                eofs_k = Vt.T [ :, :k ]
                pcs_tr = Xtr0 @ eofs_k
                pcs_te = Xte0 @ eofs_k

                for a in alpha_grid:
                    reg = _new_reg(a).fit(pcs_tr, y_tr)
                    pred = reg.predict(pcs_te)
                    acc [ (k, a) ].append(_rmse(y_te, pred))
                    fold_preds [ (k, a) ] = pred

            if not fold_preds:
                continue

            fold_store.append((te, fold_preds))
            n_folds_done += 1
 
        """if not np.isfinite(yhat_cv).any() or not fold_choices:
            self.flags [ "no_glo_var" ] = True
            self.hindcast = None
            self.pcs = None
            self.lin_model = None
            self.correlation = None
            print("Cross-validation failed (no valid folds).")
            return
 
        valid = np.isfinite(yhat_cv)
        self.hindcast = np.where(valid, yhat_cv, np.nan)
        self.correlation = corr(y [ valid ], yhat_cv [ valid ]) [ 0 ] if valid.any() else np.nan
 
        # -----------------------------
        # 5) Choose global (k, alpha) by median RMSE across folds
        # -----------------------------
        acc = defaultdict(list)
        fold_store = []
        for fc in fold_choices:
            acc [ (fc [ "k" ], fc [ "alpha" ]) ].append(fc [ "rmse" ])
 
        (k_best, a_best) = min(acc.items(), key=lambda kv: np.nanmedian(kv [ 1 ])) [ 0 ]"""

        if not acc:
            self.flags [ "no_glo_var" ] = True
            self.hindcast = None
            self.pcs = None
            self.lin_model = None
            self.correlation = None
            print("Cross-validation failed (no valid folds).")
            return

        # -----------------------------
        # 5) Choose global (k, alpha) by median RMSE across folds
        # -----------------------------
        full = {ka: v for ka, v in acc.items() if len(v)==n_folds_done} or acc
        (k_best, a_best) = min(full.items(), key=lambda kv: np.nanmedian(kv [ 1 ])) [ 0 ]
        # --- DIAGNOSTICA: superficie RMSE su (k, alpha) ---
        print(f"[DIAG] fase={self.phase} n={n_samples} feat={n_features} scelta k={k_best} "
              + ("OLS" if regressor == "ols" else f"alpha={a_best}"))
        meds = {ka: float(np.nanmedian(v)) for ka, v in full.items()}
        lo, hi = min(meds.values()), max(meds.values())
        print(f"[DIAG]   RMSE mediano: min={lo:.1f} max={hi:.1f} spread={100*(hi-lo)/lo:.1f}%")
        for ka in sorted(meds, key=meds.get)[:5]:
            print(f"[DIAG]   k={ka[0]} alpha={ka[1]:<6} rmse={meds[ka]:.2f}")
        # quali coppie sceglieva ogni singolo fold (il vecchio comportamento)
        per_fold = [min(fp, key=lambda ka: float(np.sqrt(np.mean((y[te]-fp[ka])**2))))
                    for te, fp in fold_store]
        print(f"[DIAG]   scelte per-fold: {per_fold}")

        # Hindcast: previsione di ogni fold con la coppia (k, alpha) GLOBALE,
        # scelta aggregando sui fold e non dentro il singolo fold.
        for te, fold_preds in fold_store:
            if (k_best, a_best) in fold_preds:
                yhat_cv [ te ] = fold_preds [ (k_best, a_best) ]

        valid = np.isfinite(yhat_cv)
        self.hindcast = np.where(valid, yhat_cv, np.nan)
        self.correlation = corr(y [ valid ], yhat_cv [ valid ]) [ 0 ] if valid.any() else np.nan
 
        # -----------------------------
        # 6) Final refit on ALL data with chosen (k_best, a_best)
        # -----------------------------
        x_mean_all = X.mean(axis=0)
        X0_all = X - x_mean_all
        U, S, Vt, evr_all = _svd_pca(X0_all)
        if evr_all.size==0 or not np.isfinite(evr_all).any():
            self.flags [ "no_glo_var" ] = True
            self.pcs = None
            self.lin_model = None
            self.correlation = None
            print("Final PCA refit failed.")
            return
 
        rank_cap = min(X0_all.shape [ 0 ] - 1, X0_all.shape [ 1 ])
        hard_cap = max(1, n_samples - 2)
        soft_cap = max(1, n_samples // 3)
        k_final = int(min(k_best, rank_cap, hard_cap, soft_cap))
 
        eofs_all = Vt.T [ :, :k_final ]
        pcs_full = X0_all @ eofs_all
 
        reg_full = _new_reg(a_best).fit(pcs_full, y)
 
        self.pcs = pcs_full
        self.lin_model = {
            "eofs": eofs_all,
            "regression": reg_full,
            "n_pc": k_final,
            "x_mean": x_mean_all,
            "mask_idx": mask_final,
            "col_means": col_means,
            "ridge_alpha": float(a_best) if regressor != "ols" else None,
            "regressor": regressor,
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
 
        col_means = lin_model.get("col_means", None)
 
        coef = np.asarray(reg.coef_, dtype=float).ravel()
        intercept = float(np.asarray(reg.intercept_).ravel() [ 0 ]) if np.ndim(reg.intercept_) else float(
            reg.intercept_)
 
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
            col_means=col_means if col_means is not None else np.array([ ], dtype=float),
            coef=coef,
            intercept=np.array([ intercept ], dtype=float),
        )
 
        print(f"✅ Regressor saved to {out_dir}")
 
 
    def _set_training_subset(self, X_sub: np.ndarray, y_sub: np.ndarray):
        """
        Replace the model's internal training arrays with a subset (for fold-wise bootcorr).
        X_sub: (n_samples, nlat, nlon)
        y_sub: (n_samples,)
        """
        self.clim_data = np.asarray(y_sub).ravel()
 
        # Preserve the same container type as self.glo_var (seasonal_var-like)
        gv_type = type(self.glo_var)
        self.glo_var = gv_type(np.asarray(X_sub), self.glo_var.lat, self.glo_var.lon)
 
    def build_consensus_corr_mask(
        self,
        cv_splitter,
        q: float = 0.7,
        mode: str = "vote",
        bootcorr_kwargs: dict | None = None,
        gridcheck_kwargs: dict | None = None,
    ):
        import numpy as np
        from numpy import ma
        from albatross.utils import vcorr  # NEW: Import helper
 
        bootcorr_kwargs = bootcorr_kwargs or {}
        gridcheck_kwargs = gridcheck_kwargs or {}
 
        X_all = np.asarray(self.glo_var.data)
        y_all = np.asarray(self.clim_data).ravel()
        _, nlat, nlon = X_all.shape
 
        counts = np.zeros((nlat, nlon), dtype=int)
        n_folds = 0
 
        gv_orig = self.glo_var
        y_orig = self.clim_data
 
        try:
            for tr, _ in cv_splitter.split(y_all):
                X_tr = X_all [ tr ]
                y_tr = y_all [ tr ]
 
                self._set_training_subset(X_tr, y_tr)
                self.bootcorr(**bootcorr_kwargs)
                if self.flags.get("no_glo_var", False):
                    continue
                self.gridCheck(**gridcheck_kwargs)
 
                keep = ~self.corr_grid.mask
                counts += keep.astype(int)
                n_folds += 1
        finally:
            self.glo_var = gv_orig
            self.clim_data = y_orig
 
        if n_folds==0:
            raise RuntimeError("Consensus mask failed: no valid folds produced a corr_grid.")
 
        if mode=="union":
            consensus_keep = counts >= 1
        else:
            thresh = int(np.ceil(q * n_folds))
            consensus_keep = counts >= thresh
 
        # 1. Calculate actual r-values for the full phase
        full_r_values = vcorr(X=X_all, y=y_all)
 
        # 2. Store them in corr_grid_full for the "Proportional" plot
        self.corr_grid_full = ma.masked_array(full_r_values, mask=~consensus_keep)
 
        # Binary mask (all zeros) for the "Blue" plot
        self.corr_grid = ma.masked_array(np.zeros((nlat, nlon), dtype=float), mask=~consensus_keep)
 
        self.n_pre_grid = int(consensus_keep.sum())
        self.flags [ "no_glo_var" ] = (self.n_pre_grid==0)
 
        return consensus_keep, counts
 
    def predict_from_pcr(self, X_val: np.ndarray) -> np.ndarray:
        pcs_val = self.project_pcs(X_val)
        y_pred = self.lin_model [ "regression" ].predict(pcs_val)
        return np.asarray(y_pred).ravel()

    def project_pcs(self, X_val: np.ndarray) -> np.ndarray:
        """Proietta campi (2D o 3D) sulle EOF di lin_model, con lo stesso preprocessing
        del training (pesi, maschera, imputazione, centratura). Ritorna (n, n_pc)."""
        import numpy as np
        from albatross.utils import weight_glo_var

        if self.lin_model is None:
            raise ValueError("lin_model is None; run crossvalpcr(xval=False) first.")
 
        m = self.lin_model
 
        X_val = np.asarray(X_val)
        if X_val.ndim==2:
            X_val = X_val [ None, :, : ]
        if X_val.ndim!=3:
            raise ValueError(f"X_val must be 2D or 3D, got shape {X_val.shape}")
 
        # Weight exactly like training
        tmp = seasonal_var(X_val.copy(), self.glo_var.lat, self.glo_var.lon)
        Xv_full = np.asarray(weight_glo_var(tmp).data)  # (n, nlat, nlon)
 
        # Apply mask
        Xv = Xv_full.reshape(Xv_full.shape [ 0 ], -1) [ :, m [ "mask_idx" ] ]
 
        # Impute NaNs using TRAIN col_means
        if np.isnan(Xv).any():
            col_means = m.get("col_means", None)
            if col_means is None or np.asarray(col_means).size!=Xv.shape [ 1 ]:
                raise ValueError("col_means missing or wrong size; cannot impute NaNs consistently.")
            inds = np.where(np.isnan(Xv))
            Xv [ inds ] = np.take(col_means, inds [ 1 ])
 
        # Center + project
        Xv0 = Xv - m [ "x_mean" ]
        return Xv0 @ m [ "eofs" ]
 
    def _build_X_from_current_mask(self):
        """
        Build (X, meta) using the CURRENT self.corr_grid.mask, matching crossvalpcr() preprocessing.
        Returns
        -------
        X : np.ndarray (n_samples, n_features)
        meta : dict with keys:
            - mask_idx   : boolean (nlat*nlon,) final mask over flattened grid
            - col_means  : (n_features,) or None  (for NaN imputation)
        """
        import numpy as np
        from albatross.utils import weight_glo_var
        y = np.asarray(self.clim_data).ravel()
        X3 = np.asarray(weight_glo_var(self.glo_var).data)  # (n_samples, nlat, nlon)
 
        n_samples, nlat, nlon = X3.shape
        mask_2d = ~self.corr_grid.mask
        mask_1d = mask_2d.ravel()
 
        X = X3 [ :, mask_2d ]
        if X.size==0:
            return None, None
 
        # drop all-nonfinite columns
        finite_any = np.isfinite(X).any(axis=0)
        X = X [ :, finite_any ]
        kept_within_mask = np.where(mask_1d) [ 0 ] [ finite_any ]
 
        mask_final = np.zeros(nlat * nlon, dtype=bool)
        mask_final [ kept_within_mask ] = True
 
        if X.shape [ 1 ]==0:
            return None, None
 
        # impute NaNs
        col_means = None
        if np.isnan(X).any():
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X [ inds ] = np.take(col_means, inds [ 1 ])
 
        # remove zero-variance columns
        var = X.var(axis=0)
        keep_var = var > 0
        X = X [ :, keep_var ]
        if col_means is not None:
            col_means = col_means [ keep_var ]
 
        if not keep_var.all():
            kept_within_mask = kept_within_mask [ keep_var ]
            mask_final = np.zeros(nlat * nlon, dtype=bool)
            mask_final [ kept_within_mask ] = True
 
        if X.shape [ 1 ]==0:
            return None, None
 
        meta = {"mask_idx": mask_final, "col_means": col_means}
        return X, meta
 
    def _fit_pcr_and_predict_one(self, X_val_2d, explained_variance_threshold=0.75):
        """
        Fit PCR (no internal CV) on the CURRENT self data & CURRENT self.corr_grid,
        then predict for a single validation field (2D array).
        """
        import numpy as np
        from sklearn.linear_model import Ridge
        from scipy.stats import pearsonr as corr
 
        # Build training X from current mask
        X, meta = self._build_X_from_current_mask()
        if X is None:
            return None
 
        y = np.asarray(self.clim_data).ravel()
        n_samples = y.shape [ 0 ]
 
        # Center
        x_mean = X.mean(axis=0)
        X0 = X - x_mean
 
        # PCA via SVD
        U, S, Vt = np.linalg.svd(X0, full_matrices=False)
        S2 = S ** 2
        tot = S2.sum()
        if tot <= 0 or not np.isfinite(tot):
            return None
        evr = S2 / tot
 
        # choose npc by EVR then CAP HARD
        cum = np.cumsum(evr)
        n_pc = int(np.searchsorted(cum, explained_variance_threshold) + 1)
        n_pc = max(1, min(n_pc, evr.size))
 
        # hard/soft caps for small N (strong anti-overfit)
        soft_cap = max(1, n_samples // 3)  # use //4 if you want stricter
        hard_cap = max(1, n_samples - 2)
        rank_cap = min(X0.shape [ 0 ] - 1, X0.shape [ 1 ])
        n_pc = int(min(n_pc, soft_cap, hard_cap, rank_cap))
 
        eofs = Vt.T [ :, :n_pc ]
        pcs = X0 @ eofs
 
        # Ridge regression (fixed alpha; adjust if needed)
        reg = Ridge(alpha=10.0).fit(pcs, y)
 
        # ---- Apply to validation field ----
        # Build X_val feature vector in SAME feature space
        # 1) Weighting must match training (done here using weight_glo_var on seasonal_var)
        from collections import namedtuple
        from albatross.utils import weight_glo_var
        seasonal_var_local = namedtuple("seasonal_var", ("data", "lat", "lon"))
        tmp = seasonal_var_local(np.asarray(X_val_2d) [ None, :, : ], self.glo_var.lat, self.glo_var.lon)
        Xv3 = np.asarray(weight_glo_var(tmp).data)  # (1, nlat, nlon)
 
        x_flat = Xv3.reshape(1, -1) [ 0 ]
        mask_idx = meta [ "mask_idx" ]
        x_sel = x_flat [ mask_idx ]
 
        # impute NaNs with TRAIN col_means
        if np.isnan(x_sel).any():
            col_means = meta [ "col_means" ]
            if col_means is None:
                return None
            nan = np.isnan(x_sel)
            x_sel [ nan ] = col_means [ nan ]
 
        # center + project + predict
        pcs_v = (x_sel - x_mean) @ eofs
        y_pred = float(reg.predict(pcs_v.reshape(1, -1)) [ 0 ])
        return y_pred
 
    def _consensus_mask_for_heldout(
        self,
        heldout_i: int,
        q: float = 0.6,
        corrconf: float = 0.9,
        bootconf: float = 0.6,
        ntim: int = 1000,
        seed: int = 42,
    ):
        """
        Strict no-leakage consensus mask for held-out sample i.
 
        Majority vote is computed using ONLY training years (excluding i).
        Each inner fold:
          bootcorr(quick=False) + gridCheck
        We count how often each grid cell is KEPT, and keep if count >= ceil(q * n_folds).
        """
        import numpy as np
        from numpy import ma
        from sklearn.model_selection import LeaveOneOut, KFold
 
        X_all = np.asarray(self.glo_var.data)  # (n, nlat, nlon)
        y_all = np.asarray(self.clim_data).ravel()  # (n,)
        ind_all = np.asarray(self.indicator).ravel()  # (n,)
 
        n, nlat, nlon = X_all.shape
 
        train_mask = np.ones(n, dtype=bool)
        train_mask [ heldout_i ] = False
 
        X_train = X_all [ train_mask ]
        y_train = y_all [ train_mask ]
        ind_train = ind_all [ train_mask ]
        ntr = y_train.shape [ 0 ]
 
        if ntr < 3:
            return None
 
        # Choose inner splitter: LOOCV if tiny, else 5-fold
        if ntr <= 10:
            splitter = LeaveOneOut()
            split_iter = splitter.split(np.arange(ntr))
        else:
            splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
            split_iter = splitter.split(np.arange(ntr))
 
        counts = np.zeros((nlat, nlon), dtype=int)
        n_folds = 0
 
        for tr_idx, _ in split_iter:
            X_tr = X_train [ tr_idx ]
            y_tr = y_train [ tr_idx ]
            ind_tr = ind_train [ tr_idx ]
 
            fold = NIPAphase(
                clim_data=y_tr,
                glo_var_name=self.glo_var_name,
                glo_var=seasonal_var(X_tr, self.glo_var.lat, self.glo_var.lon),
                indicator=ind_tr,
                phaseind=None,
                alt=True,
            )
            fold_seed = seed + n_folds
            fold.bootcorr(ntim=ntim, corrconf=corrconf, bootconf=bootconf, quick=False, seed=fold_seed)
            if fold.flags.get("no_glo_var", False):
                continue
 
            fold.gridCheck()
            keep = ~fold.corr_grid.mask
            counts += keep.astype(int)
            n_folds += 1
 
        if n_folds==0:
            return None
 
        thresh = int(np.ceil(q * n_folds))
        consensus_keep = counts >= thresh
 
        consensus_corr = ma.masked_array(
            np.zeros((nlat, nlon), dtype=float),
            mask=~consensus_keep
        )
        return consensus_corr
 
    def strict_loocv_majority_hindcast(
        self,
        q: float = 0.6,
        corrconf: float = 0.9,
        bootconf: float = 0.6,
        ntim: int = 1000,
        seed: int = 42,
        explained_variance_threshold: float = 0.75,
    ):
        """
        Strict no-leakage LOOCV hindcast with stability selection (majority vote).
 
        For each held-out year i:
          1) Build consensus corr mask using ONLY training years (excluding i),
             aggregating fold-wise bootcorr+gridCheck masks by majority vote.
          2) Fit PCR/Ridge on training years ONLY using that consensus mask.
          3) Predict held-out year i using fold.predict_from_pcr().
        """
        import numpy as np
 
        X_all = np.asarray(self.glo_var.data)  # (n, nlat, nlon)
        y_all = np.asarray(self.clim_data).ravel()  # (n,)
        ind_all = np.asarray(self.indicator).ravel()  # (n,)
 
        n = y_all.shape [ 0 ]
        yhat = np.full(n, np.nan)
 
        for i in range(n):
            # 1) training-only consensus mask
            consensus_corr = self._consensus_mask_for_heldout(
                heldout_i=i,
                q=q,
                corrconf=corrconf,
                bootconf=bootconf,
                ntim=ntim,
                seed=seed,
            )
            if consensus_corr is None:
                continue
 
            # 2) training subset
            tr = np.ones(n, dtype=bool)
            tr [ i ] = False
 
            X_tr = X_all [ tr ]
            y_tr = y_all [ tr ]
            ind_tr = ind_all [ tr ]
            X_val = X_all [ i ]  # (nlat, nlon)
 
            # 3) fit fold model (training-only)
            fold = NIPAphase(
                clim_data=y_tr,
                glo_var_name=self.glo_var_name,
                glo_var=seasonal_var(X_tr, self.glo_var.lat, self.glo_var.lon),
                indicator=ind_tr,
                phaseind=None,
                alt=True,
            )
            fold.phase = getattr(self, "phase", None)
 
            fold.corr_grid = consensus_corr
            fold.n_pre_grid = int((~fold.corr_grid.mask).sum())
            if fold.n_pre_grid==0:
                continue
 
            # Optional cleanup of isolated pixels (still training-only)
            fold.gridCheck()
 
            # Fit PCR/Ridge on training-only (no nested CV)
            fold.crossvalpcr(xval=False, explained_variance_threshold=explained_variance_threshold)
            if fold.lin_model is None:
                continue
 
            # 4) predict held-out
            try:
                pred = fold.predict_from_pcr(X_val)  # (1,)
                if np.isfinite(pred [ 0 ]):
                    yhat [ i ] = float(pred [ 0 ])
            except Exception:
                pass
 
        return yhat


    def loocv_hindcast(self, corrconf=0.95, bootconf=0.80, ntim=1000, seed=42,
                       explained_variance_threshold=0.75, n_pc_fixed=None,
                       regressor="ridge", quick=False, detrend=False, t=None,
                       lat_limit=60.0, orog_max_m=20000.0):
        """
        LOOCV: per ogni anno i, maschera (bootcorr + gridCheck), PCA e regressione
        stimate SOLO sugli altri n-1 anni, poi previsione di i.
        Se sul train non resta nessun pixel, la previsione e' la media del train.

        detrend=True (richiede t = anni della fase): in ogni fold si stima un trend lineare
        SOLO sul training, per il target e per ogni pixel; lo si toglie dal training e,
        estrapolato, dall'anno i. Il modello predice l'anomalia rispetto al trend:
            previsione_i = trend_y(t_i) + anomalia predetta
        self.loocv_baseline contiene la baseline di confronto per anno:
            detrend=False -> media del training (climatologia LOO)
            detrend=True  -> trend del training estrapolato a t_i
        """
        X_all = np.asarray(self.glo_var.data)
        y_all = np.asarray(self.clim_data).ravel()
        ind_all = np.asarray(self.indicator).ravel()
        n = y_all.size
        yhat = np.full(n, np.nan)
        n_fallback = 0
        keep_count = np.zeros(X_all.shape[1:], dtype=int)
        baseline = np.full(n, np.nan)
        # PC out-of-sample: per ogni anno i, la proiezione di i sulle EOF del suo fold
        # (quello che la regressione usa davvero). Max 8 PC come in crossvalpcr.
        K_MAX = 8
        pc_oos = np.full((n, K_MAX), np.nan)
        coef_oos = np.full((n, K_MAX), np.nan)
        intercept_oos = np.full(n, np.nan)
        npc_oos = np.zeros(n, dtype=int)
        npix_oos = np.zeros(n, dtype=int)
        if detrend:
            if t is None:
                raise ValueError("detrend=True richiede t (anni della fase)")
            t = np.asarray(t, dtype=float).ravel()
            if t.size != n:
                raise ValueError(f"t ha {t.size} valori, attesi {n}")

        for i in range(n):
            tr = np.ones(n, dtype=bool)
            tr[i] = False

            if detrend:
                fy = linear_trend_fit(t[tr], y_all[tr])
                fX = linear_trend_fit(t[tr], X_all[tr])
                y_tr = y_all[tr] - linear_trend_eval(fy, t[tr])
                X_tr = X_all[tr] - linear_trend_eval(fX, t[tr])
                X_i = X_all[i] - linear_trend_eval(fX, t[i])[0]
                baseline[i] = float(linear_trend_eval(fy, t[i])[0])
            else:
                y_tr, X_tr, X_i = y_all[tr], X_all[tr], X_all[i]
                baseline[i] = float(y_all[tr].mean())
            offset = baseline[i] if detrend else 0.0

            fold = NIPAphase(
                clim_data=y_tr,
                glo_var_name=self.glo_var_name,
                glo_var=seasonal_var(X_tr, self.glo_var.lat, self.glo_var.lon),
                indicator=ind_all[tr],
                phaseind=None,
                alt=True,
            )
            fold.phase = self.phase

            fold.bootcorr(ntim=ntim, corrconf=corrconf, bootconf=bootconf, quick=quick, seed=seed,
                          lat_limit=lat_limit, orog_max_m=orog_max_m)
            if not fold.flags.get("no_glo_var", False):
                fold.gridCheck()
                fold.crossvalpcr(xval=False, explained_variance_threshold=explained_variance_threshold,
                                 n_pc_fixed=n_pc_fixed, regressor=regressor)

            if fold.lin_model is None:
                yhat[i] = offset + float(np.mean(y_tr))   # nessun predittore -> baseline del train
                n_fallback += 1
                continue

            pcs_i = fold.project_pcs(X_i)[0]
            yhat[i] = offset + float(fold.lin_model["regression"].predict(pcs_i[None, :])[0])
            keep_count += ~np.ma.getmaskarray(fold.corr_grid)

            # Segno delle EOF arbitrario per fold: lo fisso con r(PC_train, y_train) >= 0.
            # Flippo PC e coefficiente insieme, quindi la previsione non cambia.
            k = min(int(fold.lin_model["n_pc"]), K_MAX)
            coef = np.asarray(fold.lin_model["regression"].coef_, dtype=float).ravel()
            for j in range(k):
                pc_tr = fold.pcs[:, j]
                s = 1.0
                if np.std(pc_tr) > 0 and np.corrcoef(pc_tr, y_tr)[0, 1] < 0:
                    s = -1.0
                pc_oos[i, j] = s * pcs_i[j]
                coef_oos[i, j] = s * coef[j]
            intercept_oos[i] = float(np.asarray(fold.lin_model["regression"].intercept_).ravel()[0])
            npc_oos[i] = k
            npix_oos[i] = int(np.asarray(fold.lin_model["mask_idx"]).sum())

        self.loocv_pc_oos = pc_oos
        self.loocv_coef_oos = coef_oos
        self.loocv_intercept_oos = intercept_oos
        self.loocv_npc_oos = npc_oos
        self.loocv_npix_oos = npix_oos
        self.loocv_keep_freq = keep_count / n
        self.loocv_n_fallback = n_fallback
        self.loocv_baseline = baseline
        self.loocv_detrend = bool(detrend)
        if n_fallback:
            print(f"[LOOCV] fase={self.phase}: {n_fallback}/{n} anni senza pixel -> climatologia")
        return yhat
    # ------------------------------------------------------------------
    # Modello operativo (forecast)
    # ------------------------------------------------------------------
    def fit_final(self, t, corrconf=0.95, bootconf=0.80, ntim=1000, seed=42,
                  explained_variance_threshold=0.75, n_pc_fixed=None,
                  regressor="ridge", quick=False, detrend=False,
                  lat_limit=60.0, orog_max_m=20000.0):
        """
        Modello operativo della fase: gli STESSI passi di un fold di loocv_hindcast
        (trend, screening, gridCheck, PCA, regressione), stimati su TUTTI gli anni
        della fase. Serve solo per il forecast di anni non usati nella stima; la skill
        riportata resta quella LOOCV.
        """
        X_all = np.asarray(self.glo_var.data, dtype=float)
        y_all = np.asarray(self.clim_data, dtype=float).ravel()
        t = np.asarray(t, dtype=float).ravel()
        if t.size != y_all.size:
            raise ValueError(f"t ha {t.size} valori, attesi {y_all.size}")

        if detrend:
            fy = linear_trend_fit(t, y_all)
            fX = linear_trend_fit(t, X_all)
            y_d = y_all - linear_trend_eval(fy, t)
            X_d = X_all - linear_trend_eval(fX, t)
        else:
            fy = fX = None
            y_d, X_d = y_all, X_all

        fold = NIPAphase(
            clim_data=y_d,
            glo_var_name=self.glo_var_name,
            glo_var=seasonal_var(X_d, self.glo_var.lat, self.glo_var.lon),
            indicator=np.asarray(self.indicator),
            phaseind=None,
            alt=True,
        )
        fold.phase = self.phase
        fold.bootcorr(ntim=ntim, corrconf=corrconf, bootconf=bootconf, quick=quick, seed=seed,
                      lat_limit=lat_limit, orog_max_m=orog_max_m)
        if not fold.flags.get("no_glo_var", False):
            fold.gridCheck()
            fold.crossvalpcr(xval=False, explained_variance_threshold=explained_variance_threshold,
                             n_pc_fixed=n_pc_fixed, regressor=regressor)

        # segno della EOF1 fissato come in LOOCV: r(PC1, y) >= 0 (solo per il reporting)
        sign = 1.0
        if fold.lin_model is not None and fold.pcs is not None:
            pc1 = fold.pcs[:, 0]
            if np.std(pc1) > 0 and np.corrcoef(pc1, y_d)[0, 1] < 0:
                sign = -1.0

        self.final_model = {
            "detrend": bool(detrend),
            "trend_y": fy,
            "trend_X": fX,
            "y_fallback": float(np.mean(y_d)),
            "fold": fold,
            "pc1_sign": sign,
            "years": t,
        }
        return self.final_model

    def predict_final(self, X_new, t_new):
        """Applica il modello operativo al campo predittore (anomalie, 2D) dell'anno t_new.
        Ritorna forecast, componente trend, componente teleconnessione, PC1 orientata."""
        fm = getattr(self, "final_model", None)
        if fm is None:
            raise ValueError("Modello operativo assente: chiamare fit_final prima.")
        X_new = np.asarray(X_new, dtype=float)
        t_new = float(t_new)

        if fm["detrend"]:
            trend = float(linear_trend_eval(fm["trend_y"], t_new)[0])
            X_new = X_new - linear_trend_eval(fm["trend_X"], t_new)[0]
        else:
            trend = 0.0

        fold = fm["fold"]
        if fold.lin_model is None:
            return {"forecast": trend + fm["y_fallback"], "trend_component": trend,
                    "teleconnection_component": fm["y_fallback"], "PC1": np.nan,
                    "n_pix": 0, "fallback": True}

        pcs = fold.project_pcs(X_new)[0]
        tele = float(fold.lin_model["regression"].predict(pcs[None, :])[0])
        return {"forecast": trend + tele, "trend_component": trend,
                "teleconnection_component": tele, "PC1": float(fm["pc1_sign"] * pcs[0]),
                "n_pix": int(np.asarray(fold.lin_model["mask_idx"]).sum()), "fallback": False}

    def save_final_model(self, out_dir, clim_2d):
        """
        Salva il modello operativo in un unico NPZ autosufficiente
        (final_model_<fase>.npz), riapplicabile offline con utils.apply_final_model.
        clim_2d: climatologia per pixel sottratta al campo grezzo prima del training.
        """
        import json
        from pathlib import Path
        fm = getattr(self, "final_model", None)
        if fm is None:
            raise ValueError("Modello operativo assente: chiamare fit_final prima.")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fold = fm["fold"]
        lm = fold.lin_model
        nlat, nlon = len(self.glo_var.lat), len(self.glo_var.lon)
        empty = np.array([], dtype=float)

        if fm["detrend"]:
            ty_m, ty_s, ty_t = fm["trend_y"]
            tX_m, tX_s, tX_t = fm["trend_X"]
        else:
            ty_m = ty_s = ty_t = tX_t = np.nan
            tX_m = tX_s = empty

        pack = dict(
            phase=np.array(str(self.phase)),
            lat=np.asarray(self.glo_var.lat, dtype=float),
            lon=np.asarray(self.glo_var.lon, dtype=float),
            clim_2d=np.asarray(clim_2d, dtype=float),
            years=np.asarray(fm["years"], dtype=float),
            detrend=np.array(fm["detrend"]),
            y_fallback=np.array(fm["y_fallback"]),
            trend_y=np.array([ty_m, ty_s, ty_t], dtype=float),
            trend_X_mean=np.asarray(tX_m, dtype=float),
            trend_X_slope=np.asarray(tX_s, dtype=float),
            trend_X_tbar=np.array(tX_t, dtype=float),
            has_model=np.array(lm is not None),
            pc1_sign=np.array(fm["pc1_sign"]),
        )
        summary = {"phase": str(self.phase), "years": [int(fm["years"][0]), int(fm["years"][-1])],
                   "n_years": int(fm["years"].size), "detrend": fm["detrend"],
                   "has_model": lm is not None}
        if lm is not None:
            reg = lm["regression"]
            col_means = lm.get("col_means")
            pack.update(
                eofs=np.asarray(lm["eofs"], dtype=float),
                x_mean=np.asarray(lm["x_mean"], dtype=float),
                mask_idx=np.asarray(lm["mask_idx"], dtype=bool),
                mask_2d=np.asarray(lm["mask_idx"], dtype=bool).reshape(nlat, nlon),
                col_means=np.asarray(col_means, dtype=float) if col_means is not None else empty,
                coef=np.asarray(reg.coef_, dtype=float).ravel(),
                intercept=np.array(float(np.asarray(reg.intercept_).ravel()[0])),
            )
            summary.update(n_pix=int(np.asarray(lm["mask_idx"]).sum()), n_pc=int(lm["n_pc"]),
                           coef_PC1_oriented=float(fm["pc1_sign"] * pack["coef"][0]),
                           intercept=float(pack["intercept"]))
        np.savez_compressed(out_dir / f"final_model_{self.phase}.npz", **pack)
        (out_dir / f"final_model_{self.phase}.json").write_text(json.dumps(summary, indent=2))
        return summary