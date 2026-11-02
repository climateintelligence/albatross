from __future__ import annotations
"""
Module containing utility functions for running NIPA.
"""
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import json
from typing import Sequence
import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from math import atanh, tanh, sqrt
try:
    from scipy.stats import t as t_dist
except Exception:
    t_dist = None

def _freeze_kwargs(kwargs: dict):
    """
    Convert a kwargs dict into a hashable tuple of sorted key-value pairs.
    Lists and tuples are converted to tuples recursively.
    """
    def freeze(v):
        if isinstance(v, dict):
            return _freeze_kwargs(v)
        elif isinstance(v, (list, tuple)):
            return tuple(freeze(i) for i in v)
        return v

    return tuple(sorted((k, freeze(v)) for k, v in kwargs.items()))

def plot_model_results(model, filepath=None, crv_flag=True, ax=None):
    """
    Scatterplot with:
      - legend bottom-right (inside axes)
      - metrics box top-left:
            r = ...
            (confidence = ...%)

    Confidence = 100 * (1 - p-value) from Pearson correlation t-test.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    y = np.asarray(model.clim_data, dtype=float)
    y_pred = np.asarray(model.hindcast, dtype=float)

    # paired finite values
    y_f, yhat_f = _paired_finite(y, y_pred)
    n = int(y_f.size)

    # correlation and confidence
    r = np.nan
    conf = np.nan
    if n >= 2 and np.std(y_f) > 0 and np.std(yhat_f) > 0:
        r = float(np.corrcoef(y_f, yhat_f)[0, 1])
        p = corr_pvalue_ttest(r, n)
        conf = corr_level_of_confidence(p)

    # regression equation (footer)
    reg = model.lin_model["regression"]
    coef_str = " + ".join(
        f"{coef:.2f}·PC{i+1}" for i, coef in enumerate(reg.coef_)
    )
    equation_str = f"y = {coef_str} + {reg.intercept_:.2f}"

    title_prefix = "Multivariate Regression" if crv_flag else "Hindcast (Full Fit)"
    title = f"{getattr(model, 'phase', 'Unknown')}\nPhase: {title_prefix}"

    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        show_plot = True
    else:
        fig = ax.figure

    # scatter + 1:1 line
    ax.scatter(y_f, yhat_f, alpha=0.7, label="Observed vs Predicted")

    mn = float(np.nanmin([np.nanmin(y_f), np.nanmin(yhat_f)]))
    mx = float(np.nanmax([np.nanmax(y_f), np.nanmax(yhat_f)]))
    ax.plot([mn, mx], [mn, mx], "r--", label="1:1 line")

    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(title)

    # legend bottom-right (inside plot)
    ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=9,
    )

    # metrics box (top-left)
    if np.isfinite(conf):
        box = f"r = {r:.2f} ({conf:.1f}%)"
    else:
        box = f"r = {r:.2f} (n/a)"

    ax.text(
        0.02, 0.98, box,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=10,
        bbox=dict(
            facecolor="white",
            edgecolor="gray",
            boxstyle="round,pad=0.4",
        ),
    )

    # footer equation
    if show_plot:
        fig.text(
            0.5, 0.01, equation_str,
            wrap=True,
            horizontalalignment="center",
            fontsize=10,
            bbox=dict(
                facecolor="white",
                edgecolor="gray",
                boxstyle="round,pad=0.4",
            ),
        )
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        if filepath:
            fig.savefig(filepath, bbox_inches="tight")
        plt.close(fig)

def weight_glo_var(glo_var):
    """Restituisce un NUOVO seasonal_var pesato per cos(lat); non modifica l'input."""
    w = np.cos(np.radians(np.asarray(glo_var.lat, dtype=float)))
    data = np.asarray(glo_var.data, dtype=float) * w[None, :, None]
    return type(glo_var)(data, glo_var.lat, glo_var.lon)

def sig_test(r, n, twotailed=True):
    import numpy as np
    from scipy.stats import t as tdist
    df = n - 2

    # Create t-statistic
    # Use absolute value to be able to deal with negative scores
    t = np.abs(r * np.sqrt(df / (1 - r**2)))
    p = (1 - tdist.cdf(t, df))
    if twotailed:
        p = p * 2
    return p

def vcorr(X, y):
    # Function to correlate a single time series with a gridded data field
    # X - Gridded data, 3 dimensions (ntim, nlat, nlon)
    # Y - Time series, 1 dimension (ntim)

    ntim, nlat, nlon = X.shape
    ngrid = nlat * nlon

    y = y.reshape(1, ntim)
    X = X.reshape(ntim, ngrid).T
    Xm = X.mean(axis=1).reshape(ngrid, 1)
    ym = y.mean()
    r_num = np.sum((X - Xm) * (y - ym), axis=1)
    r_den = np.sqrt(np.sum((X - Xm)**2, axis=1) * np.sum((y - ym)**2))
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.true_divide(r_num, r_den)
        r [ ~np.isfinite(r) ] = 0  # or np.nan, depending on your use case
        r = r.reshape(nlat, nlon)

    return r

def int_to_month():
    """
    This function is used by data_load.create_data_parameters
    """
    i2m = {
        -8: 'Apr',
        -7: 'May',
        -6: 'Jun',
        -5: 'Jul',
        -4: 'Aug',
        -3: 'Sep',
        -2: 'Oct',
        -1: 'Nov',
        0: 'Dec',
        1: 'Jan',
        2: 'Feb',
        3: 'Mar',
        4: 'Apr',
        5: 'May',
        6: 'Jun',
        7: 'Jul',
        8: 'Aug',
        9: 'Sept',
        10: 'Oct',
        11: 'Nov',
        12: 'Dec',
        13: 'Jan',
        14: 'Feb',
        15: 'Mar',
    }
    return i2m

def meteo_swiss_convert(f_in, f_out):
    data = np.loadtxt(f_in, skiprows=28)
    years = data[:, 0]
    # months = data[:, 1]
    # temp = data[:, 2]
    prcp = data[:, 3]
    startyr = years[0]
    endyr = years[-1]
    nyrs = endyr - startyr + 1
    x = prcp.reshape(nyrs, 12)
    y = np.arange(startyr, startyr + nyrs).reshape(nyrs, 1)
    array = np.concatenate((y, x), axis=1)

    fmtstr = '%i %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f'
    with open(f_out, 'w') as f:
        f.write('Description \n')
        f.write('Line 2 \n')
        np.savetxt(f, array, fmt=fmtstr)

    return

def glo_var_Map_unfiltered(nipaPhase, cmap=cm.jet):
    from mpl_toolkits.basemap import Basemap
    fig = plt.figure()
    ax = fig.add_subplot(111)
    m = Basemap(ax=ax, projection='cyl', lon_0=270, resolution='i')
    m.drawmapboundary(fill_color='#ffffff', linewidth=0.15)
    m.drawcoastlines(linewidth=0.15)
    m.fillcontinents(color='#eeeeee', lake_color='#ffffff')
    parallels = np.linspace(m.llcrnrlat, m.urcrnrlat, 4)
    meridians = np.linspace(m.llcrnrlon, m.urcrnrlon, 4)
    m.drawparallels(parallels, linewidth=0.3, labels=[0, 0, 0, 0])
    m.drawmeridians(meridians, linewidth=0.3, labels=[0, 0, 0, 0])

    lons = nipaPhase.glo_var.lon
    lats = nipaPhase.glo_var.lat
    data = nipaPhase.corr_grid_full
    data = np.where(data==0, np.nan, data)

    lons, lats = np.meshgrid(lons, lats)
    im1 = m.pcolormesh(
        lons, lats, data,
        vmin=np.nanmin(data),
        vmax=np.nanmax(data),
        cmap=cmap,
        latlon=True
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='horizontal')
    return fig

def glo_var_Map(nipaPhase, cmap="RdBu_r", fig=None, ax=None):
    from mpl_toolkits.basemap import Basemap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    import numpy as np

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    m = Basemap(ax=ax, projection='cyl', lon_0=270, resolution='i')
    m.drawmapboundary(fill_color='#ffffff', linewidth=0.15)
    m.drawcoastlines(linewidth=0.15)
    m.fillcontinents(color='#eeeeee', lake_color='#ffffff')

    # Griglia ogni 30° in lat e 60° in lon, con etichette a sinistra e in basso
    m.drawparallels(np.arange(-60, 61, 30), linewidth=0.3, color="0.5", dashes=[1, 2],
                    labels=[1, 0, 0, 0], fontsize=7)
    m.drawmeridians(np.arange(0, 360, 60), linewidth=0.3, color="0.5", dashes=[1, 2],
                    labels=[0, 0, 0, 1], fontsize=7)

    lons = nipaPhase.glo_var.lon
    lats = nipaPhase.glo_var.lat
    data = nipaPhase.corr_grid

    lons, lats = np.meshgrid(lons, lats)

    # Fixed color scale from -1 to +1
    im1 = m.pcolormesh(
        lons, lats, data,
        vmin=-1.0, vmax=1.0,
        cmap=cmap,
        latlon=True
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='5%', pad=0.3)   # spazio per le etichette di lon
    cbar = fig.colorbar(im1, cax=cax, orientation='horizontal')
    cbar.set_label('Correlation coefficient', fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    return fig, ax, m

def append_model_pcs(model, pcs_file_rows):
    """
    Extracts PCs and hindcasts from a NIPAphase model and appends rows to the given list.

    Parameters:
    -----------
    model : NIPAphase
        Fitted NIPAphase model containing .pcs, .hindcast, .years, and .phase.
    pcs_file_rows : list of dict
        List to append extracted rows to. Each row is a dict with year, phase, hindcast, and PCs.
    """
    if model.pcs is not None and model.hindcast is not None:
        n_pc = model.pcs.shape[1]
        for i in range(len(model.years)):
            row = {
                "year": int(model.years[i]),
                "phase": model.phase,
                "hindcast": float(model.hindcast[i]),
            }
            for pc_idx in range(n_pc):
                row[f"PC{pc_idx + 1}"] = float(model.pcs[i, pc_idx])
            pcs_file_rows.append(row)

from pathlib import Path

def extract_target_name(filepath_or_url):
    return Path(filepath_or_url).stem

def lag_to_month():
    d = {
        -4: '08',
            -3: '09',
            -2: '10',
            -1: '11',
        0: '12',
        1: '01',
        2: '02',
        3: '03',
        4: '04',
        5: '05',
        6: '06',
        7: '07',
        8: '08',
        9: '09',
        10: '10',
        11: '11',
        12: '12',
        13: '01',
        14: '02',
        15: '03',
    }
    return d

def _compute_phase_thresholds(index_avg: np.ndarray, phaseind: dict) -> dict:
    """
    Soglie derivate dai tagli EFFETTIVI della partizione per rango, non da np.quantile.

    La soglia fra la fase k e la k+1 e' il valore massimo dell'indice osservato
    nella fase k. Con pick_phase_from_thresholds (prima fase con x <= thr) questo
    riproduce esattamente la partizione di training per ogni anno, tranne quando
    lo stesso valore dell'indice cade a cavallo di un taglio (caso segnalato).
    """
    v = np.asarray(index_avg, dtype=float)
    labels = list(phaseind.keys())
    nph = len(labels)

    thresholds = []
    for lab in labels[:-1]:
        vals = v[np.asarray(phaseind[lab], dtype=bool)]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            raise ValueError(f"Fase '{lab}' vuota: impossibile derivare la soglia.")
        thresholds.append(float(np.max(vals)))

    for lab, nxt, thr in zip(labels[:-1], labels[1:], thresholds):
        nxt_vals = v[np.asarray(phaseind[nxt], dtype=bool)]
        if np.any(np.isclose(nxt_vals, thr)):
            import logging
            logging.getLogger("PYWPS").warning(
                f"Index value {thr} appears in both '{lab}' and '{nxt}': "
                f"the threshold cannot reproduce the training split for those years."
            )

    if any(b < a for a, b in zip(thresholds, thresholds[1:])):
        raise ValueError(f"Soglie non monotone: {thresholds}")

    qmap = {2: [0.5], 3: [1/3, 2/3], 4: [0.25, 0.5, 0.75], 5: [0.2, 0.4, 0.6, 0.8]}
    return {
        "n_phases": nph,
        "labels": labels,
        "quantiles": qmap.get(nph, []),   # informativo, non usato per i tagli
        "thresholds": thresholds,
    }

def _save_phase_artifacts(workdir: Path, years: np.ndarray, index_avg: np.ndarray,
                          phaseind: dict, thresholds: dict):
    """
    Save per-phase year masks and global thresholds.
    Files written under workdir/'pc_vs_hindcast':
      - phase_mask_<phase>.csv   (columns: year, in_phase)
      - phase_thresholds.json    (n_phases, quantiles, thresholds[])
      - index_training.csv       (columns: year, index_avg)
    """
    out_dir = workdir / "pc_vs_hindcast"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Years + index used in training
    pd.DataFrame({"year": years.astype(int), "index_avg": index_avg.astype(float)}).to_csv(
        out_dir / "index_training.csv", index=False
    )

    # Per-phase masks
    for phase, mask in phaseind.items():
        pd.DataFrame(
            {"year": years.astype(int), "in_phase": mask.astype(bool)}
        ).to_csv(out_dir / f"phase_mask_{phase}.csv", index=False)

    # Global thresholds
    with (out_dir / "phase_thresholds.json").open("w") as f:
        json.dump(thresholds, f, indent=2)

def pick_phase_from_thresholds(x: float, meta: dict) -> str:
    labels = meta["labels"]
    thresholds = meta.get("thresholds", [])
    if not thresholds:
        return labels[0]
    for thr, lab in zip(thresholds, labels):
        if x <= float(thr):
            return lab
    return labels[-1]

def latest_valid_index_value(index_arr) -> float:
    arr = np.asarray(index_arr).ravel().astype(float)
    arr = arr[np.isfinite(arr) & (arr != -999.0)]
    if arr.size == 0:
        raise ValueError("Index not available (all NaN/-999).")
    return float(arr[-1])

def predictors_available(glo_var_obj, index_arr) -> None:
    _ = latest_valid_index_value(index_arr)
    field = np.asarray(getattr(glo_var_obj, "data", glo_var_obj))
    if field.ndim == 3:
        field0 = field[0]
    elif field.ndim == 2:
        field0 = field
    else:
        raise ValueError(f"Unexpected glo_var field ndim={field.ndim}")
    if not np.isfinite(field0).any():
        raise ValueError("Global variable not available (all NaN).")

def apply_saved_pcr(workdir: Path, phase: str, field_2d: np.ndarray) -> float:
    import pandas as pd
    pc_dir = Path(workdir) / "pc_vs_hindcast"
    npz_path = pc_dir / f"pc_transform_{phase}.npz"
    coef_path = pc_dir / f"coefficients_{phase}.csv"

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing regressor package: {npz_path.name}")
    if not coef_path.exists():
        raise FileNotFoundError(f"Missing coefficients file: {coef_path.name}")

    pack = np.load(npz_path, allow_pickle=True)
    eofs = pack["eofs"]
    x_mean = pack["x_mean"]
    mask_idx = pack["mask_idx"].astype(bool)
    n_pc = int(np.asarray(pack["n_pc"]).ravel()[0])

    x_flat = np.asarray(field_2d, dtype=float).reshape(-1)
    if x_flat.size != mask_idx.size:
        raise ValueError(f"Grid mismatch: field={x_flat.size} vs mask={mask_idx.size}")

    x_sel = x_flat[mask_idx]

    # optional post-filter
    if "keep_var" in pack and np.asarray(pack["keep_var"]).size:
        kv = np.asarray(pack["keep_var"]).astype(bool)
        x_sel = x_sel[kv]

    if x_sel.shape[0] != x_mean.shape[0]:
        raise ValueError(f"Feature mismatch: x_sel={x_sel.shape[0]} vs x_mean={x_mean.shape[0]}")

    nan = np.isnan(x_sel)
    if nan.any():
        x_sel[nan] = x_mean[nan]

    pcs = (x_sel - x_mean) @ eofs[:, :n_pc]

    df = pd.read_csv(coef_path)
    idx_col = [c for c in df.columns if c.lower().startswith("unnamed")]
    if idx_col:
        df = df.set_index(idx_col[0])

    intercept = float(df.loc["Intercept", "Coefficient"])
    betas = np.array([float(df.loc[f"PC{i+1}", "Coefficient"]) for i in range(n_pc)], dtype=float)
    return intercept + float(pcs @ betas)

def resolve_target_input(inp, outdir: Path) -> Path:
    """
    Download target file if input is a URL, or accept uploaded file.
    Compatible with your original behavior.
    """
    import requests
    from urllib.parse import urlparse
    if inp and hasattr(inp, "data") and isinstance(inp.data, str) and inp.data.startswith("http"):
        url = inp.data
        filename = Path(urlparse(url).path).name or "target.txt"
        dst = outdir / filename
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        dst.write_bytes(r.content)
        return dst

    if hasattr(inp, "file") and inp.file:
        path = Path(inp.file)
        with path.open("r") as f:
            first = f.readline().strip()
            second = f.readline().strip()

        # If uploaded file contains only a URL, download it
        if first.startswith("http") and not second:
            url = first
            filename = Path(urlparse(url).path).name or "target.txt"
            dst = outdir / filename
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            dst.write_bytes(r.content)
            return dst

        return path

    raise ValueError("No valid target supplied. Provide a URL or upload a .txt file.")

# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------
def _paired_finite(y, yhat):
    """Return paired finite values (drop NaN/inf in either array)."""
    y = np.asarray(y, dtype=float).ravel()
    yhat = np.asarray(yhat, dtype=float).ravel()
    m = np.isfinite(y) & np.isfinite(yhat)
    return y[m], yhat[m]

def corr_pvalue_ttest(r: float, n: int, alternative: str = "two-sided") -> float:
    if t_dist is None or (not np.isfinite(r)) or n < 3 or abs(r) >= 1.0:
        return np.nan
    t = r * np.sqrt((n - 2) / (1.0 - r**2))
    if alternative == "greater":
        return float(t_dist.sf(t, df=n - 2))
    return float(2.0 * t_dist.sf(abs(t), df=n - 2))

def corr_level_of_confidence(p_value: float) -> float:
    """
    Convert p-value to an interpretable "level of confidence" in percent.

    Definition used here:
      LoC(%) = 100 * (1 - p_value)

    Examples:
      p=0.05  -> 95%
      p=0.01  -> 99%

    Returns np.nan if p_value is nan.
    """
    if not np.isfinite(p_value):
        return np.nan
    return float(100.0 * (1.0 - p_value))

# -----------------------------------------------------------------------------
# Metrics for CSV
# -----------------------------------------------------------------------------
def compute_metrics_row(y_obs, y_est) -> dict:
    """
    Returns ONLY what you want downstream:
      N, MAE, RMSE, R2, BIAS, r, r_level_of_confidence

    Notes:
    - r_level_of_confidence is derived from the two-sided p-value of Pearson r:
        LoC(%) = 100*(1 - p)
    - If scipy is missing, LoC will be NaN.
    """
    y, yhat = _paired_finite(y_obs, y_est)
    n = int(y.size)

    out = {
        "N": n,
        "MAE": np.nan,
        "RMSE": np.nan,
        "R2": np.nan,
        "BIAS": np.nan,
        "r": np.nan,
        "r_level_of_confidence": np.nan,
    }
    if n == 0:
        return out

    err = yhat - y
    out["MAE"] = float(np.mean(np.abs(err)))
    out["RMSE"] = float(np.sqrt(np.mean(err**2)))
    out["BIAS"] = float(np.mean(err))

    # r + "level of confidence"
    if n >= 2 and np.std(y) > 0 and np.std(yhat) > 0:
        r = float(np.corrcoef(y, yhat)[0, 1])
        out["r"] = r
        p = corr_pvalue_ttest(r, n, alternative="greater")
        out["r_pvalue"] = p

    # R2 = 1 - SSE/SST
    ybar = float(np.mean(y))
    sse = float(np.sum((yhat - y) ** 2))
    sst = float(np.sum((y - ybar) ** 2))
    if sst > 0:
        out["R2"] = float(1.0 - sse / sst)

    return out

def compute_skill_mae_rmse(y_obs, y_est, eps: float = 1e-12) -> dict:
    """
    LOOCV climatology baseline skill (returns FRACTIONS, not percent):
      SS = 1 - (error_model / error_climatology_LOOCV)

    Output:
      SSMAE, SSRMSE  (fractions)
    """
    y, yhat = _paired_finite(y_obs, y_est)
    n = int(y.size)
    out = {"SSMAE": np.nan, "SSRMSE": np.nan}
    if n < 2:
        return out

    total = float(np.sum(y))
    clim = (total - y) / (n - 1)  # LOOCV mean excluding i

    mae_model = float(np.mean(np.abs(yhat - y)))
    rmse_model = float(np.sqrt(np.mean((yhat - y) ** 2)))

    mae_clim = float(np.mean(np.abs(clim - y)))
    rmse_clim = float(np.sqrt(np.mean((clim - y) ** 2)))

    out["SSMAE"] = float(1.0 - mae_model / (mae_clim + eps))
    out["SSRMSE"] = float(1.0 - rmse_model / (rmse_clim + eps))
    return out

def _grid_shape(n: int, max_cols: int = 2) -> tuple[int, int]:
    """Return (nrows, ncols) for n panels."""
    if n <= 1:
        return 1, 1
    ncols = min(max_cols, n)
    nrows = int(math.ceil(n / ncols))
    return nrows, ncols


# Nomi delle fasi come nel paper (phase modes 2-4)
_PHASE_NAMES = {"neg": "negative", "neutneg": "neutral-negative", "neutral": "neutral",
                "neutpos": "neutral-positive", "pos": "positive"}


def phase_title(phase: str, indicator: str | None, n_phases: int = None) -> str:
    """Titolo di un pannello: 'All years' oppure 'Negative phase', ecc. (senza nome dell'indice,
    che va nella didascalia)."""
    if phase == "allyears":
        return "All years"
    name = f"{_PHASE_NAMES.get(phase, phase)} phase"
    return name[0].upper() + name[1:]


def save_combined_phase_maps_pdf(
    phase_models: Sequence[object],
    out_pdf: Path,
    indicator: str,
    *,
    max_cols: int = 2,
    figsize_per_col: float = 6.0,
    figsize_per_row: float = 3.8,
) -> Path:
    """
    Save one multi-panel correlation map PDF (one panel per phase model).
    Requires glo_var_Map(model, fig=..., ax=...) to exist in utils.
    """
    from albatross.utils import glo_var_Map  # avoids circular import if needed

    nph = len(phase_models)
    nrows, ncols = _grid_shape(nph, max_cols=max_cols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_col * ncols, figsize_per_row * nrows),
    )
    axes = np.atleast_1d(axes).ravel()

    for i, model in enumerate(phase_models):
        ax = axes[i]
        glo_var_Map(model, fig=fig, ax=ax)
        ax.set_title(phase_title(getattr(model, 'phase', f'phase{i+1}'), indicator, nph))

    for j in range(nph, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_pdf


def save_combined_phase_scatters_pdf(
    phase_models: Sequence[object],
    out_pdf: Path,
    *,
    crv_flag: bool = True,
    max_cols: int = 2,
    figsize_per_col: float = 6.0,
    figsize_per_row: float = 5.2,
) -> Path:
    """
    Save one multi-panel scatter PDF (one panel per phase model).
    Requires plot_model_results(model, filepath=None, crv_flag=..., ax=...) in utils.
    """
    nph = len(phase_models)
    nrows, ncols = _grid_shape(nph, max_cols=max_cols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_col * ncols, figsize_per_row * nrows),
    )
    axes = np.atleast_1d(axes).ravel()

    for i, model in enumerate(phase_models):
        ax = axes[i]
        has_fit = getattr(model, "lin_model", None) is not None
        has_pred = getattr(model, "hindcast", None) is not None
        if has_fit and has_pred:
            plot_model_results(model, filepath=None, crv_flag=crv_flag, ax=ax)
            ax.set_title(f"{getattr(model, 'phase', f'phase{i+1}')}")
        else:
            ax.set_visible(False)

    for j in range(nph, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_pdf


# -----------------------------------------------------------------------------
# Funzioni richieste da wps_drought.py (diagnostica LOOCV, forecast, mappe r)
# -----------------------------------------------------------------------------
def forecast_index_value(index_arr) -> float:
    """Valore dell'indice per l'anno di forecast: ultimo valore valido della serie."""
    return latest_valid_index_value(index_arr)


def compute_loocv_diagnostics(y_obs, y_est, eps: float = 1e-12) -> dict:
    """
    Diagnostiche rispetto alla climatologia LOOCV (media senza l'anno i):
      r_pvalue        p-value bilaterale della correlazione osservato/previsto
      frac_beats_clim frazione di anni in cui |errore modello| < |errore climatologia|
      sign_acc        frazione di anni con segno dell'anomalia prevista = osservata
      err_iqr         scarto interquartile dell'errore (previsto - osservato)
    """
    y, yhat = _paired_finite(y_obs, y_est)
    n = int(y.size)
    out = {"r_pvalue": np.nan, "frac_beats_clim": np.nan, "sign_acc": np.nan, "err_iqr": np.nan}
    if n < 2:
        return out

    clim = (float(np.sum(y)) - y) / (n - 1)
    out["frac_beats_clim"] = float(np.mean(np.abs(yhat - y) < np.abs(clim - y) - eps))

    obs_anom, pred_anom = y - clim, yhat - clim
    valid = np.abs(obs_anom) > eps
    if valid.any():
        out["sign_acc"] = float(np.mean(np.sign(obs_anom[valid]) == np.sign(pred_anom[valid])))

    q75, q25 = np.percentile(yhat - y, [75, 25])
    out["err_iqr"] = float(q75 - q25)

    if n >= 3 and np.std(y) > 0 and np.std(yhat) > 0:
        out["r_pvalue"] = corr_pvalue_ttest(float(np.corrcoef(y, yhat)[0, 1]), n)
    return out


def compute_skill_vs_baseline(y_obs, y_est, y_base, eps: float = 1e-12) -> dict:
    """
    Skill rispetto a una baseline esterna (in wps_drought: la climatologia LOOCV
    calcolata DENTRO ogni fase). Chiavi distinte da compute_skill_mae_rmse.
    """
    y = np.asarray(y_obs, dtype=float).ravel()
    yhat = np.asarray(y_est, dtype=float).ravel()
    b = np.asarray(y_base, dtype=float).ravel()
    out = {"SSMAE_phaseclim": np.nan, "SSRMSE_phaseclim": np.nan, "frac_beats_phaseclim": np.nan}
    if not (y.size == yhat.size == b.size):
        raise ValueError(f"Lunghezze diverse: y={y.size}, yhat={yhat.size}, baseline={b.size}")
    m = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(b)
    if m.sum() < 2:
        return out
    y, yhat, b = y[m], yhat[m], b[m]

    mae_m, mae_b = float(np.mean(np.abs(yhat - y))), float(np.mean(np.abs(b - y)))
    rmse_m, rmse_b = float(np.sqrt(np.mean((yhat - y) ** 2))), float(np.sqrt(np.mean((b - y) ** 2)))
    out["SSMAE_phaseclim"] = float(1.0 - mae_m / (mae_b + eps))
    out["SSRMSE_phaseclim"] = float(1.0 - rmse_m / (rmse_b + eps))
    out["frac_beats_phaseclim"] = float(np.mean(np.abs(yhat - y) < np.abs(b - y) - eps))
    return out


def save_combined_phase_corr_values_pdf(
    phase_models: Sequence[object],
    out_pdf: Path,
    indicator: str,
    *,
    max_cols: int = 2,
    figsize_per_col: float = 6.0,
    figsize_per_row: float = 3.8,
) -> Path:
    """Come save_combined_phase_maps_pdf, ma colora i pixel con r (corr_grid_full), scala -1..+1."""
    from mpl_toolkits.basemap import Basemap

    nph = len(phase_models)
    nrows, ncols = _grid_shape(nph, max_cols=max_cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_col * ncols, figsize_per_row * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, model in enumerate(phase_models):
        ax = axes[i]
        data = getattr(model, "corr_grid_full", None)
        if data is None:
            ax.set_visible(False)
            continue
        m = Basemap(ax=ax, projection="cyl", lon_0=270, resolution="i")
        m.drawcoastlines(linewidth=0.15)
        m.fillcontinents(color="#eeeeee", lake_color="#ffffff")
        lons, lats = np.meshgrid(model.glo_var.lon, model.glo_var.lat)
        im = m.pcolormesh(lons, lats, data, vmin=-1.0, vmax=1.0, cmap="RdBu_r", latlon=True)
        cax = make_axes_locatable(ax).append_axes("bottom", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="horizontal").set_label("r")
        ax.set_title(f"{indicator} phase: {getattr(model, 'phase', f'phase{i+1}')}")

    for j in range(nph, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_pdf


# -----------------------------------------------------------------------------
# Maschere fisiche sui predittori
# -----------------------------------------------------------------------------
from functools import lru_cache as _lru_cache


@_lru_cache(maxsize=4)
def _load_orography(path: str):
    """Orografia NCEP/NCAR R1 (hgt.sfc.nc, PSL), in metri, con lat crescente."""
    import netCDF4
    ds = netCDF4.Dataset(path)
    h = np.squeeze(np.ma.filled(ds["hgt"][:].astype(float), np.nan))
    lat = np.asarray(ds["lat"][:], dtype=float)
    lon = np.asarray(ds["lon"][:], dtype=float)
    ds.close()
    if lat[0] > lat[-1]:
        h, lat = h[::-1, :], lat[::-1]
    return h, lat, lon


def high_orography_mask(lat, lon, max_elev_m: float, path: str | None = None) -> np.ndarray:
    """True dove l'orografia NCEP R1 supera max_elev_m (per mascherare la SLP)."""
    import os
    from pathlib import Path
    if path is None:
        from albatross.atmos_ocean_data import ensure_local
        try:
            path = ensure_local("orog")
        except Exception:
            path = os.environ.get("ALBATROSS_OROG_NC", os.path.expanduser("~/data/hgt.sfc.nc"))
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Orografia non trovata: {path}. Scarica hgt.sfc.nc di NCEP/NCAR R1 da NOAA PSL "
            f"oppure passa orog_max_m=None per disattivare la maschera."
        )
    h, hlat, hlon = _load_orography(path)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if h.shape != (lat.size, lon.size) or not (np.allclose(hlat, lat) and np.allclose(hlon, lon)):
        raise ValueError(
            f"Griglia orografia {h.shape} diversa da quella del predittore {(lat.size, lon.size)}."
        )
    return np.nan_to_num(h, nan=0.0) > max_elev_m


# -----------------------------------------------------------------------------
# Applicazione offline del modello operativo (indipendente dal WPS)
# -----------------------------------------------------------------------------
def apply_final_model(npz_path, field_2d, year) -> dict:
    """
    Riapplica offline un modello salvato da NIPAphase.save_final_model.

    field_2d : campo predittore GREZZO (nlat, nlon) dell'anno `year`, mediato sugli
               stessi mesi del training e sulla stessa griglia (lat crescente).
    year     : anno della stagione target (stessa convenzione degli anni di training).

    Passi identici al servizio: anomalia (clim_2d), detrend estrapolato all'anno,
    peso cos(lat), maschera, imputazione, centratura, proiezione su EOF, regressione,
    + trend del target.
    """
    p = np.load(npz_path, allow_pickle=False)
    lat = p["lat"]
    field = np.asarray(field_2d, dtype=float)
    if field.shape != (lat.size, p["lon"].size):
        raise ValueError(f"Griglia {field.shape} diversa da quella del modello {(lat.size, p['lon'].size)}")
    field = field - p["clim_2d"]
    year = float(year)

    if bool(p["detrend"]):
        m, s, tb = p["trend_y"]
        trend = float(m + s * (year - tb))
        field = field - (p["trend_X_mean"] + p["trend_X_slope"] * (year - float(p["trend_X_tbar"])))
    else:
        trend = 0.0

    if not bool(p["has_model"]):
        yf = float(p["y_fallback"])
        return {"forecast": trend + yf, "trend_component": trend,
                "teleconnection_component": yf, "PC1": np.nan, "fallback": True}

    field = field * np.cos(np.radians(lat))[:, None]
    x = field.reshape(-1)[p["mask_idx"]]
    nan = np.isnan(x)
    if nan.any():
        fill = p["col_means"] if p["col_means"].size == x.size else p["x_mean"]
        x[nan] = fill[nan]
    pcs = (x - p["x_mean"]) @ p["eofs"]
    tele = float(pcs @ p["coef"] + float(p["intercept"]))
    return {"forecast": trend + tele, "trend_component": trend,
            "teleconnection_component": tele, "PC1": float(p["pc1_sign"] * pcs[0]),
            "fallback": False}