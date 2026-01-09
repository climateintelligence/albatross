
"""
Module containing utility functions for running NIPA.
"""
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import json

from functools import lru_cache

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
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score

    y = model.clim_data
    X = model.pcs
    reg = model.lin_model["regression"]
    y_pred = reg.predict(X)

    coef_str = " + ".join(f"{coef:.2f}·PC{i+1}" for i, coef in enumerate(reg.coef_))
    equation_str = f"y = {coef_str} + {reg.intercept_:.2f}"

    title_prefix = "Multivariate Regression" if crv_flag else "Hindcast (Full Fit)"
    title = f"{getattr(model, 'phase', 'Unknown')} Phase\n{title_prefix}"
    footer = f"{equation_str}\n$R^2$: {r2_score(y, y_pred):.2f}"
    label = "Regression Line"

    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        show_plot = True

    ax.scatter(y, y_pred, alpha=0.7, label="Observed vs Predicted")
    ax.plot([min(y), max(y)], [min(y), max(y)], "r--", label=label)
    ax.set_xlabel("Observed Precipitation")
    ax.set_ylabel("Predicted Precipitation")
    ax.set_title(title)
    ax.legend()

    if show_plot:
        fig.text(
            0.5, 0.01, footer,
            wrap=True,
            horizontalalignment="center",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.5")
        )
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        if filepath:
            fig.savefig(filepath, bbox_inches="tight")
        plt.close()

def weight_glo_var(glo_var):
    # SST needs to be downloaded using the openDAP function
    from numpy import cos, radians
    weights = cos(radians(glo_var.lat))
    for i, weight in enumerate(weights):
        glo_var.data[:, i, :] *= weight
    return glo_var

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
    #r = (r_num / r_den).reshape(nlat, nlon)
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


def glo_var_Map(nipaPhase, cmap=cm.jet, fig=None, ax=None):
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

    parallels = np.linspace(m.llcrnrlat, m.urcrnrlat, 4)
    meridians = np.linspace(m.llcrnrlon, m.urcrnrlon, 4)
    m.drawparallels(parallels, linewidth=0.3, labels=[ 0, 0, 0, 0 ])
    m.drawmeridians(meridians, linewidth=0.3, labels=[ 0, 0, 0, 0 ])

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
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    cbar = fig.colorbar(im1, cax=cax, orientation='horizontal')
    cbar.set_label('Correlation coefficient')

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
    Returns thresholds + explicit labels ordering for operational classification.
    """
    qmap = {2: [0.5], 3: [1/3, 2/3], 4: [0.25, 0.5, 0.75], 5: [0.2, 0.4, 0.6, 0.8]}
    labels = list(phaseind.keys())
    nph = len(labels)
    qs = qmap.get(nph, [])
    thresholds = list(np.quantile(index_avg, qs)) if qs else []
    return {
        "n_phases": nph,
        "labels": labels,
        "quantiles": qs,
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

import json
import numpy as np
import pandas as pd
from pathlib import Path

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
