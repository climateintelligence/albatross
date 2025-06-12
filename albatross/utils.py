
"""
Module containing utility functions for running NIPA.
"""
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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

def weightsst(sst):
    # SST needs to be downloaded using the openDAPsst function
    from numpy import cos, radians
    weights = cos(radians(sst.lat))
    for i, weight in enumerate(weights):
        sst.data[:, i, :] *= weight
    return sst

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

def slp_tf():
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

def sstMap(nipaPhase, cmap=cm.jet, fig=None, ax=None):
    from mpl_toolkits.basemap import Basemap
    if fig is None:
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

    lons = nipaPhase.sst.lon
    lats = nipaPhase.sst.lat

    data = nipaPhase.corr_grid
    levels = np.linspace(-1.0, 1.0, 41)

    lons, lats = np.meshgrid(lons, lats)

    im1 = m.pcolormesh(lons, lats, data, vmin=np.min(levels),
                       vmax=np.max(levels), cmap=cmap, latlon=True)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='horizontal')
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
