
"""
Module containing utility functions for running NIPA.
"""
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_pc1_vs_true(model, filepath):
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score

    x = getattr(model, "pc1", None)
    y = model.clim_data

    if x is None or len(x) != len(y):
        raise ValueError("PC1 scores are missing or do not align with target data.")
    slope = model.lin_model.get("slope", None)
    intercept = model.lin_model.get("intercept", None)
    y_pred = slope * x + intercept
    r2 = r2_score(y, y_pred)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, label="Observed", alpha=0.7)
    plt.plot(x, y_pred, color="red", label=f"Fit: y = {slope:.2f}x + {intercept:.2f}")
    plt.xlabel("PC1 (SST)")
    plt.ylabel("Predictand")
    plt.title(f"{model.phase} phase\nPC1 vs Predictand")

    # Add R²
    plt.text(
        0.95, 0.05,
        f"$R^2$: {r2:.2f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.5")
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


"""def plot_pc1_vs_true(model, filepath):

    from sklearn.metrics import r2_score
    x = model.pc1        # Principal component
    y = model.clim_data   # Hindcast values

    # Fit linear regression
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    r2 = r2_score(y, y_pred)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, label="Data points", alpha=0.7)
    plt.plot(x, y_pred, color="red", label=f"y = {slope:.2f}x + {intercept:.2f}")
    plt.xlabel("PC1 (SST)")
    plt.ylabel("Hindcast")
    plt.title(f"{model.phase} phase\nPC1 vs Hindcast")

    # Add R² to corner
    metrics_text = f"$R^2$: {r2:.2f}"
    plt.text(
        0.95, 0.05, metrics_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5')
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()"""

def make_scatterplot(model, fp):

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    x = model.clim_data      # true values (observed)
    y = model.hindcast       # predicted values (hindcast)

    # Calculate metrics
    r2 = r2_score(x, y)
    mae = mean_absolute_error(x, y)
    mse = mean_squared_error(x, y)
    rmse = np.sqrt(mse)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.7, label="Data points")
    # add line of best fit
    plt.plot([x.min(), x.max()], [x.min(), x.max()], 'r--', lw=2, label='Perfect fit')
    plt.xlabel("Observed")
    plt.ylabel("Hindcast")
    plt.title(f"{model.phase} phase")
    plt.grid(True)

    # Add metrics text box
    metrics_text = f"$R^2$: {r2:.2f}\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}"
    plt.text(
        0.8, 0.15,  # ⬅️ bottom-right corner in axes coordinates
        metrics_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5')
    )
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(fp)
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
