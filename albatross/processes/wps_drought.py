# albatross/processes/wps_drought.py
#
# Skill: LOOCV senza leakage. Per ogni anno i, maschera (bootcorr + gridCheck), PCA e
#   regressione sono stimate SOLO sugli altri n-1 anni (NIPAphase.loocv_hindcast).
# Forecast: per ogni fase, modello operativo con gli stessi passi stimati su tutti gli
#   anni della fase (NIPAphase.fit_final), salvato nel bundle e riapplicabile offline
#   (utils.apply_final_model). Con forecast_year lo si applica al predittore osservato
#   nei mesi che precedono la stagione target di quell'anno.
#
# Con phase_mode=1 l'indice climatico non viene caricato ne' usato.

# Standard library imports
import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
import random

# 3rd party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from importlib.resources import files

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("PYWPS")

# import modules from pywps
from pywps import Process, LiteralInput, ComplexInput, ComplexOutput, FORMATS, Format
from pywps.app.Common import Metadata

# import modules from albatross
from albatross.climdiv_data import get_data, create_kwgroups
from albatross.new_simpleNIPA import NIPAphase, linear_detrend
from albatross.utils import (
    extract_target_name,
    _compute_phase_thresholds,
    resolve_target_input,
    _paired_finite,
    compute_metrics_row,
    compute_skill_mae_rmse,
    save_combined_phase_maps_pdf,
    corr_pvalue_ttest,
    _grid_shape,
    vcorr,
    pick_phase_from_thresholds,
    phase_title,
)
from albatross.atmos_ocean_data import load_clim_file

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
FORMAT_PDF = Format("image/pdf", extension=".pdf", encoding="base64")
FORMAT_CSV = Format("text/csv", extension=".csv")
np.random.seed(42)
random.seed(42)
lag = 3                 # mesi di lag e lunghezza della stagione dei predittori

# Impostazioni LOOCV (stesse delle run Como/Giava discusse)
LOOCV_CFG = dict(
    corrconf=0.95,      # significativita' della correlazione pixel-target
    bootconf=0.80,      # usato solo se quick=False
    ntim=1000,          # bootstrap, usato solo se quick=False
    quick=True,         # True = niente bootstrap, solo t-test
    n_pc_fixed=1,       # None = scelta per varianza spiegata (0.75)
    regressor="ols",    # "ols" oppure "ridge"
    detrend=True,      # True = trend lineare tolto in ogni fold (stimato sul solo training)
)
MAP_KEEP_FREQ = 0.5     # pixel mostrati in mappa: tenuti in almeno questa frazione di fold


# -----------------------------------------------------------------------------
# Helpers locali
# -----------------------------------------------------------------------------
def _skill_vs_baseline(y_obs, y_est, y_clim, eps=1e-12):
    """SS rispetto a una baseline fornita (per la riga 'overall': climatologia
    LOO condizionata per fase, non una climatologia unica sugli anni aggregati)."""
    y = np.asarray(y_obs, dtype=float).ravel()
    yhat = np.asarray(y_est, dtype=float).ravel()
    clim = np.asarray(y_clim, dtype=float).ravel()
    m = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(clim)
    y, yhat, clim = y[m], yhat[m], clim[m]
    out = {"SSMAE": np.nan, "SSRMSE": np.nan}
    if y.size < 2:
        return out
    out["SSMAE"] = 1.0 - np.mean(np.abs(yhat - y)) / (np.mean(np.abs(clim - y)) + eps)
    out["SSRMSE"] = 1.0 - np.sqrt(np.mean((yhat - y) ** 2)) / (np.sqrt(np.mean((clim - y) ** 2)) + eps)
    return out


def _loo_climatology(y):
    """Climatologia LOO sull'intero campione (tutti gli anni, ignorando le fasi): media
    degli altri n-1 anni. E' lo stesso riferimento della riga 'allyears' con M=1, quindi
    la riga 'overall' con M>1 e' direttamente confrontabile con M=1."""
    y = np.asarray(y, dtype=float)
    return (y.sum() - y) / (y.size - 1)


def _save_loocv_scatters_pdf(phase_models, out_pdf):
    """Scatter osservato vs hindcast LOOCV. Non richiede lin_model (che in LOOCV non esiste)."""
    nph = len(phase_models)
    nrows, ncols = _grid_shape(nph, max_cols=3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.8 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, model in zip(axes, phase_models):
        if model.hindcast is None:
            ax.set_visible(False)
            continue
        y, yhat = _paired_finite(model.clim_data, model.hindcast)
        if y.size == 0:
            ax.set_visible(False)
            continue
        ax.scatter(y, yhat, alpha=0.7, label="Observed vs LOOCV hindcast")
        mn, mx = float(min(y.min(), yhat.min())), float(max(y.max(), yhat.max()))
        ax.plot([mn, mx], [mn, mx], "r--", label="1:1 line")
        ax.axhline(np.mean(y), color="gray", lw=0.8, ls=":", label="Observed mean")

        r = float(np.corrcoef(y, yhat)[0, 1]) if (y.size >= 3 and np.std(yhat) > 0) else np.nan
        p = corr_pvalue_ttest(r, y.size, alternative="greater")   # one-sided, as in metrics
        if np.isfinite(p):
            ptxt = " (p < 0.001)" if p < 0.001 else f" (p = {p:.3f})"
        else:
            ptxt = ""
        txt = f"r = {r:.2f}" + ptxt
        txt += f"\nn = {y.size}, fallback = {getattr(model, 'loocv_n_fallback', 0)}"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.4"))
        ax.set_xlabel("Observed", fontsize=9)
        ax.set_ylabel("Predicted (LOOCV)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_title(phase_title(str(model.phase), None))
        ax.legend(loc="lower right", fontsize=7)

    for ax in axes[nph:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)

def _opt(request, name):
    """Valore di un input opzionale come stringa, None se assente o vuoto."""
    inp = request.inputs.get(name)
    if not inp or inp[0].data is None:
        return None
    v = str(inp[0].data).strip()
    return v or None


def _ym(y, m):
    """Mese assoluto. m puo' essere <= 0 (anno precedente) o > 12 (anno successivo)."""
    return y * 12 + m - 1


def _series_coverage(fp):
    """Primo e ultimo mese con dato valido di un file txt (target o indice)."""
    s = load_clim_file(fp)
    ok = np.isfinite(np.asarray(s.values, dtype=float))
    if not ok.any():
        raise ValueError(f"{fp}: no valid values")
    d = s.index[ok]
    return _ym(d[0].year, d[0].month), _ym(d[-1].year, d[-1].month)


def _predictor_coverage(glo_var_name):
    """Primo e ultimo mese del campo predittore (file PSL locale); None se non verificabile."""
    import netCDF4
    from albatross.atmos_ocean_data import ensure_local
    from albatross.hgt_levels import level_of
    key = {"sst": "sst", "slp": "slp", "z500": "hgt"}.get(glo_var_name)
    if key is None and level_of(glo_var_name) is not None:
        key = "hgt"
    if key is None:
        return None                    # ERA5: copertura non verificata qui
    with netCDF4.Dataset(ensure_local(key)) as ds:
        t = ds["time"]
        cal = getattr(t, "calendar", "standard")
        d0 = netCDF4.num2date(t[0], t.units, cal)
        d1 = netCDF4.num2date(t[-1], t.units, cal)
    return _ym(d0.year, d0.month), _ym(d1.year, d1.month)


def _default_years(target_cov, months, others):
    """Primo e ultimo anno con stagione target completa nel target e con i `lag` mesi
    precedenti coperti da tutti gli altri campi (predittore, indice)."""
    t0, t1 = target_cov
    sy, ey = t0 // 12, t1 // 12
    m0, m1 = months[0], months[-1]
    while _ym(sy, m0) < t0:
        sy += 1
    while _ym(ey, m1) > t1:
        ey -= 1
    for first, last in others:
        while _ym(sy, m0 - lag) < first:
            sy += 1
        while _ym(ey, m0 - 1) > last:
            ey -= 1
    if ey < sy:
        raise ValueError("The target series, predictor field and climate index have no "
                         "common period for this target season.")
    return sy, ey


# -----------------------------------------------------------------------------
# Process
# -----------------------------------------------------------------------------
class Drought(Process):
    def __init__(self):
        inputs = [
            ComplexInput(
                identifier="target",
                title="Target time series",
                abstract="Plain-text file: description, first and last year, then monthly values "
                        "(one year per row, Jan-Dec). Missing values as NaN or <= -99.",
                supported_formats=[FORMATS.TEXT],
            ),
            LiteralInput(
                identifier="aggregation",
                title="Target aggregation over season",
                abstract="How to aggregate target months: mean (average) or sum (cumulated).",
                data_type="string",
                allowed_values=["mean", "sum"],
                default="mean",
            ),
            LiteralInput(identifier="start_year", title="Start year",
                        abstract="Default: first year covered by target, predictor and (if used) index.",
                        data_type="string", min_occurs=0),
            LiteralInput(identifier="end_year", title="End year",
                        abstract="Default: last year covered by target, predictor and (if used) index.",
                        data_type="string", min_occurs=0),
            LiteralInput(identifier="month", title="Target season",
                abstract="Consecutive months, e.g. 4,5,6 or 11,12,13,14,15 (up to March of the following year).",
                data_type="string"),
            LiteralInput(identifier="indicator", title="Climate index",
                abstract="Required if phase_mode > 1.", data_type="string",
                allowed_values=["SCA", "EAWR", "AO", "ONI", "DMI", "SOI", "WP", "PNA", "NAO", "QBO",
                                "NINO 3.4", "NINO 12", "NINO 3.4_HadiSST"],
                min_occurs=0),
            LiteralInput(identifier="phase_mode", title="Number of phases",
                abstract="1 = all years, no climate index (default); 2-4 require 'indicator'.",
                data_type="string", allowed_values=["1", "2", "3", "4"], default="1"),
            LiteralInput(identifier="glo_var_name", title="Predictor field",
                abstract="Default: sst.", data_type="string",
                allowed_values=["sst", "slp", "z500"], default="sst"),
            LiteralInput(
                identifier="forecast_year",
                title="Forecast year (optional)",
                abstract="Year of the target season to forecast. The model calibrated on all training "
                         "years is applied to the predictor field and index observed in the months "
                         "preceding the target season of that year.",
                data_type="string",
                min_occurs=0,
            ),
        ]

        outputs = [
            ComplexOutput(identifier="forecast_file", title="LOOCV hindcast time series", as_reference=True, supported_formats=[FORMATS.TEXT]),
            ComplexOutput(identifier="scatter_plot", title="Scatter Plot", as_reference=True, supported_formats=[FORMAT_PDF]),
            ComplexOutput(identifier="forecast_bundle", title="All Outputs (ZIP)", as_reference=True, supported_formats=[FORMATS.ZIP]),
            ComplexOutput(identifier="glo_var_map", title="Correlation of pixels retained in >= half of LOOCV folds", as_reference=True, supported_formats=[FORMAT_PDF]),
            ComplexOutput(identifier="metrics_table", title="Metrics table by phase (CSV)", as_reference=True, supported_formats=[FORMAT_CSV]),
            ComplexOutput(identifier="operational_forecast", title="Operational forecast with phase skill (CSV)", as_reference=True, supported_formats=[FORMAT_CSV]),
        ]

        super().__init__(
            self._handler,
            identifier="drought",
            title="Albatross",
            abstract="Seasonal hydroclimate forecast with leakage-free LOOCV skill, using EOF/PCR and climate indices.",
            metadata=[
                Metadata("PyWPS", "https://pywps.org/"),
                Metadata("Birdhouse", "http://bird-house.github.io/"),
            ],
            version="0.3",
            inputs=inputs,
            outputs=outputs,
            store_supported=True,
            status_supported=True,
        )

    def _handler(self, request, response):
        LOGGER.info("Processing...")
        response.update_status("Retrieving input data and initializing variables...", 10)

        # -------------------------
        # Inputs
        # -------------------------
        glo_var_name = (_opt(request, "glo_var_name") or "sst").lower()
        months = [int(m) for m in request.inputs["month"][0].data.split(",")]
        if (not 1 <= months[0] <= 12 or months[-1] > 15
                or any(b - a != 1 for a, b in zip(months, months[1:]))):
            raise ValueError("Target months must be consecutive, start in 1-12 and end at most "
                            "in 15 (March of the following year), e.g. 4,5,6 or 11,12,13,14,15.")
        aggregation = (_opt(request, "aggregation") or "mean").lower()
        M = int(_opt(request, "phase_mode") or 1)
        indicator = _opt(request, "indicator")
        if M > 1:
            if indicator is None:
                raise ValueError("indicator is required when phase_mode > 1.")
            indicator = indicator.upper()
            index_file = files("albatross").joinpath("data", f"{indicator}.txt")
        else:
            indicator, index_file = None, None
        index_label = indicator if M > 1 else "noindex"
        pr_input = request.inputs.get("target", [None])[0]
        fy = _opt(request, "forecast_year")
        forecast_year = int(fy) if fy else None

        # -------------------------
        # Directories
        # -------------------------
        workdir = Path(self.workdir)
        run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        run_root = workdir / f"run_{run_id}"
        run_root.mkdir(parents=True, exist_ok=True)
        clim_file = resolve_target_input(pr_input, workdir)
        pc_dir = run_root / "pc_vs_hindcast"
        pc_dir.mkdir(parents=True, exist_ok=True)

        sy_in, ey_in = _opt(request, "start_year"), _opt(request, "end_year")
        if sy_in is None or ey_in is None:
            others = []
            pc = _predictor_coverage(glo_var_name)
            if pc is not None:
                others.append(pc)
            if M > 1:
                others.append(_series_coverage(index_file))
            d_sy, d_ey = _default_years(_series_coverage(clim_file), months, others)
        startyr = int(sy_in) if sy_in else d_sy
        endyr = int(ey_in) if ey_in else d_ey
        if endyr < startyr:
            raise ValueError(f"end_year ({endyr}) is before start_year ({startyr}).")
        LOGGER.info(f"Analysis period: {startyr}-{endyr}")

        response.update_status("Loading training data...", 20)

        kwgroups = create_kwgroups(
            debug=True,
            climdata_months=months,
            climdata_startyr=startyr,
            n_yrs=endyr - startyr + 1,
            n_mon_glo_var=lag,
            n_mon_index=lag,
            glo_var_lag=lag,
            n_phases=M,
            phases_even=True,
            index_lag=lag,
            index_fp=index_file,
            climdata_fp=clim_file,
            glo_var_name=glo_var_name,
            climdata_aggregation=aggregation,
        )

        years = np.arange(startyr, endyr + 1)

        target, _, glo_var, index_avg, phaseind = get_data(
            kwgroups, glo_var_name=glo_var_name, workdir=workdir
        )

        # -------------------------
        # Drop anni con target NaN (tutti i vettori restano allineati)
        # -------------------------
        target = np.asarray(target, dtype=float)
        valid = np.isfinite(target)
        dropped_years = years[~valid].tolist()      
        if dropped_years:
            LOGGER.warning(f"Dropping {len(dropped_years)} year(s) with missing target: "
                           f"{dropped_years}")
            target = target[valid]
            glo_var = type(glo_var)(np.asarray(glo_var.data)[valid], glo_var.lat, glo_var.lon)
            index_avg = np.asarray(index_avg)[valid]
            phaseind = {k: np.asarray(v)[valid] for k, v in phaseind.items()}
            years = years[valid]

        # Indice richiesto solo se si usano le fasi
        if M > 1:
            bad_idx = ~np.isfinite(np.asarray(index_avg, dtype=float))
            if bad_idx.any():
                raise ValueError(f"Index {indicator} missing for training years {years[bad_idx].tolist()}")

        # -------------------------
        # Anomalie del predittore.
        # La climatologia e' una costante per pixel: la centratura dentro ogni fold
        # (x_mean del training) la annulla, quindi non introduce leakage nella LOOCV.
        # -------------------------
        clim_2d = np.nanmean(np.asarray(glo_var.data), axis=0)
        glo_var = type(glo_var)(np.asarray(glo_var.data) - clim_2d[None, :, :], glo_var.lat, glo_var.lon)

        phases = list(phaseind.keys())

        phase_meta = _compute_phase_thresholds(index_avg=np.asarray(index_avg, dtype=float), phaseind=phaseind)
        if M > 1:   # con una sola fase non ci sono soglie
            (pc_dir / "phase_thresholds.json").write_text(json.dumps(phase_meta, indent=2))

        training_config = {
            "indicator": indicator if M > 1 else None,
            "glo_var_name": glo_var_name,
            "season_months": list(months),
            "years": [int(years[0]), int(years[-1])],
            "n_years_used": int(years.size),
            "aggregation": aggregation,
            "n_mon_glo_var": kwgroups["glo_var"]["n_mon"],
            "glo_var_lag": kwgroups["glo_var"]["lag"],
            "n_phases": M,
            "validation": "LOOCV, pixel selection + PCA + regression refit in every fold",
            "loocv_cfg": LOOCV_CFG,
            "masks": {k: LOOCV_CFG.get(k, __import__("inspect").signature(NIPAphase.loocv_hindcast).parameters[k].default) for k in ("lat_limit", "orog_max_m")},
            "dropped_years_incomplete_season": dropped_years,
        }
        (pc_dir / "training_config.json").write_text(json.dumps(training_config, indent=2))

        response.update_status("Running LOOCV per phase...", 50)

        # -------------------------
        # Accumulators
        # -------------------------
        phase_models = []
        metrics_rows = []
        timeseries_rows = []
        overall_t, overall_y, overall_yhat, overall_phclim = [], [], [], []
        detrend_on = bool(LOOCV_CFG.get("detrend", False))

        def run_one_phase(phase: str) -> None:
            model = NIPAphase(target, glo_var_name, glo_var, index_avg, phaseind[phase])
            model.phase = phase
            model.years = years[phaseind[phase]]

            model.hindcast = model.loocv_hindcast(**LOOCV_CFG, t=np.asarray(model.years, dtype=float))

            # Modello operativo: stessi passi, stimati su tutti gli anni della fase
            model.fit_final(t=np.asarray(model.years, dtype=float), **LOOCV_CFG)
            model.save_final_model(pc_dir, clim_2d)

            # Mappa (solo visualizzazione, non entra nella skill):
            # pixel tenuti in almeno MAP_KEEP_FREQ dei fold, colorati con la r su tutta la fase.
            freq = model.loocv_keep_freq
            shown = freq >= MAP_KEEP_FREQ
            if detrend_on:   # stessa natura dei campi usati nei fold: senza trend (sull'intera fase)
                r_full = vcorr(X=linear_detrend(model.years, np.asarray(model.glo_var.data)),
                               y=linear_detrend(model.years, np.asarray(model.clim_data, dtype=float)))
            else:
                r_full = vcorr(X=np.asarray(model.glo_var.data), y=np.asarray(model.clim_data))
            model.corr_grid = np.ma.masked_array(r_full, mask=~shown)
            model.corr_grid_full = model.corr_grid
            np.savez_compressed(pc_dir / f"loocv_keep_freq_{phase}.npz",
                                freq=freq, r_full=r_full,
                                lat=np.asarray(model.glo_var.lat), lon=np.asarray(model.glo_var.lon),
                                years=model.years)

            # PC OUT-OF-SAMPLE: per ogni anno, la sua proiezione sulle EOF del fold che lo
            # esclude, con i coefficienti di quel fold (sono le PC che producono l'hindcast).
            # Osservato e hindcast sono in <var>_timeseries.csv.
            npc = np.asarray(model.loocv_npc_oos)
            kmax = int(npc.max()) if npc.size else 0
            oos = {"year": np.asarray(model.years, dtype=int),
                   "n_pc": npc, "n_pix": np.asarray(model.loocv_npix_oos)}
            for j in range(kmax):
                oos[f"PC{j + 1}_oos"] = model.loocv_pc_oos[:, j]
            for j in range(kmax):
                oos[f"coef_PC{j + 1}"] = model.loocv_coef_oos[:, j]
            oos["intercept"] = model.loocv_intercept_oos
            pd.DataFrame(oos).to_csv(pc_dir / f"pcs_oos_{phase}.csv", index=False)

            y_obs = np.asarray(model.clim_data, dtype=float)
            y_est = np.asarray(model.hindcast, dtype=float)
            y_trend = np.asarray(model.loocv_baseline, dtype=float)   # con detrend: trend LOO
            idx_ph = np.asarray(model.indicator, dtype=float)

            for k in range(len(model.years)):
                r = {"year": int(model.years[k]),
                     "observed": float(y_obs[k]),
                     "hindcast": float(y_est[k])}
                if detrend_on:
                    # hindcast = trend_component + teleconnection component
                    r["trend_component"] = float(y_trend[k])
                if M > 1:
                    r["phase"] = phase
                    r["index_value"] = float(idx_ph[k])
                timeseries_rows.append(r)

            row = {"phase": phase}
            row.update(compute_metrics_row(y_obs, y_est))
            row.update(compute_skill_mae_rmse(y_obs, y_est, eps=1e-12))
            row["n_fallback"] = int(getattr(model, "loocv_n_fallback", 0))
            metrics_rows.append(row)

            fin = np.isfinite(y_obs) & np.isfinite(y_est)
            y_f, yhat_f = y_obs[fin], y_est[fin]
            if y_f.size >= 2:
                overall_t.append(np.asarray(model.years, dtype=float)[fin])
                overall_y.append(y_f)
                overall_yhat.append(yhat_f)
                overall_phclim.append((y_f.sum() - y_f) / (y_f.size - 1))   # LOO clim della fase

            phase_models.append(model)

        for ph in phases:
            run_one_phase(ph)

        # -------------------------
        # Overall row (solo se piu' fasi).
        # Calcolata su TUTTI gli anni contro la climatologia LOO dell'intero campione,
        # come la riga 'allyears' di M=1, cosi' M=1 e M>1 sono confrontabili direttamente.
        # SSRMSE_phaseclimonly: skill della sola climatologia LOO di fase (quanto vale la
        # partizione da sola, senza il campo predittore).
        # -------------------------
        if len(phases) > 1 and overall_y:
            t_all = np.concatenate(overall_t)
            o = np.argsort(t_all)
            y_all = np.concatenate(overall_y)[o]
            yhat_all = np.concatenate(overall_yhat)[o]
            phclim_all = np.concatenate(overall_phclim)[o]
            clim_all = _loo_climatology(y_all)

            row = {"phase": "overall"}
            row.update(compute_metrics_row(y_all, yhat_all))
            row.update(_skill_vs_baseline(y_all, yhat_all, clim_all))
            row["SSRMSE_phaseclimonly"] = _skill_vs_baseline(y_all, phclim_all, clim_all)["SSRMSE"]
            row["n_fallback"] = int(sum(r.get("n_fallback", 0) for r in metrics_rows))
            metrics_rows.append(row)

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df["SSMAE_%"] = 100.0 * pd.to_numeric(metrics_df.get("SSMAE"), errors="coerce")
        metrics_df["SSRMSE_%"] = 100.0 * pd.to_numeric(metrics_df.get("SSRMSE"), errors="coerce")

        keep_cols = ["phase", "N", "MAE", "SSMAE_%", "RMSE", "SSRMSE_%", "R2", "BIAS",
                     "r", "r_pvalue", "n_fallback"]
        if len(phases) > 1:
            metrics_df["SSRMSE_phaseclimonly_%"] = (
                100.0 * pd.to_numeric(metrics_df["SSRMSE_phaseclimonly"], errors="coerce")
                if "SSRMSE_phaseclimonly" in metrics_df.columns else np.nan)
            keep_cols.append("SSRMSE_phaseclimonly_%")
        for c in keep_cols:
            if c not in metrics_df.columns:
                metrics_df[c] = np.nan
        metrics_df = metrics_df[keep_cols]

        metrics_fp = run_root / "metrics_by_phase.csv"
        metrics_df.to_csv(metrics_fp, index=False)
        LOGGER.info("\n" + metrics_df.to_string(index=False))

        # -------------------------
        # Plots
        # -------------------------
        combined_map_pdf = run_root / f"{glo_var_name}_corr_map_combined.pdf"
        combined_scatter_pdf = run_root / f"{glo_var_name}_scatter_combined.pdf"
        save_combined_phase_maps_pdf(phase_models, combined_map_pdf, index_label)
        _save_loocv_scatters_pdf(phase_models, combined_scatter_pdf)

        ts_file = run_root / f"{glo_var_name}_timeseries.csv"
        pd.DataFrame(timeseries_rows).sort_values("year").to_csv(ts_file, index=False)

        # -------------------------
        # Forecast operativo
        # -------------------------
        response.update_status("Operational forecast...", 85)
        models_by_phase = {m.phase: m for m in phase_models}
        fc = {
            "status": "not requested",
            "forecast_year": forecast_year,
            "target_months": ",".join(map(str, months)),
            "aggregation": aggregation,
            "predictor": glo_var_name,
            "indicator": indicator if M > 1 else None,
        }
        if forecast_year is not None:
            try:
                if years[0] <= forecast_year <= years[-1]:
                    LOGGER.warning(f"forecast_year {forecast_year} is inside the training period: "
                                   f"the forecast is in-sample.")
                kw_f = create_kwgroups(
                    debug=True, climdata_months=months, climdata_startyr=forecast_year, n_yrs=1,
                    n_mon_glo_var=lag, n_mon_index=lag, glo_var_lag=lag, n_phases=M,
                    phases_even=True, index_lag=lag, index_fp=index_file,
                    climdata_fp=clim_file, glo_var_name=glo_var_name,
                    climdata_aggregation=aggregation,
                )
                _, _, gv_f, idx_f, _ = get_data(kw_f, glo_var_name=glo_var_name,
                                                workdir=workdir, load_target=False)
                if not (np.allclose(np.asarray(gv_f.lat, float), np.asarray(glo_var.lat, float))
                        and np.allclose(np.asarray(gv_f.lon, float), np.asarray(glo_var.lon, float))):
                    raise ValueError("Predictor grid for the forecast year differs from training grid.")
                field = np.asarray(gv_f.data, dtype=float)[0] - clim_2d

                if M > 1:
                    x_idx = float(np.asarray(idx_f, dtype=float)[0])
                    if not np.isfinite(x_idx):
                        raise ValueError(f"Index {indicator} not available for the predictor "
                                         f"window of {forecast_year}.")
                    ph = pick_phase_from_thresholds(x_idx, phase_meta)
                else:
                    x_idx, ph = np.nan, phases[0]

                out = models_by_phase[ph].predict_final(field, forecast_year)
                mrow = metrics_df.loc[metrics_df["phase"] == ph].iloc[0]
                fc.update({
                    "status": "ok",
                    "predictor_months": ",".join(map(str, kw_f["glo_var"]["months"])),
                    "index_value": x_idx,
                    "phase": ph,
                    "forecast": out["forecast"],
                    "trend_component": out["trend_component"],
                    "teleconnection_component": out["teleconnection_component"],
                    "PC1": out["PC1"],
                    "n_pix": out["n_pix"],
                    "no_predictor_fallback": out["fallback"],
                    "phase_n_years": int(mrow["N"]),
                    "phase_r": mrow["r"],
                    "phase_r_pvalue": mrow["r_pvalue"],
                    "phase_SSRMSE_vs_clim_%": mrow["SSRMSE_%"],
                    "phase_SSMAE_vs_clim_%": mrow["SSMAE_%"],
                })
                LOGGER.info(f"Forecast {forecast_year} ({ph}): {out['forecast']:.3f}")
            except Exception as e:
                fc["status"] = f"failed: {e}"
                LOGGER.warning(f"Operational forecast failed: {e}")

        forecast_fp = run_root / "forecast.csv"
        pd.DataFrame([fc]).to_csv(forecast_fp, index=False)

        # -------------------------
        # WPS outputs
        # -------------------------
        response.outputs["scatter_plot"].file = str(combined_scatter_pdf)
        response.outputs["glo_var_map"].file = str(combined_map_pdf)
        response.outputs["metrics_table"].file = str(metrics_fp)
        response.outputs["forecast_file"].file = str(ts_file)
        response.outputs["operational_forecast"].file = str(forecast_fp)

        response.update_status("Packaging outputs...", 90)

        # Il target usato entra nell'archivio, cosi' la run e' riproducibile da sola
        try:
            shutil.copy(clim_file, run_root / f"target_{Path(clim_file).name}")
        except Exception as e:
            LOGGER.warning(f"Could not copy target file into the archive: {e}")

        zip_path = run_root / f"outputs_{index_label.replace(' ', '')}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for p in run_root.rglob("*"):
                if p.is_file() and p != zip_path:
                    zipf.write(p, arcname=str(p.relative_to(run_root)))
        response.outputs["forecast_bundle"].file = str(zip_path)

        # Copia locale della run (best effort), solo se ALBATROSS_OUT_DIR e' impostata:
        # utile in sviluppo, disattivata di default sul server.
        out_root = os.environ.get("ALBATROSS_OUT_DIR")
        if out_root:
            try:
                target_file = extract_target_name(clim_file)
                months_string = ",".join(map(str, months))
                out_folder = (
                    Path(out_root)
                    / f"{target_file}_{startyr}_{endyr}"
                    / f"{target_file}_{index_label}_{glo_var_name}_{M}_{months_string}_{startyr}_{endyr}"
                )
                out_folder.mkdir(parents=True, exist_ok=True)
                dst_run = out_folder / run_root.name
                if dst_run.exists():
                    shutil.rmtree(dst_run)
                shutil.copytree(run_root, dst_run)
            except Exception as e:
                LOGGER.warning(f"Could not copy run folder: {e}")

        response.update_status("Drought process completed successfully!", 100)
        return response