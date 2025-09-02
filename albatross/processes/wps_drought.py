import logging

from pywps import Process, LiteralInput, ComplexInput, ComplexOutput
from pywps import FORMATS, Format
from pathlib import Path
from importlib.resources import files
from pywps.app.Common import Metadata

# drought specific functions
from albatross.climdiv_data import get_data, create_kwgroups
from albatross.new_simpleNIPA import NIPAphase
from albatross.utils import sstMap, plot_model_results, append_model_pcs

# NIPA specific imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from datetime import datetime
import shutil
import zipfile
import os
from urllib.parse import urlparse
import requests

LOGGER = logging.getLogger("PYWPS")
FORMAT_PNG = Format("image/png", extension=".png", encoding="base64")
default_target_file = "APGD_prcpComo"
crv_flag = True
map_flag = True

class Drought(Process):
    """A process to forecast precipitation."""

    def __init__(self):
        inputs = [
            ComplexInput(
                "pr",
                "Monthly global precipitation file",
                abstract="text file of precipitation",
                default=f"https://raw.githubusercontent.com/climateintelligence/albatross/refs/heads/main/albatross/data/{default_target_file}.txt",
                supported_formats=[FORMATS.TEXT],
            ),
            LiteralInput(
                "start_year",
                "Start Year",
                default="1971",
                data_type="string",
            ),
            LiteralInput(
                "end_year",
                "End Year",
                default="2008",
                data_type="string",
            ),
            LiteralInput(
                "month",
                "Target season",
                default="4,5,6",  # 👈 Provide multiple months as a comma-separated string
                data_type="string",
            ),
            LiteralInput(
                "indicator",
                "Climate Indicator",
                abstract="Choose between 'NAO', 'ONI', 'QBO','EAWR','WP', or 'PNA'",
                data_type="string",
                allowed_values=[ 'NAO', 'ONI', 'QBO','EAWR','WP', 'PNA' ],
                default="NAO",
            ),
            LiteralInput(
                "phase_mode",
                "Phase mode: 1 (allyears) or 2 (positive/negative phases)",
                abstract="Set to 1 for one-phase mode (allyears), or 2 for two-phase mode (pos/neg)",
                default="2",
                data_type="string",
            )
        ]
        outputs = [
            ComplexOutput(
                "forecast_file",
                "Forecast File ",
                as_reference=True,
                supported_formats=[FORMATS.TEXT],
            ),
            ComplexOutput(
                "scatter_plot",
                "Scatter Plot",
                as_reference=True,
                supported_formats=[FORMAT_PNG],
            ),
            ComplexOutput(
                "forecast_bundle",
                "All Forecast Outputs (ZIP)",
                as_reference=True,
                supported_formats=[ FORMATS.ZIP ],
            ),
            ComplexOutput(
                "sst_map",
                "SST Map",
                as_reference=True,
                supported_formats=[FORMAT_PNG],
            ),
            ComplexOutput(
                "pcs_plot",
                "Predicted vs Observed (PCA regression)",
                as_reference=True,
                supported_formats=[ FORMAT_PNG ],
            )
        ]

        super(Drought, self).__init__(
            self._handler,
            identifier="drought",
            title="Albatross",
            abstract="Albatross is a process designed to forecast seasonal hydroclimatic variables based on climate "
                     "indicators such as the North Atlantic Oscillation (NAO) or the Oceanic Niño Index (ONI). "
                     "It is built on the NIPA model (Zimmermann et al., 2016) and uses Principal Component "
                     "Regression (PCR) to analyze historical data and construct a forecasting model.",
            metadata=[
                Metadata("PyWPS", "https://pywps.org/"),
                Metadata("Birdhouse", "http://bird-house.github.io/"),
                Metadata("PyWPS Demo", "https://pywps-demo.readthedocs.io/en/latest/"),
            ],
            version="0.1",
            inputs=inputs,
            outputs=outputs,
            store_supported=True,
            status_supported=True,
        )

    def _handler(self, request, response):
        LOGGER.info("Processing...")

        # Status update
        response.update_status("Retrieving input data and initializing variables...", 10)

        try:
            months = list(map(int, request.inputs["month"][0].data.split(",")))
            startyr = int(request.inputs["start_year"][0].data)
            endyr = int(request.inputs["end_year"][0].data)
        except KeyError as e:
            LOGGER.error(f"Missing required input parameter: {e}")
            response.status = "Failed"
            return response

        M = int(request.inputs [ "phase_mode" ] [ 0 ].data)
        if M not in [ 1, 2 ]:
            raise ValueError("phase_mode must be 1 or 2")


        LOGGER.info("Select the input-output files")
        indicator = request.inputs [ "indicator" ] [ 0 ].data.upper()
        index_file = files("albatross").joinpath("data", f"{indicator}.txt")

        pr_input = request.inputs.get("pr", [ None ]) [ 0 ]


        def resolve_precip_input(pr_input, workdir, fallback_file):
            import requests
            from pathlib import Path

            if pr_input and hasattr(pr_input, "data") and isinstance(pr_input.data, str) and pr_input.data.startswith(
                "http"):
                url = pr_input.data
                target = Path(workdir) / "downloaded_precip.txt"
                LOGGER.info(f"Downloading climate data from URL: {url}")
                try:
                    r = requests.get(url)
                    r.raise_for_status()
                    with open(target, "wb") as out:
                        out.write(r.content)
                    return target
                except Exception as e:
                    LOGGER.error(f"Failed to download climate data: {e}")
                    raise

            if hasattr(pr_input, "file") and pr_input.file:
                path = Path(pr_input.file)
                with open(path, "r") as f:
                    first_line = f.readline().strip()
                    second_line = f.readline().strip()

                if first_line.startswith("http") and not second_line:
                    url = first_line
                    target = Path(workdir) / "downloaded_precip.txt"
                    LOGGER.info(f"Downloading climate data from URL: {url}")
                    r = requests.get(url)
                    r.raise_for_status()
                    with open(target, "wb") as out:
                        out.write(r.content)
                    return target
                else:
                    return path

            return fallback_file

        clim_file = resolve_precip_input(pr_input, self.workdir, files("albatross").joinpath("data", f"{default_target_file}.txt"))

        workdir = Path(self.workdir)

        # Ensure output directories exist
        (workdir / "sst_maps").mkdir(parents=True, exist_ok=True)
        (workdir / "scatter_plots").mkdir(parents=True, exist_ok=True)
        (workdir / "pc_vs_hindcast").mkdir(parents=True, exist_ok=True)

        # Update status
        response.update_status("NIPA running...", 50)

        # Processing steps...
        kwgroups = create_kwgroups(
            debug=True,
            climdata_months=months,
            climdata_startyr=startyr,
            n_yrs= endyr - startyr + 1,
            n_mon_sst=3,
            n_mon_index=3,
            sst_lag=3,
            n_phases=M,
            phases_even=True,
            index_lag=3,
            index_fp=index_file,
            climdata_fp=clim_file,
        )

        # Years goes from start year to end year
        years = np.arange(startyr, endyr + 1)
        workdir = Path(self.workdir)

        # Run Model
        climdata, sst, index, phaseind = get_data(kwgroups, workdir=workdir)

        # Output file paths
        sst_fp = workdir / "sst_maps" / f"corr_map.png"
        ts_file = workdir / f"timeseries.csv"

        fig, axes = plt.subplots(M, 1, figsize=(6, 12))
        pcs_file_rows = [ ]
        scatter_files = []
        timeseries_rows = []


        if M==1:
            phase = 'allyears'
            model = NIPAphase(climdata, sst, index, phaseind [ phase ])
            model.phase = phase
            model.years = years [ phaseind [ phase ] ]
            model.bootcorr(corrconf=0.95)
            model.gridCheck()
            print("boot corr and grid check moved inside")
            model.crossvalpcr(xval=crv_flag)
            for i in range(len(model.years)):
                timeseries_rows.append({
                    "year": int(model.years [ i ]),
                    "observed": float(model.clim_data [ i ]),
                    "hindcast": float(model.hindcast [ i ]),
                    "phase": model.phase  # Optional: drop this if not needed
                })

            model.save_regressor(workdir)

            if map_flag:
                fig, axes, m = sstMap(model, fig=fig, ax=axes)
                axes.set_title('%s, %.2f' % (phase, model.correlation))
                fig.savefig(sst_fp)
                plt.close(fig)

            scatter_fp = workdir / "scatter_plots" / f"{phase}.png"
            plot_model_results(model, scatter_fp, crv_flag=crv_flag)
            scatter_files.append(scatter_fp)
            append_model_pcs(model, pcs_file_rows)
            print(f"✔ pcs_file_rows contains {len(pcs_file_rows)} rows")

            df = pd.DataFrame(timeseries_rows).sort_values("year")
            df.to_csv(ts_file, index=False)

        else:
            for phase, ax in zip(phaseind, axes):
                model = NIPAphase(climdata, sst, index, phaseind [ phase ])
                model.phase = phase
                model.years = years [ phaseind [ phase ] ]
                model.bootcorr(corrconf=0.95)
                model.gridCheck()
                model.crossvalpcr(xval=crv_flag)
                model.save_regressor(workdir)
                for i in range(len(model.years)):
                    timeseries_rows.append({
                        "year": int(model.years [ i ]),
                        "observed": float(model.clim_data [ i ]),
                        "hindcast": float(model.hindcast [ i ]),
                        "phase": model.phase  # Optional: drop this if not needed
                    })

                if map_flag:
                    fig, ax, m = sstMap(model, fig=fig, ax=ax)
                    ax.set_title(
                        '%s, %s' % (phase, f"{model.correlation:.2f}" if model.correlation is not None else "N/A"))
                    # ax.set_title('%s, %.2f' % (phase, model.correlation))
                    fig.savefig(sst_fp)
                    plt.close(fig)

                scatter_files.append((model, phase))
                append_model_pcs(model, pcs_file_rows)
                print(f"✔ pcs_file_rows contains {len(pcs_file_rows)} rows")

                df = pd.DataFrame(timeseries_rows).sort_values("year")
                df.to_csv(ts_file, index=False)

        df = pd.DataFrame(timeseries_rows).sort_values("year")
        df.to_csv(ts_file, index=False)

        print('NIPA run completed')

        # Update status before saving outputs
        response.update_status("Saving outputs and generating final results...", 90)

        # Assign outputs if they exist
        if ts_file.exists():
            response.outputs [ "forecast_file" ].file = ts_file
        else:
            LOGGER.warning(f"Forecast file {ts_file} not found!")

        # Only assign scatter plot if any were generated
        if scatter_files:
            if M==1:
                # Single-phase mode: just return the only plot
                scatter_fp = scatter_files [ 0 ]
                if scatter_fp.exists():
                    response.outputs [ "scatter_plot" ].file = scatter_fp
                else:
                    LOGGER.warning(f"Scatter plot file {scatter_fp} does not exist!")
            else:
                # Two-phase mode: combine both scatter plots side-by-side
                combined_fp = workdir / "scatter_plots" / "combined_scatter.png"
                fig, axs = plt.subplots(1, len(scatter_files), figsize=(12, 6))

                if len(scatter_files)==1:
                    axs = [ axs ]  # Ensure axs is iterable if only one

                for ax, (model, phase) in zip(axs, scatter_files):
                    plot_model_results(model, ax=ax, crv_flag=crv_flag)
                    ax.set_title(phase)

                plt.tight_layout()
                fig.savefig(combined_fp)
                plt.close(fig)

                if combined_fp.exists():
                    response.outputs [ "scatter_plot" ].file = combined_fp
                else:
                    LOGGER.warning("Combined scatter plot could not be created.")
        else:
            LOGGER.warning("No scatter plots were generated.")

        # SST map
        if sst_fp.exists():
            response.outputs [ "sst_map" ].file = sst_fp
        else:
            LOGGER.warning(f"SST map {sst_fp} not found!")

        pcs_df = pd.DataFrame(pcs_file_rows)
        pcs_csv_fp = workdir / "pc_vs_hindcast" / "pcs_vs_hindcast.csv"
        pcs_df.to_csv(pcs_csv_fp, index=False)

        # Mark as complete
        response.update_status("Drought process completed successfully!", 100)

        # Save the forecast script for future use
        forecast_script_path = workdir / "predict_next_year.py"
        import textwrap

        script = textwrap.dedent("""
            import numpy as np
            import pandas as pd
            from pathlib import Path
            from sklearn.linear_model import LinearRegression
            import xarray as xr

            #-----------------------------
            # Parameters to be set by user
            #-----------------------------

            PHASE = "pos"
            YEAR = 2025
            SEASON_TO_FORECAST = [1, 2, 3]

            #-----------------------------

            def weightsst(sst):

            import numpy as np
            from numpy import cos, radians

            # If 3D, remove singleton zlev
            if "zlev" in sst.dims and sst.sizes["zlev"] == 1:
            sst = sst.squeeze("zlev")

            # Get latitude
            lat = sst.coords["Y"]
            weights = cos(radians(lat))

            # Expand weights to match shape (Y, X)
            weights_2d = xr.DataArray(weights, dims=["Y"]).broadcast_like(sst)

            # Apply weights directly using xarray broadcasting
            weighted_sst = sst * weights_2d

            return weighted_sst

            def get_future_sst_3mon(year, season, anomalies=True):
            import xarray as xr

            var = "anom" if anomalies else "sst"
            url = f"https://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCDC/.ERSST/.version5/.{var}/dods"

            # Do not decode time automatically
            ds = xr.open_dataset(url, decode_times=False)

            # Select time indices manually if needed
            # For now, take the last 3 time steps (approx JFM 2025)
            # You could refine this based on exact time indexing
            sst = ds[var].isel(T=slice(-3, None)).mean(dim="T")

            return sst

            MODEL_DIR = Path(".")
            OUTPUT_CSV = MODEL_DIR / f"forecast_{PHASE}_{YEAR}.csv"

            eofs = pd.read_csv(MODEL_DIR / f"eofs_{PHASE}.csv").values
            coefs = pd.read_csv(MODEL_DIR / f"coefficients_{PHASE}.csv", index_col=0)
            reg = LinearRegression()
            reg.coef_ = coefs.loc[coefs.index != "Intercept", "Coefficient"].values
            reg.intercept_ = coefs.loc["Intercept", "Coefficient"]

            sst = get_future_sst_3mon(YEAR, SEASON_TO_FORECAST, anomalies=True)
            print(sst)
            masked_sst = weightsst(sst).values
            mask = np.load(MODEL_DIR / f"sst_mask_{PHASE}.npy")
            flat_sst = masked_sst[~mask]
            pcs = flat_sst.dot(eofs)

            forecast = reg.predict([pcs])[0]
            print(f"Forecasted precipitation for {YEAR}: {forecast:.2f}")

            pd.DataFrame({ "year": [YEAR], "forecast": [forecast], "phase": [PHASE] }).to_csv(OUTPUT_CSV, index=False)
            print(f"Saved forecast to {OUTPUT_CSV}")
            """)

        forecast_script_path.write_text(script)

        # Create ZIP archive
        zip_path = workdir / f"outputs_{indicator}.zip"
        with zipfile.ZipFile(zip_path, "w") as zipf:
            if ts_file.exists():
                zipf.write(ts_file, arcname=ts_file.name)

            if sst_fp.exists():
                zipf.write(sst_fp, arcname=sst_fp.name)

            combined_fp = workdir / "scatter_plots" / "combined_scatter.png"
            if combined_fp.exists():
                zipf.write(combined_fp, arcname=combined_fp.name)

            if pcs_csv_fp.exists():
                zipf.write(pcs_csv_fp, arcname=pcs_csv_fp.name)

            # NEW: Add regressor outputs to ZIP (for both phases if 2-phase)
            for phase in phaseind if M==2 else [ "allyears" ]:
                eofs_path = workdir / "pc_vs_hindcast" / f"eofs_{phase}.csv"
                coef_path = workdir / "pc_vs_hindcast" / f"coefficients_{phase}.csv"
                if eofs_path.exists():
                    zipf.write(eofs_path, arcname=eofs_path.name)
                if coef_path.exists():
                    zipf.write(coef_path, arcname=coef_path.name)

                mask_path = workdir / "pc_vs_hindcast" / f"sst_mask_{phase}.npy"
                if mask_path.exists():
                    zipf.write(mask_path, arcname=mask_path.name)

                if forecast_script_path.exists():
                    zipf.write(forecast_script_path, arcname=forecast_script_path.name)

        # Assign final ZIP to response
        response.outputs [ "forecast_bundle" ].file = zip_path

        """# Copy main outputs to Desktop (adjust for OS or path if needed)
        try:
            desktop_path = Path.home() / "Desktop"

            shutil.copy(ts_file, desktop_path / ts_file.name)
            if sst_fp.exists():
                shutil.copy(sst_fp, desktop_path / sst_fp.name)
            if 'combined_fp' in locals() and combined_fp.exists():
                shutil.copy(combined_fp, desktop_path / combined_fp.name)
            if zip_path.exists():
                shutil.copy(zip_path, desktop_path / zip_path.name)
            if pcs_csv_fp.exists():
                shutil.copy(pcs_csv_fp, desktop_path / pcs_csv_fp.name)

            LOGGER.info("Outputs copied to Desktop.")

        except Exception as e:
            LOGGER.warning(f"Could not copy to Desktop: {e}")"""

        return response
