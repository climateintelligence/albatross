import logging

from pywps import Process, LiteralInput, ComplexInput, ComplexOutput
from pywps import FORMATS, Format
from pathlib import Path
from importlib.resources import files
from pywps.app.Common import Metadata

# drought specific functions
from albatross.climdiv_data import get_data, create_kwgroups
from albatross.new_simpleNIPA import NIPAphase
from albatross.utils import sstMap, make_scatterplot, plot_pc1_vs_true

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


LOGGER = logging.getLogger("PYWPS")
FORMAT_PNG = Format("image/png", extension=".png", encoding="base64")


class Drought(Process):
    """A process to forecast precipitation."""

    def __init__(self):
        inputs = [
            ComplexInput(
                "pr",
                "Monthly global precipitation file",
                abstract="text file of precipitation",
                default="https://github.com/climateintelligence/albatross/blob/new_NIPA/albatross/data/APGD_prcpComo.txt",
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
                "Forecasted Month",
                default="1,2,3",  # 👈 Provide multiple months as a comma-separated string
                data_type="string",
            ),
            LiteralInput(
                "indicator",
                "Climate Indicator",
                abstract="Choose between 'nao' or 'mei'",
                data_type="string",
                allowed_values=[ "nao", "mei" ],
                default="nao",
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
                "pc1_plot",
                "PC1 vs Hindcast Plot",
                as_reference=True,
                supported_formats=[ FORMAT_PNG ],
            )
        ]

        super(Drought, self).__init__(
            self._handler,
            identifier="drought",
            title="A process to forecast precipitation.",
            abstract="A process to forecast precipitation...",
            # keywords=['hello', 'demo'],
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
        LOGGER.info("Starting drought processing...")

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

        crv_flag = True
        map_flag = True

        LOGGER.info("Select the input-output files")

        indicator = request.inputs [ "indicator" ] [ 0 ].data.lower()
        if indicator=="nao":
            index_file = files("albatross").joinpath("data", "nao.txt")
        elif indicator=="mei":
            index_file = files("albatross").joinpath("data", "mei.txt")
        else:
            raise Exception(f"Unsupported indicator: {indicator}")

        pr_input = request.inputs.get("pr", [ None ]) [ 0 ]

        # Handle remote and local references safely
        # Handle the file input (if provided)
        if hasattr(pr_input, "file") and pr_input.file:
            clim_file = Path(pr_input.file)
            LOGGER.info(f"Loaded climate file from user upload: {clim_file}")
        else:
            clim_file = files("albatross").joinpath("data", "APGD_prcpComo.txt")
            LOGGER.warning(f"Falling back to default climate file: {clim_file}")

        workdir = Path(self.workdir)

        # Ensure output directories exist
        (workdir / "sst_maps").mkdir(parents=True, exist_ok=True)
        (workdir / "scatter_plots").mkdir(parents=True, exist_ok=True)
        (workdir / "pc_vs_hindcast").mkdir(parents=True, exist_ok=True)

        # Update status
        response.update_status("Running NIPA drought model...", 50)

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

        # Run Model
        climdata, sst, index, phaseind = get_data(kwgroups)

        # Output file paths
        sst_fp = workdir / "sst_maps" / f"corr_map.png"
        ts_file = workdir / f"timeseries.csv"

        fig, axes = plt.subplots(M, 1, figsize=(6, 12))
        timeseries = {'years': [ ], 'data': [ ], 'hindcast': [ ]}
        pc1 = {'pc1': [ ]}
        scatter_fp = None
        scatter_files = []
        pc1_plot_fp = None
        pc1_plot_files = []

        print('NIPA running...')
        if M==1:
            phase = 'allyears'
            model = NIPAphase(climdata, sst, index, phaseind [ phase ])
            model.phase = phase
            model.years = years [ phaseind [ phase ] ]
            model.bootcorr(corrconf=0.95)
            model.gridCheck()
            model.crossvalpcr(xval=crv_flag)
            timeseries [ 'years' ] = model.years
            timeseries [ 'data' ] = model.clim_data
            timeseries [ 'hindcast' ] = model.hindcast

            if map_flag:
                fig, axes, m = sstMap(model, fig=fig, ax=axes)
                axes.set_title('%s, %.2f' % (phase, model.correlation))
                fig.savefig(sst_fp)
                plt.close(fig)

            if not crv_flag and hasattr(model, 'pc1'):
                # PC1 plotting
                pc1_plot_fp = workdir / "pc_vs_hindcast" / f"pc1_vs_hindcast.png"
                plot_pc1_vs_true(model, pc1_plot_fp)
                pc1 [ 'pc1' ].append(model.pc1)
                pc1_plot_files.append(pc1_plot_fp)

            if crv_flag:
                # Always generate scatter plot if cross-validation is used
                scatter_fp = workdir / "scatter_plots" / f"{phase}_scatter.png"
                make_scatterplot(model, scatter_fp)
                scatter_files.append(scatter_fp)

        else:
            for phase, ax in zip(phaseind, axes):
                model = NIPAphase(climdata, sst, index, phaseind [ phase ])
                model.phase = phase
                model.years = years [ phaseind [ phase ] ]
                model.bootcorr(corrconf=0.95)
                model.gridCheck()
                model.crossvalpcr(xval=crv_flag)
                timeseries [ 'years' ].append(model.years)
                timeseries [ 'data' ].append(model.clim_data)
                timeseries [ 'hindcast' ].append(model.hindcast)

                if not crv_flag and hasattr(model, 'pc1'):
                    # PC1 plotting
                    pc1_plot_fp = workdir / "pc_vs_hindcast" / f"pc1_vs_hindcast_phase_{phase}.png"
                    plot_pc1_vs_true(model, pc1_plot_fp)
                    pc1 [ 'pc1' ].append(model.pc1)
                    pc1_plot_files.append(pc1_plot_fp)

                if crv_flag:
                    # Always generate scatter plot if cross-validation is used
                    scatter_fp = workdir / "scatter_plots" / f"{phase}_scatter.png"
                    make_scatterplot(model, scatter_fp)
                    scatter_files.append(scatter_fp)

                if map_flag:
                    fig, ax, m = sstMap(model, fig=fig, ax=ax)
                    ax.set_title('%s, %.2f' % (phase, model.correlation))
                    fig.savefig(sst_fp)
                    plt.close(fig)

        # save timeseries (exceptions handled only for 2 phase analysis)
        if np.size(timeseries [ 'hindcast' ] [ 0 ])==1:
            if math.isnan(timeseries [ 'hindcast' ] [ 0 ]):
                # no result for the first phase -> consider only the second set of results
                timeseries [ 'years' ] = timeseries [ 'years' ] [ 1 ]
                timeseries [ 'data' ] = timeseries [ 'data' ] [ 1 ]
                timeseries [ 'hindcast' ] = timeseries [ 'hindcast' ] [ 1 ]

        elif np.size(timeseries [ 'hindcast' ] [ 1 ])==1:
            if math.isnan(timeseries [ 'hindcast' ] [ 1 ]):
                # no result for the second phase -> consider only the first set of results
                timeseries [ 'years' ] = timeseries [ 'years' ] [ 0 ]
                timeseries [ 'data' ] = timeseries [ 'data' ] [ 0 ]
                timeseries [ 'hindcast' ] = timeseries [ 'hindcast' ] [ 0 ]

        else:
            timeseries [ 'years' ] = np.concatenate(timeseries [ 'years' ])
            timeseries [ 'data' ] = np.concatenate(timeseries [ 'data' ])
            timeseries [ 'hindcast' ] = np.concatenate(timeseries [ 'hindcast' ])

        df = pd.DataFrame(timeseries)
        df.to_csv(ts_file)

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
            scatter_fp = scatter_files [ 0 ]
            if scatter_fp.exists():
                response.outputs [ "scatter_plot" ].file = scatter_fp
            else:
                LOGGER.warning(f"Scatter plot file {scatter_fp} does not exist!")
        else:
            LOGGER.warning("No scatter plots were generated.")

        if pc1_plot_files:
            selected_plot = pc1_plot_files [ 0 ]
            if selected_plot.exists():
                response.outputs [ "pc1_plot" ].file = selected_plot
            else:
                LOGGER.warning(f"PC1 plot file {selected_plot} not found.")
        else:
            LOGGER.warning("No PC1 plots were generated.")

        # SST map
        if sst_fp.exists():
            response.outputs [ "sst_map" ].file = sst_fp
        else:
            LOGGER.warning(f"SST map {sst_fp} not found!")

        # Mark as complete
        response.update_status("Drought process completed successfully!", 100)

        # Local save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path.home() / "Desktop" / "wps_outputs" / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create ZIP archive
        zip_path = workdir / f"outputs_{indicator}.zip"
        with zipfile.ZipFile(zip_path, "w") as zipf:
            if ts_file.exists():
                zipf.write(ts_file, arcname=ts_file.name)

            if sst_fp.exists():
                zipf.write(sst_fp, arcname=sst_fp.name)

            # Write all scatter plots
            for scatter_plot in scatter_files:
                if scatter_plot.exists():
                    zipf.write(scatter_plot, arcname=scatter_plot.name)

            # Write all pc1_vs_hindcast plots
            for pc1_plot in pc1_plot_files:
                if pc1_plot.exists():
                    zipf.write(pc1_plot, arcname=pc1_plot.name)

            # Extra: phase-specific scatter plots
            for phase in phaseind:
                phase_fp = workdir / "scatter_plots" / f"{phase}_scatter.png"
                if phase_fp.exists():
                    zipf.write(phase_fp, arcname=phase_fp.name)

        # Assign final ZIP to response
        response.outputs [ "forecast_bundle" ].file = zip_path

        # Optionally copy to desktop
        shutil.copy(zip_path, output_dir / zip_path.name)
        print(f"Output files copied to: {output_dir}")

        return response
