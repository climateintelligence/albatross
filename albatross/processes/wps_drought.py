import logging
from pywps import Process, LiteralInput, ComplexInput, ComplexOutput
from pywps import FORMATS, Format
from pathlib import Path
from importlib.resources import files
from pywps.app.Common import Metadata

# drought specific functions
from albatross.climdiv_data import get_data, create_kwgroups
from albatross.new_simpleNIPA import NIPAphase
from albatross.utils import sstMap, make_scatterplot

# NIPA specific imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


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
                default=(
                    "https://github.com/climateintelligence/albatross/blob/new_NIPA/albatross/data/APGD_prcpComo.txt"
                ),
                supported_formats=[FORMATS.TEXT],
            ),
            ComplexInput(
                "indicator",
                "NAO Indicator",
                abstract="examples: climate indicator of tele-connection patterns",
                default=(
                    "https://github.com/climateintelligence/albatross/blob/new_NIPA/albatross/data/nao.txt"
                ),
                supported_formats=[FORMATS.TEXT],
            ),
            LiteralInput(
                "start_year",
                "Start Year",
                abstract="1971",
                # keywords=['name', 'firstname'],
                default="1971",
                data_type="string",
            ),
            LiteralInput(
                "end_year",
                "End Year",
                abstract="2008",
                default="2008",
                data_type="string",
            ),  # this is new added user defined parameter

            LiteralInput(
                "month",
                "Forecasted Month",
                default="1,2,3",  # 👈 Provide multiple months as a comma-separated string
                data_type="string",
            )
            ,  # this is new added user defined parameter
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
                "sst_map",
                "SST Map",
                as_reference=True,
                supported_formats=[FORMAT_PNG],
            ),
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
            endyr = int(request.inputs [ "end_year" ] [ 0 ].data)
        except KeyError as e:
            LOGGER.error(f"Missing required input parameter: {e}")
            response.status = "Failed"
            return response

        LOGGER.info("Select the input-output files")
        index_file = files("albatross").joinpath("data", "nao.txt")
        clim_file = files("albatross").joinpath("data", "APGD_prcpComo.txt")
        filename = "testComoNAO"

        workdir = Path(self.workdir)

        # Ensure output directories exist
        (workdir / "sst_maps").mkdir(parents=True, exist_ok=True)
        (workdir / "scatter_plots").mkdir(parents=True, exist_ok=True)

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
            n_phases=2,
            phases_even=True,
            index_lag=3,
            index_fp=index_file,
            climdata_fp=clim_file,
        )

        M = 2
        crv_flag = True
        map_flag = True

        # Years goes from start year to end year
        years = np.arange(startyr, endyr + 1)

        # Run Model
        climdata, sst, index, phaseind = get_data(kwgroups)

        # Output file paths
        sst_fp = workdir / "sst_maps" / f"{filename}_sst.png"
        scatter_fp = workdir / "scatter_plots" / f"{filename}_scatter.png"
        ts_file = workdir / f"{filename}_timeseries.csv"

        fig, axes = plt.subplots(M, 1, figsize=(6, 12))
        timeseries = {'years': [ ], 'data': [ ], 'hindcast': [ ]}
        pc1 = {'pc1': [ ]}

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

            # plot hindcast vs data
            make_scatterplot(model, scatter_fp)

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
                if not crv_flag:
                    if hasattr(model, 'pc1'):
                        pc1 [ 'pc1' ].append(model.pc1)

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

        # plot hindcast vs data
        make_scatterplot(model, scatter_fp)
        df = pd.DataFrame(timeseries)
        df.to_csv(ts_file)

        if not crv_flag:
            # save PC
            pc1 [ 'pc1' ] = np.concatenate(pc1 [ 'pc1' ])
            pc_file = './output/%s_pc1SST.csv' % (filename)
            df1 = pd.DataFrame(pc1)
            df1.to_csv(pc_file)

        print('NIPA run completed')

        # Update status before saving outputs
        response.update_status("Saving outputs and generating final results...", 90)

        # Save and check if files exist before assigning to response
        if ts_file.exists():
            response.outputs [ "forecast_file" ].file = ts_file
        else:
            LOGGER.warning(f"Forecast file {ts_file} not found!")

        if scatter_fp.exists():
            response.outputs [ "scatter_plot" ].file = scatter_fp
        else:
            LOGGER.warning(f"Scatter plot {scatter_fp} not found!")

        if sst_fp.exists():
            response.outputs [ "sst_map" ].file = sst_fp
        else:
            LOGGER.warning(f"SST map {sst_fp} not found!")

        response.update_status("Drought process completed successfully!", 100)

        return response
