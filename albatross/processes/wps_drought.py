import logging
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from pywps import ComplexOutput, Format, LiteralInput, LiteralOutput, Process
from pywps.app.Common import Metadata

# drought specific functions
from albatross.climdiv_data import get_data, create_kwgroups
from albatross.new_simpleNIPA import NIPAphase
from albatross.utils import sstMap

LOGGER = logging.getLogger("PYWPS")
FORMAT_PNG = Format("image/png", extension=".png", encoding="base64")


class Drought(Process):
    """A process to forecast precipitation."""

    def __init__(self):
        inputs = [
            LiteralInput(
                'pr',
                'Monthly global precipitation file',
                abstract='text file of precipitation',
                # keywords=['name', 'firstname'],
                default=(
                    "https://raw.githubusercontent.com/mxgiuliani00/CSI/master/"
                    "NIPA/DATA/APGD_prcpComo.txt"
                ),
                data_type='string'
            ),

            LiteralInput(
                'indicator',
                'NAO Indicator ',
                abstract='examples: climate indicator of tele-connection patterns',
                # keywords=['name', 'firstname'],
                default=(
                    "https://raw.githubusercontent.com/mxgiuliani00/CSI/master/NIPA/DATA/nao.txt"
                ),
                data_type='string'),
            LiteralInput(
                'start_year',
                'Start Year',
                abstract='2020',
                # keywords=['name', 'firstname'],
                default='2020',
                data_type='string'),
            LiteralInput(
                'end_year',
                'End Year',
                abstract='2022',
                default='2022',
                data_type='string'),  # this is new added user defined parameter

            LiteralInput(
                'month',
                'Forecasted Month',
                abstract='8',
                default='8',
                # keywords=['name', 'firstname'],
                data_type='string'),  # this is new added user defined parameter
        ]
        outputs = [

            LiteralOutput('forecast_file',
                          'Forecast File ',
                          # abstract='negativ list',
                          # keywords=['output', 'result', 'response'],
                          data_type='string'),
            ComplexOutput('scatter_plot',
                          'Scatter Plot',
                          # abstract='Plot of observations and predictions with a 1:1 line, accuracy and equation.',
                          as_reference=True,
                          supported_formats=[FORMAT_PNG]),
            ComplexOutput('sst_map',
                          'SST Map',
                          # abstract='Plot of selected SST map',
                          as_reference=True,
                          supported_formats=[FORMAT_PNG]),
        ]

        super(Drought, self).__init__(
            self._handler,
            identifier='drought',
            title='A process to forecast precipitation.',
            abstract='A process to forecast precipitation...',
            # keywords=['hello', 'demo'],
            metadata=[
                Metadata('PyWPS', 'https://pywps.org/'),
                Metadata('Birdhouse', 'http://bird-house.github.io/'),
                Metadata('PyWPS Demo', 'https://pywps-demo.readthedocs.io/en/latest/'),
            ],
            version='0.1',
            inputs=inputs,
            outputs=outputs,
            store_supported=True,
            status_supported=True
        )

    """@staticmethod
    def _handler(request, response):

        ############################
        LOGGER.info("get input parameter")
        ############################
        # original code https://github.com/mxgiuliani00/CSI/blob/master/NIPA/run_nipa.py
        LOGGER.info("start processing")
        ###########################
        LOGGER.info("Select the input-output files")

        import os
        os.getcwd()

        index_file = os.getcwd() + '/tests/DATA/nao.txt'
        clim_file = os.getcwd() + '/tests/DATA/APGD_prcpComo.txt'
        filename = 'testComoNAO'

        # #### USER INPUT ####
        LOGGER.info("configuration of the process")

        # Settings:
        M = 2  # number of climate signal's phases; fixed
        n_obs = 3  # number of observations (months); fixed
        lag = 3  # lag-time (months) --> 3 = seasonal; fixed
        months = month  # months to consider (J,F,M): [1,2,3] ; link this parameter to input parameter; from user input
        startyr = start_year  # beginning of the time period to analyze; from user input
        endyr = end_year  # end of the time period to analyze; from user input
        n_yrs = endyr - startyr + 1  # number of years to analyze
        fp = './maps/%s' % (filename) # link to output file path
        fig, axes = plt.subplots(M, 1, figsize=(6, 12))

        # Select the type of experiment:
        # crv_flag:
        #   True  = runs NIPA with crossvalidation
        #   False = runs NIPA without crossvalidation and save the first SST-Principal Component for multi-variate model
        #
        crv_flag = True
        sst_map_flag = True
        scatter_plot_flag = True

        ####################

        # Don't mess with the next few lines
        years = np.arange(startyr, startyr + n_yrs)"""

    LOGGER = logging.getLogger(__name__)

    def _handler(request, response):
        ############################
        LOGGER.info("get input parameter")
        ############################
        # Retrieve user inputs
        try:
            months = [ int(m) for m in request.inputs [ 'months' ].data ]  # Assuming a list of integers is passed
            startyr = int(request.inputs [ 'start_year' ].data)
            endyr = int(request.inputs [ 'end_year' ].data)
        except KeyError as e:
            LOGGER.error(f"Missing required input parameter: {e}")
            response.status = "Failed"
            return response

        ###########################
        LOGGER.info("start processing")
        ###########################
        LOGGER.info("Select the input-output files")

        index_file = os.path.join(os.getcwd(), 'tests/DATA/nao.txt')
        clim_file = os.path.join(os.getcwd(), 'tests/DATA/APGD_prcpComo.txt')
        filename = 'testComoNAO'

        # #### USER INPUT ####
        LOGGER.info("configuration of the process")

        # Settings:
        M = 2  # number of climate signal's phases; fixed
        n_obs = 3  # number of observations (months); fixed
        lag = 3  # lag-time (months) --> 3 = seasonal; fixed
        n_yrs = endyr - startyr + 1  # number of years to analyze
        fp = f'./maps/{filename}'  # link to output file path
        if not os.path.exists(fp):
            os.makedirs(fp)

        fig, axes = plt.subplots(M, 1, figsize=(6, 12))

        # Select the type of experiment:
        crv_flag = True
        sst_map_flag = True
        scatter_plot_flag = True

        ####################
        # Processing
        years = np.arange(startyr, startyr + n_yrs)

        kwgroups = create_kwgroups(debug=True, climdata_months=months,
                                   climdata_startyr=startyr, n_yrs=n_yrs,
                                   n_mon_sst=n_obs, n_mon_index=n_obs, sst_lag=lag,
                                   n_phases=M, phases_even=True,
                                   index_lag=lag,
                                   index_fp=index_file,
                                   climdata_fp=clim_file)

        climdata, sst, index, phaseind = get_data(kwgroups)
        sst_fp = './sst_maps/%s' % (filename)
        scatter_fp = './scatter_plots/%s' % (filename)
        sst_fig, sst_axes = plt.subplots(M, 1, figsize=(6, 12))
        scatter_fig, scatter_axes = plt.subplots(M, 1, figsize=(6, 12))
        timeseries = {'years': [], 'data': [], 'hindcast': []}
        # pc1 = {'pc1':[]}
        # reg_stats = {'slope':[],'intercept':[]}

        LOGGER.info("NIPA running...")  # print('NIPA running...')

        if M == 1:
            phase = 'allyears'
            model = NIPAphase(climdata, sst, index, phaseind[phase])
            model.phase = phase
            model.years = years[phaseind[phase]]
            model.bootcorr(corrconf=0.95)
            model.gridCheck()
            model.crossvalpcr(xval=crv_flag)
            timeseries['years'] = model.years
            timeseries['data'] = model.clim_data
            timeseries['hindcast'] = model.hindcast
            LOGGER.info(timeseries['years'])
            LOGGER.info(timeseries['data'])

            if sst_map_flag:
                fig, axes, m = sstMap(model, fig=fig, ax=axes)
                axes.set_title('%s, %.2f' % (phase, model.correlation))
                fig.savefig(fp)
                plt.close(fig)

        else:
            for phase, sst_ax, scatter_ax in zip(phaseind, sst_axes, scatter_axes):
                model = NIPAphase(climdata, sst, index, phaseind[phase])
                model.phase = phase
                model.years = years[phaseind[phase]]
                model.bootcorr(corrconf=0.95)
                model.gridCheck()
                model.crossvalpcr(xval=crv_flag)
                timeseries['years'].append(model.years)
                timeseries['data'].append(model.clim_data)
                timeseries['hindcast'].append(model.hindcast)

                if sst_map_flag:
                    sst_fig, sst_ax, m = sstMap(model, fig=sst_fig, ax=sst_ax)
                    sst_ax.set_title('%s, %.2f' % (phase, model.correlation))

                if scatter_plot_flag:
                    scatter_ax.scatter(model.clim_data, model.hindcast)
                    # Add 1:1 line
                    min_val = min(min(model.clim_data), min(model.hindcast))
                    max_val = max(max(model.clim_data), max(model.hindcast))
                    scatter_ax.plot([min_val, max_val], [min_val, max_val], 'r--',
                                    label='1:1 Line')  # Red dashed line
                    scatter_ax.set_title('%s, %.2f' % (phase, model.correlation))
                    scatter_ax.set_xlabel('observation')
                    scatter_ax.set_ylabel('forecast')
                    scatter_ax.legend()  # Add legend

            # Save SST figure
            if sst_map_flag:
                sst_fig.tight_layout()
                sst_fig.savefig(sst_fp)
                plt.close(sst_fig)

            # Save scatter figure
            if scatter_plot_flag:
                scatter_fig.tight_layout()
                scatter_fig.savefig(scatter_fp)
                plt.close(scatter_fig)

        # save timeseries (exceptions handled only for 2 phase analysis)
        if np.size(timeseries['hindcast'][0]) == 1:
            if math.isnan(timeseries['hindcast'][0]):
                # no result for the first phase -> consider only the second set of results
                timeseries['years'] = timeseries['years'][1]
                timeseries['data'] = timeseries['data'][1]
                timeseries['hindcast'] = timeseries['hindcast'][1]
                # reg_stats['slope'] = reg_stats['slope'][1]
                # reg_stats['intercept'] = reg_stats['intercept'][1]

        elif np.size(timeseries['hindcast'][1]) == 1:
            if math.isnan(timeseries['hindcast'][1]):
                # no result for the second phase -> consider only the first set of results
                timeseries['years'] = timeseries['years'][0]
                timeseries['data'] = timeseries['data'][0]
                timeseries['hindcast'] = timeseries['hindcast'][0]

        else:
            timeseries['years'] = np.concatenate(timeseries['years'])
            timeseries['data'] = np.concatenate(timeseries['data'])
            timeseries['hindcast'] = np.concatenate(timeseries['hindcast'])

        df_timeseries = pd.DataFrame(timeseries)
        ts_file = './%s_timeseries.csv' % (filename)
        df_timeseries.to_csv(ts_file)

        # df_reg_stats = pd.DataFrame(reg_stats)
        # reg_file = './%s_reg_stats.csv' % (filename)
        # df_reg_stats.to_csv(reg_file)

        # if not crv_flag:
        #     # save PC
        #     pc1['pc1'] = np.concatenate(pc1['pc1'])
        #     pc_file = './output/%s_pc1SST.csv' % (filename)
        #     df1 = pd.DataFrame(pc1)
        #     df1.to_csv(pc_file)

        LOGGER.info('NIPA run completed')

        response.outputs['forecast_file'].data = ts_file
        response.outputs['scatter_plot'].data = scatter_fp
        response.outputs['sst_map'].data = sst_fp

        return response
