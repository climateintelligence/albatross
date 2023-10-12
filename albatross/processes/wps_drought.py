from pywps import Process, LiteralInput, LiteralOutput, UOM
from pywps.app.Common import Metadata

import logging
LOGGER = logging.getLogger("PYWPS")


class Drought(Process):
    """A process to forecast precipitation."""
    def __init__(self):
        inputs = [
            LiteralInput('pr', 'Monthly global precipitation file',
                         abstract='??? netcdf file',
                         # keywords=['name', 'firstname'],
                         data_type='string'),
            LiteralInput('sst', 'Monthly global sea surface temperature file',
                         abstract='??? netcdf file',
                         # keywords=['name', 'firstname'],
                         data_type='string'),
            LiteralInput('gph', 'Monthly global geopotential height file',
                         abstract='??? netcdf file',
                         # keywords=['name', 'firstname'],
                         data_type='string'),
            LiteralInput('slp', 'Monthly global sea level pressure file',
                         abstract='??? netcdf file',
                         # keywords=['name', 'firstname'],
                         data_type='string'),
            LiteralInput('indicator', 'Indicator ???',
                         abstract='examples: climate indicator of tele-connection patterns, ???',
                         # keywords=['name', 'firstname'],
                         data_type='string'),
            LiteralInput('start_year', 'Start Year',
                         abstract='2020',
                         # keywords=['name', 'firstname'],
                         data_type='string'),
            LiteralInput('end_year', 'End Year',
                         abstract='2022',
                         # keywords=['name', 'firstname'],
                         data_type='string'),
            LiteralInput('bbox', 'Bounding Box',
                         abstract='bbox or other geometries',
                         # keywords=['name', 'firstname'],
                         data_type='string')
        ]
        outputs = [
            LiteralOutput('negative_list', 'Negative List ...',
                          #abstract='negativ list',
                          # keywords=['output', 'result', 'response'],
                          data_type='string'),
            LiteralOutput('positive_list', 'Postive List ...',
                          #abstract='negativ list',
                          # keywords=['output', 'result', 'response'],
                          data_type='string'),
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

    @staticmethod
    def _handler(request, response):
        LOGGER.info("processing")
        response.outputs['negative_list'].data = 'PR ' + request.inputs['pr'][0].data
        response.outputs['positive_list'].data = 'sst ' + request.inputs['sst'][0].data
        return response
