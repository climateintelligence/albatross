"""Offline smoke checks for the service and scientific dependencies."""

from types import SimpleNamespace

from click.testing import CliRunner
from lxml import etree
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pywps import Service
from pywps.tests import client_for

from albatross.cli import cli
from albatross.processes.wps_drought import Drought
from albatross.utils import glo_var_Map_unfiltered


def test_service_metadata():
    client = client_for(Service(processes=[Drought()]))
    for request, expected in (
        ('GetCapabilities', 'Capabilities'),
        ('DescribeProcess&identifier=drought', 'ProcessDescriptions'),
    ):
        response = client.get(
            '/wps?service=WPS&version=1.0.0&request=' + request
        )
        assert response.status_code == 200
        root = etree.fromstring(response.data)
        assert etree.QName(root).localname == expected
        assert root.xpath('//*[local-name()="Identifier" and text()="drought"]')


def test_cli_help():
    result = CliRunner().invoke(cli, ['--help'])
    assert result.exit_code == 0, result.output
    assert 'start' in result.output


def test_basemap_plot(tmp_path):
    phase = SimpleNamespace(
        glo_var=SimpleNamespace(
            lon=np.linspace(-180, 180, 8), lat=np.linspace(-80, 80, 6)
        ),
        corr_grid_full=np.linspace(-1, 1, 48).reshape(6, 8),
    )
    figure = glo_var_Map_unfiltered(phase)
    try:
        output = tmp_path / 'map.png'
        figure.savefig(output)
        assert output.stat().st_size > 0
    finally:
        plt.close(figure)
