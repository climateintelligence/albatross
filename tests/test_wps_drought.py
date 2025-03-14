from pywps import Service
from pywps.tests import client_for
from albatross.processes.wps_drought import Drought  # Adjust path if necessary


def test_wps_drought():
    """Test the WPS Drought process"""

    # Initialize a WPS client with the Drought process
    client = client_for(Service(processes=[Drought()]))

    # Define test input parameters in correct WPS format
    datainputs = "pr=https://raw.githubusercontent.com/mxgiuliani00/CSI/master/NIPA/DATA/APGD_prcpComo.txt;" \
                 "indicator=https://raw.githubusercontent.com/mxgiuliani00/CSI/master/NIPA/DATA/nao.txt;" \
                 "start_year=2020;end_year=2022;month=8"

    # Correctly format the request as a URL with query parameters
    response = client.get(
        "/wps?service=WPS&request=Execute&version=1.0.0&identifier=drought&datainputs={}".format(datainputs)
    )

    # Convert response.data from bytes to string
    response_text = response.data.decode("utf-8")

    # Debug: Print response in case of failure
    print(response_text)

    # Assert that the process returns a successful response
    assert response.status_code == 200, f"Process failed with status {response.status_code}"
    assert "ProcessSucceeded" in response_text, "WPS process did not succeed"
