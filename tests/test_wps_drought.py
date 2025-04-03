from pywps import Service
from pywps.tests import client_for
from albatross.processes.wps_drought import Drought  # Adjust path if necessary
from pywps import configuration
from urllib.parse import urlencode
from pathlib import Path


# Print PyWPS workdir
print("PyWPS Work Directory:", configuration.get_config_value("server", "workdir"))

def test_wps_drought():
    """Test the WPS Drought process"""

    # Initialize a WPS client with the Drought process
    client = client_for(Service(processes=[Drought()]))
    # Path to the local .txt file for the test
    local_txt_file = Path("/Users/giuliopalcic/Downloads/APGD_prcpComo.txt")

    # Open the file to pass as input (simulate the upload)
    with open(local_txt_file, 'rb') as f:  # 'rb' to open in binary mode
        file_content = f.read()


    # Correct the input parameters and encode URLs properly
    datainputs_dict = {
        "pr": file_content,
        "indicator": "enso",
        "start_year": "1971",
        "end_year": "2018",
        "month": "1,2,3",  # Ensure the correct parameter name
        "phase_mode": "2",
    }

    datainputs = urlencode(datainputs_dict, safe=":/")  # Encode properly

    # Correctly format the request
    response = client.get(
        f"/wps?service=WPS&request=Execute&version=1.0.0&identifier=drought&datainputs={datainputs}"
    )

    # Convert response.data from bytes to string
    response_text = response.data.decode("utf-8")

    # 🔹 Debug: Print the full response to see what's happening
    print("WPS Response:", response_text)

    # Assert that the process returns a successful response
    assert response.status_code == 200, f"Process failed with status {response.status_code}"
    assert "ProcessSucceeded" in response_text, f"WPS process did not succeed. Response: {response_text}"
