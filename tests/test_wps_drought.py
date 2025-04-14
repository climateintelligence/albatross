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

    # Correct the input parameters and encode URLs properly
    datainputs_dict = {
        "pr": "https://raw.githubusercontent.com/climateintelligence/albatross/refs/heads/new_NIPA/albatross/data/E-OBS_precipitation_Como.txt",
        "indicator": "nao",
        "start_year": "1952",
        "end_year": "2023",
        "month": "1,2,3",  # Ensure the correct parameter name
        "phase_mode": "2",
    }

    datainputs = ";".join(f"{k}={v}" for k, v in datainputs_dict.items())

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
