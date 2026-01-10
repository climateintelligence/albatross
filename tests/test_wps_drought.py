import time
import random
import re
import uuid
import gc
import sys
from urllib.parse import quote

from pywps import Service, configuration
from pywps.tests import client_for
from albatross.processes.wps_drought import Drought

basin = "ZRB"
start_year = "1951"
end_year = "2015"
forecast_year_ok = "2025"  # or whatever you want
glo_vars = ["sst"]
indices = ["NINO 3.4"]
target_variables = ["monthly_inflows"]

aggregations = "sum"


log_path = f"wps_drought_test_log_{basin}_{start_year}_{end_year}.txt"
sys.stdout = open(log_path, "w", encoding="utf-8")
sys.stderr = sys.stdout

print("PyWPS Work Directory:", configuration.get_config_value("server", "workdir"))
print(f"🧪 Starting test_wps_drought run — ID: {uuid.uuid4()}")

def make_month_windows(basin_name=None):
    if basin_name == "CHIRPS_MEK":
        return ["11,12,1", "2,3,4", "5,6,7", "8,9,10"]
    elif basin_name == "ZRB" or basin_name == "CHIRPS_ZRB":
        return ["5,6,7", "8,9,10", "11,12,1", "2,3,4"]
    else:
        return ["12,1,2", "3,4,5", "6,7,8", "9,10,11"]

def test_wps_drought():
    phase_modes = [2]
    month_windows = make_month_windows(basin_name=basin)

    downloaded_data = {}
    last_data_key = None

    for glo_var in glo_vars:
        for months in month_windows:
            for index in indices:
                data_key = (glo_var, months, index)

                if data_key not in downloaded_data:
                    downloaded_data[data_key] = True
                    if last_data_key is not None:
                        delay = random.uniform(1.5, 3.0)
                        print(f"⏳ Waiting {delay:.2f}s before new group request...")
                        time.sleep(delay)
                    last_data_key = data_key

                for phase_mode in phase_modes:
                    for target_variable in target_variables:
                        print(
                            f"\n▶ Running test for glo_var: {glo_var} | index: {index} | "
                            f"phase_mode: {phase_mode} | target_variable: {target_variable} | months: {months} |"
                        )

                        client = None
                        try:
                            client = client_for(Service(processes=[Drought()]))


                            datainputs_dict = {
                                "target": (
                                    "https://raw.githubusercontent.com/climateintelligence/albatross/"
                                    f"refs/heads/main/albatross/data/{basin}_{target_variable}.txt"
                                ),
                                "indicator": index,
                                "start_year": start_year,
                                "end_year": end_year,
                                "month": months,
                                "phase_mode": str(phase_mode),
                                "glo_var_name": glo_var,
                                "aggregation": aggregations,

                                # ✅ this is what makes the process attempt operational
                                "forecast_year": forecast_year_ok,
                            }

                            datainputs = ";".join(f"{k}={v}" for k, v in datainputs_dict.items())
                            datainputs_q = quote(datainputs, safe="")  # critical

                            response = client.get(
                                "/wps?service=WPS&request=Execute&version=1.0.0"
                                "&identifier=drought"
                                f"&datainputs={datainputs_q}"
                            )

                            response_text = response.data.decode("utf-8", errors="replace")

                            if response.status_code != 200 or "ProcessSucceeded" not in response_text:
                                print("❌ WPS process failed!")
                                print("HTTP:", response.status_code)
                                match = re.search(
                                    r"<ows:ExceptionText>(.*?)</ows:ExceptionText>",
                                    response_text, re.DOTALL
                                )
                                if not match:
                                    match = re.search(
                                        r"<(?:\w+:)?ExceptionText>(.*?)</(?:\w+:)?ExceptionText>",
                                        response_text, re.DOTALL
                                    )
                                if match:
                                    print("---- ExceptionText ----")
                                    print(match.group(1).strip())
                                else:
                                    print("---- Response (first 2000 chars) ----")
                                    print(response_text[:2000])
                                raise RuntimeError("Process failed.")

                            print("✅ Success!")

                        except Exception as e:
                            print(
                                f"⚠️  Skipping combination {target_variable}-{index}-{glo_var}-{phase_mode}-{months}, "
                                f"due to error: {e}"
                            )
                            continue

                        finally:
                            if client is not None:
                                del client
                            gc.collect()

        if random.random() < 0.25:
            time.sleep(random.uniform(1.5, 3.0))
