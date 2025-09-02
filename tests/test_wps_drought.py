import time
import random
import re
import uuid
from pywps import Service, configuration
from pywps.tests import client_for
from albatross.processes.wps_drought import Drought  # Adjust if needed
import gc
import sys
import os

basin = "LakeComo"
log_path = f"wps_drought_test_log_{basin}_1984_2023.txt"
sys.stdout = open(log_path, "w", encoding="utf-8")
sys.stderr = sys.stdout  # optional: include errors in the same file

print("PyWPS Work Directory:", configuration.get_config_value("server", "workdir"))
print(f"🧪 Starting test_wps_drought run — ID: {uuid.uuid4()}")

def make_month_windows(window_size=3, start_month=1):
    windows = []
    for start in range(start_month, 13):
        window = [(start + i - 1) % 12 or 12 for i in range(window_size)]
        windows.append(",".join(map(str, window)))
    return windows


"""def make_month_windows(basin_name=None):
    if basin_name == "CHIRPS_MEK":
        return [
            "11,12,1",
            "2,3,4", # dry
            "5,6,7",
            "8,9,10"  # wet
        ]
    elif basin_name == "ZRB" or basin_name == "CHIRPS_ZRB":
        return [
            "5,6,7",
            "8,9,10", # dry
            "11,12,1",
            "2,3,4"  # wet
        ]
    else:
        return [
            "12,1,2",  # Inverno (DJF)
            "3,4,5",    # Primavera (MAM)
            "6,7,8",    # Estate (JJA)
            "9,10,11"  # Autunno (SON)

    ]"""

def test_wps_drought():

    # Configuration
    window_size = 4
    phase_modes = [2,3,4] #4] #1,2,3,4
    month_windows = make_month_windows(window_size=window_size, start_month=1)
    # month_windows = make_month_windows(basin_name=basin)
    glo_vars = ['sst','slp']
    indices = ['NAO', 'SCA', 'EAWR', 'PDO', 'AMO','AO','ONI']
    target_variables = ['monthly_inflows']# ['monthly_inflows']#['tmax','tmin','tmean']#,'ppt']#['tmax', 'tmin', 'tmean', 'ppt']

    # Download cache to avoid redundant fetches
    downloaded_data = {}
    last_data_key = None

    for glo_var in glo_vars:
        for months in month_windows:
            for index in indices:
                data_key = (glo_var, months, index)

                # Simulate/track preload
                if data_key not in downloaded_data:
                    downloaded_data[data_key] = True

                    # Add delay only when moving to a new dataset
                    if last_data_key is not None:
                        delay = random.uniform(1.5, 3.0)
                        print(f"⏳ Waiting {delay:.2f}s before new group request...")
                        time.sleep(delay)

                    last_data_key = data_key

                # Loop over remaining parameters
                for phase_mode in phase_modes:
                    for target_variable in target_variables:
                        print(f"\n▶ Running test for glo_var: {glo_var} | index: {index} | "
                              f"phase_mode: {phase_mode} | target_variable: {target_variable} | months: {months} |")

                        try:
                            client = client_for(Service(processes=[ Drought() ]))

                            datainputs_dict = {
                                "target": f"https://raw.githubusercontent.com/climateintelligence/albatross/refs/heads/main/albatross/data/{basin}_{target_variable}.txt",
                                "indicator": index,
                                "start_year": "1984",
                                "end_year": "2022",
                                "month": months,
                                "phase_mode": str(phase_mode),
                                "glo_var_name": glo_var,
                            }

                            datainputs = ";".join(f"{k}={v}" for k, v in datainputs_dict.items())

                            response = client.get(
                                f"/wps?service=WPS&request=Execute&version=1.0.0&identifier=drought&datainputs={datainputs}"
                            )

                            response_text = response.data.decode("utf-8")

                            if response.status_code!=200 or "ProcessSucceeded" not in response_text:
                                print("❌ WPS process failed!")
                                match = re.search(r"<ows:ExceptionText>(.*?)</ows:ExceptionText>", response_text,
                                                  re.DOTALL)
                                if match:
                                    print("---- WPS ExceptionText ----")
                                    print(match.group(1).strip())
                                raise RuntimeError("Process failed.")

                            print("✅ Success!")

                        except Exception as e:
                            print(f"⚠️  Skipping combination {target_variable}-{index}-{glo_var}-{phase_mode}-{months}, due to error: {e}")
                            continue  # Proceed to next configuration

                        finally:
                            del client
                            gc.collect()

        # Pause a little after each glo_var
        if random.random() < 0.25:  # 25% dei casi
            time.sleep(random.uniform(1.5, 3.0))
