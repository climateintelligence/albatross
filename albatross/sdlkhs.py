from pydap.client import open_url

# Open the dataset (local or remote)
dataset = open_url("http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCDC/.ERSST/.version3b/.anom/T/dods")

# Select the anomaly dataset
anom_data = dataset["anom"]
time_data = dataset["T"]  # Extract time variable if needed

# Print dataset structure
print("Available variables:", dataset.keys())
print("Anomaly Data Shape:", anom_data.shape)

# Access specific slices of data (e.g., first time step)
sst_values = anom_data.array[:, :, :, :].data.squeeze()  # Extract and squeeze dimensions
print("Extracted SST Data:", sst_values)
