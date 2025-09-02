import pandas as pd

# === CONFIGURE ===
input_file = "/Users/giuliopalcic/albatross/albatross/data/E-OBS_PO_tmean.txt"
output_file = "/Users/giuliopalcic/albatross/albatross/data/E-OBS_PO_tmean_kelvin.txt"

# === READ FILE LINES MANUALLY ===
with open(input_file, 'r') as f:
    lines = f.readlines()

# === PRESERVE FIRST TWO LINES ===
header_line = lines[0]
metadata_line = lines[1]

# === READ DATA FROM THIRD LINE ONWARDS ===
from io import StringIO
data_string = ''.join(lines[2:])
data_df = pd.read_csv(StringIO(data_string), sep=r'\s+|\t+', engine='python', header=None)

# === CONVERT CELSIUS TO KELVIN ===
data_df = data_df.apply(pd.to_numeric, errors='coerce') + 273.15

# === WRITE TO OUTPUT ===
with open(output_file, 'w') as f:
    f.write(header_line)
    f.write(metadata_line)
    data_df.to_csv(f, sep='\t', index=False, header=False)

print(f"✅ Output saved to: {output_file}")
