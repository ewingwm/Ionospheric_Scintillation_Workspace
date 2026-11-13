# Import libraries
import PyRayHF
import matplotlib.pyplot as plt
import numpy as np


# Select the day and Universal Time of interest
year = 2026
month = 11
day = 12
UT = 21

# F10.7 index for the selected day
F107 = 132.

# Define geographic location (longitude, latitude in degrees)
lon = -82.561821
lat = 36.548435

# Create an array of altitudes (km)
aalt = np.arange(80., 700., 1.)

# Generate 1D Data
input_example = PyRayHF.library.generate_input_1D(year, month, day, UT, lat, lon, aalt, F107)


# Generate array of frequencies used by a vertical ionosonde in MHz.
f_min = 0.1
f_max = 17.5
df = 0.1
ionosonde_frequency = np.arange(f_min, f_max, df)

