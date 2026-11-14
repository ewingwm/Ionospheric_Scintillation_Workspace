# Import libraries
import PyRayHF
import numpy as np
from PyRayHF.library import constants
from PyRayHF.library import find_X, find_Y, find_mu_mup
from PyRayHF.library import build_mup_function
from PyRayHF.library import build_refractive_index_interpolator_spherical
from PyRayHF.library import trace_ray_spherical_gradient
import time
from scipy.constants import mu_0, epsilon_0

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

# Define inputs
elevation_deg = 25.
f0_Hz = 14e6

delta_z = 5
# Create an array of altitudes (km)
aalt = np.arange(80., 700., 1.)

# Save to file
save_path = './Input_Day.p'

# Generate 1D Data
input_example = PyRayHF.library.generate_input_1D(year, month, day, UT, lat, lon, aalt, F107)


# Generate array of frequencies used by a vertical ionosonde in MHz.
f_min = 0.1
f_max = 17.5
df = 0.1
ionosonde_frequency = np.arange(f_min, f_max, df)

# Directory on your local machine where to save the figures
local_dir = '/home/wewing/AP_Research_GRC/python_workspace/figures'

# Load an ionospheric profile
# See Example_Generate_Input_Arrays for how to create this input file using
# PyIRI.
input_arrays = input_example
alt_km = input_arrays["alt"]
Ne = input_arrays["den"]
Babs = input_arrays["bmag"]
bpsi = input_arrays["bpsi"]

# Ensure ground level is included
h_ground = 0
if alt_km[0] > h_ground:
    Ne0 = np.interp(h_ground, alt_km, Ne)
    Babs0 = np.interp(h_ground, alt_km, Babs)
    bpsi0 = np.interp(h_ground, alt_km, bpsi)
    alt_km = np.insert(alt_km, 0, h_ground)
    Ne = np.insert(Ne, 0, Ne0)
    Babs = np.insert(Babs, 0, Babs0)
    bpsi = np.insert(bpsi, 0, bpsi0)



# Create 2D density grid (Ne assumed horizontally uniform)
nx = 1500
xmax = 1500
x_grid = np.linspace(0, xmax, nx)
z_grid = alt_km
Xg, Zg = np.meshgrid(x_grid, z_grid)
Ne_grid = np.tile(Ne[:, np.newaxis], (1, nx))
Babs_grid = np.tile(Babs[:, np.newaxis], (1, nx))
bpsi_grid = np.tile(bpsi[:, np.newaxis], (1, nx))

# Add some gradient to it
Ne_gradient = Ne_grid * (1.0 - 0.5 * (Xg / Xg.max()))

# Start the timer to time the code
start_time = time.perf_counter()

# Calculate plasma parameters
X = find_X(Ne_gradient, f0_Hz)
Y = find_Y(f0_Hz, Babs_grid)

# O-mode
mode = "O"
mu, mup = find_mu_mup(X, Y, bpsi_grid, mode)
mup = np.where((mup < 1e-3) | np.isnan(mup), np.nan, mup)

# Build a function for spherical interpolation of mup
mup_func_spherical = build_mup_function(mup_field=mup,
                                        x_grid=x_grid,
                                        z_grid=z_grid,
                                        geometry="spherical")

# Build interpolator from spherical μ grid
n_and_grad_spherical = build_refractive_index_interpolator_spherical(z_grid,
                                                                     x_grid,
                                                                     mu)

# Raytrace
result_O = trace_ray_spherical_gradient(n_and_grad_rphi=n_and_grad_spherical,
                                        x0_km=0.0,
                                        z0_km=0.0,
                                        elevation_deg=elevation_deg,
                                        s_max_km=4000.,
                                        mup_func=mup_func_spherical)

# Stop the timer
end_time = time.perf_counter()
# Calculate and print the elapsed time
elapsed_time = end_time - start_time


x_values = result_O["x"]
z_values = result_O["z"]

last_x = None
last_z = None
dists = []
for x, z in zip(x_values, z_values):
    if not(last_x or last_z):
        last_x = x
        last_z = z
        dists.append(0)
    else:
        dists.append(np.sqrt(pow(x-last_x, 2) + pow(z-last_z, 2)))
        last_x = x
        last_z = z
total_dist = []
# Fix the thing that filters out extra values.
running_value = 0
for d in dists:
    total_dist.append(d + running_value)
    running_value += d
idx = -1
for x in x_values:
  idx += 1
  if x == result_O["ground_range_km"]:
        break
total_dist = total_dist[:idx]
x_values = x_values[:idx]
z_values = z_values[:idx]

indexes = []
index = -1
current_v = 0
last_v = None
for d in total_dist:
  index += 1
  if d == current_v:
    current_v += delta_z
    indexes.append(index)
  elif last_v < current_v and d > current_v:
    current_v += delta_z
    indexes.append(index)
  last_v = d
index_zs = []
index_ds = []
index_xs = []
for i in indexes:
   index_zs.append(z_values[i])
   index_xs.append(x_values[i])
   index_ds.append(total_dist[i])

# Create Gausian Noise in the Complex Domain (Will be done in the model code as an input rather than in the python)
rng = np.random.default_rng(seed=42)
N = 1000
total_std_dev = 0.5  # Desired total standard deviation (sqrt of total noise power)

# Split power equally between real and imaginary components
component_std = total_std_dev / np.sqrt(2)

real_part = rng.normal(loc=0.0, scale=component_std, size=N)
imag_part = rng.normal(loc=0.0, scale=component_std, size=N)

# Combine into a complex128 array
complex_noise = real_part + 1j * imag_part

# TODO Real Epsilon Values
omega = 2 * np.pi * f0_Hz
epsilon_r_v = 123
k = np.sqrt( np.pow(omega, 2) * mu_0 * epsilon_0 * epsilon_r_v)
epsilon_1 = 123 # test value for now, defined later
complex_noise_through_phase = complex_noise * np.exp(1j * k * delta_z * epsilon_1 / 2)

