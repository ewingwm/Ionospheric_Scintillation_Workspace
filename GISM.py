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
import scipy
from scipy.special import gamma
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

# Phase Screen 1D for now, 2d if time
q_0 = np.pi # TODO real value (2 < pi/L0 and L0 is the outer scale of the  inhomogeneities.) Set in the paper #param
p = 3 # TODO real value (p is the spectrum slope. The spectrum is consequently  linear using Log-Log scales.) Value is set as a defualt in the paper #param
sigma_phi = 5 # TODO real value (phase variance that comes from spectrum slope)
def f(lag):
  return (pow(sigma_phi, 2) / ( pow(2, (p-3)/2) * gamma((p-1)/2) ) ) * pow((lag * q_0), ((p-1)/2)) * scipy.special.kv(((p-1)/2),(lag*q_0)) 

M = 2048
max_lag = 10.0
lag = np.linspace(0, max_lag, M)

B_phi = np.asarry(f(lag), dtype=float)

if np.isnan(B_phi[0]):
  B_phi[0] = f(1e-12)

acf_symmetric = np.concatenate([B_phi, B_phi[-2:0:-1]])
N = len(acf_symmetric)

psd = np.maximum(np.real(np.fft.fft(acf_symmetric)), 0)

rng = np.random.default_rng(seed=42)
white_noise_fft = rng.standard_normal(N) + 1j * rng.standard_normal(N)

random_spectrum = np.sqrt(psd / 2.0) * white_noise_fft

synthetic_data = np.real(np.fft.ifft(random_spectrum)) * np.sqrt(N)
