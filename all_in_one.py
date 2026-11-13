# Import libraries
import PyRayHF
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PyRayHF.library import constants
from PyRayHF.library import find_X, find_Y, find_mu_mup
from PyRayHF.library import build_mup_function
from PyRayHF.library import build_refractive_index_interpolator_spherical
from PyRayHF.library import trace_ray_spherical_gradient
import time

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

# X-mode
mode = "X"
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
# Plot results
fig, ax = plt.subplots(1, 1, figsize=(7, 3))
ax.set_xlim(0, 1500)
ax.set_ylim(0, 500)
ax.set_facecolor("lightgray")
ax.set_ylabel('Altitude (km)')
ax.set_xlabel('Surface Distance (km)')

# --- Electron density background
vmin, vmax = 0, 3e12
color_ticks = np.arange(vmin, vmax + 3e11, 3e11)
pc = ax.pcolormesh(Xg, Zg, Ne_gradient, shading='auto',
                   cmap='plasma', vmin=vmin, vmax=vmax)

# --- Magnetic field visualization (simple line segments with dots)
# Babs is in Tesla (≈3–6e-5), convert to relative visual length
B_scale = 1e6  # empirical scaling factor to make lines a few tens of km long
Bx = Babs * np.sin(np.deg2rad(bpsi)) * B_scale
Bz = -Babs * np.cos(np.deg2rad(bpsi)) * B_scale

# Subsample altitudes for clarity
step_z = max(1, len(alt_km)//5)
z_sample = alt_km[::step_z]
Bx_sample = Bx[::step_z]
Bz_sample = Bz[::step_z]

# Repeat horizontally every 300 km
x_spacing = 300  # km
x_repeat = np.arange(0, ax.get_xlim()[1], x_spacing)

for x_offset in x_repeat:
    for zs, bx, bz in zip(z_sample, Bx_sample, Bz_sample):
        xs, xe = x_offset, x_offset + bx
        ze = zs + bz
        ax.plot([xs, xe], [zs, ze], color='lightblue', lw=0.8)
        ax.plot(xs, zs, 'o', color='lightblue', ms=1.5)

# --- Plot the actual ray paths
line_O, = ax.plot(result_O['x'], result_O['z'], c='white', lw=1.8, label='O-mode')

# --- Add custom magnetic field legend handle
B_ref = 50e-6  # 50 µT reference
line_length_km = B_ref * B_scale  # visible line length
legend_line = Line2D([0, 1], [0, 0], color='lightblue', lw=1.2,
                     marker='o', markersize=3,
                     label=f'Magnetic field\n{line_length_km:.0f} km (50 µT)')

# --- Labels, colorbar, legend
ax.set_title(f"Spherical Gradient (El={elevation_deg:.1f}°, f={f0_Hz/1e6:.1f} MHz)")
plt.colorbar(pc, label='Electron Density (m$^{-3}$)', ax=ax, ticks=color_ticks)

ax.legend(handles=[line_O, legend_line], loc='upper right')

plt.tight_layout()
plt.savefig(local_dir + 'Spherical_Gradient.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Print diagnostics
print('--------------------------------------------------')
print('Spherical Gradient Raytracing O-mode:')
print('--------------------------------------------------')
print('Group Path (km): ', result_O['group_path_km'])
print('Group delay (sec): ', result_O['group_delay_sec'])
print('Ground Range (km): ', result_O['ground_range_km'])
print('x midpoint (km): ', result_O['x_midpoint'])
print('z midpoint (km): ', result_O['z_midpoint'])
print('x apex (km): ', result_O['x_apex_km'])
print('z apex (km): ', result_O['z_apex_km'])

print('--------------------------------------------------')
print('Elapsed time (sec): ', elapsed_time)
print('--------------------------------------------------')