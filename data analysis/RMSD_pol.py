import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='ticks')

# Set font sizes and styles for the plots
plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)
plt.rc('axes', labelsize=29)
plt.rc('legend', fontsize=18)
plt.rc('font', family='sans-serif')

# Data from the image
data = [
    [0.0025395324161043, 0.0028182823882270, 0.0030261970790766, 0.0047580607162448, 0.0055272446188423],
    [0.0025567434889275, 0.0028343544624412, 0.0029958246817982, 0.0048493426208918, 0.0055011542051099],
    [0.0025721952087785, 0.0028009465932603, 0.0029580989063612, 0.0048007659498062, 0.0055113496060526],
    [0.0025566903674252, 0.0027809183230450, 0.0029221749604555, np.nan, 0.0054854506299671],
    [0.0026117338599566, 0.0028014205421479, 0.0029153819528151, np.nan, 0.0055299865400108],
    [np.nan, 0.0027993706449646, 0.0029627094576208, np.nan, 0.0055194916364048],
    [np.nan, 0.0027318951756178, 0.0029009100949181, np.nan, np.nan],
    [np.nan, 0.0027793658731903, np.nan, np.nan, np.nan],
    [np.nan, 0.0027508738274564, np.nan, np.nan, np.nan]
]

# Defining the column names
columns = ['N', 'F', 'E', 'C', 'O']

# Creating the DataFrame
df = pd.DataFrame(data, columns=columns)

# Reference values for each column
reference_values = {
    'N': 5860,
    'F': 7400,
    'E': np.nan,
    'C': 9971,
    'O': np.nan
}

# Calculate mean and standard deviation for each column
mean_values = df.mean()
std_values = df.std()

# Prepare data for fitting with N, F, E
fit_mask_nf = ~np.isnan([reference_values['N'], reference_values['F'], reference_values['E']])
x_fit_nf = np.array([reference_values['N'], reference_values['F'], reference_values['E']])[fit_mask_nf]
y_fit_nf = mean_values[['N', 'F', 'E']].values[fit_mask_nf]

# Perform a linear fit (polynomial of degree 1)
fit_params_nf = np.polyfit(x_fit_nf, y_fit_nf, 1)
fit_line_nf = np.poly1d(fit_params_nf)

# Create x values for the fit line
x_line_nf = np.linspace(min(x_fit_nf), max(x_fit_nf), 100)
y_line_nf = fit_line_nf(x_line_nf)

# Calculate x value for E based on the fit
e_mean_value = mean_values['E']
e_assigned_nf = (e_mean_value - fit_params_nf[1]) / fit_params_nf[0] if not np.isnan(e_mean_value) else np.nan
reference_values['E'] = e_assigned_nf

# Update assigned x values
x_fit_assigned = [reference_values[col] for col in ['N', 'F', 'E']]

# Plotting
plt.figure(figsize=(10, 6))
plt.errorbar(x_fit_assigned, mean_values[['N', 'F', 'E']].values * 1000, 
             yerr=std_values[['N', 'F', 'E']].values * 1000, fmt='o', label='Data with error bars')
plt.plot(x_line_nf, y_line_nf * 1000, label=f'Fit line: y = {fit_params_nf[0]:.4e}x + {fit_params_nf[1]:.4e}', color='red')

# Highlight E point
if not np.isnan(e_assigned_nf):
    plt.scatter(e_assigned_nf, e_mean_value * 1000, color='green', label='E point on fit', zorder=5)

# Labels and legend
plt.xlabel('Polarizability')
plt.ylabel('NRMSD (x$10^{-3}$)')
plt.grid(which='major', color='#EEEEEE', linewidth=1)
plt.grid(which='minor', color='#EEEEEE', linestyle=':')
plt.minorticks_on()
plt.tight_layout()
plt.show()

# Print fit parameters and assigned value for E
print("Fit parameters:", fit_params_nf)
print("Assigned value for E:", e_assigned_nf)

# Update reference values for E and O
reference_values['E'] = e_assigned_nf
reference_values['O'] = o_assigned = np.nan

# Prepare data for fitting with N, F, C, and O
x_assigned = [reference_values[col] for col in df.columns]
fit_mask_all = ~np.isnan(x_assigned)
x_fit_all = np.array(x_assigned)[fit_mask_all]
y_fit_all = mean_values.values[fit_mask_all]

# Perform a linear fit (polynomial of degree 1)
fit_params_all = np.polyfit(x_fit_all, y_fit_all, 1)
fit_line_all = np.poly1d(fit_params_all)

# Create x values for the fit line
x_line_all = np.linspace(min(x_fit_all), max(x_fit_all), 100)
y_line_all = fit_line_all(x_line_all)

# Calculate x values for E and O based on the fit
e_mean_value = mean_values['E']
o_mean_value = mean_values['O']
e_assigned = (e_mean_value - fit_params_all[1]) / fit_params_all[0] if not np.isnan(e_mean_value) else np.nan
o_assigned = (o_mean_value - fit_params_all[1]) / fit_params_all[0] if not np.isnan(o_mean_value) else np.nan

# Update reference values for plotting
reference_values['E'] = e_assigned
reference_values['O'] = o_assigned
x_assigned = [reference_values[col] for col in df.columns]

# Plotting
plt.figure(figsize=(10, 6))
plt.errorbar(x_assigned, mean_values.values, yerr=std_values.values, fmt='o', label='Data with error bars')
plt.plot(x_line_all, y_line_all, label=f'Fit line: y = {fit_params_all[0]:.4e}x + {fit_params_all[1]:.4e}', color='red')

# Highlight E and O points
if not np.isnan(e_assigned):
    plt.scatter(e_assigned, e_mean_value, color='green', label='E point on fit', zorder=5)
if not np.isnan(o_assigned):
    plt.scatter(o_assigned, o_mean_value, color='purple', label='O point on fit', zorder=5)

# Labels and legend
plt.xlabel('Polarizability')
plt.ylabel('NRMSD')
plt.grid(which='major', color='#EEEEEE', linewidth=1)
plt.grid(which='minor', color='#EEEEEE', linestyle=':')
plt.minorticks_on()
plt.tight_layout()
plt.show()

# Print fit parameters
print("Fit parameters:", fit_params_all)
print("Assigned value for E:", e_assigned)
print("Assigned value for O:", o_assigned)
