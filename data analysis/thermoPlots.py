import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter
from scipy.optimize import curve_fit

sns.set_theme(style='ticks')

# Set font sizes and styles for the plots
plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)
plt.rc('axes', labelsize=29)
plt.rc('legend', fontsize=18)
plt.rc('font', family='sans-serif')

# Temperature data
temp_unlabelled = [37, 38.6, 39, 40.2, 41, 41.8, 42.3, 45]
temp_labelled = [37, 38, 38.2, 39.5, 40.6, 40.8, 41.3, 42.2, 43, 43.4, 44.2, 45, 46, 48, 50.4, 54.7]

# Data for KNF
knf_unlabelled = {
    "AVG": [3.969276274, 3.706533228, 4.597015272, None, 53.95425457, None, None, None],
    "STD": [0.105191421, 0.10946606, 0.773877719, None, 3.604489143, None, None, None]
}

knf_labelled = {
    "AVG": [0.863138633, 4.17336249, 8.968472166, 13.10204942, 5.708565741, 44.28256106, 95.07726556, 107.3584791, 76.96262396, 305.6681389, None, None, None],
    "STD": [0.033202009, 0.895454345, 2.413846013, 4.217293361, 1.261895169, 8.197278022, 5.228684234, 55.02635904, 4.405675717, 0, None, None, None]
}

# Data for KFE
kfe_unlabelled = {
    "AVG": [0.28520844, 0.10984183, 0.03231471, 0.229710702, 1.143711577, 1.31452048, None, 0.0614504],
    "STD": [0.020811344, 0.029544346, 0.002251355, 0.16278427, 0.019447652, 0.02180789, None, 0.004331]
}

kfe_labelled = {
    "AVG": [0.02653823, 0.009112358, None, 0.03979828, 0.717902708, None, 0.714558539, 1.191821308, None,  1.16626528, 0.275743213, 0.101064265, 0.064162631, None],
    "STD": [0.001774088, 0.001681442, None, 0.003662732, 0.252846787, None, 0.029653224, 0.11606009,None,  0.084773417, 0.049979795, 0.014934442, 0.013373449, None]
}

def filter_none(data, temps):
    filtered_data = [(temp, avg, std) for temp, avg, std in zip(temps, data["AVG"], data["STD"]) if avg is not None and std is not None]
    temps, avgs, stds = zip(*filtered_data)
    return list(temps), list(avgs), list(stds)

# Exponential function to fit
def exp_func(x, a, b):
    return a * np.exp(b * x)

# Filter out None values
temp_unlabelled_knf, knf_unlabelled_avg, knf_unlabelled_std = filter_none(knf_unlabelled, temp_unlabelled)
temp_labelled_knf, knf_labelled_avg, knf_labelled_std = filter_none(knf_labelled, temp_labelled)
temp_unlabelled_kfe, kfe_unlabelled_avg, kfe_unlabelled_std = filter_none(kfe_unlabelled, temp_unlabelled)
temp_labelled_kfe, kfe_labelled_avg, kfe_labelled_std = filter_none(kfe_labelled, temp_labelled)

# Fit exponential function to KNF data
initial_guess = (1, 0.01)
params_unlabelled, _ = curve_fit(exp_func, temp_unlabelled_knf, knf_unlabelled_avg, p0=initial_guess, maxfev=2000)
params_labelled, _ = curve_fit(exp_func, temp_labelled_knf, knf_labelled_avg, p0=initial_guess, maxfev=2000)

# Generate x values for plotting the fitted curves
x_fit = np.linspace(min(temp_unlabelled + temp_labelled), max(temp_unlabelled + temp_labelled), 100)
y_fit_unlabelled = exp_func(x_fit, *params_unlabelled)
y_fit_labelled = exp_func(x_fit, *params_labelled)

# Plotting KNF with fitted exponential
plt.figure(figsize=(8, 6))
plt.errorbar(temp_labelled_knf, knf_labelled_avg, yerr=knf_labelled_std, fmt='d', label='BSA+FITC', color='#008080', capsize=7.5, markersize=10, elinewidth=2, capthick=2)
plt.plot(x_fit, y_fit_labelled, '--', color='#008080', linewidth = 5)
plt.errorbar(temp_unlabelled_knf, knf_unlabelled_avg, yerr=knf_unlabelled_std, fmt='o', label='BSA', color='dodgerblue', capsize=7.5, markersize=10, elinewidth=2, capthick=2)
plt.plot(x_fit, y_fit_unlabelled, '--', color='dodgerblue', linewidth =5)
plt.xlabel('Temperature (K)')
plt.ylabel('$K_{NF}$')
plt.legend(frameon=True, framealpha=0.5, facecolor='gray')
plt.grid(which='major', color='#EEEEEE', linewidth=1)
plt.grid(which='minor', color='#EEEEEE', linestyle=':')
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
plt.xlim([36.5, 43.5])
plt.ylim([0, 350])
plt.tight_layout()
# Shift x-tick labels by adding 273.15
ax = plt.gca()
xticks = ax.get_xticks()
ax.set_xticklabels([f'{x + 273.15:.0f}' for x in xticks])

plt.show()

# Plotting KFE with lines connecting the dots
plt.figure(figsize=(10, 8))

# Plot labeled data with lines and error bars
plt.errorbar(temp_labelled_kfe, kfe_labelled_avg, yerr=kfe_labelled_std, fmt='d', label='BSA+FITC', color='k', capsize=7.5, markersize=7.5, elinewidth=2, capthick=2)
#plt.plot(temp_labelled_kfe, kfe_labelled_avg, linestyle='-', color='k', linewidth =2)  # Connect with lines

# Plot unlabeled data with lines and error bars
plt.errorbar(temp_unlabelled_kfe, kfe_unlabelled_avg, yerr=kfe_unlabelled_std, fmt='o', label='BSA', color='red', capsize=7.5, markersize=7.5, elinewidth=2, capthick=2)
#plt.plot(temp_unlabelled_kfe, kfe_unlabelled_avg, linestyle='-', color='red', linewidth = 2)  # Connect with lines

plt.xlabel('Temperature ($^\circ$C)')
plt.ylabel('$K_{FE}$')
plt.legend(frameon=False)
plt.ylim([0, 1.4])
plt.grid(which='major', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
plt.tight_layout()
plt.show()
