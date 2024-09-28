import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from nptdms import TdmsFile
from scipy.optimize import curve_fit
from scipy.fftpack import fft, ifft
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
from scipy import signal
from matplotlib.ticker import StrMethodFormatter
from numpy.polynomial.polynomial import Polynomial
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar
from scipy.signal import wiener
from scipy.signal import fftconvolve
from skimage.restoration import denoise_tv_chambolle
from skimage import img_as_float, data, restoration
from scipy.signal import savgol_filter
sns.set_theme(style='ticks')


plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)
plt.rc('axes', labelsize=29)
plt.rc('legend', fontsize=18)
plt.rc('font', family='sans-serif')
def plot_file_data(plot=True):
    # Create file dialog window
    root = Tk()
    root.withdraw()

    # Prompt user to select files
    file_names = filedialog.askopenfilenames(filetypes=[('CSV files', '*.csv'),
                                                        ('Text files', '*.txt'),
                                                        ('TDMS files', '*.tdms')],
                                             title='Select files')

    # Check if any files were selected
    if not file_names:
        raise ValueError('No files selected')

    # Initialize file data array
    num_files = len(file_names)
    file_data = []

    # Loop through each file
    for i in range(num_files):
        file_path = str(file_names[i])

        # Determine file type based on file extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            data = pd.read_csv(file_path, skiprows=7, header=None).values
            time = data[:, 0]
            voltage = data[:, 1]
        elif ext == '.txt':
            data = pd.read_csv(file_path, skiprows=16,
                               header=None, delim_whitespace=True).values
            
            bin_centers = data[:, 0]

            luc_5 = data[:, 1]
            EL_5 = data[:, 2]

            luc_55 = data[:, 3]
            EL_55 = data[:, 4]

            luc_6 = data[:, 5]
            EL_6 = data[:, 6]

        else:
            raise ValueError('File type not supported')

        # Store file data in array
        file_data.append({'bin_centers': bin_centers, 
                          'luc_5': luc_5,
                          'EL_5': EL_5,
                          'luc_55': luc_55,
                          'EL_55': EL_55,
                          'luc_6': luc_6,
                          'EL_6': EL_6,
                          'fs': 100000, 
                          'name': file_names[i]})
    return file_data

def find_local_minima(fitted_values, bin_centers):
    # Compute the derivative of the fitted function
    derivative = np.gradient(fitted_values, bin_centers)
    
    # Find the indices where the derivative changes sign from negative to positive
    minima_indices = np.where((derivative[:-1] < 0) & (derivative[1:] > 0))[0]
    
    # Get the x-values of the local minima
    local_minima_x = bin_centers[minima_indices]
    # Get the corresponding y-values of the local minima
    local_minima_y = fitted_values[minima_indices]
    
    return local_minima_x, local_minima_y

def find_local_maxima(fitted_values, bin_centers):
    # Compute the derivative of the fitted function
    derivative = np.gradient(fitted_values, bin_centers)
    
    # Find the indices where the derivative changes sign from positive to negative
    maxima_indices = np.where((derivative[:-1] > 0) & (derivative[1:] < 0))[0]
    
    # Get the x-values of the local maxima
    local_maxima_x = bin_centers[maxima_indices]
    # Get the corresponding y-values of the local maxima
    local_maxima_y = fitted_values[maxima_indices]
    
    return local_maxima_x, local_maxima_y


def histogram(bin_centers, luc_5, luc_55, luc_6, EL_5, EL_55, EL_6, name, fs, bins='auto', density=True, plot=True, run=True):
    if run:
        if plot:

            plt.figure(figsize=(8, 8))
            plt.plot(bin_centers, luc_5, color='orangered', linewidth=3, linestyle='-', label = 'SD: 5')
            plt.plot(bin_centers, luc_55, color='red', linewidth=3, linestyle='-', label = 'SD: 5.5')
            plt.plot(bin_centers, luc_6, color='maroon', linewidth=3, linestyle='-', label = 'SD: 6')
            plt.xlabel('Transmission (V)')
            plt.ylabel('Density')
            plt.legend(frameon = False)
            plt.ylim([0, 1.05])
            plt.tight_layout()
            plt.savefig(f'{name}_hist.png')
            plt.savefig(f'{name}_hist.pdf')
            plt.show()

            plt.figure(figsize=(8, 8))
            plt.plot(bin_centers, EL_5, color='orangered', linewidth=3, linestyle='-', label = 'SD: 5')
            plt.plot(bin_centers, EL_55, color='red', linewidth=3, linestyle='-', label = 'SD: 5.5')
            plt.plot(bin_centers, EL_6, color='maroon', linewidth=3, linestyle='-', label = 'SD: 6')
            plt.xlabel('Transmission (V)')
            plt.ylabel('$K_{b}$T')
            plt.legend(frameon = False)
            plt.tight_layout()
            plt.savefig(f'{name}_landscapes.png')
            plt.savefig(f'{name}_landscapes.pdf')
            plt.show()
#############################################################################################
            # Third plot with two y-axes
            fig, ax1 = plt.subplots(figsize=(8, 4))
            # Plot histogram as a line plot
            ax1.plot(bin_centers, luc_5, color='orangered', linewidth=2)
            ax1.set_xlabel('Transmission (V)')
            ax1.set_ylabel('Density', color='red')
            ax1.tick_params(axis='y', labelcolor='red')

            # Create a second y-axis
            ax2 = ax1.twinx()
            ax2.plot(bin_centers, EL_5, color='black', linewidth=3, linestyle='-')
            ax2.set_ylabel('$K_{b}$T', color='black')
            ax2.tick_params(axis='y', labelcolor='black')
            plt.xlim([0.41, 0.57])
            fig.tight_layout()
            plt.savefig(f'{name}_hist_landscapes_SD5.png')
            plt.savefig(f'{name}_hist_landscapes_SD5.pdf')
            plt.show()
#############################################################################################

            # Third plot with two y-axes
            fig, ax1 = plt.subplots(figsize=(8, 4))
            # Plot histogram as a line plot
            ax1.plot(bin_centers, luc_55, color='red', linewidth=2)
            ax1.set_xlabel('Transmission (V)')
            ax1.set_ylabel('Density', color='red')
            ax1.tick_params(axis='y', labelcolor='red')

            # Create a second y-axis
            ax2 = ax1.twinx()
            ax2.plot(bin_centers, EL_55, color='black', linewidth=3, linestyle='-')
            ax2.set_ylabel('$K_{b}$T', color='black')
            ax2.tick_params(axis='y', labelcolor='black')
            plt.xlim([0.42, 0.57])
            fig.tight_layout()
            plt.savefig(f'{name}_hist_landscapes_SD55.png')
            plt.savefig(f'{name}_hist_landscapes_SD55.pdf')
            plt.show()
#############################################################################################

            # Third plot with two y-axes
            fig, ax1 = plt.subplots(figsize=(8, 4))
            # Plot histogram as a line plot
            ax1.plot(bin_centers, luc_6, color='maroon', linewidth=2)
            ax1.set_xlabel('Transmission (V)')
            ax1.set_ylabel('Density', color='red')
            ax1.tick_params(axis='y', labelcolor='red')

            # Create a second y-axis
            ax2 = ax1.twinx()
            ax2.plot(bin_centers, EL_6, color='black', linewidth=3, linestyle='-')
            ax2.set_ylabel('$K_{b}$T', color='black')
            ax2.tick_params(axis='y', labelcolor='black')
            plt.xlim([0.42, 0.57])
            fig.tight_layout()
            plt.savefig(f'{name}_hist_landscapes_SD6.png')
            plt.savefig(f'{name}_hist_landscapes_SD6.pdf')
            plt.show()
#############################################################################################

            Vx_filtered_5 = savgol_filter(EL_5, 3, 1)
            Vx_fitted_5 = np.polyfit(bin_centers, Vx_filtered_5, 200)
            # Generate fitted values for all bin_centers
            Vx_fitted_values_5 = np.polyval(Vx_fitted_5, bin_centers)
            print(Vx_filtered_5)

            # Find local minima
            local_minima_x, local_minima_y = find_local_minima(Vx_fitted_values_5, bin_centers)
            print("Local Minima X, sd 5:", local_minima_x)
            print("Local Minima Y, sd 5:", local_minima_y)

            # Find saddle points
            local_maxima_x, local_maxima_y = find_local_maxima(Vx_fitted_values_5, bin_centers)
            print("Saddle Points X, sd 5:", local_maxima_x)
            print("Saddle Points Y, sd 5:", local_maxima_y)

            # Plot the fitted curve
            plt.figure(figsize=(6, 4))
            plt.plot(bin_centers, EL_5, color = 'maroon', label = 'SD: 5', linewidth = 3)
            #plt.plot(bin_centers, Vx_filtered_5, color = 'blue', linestyle = '-', label = 'Sav-Gol Filter')
            #plt.plot(bin_centers, Vx_fitted_values_5, color = 'red', linestyle = '--', linewidth = 2, label = 'Fit')
            plt.scatter(local_minima_x, local_minima_y, color='red', s = 100)
            plt.scatter(local_maxima_x, local_maxima_y, color='red', marker='d', s = 100)
            plt.xlabel('Transmission (V)')
            plt.ylabel('$K_{b}$T')
            #plt.legend(frameon = False)
            plt.tight_layout()
            plt.show()

#############################################################################################

            Vx_filtered_55 = savgol_filter(EL_55, 3, 1)
            Vx_fitted_55 = np.polyfit(bin_centers, Vx_filtered_55, 250)
            # Generate fitted values for all bin_centers
            Vx_fitted_values_55 = np.polyval(Vx_fitted_55, bin_centers)

            # Find local minima
            local_minima_x, local_minima_y = find_local_minima(Vx_fitted_values_55, bin_centers)
            print("Local Minima X, sd 5.5:", local_minima_x)
            print("Local Minima Y, sd 5.5:", local_minima_y)

            # Find saddle points
            local_maxima_x, local_maxima_y = find_local_maxima(Vx_fitted_values_55, bin_centers)
            print("Saddle Points X, sd 5.5:", local_maxima_x)
            print("Saddle Points Y, sd 5.5:", local_maxima_y)

            # Plot the fitted curve
            plt.figure(figsize=(8, 8))
            plt.plot(bin_centers, EL_55, '#050000', linewidth=3)
            #plt.plot(bin_centers, Vx_filtered_55, color = 'blue', linestyle = '-', label = 'Sav-Gol Filter')
            #plt.plot(bin_centers, Vx_fitted_values_55, color = 'red', linestyle = '--', linewidth = 2, label = 'Fit')
            #plt.scatter(local_minima_x, local_minima_y, color='red', s = 100)
            #plt.scatter(local_maxima_x, local_maxima_y, color='red', marker='d', s = 100)
            plt.xlabel('Transmission (V)')
            plt.ylabel('$K_{b}$T')
            plt.legend(frameon = False)
            plt.xlim([0.42, 0.57])
            plt.ylim([-15, -8])
            plt.tight_layout()
            plt.show()

#############################################################################################

            Vx_filtered_6 = savgol_filter(EL_6, 3, 1)
            Vx_fitted_6 = np.polyfit(bin_centers, Vx_filtered_6, 250)
            # Generate fitted values for all bin_centers
            Vx_fitted_values_6 = np.polyval(Vx_fitted_6, bin_centers)

            # Find local minima
            local_minima_x, local_minima_y = find_local_minima(Vx_fitted_values_6, bin_centers)
            print("Local Minima X, sd 6:", local_minima_x)
            print("Local Minima Y, sd 6:", local_minima_y)

            # Find saddle points
            local_maxima_x, local_maxima_y = find_local_maxima(Vx_fitted_values_6, bin_centers)
            print("Saddle Points X, sd 6:", local_maxima_x)
            print("Saddle Points Y, sd 6:", local_maxima_y)

            # Plot the fitted curve
            plt.figure(figsize=(8, 8))
            plt.plot(bin_centers, EL_6, 'k-', label = 'SD: 6')
            plt.plot(bin_centers, Vx_filtered_6, color = 'blue', linestyle = '-', label = 'Sav-Gol Filter')
            plt.plot(bin_centers, Vx_fitted_values_6, color = 'red', linestyle = '--', linewidth = 2, label = 'Fit')
            plt.scatter(local_minima_x, local_minima_y, color='red', s = 100)
            plt.scatter(local_maxima_x, local_maxima_y, color='red', marker='d', s = 100)
            plt.xlabel('Transmission (V)')
            plt.ylabel('$K_{b}$T')
            plt.legend(frameon = False)
            plt.tight_layout()
            plt.show()            
        else:
            pass

        return
    else:
        return
    


file_data = plot_file_data(plot=True)
if file_data:
    print("Files selected successfully!")
    for data in file_data:
###########################################################################        
        bin_centers = data['bin_centers']
      
        luc_5 = data['luc_5']
        luc_55 = data['luc_55']
        luc_6 = data['luc_6']
        
        EL_5 = data['EL_5']
        EL_55 = data['EL_55']
        EL_6 = data['EL_6']

        fs = data['fs']
        name = data['name']
        name, extension = os.path.splitext(name)

###########################################################################

    histogram(bin_centers, luc_5, luc_55, luc_6, EL_5, EL_55, EL_6, name, fs)

else:
    print("No files selected.")