#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:55:53 2023

@author: matthewpeters
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from nptdms import TdmsFile
from scipy.optimize import curve_fit
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
from scipy import signal
from matplotlib.ticker import StrMethodFormatter

plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)
plt.rc('axes', labelsize=29)
plt.rc('legend', fontsize=18)
plt.rc('font', family='sans-serif')


def plot_file_data(plot=False):
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
            voltage = data[:, 0]
            time_step = 0.00001
            time = np.arange(0, (len(voltage)-0.5)*time_step, time_step)
        elif ext == '.tdms':
            tdms_file = TdmsFile(file_path)
            properties = tdms_file['Analog'].properties
            scanrate = properties['ScanRate']
            reference = tdms_file['Analog']['AI1']
            voltage = tdms_file['Analog']['AI2']
            time_step = 0.00001
            time = np.arange(0, (len(voltage)-0.5)*time_step, time_step)
        else:
            raise ValueError('File type not supported')

        # Load your original time series data (replace this with your actual data loading code)
        # Assuming your original data is stored in the variable 'original_data'
        # original_data = ...

        # Define the original and target sampling frequencies
        # original_sampling_freq = 100000  # 100 kHz
        # target_sampling_freq = 30  # 30 samples/s

        # Calculate the resampling factor
        # resampling_factor = original_sampling_freq // target_sampling_freq

        # # Use the resample function to downsample the data
        # voltage = signal.resample(voltage, len(voltage) // resampling_factor)
        # time = signal.resample(time, len(time) // resampling_factor)
        # Now 'downsampled_data' contains your downsampled time series data

        # Save or process the downsampled data as needed

        # Store file data in array
        file_data.append({'time': time, 'voltage': voltage,
                         'fs': 100000, 'name': file_names[i]})

        # Apply a 1Hz low-pass filter to the voltage data
        cutoff_frequency = 10  # 1 Hz
        nyquist_frequency = 0.5 * 100000  # Nyquist frequency for your sampling rate
        b, a = signal.butter(1, cutoff_frequency /
                             nyquist_frequency, btype='low')
        filtered_voltage = signal.lfilter(b, a, voltage)

        # ... (rest of the existing code)

        # Plot the data
        if plot:

            plt.figure(figsize=(14, 5))

            plt.plot(time, voltage, linewidth=1,color='magenta', alpha=0.25, label='APD')
            plt.plot(time, filtered_voltage, linewidth=2, color = 'magenta', label='APD (1Hz LPF)')
            # plt.plot(time_2, signal_data, linewidth = 2, color = 'm', label = '1D Video Intensity')
            plt.xticks(np.arange(0, max(time), step=1))
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Time (s)')
            plt.ylabel('APD Signal (V)')
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
            
            xmin = 510
            xmax = xmin + 40
            ymin = 1.5
            ymax = ymin + 0.34
            
            #plt.xlim([xmin, xmax])
            #plt.ylim([ymin, ymax])
            
            plt.tight_layout()            
            

        else:
            pass

    return file_data

def Lorentzian(f, A, fc):
    return A / (f**2 + fc**2)

def compute_PSD(voltage, fs, start, ending, name, run=False):
    if run:

        if not isinstance(voltage, np.ndarray):
            voltage_array = np.array(voltage[0])
        else:
            voltage_array = voltage

        notch_freq = 59.06
        q = 40
        b_notch, a_notch = signal.iirnotch(notch_freq, q, fs)
        voltage_array = signal.filtfilt(b_notch, a_notch, voltage_array)

        notch_freq = 0.1
        q = 10
        b_notch, a_notch = signal.iirnotch(notch_freq, q, fs)
        voltage_array = signal.filtfilt(b_notch, a_notch, voltage_array)
        
        voltage_fft = scipy.fftpack.fft(voltage_array)
        voltage_PSD = np.abs(voltage_fft)**2 / (len(voltage_array)/fs)

        fftfreq = scipy.fftpack.fftfreq(len(voltage_PSD), d=1/fs)

        i = fftfreq > 0

        return fftfreq[i], voltage_PSD[i], name
    else:
        pass

def find_nearest_idx(array, value):
    """
    Returns the index of the nearest value in an array to a given value.
    """
    return np.argmin(np.abs(array - value))

def psd_fitter(psd, minFreq, maxFreq, filename, plot=True, run=False):
    if run:

        freqs = psd[0]
        psd_data = psd[1]
        file_name = psd[2]

        min_freq_idx = find_nearest_idx(freqs, minFreq)
        max_freq_idx = find_nearest_idx(freqs, maxFreq)

        fs = freqs[min_freq_idx:max_freq_idx]  # freqs for fitting x axis
        ys = psd_data[min_freq_idx:max_freq_idx]  # data for fitting y axis

        p0 = (ys[0], 100)  # A, fc
        params, cv = scipy.optimize.curve_fit(
            Lorentzian, fs, ys, p0, sigma=None, absolute_sigma=True)
        A, fc = params
        squaredDiffs = np.square(ys - Lorentzian(fs, A, fc))
        squaredDiffsFromMean = np.square(ys-np.mean(ys))
        rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
        import matplotlib as mpl
        
        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.spines.top'] = False
        if plot:
            plt.figure(figsize=(10,8))
            plt.plot(fs, ys, '.', label="data", color='magenta', alpha = 0.5)
            plt.plot(fs, Lorentzian(fs, A, fc), '--', label="fitted", color='k')
            plt.text(0.975, 0.925, f"fc = {abs(fc):.3f} Hz", fontsize=22, bbox=dict(facecolor='magenta', alpha=0.5), transform=plt.gca().transAxes, horizontalalignment='right')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('P (V$^2$/Hz)')
            plt.xscale('Log')
            plt.yscale('Log')
            plt.tight_layout()
            plt.xlim([4, 50000])
            plt.savefig(filename + 'PSD.pdf')
            # plt.close()
        else:
            pass
        return fc
    else:
        pass

def histogram(trapped, bins='auto', density=True, plot=True, run=False):
    if run:
        
        # Apply a 1Hz low-pass filter to the voltage data
        cutoff_frequency = 10  # 1 Hz
        nyquist_frequency = 0.5 * 100000  # Nyquist frequency for your sampling rate
        b, a = signal.butter(1, cutoff_frequency /
                             nyquist_frequency, btype='low')
        trapped = signal.lfilter(b, a, trapped)
        if plot:
            plt.figure(figsize=(10,8))

            if density:
                # Fit a distribution to the data
                param = stats.norm.fit(trapped)
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 100)
                pdf = stats.norm.pdf(x, *param)
                #plt.plot(x, pdf, color='tab:orange', label='PDF')

            # Use KDE to create a smooth line outlining the histogram
            kde = stats.gaussian_kde(trapped)
            x_kde = np.linspace(min(trapped), max(trapped), 1000)
            y_kde = kde(x_kde)
            plt.plot(x_kde, y_kde, color='magenta', linestyle='-', linewidth=2, label='Smooth Outline')

            # Fill area underneath the smooth line
            plt.fill_between(x_kde, y_kde, alpha=0.3, color='magenta')

            plt.xlabel('Value')
            plt.ylabel('Frequency' if not density else 'Probability Density')
            plt.show()
        else:
            pass

        return
    else:
        return

def plot_on_top(files, plot=False):
    if plot:
        plt.figure()

        for data in files:
            time = data['time']
            voltage = data['voltage']
            name = data['name']
            plt.plot(time, voltage, linewidth=1)

        plt.xticks(np.arange(0, max(time), step=100))
        plt.xlabel('Time (s)', fontsize=20, fontweight='bold')
        plt.ylabel('Voltage (V)', fontsize=20, fontweight='bold')
        plt.xticks(fontsize=16)
        plt.show()

def calculate_rmsd(signal, window_length, file_name, run=False):
    """
    Calculate the RMSD for sections of the input signal based on a single window length
    and print the file name and average RMSD value.

    Args:
        signal (array-like): Input signal.
        window_length (int): Length of the window for splitting the data.
        file_name (str): Name of the file.

    Returns:
        float: Average RMSD value.
    """
    if run:

        num_sections = len(signal) // window_length
        rmsd_values = []
        mean_signal = np.mean(signal)

        for i in range(num_sections):
            section = signal[i * window_length: (i + 1) * window_length]
            squared_diff = np.square(section - np.mean(section))
            mean_squared_diff = np.mean(squared_diff)
            rmsd = np.sqrt(mean_squared_diff)
            rmsd_values.append(rmsd)

        average_rmsd = np.mean(rmsd_values)/mean_signal

        # print("File Name:", file_name)
        # print("Average RMSD:", average_rmsd)

        return average_rmsd
    else:
        pass

def newFile(trapped, write=False):

    if write:
        # reshape data to a single column
        trapped = trapped.reshape(-1, 1)

        # save data to text file
        np.savetxt('CTC013_0802.txt', trapped, fmt='%.8f', delimiter='\n')
    else:
        pass

def acorr(x, lags):
    x_demeaned = x-x.mean()
    corr=np.correlate(x_demeaned,x_demeaned,'full')[len(x)-1:]/(np.var(x)*len(x))
    return corr[:len(lags)]


    # Step 3: Fit to a Two-Exponential Function
def two_exp_function(t, a1, tau1, a2, tau2):
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)
        
t = True
f = False
PSD_data = []
files = []
cornerFreq = []
autocorr_values = []
colors = ['darkviolet', 'tab:orange', 'magenta']
file_data = plot_file_data(plot=f)
if file_data:
    print("Files selected successfully!")
    average_rmsd_list = []
    for data in file_data:
        time = data['time']
        start = time[0]
        ending = time[-1]
        voltage = data['voltage']
        # voltage2=data['voltage2']
        minTrappedTime = find_nearest_idx(time, start)
        maxTrappedTime = find_nearest_idx(time, ending)
        trapped = voltage[minTrappedTime:maxTrappedTime]
        fs = data['fs']
        name = data['name']
        
        print(name)
        
        #psd_run = f
        #rmsd_run = f
        #hist_run = f
        
        #histogram(trapped, run=hist_run)
        #average_rmsd = calculate_rmsd(trapped, 5000, name, run=rmsd_run)


        time_step = 1/100000  # Time step in seconds
        desired_lag_ms = 0.1  # Desired lag in milliseconds
        desired_lag_steps = int(desired_lag_ms / time_step)  # Convert desired lag to time steps
        lags_range = range(desired_lag_steps + 1)
        
        autocorr_values = acorr(trapped, lags_range)
        np.savetxt(str(os.path.splitext(name)[0]) + '_autocorr.txt', autocorr_values, fmt='%.8f', delimiter='\n')
        
        # Call compute_PSD function to compute PSD
        #psd = compute_PSD(trapped, fs, start, ending, name, run=psd_run)
        #PSD_data.append(psd)
        #files.append(name)

        #average_rmsd_list.append(average_rmsd)
    # for autocorr_values, file_name, color in zip(autocorr_values,files, colors ):
    #     plt.plot(lags_range, autocorr_values, label=file_name, color = color)
        
    #newFile(trapped, write=False)

    for array, file_name in zip(PSD_data, files):
        fc = psd_fitter(array, 4, 50000, file_name, run=psd_run)
        cornerFreq.append(fc)

    print(average_rmsd_list)
    print(cornerFreq)


else:
    print("No files selected.")
