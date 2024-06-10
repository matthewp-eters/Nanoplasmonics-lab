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
from scipy.fftpack import fft, ifft
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
from scipy import signal
from matplotlib.ticker import StrMethodFormatter

sns.set_theme(style='ticks')


plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)
plt.rc('axes', labelsize=29)
plt.rc('legend', fontsize=18)
plt.rc('font', family='sans-serif')



def lowpass(voltage, cutoff_frequency, sampling):
    
    nyquist_frequency = 0.5*sampling
    b, a = signal.butter(1, cutoff_frequency /
                             nyquist_frequency, btype='low')
    filtered_voltage = signal.lfilter(b, a, voltage)
    return filtered_voltage

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
            #data = pd.read_csv(file_path, skiprows=16,
            #                   header=None, delim_whitespace=True).values
            #voltage = data[:, 0]
            #time_step = 0.00001
            #time = np.arange(0, (len(voltage)-0.5)*time_step, time_step)

            data = pd.read_csv(file_path, skiprows=1,
                               header=None, delim_whitespace=True).values
            voltage = data[:, 1]
            time_step = 0.000001
            time = data[:,0]

        else:
            raise ValueError('File type not supported')

        # Store file data in array
        file_data.append({'time': time, 'voltage': voltage,
                         'fs': 1000000, 'name': file_names[i]})

        fs = 1000000
        # Apply a 10 Hz low-pass filter to the voltage data
        filtered_voltage = lowpass(voltage, 1000, fs)

        # Plot the data
        if plot:

            plt.figure(figsize=(14, 5))

            plt.plot(time, voltage, linewidth=1,color='red', alpha=0.25, label='APD')
            plt.plot(time, filtered_voltage, linewidth=2, color = 'red', label='APD (LPF)')
            plt.xticks(np.arange(0, max(time), step=1))
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Time (s)')
            plt.ylabel('APD Signal (V)')
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
            
            #plt.xlim([xmin, xmax])
            #plt.ylim([ymin, ymax])
            
            plt.tight_layout()            
            

        else:
            pass

    return file_data

def Lorentzian(f, A, fc):
    return A / (f**2 + fc**2)

def notch(voltage_array, notch_freq, q, fs):
    b_notch, a_notch = signal.iirnotch(notch_freq, q, fs)
    voltage_array = signal.filtfilt(b_notch, a_notch, voltage_array)
    return voltage_array

def compute_PSD(voltage, fs, name, run=False):
    if run:

        if not isinstance(voltage, np.ndarray):
            voltage_array = np.array(voltage[0])
        else:
            voltage_array = voltage

        notch_freq = 59.06
        q = 40
        voltage_array = notch(voltage_array, notch_freq, q, fs)

        notch_freq = 0.1
        q = 10
        voltage_array = notch(voltage_array, notch_freq, q, fs)
        
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
            plt.xlim([4, 5000])
            # plt.close()
        else:
            pass
        return fc
    else:
        pass

def histogram(trapped, fs, bins='auto', density=True, plot=True, run=False):
    if run:
        
        # Apply a 10 Hz low-pass filter to the voltage data
        trapped = lowpass(trapped, 100, fs)
        if plot:
            plt.figure(figsize=(5,8))
            counts, bins, bars = plt.hist(trapped, bins = 10000, color = 'red', alpha = 0.25, edgecolor='red',  orientation='horizontal', density = True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            plt.xlabel('Density', labelpad=12.5)
            plt.ylabel('Transmission (V)')
            plt.ylim([0.42, 0.56])
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y')
            plt.tight_layout()
            plt.show()


            plt.figure(figsize=(8,5))
            # Calculate the bin centers
            bin_centers = (bins[:-1] + bins[1:]) / 2
            # Plot and fill the original histogram
            plt.fill_betweenx(bin_centers, 0, counts * (bins[1] - bins[0]), color='red', alpha=0.5)
            # Overlay the filled histogram with counts multiplied by 10
            plt.fill_betweenx(bin_centers, 0, counts * 10 * (bins[1] - bins[0]), color='red', alpha=0.25)
            plt.xlabel('Density', labelpad=12.5)
            plt.ylabel('Transmission (V)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y')
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(5,8))
            Vx = -1*np.log(counts)
            plt.plot(Vx, color = 'red', linewidth = 2, linestyle = '-')
            plt.ylabel('$K_{b}$T')
            plt.grid(axis='y')
            plt.tight_layout()
            plt.show()
            
        else:
            pass

        return
    else:
        return

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

        print("File Name:", file_name)
        print("Average RMSD:", average_rmsd)

        return average_rmsd
    else:
        pass
       
t = True
f = False
PSD_data = []
files = []
cornerFreq = []
file_data = plot_file_data(plot=t)
if file_data:
    print("Files selected successfully!")
    average_rmsd_list = []
    for data in file_data:
        time = data['time']

        #Select the start and end point of the data (in seconds)
        start = 10
        ending = 15 

        voltage = data['voltage']

        minTrappedTime = find_nearest_idx(time, start)
        maxTrappedTime = find_nearest_idx(time, ending)

        trapped = voltage[minTrappedTime:maxTrappedTime]

        fs = data['fs']
        name = data['name']

        filtered_trap = lowpass(trapped, 10, fs)

        print(name)


        #Select which functions to run
        psd_run = t
        rmsd_run = t
        hist_run = t
        
        #Call histogram function
        histogram(trapped, fs, run=hist_run)
        average_rmsd = calculate_rmsd(trapped, 50, name, run=rmsd_run)
        average_rmsd_list.append(average_rmsd)

        #Call compute_PSD function to compute PSD
        psd = compute_PSD(trapped, fs, name, run=psd_run)
        PSD_data.append(psd)
        files.append(name)


    for array, file_name in zip(PSD_data, files):
        fc = psd_fitter(array, 4, 5000, file_name, run=psd_run)
        cornerFreq.append(fc)

    print(average_rmsd_list)
    print(cornerFreq)


else:
    print("No files selected.")
