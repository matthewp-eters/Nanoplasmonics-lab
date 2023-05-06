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


def plot_file_data():
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
            time = data[:,0]
            voltage = data[:,2]
        elif ext == '.txt':
            data = pd.read_csv(file_path, skiprows=16, header=None, delim_whitespace=True).values
            voltage = data[:,0]
            time_step = 0.00001
            time = np.arange(0, len(voltage)*time_step, time_step)
        elif ext == '.tdms':
            tdms_file = TdmsFile(file_path)
            groups = tdms_file.groups()
            group_name = list(groups.keys())[0]
            group = groups[group_name]
            time_channel = group['time']
            voltage_channel = group['voltage']
            time = time_channel.time_track()
            voltage = voltage_channel[:]
        else:
            raise ValueError('File type not supported')

        # Store file data in array
        file_data.append({'time': time, 'voltage': voltage, 'fs': 100000, 'name': file_names[i]})

        # Plot the data
        
        plt.figure()
        plt.plot(time, voltage, linewidth=1)
        plt.xticks(np.arange(0, max(time), step=100))
        plt.xlabel('Time (s)', fontsize=20, fontweight='bold')
        plt.ylabel('Voltage (V)', fontsize=20, fontweight='bold')
        plt.xticks(fontsize=16)
        plt.title(file_names[i])
        plt.show()
        

    return file_data


def Lorentzian(f, A, fc):
    return A / (f**2 + fc**2)


def compute_PSD(voltage, fs, start, ending):
    if not isinstance(voltage, np.ndarray):
        voltage_array=np.array(voltage[0])
    else:
        voltage_array = voltage
        
    voltage_fft = scipy.fftpack.fft(voltage_array)
    voltage_PSD = np.abs(voltage_fft)**2 / (len(voltage_array)/fs)
    fftfreq = scipy.fftpack.fftfreq(len(voltage_PSD), d=1/fs)
    i = fftfreq > 0
    
    return fftfreq[i], voltage_PSD[i]


def find_nearest_idx(array, value):
    """
    Returns the index of the nearest value in an array to a given value.
    """
    return np.argmin(np.abs(array - value))


def psd_fitter(psd, minFreq, maxFreq, plot = False):
    freqs = psd[0]
    psd_data = psd[1]
    
    min_freq_idx = find_nearest_idx(freqs, minFreq)
    max_freq_idx = find_nearest_idx(freqs, maxFreq)
    
    fs = freqs[min_freq_idx:max_freq_idx] #freqs for fitting x axis
    ys = psd_data[min_freq_idx:max_freq_idx] #data for fitting y axis
    
    p0 = (ys[0], 100) #A, fc
    params, cv = scipy.optimize.curve_fit(Lorentzian, fs, ys, p0, sigma = None, absolute_sigma = True )
    A, fc = params
    squaredDiffs = np.square(ys - Lorentzian(fs, A, fc))
    squaredDiffsFromMean = np.square(ys-np.mean(ys))
    rSquared = 1- np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
    
    if plot:
        plt.figure()
        plt.plot(fs, ys, '.', label = "data")
        plt.plot(fs, Lorentzian(fs, A, fc),'--', label="fitted")
        plt.title(f"Corner freq = {-fc} Hz")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('P (V$^2$/Hz')
        plt.xscale('Log')
        plt.yscale('Log')
        plt.show()
    else:
        pass
    return params, rSquared




from scipy.signal import butter, filtfilt

def find_trapped_signal(data, threshold=0.1, window_size=10):
    # Apply low-pass filter to smooth signal
    b, a = butter(3, 0.1)
    smoothed_data = filtfilt(b, a, data)

    # Calculate derivative of smoothed signal
    derivative = np.gradient(smoothed_data)

    # Calculate standard deviation of signal in sliding window
    std_dev = np.zeros_like(smoothed_data)
    for i in range(window_size // 2, len(smoothed_data) - window_size // 2):
        std_dev[i] = np.std(smoothed_data[i - window_size // 2 : i + window_size // 2])

    # Find indices where both derivative and std_dev exceed threshold
    idx = np.argwhere((np.abs(derivative) > threshold) & (std_dev > threshold))

    # Find the largest group of consecutive indices
    groups = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    largest_group = max(groups, key=len)
    start_idx, end_idx = largest_group[0], largest_group[-1]

    print(f"Trapped signal start: {start_idx}, end: {end_idx}")
    return (start_idx, end_idx)




def main():
    file_data = plot_file_data()
    PSD_data = []
    if file_data:
        print("Files selected successfully!")
       # start = 51
       # ending = 60
        
        for signal in file_data:
            start, ending = find_trapped_signal(signal['voltage'])
            
        for data in file_data:
            time = data['time']
            voltage = data['voltage']
            minTrappedTime = find_nearest_idx(time, start)
            maxTrappedTime = find_nearest_idx(time, ending)
            trapped = voltage[minTrappedTime:maxTrappedTime]
            fs = data['fs']
            name = data['name']
            
            # Call compute_PSD function to compute PSD
            psd = compute_PSD(trapped, fs, start, ending)
            PSD_data.append(psd)
        for array in PSD_data:
            psd_fitter(array, 1, 4000, plot=True)
            
            
    else:
        print("No files selected.")

if __name__ == '__main__':
    main()
