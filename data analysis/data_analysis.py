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
            time = np.arange(0, (len(voltage)-0.5)*time_step, time_step)
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
        plt.xticks(np.arange(0, max(time), step=5))
        plt.xlabel('Time (s)', fontsize=20, fontweight='bold')
        plt.ylabel('Voltage (V)', fontsize=20, fontweight='bold')
        plt.xticks(fontsize=16)
        plt.title(file_names[i])
        plt.show()
        

    return file_data


def Lorentzian(f, A, fc):
    return A / (f**2 + fc**2)


def compute_PSD(voltage, fs, start, ending, name):
    if not isinstance(voltage, np.ndarray):
        voltage_array=np.array(voltage[0])
    else:
        voltage_array = voltage
        
    voltage_fft = scipy.fftpack.fft(voltage_array)
    voltage_PSD = np.abs(voltage_fft)**2 / (len(voltage_array)/fs)
    fftfreq = scipy.fftpack.fftfreq(len(voltage_PSD), d=1/fs)
    i = fftfreq > 0
    
    return fftfreq[i], voltage_PSD[i], name


def find_nearest_idx(array, value):
    """
    Returns the index of the nearest value in an array to a given value.
    """
    return np.argmin(np.abs(array - value))


def psd_fitter(psd, minFreq, maxFreq, plot = False):
    freqs = psd[0]
    psd_data = psd[1]
    file_name = psd[2]
    
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
        #plt.text(.01, .01, file_name, ha='left', va='bottom')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('P (V$^2$/Hz')
        plt.xscale('Log')
        plt.yscale('Log')
        plt.show()
    else:
        pass
    return params, rSquared



def gaussian(x, A, beta, B, mu, sigma):
    return (A * np.exp(-x/beta) + B * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2)))

def histogram(trapped, file_name, plot=False):


    if plot:
        #sns.set_style('darkgrid')
        sns.displot(trapped, kde=True)
        
    else:
        pass

    #print("File Name:", file_name)
    #print("FWHM:", fwhm)

    return #fwhm

    
    

def plot_on_top(files, plot = False):
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
        
def calculate_rms(signal, window_length, file_name):
    """
    Calculate the RMS for sections of the input signal based on a single window length
    and print the file name and average RMS value.

    Args:
        signal (array-like): Input signal.
        window_length (int): Length of the window for splitting the data.
        file_name (str): Name of the file.

    Returns:
        float: Average RMS value.
    """
    num_sections = len(signal) // window_length
    rms_values = []

    for i in range(num_sections):
        section = signal[i * window_length : (i + 1) * window_length]
        squared_values = np.square(section)
        mean_squared = np.mean(squared_values)
        rms = np.sqrt(mean_squared)
        rms_values.append(rms)

    average_rms = np.mean(rms_values)

    print("File Name:", file_name)
    print("Average RMS:", average_rms)

    return average_rms


        
def newFile(trapped):
        # reshape data to a single column
        trapped = trapped.reshape(-1, 1)
            
        # save data to text file
        np.savetxt('CTC009_trapped.txt', trapped, fmt='%.8f', delimiter='\n')
        
    
def main():
    file_data = plot_file_data()
    PSD_data = []
    if file_data:
        print("Files selected successfully!")
        
        plot_on_top(file_data, plot = False)
        
        # for signal in file_data:
        #     start, ending = find_trapped_signal(signal['voltage'])
            
        for data in file_data:
            time = data['time']
            start = time[0]
            ending = 5
            voltage = data['voltage']
            minTrappedTime = find_nearest_idx(time, start)
            maxTrappedTime = find_nearest_idx(time, ending)
            trapped = voltage[minTrappedTime:maxTrappedTime]
            fs = data['fs']
            name = data['name']
        #newFile(trapped)
            
            # Call compute_PSD function to compute PSD
            psd = compute_PSD(trapped, fs, start, ending, name)
            PSD_data.append(psd)
            
            calculate_rms(trapped, 5, name)
            histogram(trapped, name, plot=False)
            
        for array in PSD_data:
            psd_fitter(array, 5, 4000, plot=False)
            
            
    else:
        print("No files selected.")

if __name__ == '__main__':
    main()
