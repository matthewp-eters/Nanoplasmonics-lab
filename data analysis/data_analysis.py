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
        if plot:
            
            plt.figure()
            plt.plot(time, voltage, linewidth=1)
            plt.xticks(np.arange(0, max(time), step=5))
            plt.xlabel('Time (s)', fontsize=20, fontweight='bold')
            plt.ylabel('Voltage (V)', fontsize=20, fontweight='bold')
            plt.xticks(fontsize=16)
            plt.title(file_names[i])
            plt.show()
        else:
            pass
        

    return file_data


def Lorentzian(f, A, fc):
    return A / (f**2 + fc**2)


def compute_PSD(voltage, fs, start, ending, name, run=False):
    if run:
     
        if not isinstance(voltage, np.ndarray):
            voltage_array=np.array(voltage[0])
        else:
            voltage_array = voltage
        
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


def psd_fitter(psd, minFreq, maxFreq, plot = False, run=False):
    if run:
        
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
    else:
        pass 
    

def histogram(trapped, bins='auto', density=True, plot=True, run=False):

    if run:
        
        if plot:
            """
            Calculate and plot the histogram of a signal along with a probability density function (PDF).
        
            Args:
                signal (array-like): Input signal.
                bins (int or array-like, optional): Number of bins or bin edges for the histogram. Defaults to 'auto'.
                density (bool, optional): If True, normalize the histogram to form a probability density function (PDF).
                                          Defaults to True.
                plot_pdf (bool, optional): If True, plot the PDF on the same graph. Defaults to True.
            """
            # Calculate the histogram
            hist, bin_edges = np.histogram(trapped, bins=bins, density=density)
        
            # Calculate the bin centers
            bin_centers = (bin_edges[:1] + bin_edges[-1:]) / 2
        
            # Plot the histogram
        
                # Calculate the PDF
            pdf = stats.norm.pdf(bin_centers)
        
                # Plot the PDF
            plt.figure()
            plt.plot(bin_centers, hist, label="Histogram of samples")
            plt.plot(bin_centers, pdf, 'r-', label='PDF')
        
            plt.xlabel('Value')
            plt.ylabel('Frequency' if not density else 'Probability Density')
            plt.legend()
            plt.show()
        else:
            pass
    
        #print("File Name:", file_name)
        #print("FWHM:", fwhm)
    
        return #fwhm
    else:
        return
 

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
         
         for i in range(num_sections):
             section = signal[i * window_length : (i + 1) * window_length]
             squared_diff = np.square(section - np.mean(section))
             mean_squared_diff = np.mean(squared_diff)
             rmsd = np.sqrt(mean_squared_diff)/np.mean(section)
             rmsd_values.append(rmsd)
             
             average_rmsd = np.mean(rmsd_values)
             
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
        np.savetxt('CA011_LASER_NOISE', trapped, fmt='%.8f', delimiter='\n')
    else:
           pass 
    
def main():
    file_data = plot_file_data(plot=False)
    PSD_data = []
    if file_data:
        print("Files selected successfully!")
        
        plot_on_top(file_data, plot = False)
        
        average_rmsd_list = []
        for data in file_data:
            time = data['time']
            start = time[0]
            ending = time[-1]
            voltage = data['voltage']
            minTrappedTime = find_nearest_idx(time, start)
            maxTrappedTime = find_nearest_idx(time, ending)
            trapped = voltage[minTrappedTime:maxTrappedTime]
            fs = data['fs']
            name = data['name']
            
            # Call compute_PSD function to compute PSD
            psd = compute_PSD(trapped, fs, start, ending, name, run=False)
            PSD_data.append(psd)
            
            average_rmsd = calculate_rmsd(trapped, 5, name, run=True)
            average_rmsd_list.append(average_rmsd)
            
            histogram(trapped, plot=False, run=False)
        newFile(trapped, write=False)

        for array in PSD_data:
            psd_fitter(array, 5, 4000, plot=False, run=False)
        print(average_rmsd_list)

            
    else:
        print("No files selected.")

if __name__ == '__main__':
    main()
