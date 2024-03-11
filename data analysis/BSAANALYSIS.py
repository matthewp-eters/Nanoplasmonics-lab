#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:27:07 2024

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
from scipy.stats import expon
sns.set_theme(style='ticks')


plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24)
plt.rc('axes', labelsize=29)
plt.rc('legend', fontsize=18)
plt.rc('font', family='sans-serif')



# Define a linear function for curve fitting
def linear_fit(x, a, b):
    return a * x + b

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

        # Store file data in array
        file_data.append({'time': time, 'voltage': voltage,
                         'fs': 100000, 'name': file_names[i]})


        # Use curve_fit to fit the linear function to the data
        popt, pcov = curve_fit(linear_fit, time, voltage)

        # Calculate the linear fit using the fitted parameters
        linear_fit_data = linear_fit(time, *popt)

        # Subtract the linear fit from the original data
        voltage -= linear_fit_data
        
        
        # Apply a 1Hz low-pass filter to the voltage data
        cutoff_frequency = 10  # 1 Hz
        nyquist_frequency = 0.5 * 100000  # Nyquist frequency for your sampling rate
        b, a = signal.butter(1, cutoff_frequency /
                             nyquist_frequency, btype='low')
        filtered_voltage = signal.lfilter(b, a, voltage)

        # Plot the data
        if plot:

            plt.figure(figsize=(20, 10))

            plt.plot(time, voltage+1, linewidth=1,color='red', alpha=0.25, label='APD')
            plt.plot(time, filtered_voltage+1, linewidth=2, color = 'red', label='APD (1Hz LPF)')
            plt.axhline(1, color = 'k', linewidth = 1, linestyle = '--')
            
            
            
            plt.xticks(np.arange(0, max(time), step=10))
            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Time (s)')
            plt.ylabel('APD Signal (V)')
            #plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
            
            xmin = 100
            #xmax = len(time)*time_step
            xmax = 110
            ymin = 0.85
            ymax = 1.15
            
            plt.xlim([xmin, xmax])
            plt.ylim([ymin, ymax])
            plt.grid(axis='y')
            
            plt.tight_layout()       
            plt.savefig(file_names[i]+'trapZoom.png')
            

        else:
            pass

    return file_data

def histogram(trapped, name, bins='auto', density=True, plot=True, run=False):
    if run:
        
        #Apply a 1Hz low-pass filter to the voltage data
        cutoff_frequency = 10 # 1 Hz
        nyquist_frequency = 0.5 * 100000  # Nyquist frequency for your sampling rate
        b, a = signal.butter(1, cutoff_frequency /
                              nyquist_frequency, btype='low')
        trapped = signal.lfilter(b, a, trapped)+1
        if plot:
            plt.figure(figsize=(5,10))
            
            #counts, bins, bars = plt.hist(trapped, bins = 1000, color = 'r', alpha = 0.5, edgecolor='r',  orientation='horizontal')
            counts, bins, bars = plt.hist(trapped, bins = 100, color = 'red', alpha = 0.5, edgecolor='red',  orientation='horizontal', density = True)
            plt.xlabel('Density', labelpad=12.5)
            plt.ylabel('APD Signal (V)')
            plt.ylim([0.85,1.15])
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y')
            plt.tight_layout()
            plt.savefig(name+'Hist.png')
            plt.show()
            
            plt.figure(figsize=(5,10))
            counts, bins, bars = plt.hist(trapped, bins = 100, color = 'red', alpha = 0.5, edgecolor='red',  orientation='horizontal', density = False)
            
            plt.figure(figsize=(10,10))
            Vx = -1*np.log(counts)
            plt.plot(Vx, color = 'red', linewidth = 2, linestyle = '-')
            plt.xlabel('Reaction Coordinate')
            plt.ylabel('$K_{b}$T')
            plt.xticks(rotation=45, ha='right')
            
            plt.ylim([-18, 0])
            plt.tight_layout()
            plt.savefig(name+'energyLandscape.svg')
            plt.show()
            
        else:
            pass

        return
    else:
        return

def find_nearest_idx(array, value):
    """
    Returns the index of the nearest value in an array to a given value.
    """
    return np.argmin(np.abs(array - value))

def residence_times_high(trapped, fs, threshold):
    res_high = []
    start_high = None
    state_high = False
    

    
    for i in range(len(trapped)):
        if trapped[i] > threshold and not state_high:
            state_high = True
            start_high = i
            
            
        elif trapped[i] <= threshold and state_high:
            state_high = False
            event_end_time = i
            event_duration = (event_end_time - start_high)/fs
            res_high.append(event_duration)
    return res_high
      
def residence_times_low(trapped, fs, threshold):
    res_low = []
    
    start_low = None
    state_low = False
    
    for i in range(len(trapped)):
        if trapped[i] <= threshold and not state_low:
            state_low = True
            start_low = i
            
            
        elif trapped[i] > threshold and state_low:
            state_low = False
            event_end_time = i
            event_duration = (event_end_time - start_low)/fs
            res_low.append(event_duration)
    return res_low
            
# Define the double exponential function
def double_exponential(x, a, b, c, d):
    return a * np.exp(-b * x) + c * np.exp(-d * x)

def single_exponential(x, a, b):
    return a*np.exp(-b*x)

def res_hist(data, name, bins='auto', color='forestgreen', alpha=0.5, edge_color = 'r' ):
    # Plot histogram of event durations
    hist_values, bin_edges = np.histogram(data, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

     # Fit the double exponential function to the histogram data
    #popt, _ = curve_fit(double_exponential, bin_centers, hist_values,maxfev=5000)
    popt, _ = curve_fit(single_exponential, bin_centers, hist_values,maxfev=5000)

    # Extract the optimized parameters
    #a_opt, b_opt, c_opt, d_opt = popt
    a_opt, b_opt = popt

    # Calculate the decay rate constants
    decay_rate_constant_b = 1 / b_opt
    #decay_rate_constant_d = 1 / d_opt

    # Plot histogram
    plt.figure(figsize=(5, 5))
    plt.hist(data, bins=bins, color=color, alpha=alpha, edgecolor=edge_color, label='Histogram')
    plt.plot(bin_centers, single_exponential(bin_centers, *popt), 'k-', label='Fitted Curve')
    # Adding text on the plot.
    equation_text = rf'$y = {a_opt:.2f} \cdot e^{{-{b_opt:.2f}x}}$'
    plt.text(0.95, 0.95, equation_text, transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5), fontsize = 22)




    plt.xlabel('State Duration (s)', labelpad=12.5)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) # 2 decimal places

    plt.tight_layout()
    plt.savefig(name)
    plt.show()
    # Print the optimized parameters and decay rate constants
    print("Optimized Parameters:")
    print("a =", a_opt)
    print("b =", b_opt)
    # print("c =", c_opt)
    # print("d =", d_opt)
    print("Decay Rate Constant for b:", decay_rate_constant_b)
    # print("Decay Rate Constant for d:", decay_rate_constant_d)
    
    return a_opt, b_opt#, c_opt, d_opt

def plot_file_data_video(fps, threshold, plot=False):
    
    

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
            data = pd.read_csv(file_path, skiprows=0,
                               header=None, delim_whitespace=True).values
            voltage = data[:, 0]
            time_step = 1 / fps  # Time step based on fps
            time = np.arange(0, len(voltage) * time_step, time_step)
        elif ext == '.tdms':
            tdms_file = TdmsFile(file_path)
            properties = tdms_file['Analog'].properties
            scanrate = properties['ScanRate']
            reference = tdms_file['Analog']['AI1']
            voltage = tdms_file['Analog']['AI2']
            time_step = 1 / fps  # Time step based on fps
            time = np.arange(0, len(voltage) * time_step, time_step)
        else:
            raise ValueError('File type not supported')

        # Store file data in array
        file_data.append({'time': time, 'voltage': voltage,
                          'fs': fps, 'name': file_names[i]})


        # Plot the data
        if plot:
            plt.figure(figsize=(14, 5))

            plt.plot(time, voltage, linewidth=1, color='r', alpha=0.25, label='APD')
            plt.axhline(threshold, color='k', linewidth=1, linestyle='--')

            # # Use the time_values and tick_positions from the initial code
            # time_values = time
            # tick_positions = np.arange(0, len(time_values), int(10 * fps))
            # tick_values = np.round(time_values[tick_positions])

            # # Format tick labels to remove ".0"
            # tick_labels = [str(int(val)) for val in tick_values]
            # plt.xticks(tick_positions, tick_labels)  # Set x-axis ticks to formatted tick labels

            plt.xticks(rotation=45, ha='right')
            plt.xlabel('Time (s)')
            plt.ylabel('APD Signal (V)')
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))  # 2 decimal places

            xmin = 70
            xmax = 100
            ymin = 1.5
            ymax = ymin + 0.34

            # Uncomment these lines if you want to set specific limits for the x and y axes
            # plt.xlim([xmin, xmax])
            # plt.ylim([ymin, ymax])

            plt.tight_layout()

            # Add legend for better visualization
            plt.legend()

            # Show the plot
            plt.show()
        else:
            pass

    return file_data


t = True
f = False
files = []

file_data = plot_file_data(plot=t)
#file_data = plot_file_data_video(29.97, 189, plot = t)
if file_data:
    print("Files selected successfully!")
    average_rmsd_list = []
    for data in file_data:
        time = data['time']
        start = 60
        ending = 205
        voltage = data['voltage']
        minTrappedTime = find_nearest_idx(time, start)
        maxTrappedTime = find_nearest_idx(time, ending)
        trapped = voltage[minTrappedTime:maxTrappedTime]
        fs = data['fs']
        name = data['name']
        

        
            
        print(name)
        
        threshold_value = 1  # You can adjust this threshold as needed

        hist_run = f
        
        histogram(trapped,name, run=hist_run)
        
        # cutoff_frequency = 1  # 1 Hz
        # nyquist_frequency = 0.5 * fs  # Nyquist frequency for your sampling rate
        # b, a = signal.butter(1, cutoff_frequency /
        #                       nyquist_frequency, btype='low')
        # trapped = signal.lfilter(b, a, trapped)+1
        
        # res_high = residence_times_high(trapped, fs, threshold_value)
        # res_low = residence_times_low(trapped, fs, threshold_value)

        # high_sum = np.sum(res_high)
        # low_sum = np.sum(res_low)
        
        # Keq = high_sum/low_sum
        # print(f"Keq: {Keq}")

        # #a_high, b_high, c_high, d_high = res_hist(res_high, bins = 10, color = 'r', edge_color = 'r')

        # #a_low, b_low, c_low, d_low = res_hist(res_low, bins = 10, color = 'b', edge_color = 'b')
        
        # a_high, b_high = res_hist(res_high, name+'resHigh.png', bins = 10, color = 'red', edge_color = 'red')

        # a_low, b_low = res_hist(res_low, name + 'resLow.png', bins = 10, color = 'blue', edge_color = 'blue')
        
        # KD = b_low/b_high
        # print(f"KD: {KD}")
        
        
        # # # Read data from Excel file
        # excel_file_path = '/Users/matthewpeters/Desktop/UVIC/lab/Projects/BSA/BSATransitions.xlsx'  # Replace with the actual file path
        # df = pd.read_excel(excel_file_path, sheet_name='Sheet1')

        
        # # Process each row in the DataFrame
        # for index, row in df.iterrows():
        #     time1 = row['TransStart']
        #     time2 = row['TransEnd']
        #     transition = row['Transition']

        #     time = data['time']
        #     start = time1
        #     ending = time2
        #     voltage = data['voltage']
            
        #     minTrappedTime = find_nearest_idx(time, start)
        #     maxTrappedTime = find_nearest_idx(time, ending)
                
        #     trapped = voltage[minTrappedTime:maxTrappedTime]
            
        #     cutoff_frequency = 1  # 1 Hz
        #     nyquist_frequency = 0.5 * fs  # Nyquist frequency for your sampling rate
        #     b, a = signal.butter(1, cutoff_frequency /
        #                           nyquist_frequency, btype='low')
        #     trapped = signal.lfilter(b, a, trapped)

                    
        #     # Save the transition_array to a file
        #     np.savetxt(f'BSA001_{transition}_{index + 1}.txt', np.array(trapped))
        #     plt.figure(figsize=(10,8))
        #     plt.plot(trapped, color = 'k')
        #     plt.grid()
        #     plt.tight_layout()
        #     plt.savefig(f'BSA001_{transition}_{index + 1}.png')
        #     plt.close()
            

                




else:
    print("No files selected.")
