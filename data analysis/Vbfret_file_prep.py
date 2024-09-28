import os
from tkinter import Tk, filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
# Function to extract data within the specified time range and write to a new text file
def extract_data(input_file, output_file, start_time, end_time):
    # Load the data from the input file
    data = pd.read_csv(input_file, skiprows=16, header=None, delim_whitespace=True)
    # Calculate the indices corresponding to the specified time range
    start_index = int(start_time / 0.00001) - 16
    end_index = int(end_time / 0.00001) - 16
    # Extract the voltage data within the specified time range
    extracted_data = data.iloc[start_index:end_index, 0]
    # Write the extracted data to the output file
    extracted_data.to_csv(output_file, index=False, header=False, sep='\t')
# Function to plot data
def plot_file_data(plot=False):
    root = Tk()
    root.withdraw()
    file_names = filedialog.askopenfilenames(
        filetypes=[('CSV files', '*.csv'),
                   ('Text files', '*.txt')],
        title='Select files'
    )
    if not file_names:
        raise ValueError('No files selected')
    num_files = len(file_names)
    file_data = []
    for i in range(num_files):
        file_path = str(file_names[i])
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            data = pd.read_csv(file_path, skiprows=7, header=None).values
            time = data[:, 0]
            voltage = data[:, 1]
        elif ext == '.txt':
            # Example usage of extract_data function
            # Specify the desired time range (in seconds)
            start_time = 110
            end_time = 140 #max 140
            output_file = r'Downloads\41.8deg_VB_110-140.txt'
            # Call the extract_data function to create a new txt file
            extract_data(file_path, output_file, start_time, end_time)
            data = pd.read_csv(output_file, header=None).values
            voltage = data[:, 0]
            time_step = 0.00001
            time = np.arange(0, len(voltage) * time_step, time_step)
            # # Apply filtering to the voltage signal
            cutoff_frequency = 5  # 10 Hz
            nyquist_frequency = 0.5 * 100000
            b, a = signal.butter(1, cutoff_frequency / nyquist_frequency, btype='low')
            filtered_voltage = signal.lfilter(b, a, voltage)
            # Save the filtered voltage data to the output file
            filtered_output_file = r'Downloads\41.8deg_VB_filtered_110-140.txt'
            # Define the time range for saving, skipping the first 0.5 seconds
            start_index = int(0.5 / 0.00001)  # Calculate the index corresponding to 0.5 seconds
            filtered_voltage_to_save = filtered_voltage[start_index:]  # Slice the filtered_voltage array from the calculated index
            # Create a DataFrame with two columns: '1-filtered_voltage' and 'filtered_voltage'
            filtered_data = pd.DataFrame({
                '1-filtered_voltage': 1 - filtered_voltage_to_save,
                'filtered_voltage': filtered_voltage_to_save
            })
            # Save the DataFrame to the output file without headers
            filtered_data.to_csv(filtered_output_file, index=False, sep='\t', header=False)
            # Plot the original and filtered voltage data
            plt.plot(time, voltage, label='Original Voltage')
            plt.plot(time, filtered_voltage, label='Filtered Voltage')
            plt.xlabel('Time')
            plt.ylabel('Voltage')
            plt.title('Original vs Filtered Voltage')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            raise ValueError('File type not supported')
# Call the function to plot data
plot_file_data()