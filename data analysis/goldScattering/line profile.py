#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:54:22 2023

@author: matthewpeters
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

def find_nearest_idx(array, value):
    """
    Returns the index of the nearest value in an array to a given value.
    """
    return np.argmin(np.abs(array - value))

plot = True
root = Tk()
root.withdraw()

file_names = filedialog.askopenfilenames(filetypes=[('CSV files', '*.csv'),
                                                    ('Text files', '*.txt'),
                                                    ('TDMS files', '*.tdms')],
                                         title='Select files')

if not file_names:
    raise ValueError('No files selected')

# Initialize data containers for "glass" and "gold" files
glass_data = []
gold_data = []

for file_path in file_names:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        data = pd.read_csv(file_path, skiprows=1, header=None).values
        pixel = data[:, 0]
        intensity = data[:, 1]

        # Find the index and value of the minimum intensity
        min_index = np.argmin(intensity)
        min_intensity = np.min(intensity)

        # Shift intensity data to align minimum at pixel 0
        shifted_intensity = intensity
        shifted_pixel = pixel - pixel[min_index]

        # Determine whether the file starts with "glass" or "gold" and store data accordingly
        if os.path.basename(file_path).startswith("glass"):
            glass_data.append({'pixel': shifted_pixel, 'intensity': shifted_intensity, 'name': file_path})
        elif os.path.basename(file_path).startswith("gold"):
            gold_data.append({'pixel': shifted_pixel, 'intensity': shifted_intensity, 'name': file_path})

# Plot mean and confidence graphs for "glass" data
# ... (previous code)

# Determine a common pixel range for trimming
common_min_pixel = -32
common_max_pixel = 32

# Plot mean and confidence graphs for "glass" data
plt.figure(figsize=(8, 6))
for file in glass_data:
    min_idx = find_nearest_idx(file['pixel'], common_min_pixel)
    max_idx = find_nearest_idx(file['pixel'], common_max_pixel)
    file['pixel'] = file['pixel'][min_idx:max_idx]
    file['intensity'] = file['intensity'][min_idx:max_idx]

all_glass_intensities = np.array([file['intensity'] for file in glass_data])
mean_glass_intensity = np.mean(all_glass_intensities, axis=0)
std_glass_intensity = np.std(all_glass_intensities, axis=0)
confidence_interval_glass = 1.96 * (std_glass_intensity / np.sqrt(len(glass_data)))

plt.plot(glass_data[0]['pixel'], mean_glass_intensity, color='blue', label='Glass Mean', linewidth=2)
plt.fill_between(glass_data[0]['pixel'], mean_glass_intensity - confidence_interval_glass, mean_glass_intensity + confidence_interval_glass, color='blue', alpha=0.3)

# Plot mean and confidence graphs for "gold" data
for file in gold_data:
    min_idx = find_nearest_idx(file['pixel'], common_min_pixel)
    max_idx = find_nearest_idx(file['pixel'], common_max_pixel)
    file['pixel'] = file['pixel'][min_idx:max_idx]
    file['intensity'] = file['intensity'][min_idx:max_idx]

all_gold_intensities = np.array([file['intensity'] for file in gold_data])
mean_gold_intensity = np.mean(all_gold_intensities, axis=0)
std_gold_intensity = np.std(all_gold_intensities, axis=0)
confidence_interval_gold = 1.96 * (std_gold_intensity / np.sqrt(len(gold_data)))

plt.plot(gold_data[0]['pixel'], mean_gold_intensity, color='orange', label='Gold Mean', linewidth=2)
plt.fill_between(gold_data[0]['pixel'], mean_gold_intensity - confidence_interval_gold, mean_gold_intensity + confidence_interval_gold, color='orange', alpha=0.3)

plt.xlabel('Pixel')
plt.ylabel('Intensity')
plt.title('Glass and Gold Intensity')
plt.legend()
plt.show()
