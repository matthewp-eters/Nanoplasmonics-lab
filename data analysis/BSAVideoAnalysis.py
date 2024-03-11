#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 09:34:50 2024

@author: matthewpeters
"""
import cv2
import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
#from skimage.morphology import skeletonize, thin
import matplotlib.pyplot as plt
#from PIL import Image
#import math
#from skimage.measure import profile_line
#from skimage import io
from scipy.fftpack import fft, fftfreq
import matplotlib.patches as patches
from pprint import pprint
import sys
from matplotlib.lines import Line2D
import seaborn as sns
import re
import pandas as pd
import scipy.stats as stats
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
plt.rc('axes', labelsize=25) 
plt.rc('font', family='sans-serif')


def video_info(video_path, run=False):
    
    if run:
        
        #open video file
        video = cv2.VideoCapture(video_path)
        
        #read first frame
        success, frame = video.read()
        fps = video.get(cv2.CAP_PROP_FPS)
    
        # Get the total number of frames in the video
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        height, width, channels = frame.shape
        center_coords = (width/2, height/2)
        #release video
        video.release()
        

        return frame, fps, total_frames, center_coords

    else:
        pass 
    
def image_profile(video_path, center_coords, num_angles, total_frames, run = False):
    
    if run:
        # Open video file
        video = cv2.VideoCapture(video_path)
    
        matrices = []
        angles = []
    
        # Read the first frame
        success, frame = video.read()
    
        for angle_idx in range(num_angles):
            # Calculate the angle for the profile
            angle = (angle_idx * 180) / num_angles
            print("angle",angle)
            frame_profile_matrix = np.zeros((total_frames, 1920))
            M = cv2.getRotationMatrix2D(center_coords, angle, 1)
    
            # Reset the video capture to the beginning
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
            frame_index = 0
            while True:
                success, frame = video.read()
                if not success:
                    break
                rows, cols, channels = frame.shape
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.warpAffine(frame, M, (cols, rows))
    
                frame_profile = frame[center_coords[1], :]
    
                frame_profile_matrix[frame_index] = frame_profile
                frame_index += 1
    
            matrices.append(frame_profile_matrix)
            angles.append(angle)
    
        # Release the video capture
        video.release()
    
        return matrices, angles
    else:
        pass
    return 0, 0

def visualize_data(profiles, angles, fps, col_change, plot=False, run=False):
    
    if run:
        
        rms_values_list = []  # List to store RMS values for each column
        avg_rms_before_list = []  # List to store average RMS before values
        avg_rms_after_list = []  # List to store average RMS after values
        
        profiles_list = []

        for i, prof in enumerate(profiles):
            
            prof = prof.transpose()
            
            # Compute the one-sided FFT for each column
            fft_matrix = np.zeros((prof.shape[0] // 2, prof.shape[1]))
            for col_idx, col in enumerate(prof.T):
                fft_values = fft(col)
                frequencies = fftfreq(len(col), 1 / fps)
                positive_freq_mask = frequencies >= 0
                fft_matrix[:, col_idx] = np.abs(fft_values)[positive_freq_mask]
            
            # Normalize the frequency plot using a logarithmic scale
            fft_matrix_log = np.log10(fft_matrix + 1)  # Add 1 to avoid taking the log of zero
    
    
            # Assuming 'profile_matrix' is your profile matrix
            change_column_idx = col_change
            print("Visualize data", change_column_idx)
            
            slice_of_frame_before = []
            slice_of_frame_after = []
    
            if change_column_idx is not None:
                # Average the FFT plots before and after the change index
                fft_before = np.mean(fft_matrix[:, :change_column_idx], axis=1)
                fft_after = np.mean(fft_matrix[:, change_column_idx:], axis=1)

            

                
    
                if plot:
#%%
                    profiles_list.append(prof)
                    plt.figure(figsize = (15,4))
                    plt.imshow(prof, cmap='coolwarm', interpolation='nearest', aspect='auto')
                    #plt.title(f"Time Domain - Angle: {angles[i]}")
                    plt.xlabel('Time (s)')  # Change x-axis label to Time
                    plt.ylabel('Pixel')
                    plt.ylim([0, 1920])
                    #plt.xlim([45, 55])
                    
                    # Calculate time values for x-axis based on frame index and fps
                    frame_indices = np.arange(prof.shape[1])
                    time_values = frame_indices / fps
                    # Select tick positions at 50-second intervals
                    tick_positions = np.arange(0, len(time_values), int(10 * fps))
                    tick_values = np.round(time_values[tick_positions])  # Round to nearest whole number
                    
                    # Format tick labels to remove ".0"
                    tick_labels = [str(int(val)) for val in tick_values]
                    plt.xticks(tick_positions, tick_labels)  # Set x-axis ticks to formatted tick labels
                    
                    plt.colorbar(label='Intensity')
                    plt.tight_layout()
                    plt.show()
                    
#%%                    
#%%
                    
                    plt.figure()
                    for i in range(0, prof.shape[0], 50):
                        signal_data = prof[i, :]  # Extract the 1D signal from the current row
                        
                        plt.plot(time_values, signal_data, label=f"Row {i + 1}")  # Use time_values calculated previously
                        plt.xlabel('Time (s)')
                        plt.ylabel('Intensity')
                        plt.title("1D Signals from Every 50 Rows of waveform")
                        #plt.legend()
                        plt.grid()
                    
                    plt.tight_layout()
                    plt.show()
                    
                    selected_row = 968  # Change this to the desired row index

                    plt.figure(figsize=(14,8))  # Adjust the figure size as needed
                    
                    signal_data = prof[selected_row, :]  # Extract the 1D signal from the selected row
                    signal_data = prof[selected_row, :]
                    signal_list = list(signal_data)
                    with open('BSA_video_center.txt', 'w') as f:
                        for line in signal_list:
                            f.write(f"{line}\n")  
                        
                    plt.plot(time_values, signal_data)
                    plt.xlabel('Time (s)')
                    plt.ylabel('Intensity')
                    plt.grid()
                    
                    
                    plt.tight_layout()
                    plt.show()
                    
                else:
                    pass
                
        for matrix in profiles_list:
            for i in range(len(matrix)):
                for j in range(len(matrix[i])):
                    if matrix[i][j] < 100:
                        matrix[i][j] = 0
        def create_colormap(color):
            cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0, 0, 0, 0), color], N=100)
            return cmap
        
        # Colors for each matrix
        colors = ['red', 'green', 'blue', 'yellow']
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize = (15,4))
        
        # Plot each matrix with a different color
        for i, matrix in enumerate(profiles_list):
            cmap = create_colormap(colors[i])
            im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100)
        
        time_values = (frame_indices / fps)
        # Select tick positions at 50-second intervals
        tick_positions = np.arange(0, len(time_values), int(10 * fps))
        tick_values = np.round(time_values[tick_positions])  # Round to nearest whole number
        
        # Format tick labels to remove ".0"
        tick_labels = [str(int(val)) for val in tick_values]
        plt.xticks(tick_positions, tick_labels)  # Set x-axis ticks to formatted tick labels
        #plt.xlim([40, 60])
        plt.ylim([550, 725])
        plt.xlabel('Time (s)')  # Change x-axis label to Time
        plt.ylabel('Pixel')
        # Show the plot
        plt.tight_layout()
        plt.show()
            
    else:
        pass    

    
    return 



    
# Create a Tkinter root window
root = tk.Tk()
root.withdraw()


np.set_printoptions(threshold=sys.maxsize)
# Prompt the user to select the input video file
video_paths = filedialog.askopenfilenames(filetypes=[("Video Files", "*.mp4")])
if not video_paths:
    print("No video file selected.")


for video_path in video_paths:

    # Extract the directory and filename from the video path
    directory = os.path.dirname(video_path)
    filename = os.path.basename(video_path)

    
    
############################################# Video Information #######################################################################
    frame, fps, total_frames, center = video_info(video_path, run=True)
    print(fps, total_frames)
    center_coords_int = (int(center[0]), int(center[1]))
######################################################################################################################################################
    
######################################### Detect CHanges###########################################################################
    column_change = 1
######################################################################################################################################################


##################################### ANALYSIS ########################################################################################
    num_angle_increments = 1 # Specify the number of angle increments
    profiles, angles = image_profile(video_path, center_coords_int, num_angle_increments, total_frames, run = True)
    #print(profiles)
    visualize_data(profiles, angles, fps,column_change, plot=True, run=True)
    #needs to do this for avg rms
    profiles = np.array(profiles)
######################################################################################################################################################

