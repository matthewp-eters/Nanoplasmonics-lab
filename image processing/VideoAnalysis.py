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
plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16) 
plt.rc('axes', labelsize=18) 
plt.rc('font', family='sans-serif')


def GMM_removal(video_path, output_path, variance, history, run=False):
    
    if run:
        # Load the video
        video = cv2.VideoCapture(video_path)
    
        # Get the video's width, height, and frames per second (fps)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
        # Create a VideoWriter object to write the output video
        output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    
        # Create a background subtractor using Gaussian Mixture Model
        background_subtractor = cv2.createBackgroundSubtractorMOG2(history, variance)
    
        while True:
            # Read a frame from the video
            ret, frame = video.read()
            if not ret:
                break
    
            # Apply background subtraction
            foreground_mask = background_subtractor.apply(frame)
    
            # Write the frame with the foreground mask to the output video
            output.write(foreground_mask)
    
            # Display the foreground mask
            cv2.imshow('Foreground Mask', foreground_mask)
    
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        # Release the video, output video, and destroy windows
        video.release()
        output.release()
        cv2.destroyAllWindows()
    else:
        pass

def average_background_removal(video_path, output_path, history, column_change, run=False):
    
    if run:
        
        # Load the video
        video_source = cv2.VideoCapture(video_path)
        
        # Step 1: Accumulate frames for background generation
        num_background_frames = column_change  # Specify the number of frames to use for background generation
        background_frames = []
        frame_count = 0
        
        while frame_count < num_background_frames:
            success, frame = video_source.read()
            if not success:
                break
            background_frames.append(frame)
            frame_count += 1
        
        # Step 2: Calculate the average background
        average_background = np.mean(background_frames, axis=0).astype(np.uint8)
        
        # Step 3: Apply background subtraction to all frames and convert to black and white
        video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset the video capture to the beginning
        
        # Get video properties
        fps = video_source.get(cv2.CAP_PROP_FPS)
        frame_size = (int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT)))
       
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # Create VideoWriter object to save the processed video
        output_video = cv2.VideoWriter(output_path, fourcc, fps, frame_size, True)
        
        while True:
            success, frame = video_source.read()
            if not success:
                break
        

            # Subtract the average background from each frame
            subtracted_frame = cv2.absdiff(frame, average_background)
        
            # Convert the frame and subtracted_frame to black and white (grayscale)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_subtracted_frame = cv2.cvtColor(subtracted_frame, cv2.COLOR_BGR2GRAY)
        
            # Horizontally stack the frames
            hstacked_frames = np.hstack((gray_frame, gray_subtracted_frame))
            # Convert back to BGR for writing
            hstacked_frames_bgr = cv2.cvtColor(hstacked_frames, cv2.COLOR_GRAY2BGR)
        
            last_frame = gray_subtracted_frame

            cv2.imshow("Original vs Subtracted", hstacked_frames)
        
            # Write the processed frame to the output video file
            output_video.write(gray_subtracted_frame)
            
        
            if cv2.waitKey(30) == ord("q"):
                 break
        
        # Release video source and output video
        video_source.release()
        output_video.release()
        
        cv2.destroyAllWindows()
        return last_frame
    else:
        pass      
    
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
            frame_profile_matrix = np.zeros((total_frames, 1280))
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

def visualize_data(profiles, angles, fps, col_change, last_frame, plot=False, run=False):
    
    if run:
        
        rms_values_list = []  # List to store RMS values for each column
        avg_rms_before_list = []  # List to store average RMS before values
        avg_rms_after_list = []  # List to store average RMS after values
        for i, prof in enumerate(profiles):
            
            prof = prof.transpose()
            
            # Calculate RMS of each column
            rms_values = np.sqrt(np.mean(np.square(prof), axis=0))
            
            # Append RMS values to the list
            rms_values_list.append(rms_values.tolist())
            #print(rms_values)
    
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
                
                
                # Average the RMS values before and after the change index
                avg_rms_before = calculate_rms(rms_values[:change_column_idx], 5)
                avg_rms_after = calculate_rms(rms_values[change_column_idx:], 5)
                
                avg_rms_before_list.append(avg_rms_before)
                avg_rms_after_list.append(avg_rms_after)
            
                
    
                if plot:
                    # fig = plt.figure(figsize=(10, 5))
                    
                    # # Time Domain Plot (Subplot 1)
                    # plt.subplot(1, 2, 1)
                    # plt.imshow(prof, cmap='coolwarm', interpolation='nearest', aspect='auto')
                    # plt.title(f"Time Domain - Angle: {angles[i]}")
                    # plt.xlabel('Time (s)')  # Change x-axis label to Time
                    # plt.ylabel('Pixel')
                    # plt.ylim([400, 850])
                    
                    # # Calculate time values for x-axis based on frame index and fps
                    # frame_indices = np.arange(prof.shape[1])
                    # time_values = frame_indices / fps
                    # # Select tick positions at 50-second intervals
                    # tick_positions = np.arange(0, len(time_values), int(2 * fps))
                    # tick_values = np.round(time_values[tick_positions])  # Round to nearest whole number
                    
                    # # Format tick labels to remove ".0"
                    # tick_labels = [str(int(val)) for val in tick_values]
                    # plt.xticks(tick_positions, tick_labels)  # Set x-axis ticks to formatted tick labels
                    
                    # plt.colorbar(label='Intensity')
                    # plt.tight_layout()
                    
                    # # Spectrogram Plot (Subplot 2)
                    # plt.subplot(1, 2, 2)
                    # plt.imshow(fft_matrix_log, cmap='coolwarm', interpolation='nearest', aspect='auto')
                    # plt.title(f"Freq Domain - Angle: {angles[i]}")
                    # plt.xlabel('Time (s)')  # Change x-axis label to Time
                    # plt.ylabel('Frequency (Hz)')
                    # plt.ylim(0, 100)
                    
                    # # Calculate time values for x-axis based on frame index and fps
                    # frame_indices = np.arange(fft_matrix_log.shape[1])
                    # time_values = frame_indices / fps
                    # # Select tick positions at 50-second intervals
                    # tick_positions = np.arange(0, len(time_values), int(5 * fps))
                    # tick_values = np.round(time_values[tick_positions])  # Round to nearest whole number
                    
                    # # Format tick labels to remove ".0"
                    # tick_labels = [str(int(val)) for val in tick_values]
                    # plt.xticks(tick_positions, tick_labels)  # Set x-axis ticks to formatted tick labels
                    
                    # plt.colorbar(label='Log Intensity')
                    # plt.tight_layout()
                    
                    # plt.show()

                    plt.figure()
                    plt.imshow(prof, cmap='coolwarm', interpolation='nearest', aspect='auto')
                    plt.title(f"Time Domain - Angle: {angles[i]}")
                    plt.xlabel('Time (s)')  # Change x-axis label to Time
                    plt.ylabel('Pixel')
                    plt.ylim([400, 850])
                    
                    # Calculate time values for x-axis based on frame index and fps
                    frame_indices = np.arange(prof.shape[1])
                    time_values = frame_indices / fps
                    # Select tick positions at 50-second intervals
                    tick_positions = np.arange(0, len(time_values), int(1 * fps))
                    tick_values = np.round(time_values[tick_positions])  # Round to nearest whole number
                    
                    # Format tick labels to remove ".0"
                    tick_labels = [str(int(val)) for val in tick_values]
                    plt.xticks(tick_positions, tick_labels)  # Set x-axis ticks to formatted tick labels
                    
                    plt.colorbar(label='Intensity')
                    plt.tight_layout()
                    plt.show()
                    
                    
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
                    
                    selected_row = 465  # Change this to the desired row index

                    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
                    
                    signal_data = prof[selected_row, :]  # Extract the 1D signal from the selected row
                    signal_data = prof[selected_row, :]
                    signal_list = list(signal_data)
                    print(signal_list)
                    
                    plt.plot(time_values, signal_data)
                    plt.xlabel('Time (s)')
                    plt.ylabel('Intensity')
                    plt.title(f"1D Signal from Row {selected_row} of prof")
                    plt.grid()
                    
                    plt.tight_layout()
                    plt.show()
                    




                    
                    # plt.subplot(2,3,3)
                    # prof=prof.transpose()
                    # p=0
                    # for col in prof:
                    #     if p < change_column_idx:
                    #         plt.plot(col, color="red", alpha=0.01)
                    #         p +=1
                    #     else:
                    #         plt.plot(col, color="blue", alpha=0.01 )
                    #         p += 1
                                
                    # plt.title(f"Angle: {angles[i]}")
                    # plt.xlabel('Pixel')
                    # plt.ylabel('Intensity')
                    # # Set custom colors for the legend
                    # legend_labels = ["Untrapped", "Trapped"]
                    # legend_colors = ['red', 'blue']
                    
                    # legend_lines = [Line2D([0], [0], color=color, linewidth=1.5) for color in legend_colors]

                    # # Create the legend with custom labels and colors
                    # plt.legend(legend_lines, legend_labels, facecolor='white',loc='best', 
                    #            bbox_to_anchor=(1, 1), labelcolor='black')                   
                    # plt.show()
                    
                    # # Plot the averaged FFT plots
                    # plt.subplot(2,3,4)
                    # plt.plot(fft_before, label='Untrapped', color='red')
                    # plt.plot(fft_after, label='Trapped', color='blue')
                    # plt.title(f"Averaged 1D FFT - Angle: {angles[i]}")
                    # plt.xlabel('Frequency')
                    # plt.ylabel('Magnitude')
                    # plt.xlim(8, 130)  # Set the y-axis limits to 0 and 200 Hz
                    # plt.legend()
                    # plt.show()
                    # plt.tight_layout()
                    
                    # plt.subplot(2,3,5)
                    # plt.plot(fft_after-fft_before, color='black')
                    # plt.title(f"FFT Difference - Angle: {angles[i]}")
                    # plt.xlabel('Frequency')
                    # plt.ylabel('Magnitude')
                    # plt.xlim([0, 200])
                    # plt.show()
                    # plt.tight_layout()
                    
                    # normalized_frame = last_frame.astype(float) / np.max(last_frame)

                    # plt.subplot(2,3,6)
                    # plt.imshow(normalized_frame)  # Convert BGR to RGB for matplotlib
                    # plt.axis('off')  # Hide the axes
                    # plt.show()
                    # plt.tight_layout()
                else:
                    pass
                
        # # Create scatter plot for Average RMS Before
        # plt.figure(figsize=(8, 6))
        # plt.scatter(angles, avg_rms_before_list, c='blue', label='Untrapped')
        # plt.scatter(angles, avg_rms_after_list, c='red', label='Trapped')
        # plt.title("Average RMS")
        # plt.xlabel("Angle")
        # plt.ylabel("Average RMS")
        # plt.legend()
        # plt.show()
    else:
        pass    

    
    return 

def detect_change_column(video_path, center_coords, total_frames):
    
    profiles, angles = image_profile(video_path, center_coords, 1, total_frames, run=True)
    
    
    num_cols = len(profiles[0])
    #print(num_cols)
    profiles = np.array(profiles)
    noise_range = slice(center_coords[0]-50, center_coords[0]+50)  # Rows range to calculate noise change

    row_std = np.std(profiles[0, :, noise_range], axis=1)
    prev_std = row_std[0]  # Initialize the previous standard deviation
    for i in range(1, len(row_std)):
        curr_std = row_std[i]
    
        if np.abs(curr_std - prev_std) >= 0.1 * prev_std:
            print("Row index:", i)
            return i
            # Perform additional actions if needed
    
        prev_std = curr_std
    return None

def calculate_rms(signal, window_length):
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

    return average_rms

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    

    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
def compute_optical_flow(video_path, region_size, total_frames, col_idx,filename, plot=False, run=False):
    
    if run:
        cap = cv2.VideoCapture(video_path)
        
        # Read the first frame to get its dimensions
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            return 0, 0
        
        height, width, _ = first_frame.shape
        
        # Calculate the new width to make the video square
        new_width = height
        
        # Calculate the cropping region to keep the center point fixed
        crop_start = (width - new_width) // 2
        crop_end = crop_start + new_width
        
        # Read the first frame again to start processing
        cap.release()
        cap = cv2.VideoCapture(video_path)
        ret, prev_frame = cap.read()
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
        prev_frame = prev_frame[:, crop_start:crop_end, :]
        prev_frame = cv2.resize(prev_frame, (new_width, height))
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        
        # Get the center pixel coordinates
        height, width = prev_gray.shape
        center_x = width // 2
        center_y = height // 2
        
        #matrix, angle = image_profile(video_path, (center_x, center_y), 1, total_frames, run=True)
              
        # Initialize an empty array to store net movements
        net_movements = []
        # Initialize empty lists to store net movements before and after the split index
        net_movements_before = []
        net_movements_after = []
        
        full_movement_before = []
        full_movement_after = []
        full_movement_intermediate = []
        full_movements = []
        
        while True:
            ret, next_frame = cap.read()
            if not ret:
                break
            next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)
            next_frame = next_frame[:, crop_start:crop_end, :]
            next_frame = cv2.resize(next_frame, (new_width, height))
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
            
            # Compute optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Get the optical flow vectors within the region of interest
            flow_roi = flow[center_y - region_size // 2:center_y + region_size // 2,
                            center_x - region_size // 2:center_x + region_size // 2]
            
            # Compute the net movement in the region
            net_movement = np.sum(flow_roi, axis=(0, 1))
            
            # Store the net movement for the region
            net_movements.append(net_movement)
            
            # Determine whether to store the net movement in the before or after list
            if len(net_movements_before) < col_idx:
                net_movements_before.append(net_movement)
            else:
                net_movements_after.append(net_movement)
                
            full_movement = np.sum(flow, axis=(0,1))
            full_movements.append(full_movement)
            
            if len(full_movement_before) < col_idx:
                full_movement_before.append(full_movement)
            else:
                full_movement_after.append(full_movement)
            
            # Update the previous frame and previous gray image
            prev_frame = next_frame
            prev_gray = next_gray
        
        

        # Convert the net movements lists to numpy arrays
        net_movements_before = np.array(net_movements_before)
        net_movements_after = np.array(net_movements_after)
        
        #print("np array", net_movements_after)
        # # Convert the full movements lists to numpy arrays
        full_movement_before = np.array(full_movement_before)
        full_movement_after = np.array(full_movement_after)
        
        # Normalize the net movements based on the region size
        net_movements_before /= region_size
        net_movements_after /= region_size
        
        
        # Normalize the net movements based on the region size
        full_movement_before /= width
        full_movement_after /= width
        
        #print("Full movement", full_movements)
        #print("ROI movement", net_movements)
        
        r_center, p_center = stats.pearsonr(net_movements_after[:, 0], net_movements_after[:, 1])
        r_full, p_full = stats.pearsonr(full_movement_after[:, 0], full_movement_after[:, 1])
        
        # Calculate the offset to position the scatter plot at the center of the canvas
        offset_x = center_x #- region_size // 2
        offset_y = center_y #- region_size // 2
    
        
        if plot:
            

            
            # Define the linear function
            def linear_func(x, m, c):
                return m * x + c
            
            # Fit the data to the linear function
            params, _ = curve_fit(linear_func, net_movements_after[:, 0], net_movements_after[:, 1])
            
            # Get the slope and intercept
            slope = params[0]
            intercept = params[1]
            
            # Calculate the angle
            angle_rad = np.arctan(slope)
            angle_deg = np.degrees(angle_rad)
            angle_deg = '%.3f'%(angle_deg)

            
            fig = plt.figure(figsize = (10,10))
            r_center = '%.3f'%(r_center)
            ax = fig.add_subplot() 
            # square plot
            # Plot the net movements after the split index in blue
            plt.scatter(net_movements_after[:, 0], net_movements_after[:, 1], color='blue', alpha=0.3)
            # Overlay the scatter plot on the last frame of the video
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f"Optical Flow: Center ROI {filename}")
            plt.grid()
            plt.xlim([-175, 175])
            plt.ylim([-75, 75])
            annotation_text = f"r = {r_center}, Θ = {angle_deg}"
            annotation_box_props = dict(boxstyle='round', facecolor='white', edgecolor='black')
            
            plt.annotate(annotation_text, xy=(-150, 55), weight='bold', bbox=annotation_box_props)
            
            #plt.legend()
            ax.set_aspect('equal', adjustable='box')
            plt.savefig(filename + 'center.pdf', format="pdf", bbox_inches="tight")

            plt.show()
            
            
            r_full = '%.3f'%(r_full)
            fig = plt.figure(figsize=(10, 10))
            
            # Fit the data to the linear function
            params2, _ = curve_fit(linear_func, full_movement_after[:, 0], full_movement_after[:, 1])
            
            # Get the slope and intercept
            slope2 = params2[0]
            intercept2 = params2[1]
            
            # Calculate the angle
            angle_rad2 = np.arctan(slope2)
            angle_deg2 = np.degrees(angle_rad2)
            angle_deg2 = '%.3f'%(angle_deg2)
            
            ax = fig.add_subplot()
            # square plot
            # Plot the net movements after the split index in blue
            plt.scatter(full_movement_after[:, 0], full_movement_after[:, 1], color='blue', alpha=0.3)
            # Overlay the scatter plot on the last frame of the video
            plt.xlabel('X ')
            plt.ylabel('Y ')
            plt.title(f"Optical Flow: Full ROI {filename}")
            plt.grid()
            plt.xlim([-30, 30])
            plt.ylim([-30, 30])
            
            annotation_text = f"r = {r_full}, Θ = {angle_deg2}"
            annotation_box_props = dict(boxstyle='round', facecolor='white', edgecolor='black')
            
            plt.annotate(annotation_text, xy=(-25, 25), weight='bold', bbox=annotation_box_props)
            
            ax.set_aspect('equal', adjustable='box')
            plt.savefig(filename + 'full.pdf', format="pdf", bbox_inches="tight")

            plt.show()
            
            
            cap.release()
        else:
            pass
        
        return net_movements_after, full_movement_after
    else:
        return 0, 0
    

from matplotlib.gridspec import GridSpec

def hist(x, y, filename, toggle=False):
    
    # Create histograms
    bins = 10  # Number of bins for the histograms
    
    plt.figure()
    x_hist, x_bins, _ = plt.hist(x, bins=bins, alpha=0.5, label='X')
    
    # Histogram for y data
    y_hist, y_bins, _ = plt.hist(y, bins=bins, alpha=0.5, label='Y')
    plt.close()
    
    # Calculate FWHM for x data
    x_peak = x_bins[np.argmax(x_hist)]  # Find the peak bin
    x_half_max = np.max(x_hist) / 2  # Calculate half of the maximum bin value
    x_left_idx = np.where(x_hist[:np.argmax(x_hist)] <= x_half_max)[0][-1]  # Left index of half-max bin
    x_right_idx = np.where(x_hist[np.argmax(x_hist):] <= x_half_max)[0][0] + np.argmax(x_hist)  # Right index of half-max bin
    x_fwhm = x_bins[x_right_idx] - x_bins[x_left_idx]  # FWHM for x data
    
    # Calculate FWHM for y data
    y_peak = y_bins[np.argmax(y_hist)]  # Find the peak bin
    y_half_max = np.max(y_hist) / 2  # Calculate half of the maximum bin value
    y_left_idx = np.where(y_hist[:np.argmax(y_hist)] <= y_half_max)[0][-1]  # Left index of half-max bin
    y_right_idx = np.where(y_hist[np.argmax(y_hist):] <= y_half_max)[0][0] + np.argmax(y_hist)  # Right index of half-max bin
    y_fwhm = y_bins[y_right_idx] - y_bins[y_left_idx]  # FWHM for y data

    
    if toggle:
        #plt.savefig(filename + '_plot.png')
        plt.show()
        # Print the FWHM values
        print("FWHM for X data center:", x_fwhm, ' ', filename)
        print("FWHM for Y data center:", y_fwhm, ' ', filename)
    else:
        plt.show()
        # Print the FWHM values
        print("FWHM for X data full:", x_fwhm, ' ', filename)
        print("FWHM for Y data full:", y_fwhm, ' ', filename)



    
# Create a Tkinter root window
root = tk.Tk()
root.withdraw()

center_trap = []
full_trap = []
names = []

dfs = []

bsa_pearson_center_r = []
ca_pearson_center_r = []
ctc_pearson_center_r = []

bsa_pearson_full_r = []
ca_pearson_full_r = []
ctc_pearson_full_r = []

bsa_pearson_center_p = []
ca_pearson_center_p = []
ctc_pearson_center_p = []

bsa_pearson_full_p = []
ca_pearson_full_p = []
ctc_pearson_full_p = []


    
def scatter_hist(x, y, ax, ax_histx, ax_histy, filename):
    ax_histx.tick_params(axis='x', labelbottom=False)
    ax_histy.tick_params(axis='y', labelleft=False)
    
    scatter_color = 'blue'  # Set the scatter plot color
    
    ax.scatter(x, y, color=scatter_color, alpha=0.3)
    
    ax.set_xlim([-110, 110])
    ax.set_ylim([-110, 110])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    ax.grid(True)
    
    binwidth = 2
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth
    
    bins = np.arange(-lim, lim + binwidth, binwidth)
    x_hist, _ = np.histogram(x, bins=bins)
    y_hist, _ = np.histogram(y, bins=bins)
    
    max_bin_count = 100  # Normalize to a maximum of 100
    
    # Normalize histograms
    x_hist_norm = (x_hist / np.max(x_hist)) * max_bin_count
    y_hist_norm = (y_hist / np.max(y_hist)) * max_bin_count
    
    ax_histx.bar(bins[:-1], x_hist_norm, width=binwidth, color=scatter_color, alpha=0.7)  # Use the same color for histogram
    
    # Set the same y-axis limits for both histograms
    ax_histx.set_ylim([0, max(ax_histx.get_ylim()[1], ax_histy.get_ylim()[1])])
    
    ax_histy.barh(bins[:-1], y_hist_norm, height=binwidth, color=scatter_color, alpha=0.7)  # Use the same color for histogram

    # Save the figure as PDF
    plt.savefig(filename + '_hist.pdf', format="pdf", bbox_inches="tight")
    plt.close()
    
def scatter_hist_2(x, y, ax, ax_histx, ax_histy, filename):
    ax_histx.tick_params(axis='x', labelbottom=False)
    ax_histy.tick_params(axis='y', labelleft=False)
    
    scatter_color = 'black'  # Set the scatter plot color
    
    ax.scatter(x, y, color=scatter_color, alpha=0.3)
    
    ax.set_xlim([-30, 30])
    ax.set_ylim([-30, 30])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    ax.grid(True)
    
    binwidth = 2
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth
    
    bins = np.arange(-lim, lim + binwidth, binwidth)
    x_hist, _ = np.histogram(x, bins=bins)
    y_hist, _ = np.histogram(y, bins=bins)
    
    max_bin_count = 100  # Normalize to a maximum of 100
    
    # Normalize histograms
    x_hist_norm = (x_hist / np.max(x_hist)) * max_bin_count
    y_hist_norm = (y_hist / np.max(y_hist)) * max_bin_count
    
    ax_histx.bar(bins[:-1], x_hist_norm, width=binwidth, color=scatter_color, alpha=0.7)  # Use the same color for histogram
    
    # Set the same y-axis limits for both histograms
    ax_histx.set_ylim([0, max(ax_histx.get_ylim()[1], ax_histy.get_ylim()[1])])
    
    ax_histy.barh(bins[:-1], y_hist_norm, height=binwidth, color=scatter_color, alpha=0.7)  # Use the same color for histogram

    # Save the figure as PDF
    plt.savefig(filename + '_hist.pdf', format="pdf", bbox_inches="tight")
    plt.close()

    

np.set_printoptions(threshold=sys.maxsize)
# Prompt the user to select the input video file
video_paths = filedialog.askopenfilenames(filetypes=[("Video Files", "*.mp4")])
if not video_paths:
    print("No video file selected.")


for video_path in video_paths:

    # Extract the directory and filename from the video path
    directory = os.path.dirname(video_path)
    filename = os.path.basename(video_path)

    # Generate the output video path with a modified filename
    output_filename = os.path.splitext(filename)[0] + "_removed.mp4"
    output_path = os.path.join(directory, output_filename)
    
    
############################################# Video Information #######################################################################
    frame, fps, total_frames, center = video_info(video_path, run=True)
    center_coords_int = (int(center[0]), int(center[1]))
######################################################################################################################################################
    
######################################### Detect CHanges###########################################################################
    #column_change = detect_change_column(video_path, center_coords_int, total_frames)
    column_change = 1
######################################################################################################################################################

    ######################## optical flow #####################################
    region_size = 50
    net_movement_after, full_movement_after = compute_optical_flow(video_path, region_size, total_frames,column_change, filename, run=False)
    #hist(net_movement_after[:, 0], net_movement_after[:, 1], filename, toggle = False)
          
            
    # fig = plt.figure(layout='constrained', figsize = (8,8))
    
    # ax = fig.add_gridspec(top=0.75, right = 0.75).subplots()
    
    # ax.set(aspect=1)
    
    # ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    # ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    
    # scatter_hist(net_movement_after[:, 0], net_movement_after[:, 1], ax, ax_histx, ax_histy, filename+'center')
    
    # fig = plt.figure(layout='constrained', figsize=(8,8))
    
    # ax = fig.add_gridspec(top=0.75, right = 0.75).subplots()
    
    # ax.set(aspect=1)
    
    # ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    # ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    # scatter_hist_2(full_movement_after[:, 0], full_movement_after[:, 1], ax, ax_histx, ax_histy, filename+'full')
        
    
    #hist(full_movement_after[:, 0], full_movement_after[:, 1], filename, toggle = False)
######################################################################################################################################################

        
############################## Background removal ###################################################################################
    #GMM PARAMS
    variance = 40
    history = 100
    # Perform background subtraction and write the output video
    GMM_removal(video_path, output_path, variance, history, run=False)
    last_frame = average_background_removal(video_path, output_path, history, column_change, run=False)
######################################################################################################################################################


##################################### ANALYSIS ########################################################################################
    num_angle_increments = 1 # Specify the number of angle increments
    profiles, angles = image_profile(video_path, center_coords_int, num_angle_increments, total_frames, run = True)
    #print(profiles)
    visualize_data(profiles, angles, fps,column_change, last_frame, plot=True, run=True)
    #needs to do this for avg rms
    profiles = np.array(profiles)
######################################################################################################################################################
#print(center_trap)

#     r_center, p_center = stats.pearsonr(net_movement_after[:, 0], net_movement_after[:, 1])
#     r_full, p_full = stats.pearsonr(full_movement_after[:, 0], full_movement_after[:, 1])

    

    # Remove ".mp4" extension
    filename = re.sub(r"\.mp4$", "", filename)
    # Remove numbers and dashes
    filename = re.sub(r"[\d-]", "", filename)  
    filename = re.sub(r"_AFTER", "", filename)
    filename = re.sub(r"_trap", "", filename)
    filename = re.sub(r"_TRAP", "", filename)

    
#     if filename == 'BSA':
#         bsa_pearson_center_r.append(r_center)
#         bsa_pearson_full_r.append(r_full)
        
#         bsa_pearson_center_p.append(p_center)
#         bsa_pearson_full_p.append(p_full)
        
#     elif filename == 'CA':
#         ca_pearson_center_r.append(r_center)
#         ca_pearson_full_r.append(r_full)
        
#         ca_pearson_center_p.append(p_center)
#         ca_pearson_full_p.append(p_full)
#     elif filename == 'CTC':
#         ctc_pearson_center_r.append(r_center)
#         ctc_pearson_full_r.append(r_full)
        
#         ctc_pearson_center_p.append(p_center)
#         ctc_pearson_full_p.append(p_full)

    
    center_trap.append(net_movement_after)
    full_trap.append(full_movement_after)
    
    
    # Create a DataFrame for the current file's scatter plot data
    scatter_data = {
        'x': net_movement_after[:, 0],
        'y': net_movement_after[:, 1],
        'full_x': full_movement_after[:,0],
        'full_y': full_movement_after[:,1],
        'protein': filename
    }
    df = pd.DataFrame(scatter_data)

    # Append the DataFrame to the list
    dfs.append(df)
    
    
#     # hist(net_movement_after[:, 0], net_movement_after[:, 1], filename, toggle = True)
#     # hist(full_movement_after[:, 0], full_movement_after[:, 1], filename, toggle = False)

# bsa_r_avg_center = np.mean(bsa_pearson_center_r)
# bsa_p_avg_center = np.mean(bsa_pearson_center_p)

# ca_r_avg_center = np.mean(ca_pearson_center_r)
# ca_p_avg_center = np.mean(ca_pearson_center_p)

# ctc_r_avg_center = np.mean(ctc_pearson_center_r)
# ctc_p_avg_center = np.mean(ctc_pearson_center_p)

# bsa_r_avg_full = np.mean(bsa_pearson_full_r)
# bsa_p_avg_full = np.mean(bsa_pearson_full_p)

# ca_r_avg_full = np.mean(ca_pearson_full_r)
# ca_p_avg_full = np.mean(ca_pearson_full_p)

# ctc_r_avg_full = np.mean(ctc_pearson_full_r)
# ctc_p_avg_full = np.mean(ctc_pearson_full_p)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)
# Assuming combined_df is the pandas DataFrame containing the combined scatter plot data
# g = sns.jointplot(data=combined_df, x='x', y='y', hue='protein', kind='scatter', palette='Set1', alpha=0.1)
# g.ax_joint.annotate(f'$r = {bsa_r_avg_center:.3f}, p = {bsa_p_avg_center:.3f}$',
#                     xy=(0.05, 0.83), xycoords='axes fraction',
#                     ha='left', va='center',
#                     bbox={'boxstyle': 'round', 'fc': '#0cdc73', 'ec': '#048243'})
# g.ax_joint.annotate(f'$r = {ca_r_avg_center:.3f}, p = {ca_p_avg_center:.3f}$',
#                     xy=(0.05, 0.89), xycoords='axes fraction',
#                     ha='left', va='center',
#                     bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})
# g.ax_joint.annotate(f'$r = {ctc_r_avg_center:.3f}, p = {ctc_p_avg_center:.3f}$',
#                     xy=(0.05, 0.95), xycoords='axes fraction',
#                     ha='left', va='center',
#                     bbox={'boxstyle': 'round', 'fc': '#fc5a50', 'ec': '#9a0200'})

sns.jointplot(data=combined_df, x='x', y='y', hue='protein', kind='kde', palette='Set1')
sns.jointplot(data=combined_df, x='full_x', y='full_y', hue='protein', kind='kde', palette='Set1')

# # Show the plot
#sns.jointplot(data=combined_df, x='x', y='y', hue='protein', kind='hist', palette='Set1')
# # Show the plot

# # sns.violinplot(data=combined_df, x='x', y='protein', palette='Set1', bw=10, alpha=0.1)
# # sns.violinplot(data=combined_df, x='y', y='protein', palette='Set1', bw=10)
# # sns.violinplot(data=combined_df, x='full_x', y='protein', palette='Set1', bw=10)
# # sns.violinplot(data=combined_df, x='full_y', y='protein', palette='Set1', bw=10)

# # sns.kdeplot(data=combined_df, x='x', hue='protein', palette='Set1', fill=True, bw_adjust=10, common_norm=False, alpha=0.5, linewidth=0.5)
# # sns.kdeplot(data=combined_df, x='y', hue='protein', palette='Set1', fill=True, bw_adjust=10, common_norm=False, alpha=0.5, linewidth=0.5)
# # sns.kdeplot(data=combined_df, x='full_x', hue='protein', palette='Set1', fill=True, bw_adjust=10, common_norm=False, alpha=0.5, linewidth=0.5)
# # sns.kdeplot(data=combined_df, x='full_y', hue='protein', palette='Set1', fill=True, bw_adjust=10, common_norm=False, alpha=0.5, linewidth=0.5)

# j = sns.jointplot(data=combined_df, x='full_x', y='full_y', hue='protein', kind='scatter', palette='Set1', alpha=0.1)
# j.ax_joint.annotate(f'$r = {bsa_r_avg_full:.3f}, p = {bsa_p_avg_full:.3f}$',
#                     xy=(0.05, 0.83), xycoords='axes fraction',
#                     ha='left', va='center',
#                     bbox={'boxstyle': 'round', 'fc': '#0cdc73', 'ec': '#048243'})
# j.ax_joint.annotate(f'$r = {ca_r_avg_full:.3f}, p = {ca_p_avg_full:.3f}$',
#                     xy=(0.05, 0.89), xycoords='axes fraction',
#                     ha='left', va='center',
#                     bbox={'boxstyle': 'round', 'fc': 'powderblue', 'ec': 'navy'})
# j.ax_joint.annotate(f'$r = {ctc_r_avg_full:.3f}, p = {ctc_p_avg_full:.3f}$',
#                     xy=(0.05, 0.95), xycoords='axes fraction',
#                     ha='left', va='center',
#                     bbox={'boxstyle': 'round', 'fc': '#fc5a50', 'ec': '#9a0200'})

# #sns.jointplot(data=combined_df, x='full_x', y='full_y', hue='protein', kind='kde', palette='Set1')
# # Show the plot
# #sns.jointplot(data=combined_df, x='full_x', y='full_y', hue='protein', kind='hist', palette='Set1')
# # Show the plot
