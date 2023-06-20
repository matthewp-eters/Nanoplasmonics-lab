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
    return 0, 0, 0

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
                    
                    
                    #time domain
                    fig = plt.figure(figsize=(10, 5))
                    
                    # Plot the time domain profile
                    plt.subplot(2,3,1)
                    plt.imshow(prof, cmap='plasma', interpolation='nearest', aspect='auto')
                    plt.title(f"Time Domain - Angle: {angles[i]}")
                    plt.xlabel('Frame Index')
                    plt.ylabel('Pixel Position')
                    plt.colorbar(label='Intensity')
                    # Adjust the layout to prevent overlapping of subplots
                    plt.tight_layout()
                    # Show the plot
                    plt.show()
                    
                    
                    #spectrogram
                    plt.subplot(2,3,2)
                    plt.imshow(fft_matrix_log, cmap='plasma', interpolation='nearest', aspect='auto')
                    plt.title(f"Freq Domain - Angle: {angles[i]}")
                    plt.xlabel('Frame Index')
                    plt.ylabel('Frequency')
                    plt.ylim(0, 200)  # Set the y-axis limits to 0 and 200 Hz
                    plt.colorbar(label='Log Intensity')  # Update the colorbar label
                    # Adjust the layout to prevent overlapping of subplots
                    plt.tight_layout()
                    # Show the plot
                    plt.show()
                    
                    plt.subplot(2,3,3)
                    prof=prof.transpose()
                    p=0
                    for col in prof:
                        if p < change_column_idx:
                            plt.plot(col, color="red", alpha=0.01)
                            p +=1
                        else:
                            plt.plot(col, color="blue", alpha=0.01 )
                            p += 1
                                
                    plt.title(f"Angle: {angles[i]}")
                    plt.xlabel('Pixel')
                    plt.ylabel('Intensity')
                    # Set custom colors for the legend
                    legend_labels = ["Untrapped", "Trapped"]
                    legend_colors = ['red', 'blue']
                    
                    legend_lines = [Line2D([0], [0], color=color, linewidth=1.5) for color in legend_colors]

                    # Create the legend with custom labels and colors
                    plt.legend(legend_lines, legend_labels, facecolor='white',loc='best', 
                               bbox_to_anchor=(1, 1), labelcolor='black')                   
                    plt.show()
                    
                    # Plot the averaged FFT plots
                    plt.subplot(2,3,4)
                    plt.plot(fft_before, label='Untrapped', color='red')
                    plt.plot(fft_after, label='Trapped', color='blue')
                    plt.title(f"Averaged 1D FFT - Angle: {angles[i]}")
                    plt.xlabel('Frequency')
                    plt.ylabel('Magnitude')
                    plt.xlim(8, 130)  # Set the y-axis limits to 0 and 200 Hz
                    plt.legend()
                    plt.show()
                    plt.tight_layout()
                    
                    plt.subplot(2,3,5)
                    plt.plot(fft_after-fft_before, color='black')
                    plt.title(f"FFT Difference - Angle: {angles[i]}")
                    plt.xlabel('Frequency')
                    plt.ylabel('Magnitude')
                    plt.xlim([0, 200])
                    plt.show()
                    plt.tight_layout()
                    
                    normalized_frame = last_frame.astype(float) / np.max(last_frame)

                    plt.subplot(2,3,6)
                    plt.imshow(normalized_frame)  # Convert BGR to RGB for matplotlib
                    plt.axis('off')  # Hide the axes
                    plt.show()
                    plt.tight_layout()
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

def compute_optical_flow(video_path, region_size, total_frames, col_idx, run=False):
    
    if run:
        cap = cv2.VideoCapture(video_path)
        
        # Read the first frame
        ret, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Get the center pixel coordinates
        height, width = prev_gray.shape
        center_x = width // 2
        center_y = height // 2
        
        matrix, angle = image_profile(video_path, (center_x, center_y), 1, total_frames, run=True)
              
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
            # Read the next frame
            ret, next_frame = cap.read()
            if not ret:
                break
            
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
            
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
        
        # Convert the full movements lists to numpy arrays
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
        
        
        # Calculate the offset to position the scatter plot at the center of the canvas
        offset_x = center_x #- region_size // 2
        offset_y = center_y #- region_size // 2
    
        # Plot the net movements on the canvas
        plt.figure()
        # Plot the net movements before the split index in red
        plt.scatter(net_movements_before[:, 0] + offset_x, net_movements_before[:, 1] + offset_y, color='red', alpha=0.25,label='Untrapped')
        # Plot the net movements after the split index in blue
        plt.scatter(net_movements_after[:, 0] + offset_x, net_movements_after[:, 1] + offset_y, color='blue', alpha=0.25,label='Trapped')
        # Overlay the scatter plot on the last frame of the video
        plt.imshow(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB))
        # Get the current reference
        ax = plt.gca()
        
        # Create a Rectangle patch
        rect = patches.Rectangle((center_x - region_size // 2,center_y - region_size // 2),region_size,region_size,linewidth=1,edgecolor='r',facecolor='none')
        
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.xlabel('Net Movement in X direction')
        plt.ylabel('Net Movement in Y direction')
        plt.title('Optical Flow: Net Movement Scatter Plot')
        plt.show()
        
        fig = plt.figure()
        
        ax = fig.add_subplot() 
        # square plot
        # Plot the net movements before the split index in red
        plt.scatter(net_movements_before[:, 0] , net_movements_before[:, 1], color='red', alpha=0.3,label='Untrapped')
        # Plot the net movements after the split index in blue
        plt.scatter(net_movements_after[:, 0], net_movements_after[:, 1], color='blue', alpha=0.3,label='Trapped')
        # Overlay the scatter plot on the last frame of the video
        plt.xlabel('Net Movement in X direction')
        plt.ylabel('Net Movement in Y direction')
        plt.title('Optical Flow: Net Movement Scatter Plot Center ROI')
        plt.grid()
        plt.legend()
        ax.set_aspect('equal', adjustable='box')
        plt.show()
        
        
        fig = plt.figure()
        
        ax = fig.add_subplot() 
        # square plot
        # Plot the net movements before the split index in red
        plt.scatter(full_movement_before[:, 0] , full_movement_before[:, 1], color='red', alpha=0.3,label='Untrapped')
        # Plot the net movements after the split index in blue
        plt.scatter(full_movement_after[:, 0], full_movement_after[:, 1], color='blue', alpha=0.3,label='Trapped')
        # Overlay the scatter plot on the last frame of the video
        plt.xlabel('Net Movement in X direction')
        plt.ylabel('Net Movement in Y direction')
        plt.title('Optical Flow: Net Movement Scatter Plot Full ROI')
        plt.grid()
        plt.legend()
        ax.set_aspect('equal', adjustable='box')
        plt.show()
        
        cap.release()
    else:
        pass
    

def main():
    # Create a Tkinter root window
    root = tk.Tk()
    root.withdraw()

    np.set_printoptions(threshold=sys.maxsize)
    # Prompt the user to select the input video file
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
    if not video_path:
        print("No video file selected.")
        return

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
    column_change = detect_change_column(video_path, center_coords_int, total_frames)
    column_change = 13*12
######################################################################################################################################################

    ######################## optical flow #####################################
    region_size = 50
    compute_optical_flow(video_path, region_size, total_frames,column_change, run=True)
######################################################################################################################################################
    
    
############################## Background removal ###################################################################################
    #GMM PARAMS
    variance = 40
    history = 100
    # Perform background subtraction and write the output video
    GMM_removal(video_path, output_path, variance, history, run=False)
    last_frame = average_background_removal(video_path, output_path, history, column_change, run=True)
######################################################################################################################################################


##################################### ANALYSIS ########################################################################################
    num_angle_increments = 4 # Specify the number of angle increments
    profiles, angles = image_profile(video_path, center_coords_int, num_angle_increments, total_frames, run = True)
    #print(profiles)
    visualize_data(profiles, angles, fps,column_change, last_frame, plot=True, run=True)
    #needs to do this for avg rms
    profiles = np.array(profiles)
######################################################################################################################################################




if __name__ == "__main__":
    main()
