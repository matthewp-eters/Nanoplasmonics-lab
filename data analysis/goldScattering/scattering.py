import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize

# Path to directories containing gold and glass images
gold_dir = '/Users/matthewpeters/Desktop/AstroDMx_DATA/2023-08-10/Snapshot/gold'
glass_dir = '/Users/matthewpeters/Desktop/AstroDMx_DATA/2023-08-10/Snapshot/glass'

# Get lists of image filenames for gold and glass sets
gold_files = [f for f in os.listdir(gold_dir) if f.startswith('gold')]
glass_files = [f for f in os.listdir(glass_dir) if f.startswith('glass')]

num_channels = 3  # RGB channels
color_labels = ['Red', 'Green', 'Blue']
common_image_size = (256, 256)  # Common size for resizing

def extract_middle_line(image):
    middle_row = image[image.shape[0] // 2, :, :]
    return middle_row

def plot_average_line_profiles(image_files, title_prefix):
    line_profiles = np.zeros((len(image_files), num_channels, common_image_size[1]), dtype=np.float)

    for i, file in enumerate(image_files):
        image = imread(file)
        image_resized = resize(image, common_image_size, mode='reflect', anti_aliasing=True)
        for channel in range(num_channels):
            line_profiles[i, channel] = extract_middle_line(image_resized)[:, channel]

    return np.mean(line_profiles, axis=0)

# Calculate the average line profiles for gold and glass images
avg_line_profiles_gold = plot_average_line_profiles([os.path.join(gold_dir, f) for f in gold_files], 'Gold')
avg_line_profiles_glass = plot_average_line_profiles([os.path.join(glass_dir, f) for f in glass_files], 'Glass')

# Plotting and Normalizing
plt.figure(figsize=(10, 8))

# Set the y-axis limits for all subplots
y_min, y_max = 0.8, 1.05

for channel in range(num_channels):
    plt.subplot(num_channels, 1, channel + 1)
    plt.title(f'{color_labels[channel]} Channel')
    plt.xlabel('pixel')

    # Set y-axis limits
    plt.ylim(y_min, y_max)

    # Find the first intensity value
    first_intensity_gold = avg_line_profiles_gold[channel, 0]
    first_intensity_glass = avg_line_profiles_glass[channel, 0]

    # Normalize the line profiles by dividing by the first intensity value
    normalized_intensity_gold = avg_line_profiles_gold[channel] / first_intensity_gold
    normalized_intensity_glass = avg_line_profiles_glass[channel] / first_intensity_glass

    # Define shifted_pixel based on min_index_gold (you can calculate min_index_glass similarly)
    min_index_gold = np.argmin(avg_line_profiles_gold[channel])
    shifted_pixel = np.arange(common_image_size[1]) - min_index_gold

    plt.plot(shifted_pixel, normalized_intensity_gold, label='Gold', color='gold', linewidth=2)
    plt.plot(shifted_pixel, normalized_intensity_glass, label='Glass', color='blue', linewidth=2)
    plt.grid()
    if channel == num_channels - 1:
        plt.ylabel('Normalized Intensity')  # Set y-axis label for the last subplot
        plt.legend()

plt.tight_layout()
plt.savefig("normalized_channels.pdf", format="pdf", bbox_inches="tight")
plt.show()
