#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:12:20 2023

@author: matthewpeters
"""
import matplotlib.pyplot as plt
import numpy as np

# Data sets
DATARMSD1 = [0.01836677321936761, 0.014969537198209714, 0.011062365428248644, 0.014969886488361176, 0.013421351652361605, 0.011601775199746986]

LASERRMSD1 = [0.006984500036178186,  0.007565061930774559,  0.00715897994636677, 0.007631665174533776, 0.009963277550937922, 0.010959648279343396]

DATARMSD2 = [0.010779967062681077, 0.012411465784235843, 0.00828409525464512,0.008473643567083208, 0.008771923686338664, 0.010209994551778297, 0.007667265651191853, 0.011693589742389293, 0.016061783607452133,  0.0060702312262221, 0.0143951556434199, 0.010087535456293149, 0.009818776631125122, 0.010331333480147926, 0.012641831993608797]

LASERRMSD2 = [0.002293618860665848,0.002293618860665848,0.002293618860665848,0.002293618860665848,0.002293618860665848,0.002293618860665848, 0.0017569184135447528, 0.0024634997787176846, 0.0024634997787176846, 0.002442085138465982, 0.0022678201494274177, 0.0021453541997454193, 0.005182302477818676, 0.005182302477818676, 0.005182302477818676]

DATARMSD3 = [0.004175747750790176, 0.004183531858948534, 0.009176191598764032, 0.004954191427130261, 0.004953707509969634, 0.005005176247478226, 0.011781928922431526, 0.009125618615383569, 0.007617045381743546, 0.007800227449602151, 0.0076697333436274015]

LASERRMSD3 = [0.002289226481794128, 0.0020660669935010163, 0.0021203952018481807, 0.002642978589641964, 0.0026073317834525017, 0.0027821391733247, 0.0029496642079130998, 0.002536589342045116, 0.0033260998716865132, 0.0033260998716865132, 0.0033260998716865132]


# Calculate the division results
DATARMSD1_divided = [rmsd - laser for rmsd, laser in zip(DATARMSD1, LASERRMSD1)]
DATARMSD2_divided = [rmsd - laser for rmsd, laser in zip(DATARMSD2, LASERRMSD2)]
DATARMSD3_divided = [rmsd - laser for rmsd, laser in zip(DATARMSD3, LASERRMSD3)]

# Calculate averages
DATA1AVG = np.mean(DATARMSD1_divided)
DATA2AVG = np.mean(DATARMSD2_divided)
DATA3AVG = np.mean(DATARMSD3_divided)

DATA1_x = 12
DATA2_x = 28
DATA3_x = 65

label1 = 'CTC'
label2 = 'CA'
label3 = 'BSA'


Averages = [DATA1AVG, DATA3AVG, DATA2AVG]
x_fit = [DATA1_x, DATA2_x, DATA3_x]

m, b = np.polyfit(x_fit, Averages, 1)

# Colors for each data set (with adjusted alpha)
dimmed_colors = ['green', 'blue', 'red']  # Dimmed colors for data points
avg_color = ['black', 'black', 'black']  # Opaque colors for average points

# X-axis points for each group
x_points_data1 = [DATA1_x] * len(DATARMSD1_divided)  # CTC mass in kDa
x_points_data2 = [DATA2_x] * len(DATARMSD2_divided)  # BSA mass in kDa
x_points_data3 = [DATA3_x] * len(DATARMSD3_divided)  # CA mass in kDa


#######################################################################################################################
# For label BSA
x_center_data3 = [3.3510206, 4.135784, 4.4335313, 3.4414983, 11.224236, 8.646858, 4.522126, 5.1918054, 3.731875]
y_center_data3 = [4.6267705, 4.0439115, 4.5731897, 2.6524134, 4.8658924, 3.9007914, 3.433044, 2.7269068, 2.8192644]
x_full_data3 = [0.7722608, 0.8046545, 0.6842221, 0.7653183, 1.2190813, 1.1727794, 1.9540932, 1.0465759, 1.4917953]
y_full_data3 = [0.37357682, 0.4371362, 0.34828484, 0.46672347, 0.8003492, 0.6031494, 1.3032647, 1.2153101, 1.4707481]

# For label CA
x_center_data2 = [16.12157, 9.832638, 9.158186, 11.991603]
y_center_data2 = [4.445848, 4.0589037, 6.3823414, 5.0372977]
x_full_data2 = [2.0943937, 1.2954388, 1.770933, 1.977694]
y_full_data2 = [1.4091578, 0.8251859, 5.7734585, 0.99627316]

# For label CTC
x_center_data1 = [9.424178, 8.231652, 9.88731, 13.9992, 5.9131403, 22.856026]
y_center_data1 = [6.452335, 5.338526, 6.3638716, 5.5284433, 3.8371532, 9.341224]
x_full_data1 = [1.9952048, 1.6462984, 1.6052369, 4.043391, 0.9238795, 1.8888711]
y_full_data1 = [2.8200831, 2.3253348, 1.6323985, 1.7524946, 0.9712976, 2.3635461]

# Calculate averages
DATA3_x_center_avg = np.mean(x_center_data3)
DATA3_y_center_avg = np.mean(y_center_data3)

DATA2_x_center_avg = np.mean(x_center_data2)
DATA2_y_center_avg = np.mean(y_center_data2)

DATA1_x_center_avg = np.mean(x_center_data1)
DATA1_y_center_avg = np.mean(y_center_data1)

DATA3_x_full_avg = np.mean(x_full_data3)
DATA3_y_full_avg = np.mean(y_full_data3)

DATA2_x_full_avg = np.mean(x_full_data2)
DATA2_y_full_avg = np.mean(y_full_data2)

DATA1_x_full_avg = np.mean(x_full_data1)
DATA1_y_full_avg = np.mean(y_full_data1)

AveragesX = [DATA1_x_center_avg, DATA2_x_center_avg, DATA3_x_center_avg]
x_fit = [DATA1_x, DATA2_x, DATA3_x]

mX,bX = np.polyfit(x_fit, AveragesX, 1)

# Averages
averagesY = [DATA1_y_center_avg, DATA2_y_center_avg, DATA3_y_center_avg]


mY, bY = np.polyfit(x_fit, averagesY, 1)

averagesXFull = [DATA1_x_full_avg, DATA2_x_full_avg, DATA3_x_full_avg]

mXF, bXF = np.polyfit(x_fit, averagesXFull, 1)
averagesYFull = [DATA1_y_full_avg, DATA2_y_full_avg, DATA3_y_full_avg]

mYF, bYF = np.polyfit(x_fit, averagesYFull, 1)

# Colors for each data set (with adjusted alpha)
dimmed_colors = ['green', 'blue', 'red']  # Dimmed colors for data points
avg_color = ['black', 'black', 'black']  # Opaque colors for average points

# X-axis points for each group
x_points_data1_fwhm = [DATA1_x] * len(x_center_data1)  # CTC mass in kDa
x_points_data2_fwhm = [DATA2_x] * len(x_center_data2)  # BSA mass in kDa
x_points_data3_fwhm = [DATA3_x] * len(x_center_data3)  # CA mass in kDa

#######################################################################################################
####################################################################################
# Plotting
fig, ax1 = plt.subplots()

# Plot dimmed data points
ax1.scatter(x_points_data1, DATARMSD1_divided, color=dimmed_colors[2], alpha=0.5, label=label1)
ax1.scatter(x_points_data2, DATARMSD2_divided, color=dimmed_colors[0], alpha=0.5, label=label2)
ax1.scatter(x_points_data3, DATARMSD3_divided, color=dimmed_colors[1], alpha=0.5, label=label3)

# Plot opaque average points
ax1.scatter(DATA1_x, DATA1AVG, color="black", alpha=1.0, marker='x')
ax1.scatter(DATA3_x, DATA2AVG, color="black", alpha=1.0, marker='x')
ax1.scatter(DATA2_x, DATA3AVG, color="black", alpha=1.0, marker='x')

# Plot line of best fit
#ax1.plot([0, 80], [b, m * 80 + b], color='black', linestyle='dashed') 

# Add legends and labels
ax1.legend()
ax1.set_xlabel('Mass (kDa)')
ax1.set_ylabel('NRMSD', color='black')
ax1.tick_params(axis='y', labelcolor='black')
# Plotting
# Plot dimmed data points

ax2 = ax1.twinx()
ax2.scatter(x_points_data1_fwhm, x_center_data1, color=dimmed_colors[2],marker="D", alpha=0.5, label=label1, facecolors='none')
ax2.scatter(x_points_data2_fwhm, x_center_data2, color=dimmed_colors[0], marker="D", alpha=0.5, label=label2, facecolors='none')
ax2.scatter(x_points_data3_fwhm, x_center_data3, color=dimmed_colors[1], marker="D",alpha=0.5, label=label3, facecolors='none')

# Plot opaque average points
ax2.scatter(DATA1_x, DATA1_x_center_avg, color="tab:purple", alpha=1.0, marker='x')
ax2.scatter(DATA2_x, DATA2_x_center_avg, color="tab:purple", alpha=1.0, marker='x')
ax2.scatter(DATA3_x, DATA3_x_center_avg, color="tab:purple", alpha=1.0, marker='x')

# Plot line of best fit
ax2.plot([0, 80], [bX, mX * 80 + bX], color='tab:purple', linestyle='dashed') 
ax2.set_ylabel('FWHM (X - Center)', color='tab:purple')
ax2.tick_params(axis='y', labelcolor='tab:purple')


# Show the plot
plt.show()
##############################################################################
# Plotting
fig, ax3 = plt.subplots()

# Plot dimmed data points
ax3.scatter(x_points_data1, DATARMSD1_divided, color=dimmed_colors[2], alpha=0.5, label=label1)
ax3.scatter(x_points_data2, DATARMSD2_divided, color=dimmed_colors[0], alpha=0.5, label=label2)
ax3.scatter(x_points_data3, DATARMSD3_divided, color=dimmed_colors[1], alpha=0.5, label=label3)

# Plot opaque average points
ax3.scatter(DATA1_x, DATA1AVG, color="black", alpha=1.0, marker='x')
ax3.scatter(DATA2_x, DATA2AVG, color="black", alpha=1.0, marker='x')
ax3.scatter(DATA3_x, DATA3AVG, color="black", alpha=1.0, marker='x')

# Plot line of best fit
ax3.plot([0, 80], [b, m * 80 + b], color='black', linestyle='dashed') 

# Add legends and labels
ax3.legend()
ax3.set_xlabel('Mass (kDa)')
ax3.set_ylabel('NRMSD', color='black')
ax3.tick_params(axis='y', labelcolor='black')

ax4 = ax3.twinx()
ax4.scatter(x_points_data1_fwhm, y_center_data1, color=dimmed_colors[2],marker="D", alpha=0.5, label=label1, facecolors='none')
ax4.scatter(x_points_data2_fwhm, y_center_data2, color=dimmed_colors[0], marker="D", alpha=0.5, label=label2, facecolors='none')
ax4.scatter(x_points_data3_fwhm, y_center_data3, color=dimmed_colors[1], marker="D",alpha=0.5, label=label3, facecolors='none')

# Plot opaque average points
ax4.scatter(DATA1_x, DATA1_y_center_avg, color="tab:purple", alpha=1.0, marker='x')
ax4.scatter(DATA2_x, DATA2_y_center_avg, color="tab:purple", alpha=1.0, marker='x')
ax4.scatter(DATA3_x, DATA3_y_center_avg, color="tab:purple", alpha=1.0, marker='x')

# Plot line of best fit
ax4.plot([0, 80], [bY, mY * 80 + bY], color='tab:purple', linestyle='dashed') 
ax4.set_ylabel('FWHM (Y - Center)', color='tab:purple')
ax4.tick_params(axis='y', labelcolor='tab:purple')
plt.show()

##############################################################################
# Plotting
fig, ax5 = plt.subplots()

# Plot dimmed data points
ax5.scatter(x_points_data1, DATARMSD1_divided, color=dimmed_colors[2], alpha=0.5, label=label1)
ax5.scatter(x_points_data2, DATARMSD2_divided, color=dimmed_colors[0], alpha=0.5, label=label2)
ax5.scatter(x_points_data3, DATARMSD3_divided, color=dimmed_colors[1], alpha=0.5, label=label3)

# Plot opaque average points
ax5.scatter(DATA1_x, DATA1AVG, color="black", alpha=1.0, marker='x')
ax5.scatter(DATA2_x, DATA2AVG, color="black", alpha=1.0, marker='x')
ax5.scatter(DATA3_x, DATA3AVG, color="black", alpha=1.0, marker='x')

# Plot line of best fit
ax5.plot([0, 80], [b, m * 80 + b], color='black', linestyle='dashed') 

# Add legends and labels
ax5.legend()
ax5.set_xlabel('Mass (kDa)')
ax5.set_ylabel('NRMSD', color='black')
ax5.tick_params(axis='y', labelcolor='black')

ax6 = ax5.twinx()
ax6.scatter(x_points_data1_fwhm, y_full_data1, color=dimmed_colors[2],marker="D", alpha=0.5, label=label1, facecolors='none')
ax6.scatter(x_points_data2_fwhm, y_full_data2, color=dimmed_colors[0], marker="D", alpha=0.5, label=label2, facecolors='none')
ax6.scatter(x_points_data3_fwhm, y_full_data3, color=dimmed_colors[1], marker="D",alpha=0.5, label=label3, facecolors='none')

# Plot opaque average points
ax6.scatter(DATA1_x, DATA1_y_full_avg, color="tab:purple", alpha=1.0, marker='x')
ax6.scatter(DATA2_x, DATA2_y_full_avg, color="tab:purple", alpha=1.0, marker='x')
ax6.scatter(DATA3_x, DATA3_y_full_avg, color="tab:purple", alpha=1.0, marker='x')

# Plot line of best fit
ax6.plot([0, 80], [bYF, mYF * 80 + bYF], color='tab:purple', linestyle='dashed') 
ax6.set_ylabel('FWHM (Y - Full)', color='tab:purple')
ax6.tick_params(axis='y', labelcolor='tab:purple')
plt.show()

##############################################################################
# Plotting
fig, ax7 = plt.subplots()

# Plot dimmed data points
ax7.scatter(x_points_data1, DATARMSD1_divided, color=dimmed_colors[2], alpha=0.5, label=label1)
ax7.scatter(x_points_data2, DATARMSD2_divided, color=dimmed_colors[0], alpha=0.5, label=label2)
ax7.scatter(x_points_data3, DATARMSD3_divided, color=dimmed_colors[1], alpha=0.5, label=label3)

# Plot opaque average points
ax7.scatter(DATA1_x, DATA1AVG, color="black", alpha=1.0, marker='x')
ax7.scatter(DATA2_x, DATA3AVG, color="black", alpha=1.0, marker='x')
ax7.scatter(DATA3_x, DATA3AVG, color="black", alpha=1.0, marker='x')

# Plot line of best fit
ax7.plot([0, 80], [b, m * 80 + b], color='black', linestyle='dashed') 

# Add legends and labels
ax7.legend()
ax7.set_xlabel('Mass (kDa)')
ax7.set_ylabel('NRMSD', color='black')
ax7.tick_params(axis='y', labelcolor='black')

ax8 = ax7.twinx()
ax8.scatter(x_points_data1_fwhm, x_full_data1, color=dimmed_colors[2],marker="D", alpha=0.5, label=label1, facecolors='none')
ax8.scatter(x_points_data2_fwhm, x_full_data2, color=dimmed_colors[0], marker="D", alpha=0.5, label=label2, facecolors='none')
ax8.scatter(x_points_data3_fwhm, x_full_data3, color=dimmed_colors[1], marker="D",alpha=0.5, label=label3, facecolors='none')

# Plot opaque average points
ax8.scatter(DATA1_x, DATA1_x_full_avg, color="tab:purple", alpha=1.0, marker='x')
ax8.scatter(DATA2_x, DATA2_x_full_avg, color="tab:purple", alpha=1.0, marker='x')
ax8.scatter(DATA3_x, DATA3_x_full_avg, color="tab:purple", alpha=1.0, marker='x')

# Plot line of best fit
ax8.plot([0, 80], [bXF, mXF * 80 + bXF], color='tab:purple', linestyle='dashed') 
ax8.set_ylabel('FWHM (X - Full)', color='tab:purple')
ax8.tick_params(axis='y', labelcolor='tab:purple')
plt.show()

##############################################################################
# Plotting
fig, ax9 = plt.subplots()

# Plot dimmed data points
ax9.scatter(x_points_data1, DATARMSD1_divided, color=dimmed_colors[2], alpha=0.5, label=label1)
ax9.scatter(x_points_data2, DATARMSD2_divided, color=dimmed_colors[0], alpha=0.5, label=label2)
ax9.scatter(x_points_data3, DATARMSD3_divided, color=dimmed_colors[1], alpha=0.5, label=label3)

# Plot opaque average points
ax9.scatter(DATA1_x, DATA1AVG, color="black", alpha=1.0, marker='x')
ax9.scatter(DATA2_x, DATA2AVG, color="black", alpha=1.0, marker='x')
ax9.scatter(DATA3_x, DATA3AVG, color="black", alpha=1.0, marker='x')

# Plot line of best fit
ax9.plot([0, 80], [b, m * 80 + b], color='black', linestyle='dashed') 

# Add legends and labels
ax9.legend()
ax9.set_xlabel('Mass (kDa)')
ax9.set_ylabel('NRMSD', color='black')
ax9.tick_params(axis='y', labelcolor='black')


