#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:35:52 2023

@author: matthewpeters
"""
import glob
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16) 
plt.rc('axes', labelsize = 18)
plt.rc('legend', fontsize = 18)
plt.rc('font', family='sans-serif')

# Data sets
data3 = [82.58*0.0465, 275.603*0.0465, 272.27*0.0465]#,441.03, 414.62]
data2 = [305.06*0.0465, 283.82*0.0465]
data1 = [487.36*0.0465, 596.06*0.0465,325.04*0.0465, 310.43*0.0465]

label1 = 'CTC'
label2 = 'CA'
label3 = 'BSA'
label4 = 'NPY'

#data1 = [29.75, 45.5, 36.4, 33.1, 40.33, 34.39,43.72]
#data2= [64.3, 59.54, 52.1, 51.11, 64.33, 59.48,73.48, 74.03,75.24]
#data3 = [158.73, 155.72, 101.28, 91.88, 107.2 ]
#data4 = [17.3]
# Calculate averages
avgData1 = np.mean(data1)
avgData2 = np.mean(data2)
avgData3 = np.mean(data3)
#avgData4 = np.mean(data4)


Averages = [ avgData1, avgData2, avgData3]

#Averages = [avgData4, avgData1, avgData2, avgData3]

x_point1 = 14.3
x_point2 = 29.2
x_point3 = 66.4
x_point4= 4.3

x_fit = [x_point1, x_point2, x_point3]

#x_fit = [x_point1, x_point2, x_point3, x_point4]

m,b = np.polyfit(x_fit, Averages, 1)


# Calculate time constants
tcData1 = 1 / (2 * np.pi * np.array(data1))
tcData2 = 1 / (2 * np.pi * np.array(data2))
tcData3 = 1 / (2 * np.pi * np.array(data3))
#tcData4 = 1 / (2 * np.pi * np.array(data4))

# Calculate average time constants
avg_tcData1 = np.mean(tcData1)
avg_tcData2 = np.mean(tcData2)
avg_tcData3 = np.mean(tcData3)
#avg_tcData4 = np.mean(tcData4)

# Averages
averagesTC = [avg_tcData1, avg_tcData2, avg_tcData3]

#averagesTC = [avg_tcData4, avg_tcData1, avg_tcData2, avg_tcData3]
x_fit = [x_point1, x_point2, x_point3]

#x_fit = [x_point4, x_point1, x_point2, x_point3]

mTC, bTC = np.polyfit(x_fit, averagesTC, 1)


# Colors for each data set (with adjusted alpha)
dimmed_colors = ['green', 'blue', 'red']  # Dimmed colors for data points

#dimmed_colors = ['orange', 'green', 'blue', 'red']  # Dimmed colors for data points
avg_color = ['black', 'black', 'black', 'black']  # Opaque colors for average points

# X-axis points for each group
x_length_data1 = [x_point1] * len(data1)  # BSA mass in kDa
x_length_data2 = [x_point2] * len(data2)  # CA mass in kDa
x_length_data3 = [x_point3] * len(data3)  # CTC mass in kDa
#x_length_data4 = [x_point4 * len(data4)]


# # Create a 2x2 grid of subplots
# fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# # Panel 1: Corner Frequency - Dimmed Data Points and Average Points
# axs[0, 0].scatter(x_length_data1, data1, color=dimmed_colors[0], alpha=0.5, label=label1)
# axs[0, 0].scatter(x_length_data2, data2, color=dimmed_colors[1], alpha=0.5, label=label2)
# axs[0, 0].scatter(x_length_data3, data3, color=dimmed_colors[2], alpha=0.5, label=label3)
# axs[0, 0].scatter(x_point1, avgData1, color=avg_color[0], alpha=1.0, marker='x')
# axs[0, 0].scatter(x_point2, avgData2, color=avg_color[1], alpha=1.0, marker='x')
# axs[0, 0].scatter(x_point3, avgData3, color=avg_color[2], alpha=1.0, marker='x')
# axs[0, 0].plot([0, 120], [b, m * 130 + b], color='k', linestyle='dashed')
# axs[0, 0].set_xlim([0, 80])
# axs[0, 0].set_ylim([0, 175])
# axs[0, 0].set_xlabel('Mass (kDa)')
# axs[0, 0].set_ylabel('Corner Frequency')
# axs[0, 0].legend()

# # Panel 2: Time Constant - Dimmed Data Points and Average Points
# axs[0, 1].scatter(x_length_data1, tcData1, color=dimmed_colors[0], alpha=0.5, label=label1)
# axs[0, 1].scatter(x_length_data2, tcData2, color=dimmed_colors[1], alpha=0.5, label=label2)
# axs[0, 1].scatter(x_length_data3, tcData3, color=dimmed_colors[2], alpha=0.5, label=label3)
# axs[0, 1].scatter(x_point1, avg_tcData1, color=avg_color[0], alpha=1.0, marker='x')
# axs[0, 1].scatter(x_point2, avg_tcData2, color=avg_color[1], alpha=1.0, marker='x')
# axs[0, 1].scatter(x_point3, avg_tcData3, color=avg_color[2], alpha=1.0, marker='x')
# axs[0, 1].plot(x_fit, averagesTC, marker='o', color='k', markersize=1, linestyle='dashed')
# axs[0, 1].set_xlim([0, 80])
# axs[0, 1].set_ylim([0, 0.006])
# axs[0, 1].set_xlabel('Mass (kDa)')
# axs[0, 1].set_ylabel('Time Constant')
# axs[0, 1].legend()

# # Panel 3: Corner Frequency - Boxplot
# data = [data1, data2, data3]
# positions = [x_point1, x_point2, x_point3]
# bp1 = axs[1, 0].boxplot(data, positions=positions, widths=1)

# for i in range(0, len(bp1['boxes'])):
#     bp1['boxes'][i].set_color(dimmed_colors[i])
#     bp1['whiskers'][i*2].set_color(dimmed_colors[i])
#     bp1['whiskers'][i*2 + 1].set_color(dimmed_colors[i])
#     bp1['whiskers'][i*2].set_linewidth(1)
#     bp1['whiskers'][i*2 + 1].set_linewidth(1)
#     bp1['fliers'][i].set(markerfacecolor=dimmed_colors[i], marker='o', alpha=0.75, markersize=6, markeredgecolor='none')
#     bp1['medians'][i].set_color('black')
#     bp1['medians'][i].set_linewidth(1)
#     for c in bp1['caps']:
#         c.set_linewidth(0)
        
# axs[1, 0].legend([bp1["boxes"][0], bp1["boxes"][1], bp1["boxes"][2]], [label1, label2, label3], loc='upper left')      
# axs[1, 0].set_xticks(positions)
# axs[1, 0].plot([0, 120], [b, m * 130 + b], color='k', linestyle='dashed') 
# #axs[1, 0].set_ylim([0, 175])
# axs[1, 0].set_xlim([10, 70])
# axs[1, 0].set_xlabel("Mass (kDa)")
# axs[1, 0].set_ylabel("Corner Frequency")

# # Panel 4: Time Constant - Boxplot
# dataTC = [tcData1, tcData2, tcData3]
# bp2 = axs[1, 1].boxplot(dataTC, positions=positions, widths=1)

# for i in range(0, len(bp2['boxes'])):
#     bp2['boxes'][i].set_color(dimmed_colors[i])
#     bp2['whiskers'][i*2].set_color(dimmed_colors[i])
#     bp2['whiskers'][i*2 + 1].set_color(dimmed_colors[i])
#     bp2['whiskers'][i*2].set_linewidth(1)
#     bp2['whiskers'][i*2 + 1].set_linewidth(1)
#     bp2['fliers'][i].set(markerfacecolor=dimmed_colors[i], marker='o', alpha=0.75, markersize=6, markeredgecolor='none')
#     bp2['medians'][i].set_color('black')
#     bp2['medians'][i].set_linewidth(1)
#     for c in bp2['caps']:
#         c.set_linewidth(0)

# axs[1, 1].legend([bp2["boxes"][0], bp2["boxes"][1], bp2["boxes"][2]], [label1, label2, label3], loc='upper right')      
# axs[1, 1].set_xticks(positions)
# axs[1, 1].plot(positions, averagesTC, marker='o', color='k', markersize=0, linestyle='dashed')
# axs[1, 1].set_ylim([0, 0.006])
# axs[1, 1].set_xlim([10, 70])
# axs[1, 1].set_xlabel("Mass (kDa)")
# axs[1, 1].set_ylabel("Time Constant")

# # Adjust layout and spacing
# plt.tight_layout()

# # Show the figure with all panels
# plt.show()















# # Plotting
# plt.figure()

# # Plot dimmed data points
# plt.scatter(x_length_data1, data1, color=dimmed_colors[0], alpha=0.5, label=label1)
# plt.scatter(x_length_data2, data2, color=dimmed_colors[1], alpha=0.5, label=label2)
# plt.scatter(x_length_data3, data3, color=dimmed_colors[2], alpha=0.5, label=label3)

# # Plot opaque average points
# plt.scatter(x_point1, avgData1, color=avg_color[0], alpha=1.0, marker='x')
# plt.scatter(x_point2, avgData2, color=avg_color[1], alpha=1.0, marker='x')
# plt.scatter(x_point3, avgData3, color=avg_color[2], alpha=1.0, marker='x')

# # Plot line of best fit
# plt.plot([0, 120], [b, m * 130 + b], color='gray', linestyle='dashed') 



# # Add legends and labels
# plt.xlim([0, 80])
# plt.ylim([0, 175])
# #plt.xscale('log')
# #plt.yscale('log')
# plt.legend()
# plt.xlabel('Mass (kDa)')
# plt.ylabel('Corner Frequency')

# # Show the plot
# plt.show()

# # Plotting
# plt.figure()

# # Plot dimmed data points
# plt.scatter(x_length_data1, tcData1, color=dimmed_colors[0], alpha=0.5, label=label1)
# plt.scatter(x_length_data2, tcData2, color=dimmed_colors[1], alpha=0.5, label=label2)
# plt.scatter(x_length_data3, tcData3, color=dimmed_colors[2], alpha=0.5, label=label3)

# # Plot opaque average points
# plt.scatter(x_point1, avg_tcData1, color=avg_color[0], alpha=1.0, marker='x')
# plt.scatter(x_point2, avg_tcData2, color=avg_color[1], alpha=1.0, marker='x')
# plt.scatter(x_point3, avg_tcData3, color=avg_color[2], alpha=1.0, marker='x')

# plt.plot(x_fit, averagesTC, marker='o', color='red', markersize=1, linestyle='dashed')

# # Plot line of best fit
# #plt.plot([0, 120], [bTC, mTC * 130 + bTC], color='gray', linestyle='dashed') 


# # Add legends and labels
# plt.xlim([0, 80])
# plt.ylim([0, 0.006])
# #plt.xscale('log')
# #plt.yscale('log')
# plt.legend()
# plt.xlabel('Mass (kDa)')
# plt.ylabel('Time Constant')

# # Show the plot
# plt.show()


data = [data1, data2, data3]
positions = [x_point1, x_point2, x_point3]

# fig7, ax7 = plt.subplots()

# bp1 = ax7.boxplot(data, positions=positions, widths =1)

# for i in range(0, len(bp1['boxes'])):
#     bp1['boxes'][i].set_color(dimmed_colors[i])
#     # we have two whiskers!
#     bp1['whiskers'][i*2].set_color(dimmed_colors[i])
#     bp1['whiskers'][i*2 + 1].set_color(dimmed_colors[i])
#     bp1['whiskers'][i*2].set_linewidth(1)
#     bp1['whiskers'][i*2 + 1].set_linewidth(1)
#     # fliers
#     # (set allows us to set many parameters at once)
#     bp1['fliers'][i].set(markerfacecolor=dimmed_colors[i],
#                     marker='o', alpha=0.75, markersize=6,
#                     markeredgecolor='none')
#     bp1['medians'][i].set_color('black')
#     bp1['medians'][i].set_linewidth(1)
#     # and 4 caps to remove
#     for c in bp1['caps']:
#         c.set_linewidth(0)

# ax7.legend([bp1["boxes"][0], bp1["boxes"][1], bp1["boxes"][2]], [label1,label2,label3], loc='upper left')      

# # Set the x-axis labels
# ax7.set_xticks(positions)
# #ax7.set_xticklabels([x_point1, x_point2, x_point3])
# plt.plot([0, 120], [b, m * 130 + b], color='gray', linestyle='dashed') 
# #plt.plot(positions, Averages, marker='x', color='k', markersize=5, linestyle='dashed')

# plt.ylim([0, 175])
# plt.xlim([10, 70])
# plt.xlabel("Mass (kDa)")
# plt.ylabel("Corner Frequency (Hz)")
# plt.savefig("CornerFreq.pdf", format="pdf", bbox_inches="tight")

# plt.show()

#dataTC = [tcData4, tcData1, tcData2, tcData3]
#data = [data1, data2, data3, data4]


d2 = 1.004
d_CTC = 1.41 + 0.145*exp(-1*x_point1/13)
d_CA = 1.41 + 0.145*exp(-1*x_point2/13)
d_BSA = 1.41 + 0.145*exp(-1*x_point3/13)

v_CTC = (2/9)*((d_CTC-d2)*9.8*(1.7E-9)**2)/6.53E-4
v_CA = (2/9)*((d_CA-d2)*9.8*(2.01E-9)**2)/6.53E-4
v_BSA = (2/9)*((d_BSA-d2)*9.8*(3.48E-9)**2)/6.53E-4


D_CTC = 1 / (6*np.pi*6.53E-4 * 1.7E-9 *v_CTC)
D_CA = 1 / (6*np.pi*6.53E-4 * 2.01E-9 *v_CA)
D_BSA = 1 / (6*np.pi*6.53E-4 * 3.48E-9 *v_BSA)


your_points_data = [D_CTC, D_CA, D_BSA]

fig8 = plt.figure()

ax8 = fig8.add_subplot(1,1,1)

bp = ax8.boxplot(data, positions=positions, widths = 3)

for i in range(0, len(bp['boxes'])):
    bp['boxes'][i].set_color(dimmed_colors[i])
    bp['boxes'][i].set_linewidth(4)
    # we have two whiskers!
    bp['whiskers'][i*2].set_color(dimmed_colors[i])
    bp['whiskers'][i*2 + 1].set_color(dimmed_colors[i])
    bp['whiskers'][i*2].set_linewidth(3)
    bp['whiskers'][i*2 + 1].set_linewidth(3)
    # fliers
    # (set allows us to set many parameters at once)
    bp['fliers'][i].set(markerfacecolor=dimmed_colors[i],
                    marker='o', alpha=0.75, markersize=10,
                    markeredgecolor='none')
    bp['medians'][i].set_color('black')
    bp['medians'][i].set_linewidth(3)
    # and 4 caps to remove
    for c in bp['caps']:
        c.set_linewidth(0)
       
ax8.legend([bp["boxes"][0], bp["boxes"][1], bp["boxes"][2]], [label1,label2,label3], loc='upper right')      
# Set the x-axis labels
ax8.set_xticks(positions)
#ax7.set_xticklabels([x_point1, x_point2, x_point3])
plt.plot(positions, Averages, marker='o', color='k', markersize=0, linestyle='dashed')
#plt.plot([0, 120], [bTC, mTC * 130 + bTC], color='gray', linestyle='dashed')
ax8.set_ylabel("Velocity ($\mu m$/s)") 
ax8.set_ylim([0, 30])
plt.xlim([10, 70])
ax8.set_xlabel("Mass (kDa)")

# Create a twin Axes for the second y-axis
ax8_2 = ax8.twinx()

# Plot points on the second y-axis
ax8_2.plot(positions, your_points_data, marker='o', color='red', markersize=6, linestyle='dashed')

# Set labels and limits for the second y-axis
ax8_2.set_ylabel("Calculated Drag Force (kg $m^{-1} s^{-1}$)", color='red')  # Set label color to red
ax8_2.tick_params(axis='y', labelcolor='red')
#ax8_2.set_ylim([D_BSA/1.25, D_CTC*1.25])
ax8_2.set_ylim([0, 1.5E25])




#ax8.ylabel("Time Constant (s)")
#plt.savefig("Diffusion.pdf", format="pdf", bbox_inches="tight")
plt.show()
