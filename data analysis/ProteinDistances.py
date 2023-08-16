import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.special import erf
from scipy.stats import norm
from scipy.stats import rayleigh


plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16) 
plt.rc('axes', labelsize = 18)
plt.rc('legend', fontsize = 18)
plt.rc('font', family='sans-serif')

def distance(x1, y1, x2, y2):
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

def probability(x, t, D):
    return 2 * x * np.exp(-x**2 / (4 * D * t)) / (4 * D * t)

# Sample data for BSA, CA, and CTC categories
categories = {
    'BSA': [
        {'time': 13.27, 'positions': [(470, 668), (395, 596), (383, 579), (394, 550), (406, 598), (428, 602), (444, 584), (451, 634), (412, 599), (456, 627), (486, 582), (442, 564), (437, 580), (472, 555), (468, 582), (429, 559), (428, 558), (411, 599), (438, 599), (409, 562), (444, 522), (495, 592), (486, 656), (464, 637), (445, 564), (469, 636), (589, 486), (590, 492), (622, 510), (572, 479), (593, 495), (616, 500), (631, 402), (646, 370), (643, 377), (613, 365), (650, 348)]},
        {'time': 11.42, 'positions': [(433, 613), (483, 645), (486, 621), (461, 587), (487, 557), (466, 508), (476, 536), (450, 373), (472, 400), (454, 447), (585, 336), (595, 353), (595, 333), (610, 500), (638, 363)]},
        {'time': 7.57, 'positions': [(371, 461), (398, 323), (437, 298), (554, 465), (571, 476), (639, 360), (650, 368), (651, 369), (637, 350), (629, 357), (633, 362), (629, 383), (630, 381)]},
        {'time': 6.3, 'positions': [(578, 568), (621, 395), (625, 397), (619, 440), (680, 431), (648, 381), (630, 379), (608, 346), (601, 342), (625, 347), (599, 337), (618, 362), (600, 337), (613, 353), (613, 349), (594, 343), (604, 360), (574, 389), (554, 313), (555, 316), (559, 389), (611, 298), (652, 323), (642, 340), (625, 350), (597, 333), (589, 344), (600, 325), (596, 325), (596, 328), (614, 366), (636, 320), (662, 366), (655, 307), (641, 361), (643, 334), (665, 375), (605, 335), (625, 336), (636, 356), (641, 359), (649, 362), (673, 351), (710, 313), (711, 313), (645, 361), (636, 350), (634, 365), (648, 370), (648, 371), (657, 371)]},
        {'time': 3.93, 'positions': [(588, 696), (541, 681), (557, 643), (543, 632), (617, 652), (623, 644), (594, 689), (562, 676), (579, 655), (568, 648), (582, 650), (579, 644), (587, 642), (487, 618), (505, 502), (453, 426), (492, 542), (566, 486), (575, 489), (591, 494), (572, 486), (548, 468), (606, 395), (624, 406)]}
    ],
    'CA': [
        {'time': 6.67, 'positions': [(378, 676), (339, 644), (368, 667), (351, 630), (329, 631), (289, 584), (394, 659), (387, 615), (399, 633), (408, 639), (421, 659), (390, 639), (392, 630), (387, 590), (389, 597), (412, 572), (436, 573), (435, 654), (444, 644), (699, 699), (417, 541), (441, 588), (529, 679), (548, 652), (565, 547), (507, 568), (635, 316), (645, 412), (636, 466), (616, 455), (609, 460), (629, 417), (639, 415), (642, 364)]},
        {'time': 4.2, 'positions': [(812, 384), (842, 325), (855, 391), (864, 380), (885, 376), (848, 419), (841, 440), (841, 426), (867, 476), (801, 478), (833, 441), (801, 483), (808, 369), (796, 367), (814, 377), (806, 402), (724, 391), (732, 382), (649, 374), (689, 358), (701, 360), (650, 357), (658, 340), (668, 363), (644, 361), (642, 359), (633, 379), (648, 359)]},
        {'time': 3.77, 'positions': [(392, 579), (451, 635), (452, 624), (520, 628), (467, 642), (508, 696), (498, 691), (517, 669), (441, 622), (523, 625), (507, 613), (529, 580), (534, 584), (545, 590), (587, 573), (562, 575), (569, 532), (569, 563), (589, 576), (610, 569), (565, 565), (668, 605), (626, 589), (632, 405)]},
        {'time': 7, 'positions': [(661, 362), (648, 313), (609, 343), (633, 391), (636, 382), (638, 395), (629, 397), (622, 401), (635, 396), (564, 390), (637, 395), (628, 350), (594, 349), (636, 402), (635, 401), (633, 399), (649, 325), (648, 326), (702, 364), (703, 363), (833, 329), (841, 335), (729, 342), (723, 352), (697, 371), (682, 362), (725, 358), (628, 396), (625, 398), (630, 405), (635, 392), (592, 352), (594, 344), (598, 344), (600, 348)]}
    ],
    'CTC': [
        {'time': 5.3, 'positions': [(578, 697), (599, 702), (615, 691), (601, 700), (632, 705), (575, 681), (609, 684), (564, 690), (544, 675), (536, 679), (559, 655), (574, 649), (536, 663), (578, 634), (608, 632), (623, 642), (597, 626), (579, 626), (613, 634), (588, 643), (620, 635), (598, 644), (611, 643), (590, 613), (580, 647), (574, 634), (607, 622), (627, 633), (590, 633), (605, 626), (592, 638), (671, 652), (639, 647), (607, 643), (589, 599), (589, 659), (550, 601), (562, 619), (556, 625), (570, 640), (648, 599), (563, 592), (557, 607), (553, 590), (526, 587), (578, 510), (519, 519), (533, 414), (586, 460), (634, 369), (632, 370)]},
        {'time': 1.5, 'positions': [(827, 438), (803, 479), (758, 449), (733, 469), (763, 496), (746, 464), (768, 504), (728, 460), (725, 460), (658, 460), (649, 460), (639, 455), (671, 405), (614, 458), (680, 269), (588, 389)]},
        {'time': 2, 'positions': [(640, 155), (681, 144), (692, 155), (664, 156), (654, 173), (534, 218), (518, 201), (510, 198), (477, 279), (515, 211), (637, 334), (561, 244), (642, 187), (586, 191), (614, 247), (613, 219), (607, 216)]},
        {'time': 7.07, 'positions': [(280, 505), (240, 439), (288, 308), (285, 316), (352, 287), (356, 272), (379, 261), (350, 285), (349, 288), (350, 322), (348, 343), (357, 336), (348, 352), (339, 354), (359, 416), (343, 375), (383, 491), (375, 461), (382, 491), (413, 543), (414, 538), (384, 533), (410, 534), (386, 523), (385, 515), (392, 503), (426, 539), (411, 499), (418, 466), (423, 483), (387, 478), (409, 505), (407, 517), (411, 522), (450, 512), (445, 539), (537, 607), (472, 583), (559, 622), (532, 442), (655, 505), (526, 433), (629, 479), (632, 458), (625, 455), (629, 439), (630, 410), (644, 363), (642, 347), (640, 360), (640, 380), (640, 381), (642, 357)]},
        {'time': 8, 'positions': [(954, 354), (951, 357), (963, 375), (906, 377), (942, 427), (923, 412), (932, 409), (953, 391), (906, 395), (947, 373), (950, 330), (949, 341), (937, 441), (945, 390), (896, 469), (899, 391), (927, 426), (901, 387), (901, 391), (892, 406), (861, 422), (880, 433), (846, 464), (847, 456), (865, 447), (880, 454), (852, 419), (851, 413), (852, 389), (813, 473), (832, 457), (839, 428), (854, 353), (727, 374), (793, 512), (698, 365), (712, 462), (740, 434), (642, 380), (660, 323), (713, 460), (649, 314), (712, 354), (698, 360), (641, 340), (649, 363)]},
        {'time': 3, 'positions': [(438, 209), (429, 284), (429, 296), (446, 257), (463, 225), (559, 240), (567, 305), (635, 408), (638, 316), (658, 314), (635, 406), (665, 356), (651, 367), (630, 366), (599, 355), (660, 363), (668, 367), (696, 356), (693, 378), (693, 376), (696, 378), (693, 361), (692, 371), (686, 364), (648, 313), (649, 341), (609, 361), (616, 360), (585, 365), (622, 354), (619, 406), (637, 360), (630, 408), (645, 362), (659, 363), (644, 335), (633, 387), (638, 335), (644, 326), (650, 325), (645, 358), (645, 324), (643, 331), (642, 337)]}
    ]
}

colormap = {
    'BSA': 'red',
    'CA': 'blue',
    'CTC': 'green'
}


# Diffusion coefficients for each category
diffusion_coefficients = {
    'BSA': (1.380649E-23*300) / (6*np.pi*6.53E-4 * 1.7E-9) ,
    'CA': (1.380649E-23*300)/ (6*np.pi*6.53E-4 * 2.01E-9),
    'CTC': (1.380649E-23*300) / (6*np.pi*6.53E-4 * 3.48E-9)
}



# Multiply all data points in positions by 0.0465
for category, datasets in categories.items():
    for dataset in datasets:
        for i in range(len(dataset['positions'])):
            dataset['positions'][i] = (dataset['positions'][i][0] * 0.0465, dataset['positions'][i][1] * 0.0465)

# Plotting
plt.figure(figsize=(10, 6))

legend_added = {}  # Track if legend has been added for each category

for category, datasets in categories.items():
    for dataset in datasets:
        protein_distances = []
        time = dataset['time']  # Get the time value from the dataset
        for i in range(1, len(dataset['positions'])):
            x1, y1 = dataset['positions'][i-1][0], dataset['positions'][i-1][1]
            x2, y2 = dataset['positions'][i][0], dataset['positions'][i][1]
            dist = distance(x1, y1, x2, y2)
            protein_distances.append(dist)
        
        cumulative_distances = [0] + [sum(protein_distances[:i+1]) for i in range(len(protein_distances))]
        
        # Adjust the time values to match the length of cumulative_distances
        times = np.linspace(0, time, num=len(cumulative_distances))
        
        plt.plot(times, cumulative_distances, marker='o', color=colormap[category])
        
        if category not in legend_added:
            plt.plot([], [], marker='o', color=colormap[category], label=category)
            legend_added[category] = True

plt.xlabel('Time (s)')
plt.ylabel('Protein Distance Traveled ($\mu$m)')
plt.grid(True)
plt.legend()
plt.savefig("AllDistances.pdf", format="pdf", bbox_inches="tight")
plt.show()


# Second plot: Average Distance vs Time (based on shortest list length)
plt.figure(figsize=(8,8))

for category, datasets in categories.items():
    shortest_length = min(len(dataset['positions']) for dataset in datasets)
    time_shortest = min(dataset['time'] for dataset in datasets)
    average_cumulative_distances = np.zeros(shortest_length)  # Corrected length
    
    for dataset in datasets:
        dataset_positions = dataset['positions'][:shortest_length]  # Cut positions list to shortest length
        cumulative_distances = np.cumsum([0] + [distance(dataset_positions[i-1][0], dataset_positions[i-1][1], dataset_positions[i][0], dataset_positions[i][1]) for i in range(1, len(dataset_positions))])
        average_cumulative_distances += cumulative_distances
    
    average_cumulative_distances /= len(datasets)  # Calculate the average
    
    times = np.linspace(0, time_shortest, num=shortest_length)  # Corrected length
    
    plt.plot(times, average_cumulative_distances, marker='o', color=colormap[category])

plt.xlabel('Time (s)')
plt.ylabel('Average Protein Distance Traveled ($\mu$m)')
plt.grid(True)
plt.legend(categories.keys())

plt.tight_layout()
plt.savefig("AvgDistances.pdf", format="pdf", bbox_inches="tight")

plt.show()


plt.figure(figsize = (6.4, 8))
shortest_length = min(len(dataset['positions']) for datasets in categories.values())
average_distances = {category: np.zeros(shortest_length - 1) for category in categories}

for category, datasets in categories.items():
    for dataset in datasets:
        dataset_positions = dataset['positions'][:shortest_length]  # Cut positions list to shortest length
        distances = [distance(dataset_positions[i-1][0], dataset_positions[i-1][1], dataset_positions[i][0], dataset_positions[i][1]) for i in range(1, len(dataset_positions))]
        average_distances[category][:len(distances)] += distances
    
    average_distances[category] /= len(datasets)

def rayleigh_pdf(x, sigma):
    return (x/sigma**2) * np.exp(-x**2 / (sigma**2))

for category, avg_dist in average_distances.items():
    

    N = len(avg_dist)
    
    # Fit a Rayleigh distribution to the actual data
    scale = np.mean(avg_dist)/np.sqrt(np.pi / 2)
    
    num_bins=5
    # Calculate histogram of the actual data
    _binvalues, bins, _patches = plt.hist(avg_dist, bins=num_bins, density=False, rwidth=1, ec=colormap[category], color=colormap[category], alpha = 0.3)
    
    x = np.linspace(bins[0], bins[-1], 100)

    # Calculate bin centers for x values
    binwidth = (bins[-1] - bins[0]) / num_bins
    
    plt.plot(x, rayleigh(loc=0, scale=scale).pdf(x)*len(avg_dist)*binwidth, lw=2, alpha=1, label=category, color = colormap[category])
    plt.fill_between(x, rayleigh(loc=0, scale=scale).pdf(x)*len(avg_dist)*binwidth, color=colormap[category], alpha=0.0)

    # Plot the histogram of the actual data
    #plt.hist(avg_dist, bins=5, alpha=0.2, color=colormap[category], edgecolor='black', density=True)

    plt.axvline(x = x[np.argmax(rayleigh(loc=0, scale=scale).pdf(x)*len(avg_dist)*binwidth)], color = colormap[category], linewidth =2, linestyle = 'dashed')
    
    


plt.xlabel('Point-Point Distance ($\mu$m)')
plt.ylabel('Frequency')
#plt.grid(True)
plt.legend()
plt.xlim([0, 6])

plt.tight_layout()
plt.show()