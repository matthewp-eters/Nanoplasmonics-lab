# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:07:40 2022

@author: daw85
"""

import numpy as np
from nptdms import TdmsFile
from nptdms import tdms
import scipy
import scipy.fftpack
from scipy import signal
from matplotlib import pyplot as plt
import statsmodels.api as sm

def data_load(path):
    tdms_file = TdmsFile(path)
    properties = tdms_file['Analog'].properties
    scanrate=properties['ScanRate']
    data= tdms_file['Analog']['AI1']
    return data, scanrate
def butter_lowpass_filter(data, cutoff, order, highorlow='low', dt=1e-5):
    if not isinstance(data, np.ndarray):
        data=np.array(data[0]).flatten()
    else:
        pass
    fs=1/dt # sample rate, Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = signal.butter(order, normal_cutoff, btype=highorlow, analog=False)
    y = signal.filtfilt(b, a, data)
    return y

# exampledata=data_load(r'PATH')
# plt.figure(figsize=(4,3))
# plt.plot(np.arange(len(exampledata))/exampledata[1],exampledata)
# plt.xlabel('Time (s)')
# plt.ylabel('PD Response (V)')