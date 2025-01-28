#This code was based on code developed by Agarwal et al. available at: https://github.com/vismayagrawal/RESPCO/blob/main/code/utils.py
#Citation: Agrawal, V., Zhong, X. Z., & Chen, J. J. (2023). Generating dynamic carbon-dioxide traces from respiration-belt recordings: Feasibility using neural networks and application in functional magnetic resonance imaging. Frontiers in Neuroimaging, 2. https://doi.org/10.3389/fnimg.2023.1119539

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, sosfilt
import peakutils

def get_peaks(signal, Fs, thres=0.5):
 	"""
 	Useful for getting PETCO2 or in general peak of a signal
 	"""
 	peak_index = peakutils.indexes(signal, thres, min_dist=30*Fs)
 	peak_amplitude = signal[peak_index]

 	return peak_index, peak_amplitude
