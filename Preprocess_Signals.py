import numpy as np
import pandas as pd
import os
from numpy.fft import fft, ifft, fftshift

#The function below (delay_correction_with_limit) was heavily based on code written by Agrawal et al. (citation below) that is available in their GitHub repository: https://github.com/vismayagrawal/RESPCO/blob/main/code/utils.py
#Their code was adapted to only allow for negative correlations between CO2 and RVT and to only allow for negative shifts. 
#Agrawal, V., Zhong, X. Z., & Chen, J. J. (2023). Generating dynamic carbon-dioxide traces from respiration-belt recordings: Feasibility using neural networks and application in functional magnetic resonance imaging. Frontiers in Neuroimaging, 2. https://doi.org/10.3389/fnimg.2023.1119539

def delay_correction_with_limit(co2, rvt, max_shift_secs, sampling_freq):
    """
    Computes the time shift between a PETCO2 and RVT timeseries 
    that maximizes their negative correlation. If no shifts result in a negative 
    correlation, 0 shift is applied.
    
    The PETCO2 timeseries is assumed to have a longer measurement delay compared 
    to RVT, and thus PETCO2 can only be shifted earlier in time within a defined limit.
    
    After the PETCO2 timeseries is delay-corrected, the RVT timeseries is trimmed 
    to match the corrected length of the PETCO2 timeseries.
    
    Parameters
    ----------
    co2 : numpy array
        PETCO2 timeseries
    rvt : numpy array
        RVT timeseries
    max_shift_secs : int
        Maximum time shift (in seconds) that PETCO2 can be shifted earlier in time
    sampling_freq : int
        The sampling frequency (Hz) of the input timeseries

    Returns
    -------
    corrected_co2 : numpy array
        Delay-corrected PETCO2 timeseries
    trimmed_rvt : numpy array
        Trimmed RVT timeseries to match the corrected PETCO2 length
    shift : int
        Time shift (in samples) applied to PETCO2 
    """
    assert len(co2) == len(rvt), "co2 and rvt must have the same length"
    fft_co2 = fft(co2)
    fft_rvt = fft(np.flipud(rvt))
    cross_corr = np.real(ifft(fft_co2 * fft_rvt))  # Inverse FFT to get cross-correlation
    correlation = fftshift(cross_corr)
    zero_index = len(co2) // 2
    correlation_mean = np.mean(correlation)
    limit_samples = max_shift_secs * sampling_freq
    if zero_index - limit_samples > 0:
        correlation[:zero_index - limit_samples] = correlation_mean
    if zero_index + limit_samples < len(correlation):
        correlation[zero_index + limit_samples:] = correlation_mean
    # Find the shift that results in the maximum negative correlation
    min_corr_idx = np.argmin(correlation)
    if np.min(correlation) < correlation_mean:
        shift = zero_index - min_corr_idx
    else:
        shift = 0 #If no shifts result in a negative correlation, 0 shift is applied
        
    # Apply the shift to the PETCO2 timeseries and trim RVT
    if shift == 0:
        corrected_co2 = co2
        trimmed_rvt = rvt
    elif shift > 0:
        corrected_co2 = co2[:-shift]
        trimmed_rvt = rvt[shift:]
    else:
        corrected_co2 = co2[abs(shift):]
        trimmed_rvt = rvt[:-abs(shift)]
    return corrected_co2, trimmed_rvt, shift

def standardize(x):
    return (x-np.mean(x))/np.std(x)
