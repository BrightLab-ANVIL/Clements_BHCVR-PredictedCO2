import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_BH_locations(co2_trace_path, peakcheck_path, num_BHs, output_dir, co2_column=0, peakcheck_column=0, ID=None): 
    """
    Outputs the start and end indices of each breath hold in a CO2 timeseries, 
    and saves a plot displaying the timeseries with marked breath-hold periods for manual verification.
    
    Parameters
    ----------
    co2_trace_path : string
        Path to an exhaled CO2 timeseries recorded during a breath-hold task
    peakcheck_path : string
        Path to a binary file consisting of 0s and 1s that is the same length as "co2_trace." This file is typically generated to calculate end-tidal CO2 and contains 1s at the peaks in the CO2 trace and 0s elsewhere
    num_BHs : int
        The number of breath holds that were completed
    output_dir : string
        Full path to the output directory
    co2_column : int, optional (default value = 0)
        Column number (indexing starts at 0) in the CO2 timeseries file containing the CO2 timeseries of interest
    peakcheck_colunmn : int, optional (default value = 0)
        Column number (indexing starts at 0) in the peakcheck file containing peak information of interest
    ID : string, optional (default value = None)
        An identifier that can be added to the outputted files (useful if you are running this function for multiple files).
        
    Returns
    -------
    indices : numpy array
        Start and end indices of each breath hold in co2_trace. In order of BH 1 start, Bh 1 end, BH 2 start, BH 2 end, and so on. 
    
    """
    
    co2_trace = np.loadtxt(co2_trace_path)
    co2_trace = co2_trace[:,co2_column]
    peakcheck = np.loadtxt(peakcheck_path)
    peakcheck = peakcheck[:,peakcheck_column]
    BH_locations = np.zeros(num_BHs*2)
    indices_of_ones = [i for i, bit in enumerate(peakcheck) if bit == 1]
    max_distances = []
    max_distance_pairs = []
    
    # Calculate the distance between each "1" in your peakcheck file
    for i in range(len(indices_of_ones) - 1):
        for j in range(i + 1, len(indices_of_ones)):
            if all(bit == 0 for bit in peakcheck[indices_of_ones[i] + 1 : indices_of_ones[j]]):
                distance = abs(indices_of_ones[j] - indices_of_ones[i])
                max_distances.append(distance)
                max_distance_pairs.append((indices_of_ones[i], indices_of_ones[j]))
    # Sort pairs of 1s in descending order by distance
    sorted_pairs = [pair for _, pair in sorted(zip(max_distances, max_distance_pairs), reverse=True)]
    sorted_pairs_chronological = sorted(sorted_pairs[:num_BHs], key=lambda pair: pair[0])
    # Plot co2_timeseries with marked breath-hold periods
    plt.figure(figsize=(15, 8))
    plt.plot(co2_trace)
    BH_start_end_column = 0
    for pair in sorted_pairs_chronological[:num_BHs]:
        index1, index2 = pair
        BH_locations[BH_start_end_column] = index1
        BH_locations[BH_start_end_column+1] = index2
        plt.axvline(pair[0], color='gray', linestyle='--', label='Vertical Line')
        plt.axvline(pair[1], color='gray', linestyle='--', label='Vertical Line')
        plt.axvspan(pair[0],pair[1], facecolor='lightgray', alpha=0.5)
        BH_start_end_column+=2
    plt.title('Breath-hold Periods Identified in CO2 Timeseries')
    if ID:
        fig_save_path=output_dir + f'/{ID}_Plotted_Breath_Holds.PNG'
    else:
        fig_save_path=output_dir + '/Plotted_Breath_Holds.PNG'
    plt.savefig(fig_save_path) 
    return BH_locations

def get_BH_CO2_changes(co2_trace_path, BH_locations, num_BHs, co2_column=0): 
    """
    Given the start and end locations of each breath hold in a CO2 timeseries, calculates the CO2 change caused by each breath hold in the timeseries 
    
    Parameters
    ----------
    co2_trace_path : string
        Path to an exhaled CO2 timeseries recorded during a breath-hold task.
    BH_locations : numpy array
        Start and end indices of each breath hold in co2_trace. In order of BH 1 start, Bh 1 end, BH 2 start, BH 2 end, and so on. Output of get_BH_locations()
    num_BHs : int
        The number of breath holds that were completed
    co2_column : int, optional (default value = 0)
        Column number (indexing starts at 0) in CO2 timeseries containing the CO2 timeseries of interest    
        
    Returns
    -------
    BH_changes : numpy array 
        CO2 change caused by each breath-hold in the timeseries
    
    """
    
    co2_trace = np.loadtxt(co2_trace_path)
    co2_trace = co2_trace[:,co2_column]
    BH_changes = np.full(num_BHs, np.nan)
    i=0
    for BH in range(0,2*num_BHs,2):
        BH_start_index = int(BH_locations[BH])
        BH_end_index = int(BH_locations[BH+1])
        co2_value1, co2_value2 = co2_trace[BH_start_index], co2_trace[BH_end_index]
        change_co2 = co2_value2 - co2_value1
        BH_changes[i] = change_co2
        i+=1
    return BH_changes

def check_BH_quality(BH_changes,threshold): 
    """
    Given a threshold CO2 increase for classifying breath holds as high-quality, and an array of CO2 increases for each breath hold in a timeseries, this function outputs an array with 1s for high-quality breath holds and 0s for low-quality breath holds.
    
    Parameters
    ----------
    BH_changes : numpy array
        The output of get_BH_CO2_increases(). An array containing the CO2 change caused by each breath hold in a CO2 timeseries.
    threshold : float or int
        The threshold CO2 increase for a breath hold to be classified as high-quality.
        
    Returns
    -------
    quality : numpy array
        For each high-quality breath hold in BH_changes, contains a 1 if the breath hold is high-quality and a 0 if the breath hold is low-quality.
    
    """
    quality = (BH_changes >= threshold).astype(int)
    return quality

def calculate_usable_segments(quality):
    """
    Given the output of check_BH_quality, which is a binary array containing 1s for each 
    high quality breath hold and 0s for each low quality of breath hold, this function
    returns the length of each "usable" data segments. In this case, a "usable" data segment contains 
    2 or more consecutive high-quality breath holds. 
    
    Note: This funciton assumes that you can only have up to 3 "usable" segments in your data,
    which is only appropriate if your data contains 10 or less breath holds.
    
    Parameters
    ----------
    quality : numpy array
        An array that is as long as the number of breath holds in your CO2 timeseries and
        contains 1s corresponding to each high-quality breath hold and 0s corresponding
        to each low-quality breath hold. The output of check_BH_quality().
        
    Returns
    -------
    usable_segments : numpy array
        An array that contains the length of each usable data segment in your CO2 timeseries.
    """
    current_length = 0
    usable_lengths = []
    usable_segments = []
    for value in quality:
        if value == 1:
            current_length += 1
        else:
            if current_length >= 2:
                usable_lengths.append(current_length)
            current_length = 0
    if current_length >= 2:
        usable_lengths.append(current_length)
    # Append the lengths of usable data for the current dataset
    if len(usable_lengths) == 0:
        usable_segments.append([0, 0, 0])  # Append [0, 0, 0] if no usable segments
    elif len(usable_lengths) == 1:
        usable_segments.append([usable_lengths[0], 0, 0])  # Append [length, 0, 0] if one usable segment
    elif len(usable_lengths) == 2:
        usable_segments.append([usable_lengths[0], usable_lengths[1], 0])  # Append [length1, length2, 0] if two usable segments
    else:
        usable_segments.append(usable_lengths) # Append [length1, length2, length3] if 2 usable segments
    return usable_segments