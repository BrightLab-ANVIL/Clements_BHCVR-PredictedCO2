import numpy as np
import matplotlib.pyplot as plt

def get_BH_locations(co2_trace_path, peakcheck_path, num_BHs, output_dir, ID=None): 
    """
    Outputs the start and end indices of each breath hold in a CO2 timeseries, 
    and saves a plot displaying the timeseries with marked breath-hold periods for manual verification.
    
    Parameters
    ----------
    co2_trace : string
        Path to an exhaled CO2 timeseries recorded during a breath-hold task.

    peakcheck_path : string
        Path to a binary file consisting of 0s and 1s that is the same length as "co2_trace." This file is typically generated to calculate end-tidal CO2 and contains 1s at the peaks in the CO2 trace and 0s elsewhere.
    num_BHs : int
        The number of breath holds that were completed
    output_dir : string
        Full path to the output directory
    ID : string, optional
        An identifier that can be added to the outputted files (useful if you are running this function for multiple files).
        
    Returns
    -------
    indices : np.ndarray
        Start and end indices of each breath hold in co2_trace. In order of BH 1 start, Bh 1 end, BH 2 start, BH 2 end, and so on. 
    
    """
    
    co2_trace = np.loadtxt(co2_trace_path)
    peakcheck = np.loadtxt(peakcheck_path)
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

def get_BH_CO2_increases(co2_trace_path, BH_locations, num_BHs): 
    """
    Given the start and end indices of each breath hold in a PETCO2 timeseries, calculates the CO2 change caused by each breath hold in the timeseries 
    
    Parameters
    ----------
    co2_trace : string
        Path to an exhaled CO2 timeseries recorded during a breath-hold task.
    BH_locations : np.ndarray
        Start and end indices of each breath hold in co2_trace. In order of BH 1 start, Bh 1 end, BH 2 start, BH 2 end, and so on. Output of get_BH_locations()
    num_BHs : int
        The number of breath holds that were completed
        
    Returns
    -------
    indices : np.ndarray
        CO2 change caused by each breath-hold in the timeseries
    
    """
    
    co2_trace = np.loadtxt(co2_trace_path)
    BH_increases = np.full(num_BHs, np.nan)
    i=0
    for BH in range(0,2*num_BHs,2):
        BH_start_index = int(BH_locations[BH])
        BH_end_index = int(BH_locations[BH+1])
        co2_value1, co2_value2 = co2_trace[BH_start_index], co2_trace[BH_end_index]
        change_co2 = co2_value2 - co2_value1
        BH_increases[i] = change_co2
        i+=1
    return BH_increases


