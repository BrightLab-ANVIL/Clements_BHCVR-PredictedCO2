# Quantitative mapping of cerebrovascular reactivity amplitude and delay with breath-hold BOLD fMRI when end-tidal CO<sub>2</sub> quality is low
This analysis code is being shared to accompany the manuscript available here: https://www.biorxiv.org/content/10.1101/2024.11.18.624159v1

# Code (More to Come)

`Get_CO2_increases.py` contains functions that can be used to classify breath holds in a CO₂ timeseries as high-quality based on the CO₂ change they induce. This can provide information about a participant's compliance during a breath hold task.

* `get_BH_locations()` outputs the start and end indices of each breath-hold in an exhaled CO₂ timeseries and plots the timeseries with marked breath-hold periods for manual verification.
* After manually verifying that the outputted start and end indices are correct, `get_BH_CO2_increases()` can be used to calculate the CO₂ increase associated with each breath hold.
* After deciding on a threshold to classify breath holds as "high-quality," `check_BH_quality()` can be used to output a binary array, with 1s corresponding to each high-quality breath hold and 0s elsewhere.


