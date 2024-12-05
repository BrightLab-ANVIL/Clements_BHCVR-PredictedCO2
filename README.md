# Quantitative mapping of cerebrovascular reactivity amplitude and delay with breath-hold BOLD fMRI when end-tidal CO<sub>2</sub> quality is low
This analysis code is being shared to accompany the manuscript available here: https://www.biorxiv.org/content/10.1101/2024.11.18.624159v1

# Code (More to Come)

`Check_CO2_quality.py` contains functions that can be used to assess the quality of an exhaled CO<sub>2</sub> timeseries and better understand a participant's compliance during a breath hold task. We recommend using the functions in the following order:

1. Use `get_BH_locations()` to identify the start and end locations of each breath-hold in an exhaled CO<sub>2</sub> timeseries and plot the timeseries with marked breath-hold periods for manual verification.
2. After manually verifying that the outputted start and end locations are correct, use `get_BH_CO2_changes()` to calculate the CO<sub>2</sub> increase associated with each breath hold. 
3. Decide on a threshold to classify breath holds as "high-quality," and then use `check_BH_quality()` to output a binary array that is as long as the number of breath holds, with 1s corresponding to each high-quality breath hold and 0s elsewhere.
4. Use `calculate_usable_segments()` to identify the lengths of usable "data segments" consisting or 2 or more high-quality, consecutive breath holds.

`Preprocess_Signals.py` can be used to pre-process P<sub>ET</sub>CO<sub>2</sub> and RVT timeseries. 
* `delay_correction_with_limit()` delay corrects P<sub>ET</sub>CO<sub>2</sub> to maximize its negative correlation with RVT. This code is based on code developed by [Agrawal et al.](https://github.com/vismayagrawal/RESPCO/blob/main/code/utils.py)
* `standardize()` z-normalizes a timeseries so that it has a mean of 0 and a standard deviation of 1

`Models.py` contains the code for the FCN models with 8, 10, 12, and 14 convolutional layers. The remaining models described in the paper used code developed by [Agrawal et al.]([https://github.com/vismayagrawal/RESPCO/blob/main/code/utils.py](https://github.com/vismayagrawal/RESPCO/blob/main/code/models.py))

# References
Agrawal, V., Zhong, X. Z., & Chen, J. J. (2023). Generating dynamic carbon-dioxide traces from respiration-belt recordings: Feasibility using neural networks and application in functional magnetic resonance imaging. Frontiers in Neuroimaging, 2. https://doi.org/10.3389/fnimg.2023.1119539
