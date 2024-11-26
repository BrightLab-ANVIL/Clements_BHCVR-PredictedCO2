# Quantitative mapping of cerebrovascular reactivity amplitude and delay with breath-hold BOLD fMRI when end-tidal CO<sub>2</sub> quality is low
This analysis code is being shared to accompany the manuscript available here: https://www.biorxiv.org/content/10.1101/2024.11.18.624159v1

# Code (more to come)
Get_CO2_increases.py is used to calculate changes in CO<sub>2</sub> caused by breath holds. This can provide information about the participant's compliance during the task and the overall quality of the CO<sub>2</sub> recordings.
* get_BH_locations() outputs the start and end indices of each breath-hold in an exhaled CO<sub>2</sub> timeseries and plot the timeseries with marked breath-hold periods for manual verification
* After you manually verify that the outputted start and end indices are correct, get_BH_CO2_increases() can be used to calculate the CO<sub>2</sub> increase associated with each breath hold

