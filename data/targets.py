from estival import targets as est
import pandas as pd

# Define calibration targets (dummy data for now)
targets = [
    est.NormalTarget(
        name='tb_prevalence_per100k', 
        data=pd.Series(data=[600,], index=[2020]), 
        stdev=100.
    ),
    est.NormalTarget(
        name='tbi_prevalence_perc', 
        data=pd.Series(data=[40,], index=[2020]), 
        stdev=5.
    ),
    est.NormalTarget(
        name='perc_prev_subclinical', 
        data=pd.Series(data=[50], index=[2020]), 
        stdev=5.
    ),
]