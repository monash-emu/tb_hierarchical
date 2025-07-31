import pandas as pd
from summer2.functions import time as stf

from tbh.paths import DATA_FOLDER


def get_pop_size(model_config):

    pop_data = pd.read_csv(DATA_FOLDER / "un_population.csv")
    # filter for country and truncate historical pre-analysis years
    pop_data = pop_data[(pop_data["ISO3_code"] == model_config['iso3']) & (pop_data["Time"] >= model_config['start_time'])]

    # Aggregate accross agegroups for each year
    agg_pop_data = 1000. * pop_data.groupby('Time')['PopTotal'].sum().sort_index().cummax()  # cummax to avoid transcient population decline

    return agg_pop_data


def get_death_rates_by_age(model_config):
    """
    Compute death rates using AgeGrpStart as group labels, aggregated over defined bins.
    
    Args:
        model_config (dict): must contain 'iso3', 'start_time'
        age_bins (list of int): list of age group starting points (e.g., [0, 15, 65])
    
    Returns:
        dict: {AgeGrpStart: pd.Series of death rates indexed by year}
    """
    age_bins = [int(a) for a in model_config['age_groups']]

    pop_data = pd.read_csv(DATA_FOLDER / "un_population.csv")
    mort_data = pd.read_csv(DATA_FOLDER / "un_mortality.csv")

    # Filter by country and start year
    pop_data = pop_data[(pop_data["ISO3_code"] == model_config["iso3"]) & 
                        (pop_data["Time"] >= model_config["start_time"])]
    mort_data = mort_data[(mort_data["ISO3_code"] == model_config["iso3"]) & 
                          (mort_data["Time"] >= model_config["start_time"])]

    # Define bin edges and labels
    bin_edges = age_bins + [200]  # use 200 as an upper cap beyond realistic ages
    bin_labels = age_bins  # label each bin by its lower bound

    pop_data["age_group"] = pd.cut(pop_data["AgeGrpStart"], bins=bin_edges, labels=bin_labels, right=False)
    mort_data["age_group"] = pd.cut(mort_data["AgeGrpStart"], bins=bin_edges, labels=bin_labels, right=False)

    # Drop rows outside specified bins (age_group == NaN)
    pop_data = pop_data.dropna(subset=["age_group"])
    mort_data = mort_data.dropna(subset=["age_group"])

    # Convert category labels back to integers
    pop_data["age_group"] = pop_data["age_group"].astype(int)
    mort_data["age_group"] = mort_data["age_group"].astype(int)

    # Aggregate by year and age group
    pop_summary = pop_data.groupby(["Time", "age_group"])["PopTotal"].sum().reset_index()
    mort_summary = mort_data.groupby(["Time", "age_group"])["DeathTotal"].sum().reset_index()

    merged = pd.merge(mort_summary, pop_summary, on=["Time", "age_group"])
    merged["death_rate"] = merged["DeathTotal"] / merged["PopTotal"]

    # dictionary of series
    death_rate_series = {
        str(age_group): group.set_index("Time")["death_rate"]
        for age_group, group in merged.groupby("age_group")
    }

    # convert to functions
    death_rate_funcs = {
        age_group: stf.get_sigmoidal_interpolation_function(series.index, series)
        for age_group, series in death_rate_series.items()
    }

    return death_rate_funcs
