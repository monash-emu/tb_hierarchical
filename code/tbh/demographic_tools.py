import pandas as pd
from jax import numpy as jnp
from jax import lax
from summer2.functions import time as stf
import numpy as np

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


def gen_mixing_matrix_func(age_groups):
    """
        Returns a JAX-compatible function to build a symmetric age-structured mixing matrix
        for a given set of age group lower bounds.

        Parameters
        ----------
        age_groups : list of str
            List of lower bounds of age intervals, e.g. ["0", "5", "15", "50"].

        Returns
        -------
        function
            A function that takes two parameters (mixing_factor_cc and mixing_factor_ca) and returns
            a (n_groups x n_groups) mixing matrix as a JAX array.
    """
    age_groups = np.array(age_groups, dtype=int)
    n_groups = len(age_groups)
 
    # Children: age < 15
    n_child = (age_groups < 15).sum()  # number of child groups
 
    def build_mixing_matrix(mixing_factor_cc, mixing_factor_ca):
        """
            Constructs a symmetric mixing matrix between age groups using the provided mixing factors.
            Children are defined as age groups with lower bound < 15. Adults are all others.

            Parameters
            ----------
            mixing_factor_cc : float, Relative mixing rate between children (child-child interactions), ref: adult-adult interactions.
            mixing_factor_ca : float, Relative mixing rate between children and adults (child-adult interactions), ref: adult-adult interactions.

            Returns
            -------
            matrix : jax.Array (n_groups x n_groups) symmetric matrix of mixing rates.
        """
     
        M = jnp.full((n_groups, n_groups), mixing_factor_ca)
        M = M.at[:n_child,:n_child].set(mixing_factor_cc)
        M = M.at[n_child:, n_child:].set(1.0)
        return M
 
    return build_mixing_matrix