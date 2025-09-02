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
    age_bins = [int(a) for a in model_config['age_groups']]
    last_bin_start = age_bins[-1]

    pop_data = pd.read_csv(DATA_FOLDER / "un_population.csv")
    mort_data = pd.read_csv(DATA_FOLDER / "un_mortality.csv")

    # Filter
    pop_data = pop_data[(pop_data["ISO3_code"] == model_config["iso3"]) & 
                        (pop_data["Time"] >= model_config["start_time"])]
    mort_data = mort_data[(mort_data["ISO3_code"] == model_config["iso3"]) & 
                          (mort_data["Time"] >= model_config["start_time"])]

    # --- Step 1: expand population using one-year age-groups, except 100+ ---
    expanded_pop = []
    for _, row in pop_data.iterrows():
        start_age = row["AgeGrpStart"]
        if start_age == 100:
            expanded_pop.append({
                "Time": row["Time"],
                "Age": "100+",
                "PopTotal": row["PopTotal"]
            })
        else:
            end_age = start_age + 5
            for age in range(start_age, end_age):
                expanded_pop.append({
                    "Time": row["Time"],
                    "Age": str(age),
                    "PopTotal": row["PopTotal"] / 5.  # assumed population uniformly distributed within 5-year age group 
                })              
    expanded_pop = pd.DataFrame(expanded_pop)

    # --- Step 2: merge ---
    mort_data["Age"] = mort_data["AgeGrp"]
    merged = pd.merge(
        expanded_pop,
        mort_data,
        on=["Time", "Age"],
        how="left"
    ).fillna({"DeathTotal": 0})

    # --- Step 3: assign to model bins ---
    def assign_bin(age):
        if age == "100+":
            return last_bin_start
        elif int(age) >= last_bin_start:
            return last_bin_start
        else:
            # find appropriate bin
            for i in range(len(age_bins) - 1):
                if age_bins[i] <= int(age) < age_bins[i+1]:
                    return age_bins[i]
        return None

    merged["age_group"] = merged["Age"].apply(assign_bin)
    merged = merged.dropna(subset=["age_group"]).astype({"age_group": int})

    # --- Step 4: aggregate ---
    agg = merged.groupby(["Time", "age_group"])[["PopTotal", "DeathTotal"]].sum().reset_index()
    agg["death_rate"] = agg["DeathTotal"] / agg["PopTotal"]

    # --- Step 6: wrap ---
    death_rate_series = {
        str(age_group): group.set_index("Time")["death_rate"]
        for age_group, group in agg.groupby("age_group")
    }
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