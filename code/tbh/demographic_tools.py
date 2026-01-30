import pandas as pd
from jax import numpy as jnp
from jax.numpy.linalg import eigvals
from jax import lax
from summer2.functions import time as stf
import numpy as np

from tbh.paths import DATA_FOLDER


def get_pop_size(model_config):

    pop_data = pd.read_csv(DATA_FOLDER / "un_population.csv")
    # filter for country and truncate historical pre-analysis years
    pop_data = pop_data[(pop_data["ISO3_code"] == model_config['iso3']) & (pop_data["Time"] >= model_config['start_time'])]

    # Aggregate accross agegroups for each year
    agg_pop_data = model_config['pop_scaling'] * 1000. * pop_data.groupby('Time')['PopTotal'].sum().sort_index().cummax()  # cummax to avoid transcient population decline

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

"""
    Functions below are used to generate age mixing matrices using demographic data
"""
def get_normalised_fertility_data(iso3_code):
    """
    Load and normalise fertility data for a given country ISO3 code.    
    
    param iso3_code: str
        The ISO3 code of the country to load fertility data for.
    """
    fertility_data = pd.read_csv(DATA_FOLDER / f'un_fertility_rates_{iso3_code}.csv', index_col=0)
    normalised_fertility_data = fertility_data.div(fertility_data.sum(axis=1), axis=0)
    return normalised_fertility_data


def get_agegap_prob(normalised_fertility_data, age_gap, birth_year):
    """
    Retrieve the probability of a given parent-child age gap for a specific birth year.

    Parameters:
        normalised_fertility_data (pd.DataFrame): Fertility probabilities indexed by birth year, columns are mothers' ages.
        age_gap (int): Age difference between parent and child.
        birth_year (int): Year of birth of the child.

    Returns:
        float: Probability corresponding to the age gap and birth year, or 0.0 if out of range.
    """

    # Ensure birth_year is within the data range, if not, clamp it to the nearest available year
    earliest_year, latest_year = normalised_fertility_data.index.min(), normalised_fertility_data.index.max()
    birth_year = jnp.clip(birth_year, earliest_year, latest_year)

    # Return 0.0 if age_gap is less than youngest age in the data or greater than oldest age
    youngest_age, oldest_age = normalised_fertility_data.columns.astype(int).min(), normalised_fertility_data.columns.astype(int).max()
    if age_gap < youngest_age or age_gap > oldest_age:
        return 0.0
    else:
        return normalised_fertility_data[str(age_gap)].loc[birth_year]


def get_single_age_population(iso_3, max_age=120):
    pop_data = pd.read_csv(DATA_FOLDER / "un_population.csv")
    pop_data = pop_data[pop_data["ISO3_code"] == iso_3][["Time", "AgeGrp", "PopTotal"]]

    rows = []
    for _, r in pop_data.iterrows():
        year = r["Time"]
        pop = r["PopTotal"]
        agegrp = r["AgeGrp"]

        if agegrp.endswith("+"):   # normally "100+"
            a0 = int(agegrp[:-1]) 
            a1 = max_age
            assert a1 >= a0
        else:
            a0, a1 = map(int, agegrp.split("-"))
            
        n = a1 - a0 + 1
        for age in range(a0, a1 + 1):
            rows.append({
                "Time": year,
                "Age": age,
                "Pop": pop / n
            })

    return pd.DataFrame(rows)


def get_relative_age_weight(age, lower_bound, upper_bound, single_age_pop_df, year):
    earliest_year, latest_year = single_age_pop_df['Time'].min(), single_age_pop_df['Time'].max()
    year = jnp.clip(year, earliest_year, latest_year)

    # subset_df = single_age_pop_df[(single_age_pop_df["Time"] == year) & (single_age_pop_df["Age"] >= lower_bound) & (single_age_pop_df["Age"] <= upper_bound)] 
    
    time = jnp.array(single_age_pop_df["Time"])
    age_array = jnp.array(single_age_pop_df["Age"])
    pop = jnp.array(single_age_pop_df["Pop"])  # population column

    mask = (time == year) & (age_array >= lower_bound) & (age_array <= upper_bound)
    
    # Apply mask: use jnp.where for JIT-compatibility
    subset_pop = pop * mask   # zeros out entries not in the subset
    # Total population in subset
    total_pop = jnp.sum(subset_pop)

    # Population for a specific age
    age_mask = (age_array == age) & mask
    age_pop = jnp.sum(pop * age_mask)  # sums over exactly one entry
    
    # age_pop = subset_df[subset_df["Age"] == age]["Pop"].iloc[0]
    # total_pop = subset_df["Pop"].sum()  

    return age_pop / total_pop


# def gen_mixing_matrix_func(age_groups):
#     """
#     Returns a JAX-compatible function to build a symmetric age-structured mixing matrix
#     for a given set of age group lower bounds, with configurable socialising parameters.

#     Parameters
#     ----------
#     age_groups : list of int
#         List of lower bounds of age intervals, e.g. [0, 5, 15, 50].

#     Returns
#     -------
#     function
#         A function that takes child_socialising and elderly_socialising parameters and returns
#         an (n_groups x n_groups) mixing matrix as a JAX array.
#     """
#     age_groups = np.array(age_groups, dtype=int)

#     # Determine socialising parameter for each group
#     def build_mixing_matrix(child_socialising, elderly_socialising):
#         """
#         Constructs a symmetric mixing matrix where each age group has a socialising parameter.
#         Socialising parameter model:
#         Each age group is assigned a socialising parameter that reflects their relative level of social contacts. 
#         - Children (<15 years) use `child_socialising`.
#         - Middle-aged adults (15–64 years) have baseline socialising = 1.0.
#         - Elderly (≥65 years) use `elderly_socialising`.

#         The mixing matrix is constructed such that:
#         - Diagonal elements (within-group contacts) equal the group's socialising parameter.
#         - Off-diagonal elements (between-group contacts) equal the product of the socialising parameters of the two groups.

#         This assumes that contact intensity between two groups is proportional to the product of their social activity levels.

#         Parameters
#         ----------
#         child_socialising : float
#             Socialising factor for children (age < 15)
#         elderly_socialising : float
#             Socialising factor for elderly (age >= 65)

#         Returns
#         -------
#         matrix : jax.Array (n_groups x n_groups)
#         """
#         # Assign socialising parameters per age group
#         socialising = jnp.array([
#             child_socialising if age < 15 else
#             elderly_socialising if age >= 65 else
#             1.0
#             for age in age_groups
#         ])

#         # Construct the mixing matrix: outer product
#         M = jnp.outer(socialising, socialising)
#         # Compute spectral radius (largest absolute eigenvalue)
#         rho = jnp.max(jnp.abs(eigvals(M)))

#         # Rescale so spectral radius = 1
#         M = M / rho
#         return M

#     return build_mixing_matrix