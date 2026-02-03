import pandas as pd
from jax import numpy as jnp
from jax.numpy.linalg import eigvals

from summer2.functions import time as stf
import numpy as np

from tbh.paths import DATA_FOLDER


def get_population_over_time(iso3, age_groups, max_age=120, scaling_factor=1.):
    """
    Return two DataFrames of population size over time. First DataFrame has single-age rows, second DataFrame has age-groups as columns.

    Parameters:
        iso3 (str): ISO3 country code.
        age_groups (list[str|int]): list of age-group lower bounds, e.g. ["0","10","20"].
        max_age (int): upper bound for the final open-ended group.
        scaling_factor (float): factor to scale population sizes by (e.g. for subnational models).

    Returns:
        sap (pd.DataFrame): DataFrame with columns ["Time", "Age", "Pop"] for single-age population.
        group_popsize (pd.DataFrame): DataFrame with Time as index and age-groups as columns.    
    """
    # load raw population data for iso3 and expand grouped AgeGrp into single-age rows
    pop_data = pd.read_csv(DATA_FOLDER / "un_population.csv")
    # filter to iso3 and relevant columns
    pop_data = pop_data[pop_data["ISO3_code"] == iso3][["Time", "AgeGrp", "PopTotal"]]

    # Build single-age population DataFrame
    single_rows = []
    for _, r in pop_data.iterrows():
        pop = r["PopTotal"] * 1000. * scaling_factor # convert from thousands, and apply scaling factor if requested (e.g. subnational model)
        agegrp = r["AgeGrp"]

        if isinstance(agegrp, str) and agegrp.endswith("+"):
            a0 = int(agegrp[:-1])
            a1 = max_age
        else:
            a0, a1 = map(int, str(agegrp).split("-"))

        n_ages = a1 - a0 + 1
        for age in range(a0, a1 + 1):
            single_rows.append({"Time": r["Time"], "Age": age, "Pop": pop / n_ages})

    sap = pd.DataFrame(single_rows)

    # build group definitions (lb, ub, label)
    lbs = sorted(int(x) for x in age_groups)
    groups = []
    for i, lb in enumerate(lbs):
        ub = (lbs[i + 1] - 1) if i < len(lbs) - 1 else max_age
        label = age_groups[i]
        groups.append((lb, ub, label))

    # aggregate by year
    years = sorted(sap["Time"].unique())
    rows = []
    for year in years:
        sub = sap[sap["Time"] == year]
        row = {"Time": year}
        for lb, ub, label in groups:
            row[label] = sub.loc[(sub["Age"] >= lb) & (sub["Age"] <= ub), "Pop"].sum()
        rows.append(row)

    group_popsize = pd.DataFrame(rows).set_index("Time").sort_index()
    return sap, group_popsize


def get_death_rates_by_age(model_config, group_popsize):
    """
    Generate age-specific death rate functions based on UN mortality and popsize by age data and model configuration.
    Parameters
    ----------
    model_config : dict
        Model configuration dictionary containing 'age_groups', 'iso3', and 'start_time'.
    group_popsize : pd.DataFrame
        DataFrame with Time as index and age-groups as columns representing population sizes.   
    Returns -------
    death_rate_funcs : dict
        Dictionary mapping age group lower bounds to JAX-compatible sigmoidal interpolation functions of death rates over time.
    """
    age_bins = [int(a) for a in model_config['age_groups']]
    last_bin_start = age_bins[-1]

    mort_data = pd.read_csv(DATA_FOLDER / "un_mortality.csv") 
    # Filter and clean the mortality data
    mort_data = mort_data[(mort_data["ISO3_code"] == model_config["iso3"]) & 
                          (mort_data["Time"] >= model_config["start_time"])]
    mort_data = mort_data[["Time", "AgeGrp", "DeathTotal"]] # only keep relevant columns
    mort_data["DeathTotal"] *= 1000. # convert from thousands

    # Helper function to assign age to model bins
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

    mort_data["age_group"] = mort_data["AgeGrp"].apply(assign_bin)

    
    mort_data = mort_data.groupby(["Time", "age_group"], as_index=False).agg({"DeathTotal": "sum"})
    mort_data = mort_data.pivot(index="Time", columns="age_group", values="DeathTotal")
    mort_data.reset_index()
    mort_data.columns = mort_data.columns.astype(str)

    death_rates = mort_data.div(group_popsize, axis=0).dropna()
    death_rate_funcs = {
        age_group: stf.get_sigmoidal_interpolation_function(death_rates.index, death_rates[age_group].values)
        for age_group in model_config['age_groups']
    }

    return death_rate_funcs


"""
    Functions below are used to compute age-mixing matrices based on age-gap distributions.
"""
def build_agegap_lookup(normalised_fertility_data):
    """
    Build JAX-friendly lookup structure for parent-child age gap probabilities from normalised fertility data.

    Parameters:
        normalised_fertility_data (pd.DataFrame): DataFrame with years as index and ages as columns, containing normalised fertility rates.
    
    Returns:
        probs: jnp.ndarray [n_years, n_ages]
        year0: int (first year available in fertility data)
        age0: int (youngest age available in fertility data)
    """
    # Ensure numeric column names
    fert = normalised_fertility_data.copy()
    fert.columns = fert.columns.astype(int)

    year0 = fert.index.min()
    age0 = fert.columns.astype(int).min()
    probs = jnp.asarray(fert.to_numpy())  # shape (n_years, n_ages)

    return probs, year0, age0


def get_agegap_prob_jax(
    probs,        # Array with shape (n_years, n_ages)
    year0,     
    age0,       
    birth_year,   
    age_gap      
):
    """
    Retrieve the probability of a parent-child age gap for a given birth year using precomputed data,
    in a JAX-compatible, vectorized way.

    Years outside the available range are clamped to the nearest year, while age gaps outside the
    supported range return 0.0. Fully compatible with JAX JIT compilation and vectorization.

    Parameters
    ----------
    probs : jax.Array
        Precomputed 2D array of probabilities, shape (n_years, n_ages),
        where rows correspond to consecutive years starting from `year0`
        and columns correspond to consecutive ages starting from `age0`.
    year0 : int
        The earliest year in the `probs` array (row index 0).
    age0 : int
        The youngest age in the `probs` array (column index 0).
    birth_year : int or jax.Array
        Year(s) of birth of the child. Can be a scalar or an array of years.
    age_gap : int or jax.Array
        Parent-child age gap(s). Can be a scalar or an array of ages.

    Returns
    -------
    jax.Array
        Probability value(s) corresponding to the requested `birth_year` and `age_gap`.
    """
    n_years, n_ages = probs.shape

    # Convert to indices
    year_idx = birth_year.astype(jnp.int32) - year0
    age_idx = age_gap - age0

    # Clamp years to nearest available year
    year_idx = jnp.clip(year_idx, 0, n_years - 1)

    # Mask invalid age gaps
    valid_age = (age_idx >= 0) & (age_idx < n_ages)

    # Safe indexing
    age_idx_safe = jnp.clip(age_idx, 0, n_ages - 1)

    prob = probs[year_idx, age_idx_safe]

    return jnp.where(valid_age, prob, 0.0)


def build_age_weight_lookup(age_groups: list[str], single_age_pop_df: pd.DataFrame) -> jnp.ndarray:
    """
    Build a JAX-compatible lookup of relative population weights by age and year.

    Parameters
    ----------
    age_groups : list[str]
        Lower bounds of age groups (e.g., ["0", "15", "65"]).

    single_age_pop_df : pd.DataFrame
        Columns: ["Time", "Age", "Pop"], single-age population data.

    Returns
    -------
    jax.numpy.ndarray
        Array of shape (n_years, n_ages) giving the relative population weight of
        each age within its age group, per year. Years are indexed based on those in single_age_pop_df.
    """
    # Convert age_groups to int lower bounds
    age_bins = sorted(int(a) for a in age_groups)
    age_max = single_age_pop_df['Age'].max() + 1
    age_bins.append(age_max)  # upper bound for last group

    # Assign each age to an age group
    age_group_map = pd.cut(single_age_pop_df['Age'], bins=age_bins, right=False, labels=False)
    single_age_pop_df = single_age_pop_df.copy()
    single_age_pop_df['AgeGroup'] = age_group_map

    # Pivot to (Time x Age)
    pivot = single_age_pop_df.pivot(index='Time', columns='Age', values='Pop').fillna(0.0)

    # Compute relative weights within each age group
    # result will be same shape as pivot (Time x Age)
    age_weights = pivot.copy()
    for group in range(len(age_bins) - 1):
        ages_in_group = [age for age in pivot.columns if age_bins[group] <= age < age_bins[group + 1]]
        # sum population in this group for each year
        pop_sum = pivot[ages_in_group].sum(axis=1)
        # divide each age by total in the group (avoid division by zero)
        age_weights[ages_in_group] = pivot[ages_in_group].div(pop_sum.replace(0, 1), axis=0)

    # Convert to JAX array
    age_weights_jax = jnp.asarray(age_weights.values)
    year0 = pivot.index.min()

    return age_weights_jax, year0


def get_age_weight_jax(age, time, age_weight_lookup, year0):
    """
    Retrieve the relative population weight for a given age and year from a precomputed JAX lookup.

    Years outside the available range are clamped to the nearest row in the lookup.
    Ages outside the available range are clamped to the nearest column (should never happen in practice).

    Parameters
    ----------
    age : int or jax.Array
        Age(s) to retrieve weights for.

    time : int or jax.Array
        Year(s) corresponding to the rows in the lookup.

    age_weight_lookup : jax.Array
        Precomputed 2D array of shape (n_years, n_ages) from build_age_weight_lookup.

    year0 : int
        First year in the lookup array (row 0).

    Returns
    -------
    jax.Array
        Relative population weight(s) corresponding to the given age(s) and time(s).
    """
    n_years, n_ages = age_weight_lookup.shape

    # Compute indices
    year_idx = jnp.asarray(time.astype(jnp.int32)) - year0
    age_idx = jnp.asarray(age)

    # Clamp indices to valid range
    year_idx = jnp.clip(year_idx, 0, n_years - 1)
    age_idx = jnp.clip(age_idx, 0, n_ages - 1)

    # Retrieve weights
    weight = age_weight_lookup[year_idx, age_idx]

    return weight
