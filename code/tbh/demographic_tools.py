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
    Build JAX-friendly lookup structures.

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
    year_idx = birth_year - year0
    age_idx = age_gap - age0

    # Clamp years to nearest available year
    year_idx = jnp.clip(year_idx, 0, n_years - 1)

    # Mask invalid age gaps
    valid_age = (age_idx >= 0) & (age_idx < n_ages)

    # Safe indexing
    age_idx_safe = jnp.clip(age_idx, 0, n_ages - 1)

    prob = probs[year_idx, age_idx_safe]

    return jnp.where(valid_age, prob, 0.0)




def gen_mixing_matrix_func(age_groups):
    """
    Returns a JAX-compatible function to build a symmetric age-structured mixing matrix
    for a given set of age group lower bounds, with configurable socialising parameters.

    Parameters
    ----------
    age_groups : list of int
        List of lower bounds of age intervals, e.g. [0, 5, 15, 50].

    Returns
    -------
    function
        A function that takes child_socialising and elderly_socialising parameters and returns
        an (n_groups x n_groups) mixing matrix as a JAX array.
    """
    age_groups = np.array(age_groups, dtype=int)

    # Determine socialising parameter for each group
    def build_mixing_matrix(child_socialising, elderly_socialising):
        """
        Constructs a symmetric mixing matrix where each age group has a socialising parameter.
        Socialising parameter model:
        Each age group is assigned a socialising parameter that reflects their relative level of social contacts. 
        - Children (<15 years) use `child_socialising`.
        - Middle-aged adults (15–64 years) have baseline socialising = 1.0.
        - Elderly (≥65 years) use `elderly_socialising`.

        The mixing matrix is constructed such that:
        - Diagonal elements (within-group contacts) equal the group's socialising parameter.
        - Off-diagonal elements (between-group contacts) equal the product of the socialising parameters of the two groups.

        This assumes that contact intensity between two groups is proportional to the product of their social activity levels.

        Parameters
        ----------
        child_socialising : float
            Socialising factor for children (age < 15)
        elderly_socialising : float
            Socialising factor for elderly (age >= 65)

        Returns
        -------
        matrix : jax.Array (n_groups x n_groups)
        """
        # Assign socialising parameters per age group
        socialising = jnp.array([
            child_socialising if age < 15 else
            elderly_socialising if age >= 65 else
            1.0
            for age in age_groups
        ])

        # Construct the mixing matrix: outer product
        M = jnp.outer(socialising, socialising)
        # Compute spectral radius (largest absolute eigenvalue)
        rho = jnp.max(jnp.abs(eigvals(M)))

        # Rescale so spectral radius = 1
        M = M / rho
        return M

    return build_mixing_matrix