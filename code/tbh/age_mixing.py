from jax import numpy as jnp
from jax import vmap
import pandas as pd

from summer2.parameters import Parameter, Function, Time

from tbh.demographic_tools import get_age_weight_jax, get_agegap_prob_jax
from tbh.demographic_tools import build_agegap_lookup, build_age_weight_lookup
from tbh.paths import DATA_FOLDER


def gen_mixing_matrix_func(grouped_pop_df, fert_probs, fert_year0, fert_age0, age_weights_lookup, ageweights_year0, age_groups, max_age=120):
    """
    Generates a function to build the age mixing matrix based on parameters and time. 
    The mixing matrix incorporates background mixing, assortative mixing by age, and parent-child mixing based on fertility rates by age.

    Parameters
    ----------
    grouped_pop_df : pd.DataFrame
        DataFrame with population data grouped by age.
    fert_probs : jnp.array
        Fertility probabilities lookup table.
    fert_year0 : int
        Starting year for fertility data.
    fert_age0 : int
        Starting age for fertility data.
    age_weights_lookup : jnp.array
        Age weights lookup table.
    ageweights_year0 : int
        Starting year for age weights data.
    age_groups : list of str
        List of age group labels.
    max_age : int, optional
        Maximum age considered in the model, by default 120.
    Returns
    -------
    build_mixing_matrix : Function
        A function that builds the age mixing matrix given parameters and time. 
    """
    
    # Convert age group labels to integers
    age_lb = jnp.array([int(a) for a in age_groups])
    n_groups = len(age_lb)

    grouped_pop_df_jaxarray = jnp.array(grouped_pop_df) # row for years col for age_groups
    pop_df_year0 = grouped_pop_df.index.min()
    n_years = grouped_pop_df_jaxarray.shape[0]

    def get_group_popsize_jax(time):
        year_idx = time.astype(jnp.int32) - pop_df_year0
        year_idx = jnp.clip(year_idx, 0, n_years - 1)
        return grouped_pop_df_jaxarray[year_idx, :][:, None]

    def build_mixing_matrix(bg_mixing, a_spread, pc_strength, time):
        
        # Build symetric matrix S with rates of contact PER PAIR of individuals
        S = jnp.zeros((n_groups, n_groups))
        for i in range(n_groups):
            # List all relevant single ages in group i and calculate their population weights relative to group i
            lb_i = int(age_groups[i])
            ub_i = int(age_groups[i + 1]) - 1 if i < len(age_groups) - 1 else max_age
            ages_i = jnp.arange(lb_i, ub_i + 1)
            w_i = vmap(lambda a: get_age_weight_jax(a, time, age_weights_lookup, ageweights_year0))(ages_i)

            for j in range(i, n_groups):
                # List all relevant single ages in group j and calculate their population weights relative to group j
                lb_j = int(age_groups[j])
                ub_j = int(age_groups[j + 1]) - 1 if j < len(age_groups) - 1 else max_age
                ages_j = jnp.arange(lb_j, ub_j + 1)
                w_j = vmap(lambda a: get_age_weight_jax(a, time, age_weights_lookup, ageweights_year0))(ages_j)

                # Outer product of weights
                weight_prod = w_i[:, None] * w_j[None, :]

                # Age gap and child age matrices
                age_gap_mat = jnp.abs(ages_i[:, None] - ages_j[None, :])
                child_age_mat = jnp.minimum(ages_i[:, None], ages_j[None, :])

                # Assortative component (exponential decay with age gap)
                assortative_component = jnp.sum(weight_prod * (1.0 / a_spread) * jnp.exp(-age_gap_mat / a_spread))

                # Parent-child component (using fertility rates by age)
                pc_component = jnp.sum(weight_prod * get_agegap_prob_jax(fert_probs, fert_year0, fert_age0, time - child_age_mat, age_gap_mat))

                # Set S[i,j] as sum of Background, assortative and parent-chuld components (as well as symmetric S[j,i])
                value = bg_mixing + assortative_component + pc_strength * pc_component
                S = S.at[i, j].set(value)
                S = S.at[j, i].set(value)

        # Scale by population size of each age group      
        C = S * get_group_popsize_jax(time) # matrix is now asymmetric

        # Normalize C by its spectral radius
        eigvals = jnp.linalg.eigvals(C)
        spectral_radius = jnp.max(jnp.abs(eigvals))
        normalised_C = C / spectral_radius

        return normalised_C
    
    return build_mixing_matrix


def get_model_ready_age_mixing_matrix(iso3: str, age_groups: list[str], grouped_pop_df: pd.DataFrame, single_age_pop_df: pd.DataFrame):
    """
    Prepares the age mixing matrix function for use in the model. 
    Loads fertility data, builds lookup tables, and constructs the mixing matrix function. 
    Parameters
    ----------
    iso3 : str
        ISO3 country code for loading fertility data.
    age_groups : list of str
        List of age group labels.
    grouped_pop_df : pd.DataFrame
        DataFrame with population data grouped by age.
    single_age_pop_df : pd.DataFrame
        DataFrame with single age population data.
    Returns
    -------
    age_mixing_matrix : Function
        A Summer2 Function that generates the age mixing matrix based on parameters and time.
    """
    fertility_data = pd.read_csv(DATA_FOLDER / f"un_fertility_rates_{iso3}.csv",index_col=0)
    normalised_fertility_data = fertility_data.div(fertility_data.sum(axis=1), axis=0)
    fert_probs, fert_year0, fert_age0 = build_agegap_lookup(normalised_fertility_data)
    age_weights_lookup, ageweights_year0 = build_age_weight_lookup(age_groups, single_age_pop_df)

    build_mixing_matrix = gen_mixing_matrix_func(grouped_pop_df, fert_probs, fert_year0, fert_age0, age_weights_lookup, ageweights_year0, age_groups)
    age_mixing_matrix = Function(build_mixing_matrix, [Parameter("bg_mixing"), Parameter("a_spread"), Parameter("pc_strength"), Time]) # the function generating the matrix
    
    return age_mixing_matrix