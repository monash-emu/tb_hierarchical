from jax import numpy as jnp
from jax import vmap

from tbh.demographic_tools import get_age_weight_jax, get_agegap_prob_jax


def gen_mixing_matrix_func(grouped_pop_df, fert_probs, fert_year0, fert_age0, age_weights_lookup, ageweights_year0, age_groups, max_age=120):
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
        S = jnp.zeros((n_groups, n_groups))  # JAX array

        for i in range(n_groups):
            lb_i = int(age_groups[i])
            ub_i = int(age_groups[i + 1]) - 1 if i < len(age_groups) - 1 else max_age
            ages_i = jnp.arange(lb_i, ub_i + 1)
            w_i = vmap(lambda a: get_age_weight_jax(a, time, age_weights_lookup, ageweights_year0))(ages_i)

            for j in range(i, n_groups):
                lb_j = int(age_groups[j])
                ub_j = int(age_groups[j + 1]) - 1 if j < len(age_groups) - 1 else max_age
                ages_j = jnp.arange(lb_j, ub_j + 1)
                w_j = vmap(lambda a: get_age_weight_jax(a, time, age_weights_lookup, ageweights_year0))(ages_j)

                # Outer product of weights
                weight_prod = w_i[:, None] * w_j[None, :]

                # Age gap and child age matrices
                age_gap_mat = jnp.abs(ages_i[:, None] - ages_j[None, :])
                child_age_mat = jnp.minimum(ages_i[:, None], ages_j[None, :])

                # Assortative component
                assortative_component = jnp.sum(weight_prod * (1.0 / a_spread) * jnp.exp(-age_gap_mat / a_spread))

                # Parent-child component
                pc_component = jnp.sum(weight_prod * get_agegap_prob_jax(fert_probs, fert_year0, fert_age0, time - child_age_mat, age_gap_mat))

                # Set S[i,j] and symmetric S[j,i]
                value = bg_mixing + assortative_component + pc_strength * pc_component
                S = S.at[i, j].set(value)
                S = S.at[j, i].set(value)

        # Scale by population size of each age group      
        C = S * get_group_popsize_jax(time)  #pop_sizes[:, None]  # asymmetric contacts

        # Normalize C by its spectral radius
        eigvals = jnp.linalg.eigvals(C)
        spectral_radius = jnp.max(jnp.abs(eigvals))
        normalised_C = C / spectral_radius

        return normalised_C
    
    return build_mixing_matrix