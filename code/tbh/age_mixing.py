
from jax import numpy as jnp
from jax import vmap
from summer2.parameters import Parameter
from tbh.demographic_tools import get_relative_age_weight, get_agegap_prob, get_normalised_fertility_data, get_single_age_population


# def gen_age_mixing_matrix_func(iso3, age_groups, max_age=120):
#     normalised_fertility_data = get_normalised_fertility_data(iso3)

#     single_age_pop_df = get_single_age_population(iso3, max_age=max_age)
#     # Convert age group labels to integers
#     age_lb = jnp.array([int(a) for a in age_groups])
#     n_groups = len(age_lb)

#     def age_mixing_matrix_func(a_spread, pc_strength, time):
#         # Build symmetric per-pair contact matrix S, representing contact rate per pair of individuals
#         S = jnp.zeros((n_groups, n_groups))
#         for i in range(n_groups):
#             print("i:", i)  
#             lb_i = int(age_groups[i])
#             ub_i = int(age_groups[i+1]) - 1 if i < len(age_groups) - 1 else max_age
#             for j in range(i, n_groups):
#                 lb_j = int(age_groups[j])
#                 ub_j = int(age_groups[j+1]) - 1 if j < len(age_groups) - 1 else max_age
                
#                 assortative_component, pc_component = 0., 0.
#                 for age_i in range(lb_i, ub_i + 1):
#                     age_i_weight = get_relative_age_weight(age_i, lb_i, ub_i + 1, single_age_pop_df, time)
#                     for age_j in range(lb_j, ub_j + 1):
#                         age_j_weight = get_relative_age_weight(age_j, lb_j, ub_j + 1, single_age_pop_df, time)  
#                         age_gap = abs(age_i - age_j)
                        
#                         assortative_component = assortative_component + age_i_weight * age_j_weight * (1. / a_spread) * jnp.exp(-age_gap / a_spread)

#                         child_age = min(age_i, age_j)

#                         # compute index as JAX integer
#                         year_index = jnp.floor(time - child_age).astype(jnp.int32)
#                         pc_component = pc_component + age_i_weight * age_j_weight * get_agegap_prob(normalised_fertility_data, age_gap, year_index)
                
#                 S_i_j = assortative_component + pc_strength * pc_component
#                 S = S.at[i, j].set(S_i_j).at[j, i].set(S_i_j)

#         # Convert to asymmetric matrix C, representing number of contacts per individual
#         pop_sizes = {age: 1000. for age in age_groups}

#         C = S * jnp.array([pop_sizes[age] for age in age_groups])[:, None]

#         # Rescale C so its spectral radius is 1
#         eigvals = jnp.linalg.eigvals(C)
#         spectral_radius = jnp.max(jnp.abs(eigvals))
#         normalised_C = C / spectral_radius

#         return normalised_C
    
#     return age_mixing_matrix_func
def gen_age_mixing_matrix_func(iso3, age_groups, max_age=120):
    """
    Returns a function age_mixing_matrix_func(a_spread, pc_strength, time)
    ready to be called by the model framework.
    All static computations are done here, only things that depend on
    time and the parameters are computed inside the returned function.
    """
    # --- Static precomputation ---
    normalised_fertility_data = get_normalised_fertility_data(iso3)
    single_age_pop_df = get_single_age_population(iso3, max_age=max_age)
    n_groups = len(age_groups)
    all_ages = jnp.arange(0, max_age + 1)

    # Precompute group masks for aggregation
    group_masks = []
    for i in range(n_groups):
        lb_i = int(age_groups[i])
        ub_i = int(age_groups[i+1]) - 1 if i < n_groups - 1 else max_age
        for j in range(n_groups):
            lb_j = int(age_groups[j])
            ub_j = int(age_groups[j+1]) - 1 if j < n_groups - 1 else max_age
            row_mask = (all_ages[:, None] >= lb_i) & (all_ages[:, None] <= ub_i)
            col_mask = (all_ages[None, :] >= lb_j) & (all_ages[None, :] <= ub_j)
            group_masks.append((i, j, row_mask & col_mask))

    # --- Runtime function ---
    def age_mixing_matrix_func(a_spread, pc_strength, time):
        # Compute age weights at this time
        # age_weights = jnp.array([
        #     get_relative_age_weight(age, 0, max_age + 1, single_age_pop_df, time)
        #     for age in all_ages
        # ])

        v_get_relative_age_weight = vmap(lambda age: get_relative_age_weight(age, 0, max_age+1, single_age_pop_df, time))
        age_weights = v_get_relative_age_weight(jnp.arange(0, max_age+1))


        # Assortative component
        age_i = all_ages[:, None]
        age_j = all_ages[None, :]
        age_gap = jnp.abs(age_i - age_j)
        weight_matrix = age_weights[:, None] * age_weights[None, :]
        assortative_matrix = weight_matrix * (1. / a_spread) * jnp.exp(-age_gap / a_spread)

        # Fertility (parent-child) component
        child_age = jnp.minimum(age_i, age_j)
        year_index = jnp.floor(time - child_age).astype(jnp.int32)
        year_index = jnp.clip(year_index, 0, normalised_fertility_data.shape[0] - 1)

        # Compute pc_matrix
        pc_matrix = jnp.zeros_like(age_gap, dtype=jnp.float32)
        for gap in range(max_age + 1):
            mask = age_gap == gap
            pc_matrix = pc_matrix + mask * jnp.array([
                get_agegap_prob(normalised_fertility_data, gap, yi)
                for yi in year_index[mask]
            ])

        # Total age-age contact matrix
        total_matrix = assortative_matrix + pc_strength * pc_matrix

        # Aggregate by age groups
        S = jnp.zeros((n_groups, n_groups))
        for i, j, mask in group_masks:
            S_ij = jnp.sum(total_matrix[mask])
            S = S.at[i, j].set(S_ij).at[j, i].set(S_ij)

        # Convert to asymmetric matrix C
        pop_sizes = jnp.array([1000. for _ in age_groups])
        C = S * pop_sizes[:, None]

        # Normalize spectral radius
        eigvals = jnp.linalg.eigvals(C)
        spectral_radius = jnp.max(jnp.abs(eigvals))
        normalised_C = C / spectral_radius

        return normalised_C

    return age_mixing_matrix_func