
from summer2 import CompartmentalModel
from summer2.parameters import Parameter, DerivedOutput


def request_model_outputs(model: CompartmentalModel, compartments: list, active_compartments: list, latent_compartments: list, nat_death_flows: list, tb_death_flows: list, screening_flows: list, model_config: dict):
    """
    Define model outputs that can later be requested from model.get_derived_outputs_df()

    Args:
        model (CompartmentalModel): the 'fully-built' TB model
        compartments (list): list of all model compartments (required to create population size output)
        active_compartments: list of active TB compartments
        tb_death_flows: list of flow names for TB mortality
    """
    age_strata = model.stratifications['age'].strata
    reach_strata = model.stratifications['reachability'].strata
    reachable_stratum = "reachable"
    unreachable_stratum = "unreachable"

    # Population size (incl. age- and subgroup-specific)
    model.request_output_for_compartments(
        name=f"population", compartments=compartments
    )
    for age in age_strata:
        model.request_output_for_compartments(
            name=f"populationXage_{age}", compartments=compartments, strata={"age": age}
        )
        model.request_output_for_compartments(
            name=f"populationXage_{age}Xreach_{reachable_stratum}",
            compartments=compartments,
            strata={"age": age, "reachability": reachable_stratum},
            save_results=False,
        )
    model.request_aggregate_output(
        name="populationXage_3_9", sources=[f"populationXage_{age}" for age in ['3', '5']]
    )
    model.request_aggregate_output(
        name=f"populationXage_3_9Xreach_{reachable_stratum}",
        sources=[f"populationXage_{age}Xreach_{reachable_stratum}" for age in ['3', '5']],
        save_results=False,
    )
    model.request_aggregate_output(
        name="populationXage_15+", sources=[f"populationXage_{age}" for age in age_strata if int(age) >= 15]
    )
    model.request_aggregate_output(
        name=f"populationXage_15+Xreach_{reachable_stratum}",
        sources=[f"populationXage_{age}Xreach_{reachable_stratum}" for age in age_strata if int(age) >= 15],
        save_results=False,
    )
    model.request_aggregate_output(
        name="populationXage_18+", sources=[f"populationXage_{age}" for age in age_strata if int(age) >= 18]
    )
    model.request_aggregate_output(
        name=f"populationXage_18+Xreach_{reachable_stratum}",
        sources=[f"populationXage_{age}Xreach_{reachable_stratum}" for age in age_strata if int(age) >= 18],
        save_results=False,
    )
    for reach in reach_strata:
        model.request_output_for_compartments(
            name=f"populationXreach_{reach}", compartments=compartments, strata={"reachability": reach}
        )

    # Track births (only those incrementally added to the population, i.e. not those that die and get reintroduced back into the population)
    for reach in reach_strata:
        model.request_output_for_flow(f"births_{reach}", f"births_{reach}")
    model.request_aggregate_output(
        name="births", sources=[f"births_{reach}" for reach in reach_strata]
    )
    
    # TB incidence (and cumulative)
    for inf_cat in ["lowinf", "inf"]:
        model.request_output_for_flow(
            name=f"tb_incidence_{inf_cat}",
            flow_name=f"progression_{inf_cat}",
        )
        for reach in reach_strata:
            model.request_output_for_flow(
                name=f"tb_incidence_{inf_cat}Xreach_{reach}",
                flow_name=f"progression_{inf_cat}",
                source_strata={"reachability": reach},
                save_results=False,
            )
    for reach in reach_strata:
        model.request_aggregate_output(
            name=f"tb_incidenceXreach_{reach}",
            sources=[f"tb_incidence_{inf_cat}Xreach_{reach}" for inf_cat in ["lowinf", "inf"]],
        )
    model.request_aggregate_output(
        name=f"tb_incidence",
        sources=[f"tb_incidenceXreach_{reach}" for reach in reach_strata]
    )    
    request_per_capita_output(model, "tb_incidence", per=100000.)
    model.request_function_output(
        name=f"prop_tb_incidenceXreach_{unreachable_stratum}",
        func=DerivedOutput(f"tb_incidenceXreach_{unreachable_stratum}") / DerivedOutput("tb_incidence"),
    )
    model.request_cumulative_output(name="cum_tb_incidence", source="tb_incidence", start_time=2026)

    """ 
        Prevalence outputs (TB and TBI)
    """
    # True absolute prevalence (i.e. compartment sizes) for TB and TBI
    for comp in compartments:
        for age in age_strata:
            model.request_output_for_compartments(
                name=f"prev_{comp}Xage_{age}", compartments=comp, strata={"age": age}, save_results=False
            )
            for reach in reach_strata:
                model.request_output_for_compartments(
                    name=f"prev_{comp}Xage_{age}Xreach_{reach}",
                    compartments=comp,
                    strata={"age": age, "reachability": reach},
                    save_results=False,
                )
        model.request_aggregate_output(name=f"prev_{comp}", sources=[f"prev_{comp}Xage_{age}" for age in age_strata], save_results=False)
        for reach in reach_strata:
            model.request_aggregate_output(
                name=f"prev_{comp}Xreach_{reach}",
                sources=[f"prev_{comp}Xage_{age}Xreach_{reach}" for age in age_strata],
                save_results=False,
            )
    # True per-capita prevalence for TB and TBI
    for state, comp_list, per in zip(["tbi", "tb"], [latent_compartments, active_compartments], [100., 100000.]):
        model.request_aggregate_output(
                name=f"{state}_prevalence", sources=[f"prev_{comp}" for comp in comp_list]
        )
        request_per_capita_output(model, f"{state}_prevalence", per=per)


    # Measured n TST positive (accounting for compartment-specific sensitivity for TST) 
    for comp in compartments:
        if comp in latent_compartments:
            tst_sensitivity = Parameter(f"prev_se_{comp}_tst")
        elif comp in active_compartments or comp == "treatment":
            tst_sensitivity = 1.
        elif comp == "recovered":
            tst_sensitivity = Parameter(f"prev_se_cleared_tst")
        else: # susceptible
            tst_sensitivity = 0.

        for age in age_strata:
            model.request_function_output(
                name=f"tst_pos_{comp}Xage_{age}Xreach_{reachable_stratum}", func=DerivedOutput(f"prev_{comp}Xage_{age}Xreach_{reachable_stratum}") * tst_sensitivity, save_results=False
            )
        # manually add '3-9', '15+' and '18+' age group
        model.request_aggregate_output(
            name=f"tst_pos_{comp}Xage_3_9Xreach_{reachable_stratum}", sources=[f"tst_pos_{comp}Xage_{age}Xreach_{reachable_stratum}" for age in ['3', '5']]
        )
        model.request_aggregate_output(
            name=f"tst_pos_{comp}Xage_15+Xreach_{reachable_stratum}", sources=[f"tst_pos_{comp}Xage_{age}Xreach_{reachable_stratum}" for age in age_strata if int(age) >= 15]
        )
        model.request_aggregate_output(
            name=f"tst_pos_{comp}Xage_18+Xreach_{reachable_stratum}", sources=[f"tst_pos_{comp}Xage_{age}Xreach_{reachable_stratum}" for age in age_strata if int(age) >= 18]
        )

    # Per-capita TST positivity for each age-group and aggregated
    for age in age_strata + ['3_9', '15+', '18+']:
        model.request_aggregate_output(
            name=f"tst_posXage_{age}Xreach_{reachable_stratum}", sources=[f"tst_pos_{comp}Xage_{age}Xreach_{reachable_stratum}" for comp in compartments]
        )
        request_per_capita_output(model, f"tst_posXage_{age}Xreach_{reachable_stratum}", per=100, denominator_output=f"populationXage_{age}Xreach_{reachable_stratum}")
    
    model.request_aggregate_output(
        name=f"tst_posXreach_{reachable_stratum}", sources=[f"tst_posXage_{age}Xreach_{reachable_stratum}" for age in age_strata]
    )
    request_per_capita_output(model, f"tst_posXreach_{reachable_stratum}", per=100, denominator_output=f"populationXreach_{reachable_stratum}")


    # Measured PEARL (i.e. Xpert) and CXR positivity (accounting for compartment-specific sensitivity for different tests) 
    for comp in active_compartments:
        for age in age_strata:
            for test in ['pearl', 'cxr']:
                model.request_function_output(
                    name=f"{test}_prev_{comp}Xage_{age}Xreach_{reachable_stratum}", func=DerivedOutput(f"prev_{comp}Xage_{age}Xreach_{reachable_stratum}") * Parameter(f"prev_se_{comp}_{test}"), save_results=False
                )

    # Per-capita PEARL and CXR positivity for each age-group and aggregated
    for test in ['pearl', 'cxr']:
        for age in age_strata:
            model.request_aggregate_output(
                name=f"{test}_posXage_{age}Xreach_{reachable_stratum}", sources=[f"{test}_prev_{comp}Xage_{age}Xreach_{reachable_stratum}" for comp in active_compartments]
            )
            request_per_capita_output(model, f"{test}_posXage_{age}Xreach_{reachable_stratum}", per=100000., denominator_output=f"populationXage_{age}Xreach_{reachable_stratum}")
        model.request_aggregate_output(
            name=f"{test}_posXreach_{reachable_stratum}", sources=[f"{test}_posXage_{age}Xreach_{reachable_stratum}" for age in age_strata]
        )
        request_per_capita_output(model, f"{test}_posXreach_{reachable_stratum}", per=100000., denominator_output=f"populationXreach_{reachable_stratum}")


    # Prevalence of viable TB infection ('incipient' and 'contained')
    model.request_aggregate_output(
        name="viable_tbi_prevalence", sources=[f"prev_{comp}" for comp in ['incipient', 'contained']]
    )
    request_per_capita_output(model, "viable_tbi_prevalence", per=100.)           

    # Percentage subclinical (compare with Frascella et al. CID 2020 doi: 10.1093/cid/ciaa1402)
    model.request_output_for_compartments(
        name=f"subclin_tb_prevalenceXreach_{reachable_stratum}",  #FIXME
        compartments=[c for c in active_compartments if c.startswith('subclin_')],
        strata={"reachability": reachable_stratum},
        save_results=False
    )
    for reach in reach_strata:
        model.request_aggregate_output(
            name=f"tb_prevalenceXreach_{reach}",
            sources=[f"prev_{comp}Xreach_{reach}" for comp in active_compartments],
        )
        request_per_capita_output(
            model,
            f"tb_prevalenceXreach_{reach}",
            per=100000.,
            denominator_output=f"populationXreach_{reach}",
        )
    model.request_function_output(
        name="perc_prev_subclinicalXreach_reachable", 
        func= 100. * DerivedOutput(f"subclin_tb_prevalenceXreach_{reachable_stratum}") / DerivedOutput(f"tb_prevalenceXreach_{reachable_stratum}")
    )

    # Percentage infectious prevalence
    model.request_output_for_compartments(
        name=f"infectious_tb_prevalenceXreach_{reachable_stratum}",
        compartments=[c for c in active_compartments if c.endswith('_inf')],
        strata={"reachability": reachable_stratum},
        save_results=False
    )
    model.request_function_output(
        name="perc_prev_infectiousXreach_reachable", 
        func= 100. * DerivedOutput(f"infectious_tb_prevalenceXreach_{reachable_stratum}") / DerivedOutput(f"tb_prevalenceXreach_{reachable_stratum}")
    )

    # TB notifications
    for active_comp in active_compartments:
        model.request_output_for_flow(
            name=f"notifications_{active_comp}",
            flow_name=f"tb_detection_{active_comp}",
            save_results=False
        )
    model.request_aggregate_output(
        name="notifications",
        sources=[f"notifications_{active_comp}" for active_comp in active_compartments]
    )
    model.request_function_output(
        name="perc_notifications_clin",
        func= 100. * (DerivedOutput("notifications_clin_lowinf") + DerivedOutput("notifications_clin_inf")) / DerivedOutput("notifications")
    )

    # Screening
    for scr_flow in screening_flows:
        model.request_output_for_flow(
            name=scr_flow,
            flow_name=scr_flow,
            save_results=False
        )
    model.request_aggregate_output(
        name="screening",
        sources=screening_flows
    )

    # Mortality
    for nat_death_flow in nat_death_flows:
        model.request_output_for_flow(
            name=nat_death_flow,
            flow_name=nat_death_flow,
            save_results=False
        )
    model.request_aggregate_output(
        name="nat_mortality",
        sources=nat_death_flows
    )

    for tb_death_flow in tb_death_flows:
        model.request_output_for_flow(
            name=tb_death_flow,
            flow_name=tb_death_flow,
            save_results=False
        )
    model.request_aggregate_output(
        name="tb_mortality",
        sources=tb_death_flows
    )
    request_per_capita_output(model, "tb_mortality", per=100000.)

    model.request_cumulative_output(name="cum_tb_mortality", source="tb_mortality", start_time=2026)

    # Track computed values for passive case detection and mixign matrix distance
    computed_values_to_save = ['passive_detection_rate_clin', 'passive_detection_rate_subclin']
    if model_config.get("heterogeneous_mixing", True):
        computed_values_to_save.append('mixing_matrix_distance')
    for comp_val in computed_values_to_save:
        model.request_computed_value_output(comp_val)


def request_per_capita_output(model: CompartmentalModel, output, per=100., denominator_output="population"):

    if per == 100.:
        suffix = "perc"
    elif per == 100000.:
        suffix = "per100k"
    else:
        suffix = f"per{per}"

    model.request_function_output(
        name=f"{output}_{suffix}", 
        func= per * DerivedOutput(output) / DerivedOutput(denominator_output)
    )