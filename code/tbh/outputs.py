
from summer2 import CompartmentalModel
from summer2.parameters import Parameter, DerivedOutput


def request_model_outputs(model: CompartmentalModel, compartments: list, active_compartments: list, latent_compartments: list, nat_death_flows: list, tb_death_flows: list, screening_flows: list):
    """
    Define model outputs that can later be requested from model.get_derived_outputs_df()

    Args:
        model (CompartmentalModel): the 'fully-built' TB model
        compartments (list): list of all model compartments (required to create population size output)
        active_compartments: list of active TB compartments
        tb_death_flows: list of flow names for TB mortality
    """
    age_strata = model.stratifications['age'].strata

    # Population size (incl. age-specific)
    model.request_output_for_compartments(
        name=f"population", compartments=compartments
    )
    for age in age_strata:
        model.request_output_for_compartments(
            name=f"populationXage_{age}", compartments=compartments, strata={"age": age}
        )
    model.request_aggregate_output(
        name="populationXage_3_9", sources=[f"populationXage_{age}" for age in ['3', '5']]
    )
    model.request_aggregate_output(
        name="populationXage_15+", sources=[f"populationXage_{age}" for age in age_strata if int(age) >= 15]
    )
    model.request_aggregate_output(
        name="populationXage_18+", sources=[f"populationXage_{age}" for age in age_strata if int(age) >= 18]
    )

    model.request_output_for_flow("births", "births")

    # TB incidence (and cumulative)
    for inf_cat in ["lowinf", "inf"]:
        model.request_output_for_flow(
            name=f"tb_incidence_{inf_cat}",
            flow_name=f"progression_{inf_cat}",
        )
    model.request_aggregate_output(
        name=f"tb_incidence",
        sources=[f"tb_incidence_{inf_cat}" for inf_cat in ["lowinf", "inf"]]
    )    
    request_per_capita_output(model, "tb_incidence", per=100000.)
    model.request_cumulative_output(name="cum_tb_incidence", source="tb_incidence", start_time=2026)

    """ 
        Prevalence outputs (TB and TBI)
    """
    # True absolute prevalence (i.e. compartment sizes) for TB and TBI
    for comp in latent_compartments + active_compartments:
        for age in age_strata:
            model.request_output_for_compartments(
                name=f"prev_{comp}Xage_{age}", compartments=comp, strata={"age": age}, save_results=False
            )           
        model.request_aggregate_output(name=f"prev_{comp}", sources=[f"prev_{comp}Xage_{age}" for age in age_strata], save_results=False)
    # True per-capita prevalence for TB and TBI
    for state, comp_list, per in zip(["tbi", "tb"], [latent_compartments, active_compartments], [100., 100000.]):
        model.request_aggregate_output(
                name=f"{state}_prevalence", sources=[f"prev_{comp}" for comp in comp_list]
        )
        request_per_capita_output(model, f"{state}_prevalence", per=per)


    # Measured n TST positive (accounting for compartment-specific sensitivity for TST) 
    for comp in latent_compartments:
        for age in age_strata:
            model.request_function_output(
                name=f"tst_pos_{comp}Xage_{age}", func=DerivedOutput(f"prev_{comp}Xage_{age}") * Parameter(f"prev_se_{comp}"), save_results=False
            )
        # manually add '3-9', '15+' and '18+' age group
        model.request_aggregate_output(
            name=f"tst_pos_{comp}Xage_3_9", sources=[f"tst_pos_{comp}Xage_{age}" for age in ['3', '5']]
        )
        model.request_aggregate_output(
            name=f"tst_pos_{comp}Xage_15+", sources=[f"tst_pos_{comp}Xage_{age}" for age in age_strata if int(age) >= 15]
        )
        model.request_aggregate_output(
            name=f"tst_pos_{comp}Xage_18+", sources=[f"tst_pos_{comp}Xage_{age}" for age in age_strata if int(age) >= 18]
        )

    # Per-capita TST positivity for each age-group and aggregated
    for age in age_strata + ['3_9', '15+', '18+']:
        model.request_aggregate_output(
            name=f"tst_posXage_{age}", sources=[f"tst_pos_{comp}Xage_{age}" for comp in latent_compartments]
        )
        request_per_capita_output(model, f"tst_posXage_{age}", per=100, denominator_output=f"populationXage_{age}")           
    
    model.request_aggregate_output(
        name="tst_pos", sources=[f"tst_posXage_{age}" for age in age_strata]
    )
    request_per_capita_output(model, "tst_pos", per=100)


    # Measured PEARL (i.e. Xpert) and CXR positivity (accounting for compartment-specific sensitivity for different tests) 
    for comp in active_compartments:
        for age in age_strata:
            for test in ['pearl', 'cxr']:
                model.request_function_output(
                    name=f"{test}_prev_{comp}Xage_{age}", func=DerivedOutput(f"prev_{comp}Xage_{age}") * Parameter(f"prev_se_{comp}_{test}"), save_results=False
                )

    # Per-capita PEARL and CXR positivity for each age-group and aggregated
    for test in ['pearl', 'cxr']:
        for age in age_strata:
            model.request_aggregate_output(
                name=f"{test}_posXage_{age}", sources=[f"{test}_prev_{comp}Xage_{age}" for comp in active_compartments]
            )
            request_per_capita_output(model, f"{test}_posXage_{age}", per=100000., denominator_output=f"populationXage_{age}")           
        model.request_aggregate_output(
            name=f"{test}_pos", sources=[f"{test}_posXage_{age}" for age in age_strata]
        )
        request_per_capita_output(model, f"{test}_pos", per=100000.)


    # Prevalence of viable TB infection ('incipient' and 'contained')
    model.request_aggregate_output(
        name="viable_tbi_prevalence", sources=[f"prev_{comp}" for comp in ['incipient', 'contained']]
    )
    request_per_capita_output(model, "viable_tbi_prevalence", per=100.)           

    # Percentage subclinical (compare with Frascella et al. CID 2020 doi: 10.1093/cid/ciaa1402)
    model.request_output_for_compartments(
        name="subclin_tb_prevalence",
        compartments=[c for c in active_compartments if c.startswith('subclin_')],
        save_results=False
    )
    model.request_function_output(
        name="perc_prev_subclinical", 
        func= 100. * DerivedOutput("subclin_tb_prevalence") / DerivedOutput("tb_prevalence")
    )

    # Percentage infectious prevalence
    model.request_output_for_compartments(
        name="infectious_tb_prevalence",
        compartments=[c for c in active_compartments if c.endswith('_inf')],
        save_results=False
    )
    model.request_function_output(
        name="perc_prev_infectious", 
        func= 100. * DerivedOutput("infectious_tb_prevalence") / DerivedOutput("tb_prevalence")
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


    # Track computed values 
    computed_values_to_save = ['passive_detection_rate_clin', 'passive_detection_rate_subclin']
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