
from summer2 import CompartmentalModel
from summer2.parameters import Parameter, DerivedOutput


def request_model_outputs(model: CompartmentalModel, compartments: list, active_compartments: list, nat_death_flows: list, tb_death_flows: list):
    """
    Define model outputs that can later be requested from model.get_derived_outputs_df()

    Args:
        model (CompartmentalModel): the 'fully-built' TB model
        compartments (list): list of all model compartments (required to create population size output)
        active_compartments: list of active TB compartments
        tb_death_flows: list of flow names for TB mortality
    """
    model.request_output_for_compartments(
        name=f"population", compartments=compartments
    )

    model.request_output_for_flow("births", "births")

    # TB incidence
    model.request_output_for_flow(
        name="tb_incidence",
        flow_name="progression",
    )
    request_per_capita_output(model, "tb_incidence", per=100000.)


    # TBI prevalence
    model.request_output_for_compartments(
        name="tbi_prevalence",
        compartments=["incipient", "contained", "cleared"]  # FIXME, we might want to use different assumptions
    )
    request_per_capita_output(model, "tbi_prevalence", per=100.)


    # TB prevalence
    model.request_output_for_compartments(
        name="tb_prevalence",
        compartments=active_compartments
    )
    request_per_capita_output(model, "tb_prevalence", per=100000.)

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
        func= 100. * (DerivedOutput("notifications_clin_noninf") + DerivedOutput("notifications_clin_inf")) / DerivedOutput("notifications")
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

    # Track computed values 
    computed_values_to_save = ['passive_detection_rate_clin', 'passive_detection_rate_subclin']
    for comp_val in computed_values_to_save:
        model.request_computed_value_output(comp_val)


def request_per_capita_output(model: CompartmentalModel, output, per=100.):

    if per == 100.:
        suffix = "perc"
    elif per == 100000.:
        suffix = "per100k"
    else:
        suffix = f"per{per}"

    model.request_function_output(
        name=f"{output}_{suffix}", 
        func= per * DerivedOutput(output) / DerivedOutput("population")
    )