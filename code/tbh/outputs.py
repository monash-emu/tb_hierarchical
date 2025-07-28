
from summer2 import CompartmentalModel
from summer2.parameters import Parameter, DerivedOutput


def request_model_outputs(model: CompartmentalModel, compartments: list, active_compartments: list, tb_death_flows: list):
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

    # TB incidence
    model.request_output_for_flow(
        name="raw_incidence",
        flow_name="progression",
    )

    # TBI prevalence
    model.request_output_for_compartments(
        name="tbi_prevalence",
        compartments=["incipient", "contained", "cleared"]  # FIXME, we might want to use different assumptions
    )

    # TB prevalence
    model.request_output_for_compartments(
        name="tb_prevalence",
        compartments=active_compartments
    )

    request_per_capita_output(model, "tb_prevalence", per=100000.)

    # TB notifications
    for active_comp in active_compartments:
        model.request_output_for_flow(
            name=f"notifications_{active_comp}",
            flow_name=f"tb_detection_{active_comp}"
        )
    model.request_aggregate_output(
        name="notifications",
        sources=[f"notifications_{active_comp}" for active_comp in active_compartments]
    )

    # Mortality
    for tb_death_flow in tb_death_flows:
        model.request_output_for_flow(
            name=tb_death_flow,
            flow_name=tb_death_flow,
            save_results=False
        )
    model.request_aggregate_output(
        name="raw_tb_mortality",
        sources=tb_death_flows
    )


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