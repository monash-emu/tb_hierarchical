
from summer2 import CompartmentalModel
from summer2.parameters import Parameter, DerivedOutput


def request_model_outputs(model: CompartmentalModel, compartments: list):
    """
    Define model outputs that can later be requested from model.get_derived_outputs_df()

    Args:
        model (CompartmentalModel): the 'fully-built' TB model
        compartments (list): list of all model compartments (required to create population size output)
        studies (list): list of studies (required to disaggregate outputs by study)
    """
    model.request_output_for_compartments(
        name=f"population", compartments=compartments
    )

    model.request_output_for_flow(
        name="raw_incidence",
        flow_name="progression",
    )

    for infectious_status in ["inf", "noninf"]:
        model.request_output_for_flow(
            name=f"tb_mortality_{infectious_status}",
            flow_name=f"tb_mortality_{infectious_status}"
        )
  
    model.request_aggregate_output(
        name="raw_tb_mortality",
        sources=[f"tb_mortality_{infectious_status}" for infectious_status in ["inf", "noninf"]] 
    )