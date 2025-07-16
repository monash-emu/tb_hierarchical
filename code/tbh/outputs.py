
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