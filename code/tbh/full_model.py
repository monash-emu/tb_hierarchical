from summer2 import CompartmentalModel, Stratification
from summer2.parameters import Parameter, DerivedOutput
from summer2.functions import time as stf

from tbh.outputs import request_model_outputs

from jax import numpy as jnp
import yaml
from pathlib import Path
from typing import Callable

HOME_PATH = Path(__file__).parent.parent.parent

PLACEHOLDER_VALUE = 0.

COMPARTMENTS = (
    "mtb_naive",
    "incipient",
    "contained",
    "cleared",
    "subclin_noninf",
    "clin_noninf",
    "subclin_inf",
    "clin_inf",
    "treatment",
    "recovered"
)

INFECTIOUS_COMPARTMENTS = ("subclin_inf", "clin_inf")


def get_tb_model(model_config: dict, home_path=HOME_PATH):
    """
    Prepare time-variant parameters and other quantities requiring pre-processsing
    """

    model = get_natural_tb_model(model_config)
    
    # add_detection_and_treatment(model)

    # stratify by age

    request_model_outputs(model, COMPARTMENTS)

    return model


def get_natural_tb_model(model_config):

    model = CompartmentalModel(
        times=(model_config["start_time"], model_config["end_time"]),
        compartments=COMPARTMENTS,
        infectious_compartments=INFECTIOUS_COMPARTMENTS,
    )

    model.set_initial_population(
        distribution={
            "mtb_naive": Parameter("init_pop_size") - model_config["seed"],
            "clin_inf": model_config["seed"],
        },
    )

    # Natural mortality modelled as transition back to mtb_naive compartment
    #FIXME! will need to make sure all goes to children after age-stratification
    #FIXME!: will need to add flow from mtb_naiveXadult to mtb_naiveXchild after age-stratification
    for compartment in [c for c in COMPARTMENTS if c != "mtb_naive"]:
        model.add_transition_flow(
            name=f"all_cause_mortality_from_{compartment}",
            fractional_rate=PLACEHOLDER_VALUE,  #FIXME! placeholder
            source=compartment,
            dest="mtb_naive",
        )

    # Excess birth to capture population growth
    #FIXME! will need to make sure all goes to children after age-stratification
    model.add_crude_birth_flow(
        name="excess_birth",
        birth_rate=PLACEHOLDER_VALUE, #FIXME! placeholder
        dest="mtb_naive"
    )


    # Early TB infection flows
    # model.add_transition_flow(
    #     name=,
    #     fractional_rate=,
    #     source=,
    #     dest=
    # )

    return model

