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
    "mtb_naive", # historically called "susceptible"
    # Early TB infection states
    "incipient",
    "contained",
    "cleared",
    # TB disease states
    "subclin_noninf",
    "clin_noninf",
    "subclin_inf",
    "clin_inf",
    # Treatment and recovered
    "treatment",
    "recovered"
)

INFECTIOUS_COMPARTMENTS = ("subclin_inf", "clin_inf")
ACTIVE_COMPS = ["subclin_noninf", "clin_noninf", "subclin_inf", "clin_inf"]


def get_tb_model(model_config: dict, home_path=HOME_PATH):

    model = get_natural_tb_model(model_config)
    
    add_detection_and_treatment(model)

    # stratify by age

    request_model_outputs(model, COMPARTMENTS, ACTIVE_COMPS)

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

    # All-cause non-TB mortality modelled as transition back to mtb_naive compartment
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

    # Transmission flows (including reinfection)
    for susceptible_comp in ["mtb_naive", "contained", "cleared","recovered"]:
        rel_susceptibility = Parameter(f"rel_sus_{susceptible_comp}")
        model.add_infection_frequency_flow(
            name=f"infection_from_{susceptible_comp}",
            contact_rate=Parameter("raw_transmission_rate") * rel_susceptibility,  # will be adjusted later by study
            source=susceptible_comp,
            dest="incipient",
        )

    """
         Early TB infection dynamics
    """
    model.add_transition_flow(
        name="containment",
        fractional_rate=Parameter("containment_rate"),
        source="incipient",
        dest="contained",
    )
    model.add_transition_flow(
        name="clearance",
        fractional_rate=Parameter("clearance_rate"),
        source="contained",
        dest="cleared",
    )
    model.add_transition_flow(
        name="breakdown",
        fractional_rate=Parameter("breakdown_rate"),
        source="contained",
        dest="incipient",
    )    
    model.add_transition_flow(
        name="progression",
        fractional_rate=Parameter("progression_rate"),
        source="incipient",
        dest="subclin_noninf",
    )

    """
        Active TB dynamics
    """
    # Clinical progression and regression flows
    for infectious_status in ["inf", "noninf"]:
        model.add_transition_flow(
            name=f"clinical_progression_{infectious_status}",
            fractional_rate=Parameter("clinical_progression_rate"),
            source=f"subclin_{infectious_status}",
            dest=f"clin_{infectious_status}",
        )
        model.add_transition_flow(
            name=f"clinical_regression_{infectious_status}",
            fractional_rate=Parameter("clinical_regression_rate"),
            source=f"clin_{infectious_status}",
            dest=f"subclin_{infectious_status}",
        )

    # Infectioussness onset and loss flows
    for clinical_status in ["clin", "subclin"]:
        model.add_transition_flow(
            name=f"infectiousnnes_gain_{clinical_status}",
            fractional_rate=Parameter("infectiousness_gain_rate"),
            source=f"{clinical_status}_noninf",
            dest=f"{clinical_status}_inf",
        )
        model.add_transition_flow(
            name=f"infectiousnnes_loss_{clinical_status}",
            fractional_rate=Parameter("infectiousness_loss_rate"),
            source=f"{clinical_status}_inf",
            dest=f"{clinical_status}_noninf",
        )

    # TB mortality and self-recovery
    for infectious_status in ["inf", "noninf"]:
        # mortality only applies to clinical TB
        model.add_transition_flow(
            name=f"tb_mortality_{infectious_status}",
            fractional_rate=Parameter(f"tb_mortality_rate_{infectious_status}"),
            source=f"clin_{infectious_status}",
            dest="mtb_naive"
        )
        # self-recovery only applies to subclinical TB
        model.add_transition_flow(
            name=f"self_recovery_{infectious_status}",
            fractional_rate=Parameter("self_recovery_rate"),
            source=f"subclin_{infectious_status}",
            dest="recovered"
        )

    return model


def add_detection_and_treatment(model: CompartmentalModel):

    # Active disease detection 
    for active_comp in ACTIVE_COMPS:
        model.add_transition_flow(
            name=f"tb_detection_{active_comp}",
            fractional_rate=Parameter("tb_detection_rate"),  #FIXME! will differ by active state
            source=active_comp,
            dest="treatment"
        )

    # TB treatment outcomes
    model.add_transition_flow(
        name="tx_recovery",
        fractional_rate=Parameter("tx_recovery_rate"),
        source="treatment",
        dest="recovered"
    )
    model.add_transition_flow(
        name="tx_relapse",
        fractional_rate=Parameter("tx_relapse_rate"),
        source="treatment",
        dest="subclin_noninf"  #FIXME! may want to use different assumptions
    
    )
    model.add_transition_flow(
        name="tx_death",
        fractional_rate=Parameter("tx_death_rate"),
        source="treatment",
        dest="mtb_naive"
    )
    
