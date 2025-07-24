from summer2 import CompartmentalModel, AgeStratification
from summer2.parameters import Parameter
from summer2.functions import time as stf

from tbh.outputs import request_model_outputs

import pandas as pd
from jax import numpy as jnp
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

AGE_GROUPS = ["0", "15"]

def get_tb_model(model_config: dict, tv_params: dict):

    agg_pop_data = get_pop_size(model_config)

    model = get_natural_tb_model(model_config, agg_pop_data.iloc[0])
    
    add_detection_and_treatment(model)

    stratify_model_by_age(model)

    tb_death_flows = add_births_and_deaths(model, agg_pop_data)

    request_model_outputs(model, COMPARTMENTS, ACTIVE_COMPS, tb_death_flows)

    return model


def get_natural_tb_model(model_config, init_pop_size):

    model = CompartmentalModel(
        times=(model_config["start_time"], model_config["end_time"]),
        compartments=COMPARTMENTS,
        infectious_compartments=INFECTIOUS_COMPARTMENTS,
    )

    model.set_initial_population(
        distribution={
            "mtb_naive": init_pop_size - model_config["seed"],
            "clin_inf": model_config["seed"],
        },
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

    # TB self-recovery
    for infectious_status in ["inf", "noninf"]:
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

def stratify_model_by_age(model: CompartmentalModel):

    age_stratification = AgeStratification(
        name="age",
        strata=AGE_GROUPS,
        compartments=COMPARTMENTS
    )

    model.stratify_with(age_stratification)


def add_births_and_deaths(model, agg_pop_data):

    """
        All deaths are modelled as transitions back to mtb_naive compartment, stratum age_0
        If used alone, this approach would maintain constant population size, but extra births will be injected next.
    """
    # All cause (non-TB) mortality
    for compartment in COMPARTMENTS:
        for i_age, source_age in enumerate(AGE_GROUPS):    
            if i_age > 0 or compartment != "mtb_naive":
                model.add_transition_flow(
                    name=f"all_cause_mortality_from_{compartment}_age_{source_age}",
                    fractional_rate=PLACEHOLDER_VALUE,  #FIXME! placeholder for now, will need to be age-specific
                    source=compartment,
                    source_strata={"age": source_age},
                    dest="mtb_naive",
                    dest_strata={"age": AGE_GROUPS[0]}
                )
                
    
    # Death caused by TB (Untreated), only applied to clinical TB
    tb_death_flows = []
    for infectious_status in ["inf", "noninf"]:
        for source_age in AGE_GROUPS:  
            flow_name = f"tb_mortality_{infectious_status}_age_{source_age}"
            model.add_transition_flow(
                name=flow_name,
                fractional_rate=Parameter(f"tb_mortality_rate_{infectious_status}"),
                source=f"clin_{infectious_status}",
                source_strata={"age": source_age},
                dest="mtb_naive",
                dest_strata={"age": AGE_GROUPS[0]}
            )
            tb_death_flows.append(flow_name)

    # Death on TB treatment
    for source_age in AGE_GROUPS:  
        flow_name = f"tx_death_age_{source_age}"
        model.add_transition_flow(
            name=flow_name,
            fractional_rate=Parameter("tx_death_rate"),
            source="treatment",
            source_strata={"age": source_age},
            dest="mtb_naive",
            dest_strata={"age": AGE_GROUPS[0]}
        )
        tb_death_flows.append(flow_name)

    """
        Will now inject extra births to capture population growth
    """
    # linear interpolation for missing years
    full_index = pd.Index(range(agg_pop_data.index.min(), agg_pop_data.index.max() + 1))
    full_population_series = agg_pop_data.reindex(full_index).interpolate() # reindex with full index and linear-interpolate
    pop_entry = full_population_series.diff().dropna() # calculate delta between successive years

    # add 0 growth before first data point, then use sigmoidal interpolation for entry rate
    entry_rate = stf.get_sigmoidal_interpolation_function(
        [pop_entry.index.min() - 1] + pop_entry.index.to_list(), 
        [0.] + pop_entry.to_list()
    )

    model.add_importation_flow( # Add births as additional entry rate, (split imports in case the susceptible compartments are further stratified later)
        "births", entry_rate, dest="mtb_naive", split_imports=True, dest_strata={"age": "0"}
    )

    return tb_death_flows



from tbh.paths import DATA_FOLDER

def get_pop_size(model_config):

    pop_data = pd.read_csv(DATA_FOLDER / "un_population.csv")
    # filter for country and truncate historical pre-analysis years
    pop_data = pop_data[(pop_data["ISO3_code"] == model_config['iso3']) & (pop_data["Time"] >= model_config['start_time'])]

    # Aggregate accross agegroups for each year
    agg_pop_data = pop_data.groupby('Time')['PopTotal'].sum().sort_index()

    return agg_pop_data