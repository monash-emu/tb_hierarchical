from jax import numpy as jnp
from summer2 import CompartmentalModel, AgeStratification, Multiply
from summer2.parameters import Parameter, Function
from summer2.functions import time as stf

from tbh.demographic_tools import get_pop_size, get_death_rates_by_age, gen_mixing_matrix_func
from tbh.outputs import request_model_outputs

import pandas as pd
from pathlib import Path


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


def get_tb_model(model_config: dict, tv_params: dict):

    # Preparing population, mortality and treatment outcome processes 
    agg_pop_data = get_pop_size(model_config)
    bckd_death_funcs = get_death_rates_by_age(model_config)
    time_variant_tsr = stf.get_linear_interpolation_function(tv_params['tx_success_pct'].index.to_list(), (tv_params['tx_success_pct'] / 100.).to_list())
    neg_tx_outcome_funcs = get_neg_tx_outcome_funcs(bckd_death_funcs, time_variant_tsr)

    # Model building
    model = get_natural_tb_model(model_config, agg_pop_data.iloc[0])
    add_detection_and_treatment(model, time_variant_tsr)
    stratify_model_by_age(model, model_config["age_groups"], neg_tx_outcome_funcs)
    nat_death_flows, tb_death_flows = add_births_and_deaths(model, agg_pop_data, bckd_death_funcs, neg_tx_outcome_funcs, model_config["age_groups"])

    request_model_outputs(model, COMPARTMENTS, ACTIVE_COMPS, nat_death_flows, tb_death_flows)

    return model


def get_natural_tb_model(model_config, init_pop_size):

    model = CompartmentalModel(
        times=(model_config["start_time"], model_config["end_time"]),
        compartments=COMPARTMENTS,
        infectious_compartments=INFECTIOUS_COMPARTMENTS
    )

    model.set_initial_population(
        distribution={
            "mtb_naive": init_pop_size - model_config["seed"],
            "clin_inf": model_config["seed"],
        },
    )

    # Transmission flows (including reinfection)
    for susceptible_comp in ["mtb_naive", "contained", "cleared","recovered"]:
        rel_susceptibility = 1. if susceptible_comp == "mtb_naive" else Parameter(f"rel_sus_{susceptible_comp}")
        model.add_infection_frequency_flow(
            name=f"infection_from_{susceptible_comp}",
            contact_rate=Parameter("raw_transmission_rate") * rel_susceptibility,
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


def add_detection_and_treatment(model: CompartmentalModel, time_variant_tsr):

    # Active disease detection, adjusted based on clinical status
    tv_detection_rate = stf.get_sigmoidal_interpolation_function([1950., 2020], [0., Parameter("recent_detection_rate")])
    for active_comp in ACTIVE_COMPS:
        multiplier = Parameter("rel_detection_subclin") if active_comp.startswith("subclin_") else 1.
        model.add_transition_flow(
            name=f"tb_detection_{active_comp}",
            fractional_rate=multiplier * tv_detection_rate,
            source=active_comp,
            dest="treatment"
        )
    # Track detection rates so they can later be exported as outputs
    model.add_computed_value_func("detection_rate_clin", tv_detection_rate)
    model.add_computed_value_func("detection_rate_subclin", Parameter("rel_detection_subclin") * tv_detection_rate)


    # TB treatment outcomes (only recovery and relapse here. Tx deaths implemented within the "add_births_and_deaths" function)
    model.add_transition_flow(
        name="tx_recovery",
        fractional_rate= time_variant_tsr /Parameter("tx_duration"),
        source="treatment",
        dest="recovered"
    )
    model.add_transition_flow(
        name="tx_relapse",
        fractional_rate=1.,  # Placehodler only, age-specific value will be specified during age stratification
        source="treatment",
        dest="subclin_noninf"
    
    ) 


def stratify_model_by_age(
        model: CompartmentalModel, age_groups: list, neg_tx_outcome_funcs: dict
    ):
    """
        Applies age stratification to the model with specified age groups.

        Adjusts susceptibility for children (<15) to capture BCG effects, sets infectiousness 
        of subclinical cases, and applies an age-based mixing matrix.

        Parameters
        ----------
        model : CompartmentalModel. The model to be stratified by age.
        age_groups : list of str. List of lower bounds for age strata (must include "15").
    """

    assert "15" in age_groups, "We need 15 years old as an age break for compatibility with BCG effect and mixing matrix"

    # Create a stratification object
    age_strat = AgeStratification(
        name="age",
        strata=age_groups,
        compartments=COMPARTMENTS
    )

    # Age-mixing matrix
    build_mixing_matrix = gen_mixing_matrix_func(age_groups)  # create a function for a given set of age breakpoints
    age_mixing_matrix = Function(build_mixing_matrix, [Parameter("mixing_factor_cc"), Parameter("mixing_factor_ca")]) # the function generating the matrix
    age_strat.set_mixing_matrix(age_mixing_matrix)  # apply the mixing matrix to the stratification object

    # Adjust children's susceptibility to infection to capture BCG effect
    age_strat.set_flow_adjustments(
        flow_name="infection_from_mtb_naive",
        adjustments={age: (Parameter("rel_sus_children") if int(age) < 15 else 1.) for age in age_groups}
    )

    # Treatment outcomes are age-specific due to natural mortality being age-specific
    age_strat.set_flow_adjustments(
        "tx_relapse", 
        {age: funcs['relapse'] for age, funcs in neg_tx_outcome_funcs.items()}
    )

    # Adjust infectiousness of clinical vs nonclinical compartments. Not age-related but summer2 requires this to be done through stratification.
    age_strat.add_infectiousness_adjustments("subclin_inf", {age: Parameter("rel_infectiousness_subclin") for age in age_groups})

    # Prevent children from progressing towards the infectious forms of TB
    for clinical_status in ["clin", "subclin"]:
        age_strat.set_flow_adjustments(
            f"infectiousnnes_gain_{clinical_status}", 
            {age: (0. if int(age) < 15 else 1.) for age in age_groups}
        )

    # Apply stratification
    model.stratify_with(age_strat)


def add_births_and_deaths(model, agg_pop_data, bckd_death_rates_funcs, neg_tx_outcome_funcs, age_groups):

    """
        All deaths are modelled as transitions back to mtb_naive compartment, stratum age_0
        If used alone, this approach would maintain constant population size, but extra births will be injected next.
    """
    # All cause (non-TB) mortality
    nat_death_flows = []
    for compartment in COMPARTMENTS:
        for i_age, source_age in enumerate(age_groups):    
            if i_age > 0 or compartment != "mtb_naive":
                name = f"all_cause_mortality_from_{compartment}_age_{source_age}"
                model.add_transition_flow(
                    name=name,
                    fractional_rate=bckd_death_rates_funcs[source_age],
                    source=compartment,
                    source_strata={"age": source_age},
                    dest="mtb_naive",
                    dest_strata={"age": age_groups[0]}
                )
                nat_death_flows.append(name)
                
    # Death caused by TB (Untreated), only applied to clinical TB
    tb_death_flows = []
    for infectious_status in ["inf", "noninf"]:
        for source_age in age_groups:  
            flow_name = f"tb_mortality_{infectious_status}_age_{source_age}"
            model.add_transition_flow(
                name=flow_name,
                fractional_rate=Parameter(f"tb_mortality_rate_{infectious_status}"),
                source=f"clin_{infectious_status}",
                source_strata={"age": source_age},
                dest="mtb_naive",
                dest_strata={"age": age_groups[0]}
            )
            tb_death_flows.append(flow_name)

    # Death on TB treatment
    for source_age in age_groups:  
        flow_name = f"tx_death_age_{source_age}"
        model.add_transition_flow(
            name=flow_name,
            fractional_rate=neg_tx_outcome_funcs[source_age]['death'],
            source="treatment",
            source_strata={"age": source_age},
            dest="mtb_naive",
            dest_strata={"age": age_groups[0]}
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

    return nat_death_flows, tb_death_flows


def get_neg_tx_outcome_funcs(bckd_death_funcs, time_variant_tsr):

    
    def get_neg_tx_outcome_funcs_age(bckd_death_rates_func, time_variant_tsr, tx_duration, pct_neg_tx_death):
    
        # Calculate the proportion of people dying from natural causes while on treatment
        prop_natural_death_while_on_treatment = 1.0 - jnp.exp(
            - tx_duration * bckd_death_rates_func
        )

        # Calculate the target proportion of treatment outcomes resulting in death based on requests
        requested_prop_death_on_treatment = (1.0 - time_variant_tsr) * pct_neg_tx_death / 100.

        # Calculate the actual rate of deaths on treatment, with floor of zero
        prop_death_from_treatment = jnp.max(
            jnp.array(
                (
                    requested_prop_death_on_treatment
                    - prop_natural_death_while_on_treatment,
                    0.0,
                )
            )
        )

        # Calculate the proportion of treatment episodes resulting in relapse
        relapse_prop = (
            1.0 - time_variant_tsr - prop_death_from_treatment - prop_natural_death_while_on_treatment
        )
        neg_tx_outcome_funcs = {
            "relapse": relapse_prop / tx_duration,
            "death": prop_death_from_treatment / tx_duration
        }

        return neg_tx_outcome_funcs

    return {
        age: Function(
            get_neg_tx_outcome_funcs_age,
            [bckd_death_func, time_variant_tsr, Parameter('tx_duration'), Parameter("pct_neg_tx_death")]
        ) for age, bckd_death_func in bckd_death_funcs.items()
    }
    