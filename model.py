from summer2 import CompartmentalModel, Stratification
from summer2.parameters import Parameter, DerivedOutput
from summer2.functions import time as stf

from jax import numpy as jnp
import yaml
from pathlib import Path   


def get_tb_model(model_config: dict, studies_dict: dict, home_path=Path.cwd()):

    """
    Prepare time-variant parameters and other quantities requiring pre-processsing
    """
    tv_data_path = home_path / 'data' / 'time_variant_params.yml'
    with open(tv_data_path, 'r') as file:
        tv_data = yaml.safe_load(file)

    crude_birth_rate_func = stf.get_linear_interpolation_function(
        x_pts=tv_data['crude_birth_rate']['times'], y_pts=[cbr / 1000. for cbr in tv_data['crude_birth_rate']['values']]
    )

    life_expectancy_func = stf.get_linear_interpolation_function(
        x_pts=tv_data['life_expectancy']['times'], y_pts=tv_data['life_expectancy']['values']
    )
    all_cause_mortality_func = 1. / life_expectancy_func
    
    detection_func = stf.get_sigmoidal_interpolation_function(
        x_pts=[1950., 2025.], y_pts=[0., Parameter('current_passive_detection_rate')], curvature=16
    )

    # Treatment outcomes
    # * tx recovery rate is 1/Tx duration
    # * write equations for TSR and for prop deaths among all treatment outcomes (Pi). Solve for treatment death rate (mu_Tx) and relapse rate (phi).

    tsr_func = stf.get_linear_interpolation_function(
        x_pts=tv_data['treatment_success_perc']['times'], 
        y_pts=[ts_perc / 100. for ts_perc in tv_data['treatment_success_perc']['values']]
    )

    tx_recovery_rate = 1. / Parameter("tx_duration") 
    tx_death_func = tx_recovery_rate * Parameter("tx_prop_death") / tsr_func - all_cause_mortality_func
    tx_relapse_func = (all_cause_mortality_func + tx_death_func) * (1. / Parameter("tx_prop_death") - 1.) - tx_recovery_rate

    """
    Build the model
    """
    compartments = (
        "susceptible", 
        "latent_early",
        "latent_late",
        "infectious",
        "treatment", 
        "recovered",
    )
    model = CompartmentalModel(
        times=(model_config["start_time"], model_config["end_time"]),
        compartments=compartments,
        infectious_compartments=("infectious",),
    )

    total_pop_size = sum([studies_dict[s]['pop_size'] for s in studies_dict])
    model.set_initial_population(
        distribution=
        {
            "susceptible": total_pop_size - model_config["seed"], 
            "infectious": model_config["seed"],
        },
    )
    
    # add birth and all cause mortality
    #FIXME! WIll need to make demographics study-specific
    model.add_crude_birth_flow(
        name="birth",
        birth_rate=crude_birth_rate_func,
        dest="susceptible"
    )

    model.add_universal_death_flows(
        name="all_cause_mortality",
        death_rate= all_cause_mortality_func
    )

    # infection and reinfection flows
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=1., # will be adjusted later by study 
        source="susceptible", 
        dest="latent_early",
    )
    for reinfection_source in ["latent_late", "recovered"]:
        model.add_infection_frequency_flow(
            name=f"reinfection_{reinfection_source}", 
            contact_rate=Parameter(f"rr_reinfection_{reinfection_source}"),  # will be adjusted later by study 
            source=reinfection_source, 
            dest="latent_early",
        )

    # latency progression
    model.add_transition_flow(
        name="stabilisation",
        fractional_rate=Parameter("stabilisation_rate"),
        source="latent_early",
        dest="latent_late",
    )
    for progression_type in ["early", "late"]:
        model.add_transition_flow(
            name=f"progression_{progression_type}",
            fractional_rate=Parameter(f"activation_rate_{progression_type}"),
            source=f"latent_{progression_type}",
            dest="infectious",
        )

    # natural recovery
    model.add_transition_flow(
        name="self_recovery",
        fractional_rate=Parameter("self_recovery_rate"),
        source="infectious",
        dest="recovered",
    )

    # TB-specific death
    model.add_death_flow(
        name="active_tb_death",
        death_rate=Parameter("tb_death_rate"),
        source="infectious",
    )

    # detection of active TB
    model.add_transition_flow(
        name="tb_detection",
        fractional_rate=detection_func,
        source="infectious",
        dest="treatment",
    )

    # treatment exit flows
    model.add_transition_flow(
        name="tx_recovery",
        fractional_rate=tx_recovery_rate,
        source="treatment",
        dest="recovered",
    )
    model.add_transition_flow(
        name="tx_relapse",
        fractional_rate=tx_relapse_func,
        source="treatment",
        dest="infectious",
    )
    model.add_death_flow(
        name="tx_death",
        death_rate=tx_death_func,
        source="treatment",
    )

    stratify_by_study(model, compartments, studies_dict, total_pop_size)

    """
       Request Derived Outputs
    """
    request_model_outputs(model, compartments, list(studies_dict.keys()))

    return model


def stratify_by_study(model:CompartmentalModel, compartments:list, studies_dict: dict, total_pop_size: float):
    """
    Stratify the base TB model to capture the different studies.

    Args:
        model (CompartmentalModel): TB model before stratification
        compartments (list): list of all model compartments
        studies_dict (dict): Information about the different studies
    """
    studies = list(studies_dict.keys())
    study_stratification = Stratification(name="study", strata=studies, compartments=compartments)
    
    # Decouple the different strata (i.e. studies) using an identity matrix as mixing matrix
    study_stratification.set_mixing_matrix(jnp.eye(len(studies)))
    
    # Split initial population and births according to studies' population sizes
    study_stratification.set_population_split(
        {s: studies_dict[s]['pop_size'] / total_pop_size for s in studies}
    )
    study_stratification.set_flow_adjustments("birth", adjustments={s: studies_dict[s]['pop_size'] / total_pop_size for s in studies})

    # Adjust all transmission flows by study
    for flow_name in ["infection", "reinfection_latent_late", "reinfection_recovered"]:
        study_stratification.set_flow_adjustments(
            flow_name=flow_name , adjustments={s: Parameter(f"transmission_rateX{s}") for s in studies}
        )

    # apply stratification to the model
    model.stratify_with(study_stratification)


def request_model_outputs(model:CompartmentalModel, compartments:list, studies:list):
    """
    Define model outputs that can later be requested from model.get_derived_outputs_df()

    Args:
        model (CompartmentalModel): the 'fully-built' TB model
        compartments (list): list of all model compartments (required to create population size output)
        studies (list): list of studies (required to disaggregate outputs by study)
    """

    for study in studies:
        study_filter = {"study": study}

        ## Raw compartment size outputs
        model.request_output_for_compartments(name=f"raw_ltbi_prevalenceX{study}", compartments=["latent_early", "latent_late"], strata=study_filter, save_results=False,)
        model.request_output_for_compartments(name=f"raw_tb_prevalenceX{study}", compartments=["infectious"], strata=study_filter, save_results=False)

        ## Raw flow outputs
        # Incidence
        model.request_output_for_flow(name=f"progression_earlyX{study}", flow_name="progression_early",source_strata=study_filter, save_results=False)
        model.request_output_for_flow(name=f"progression_lateX{study}", flow_name="progression_late", source_strata=study_filter, save_results=False)
        model.request_aggregate_output(name=f"raw_tb_incidenceX{study}", sources=[f"progression_earlyX{study}", f"progression_lateX{study}"], save_results=False)

        # Notifications
        model.request_output_for_flow(name=f"raw_notificationsX{study}", flow_name="tb_detection", source_strata=study_filter)

        # TB Mortality
        model.request_output_for_flow(name=f"active_tb_deathX{study}", flow_name="active_tb_death", source_strata=study_filter, save_results=False)
        model.request_output_for_flow(name=f"tx_deathX{study}", flow_name="tx_death", source_strata=study_filter, save_results=False)
        model.request_aggregate_output(name=f"all_tb_deathsX{study}", sources=[f"active_tb_deathX{study}", f"tx_deathX{study}"])

        ## Outputs relative to population size
        model.request_output_for_compartments(name=f"populationX{study}", compartments=compartments, strata=study_filter)
        model.request_function_output(name=f"ltbi_propX{study}", func=DerivedOutput(f"raw_ltbi_prevalenceX{study}") / DerivedOutput(f"populationX{study}"))
        model.request_function_output(name=f"tb_prevalence_per100kX{study}", func=1.e5 * DerivedOutput(f"raw_tb_prevalenceX{study}") / DerivedOutput(f"populationX{study}"))
        model.request_function_output(name=f"tb_incidence_per100kX{study}", func=1.e5 * DerivedOutput(f"raw_tb_incidenceX{study}") / DerivedOutput(f"populationX{study}"))
        model.request_function_output(name=f"tb_mortality_per100kX{study}", func=1.e5 * DerivedOutput(f"all_tb_deathsX{study}") / DerivedOutput(f"populationX{study}"))
