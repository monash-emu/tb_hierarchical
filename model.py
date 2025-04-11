from summer2 import CompartmentalModel, Stratification
from summer2.parameters import Parameter, DerivedOutput
from summer2.functions import time as stf

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
    transmission_rate = Parameter("transmission_rate")
    model.add_infection_frequency_flow(
        name="infection", 
        contact_rate=transmission_rate,
        source="susceptible", 
        dest="latent_early",
    )
    for reinfection_source in ["latent_late", "recovered"]:
        model.add_infection_frequency_flow(
            name=f"reinfection_{reinfection_source}", 
            contact_rate=transmission_rate * Parameter(f"rr_reinfection_{reinfection_source}"),
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

    """
        Stratify the base model to capture the different studies
    """
    study_stratification = Stratification(name="study", strata=studies_dict.keys(), compartments=compartments)
    study_stratification.set_population_split(
        {s: studies_dict[s]['pop_size'] / total_pop_size for s in studies_dict}
    )
    model.stratify_with(study_stratification)

    """
       Request Derived Outputs
    """
    # Raw outputs
    model.request_output_for_compartments(name="raw_ltbi_prevalence", compartments=["latent_early", "latent_late"], save_results=False)
    model.request_output_for_compartments(name="raw_tb_prevalence", compartments=["infectious"], save_results=False)

    model.request_output_for_flow(name="progression_early", flow_name="progression_early", save_results=False)
    model.request_output_for_flow(name="progression_late", flow_name="progression_late", save_results=False)
    model.request_aggregate_output(name="raw_tb_incidence", sources=["progression_early", "progression_late"], save_results=False)

    model.request_output_for_flow(name="raw_notifications", flow_name="tb_detection")

    model.request_output_for_flow(name="active_tb_death", flow_name="active_tb_death", save_results=False)
    model.request_output_for_flow(name="tx_death", flow_name="tx_death", save_results=False)
    model.request_aggregate_output(name="all_tb_deaths", sources=["active_tb_death", "tx_death"])

    # Outputs relative to population size
    model.request_output_for_compartments(name="population", compartments=compartments)
    model.request_function_output(name="ltbi_prop", func=DerivedOutput("raw_ltbi_prevalence") / DerivedOutput("population"))
    model.request_function_output(name="tb_prevalence_per100k", func=1.e5 * DerivedOutput("raw_tb_prevalence") / DerivedOutput("population"))
    model.request_function_output(name="tb_incidence_per100k", func=1.e5 * DerivedOutput("raw_tb_incidence") / DerivedOutput("population"))
    model.request_function_output(name="tb_mortality_per100k", func=1.e5 * DerivedOutput("all_tb_deaths") / DerivedOutput("population"))

    return model
