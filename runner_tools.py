import pandas as pd
import pymc as pm
import arviz as az

from estival import priors as esp
from estival import targets as est
from estival.model import BayesianCompartmentalModel
from estival.wrappers import pymc as epm
from estival.sampling import tools as esamp

from model import get_tb_model

DEFAULT_MODEL_CONFIG = {
    "start_time": 1850,
    "end_time": 2050,
    "seed": 100,   
}

DEFAULT_PARAMS = {
    # Study-specific parameters
    'transmission_rateXmajuro': 10,
    'transmission_rateXstudy_2': 10,

    'lifelong_activation_riskXmajuro': .15,
    'lifelong_activation_riskXstudy_2': .10,
    'prop_early_among_activatorsXmajuro': .90,
    'prop_early_among_activatorsXstudy_2': .90,

    'current_passive_detection_rate': 1.,

    # Universal parameters
    'mean_duration_early_latent': .5,
    'rr_reinfection_latent_late': .2,
    'rr_reinfection_recovered': 1.,
    'self_recovery_rate': .2,
    'tb_death_rate': .2,
    'tx_duration': .5,
    'tx_prop_death': .04
}


def model_single_run(model_config: dict, studies_dict: dict, params: dict):
    """
    Run the TB model for a given parameter set

    Args:
        model_config (dict): Model run configuration
        studies_dict (dict): Information about different studies
        params (dict): The model parameters

    Returns:
        model: the run model
        derived_outputs_df: pandas dataframe containing derived outputs 
    """
    
    model = get_tb_model(model_config, studies_dict)
    model.run(params)
    derived_outputs_df = model.get_derived_outputs_df()

    return model, derived_outputs_df


def get_priors(studies_dict: dict) -> list:
    """
    Define the list of prior distributions to be used for Bayesian calibration.

    Args:
        studies_dict (dict): Information about different studies

    Returns:
        prior_list: List of prior distributions
    """

    # Define hyper-prior distributions
    hyper_mean_lifelong = esp.UniformPrior("hyper_mean_lifelong", [0., 1.])
    hyper_sd_lifelong = esp.UniformPrior("hyper_sd_lifelong", [0., .1])
    # hyper_mean_early = esp.UniformPrior("hyper_mean_early", [0., 1.])
    # hyper_sd_early = esp.UniformPrior("hyper_sd_early", [0., 10.])
    
    # Initialise the list of priors with "universal" priors and hyper-priors
    priors = [
        esp.UniformPrior("current_passive_detection_rate", [.1, 10.]),
        hyper_mean_lifelong,
        hyper_sd_lifelong,
        # hyper_mean_early,
        # hyper_sd_early
    ]
    
    # Complete the list of priors using study-specific priors
    for study in studies_dict:
        priors.extend(
            [
                esp.UniformPrior(f"transmission_rateX{study}", [1., 15.]),

                # the two priors below linked through the previously defined hyper-prior distributions 

                esp.TruncNormalPrior(f"lifelong_activation_riskX{study}", hyper_mean_lifelong, hyper_sd_lifelong, [0., 1.]),
                # esp.TruncNormalPrior(f"lifelong_activation_riskX{study}", hyper_mean_lifelong, .01, [0., 1.]),


                # esp.TruncNormalPrior(f"prop_early_among_activatorsX{study}", hyper_mean_early, hyper_sd_early, [0., 1.]),
            ]
        )
    return priors


def get_targets():
    """
    Define calibration targets
    """
    return [
        est.NormalTarget("ltbi_propXmajuro", data=pd.Series(data=[.38], index=[2018]), stdev=esp.UniformPrior("std_ltbi", [.001, .1])),
        est.NormalTarget("tb_prevalence_per100kXmajuro", data=pd.Series(data=[1366], index=[2018]), stdev=esp.UniformPrior("std_tb", [10., 250.])),
        est.NormalTarget("raw_notificationsXmajuro", data=pd.Series(data=[100], index=[2015]), stdev=esp.UniformPrior("std_not", [1., 25.])),
    ]


def get_bcm_object(model_config: dict, studies_dict: dict, params: dict) -> BayesianCompartmentalModel:
    """

    Args:
        model_config (dict): _description_
        studies_dict (dict): _description_
        params (dict): model parameters (used to inform non-calibrated parameters)

    Returns:
        bcm: estival BayesianCompartmentalModel object ready for calibration

    """
    priors = get_priors(studies_dict)
    targets = get_targets()
    model = get_tb_model(model_config, studies_dict)
    
    return BayesianCompartmentalModel(model, params, priors, targets)


def run_metropolis_calibration(bcm: BayesianCompartmentalModel, draws=20000, tune=2000, cores=4, chains=4, method="DEMetropolisZ"):
    """
    Run bayesian sampling using pymc methods

    Args:
        bcm (BayesianCompartmentalModel): estival Calibration object containing model, priors and targets
        draws (int, optional): Number of iterations per chain. Defaults to 20000.
        tune (int, optional): Number of iterations used for tuning (will add to draws). Defaults to 2000.
        cores (int, optional): Number of cores. Defaults to 4.
        chains (int, optional): Number of chains. Defaults to 4.
        method (str, optional): pymc calibration algorithm used. Defaults to "DEMetropolisZ".
    """
    if method == "DEMetropolis":
        sampler = pm.DEMetropolis
    elif method == "DEMetropolisZ":
        sampler = pm.DEMetropolisZ
    else:
        raise ValueError(f"Requested sampling method '{method}' not currently supported.")

    with pm.Model() as model:
        variables = epm.use_model(bcm)
        idata = pm.sample(step=[sampler(variables)], draws=draws, tune=tune, cores=cores, chains=chains)

    return idata


def run_full_runs(bcm: BayesianCompartmentalModel, idata, burn_in: int, full_runs_samples: int):

    # select samples
    chain_length = idata.sample_stats.sizes['draw']
    assert full_runs_samples <= chain_length - burn_in, "Too many full-run samples requested."
    burnt_idata = idata.sel(draw=range(burn_in, chain_length))  # Discard burn-in
    full_run_params = az.extract(burnt_idata, num_samples=full_runs_samples)

    full_runs = esamp.model_results_for_samples(full_run_params, bcm, include_extras=False)
    unc_df = esamp.quantiles_for_results(full_runs.results, [.025, .25, .5, .75, .975])

    return full_runs, unc_df
