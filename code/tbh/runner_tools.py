import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


from estival.wrappers import pymc as epm
from estival.sampling import tools as esamp
from estival.model import BayesianCompartmentalModel

from .model import get_tb_model
from .calibration import get_bcm_object, get_priors, get_targets

import tbh.plotting as pl
from tbh.paths import OUTPUT_PARENT_FOLDER

from pathlib import Path

DEFAULT_MODEL_CONFIG = {
    "start_time": 1850,
    "end_time": 2050,
    "seed": 100,
}

DEFAULT_PARAMS = {
    # Study-specific parameters
    "transmission_rateXmajuro": 10,
    "transmission_rateXstudy_2": 10,
    "lifelong_activation_riskXmajuro": 0.15,
    "lifelong_activation_riskXstudy_2": 0.10,
    "prop_early_among_activatorsXmajuro": 0.90,
    "prop_early_among_activatorsXstudy_2": 0.90,
    "current_passive_detection_rateXmajuro": 1.0,
    "current_passive_detection_rateXstudy_2": 1.0,
    # Universal parameters
    "mean_duration_early_latent": 0.5,
    "rr_reinfection_latent_late": 0.2,
    "rr_reinfection_recovered": 1.0,
    "self_recovery_rate": 0.2,
    "tb_death_rate": 0.2,
    "tx_duration": 0.5,
    "tx_prop_death": 0.04,
}

DEFAULT_STUDIES_DICT = {
    "majuro": {
        "pop_size": 27797,
    },
    "study_2": {  # vietnam like
        "pop_size": 100.e6,
    }    
}

DEFAULT_ANALYSIS_CONFIG = {
    # Metropolis config
    'chains': 4,
    'tune': 5000,
    'draws': 20000,

    # Full runs config
    'burn_in': 10000,
    'full_runs_samples': 1000
}

TEST_ANALYSIS_CONFIG = {
    # Metropolis config
    'chains': 4,
    'tune': 50,
    'draws': 200,

    # Full runs config
    'burn_in': 50,
    'full_runs_samples': 100
}


def create_output_dir(array_job_id, task_id, analysis_name):
    output_dir = OUTPUT_PARENT_FOLDER / f"{array_job_id}_{analysis_name}" / f"task_{task_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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


def run_metropolis_calibration(
    bcm: BayesianCompartmentalModel,
    draws=20000,
    tune=2000,
    cores=4,
    chains=4,
    method="DEMetropolisZ",
):
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
        raise ValueError(
            f"Requested sampling method '{method}' not currently supported."
        )

    with pm.Model() as model:
        variables = epm.use_model(bcm)
        idata = pm.sample(
            step=[sampler(variables)],
            draws=draws,
            tune=tune,
            cores=cores,
            chains=chains,
            progressbar=False,
        )

    return idata


def run_full_runs(
    bcm: BayesianCompartmentalModel, idata, burn_in: int, full_runs_samples: int
):

    # select samples
    chain_length = idata.sample_stats.sizes["draw"]
    assert (
        full_runs_samples <= chain_length - burn_in
    ), "Too many full-run samples requested."
    burnt_idata = idata.sel(draw=range(burn_in, chain_length))  # Discard burn-in
    full_run_params = az.extract(burnt_idata, num_samples=full_runs_samples)

    full_runs = esamp.model_results_for_samples(
        full_run_params, bcm, include_extras=False
    )
    unc_df = esamp.quantiles_for_results(
        full_runs.results, [0.025, 0.25, 0.5, 0.75, 0.975]
    )

    return full_runs, unc_df


def run_full_analysis(studies_dict=DEFAULT_STUDIES_DICT, params=DEFAULT_PARAMS, model_config=DEFAULT_MODEL_CONFIG, analysis_config=DEFAULT_ANALYSIS_CONFIG, output_folder=None):
    """
    Run full analysis including Metropolis-sampling-based calibration, full runs, quantiles computation and plotting.

    Args:
        studies_dict (_type_, optional): _description_. Defaults to DEFAULT_STUDIES_DICT.
        params (_type_, optional): _description_. Defaults to DEFAULT_PARAMS.
        model_config (_type_, optional): _description_. Defaults to DEFAULT_MODEL_CONFIG.
        analysis_config (_type_, optional): _description_. Defaults to DEFAULT_ANALYSIS_CONFIG.
        output_folder (_type_, optional): _description_. Defaults to None.

    """
    a_c = analysis_config

    output_folder.mkdir(parents=True, exist_ok=True) 

    bcm = get_bcm_object(model_config, studies_dict, params)

    print(">>> Run Metropolis sampling")
    idata = run_metropolis_calibration(
        bcm, draws=a_c['draws'], tune=a_c['tune'], cores=a_c['chains'], chains=a_c['chains']
    )
    az.to_netcdf(idata, output_folder / "idata.nc")

    pl.plot_traces(idata, a_c['burn_in'], output_folder)
    pl.plot_post_prior_comparison(idata, list(bcm.priors.keys()), list(bcm.priors.values()), req_grid=[3, 4], output_folder_path=output_folder)

    print(">>> Run full runs")
    full_runs, unc_df = run_full_runs(bcm, idata, a_c['burn_in'], a_c['full_runs_samples'])

    selected_outputs = bcm.targets.keys()

    for output in selected_outputs:
        _, ax = plt.subplots()
        pl.plot_model_fit_with_uncertainty(ax, unc_df, output, bcm, x_min=2010)
        if output_folder:
            plt.savefig(output_folder / f"quantiles_{output}.jpg", facecolor="white", bbox_inches='tight')
            plt.close()