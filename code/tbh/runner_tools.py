import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from time import time
import yaml

from estival.wrappers import pymc as epm
from estival.sampling import tools as esamp
from estival.model import BayesianCompartmentalModel
from estival import priors as esp
from estival import targets as est

from .model import get_tb_model

import tbh.plotting as pl
from tbh.paths import OUTPUT_PARENT_FOLDER, DATA_FOLDER

from pathlib import Path

DEFAULT_MODEL_CONFIG = {
    "start_time": 1850,
    "end_time": 2050,
    "seed": 100,
    "iso3": "KIR",
    "age_groups": ["0", "15", "75"],
}

DEFAULT_ANALYSIS_CONFIG = {
    # Metropolis config
    'chains': 4,
    'cores': 4.,
    'tune': 5000,
    'draws': 20000,

    # Full runs config
    'burn_in': 10000,
    'full_runs_samples': 1000
}

TEST_ANALYSIS_CONFIG = {
    # Metropolis config
    'chains': 4,
    'cores': 4,
    'tune': 500,
    'draws': 2000,

    # Full runs config
    'burn_in': 500,
    'full_runs_samples': 1000
}

# !FIXME this code doesn't belong here
targets = [
    est.NormalTarget(
        name='tb_prevalence_per100k', 
        data=pd.Series(data=[600,], index=[2020]), 
        stdev=100.
    ),
    est.NormalTarget(
        name='tbi_prevalence_perc', 
        data=pd.Series(data=[40,], index=[2020]), 
        stdev=5.
    ),
    est.NormalTarget(
        name='perc_prev_subclinical', 
        data=pd.Series(data=[50], index=[2020]), 
        stdev=5.
    ),
]


def get_prior(param_name, distribution, distri_param1, distri_param2=None):
    
    if distribution == "uniform":
        return esp.UniformPrior(param_name, [distri_param1, distri_param2])
    else:
        raise ValueError(f"{distribution} is not currently a supported distribution")


def get_parameters_and_priors(params_file_path=DATA_FOLDER / "parameters.xlsx"):
    """
    Read parameter values (for fixed parameters) and prior distribution details from xlsx file.
    
    Returns:
        params: Dictionary with parameter values
        priors: List of estival prior objects
        tv_df: pandas Dataframe with time-variant parameters
    """

    """
        Read constant (i.e. non-time-variant) parameters, including fixed params and priors
    """
    df = pd.read_excel(params_file_path, sheet_name="constant")
    df = df.where(pd.notna(df), None)  # Replace Nas (and empty cells) with None

    # Fixed parameters
    cst_params = dict(zip(df['parameter'], df['value']))

    # Prior distributions
    priors = []
    priors_df = df[df['distribution'].notnull()]
    priors = [get_prior(row['parameter'], row['distribution'], row['distri_param1'], row['distri_param2']) for _, row in priors_df.iterrows()]        

    """
        Read time-variant parameters
    """
    tv_df = pd.read_excel(params_file_path, sheet_name="time_variant", index_col=0)
    tv_params = {col: tv_df[col].dropna() for col in tv_df.columns}

    return cst_params, priors, tv_params


def create_output_dir(array_job_id, task_id, analysis_name):
    output_dir = OUTPUT_PARENT_FOLDER / f"{array_job_id}_{analysis_name}" / f"task_{task_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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


def run_full_analysis(model_config=DEFAULT_MODEL_CONFIG, analysis_config=DEFAULT_ANALYSIS_CONFIG, output_folder=None):
    """
    Run full analysis including Metropolis-sampling-based calibration, full runs, quantiles computation and plotting.

    Args:
        params (_type_, optional): _description_.
        model_config (_type_, optional): _description_. Defaults to DEFAULT_MODEL_CONFIG.
        analysis_config (_type_, optional): _description_. Defaults to DEFAULT_ANALYSIS_CONFIG.
        output_folder (_type_, optional): _description_. Defaults to None.

    """
    a_c = analysis_config
    output_folder.mkdir(parents=True, exist_ok=True) 
    params, priors, tv_params = get_parameters_and_priors()

    model = get_tb_model(model_config, tv_params)
    bcm = BayesianCompartmentalModel(model, params, priors, targets)

    print(">>> Run Metropolis sampling")
    
    times = {}
    t0 = time()
    idata = run_metropolis_calibration(
        bcm, draws=a_c['draws'], tune=a_c['tune'], cores=a_c['cores'], chains=a_c['chains']
    )
    mcmc_time = time() - t0
    times["mcmc_time"] = f"{round(mcmc_time)} sec (i.e. {round(mcmc_time / 60)} min) --> {round(3600 * (a_c['tune'] + a_c['draws'])/ mcmc_time)} runs per hour"
    az.to_netcdf(idata, output_folder / "idata.nc")

    pl.plot_traces(idata, a_c['burn_in'], output_folder)
    pl.plot_post_prior_comparison(idata, a_c['burn_in'], list(bcm.priors.keys()), list(bcm.priors.values()), n_col=4, output_folder_path=output_folder)

    print(">>> Run full runs")
    t0 = time()
    full_runs, unc_df = run_full_runs(bcm, idata, a_c['burn_in'], a_c['full_runs_samples'])
    fullruns_time = time() - t0
    times["full_runs_time"] = f"{round(fullruns_time)} sec (i.e. {round(fullruns_time / 60)} min) --> {round(3600 * (a_c['full_runs_samples'])/ fullruns_time)} runs per hour"
    
    selected_outputs = bcm.targets.keys()

    for output in selected_outputs:
        _, ax = plt.subplots()
        pl.plot_model_fit_with_uncertainty(ax, unc_df, output, bcm, x_min=2010)
        if output_folder:
            plt.savefig(output_folder / f"quantiles_{output}.jpg", facecolor="white", bbox_inches='tight')
            plt.close()
    
    if output_folder:
        with open(output_folder / 'timings.yaml', 'w') as file:
            yaml.dump_all([times, model_config, analysis_config], file, default_flow_style=False)
