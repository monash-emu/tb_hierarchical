import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from time import time
import yaml
from git import Repo

from estival.wrappers import pymc as epm
from estival.sampling import tools as esamp
from estival.model import BayesianCompartmentalModel
from estival import priors as esp
from estival import targets as est

from .model import get_tb_model

import tbh.plotting as pl
from tbh.paths import OUTPUT_PARENT_FOLDER, DATA_FOLDER

#FIXME!: This is a messy import
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[2]
sys.path.append(str(root))

from data import scenarios as sc


DEFAULT_MODEL_CONFIG = {
    "start_time": 1850,
    "end_time": 2050,
    "seed": 100,
    "iso3": "KIR",
    "age_groups": ["0", "3", "10", "15", "65"],
    "pop_scaling": 1.0,
}

DEFAULT_ANALYSIS_CONFIG = {
    # Metropolis config
    'chains': 4,
    'cores': 4.,
    'tune': 5000,
    'draws': 20000,

    # Full runs config
    'burn_in': 10000,
    'full_runs_samples': 4000,
    'scenarios': [sc.scenario_1, sc.scenario_2, sc.scenario_3]
}

TEST_ANALYSIS_CONFIG = {
    # Metropolis config
    'chains': 4,
    'cores': 4,
    'tune': 50,
    'draws': 200,

    # Full runs config
    'burn_in': 50,
    'full_runs_samples': 100,
    'scenarios': [sc.scenario_1, sc.scenario_2, sc.scenario_3]
}

# !FIXME this code doesn't belong here
targets = [
    est.NormalTarget(
        name='measured_tb_prevalence_per100k', 
        data=pd.Series(data=[846.,], index=[2024]), 
        stdev=100.
    ),
    est.NormalTarget(
        name='measured_tbi_prevalence_perc', 
        data=pd.Series(data=[25.,], index=[2024]), 
        stdev=5.
    ),
    est.NormalTarget(
        name='perc_prev_subclinical', 
        data=pd.Series(data=[50], index=[2024]), 
        stdev=5.
    ),
    est.NormalTarget(
        name='perc_prev_infectious', 
        data=pd.Series(data=[50], index=[2024]), 
        stdev=5.
    ),
    est.NormalTarget(
        name='notifications', 
        data=pd.Series(data=[384], index=[2024]),     # WHO reports 305/100k in 2020, 126099 population
        stdev=50.
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
    bcm_dict: dict[str, BayesianCompartmentalModel], idata, burn_in: int, full_runs_samples: int
):

    # select samples
    chain_length = idata.sample_stats.sizes["draw"]
    assert (
        full_runs_samples <= chain_length - burn_in
    ), "Too many full-run samples requested."
    burnt_idata = idata.sel(draw=range(burn_in, chain_length))  # Discard burn-in
    full_run_params = az.extract(burnt_idata, num_samples=full_runs_samples)

    full_runs, unc_dfs = {}, {}
    for sc, bcm in bcm_dict.items():
        full_run = esamp.model_results_for_samples(
            full_run_params, bcm, include_extras=False
        )
        unc_df = esamp.quantiles_for_results(
            full_run.results, [0.025, 0.25, 0.5, 0.75, 0.975]
        )
        unc_df.columns = unc_df.columns.set_levels([str(q) for q in unc_df.columns.levels[1]], level=1) # to avoid using floats as column names (not parquet-compatible)

        full_runs[sc] = full_run
        unc_dfs[sc] = convert_jax_columns(unc_df)

    return full_runs, unc_dfs


def convert_jax_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns containing JAX arrays into plain Python scalars or lists.
    Only modifies problematic columns.
    """
    def _convert_value(x):
        if hasattr(x, "item"):  # scalar-like JAX or NumPy
            try:
                return x.item()
            except Exception:
                return x
        if hasattr(x, "tolist"):  # array-like JAX or NumPy
            try:
                return x.tolist()
            except Exception:
                return x
        return x

    for col in df.columns:
        if df[col].apply(lambda x: hasattr(x, "item") or hasattr(x, "tolist")).any():
            df[col] = df[col].map(_convert_value)

    return df


def run_full_analysis(model_config=DEFAULT_MODEL_CONFIG, analysis_config=DEFAULT_ANALYSIS_CONFIG, output_folder=None, idata_path=None):
    """
    Run full analysis including Metropolis-sampling-based calibration, full runs, quantiles computation and plotting.

    Args:
        params (_type_, optional): _description_.
        model_config (_type_, optional): _description_. Defaults to DEFAULT_MODEL_CONFIG.
        analysis_config (_type_, optional): _description_. Defaults to DEFAULT_ANALYSIS_CONFIG.
        output_folder (_type_, optional): _description_. Defaults to None.

    """
    a_c = analysis_config

    # check scenario ids are unique
    sc_ids = [scenario.sc_id for scenario in a_c['scenarios']]
    assert len(sc_ids) == len(set(sc_ids)), "Please use unique scenario ids."

    output_folder.mkdir(parents=True, exist_ok=True) 
    params, priors, tv_params = get_parameters_and_priors()

    model = get_tb_model(model_config, tv_params)
    bcm = BayesianCompartmentalModel(model, params, priors, targets)

    print(">>> Run Metropolis sampling")
    
    times = {}
    t0 = time()

    if idata_path:
        idata = az.from_netcdf(idata_path / "idata.nc")
    else:
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
    bcm_dict = {"baseline": bcm}
    for scenario in a_c['scenarios']:
        assert scenario.sc_name != "baseline", "Please use scenario name different from 'baseline'"
        sc_params = params | scenario.params_ow
        sc_model = get_tb_model(model_config, tv_params, screening_programs=scenario.scr_prgs)
        sc_bcm = BayesianCompartmentalModel(sc_model, sc_params, priors, targets)
        bcm_dict[scenario.sc_id] = sc_bcm

    full_runs, unc_dfs = run_full_runs(bcm_dict, idata, a_c['burn_in'], a_c['full_runs_samples'])
    fullruns_time = time() - t0
    times["full_runs_time"] = f"{round(fullruns_time)} sec (i.e. {round(fullruns_time / 60)} min) --> {round(3600 * (a_c['full_runs_samples'])/ fullruns_time)} runs per hour"
    
    # Plot targetted outputs
    selected_outputs = bcm.targets.keys()
    for output in selected_outputs:
        _, ax = plt.subplots()
        pl.plot_model_fit_with_uncertainty(ax, unc_dfs['baseline'], output, bcm, x_lim=(2010, model_config['end_time']))
        if output_folder:
            plt.savefig(output_folder / f"quantiles_{output}.jpg", facecolor="white", bbox_inches='tight')
            plt.close()

    # Save quantile df outputs
    diff_output_quantiles = calculate_diff_output_quantiles(full_runs)
    for sc, unc_df in unc_dfs.items():
        unc_df.to_parquet(output_folder / f"uncertainty_df_{sc}.parquet")
        if sc in diff_output_quantiles:
            diff_output_quantiles[sc].to_parquet(output_folder / f"diff_quantiles_df_{sc}.parquet")

    if output_folder:
        # Read git commit id to be saved as part of details log file
        repo = Repo(search_parent_directories=True)
        commit = repo.head.commit.hexsha

        with open(output_folder / 'details.yaml', 'w') as file:
            yaml.dump_all([
                times, 
                model_config, 
                a_c | {"scenarios": sc_ids},
                {"commit": commit}
            ], file, default_flow_style=False)


def calculate_diff_output_quantiles(full_runs, quantiles=[.025, .25, .5, .75, .975], ref_sc='baseline'):
    diff_names = {
        "TB_averted": "tb_incidence", # need to work with cumulative outputs here
    }
    
    latest_time = full_runs['baseline'].results.index.max()
    
    runs_0_latest = full_runs[ref_sc].results.loc[latest_time]
    
    diff_output_quantiles = {}
    for sc in full_runs:
        if sc == ref_sc:
             continue
        
        sc_runs_latest = full_runs[sc].results.loc[latest_time]

        abs_diff = runs_0_latest - sc_runs_latest
        rel_diff = (runs_0_latest - sc_runs_latest) / sc_runs_latest
        
        diff_quantiles_df_abs = pd.DataFrame(
            index=quantiles, 
            data={colname: abs_diff[output_name].quantile(quantiles) for colname, output_name in diff_names.items()}
        )   
        diff_quantiles_df_rel = pd.DataFrame(
            index=quantiles, 
            data={f"{colname}_relative" : rel_diff[output_name].quantile(quantiles) for colname, output_name in diff_names.items()}
        ) 

        diff_output_quantiles[sc] = pd.concat([diff_quantiles_df_abs, diff_quantiles_df_rel], axis=1)


    return diff_output_quantiles
