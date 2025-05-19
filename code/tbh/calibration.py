import pandas as pd

from estival import priors as esp
from estival import targets as est
from estival.model import BayesianCompartmentalModel

from .model import get_tb_model

TARGETS = {
    "majuro": {
        "ltbi_prop": pd.Series(data=[.38], index=[2018]),
        "tb_prevalence_per100k": pd.Series(data=[1366], index=[2018]),
        "raw_notifications": pd.Series(data=[100], index=[2015])
    },
    "camau": {
        "ltbi_prop": pd.Series(data=[.368], index=[2016]),   #FIXME: Need to check year
        "tb_prevalence_per100k": pd.Series(data=[389], index=[2014]),
        "raw_notifications": pd.Series(data=[1040], index=[2018])  #FIXME: only took 10% of Vietnam notifications
    },

    "vietnam": {
        "ltbi_prop": pd.Series(data=[.45], index=[2019]),
        "tb_prevalence_per100k": pd.Series(data=[400], index=[2017]),
        "raw_notifications": pd.Series(data=[104000], index=[2018])
    },

    # For testing
    "majuro_copy": {
        "ltbi_prop": pd.Series(data=[.38], index=[2018]),
        "tb_prevalence_per100k": pd.Series(data=[1366], index=[2018]),
        "raw_notifications": pd.Series(data=[100], index=[2015])
    },
    "vietnam_no_target": {},
}


def get_priors(studies_dict: dict) -> list:
    """
    Define the list of prior distributions to be used for Bayesian calibration.

    Args:
        studies_dict (dict): Information about different studies

    Returns:
        prior_list: List of prior distributions
    """

    # Define hyper-prior distributions
    hyper_mean_lifelong = esp.UniformPrior("hyper_mean_lifelong", [0.0, 1.0])
    hyper_sd_lifelong = esp.UniformPrior("hyper_sd_lifelong", [0.0, 0.1])
    # hyper_mean_early = esp.UniformPrior("hyper_mean_early", [0., 1.])
    # hyper_sd_early = esp.UniformPrior("hyper_sd_early", [0., 10.])

    # Initialise the list of priors with "universal" priors and hyper-priors
    priors = [
        # esp.UniformPrior(f"current_passive_detection_rateXmajuro", [.1, 5.]),
        hyper_mean_lifelong,
        hyper_sd_lifelong,
        # hyper_mean_early,
        # hyper_sd_early
    ]

    # Complete the list of priors using study-specific priors
    for study in studies_dict:
        priors.extend(
            [
                # Independent priors (not linked through hierarchical structure)
                esp.UniformPrior(f"transmission_rateX{study}", [.01, 5.0]),
                esp.UniformPrior(f"current_passive_detection_rateX{study}", [0.1, 1.0]),
                # Priors linked through the previously defined hyper-prior distributions
                esp.TruncNormalPrior(
                    f"lifelong_activation_riskX{study}",
                    hyper_mean_lifelong,
                    hyper_sd_lifelong,
                    [0.0, 1.0],
                ),
                # esp.TruncNormalPrior(f"lifelong_activation_riskX{study}", hyper_mean_lifelong, .01, [0., 1.]),
                # esp.TruncNormalPrior(f"prop_early_among_activatorsX{study}", hyper_mean_early, hyper_sd_early, [0., 1.]),
            ]
        )
    return priors


def get_targets(studies_dict: dict) -> list:
    """
    Define calibration targets based on requested studies
    For each target, we use a normal likelihood with a fixed 
    standard deviation set to ensure that the 95% confidence 
    interval covers +/-20% of the central value.
    """

    return [
        est.NormalTarget(
            f"{key}X{study_name}", 
            data=data, 
            stdev=float(data.iloc[0]) / 10.,  # 4.sd = 95%CI = 40% of central estimate
        )
        for study_name, study_details in studies_dict.items()
        for key, data in TARGETS[study_name].items() if key in study_details["included_targets"]
    ]


def get_bcm_object(
    model_config: dict, studies_dict: dict, params: dict
) -> BayesianCompartmentalModel:
    """

    Args:
        model_config (dict): _description_
        studies_dict (dict): _description_
        params (dict): model parameters (used to inform non-calibrated parameters)

    Returns:
        bcm: estival BayesianCompartmentalModel object ready for calibration

    """
    priors = get_priors(studies_dict)
    targets = get_targets(studies_dict)
    model = get_tb_model(model_config, studies_dict)

    return BayesianCompartmentalModel(model, params, priors, targets)

