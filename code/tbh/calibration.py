import pandas as pd

from estival import priors as esp
from estival import targets as est
from estival.model import BayesianCompartmentalModel

from .model import get_tb_model


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


def get_targets():
    """
    Define calibration targets
    """
    return [
        # est.BinomialTarget(
        #     "ltbi_propXmajuro",
        #     data=pd.Series(data=[0.38], index=[2018.0]),
        #     sample_sizes=pd.Series(data=[19583], index=[2018.0]),
        # ),
        est.NormalTarget(
            "ltbi_propXmajuro", 
            data=pd.Series(data=[.38], index=[2018]), 
            stdev=.038,  # to get 95% CI within +-20% of central value   #esp.UniformPrior("std_ltbi", [.001, .1])
        ),
        est.NormalTarget(
            "tb_prevalence_per100kXmajuro",
            data=pd.Series(data=[1366], index=[2018]),
            stdev= 136.6 # to get 95% CI within +-20% of central value   # esp.UniformPrior("std_tb", [10.0, 250.0]),
        ),
        est.NormalTarget(
            "raw_notificationsXmajuro",
            data=pd.Series(data=[100], index=[2015]),
            stdev=10. # to get 95% CI within +-20% of central value  # esp.UniformPrior("std_not", [1.0, 25.0]),
        ),


        est.NormalTarget(
            "ltbi_propXstudy_2", 
            data=pd.Series(data=[.45], index=[2019]), 
            stdev=.045,  # to get 95% CI within +-20% of central value   #esp.UniformPrior("std_ltbi", [.001, .1])
        ),
        est.NormalTarget(
            "tb_prevalence_per100kXstudy_2",
            data=pd.Series(data=[400.], index=[2017]),
            stdev= 40. # to get 95% CI within +-20% of central value   # esp.UniformPrior("std_tb", [10.0, 250.0]),
        ),
        est.NormalTarget(
            "raw_notificationsXstudy_2",
            data=pd.Series(data=[104000], index=[2018]),
            stdev=10400. # to get 95% CI within +-20% of central value  # esp.UniformPrior("std_not", [1.0, 25.0]),
        ),


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
    targets = get_targets()
    model = get_tb_model(model_config, studies_dict)

    return BayesianCompartmentalModel(model, params, priors, targets)

