from estival import priors as esp


def get_prior(param_name, distribution, distri_param1, distri_param2=None):
    
    if distribution == "uniform":
        return esp.UniformPrior(param_name, [distri_param1, distri_param2])
    else:
        raise ValueError(f"{distribution} is not currently a supported distribution")
    

def get_targets():
    """
    Placeholder for now
    """
    return []