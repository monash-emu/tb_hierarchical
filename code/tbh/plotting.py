from pathlib import Path
import arviz as az
from math import ceil

from copy import copy
import matplotlib.pyplot as plt
import numpy as np

from estival import priors as esp


def plot_traces(idata, burn_in, output_folder_path=None):
    az.rcParams["plot.max_subplots"] = 60 # to make sure all parameters are included in trace plots
    chain_length = idata.sample_stats.sizes['draw']

    # burn data
    burnt_idata = idata.sel(draw=range(burn_in, chain_length))  # Discard burn-in

    # Traces (after burn-in)
    az.plot_trace(burnt_idata, figsize=(16, 3.0 * len(idata.posterior)), compact=False);
    plt.subplots_adjust(hspace=.7)
    if output_folder_path:
        plt.savefig(output_folder_path / "mc_traces.jpg", facecolor="white", bbox_inches='tight')
        plt.close()


def plot_post_prior_comparison(
    idata: az.InferenceData,
    req_vars: list,
    priors: list,
    n_col=4,
    req_size=None,
    output_folder_path=None
) -> plt.figure:
    """Plot comparison of calibration posterior estimates
    for parameters against their prior distributions.

    Args:
        idata: Calibration inference data
        req_vars: Names of the parameters to plot
        priors: Prior distributions for the parameters
        n_col: Requested number of columns
        req_size: Figure size request

    Returns:
        The figure
    """
    n_row = ceil(len(req_vars) / n_col) 
    grid = [n_row, n_col]
    size = req_size if req_size else None
    fig = az.plot_density(idata, var_names=req_vars, shade=0.3, grid=grid, figsize=size, hdi_prob=1.)
    for i_ax, ax in enumerate(fig.ravel()):
        ax_limits = ax.get_xlim()
        param = ax.title.get_text().split("\n")[0]
        if param:
            x_vals = np.linspace(*ax_limits, 50)
            distri = priors[i_ax]

            if type(distri) != esp.TruncNormalPrior:
                y_vals = np.exp(distri.logpdf(x_vals))
                
                ax.fill_between(x_vals, y_vals, color="k", alpha=0.2, linewidth=2)
    # ax.figure.suptitle(country, fontsize=30, y=1.0)

    if output_folder_path:
        plt.savefig(output_folder_path / "mc_posteriors.jpg", facecolor="white", bbox_inches='tight')
        plt.close()
        
    return ax.figure.tight_layout()


def plot_model_fit_with_uncertainty(axis, uncertainty_df, output_name, bcm, include_legend=True, x_min=None):

    # update_rcparams() 
   
    df = uncertainty_df[output_name]

    if output_name in bcm.targets:
        t = copy(bcm.targets[output_name].data)
        axis.scatter(list(t.index), t, marker=".", color='black', label='observations', zorder=11, s=5.)

    colour = (0.2, 0.2, 0.8)   

    time = df.index
    axis.plot(time, df[0.5], color=colour, zorder=10, label="model (median)")

    axis.fill_between(
        time, 
        df[0.25], df[0.75], 
        color=colour, 
        alpha=0.5, 
        edgecolor=None,
        label="model (IQR)"
    )
    axis.fill_between(
        time, 
        df[0.025], df[0.975],
        color=colour, 
        alpha=0.3,
        edgecolor=None,
        label="model (95% CI)",
    )

    if output_name == "transformed_random_process":
        axis.set_ylim((0., axis.get_ylim()[1]))
    
    if x_min:
        axis.set_xlim((x_min, axis.get_xlim()[1]))

    # axis.tick_params(axis="x", labelrotation=45)
    title = output_name # if output_name not in title_lookup else title_lookup[output_name]

    axis.set_ylabel(title)
    # plt.tight_layout()

    if include_legend:
        plt.legend(markerscale=2.)