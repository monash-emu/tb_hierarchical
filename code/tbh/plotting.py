from pathlib import Path
import arviz as az
from math import ceil

from copy import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import numpy as np

from estival import priors as esp


title_lookup = {
    "tb_incidence": "TB incidence",
    "tb_incidence_per100k": "TB incidence (/100k)",
    "tbi_prevalence_perc": "TB infection prev. (%)",
    "tb_mortality_per100k": "TB mortality (/100k)",
    "cum_tb_incidence": "N TB episodes 2020-2050", 
    "cum_tb_mortality": "N TB deaths 2020-2050",
    "TB_averted": "N TB episodes averted (2020-2050)", 
    "TB_averted_relative": "% TB episodes averted (2020-2050)",

    "tb_prevalence_per100k": "TB prevalence (/100k)",
    "tbi_prevalence_perc": "TBI prevalence (%)",
    "perc_prev_subclinical": "Prevalent subclinical TB (%)",
    "perc_prev_infectious": "Prevalent infectious TB (%)",
    "notifications": "TB notifications",
    "perc_notifications_clin": "Clinical notifications (%)"
}

from tbh.runner_tools import DEFAULT_ANALYSIS_CONFIG
sc_names = {
    "baseline": "No intervention", 
} | {scenario.sc_id: scenario.sc_name for scenario in DEFAULT_ANALYSIS_CONFIG['scenarios']}

sc_colours = ["black", "crimson"]
unc_sc_colours = ((0.2, 0.2, 0.8), (0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.8, 0.8, 0.2), (0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.8, 0.8, 0.2))


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
    burn_in: int,
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

    chain_length = idata.sample_stats.sizes['draw']
    burnt_idata = idata.sel(draw=range(burn_in, chain_length))  # Discard burn-in

    fig = az.plot_density(burnt_idata, var_names=req_vars, shade=0.3, grid=grid, figsize=size, hdi_prob=1.)   
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


def plot_multiple_posteriors(idata, burn_in=0, req_vars=None, output_folder_path=None):
    """
    Plot overlaid posterior densities of selected variables on the same axis.

    Parameters:
    -----------
    idata : arviz.InferenceData
        The inference data containing posterior samples.
    burn_in : int
        Number of initial samples to discard from the beginning of the chain.
    req_vars : list of str
        List of variable names to plot from the posterior.
    output_folder_path : str or None
        If provided, saves the plot to this folder. Otherwise, shows the plot.
    """

    if req_vars is None:
        raise ValueError("You must specify the list of variables to plot via `req_vars`.")

    posterior = idata.posterior
    if burn_in > 0:
        posterior = posterior.isel(draw=slice(burn_in, None))

    # Set a colormap and get N distinct colors
    cmap = cm.get_cmap("tab10", len(req_vars))
    colors = [cmap(i) for i in range(len(req_vars))]

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, var in enumerate(req_vars):
        values = posterior[var].values.flatten()
        az.plot_kde(
            values, ax=ax, label=var, plot_kwargs={"color": colors[i]}, bw='silverman',
            fill_kwargs={"alpha": 0.3, "color": colors[i]}
            )

    ax.set_title("Posterior Distributions")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if output_folder_path:
        plt.savefig(output_folder_path / "overlaid_posteriors.jpg", facecolor="white", bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_model_fit_with_uncertainty(axis, uncertainty_df, output_name, bcm, include_legend=True, x_lim=None):

    # update_rcparams() 
   
    df = uncertainty_df[output_name]
    if x_lim:
        df = df.loc[x_lim[0]:x_lim[1]]

    if output_name in bcm.targets:
        t = copy(bcm.targets[output_name].data)
        axis.scatter(list(t.index), t, marker=".", color='black', label='observations', zorder=11, s=30.)

    colour = (0.2, 0.2, 0.8)   

    time = df.index
    axis.plot(time, df['0.5'], color=colour, zorder=10, label="model (median)")

    axis.fill_between(
        time, 
        df['0.25'], df['0.75'], 
        color=colour, 
        alpha=0.5, 
        edgecolor=None,
        label="model (IQR)"
    )
    axis.fill_between(
        time, 
        df['0.025'], df['0.975'],
        color=colour, 
        alpha=0.3,
        edgecolor=None,
        label="model (95% CI)",
    )

    # axis.tick_params(axis="x", labelrotation=45)
    title = output_name if output_name not in title_lookup else title_lookup[output_name]

    axis.set_ylabel(title)
    # plt.tight_layout()

    # Get existing y-limits
    ymin, ymax = axis.get_ylim()
    axis.set_ylim(0., 1.2 * ymax)

    if include_legend:
        plt.legend(markerscale=2.)


def plot_two_scenarios(axis, uncertainty_dfs, output_name, scenarios, xlim, include_unc=False, include_legend=True):
    ymax = 0.
    for i_sc, scenario in enumerate(scenarios):
        df = uncertainty_dfs[scenario][output_name].loc[xlim[0]:xlim[1]]
        median_df = df['0.5']
        time = df.index
        
        colour = unc_sc_colours[i_sc]
        label = sc_names[scenario]
        scenario_zorder = 10 if i_sc == 0 else i_sc + 2

        if include_unc:
            axis.fill_between(
                time, 
                df['0.25'], df['0.75'], 
                color=colour, alpha=0.7, 
                edgecolor=None,
                zorder=scenario_zorder
            )
            ymax = max(ymax, df['0.75'].max())
        else:
            ymax = median_df.max()

        axis.plot(time, median_df, color=colour, label=label, lw=1.)
        
    plot_ymax = ymax * 1.1    

    # axis.tick_params(axis="x", labelrotation=45)
    title = output_name if output_name not in title_lookup else title_lookup[output_name]
    axis.set_ylabel(title)
    # axis.set_xlim((model_start, model_end))
    axis.set_ylim((0, plot_ymax))

    if include_legend:
        axis.legend(title="(median and IQR)")


def plot_final_size_compare(axis, uncertainty_dfs, output_name, scenarios):
    box_width = .5
    color = 'black'
    box_color= 'lightcoral'
    y_max = 0
    for i, scenario in enumerate(scenarios):      
        df = uncertainty_dfs[scenario][output_name].iloc[-1]

        x = 1 + i
        # median
        axis.hlines(y=df['0.5'], xmin=x - box_width / 2. , xmax= x + box_width / 2., lw=1., color=color, zorder=3)    
        
        # IQR
        q_75 = float(df['0.75'])
        q_25 = float(df['0.25'])
        rect = Rectangle(xy=(x - box_width / 2., q_25), width=box_width, height=q_75 - q_25, zorder=2, facecolor=box_color)
        axis.add_patch(rect)

        # 95% CI
        q_025 = float(df['0.025'])
        q_975 = float(df['0.975'])
        axis.vlines(x=x, ymin=q_025 , ymax=q_975, lw=.7, color=color, zorder=1)

        y_max = max(y_max, q_975)
        
    title = output_name if output_name not in title_lookup else title_lookup[output_name]
    axis.set_ylabel(title)
    axis.set_xticks(ticks=range(1, len(scenarios) + 1), labels=[sc_names[sc] for sc in scenarios]) #, fontsize=15)

    axis.set_xlim((0.5, 0.5 + len(scenarios)))
    axis.set_ylim((0, y_max * 1.2))


def plot_diff_outputs(axis, diff_quantiles_dfs, output_name, scenarios):

    box_width = .2
    med_color = 'white'
    box_color= 'black'
    y_max_abs = 0.
    for i, sc in enumerate(scenarios):

        diff_output_df = diff_quantiles_dfs[sc]
        data = diff_output_df[output_name] 
        
        if output_name.endswith("_relative"):  # use %
            data = data * 100.

        # use %. And use "-" so positive nbs indicate positive effect of closures
        x = 1 + i
        # median
        axis.hlines(y=data.loc[0.5], xmin=x - box_width / 2. , xmax= x + box_width / 2., lw=2., color=med_color, zorder=3)    
        
        # IQR
        q_75 = data.loc[0.75]
        q_25 = data.loc[0.25]
        rect = Rectangle(xy=(x - box_width / 2., q_25), width=box_width, height=q_75 - q_25, zorder=2, facecolor=box_color)
        axis.add_patch(rect)

        # 95% CI
        q_025 = data.loc[0.025]
        q_975 = data.loc[0.975]
        axis.vlines(x=x, ymin=q_025 , ymax=q_975, lw=1.5, color=box_color, zorder=1)

        y_max_abs = max(abs(q_975), y_max_abs)
        y_max_abs = max(abs(q_025), y_max_abs)
 
    y_label = output_name if output_name not in title_lookup else title_lookup[output_name]  
    axis.set_ylabel(y_label)
   
    x_labels = [sc_names[sc] for sc in scenarios]
    axis.set_xticks(ticks=range(1, len(scenarios) + 1), labels=x_labels) #, fontsize=15)

    axis.set_xlim((0.5, len(scenarios) + 0.5))
    axis.set_ylim(0., 1.05 * y_max_abs)