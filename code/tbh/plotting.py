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



def plot_single_fit(bcm, params):

    res = bcm.run(params)
        
    # Number of targets
    n_targets = len(bcm.targets)
    n_cols = 3
    n_rows = ceil(n_targets / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()  # make it easy to index

    for i, (t_name, t) in enumerate(bcm.targets.items()):
        ax = axes[i]
        
        t_data = t.data
        series = res.derived_outputs[t_name].loc[2010:2025]
        
        # Plot main line
        series.plot(ax=ax, title=t_name)
        
        # Plot single-point data as dot
        t_data.plot(ax=ax, style="o")
        
        # Set y-limits
        ax.set_ylim(bottom=0)
        ymax = series.max()
        ax.set_ylim(top=max([1.3 * ymax, t_data.max()]))
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def visualise_mle_params(priors, mle_params):
    """
    Visualise MLE parameters relative to their priors.
    Each prior range is scaled to the same visual length (0–1).
    """
    fig, ax = plt.subplots(figsize=(6, len(mle_params) * 0.6))

    # Reverse order so the first param appears on top
    for i, (param_name, mle_val) in enumerate(reversed(mle_params.items())):
        lower = priors[param_name].start
        upper = priors[param_name].end
        y = i
        
        # Normalise the MLE to 0–1
        norm_mle = (mle_val - lower) / (upper - lower)

        # Draw normalised prior line (0–1 visually)
        ax.hlines(y, 0, 1, color='lightgrey', linewidth=4)
        
        # Plot the MLE point
        ax.plot(norm_mle, y, 'o', color='tab:red', markersize=8)
        
        # Add label on the right
        ax.text(1.05, y, f"{param_name}", va='center', fontsize=10)
        
        # Optionally show numeric values for reference
        ax.text(-0.05, y, f"[{lower}, {upper}]", va='center', ha='right', color='grey', fontsize=8)

    ax.set_xlim(-0.1, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("Normalised prior scale (0–1)")
    ax.set_title("MLE position within each prior range (equal visual lengths)")
    
        # Remove box (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_age_spec_tbi_prev(unc_df, bcm):
    agegroups = ["5", "10", "15", "65"]
    
    box_data = []
    targets = []

    # Collect quantile info per age group
    for age in agegroups:
        output_name = f"measured_tbi_prevalenceXage_{age}_perc"

        year = bcm.targets[output_name].data.index[0]
        quantiles = unc_df[output_name].loc[year]
        target = bcm.targets[output_name].data.iloc[0]

        # Store quantiles in order for boxplot
        box_data.append([
            quantiles['0.025'],
            quantiles['0.25'],
            quantiles['0.5'],
            quantiles['0.75'],
            quantiles['0.975']
        ])
        targets.append(target)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    # Custom boxplot (using pre-computed quantiles)
    bp = ax.bxp(
        [
            {
                'med': d[2],
                'q1': d[1],
                'q3': d[3],
                'whislo': d[0],
                'whishi': d[4],
                'fliers': []
            } for d in box_data
        ],
        positions=range(len(agegroups)),
        showfliers=False,
        patch_artist=True
    )

    # Style boxes
    for box in bp['boxes']:
        box.set(facecolor='lightblue', alpha=0.6, edgecolor='navy')
    for whisker in bp['whiskers']:
        whisker.set(color='navy', linewidth=1)
    for cap in bp['caps']:
        cap.set(color='navy', linewidth=1)
    for median in bp['medians']:
        median.set(color='darkblue', linewidth=2)

    # Overlay target points
    ax.scatter(range(len(agegroups)), targets, color='red', marker='x', s=80, label='Target')

    # Labels and formatting
    ax.set_xticks(range(len(agegroups)))
    ax.set_xticklabels(agegroups)
    ax.set_xlabel("Age group")
    ax.set_ylabel("TBI prevalence (%)")
    ax.set_title("Measured vs modelled TBI prevalence by age group")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()