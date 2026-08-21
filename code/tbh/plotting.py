from pathlib import Path
import arviz as az
from math import ceil

from copy import copy
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

import numpy as np

from estival import priors as esp

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None


title_lookup = {
    "tb_incidence": "TB incidence",
    "tb_incidence_per100k": "TB incidence (/100k/y)",
    "tb_mortality_per100k": "TB mortality (/100k/y)",
    "cum_tb_incidence": "N TB episodes 2026-2035", 
    "cum_tb_mortality": "N TB deaths 2026-2035",
    "TB_averted": "N TB episodes averted (2026-2035)", 
    "TB_averted_relative": "% TB episodes averted (2026-2035)",
    "deaths_averted": "N TB deaths averted (2026-2035)", 
    "deaths_averted_relative": "% TB deaths averted (2026-2035)",

    "tb_prevalence_per100k": "TB prevalence (/100k)",
    "tbi_prevalence_perc": "TBI prevalence (%)",
    "perc_prev_subclinicalXreach_reachable": "% TB subclinical",
    "perc_prev_infectiousXreach_reachable": "% TB more infectious",
    "notifications": "TB notifications (n)",
    "perc_notifications_clin": "Clinical notifications (%)",

    "viable_tbi_prevalence_perc": "Viable infection prevalence (%)",
    "tst_posXreach_reachable_perc": "TST positivity (%)",
    "pearl_posXreach_reachable_per100k": "PEARL TB prevalence (/100k)",
    "cxr_posXreach_reachable_per100k": "CXR TB prevalence (/100k)",

    "passive_detection_rate_clin": "Passive detec. rate (/y), clinical TB",

    "tst_posXage_3_9Xreach_reachable_perc": "TST positivity 3-9yrs old (%)",
    "tst_posXage_10Xreach_reachable_perc": "TST positivity 10-14yrs old (%)",
    "tst_posXage_15+Xreach_reachable_perc": "TST positivity 15+yrs old (%)",
    "tst_posXage_18+Xreach_reachable_perc": "TST positivity 18+yrs old (%)",

}

from tbh.runner_tools import DEFAULT_ANALYSIS_CONFIG
SC_NAMES = {
    "baseline": "No intervention", 
} | {scenario.sc_id: scenario.sc_name for scenario in DEFAULT_ANALYSIS_CONFIG["scenarios"]}

sc_colours = ["black", "crimson"]
UNC_SC_COLORS = ((0.2, 0.2, 0.8), (0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.8, 0.8, 0.2), (0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.8, 0.8, 0.2))



def plot_traces(idata, bcm, burn_in=0, n_col=3):
    posterior = idata.posterior
    if burn_in > 0 and "draw" in posterior.dims:
        posterior = posterior.isel(draw=slice(burn_in, None))

    trace_params = [p for p in bcm.priors.keys() if p in posterior.data_vars]

    n_row = ceil(len(trace_params) / n_col)
    fig, axes = plt.subplots(n_row, n_col, figsize=(6.5 * n_col, 2.8 * n_row), sharex=False)

    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Fix one color per chain across all parameter panels.
    chain_ids = sorted(list(posterior[trace_params[0]]["chain"].values))
    base_colors = list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors)
    if len(chain_ids) > len(base_colors):
        repeats = ceil(len(chain_ids) / len(base_colors))
        base_colors = (base_colors * repeats)[:len(chain_ids)]
    chain_color_map = {chain_id: base_colors[i] for i, chain_id in enumerate(chain_ids)}

    for i, param_name in enumerate(trace_params):
        ax = axes[i]
        da = posterior[param_name]

        # For non-scalar parameters, average over extra dims to keep one trace per chain.
        extra_dims = [d for d in da.dims if d not in ["chain", "draw"]]
        if extra_dims:
            da = da.mean(dim=extra_dims)

        draw_values = da["draw"].values
        for chain_value in chain_ids:
            chain_series = da.sel(chain=chain_value).values
            ax.plot(
                draw_values,
                chain_series,
                linewidth=0.9,
                alpha=0.85,
                color=chain_color_map[chain_value],
            )

        title_suffix = " (mean over extra dims)" if extra_dims else ""
        ax.set_title(f"{param_name}{title_suffix}", fontsize=10)
        ax.set_xlabel("Draw")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)

    for j in range(len(trace_params), len(axes)):
        fig.delaxes(axes[j])

    legend_handles = [
        mlines.Line2D([], [], color=chain_color_map[c], linewidth=1.8, label=f"chain {c}")
        for c in chain_ids
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02), 
        ncol=min(9, len(chain_ids)),
        frameon=False,
        fontsize=14
    )

    return fig


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
        
    ax.figure.tight_layout()
    
    return ax.figure


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


def plot_posterior_pairs(
    idata: az.InferenceData,
    burn_in: int,
    req_vars: list,
    kind='scatter',
    output_folder_path=None
) -> plt.Figure:
    """
    Generate a pairwise posterior plot for selected parameters after burn-in.

    This function discards the specified number of initial MCMC samples (burn-in) and
    produces a pair plot showing the marginal posterior distributions and pairwise
    joint distributions for the requested variables.

    Parameters
    ----------
    idata : az.InferenceData
        ArviZ InferenceData object containing posterior samples.
    burn_in : int
        Number of initial samples to discard as burn-in.
    req_vars : list of str
        List of variable names to include in the pairwise posterior plot.
    output_folder_path : pathlib.Path or str, optional
        Directory path where the figure will be saved as 'mc_posteriors.jpg'.
        If None, the figure is displayed but not saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the pairwise posterior plot.
    """
    
    az.rcParams["plot.max_subplots"]=200

    # Discard burn-in samples
    chain_length = idata.sample_stats.sizes['draw']
    burnt_idata = idata.sel(draw=range(burn_in, chain_length))

    # --- Plot pairwise posterior distributions ---
    axes = az.plot_pair(
        burnt_idata,
        var_names=req_vars,
        kind=kind,           # Smooth density estimates
        marginals=True,       # Include 1D marginal distributions
        point_estimate="mode",
        divergences=True,
        figsize=(30, 27),
        kde_kwargs={
             "contourf_kwargs": {"cmap": "hot_r"},
        },
    )

    # Compute correlations
    samples = burnt_idata.posterior.stack(draws=("chain", "draw"))[req_vars].to_dataframe()
    corr_matrix = samples.corr()

    # n = len(req_vars)
    # for i in range(n):
    #     for j in range(i):  # Only lower triangle
    #         ax = axes[i, j]  # safe, lower triangle only
    #         corr = corr_matrix.iloc[i, j]
    #         ax.text(
    #             0.5, 0.9, f"r = {corr:.2f}",  # Place near top-centre
    #             transform=ax.transAxes,
    #             ha="center",
    #             fontsize=16,
    #             color="black"
    #         )


    for ax in np.ravel(axes):
        ax.set_xlabel(ax.get_xlabel(), fontsize=17, rotation=15) #, ha='right', va='top')
        ax.set_ylabel(ax.get_ylabel(), fontsize=17, rotation=15, ha='right', va='bottom')
        ax.tick_params(axis='both', labelsize=10)

        if kind == 'scatter':
            for coll in ax.collections:
                coll.set_rasterized(True)

    # Ensure we have a Figure handle
    fig = axes.flat[0].figure

    # Save or display
    if output_folder_path:
        plt.savefig(output_folder_path / "mc_posteriors.jpg", facecolor="white", bbox_inches="tight")
        plt.close(fig)
    else:
        fig.tight_layout()

    return fig


def plot_model_fit_with_uncertainty(axis, uncertainty_df, output_name, bcm, x_lim=None, colour="#B22222",target_ms=15):

    # update_rcparams() 
   
    df = uncertainty_df[output_name]
    if x_lim:
        df = df.loc[x_lim[0]:x_lim[1]]

    if output_name in bcm.targets:
        t = copy(bcm.targets[output_name].data)
        axis.scatter(list(t.index), t, marker=".", color='black', label='Observed', zorder=11, s=target_ms)

    time = df.index
    axis.plot(time, df['0.5'], color=colour, zorder=10, label="Model (median)")

    axis.fill_between(
        time, 
        df['0.25'], df['0.75'], 
        color=colour, 
        alpha=0.35,  # 0.5, 
        edgecolor=None,
        label="Model (IQR)"
    )
    axis.fill_between(
        time, 
        df['0.025'], df['0.975'],
        color=colour, 
        alpha=0.20, # 0.3,
        edgecolor=None,
        label="Model (95% CrI)",
    )

    # axis.tick_params(axis="x", labelrotation=45)
    title = output_name if output_name not in title_lookup else title_lookup[output_name]

    axis.set_ylabel(title)
    
    # Format x-axis to show years as integers
    axis.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    
    # plt.tight_layout()

    # Get existing y-limits
    ymin, ymax = axis.get_ylim()
    axis.set_ylim(0., 1.2 * ymax)


def plot_all_model_fits(uncertainty_df, bcm, n_col=3, excluded_outputs=[]):

    selected_outputs = list(bcm.targets.keys())
    selected_outputs = [o for o in selected_outputs if o not in excluded_outputs]

    n_row = ceil(len(selected_outputs) / n_col)

    fig, axes = plt.subplots(n_row, n_col, figsize=(5 * n_col, 3.6 * n_row))
    axes = axes.flatten()  # Flatten to simplify indexing

    for i, output in enumerate(selected_outputs):
        ax = axes[i]
        out_name = output if output not in title_lookup else title_lookup[output]
        x_min = 1995 if output == "notifications" else 2010
        plot_model_fit_with_uncertainty(ax, uncertainty_df, output, bcm, x_lim=(x_min, 2025))
        ax.set_title(out_name)
        if i == 0:
            ax.legend()

    # Hide any unused axes
    for j in range(len(selected_outputs), len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()

    return fig


def plot_two_scenarios(axis, uncertainty_dfs, output_name, scenarios, xlim, include_unc=False, include_legend=True, ylab_fontsize=12, unc_sc_colours=UNC_SC_COLORS, sc_names=SC_NAMES):
    ymax = 0.
    for i_sc, scenario in enumerate(scenarios):
        data_xmin = xlim[0] if scenario == "baseline" else 2026

        df = uncertainty_dfs[scenario][output_name].loc[data_xmin:xlim[1]]
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
            axis.fill_between(
                time, 
                df['0.025'], df['0.975'], 
                color=colour, alpha=0.4, 
                edgecolor=None,
                zorder=scenario_zorder
            )
            ymax = max(ymax, df['0.975'].max())
        else:
            ymax = median_df.max()

        axis.plot(time, median_df, color=colour, label=label, lw=1.)
        
    plot_ymax = ymax * 1.1    

    # axis.tick_params(axis="x", labelrotation=45)
    title = output_name if output_name not in title_lookup else title_lookup[output_name]
    axis.set_ylabel(title, fontsize=ylab_fontsize)
    axis.set_xlim(xlim)
    axis.set_ylim((0, plot_ymax))
    
    # Format x-axis to show years as integers
    axis.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # years = range(xlim[0], xlim[1], 5)
    # axis.set_xticks(years)
    # axis.set_xticklabels([str(y) for y in years])

    if include_legend:
        leg = axis.legend(title="(median, IQR, 95% CrI)")
        leg._legend_box.align = "left"
        for handle in leg.legend_handles:
            handle.set_linewidth(2)


def plot_final_size_compare(axis, uncertainty_dfs, output_name, scenarios, end_year=2035, sc_names=SC_NAMES):
    box_width = .5
    color = 'black'
    box_color= 'lightcoral'
    y_max = 0
    for i, scenario in enumerate(scenarios):      
        df = uncertainty_dfs[scenario][output_name].loc[end_year]

        x = 1 + i
        # median
        axis.hlines(y=df['0.5'], xmin=x - box_width / 2. , xmax= x + box_width / 2., lw=1., color=color, zorder=3)    
        
        # IQR
        q_75 = float(df['0.75'])
        q_25 = float(df['0.25'])
        rect = mpatches.Rectangle(xy=(x - box_width / 2., q_25), width=box_width, height=q_75 - q_25, zorder=2, facecolor=box_color)
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


def plot_diff_outputs(axis, diff_quantiles_dfs, output_name, scenarios, sc_names=SC_NAMES, colour="#B22222"):

    box_width = .4
    med_color = 'white'
    box_color= colour
    y_max_abs = 0.
    for i, sc in enumerate(scenarios):

        diff_output_df = diff_quantiles_dfs[sc]
        data = diff_output_df[output_name] 
        
        if output_name.endswith("_relative"):  # use %
            data = data * 100.

        # use %. And use "-" so positive nbs indicate positive effect of closures
        x = 1 + i
        # median
        axis.hlines(y=data.loc[0.5], xmin=x - box_width / 2. , xmax= x + box_width / 2., lw=.7, color=med_color, zorder=3)    

        # axis.scatter(x, data.loc[0.5], color='black', s=3, zorder=4)

        
        # IQR
        q_75 = data.loc[0.75]
        q_25 = data.loc[0.25]
        rect = mpatches.Rectangle(xy=(x - box_width / 2., q_25), width=box_width, height=q_75 - q_25, zorder=2, facecolor=box_color)
        axis.add_patch(rect)

        # 95% CI
        q_025 = data.loc[0.025]
        q_975 = data.loc[0.975]
        axis.vlines(x=x, ymin=q_025 , ymax=q_975, lw=1, color=box_color, zorder=1)

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
        series = res.derived_outputs[t_name].loc[1980:2025]
        
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
    # agegroups = ["3_9", "10", "15", "65"]
    agegroups = ["3_9", "10", "15+", "18+"]


    box_data = []
    targets = []

    # Collect quantile info per age group
    x_tick_labels = []
    for i_age, age in enumerate(agegroups):
        output_name = f"tst_posXage_{age}Xreach_reachable_perc"

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

        suffix = f"\n (year {year})"
        if age == "3_9":
            x_tick_labels.append("3-9" + suffix)
        elif age == "15+":
            x_tick_labels.append("15+" + suffix)
        elif age == "18+":
            x_tick_labels.append("18+" + suffix)
        else:
            if i_age < (len(agegroups) - 1):
                next_age = agegroups[i_age + 1].replace("+", "")
                x_tick_labels.append(f"{age}-{int(next_age) - 1}" + suffix)
            else:
                x_tick_labels.append(f"{age}+" + suffix) 


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
    ax.scatter(range(len(agegroups)), targets, color='red', marker='x', s=80, label='Observed')

    # Create proxy artists for legend
    model_patch = mpatches.Patch(facecolor='lightblue', edgecolor='navy', alpha=0.6, label='Modelled (quantiles)')
    obs_marker = mlines.Line2D([], [], color='red', marker='x', linestyle='None', markersize=8, label='Observed')

    # Labels and formatting
    ax.set_xticks(range(len(agegroups)))  
    ax.set_xticklabels(x_tick_labels)

    ax.set_xlabel("Age group (years)")
    ax.set_ylabel(title_lookup["tst_posXreach_reachable_perc"])
    ax.set_title(f"Observed vs modelled TST positivity fraction by age group in {year}")
    ax.legend(handles=[model_patch, obs_marker], loc='best')
    ax.grid(alpha=0.3)

    ax.set_ylim(bottom=0)  # ensures y-axis starts at 0

    plt.tight_layout()
    # plt.show()

    return fig


def plot_contact_matrix(M, age_groups, title, ax, cmap="viridis"):
    """
    Plot a contact matrix as a heatmap with ticks aligned to cell centres.
    """
    n = len(age_groups)

    # fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        M,
        origin="upper",
        cmap=cmap,
        aspect="auto",
        interpolation="none"
    )

    # Major ticks at cell centres
    age_lb = [int(a) for a in age_groups]
    labels = (
        [f"{age_lb[i]}-{age_lb[i+1] - 1}" for i in range(len(age_lb) - 1)]
        + [f"{age_lb[-1]}+"]
    )

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Move x-axis to the top
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", top=True, bottom=False)

    # Set axis limits to match matrix extent exactly
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    # Draw gridlines on cell boundaries
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_ylabel("Contacting individual age group (i)")
    ax.set_xlabel("Contacted individual age group (j)")
    ax.set_title(title)

    plt.setp(ax.get_xticklabels(), rotation=20, ha="center")

    # cbar = fig.colorbar(im, ax=ax)
    # cbar.ax.set_ylabel("Contact rate", rotation=270, labelpad=15)

    plt.tight_layout()
    # plt.show()


# ============================================================================
# Manuscript-specific plotting functions
# ============================================================================


def plot_diff_outputs_horizontal(axis, diff_quantiles_dfs, output_name, scenarios, sc_names=SC_NAMES):
    """
    Plot diff outputs as horizontal bars instead of vertical.
    
    Parameters
    ----------
    axis : matplotlib.axes.Axes
        The axis to plot on
    diff_quantiles_dfs : dict
        Dictionary of scenario -> diff_df DataFrames
    output_name : str
        Column name in diff DataFrames to plot (e.g., "TB_averted_relative")
    scenarios : list
        List of scenario IDs to plot
    sc_names : dict
        Dictionary mapping scenario IDs to display names
    """
    box_height = 0.5
    med_color = 'white'
    box_color = 'black'
    x_max_abs = 0.
    
    for i, sc in enumerate(scenarios):
        diff_output_df = diff_quantiles_dfs[sc]
        data = diff_output_df[output_name]
        
        if output_name.endswith("_relative"):  # use %
            data = data * 100.
        
        y = i  # y-position on axis
        
        # median
        axis.vlines(x=data.loc[0.5], ymin=y - box_height / 2., ymax=y + box_height / 2., lw=2., color=med_color, zorder=3)
        
        # IQR
        q_75 = data.loc[0.75]
        q_25 = data.loc[0.25]
        rect = mpatches.Rectangle(
            xy=(q_25, y - box_height / 2.), 
            width=q_75 - q_25, 
            height=box_height, 
            zorder=2, 
            facecolor=box_color
        )
        axis.add_patch(rect)
        
        # 95% CI
        q_025 = data.loc[0.025]
        q_975 = data.loc[0.975]
        axis.hlines(y=y, xmin=q_025, xmax=q_975, lw=1.5, color=box_color, zorder=1)
        
        x_max_abs = max(abs(q_975), x_max_abs)
        x_max_abs = max(abs(q_025), x_max_abs)
    
    x_label = output_name if output_name not in title_lookup else title_lookup[output_name]
    axis.set_xlabel(x_label)
    
    y_labels = [sc_names[sc] for sc in scenarios]
    axis.set_yticks(ticks=range(len(scenarios)), labels=y_labels)
    
    axis.set_xlim(0., 1.05 * x_max_abs)
    axis.set_ylim(-0.5, len(scenarios) - 0.5)


def plot_sensitivity_horizontal_bars(axis, sensitivity_df, output_col, scenarios, group_by, 
                                     colour_palette=None, group_label="Configuration"):
    """
    Plot sensitivity analysis as horizontal grouped bars.
    
    Parameters
    ----------
    axis : matplotlib.axes.Axes
        The axis to plot on
    sensitivity_df : pd.DataFrame
        DataFrame with columns: group_by, scenarios, output_col, and uncertainty cols
    output_col : str
        Name of column containing values to plot
    scenarios : list
        List of scenario values to group by
    group_by : str
        Column name to group by (e.g., "rel_sus_unreachable")
    colour_palette : dict, optional
        Dictionary mapping scenario values to colors
    group_label : str
        Label for the grouping variable
    """
    if colour_palette is None:
        colour_palette = {}
    
    groups = sorted(sensitivity_df[group_by].unique())
    n_groups = len(groups)
    bar_height = 0.15
    
    for i_scen, scenario in enumerate(scenarios):
        sub = sensitivity_df[sensitivity_df["scenario"] == scenario]
        
        y_positions = np.arange(n_groups) + (i_scen - len(scenarios)/2 + 0.5) * bar_height
        values = sub.set_index(group_by).reindex(groups)[output_col].values
        
        color = colour_palette.get(scenario, None)
        
        axis.barh(y_positions, values, bar_height, label=str(scenario), color=color, alpha=0.8)
    
    axis.set_yticks(np.arange(n_groups))
    axis.set_yticklabels([str(g) for g in groups])
    axis.set_ylabel(group_label)
    axis.set_xlabel("% TB episodes averted")
    axis.legend(title="Scenario", frameon=False)
    axis.grid(axis="x", alpha=0.3)


def save_figure_high_res(fig, output_path, dpi=300, formats=None):
    """
    Save a figure in multiple formats with high resolution.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    output_path : str or Path
        Base path without extension (will add .png, .pdf, etc.)
    dpi : int
        Resolution for raster formats
    formats : list, optional
        List of formats to save (default: ['png', 'pdf'])
    """
    from pathlib import Path
    
    if formats is None:
        formats = ['png', 'pdf']
    
    output_path = Path(output_path)
    
    for fmt in formats:
        file_path = output_path.parent / f"{output_path.stem}.{fmt}"
        if fmt == 'png':
            fig.savefig(file_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        else:
            fig.savefig(file_path, bbox_inches='tight')
        print(f"Saved: {file_path}")


def load_uncertainty_outputs_for_task(task_path, scenarios):
    """Load uncertainty output tables for the requested scenarios from one task folder."""
    task_path = Path(task_path)
    uncertainty_dfs = {}
    for scenario in scenarios:
        file_path = task_path / f"uncertainty_df_{scenario}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing uncertainty file for scenario '{scenario}': {file_path}")
        uncertainty_dfs[scenario] = pd.read_parquet(file_path)
    return uncertainty_dfs


def load_diff_outputs_for_task(task_path, scenarios):
    """Load baseline-referenced diff quantile tables for the requested scenarios from one task folder."""
    task_path = Path(task_path)
    diff_dfs = {}
    for scenario in scenarios:
        file_path = task_path / f"diff_quantiles_df_ref_baseline_{scenario}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing diff quantiles file for scenario '{scenario}': {file_path}")
        diff_dfs[scenario] = pd.read_parquet(file_path)
    return diff_dfs


def build_bcm_from_task_path(task_path):
    """Recreate the calibrated BCM context from a task folder's model configuration."""
    from estival.model import BayesianCompartmentalModel
    import tbh.runner_tools as rt
    from tbh.model import get_tb_model

    task_path = Path(task_path)
    details_path = task_path / "details.yaml"
    if not details_path.exists():
        raise FileNotFoundError(f"Missing task metadata file: {details_path}")

    with open(details_path, "r") as f:
        docs = list(yaml.safe_load_all(f))

    model_config = docs[1] if len(docs) > 1 else {}
    if not isinstance(model_config, dict):
        model_config = {}

    params, priors, tv_params = rt.get_parameters_and_priors()
    model = get_tb_model(model_config, tv_params)

    # Prevent crash if the mixing_matrix_distance not produced by the model (e.g., if not using age-structured mixing)
    targets = [t for t in rt.targets if t.name != "mixing_matrix_distance"]

    return BayesianCompartmentalModel(model, params, priors, targets)


def make_figure_1_calibration_from_task(
    task_path,
    model_pdf_path=None,
    include_model_panel=False,
    colour="#B22222",
    figsize=None,
    selected_outputs=None,
):
    """Generate manuscript Figure 1 from a task folder path."""
    if selected_outputs is None:
        selected_outputs = [
            "pearl_posXreach_reachable_per100k",
            "cxr_posXreach_reachable_per100k",
            "perc_prev_subclinicalXreach_reachable",
            "perc_prev_infectiousXreach_reachable",
            "notifications",
        ]

    unc_baseline = load_uncertainty_outputs_for_task(task_path, ["baseline"])["baseline"]
    bcm = build_bcm_from_task_path(task_path)

    use_model_panel = bool(include_model_panel and model_pdf_path)

    n_col = 3
    n_data_panels = len(selected_outputs) + 1
    n_data_row = ceil(n_data_panels / n_col)

    if figsize is None:
        figsize = (10.8, 8.5) if use_model_panel else (10.8, 5.2)

    fig = plt.figure(figsize=figsize)
    if use_model_panel:
        gs = gridspec.GridSpec(
            n_data_row + 1,
            n_col,
            figure=fig,
            height_ratios=[1.8] + [1] * n_data_row,
            hspace=0.2,
            wspace=0.3,
        )

        ax_pdf = fig.add_subplot(gs[0, :])
        ax_pdf.axis("off")
        if convert_from_path is not None:
            try:
                images = convert_from_path(model_pdf_path, dpi=300)
                if images:
                    ax_pdf.imshow(images[0])
            except Exception as exc:
                ax_pdf.text(0.5, 0.5, f"Error loading PDF: {exc}", ha="center", va="center")
        else:
            ax_pdf.text(
                0.5,
                0.5,
                "pdf2image not installed; skipping PDF panel rendering.",
                ha="center",
                va="center",
            )
        row_offset = 1
    else:
        gs = gridspec.GridSpec(
            n_data_row,
            n_col,
            figure=fig,
            hspace=0.35,
            wspace=0.3,
        )
        ax_pdf = None
        row_offset = 0

    axes = []
    for i in range(n_data_panels):
        row = row_offset + (i // n_col)
        col = i % n_col
        axes.append(fig.add_subplot(gs[row, col]))

    for i, output in enumerate(selected_outputs):
        ax = axes[i]
        x_min = 1995 if output == "notifications" else 2010
        plot_model_fit_with_uncertainty(
            ax,
            unc_baseline,
            output,
            bcm,
            x_lim=(x_min, 2025),
            colour=colour,
            target_ms=15,
        )
        if i == 0:
            ax.legend()

    ax = axes[len(selected_outputs)]
    agegroups = ["3_9", "10", "15+", "18+"]
    model_median, model_low, model_high, observed, x_tick_labels = [], [], [], [], []
    for age in agegroups:
        output_name = f"tst_posXage_{age}Xreach_reachable_perc"
        year = bcm.targets[output_name].data.index[0]
        quantiles = unc_baseline[output_name].loc[year]
        obs = bcm.targets[output_name].data.iloc[0]

        model_median.append(quantiles["0.5"])
        model_low.append(quantiles["0.025"])
        model_high.append(quantiles["0.975"])
        observed.append(obs)

        suffix = f" y.o.\\n({year})"
        if age == "3_9":
            x_tick_labels.append("3-9" + suffix)
        elif age == "15+":
            x_tick_labels.append("15+" + suffix)
        else:
            x_tick_labels.append(f"{age}" + suffix)

    x = range(len(agegroups))
    ax.errorbar(
        [i - 0.06 for i in x],
        model_median,
        yerr=[
            [m - l for m, l in zip(model_median, model_low)],
            [h - m for h, m in zip(model_high, model_median)],
        ],
        fmt="D",
        color=colour,
        ecolor=colour,
        markersize=3,
        elinewidth=2.0,
        capsize=0,
        label="Model (median, 95% CI)",
    )
    ax.scatter([i + 0.06 for i in x], observed, color="black", s=7, zorder=5, label="Observed")
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_tick_labels)
    ax.set_ylabel(title_lookup["tst_posXreach_reachable_perc"])

    model_handle = mlines.Line2D(
        [], [], color=colour, marker="D", markersize=3, linestyle="-", label="Model (median, 95% CrI)"
    )
    obs_handle = mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=3, label="Observed")
    ax.legend(handles=[obs_handle, model_handle], frameon=False, loc="best")

    letter_fontsize = 9
    letter_start = 0
    if use_model_panel:
        ax_pdf.text(
            -0.05,
            0.98,
            "a)",
            transform=ax_pdf.transAxes,
            fontsize=letter_fontsize,
            fontweight="bold",
            va="top",
        )
        letter_start = 1

    for i, axis in enumerate(axes):
        letter_idx = i + letter_start
        axis.text(
            -0.15,
            1.1,
            f"{chr(97 + letter_idx)})",
            transform=axis.transAxes,
            fontsize=letter_fontsize,
            fontweight="bold",
            va="top",
        )

    return fig


def make_figure_2_trajectories_from_task(
    task_path,
    trajectory_outputs,
    scenarios=("baseline", "scenario_3"),
    xlim=(2020, 2035),
    unc_sc_colours=("#B22222", "#54992c"),
    sc_names=SC_NAMES,
    figsize=(7, 4.65),
):
    """Generate manuscript Figure 2 from a task folder path."""
    unc_dfs = load_uncertainty_outputs_for_task(task_path, scenarios)

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=False)
    axes = axes.flatten()
    for ax, output in zip(axes, trajectory_outputs):
        plot_two_scenarios(
            ax,
            unc_dfs,
            output,
            scenarios=list(scenarios),
            xlim=xlim,
            include_unc=True,
            ylab_fontsize=9,
            unc_sc_colours=unc_sc_colours,
            include_legend=ax == axes[0],
            sc_names=sc_names,
        )

    panel_letters = [f"{chr(97 + i)})" for i in range(len(trajectory_outputs))]
    for i, letter in enumerate(panel_letters):
        axes[i].text(
            -0.15,
            1.05,
            letter,
            transform=axes[i].transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
        )

    fig.tight_layout()
    return fig


def make_figure_4_algorithms_coverage_from_task(
    task_path,
    scenarios_to_compare=None,
    colour="#B22222",
    figsize=(5, 5),
):
    """Generate manuscript Figure 4 from a task folder path."""
    if scenarios_to_compare is None:
        scenarios_to_compare = [
            "scenario_1",
            "scenario_2",
            "scenario_3",
            "scenario_6",
            "scenario_7",
            "scenario_8",
            "scenario_16",
            "scenario_17",
            "scenario_18",
        ]

    diff_dfs = load_diff_outputs_for_task(task_path, scenarios_to_compare)

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=False)

    group_labels = ["PEARL (CXR-Xpert-TST)", "CXR-TST", "Disease screening only (CXR)"]
    coverage_labels = ["65%", "75%", "85%"]
    xtick_labels = coverage_labels * len(group_labels)

    def _format_fig4_axis(ax):
        ax.set_xticks(range(1, len(scenarios_to_compare) + 1), xtick_labels)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

        for boundary in (3.5, 6.5):
            ax.axvline(boundary, color="0.5", linestyle="--", linewidth=0.9, alpha=0.9, zorder=0)

        for center, label in zip([2, 5, 8], group_labels):
            ax.text(
                center,
                0.98,
                label,
                ha="center",
                va="bottom",
                transform=ax.get_xaxis_transform(),
                fontsize=7,
            )
        ax.set_xlabel("Screening coverage")
        ax.set_ylim(0, 65)

    ax = axes[0]
    plot_diff_outputs(ax, diff_dfs, "TB_averted_relative", scenarios_to_compare, colour=colour)
    _format_fig4_axis(ax)
    ax.grid(axis="y", linestyle="-", linewidth=0.7, alpha=0.4)
    ax.set_title("a)", loc="left")

    ax = axes[1]
    plot_diff_outputs(ax, diff_dfs, "deaths_averted_relative", scenarios_to_compare, colour=colour)
    _format_fig4_axis(ax)
    ax.grid(axis="y", linestyle="-", linewidth=0.7, alpha=0.4)
    ax.set_title("b)", loc="left")

    fig.tight_layout()
    return fig
   
