import math
from math import pi

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, patches as mpatches
import seaborn as sns
from matplotlib.ticker import FixedLocator

from promisces.simulate_removal import SimulationResult


def er_profiles(sim_result: SimulationResult,
                save_as=None
                ):
    output_c_df = sim_result.output_c_df
    rmv_factor_df = sim_result.rmv_factor_df
    reference_value = sim_result.reference.ref_value_ng_l
    case_study = sim_result.case_study
    substance_name = sim_result.substance.name
    # set the scale for concentration y axis
    max_c = output_c_df.to_numpy().max()
    if reference_value is not None and (max_c / reference_value) > 100:
        c_scale = "log"
    else:
        c_scale = "linear"

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, hspace=.5)
    axs = gs.subplots()

    # Negative removal should be displayed reverse (input concentration / output Concentration),
    # so that every process is between 0 and 100%
    # maybe displayed in another color 
    # not implemented yet!!!!!
    sns.violinplot(
        data=output_c_df,
        color="silver",
        cut=0,
        density_norm="width",
        inner=None,
        saturation=0.5,
        ax=axs[0]
    )
    if case_study is not None:
        axs[0].set_title("At CS '" + case_study.name + "'" + f" - '{sim_result.scenario}'")
    axs[0].set_ylabel(substance_name + " (ng/L)")
    if reference_value is not None:
        axs[0].axhline(y=reference_value, color='black', linestyle='--')
    axs[0].set_xlabel('Treatment step outlet')
    axs[0].set_yscale(c_scale)

    sns.violinplot(
        data=rmv_factor_df,
        color="silver",
        cut=0,
        density_norm="width",
        inner=None,
        saturation=1,
        orient="h",
        ax=axs[1]
    )

    if case_study is not None:
        axs[1].set_title(substance_name + " at CS '" + case_study.name + "'" + f" - '{sim_result.scenario}'")
    axs[1].set_xlabel('Reduction of concentration (removal or dilution) in %')
    left, right = plt.xlim()
    if left < -1000:
        axs[1].set_xlim(-1000, 150)
    if save_as is not None:
        plt.savefig(save_as, dpi=300)
    return fig


def spider_plot(sim_results: list[SimulationResult]):
    rq = [r.final_concentration / r.reference.ref_value_ng_l for r in sim_results]

    output_c_df = pd.DataFrame(np.column_stack(rq),
                               columns=[
                                   f"{r.case_study.name if r.case_study is not None else f'result {i}'} - {r.scenario}"
                                   for i, r in enumerate(sim_results)]
                               )

    percentiles = [0.50, 0.75, 0.95, 0.99]
    stats = output_c_df.describe(percentiles)
    categories = stats.columns
    N = len(categories)
    angles = np.linspace(0, 2 * pi, N, endpoint=False)
    fig = plt.figure(figsize=(8, 9))
    ax = plt.subplot(111, polar=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor('white')
    ax.grid(False)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontweight='bold', fontsize=12)
    ax.xaxis.set_minor_locator(FixedLocator(angles))

    XTICKS = ax.xaxis.get_major_ticks()
    for tick in XTICKS:
        tick.set_pad(15)
    y_max = math.ceil(stats.loc['50%'].max())
    ax.set_ylim(0, y_max)
    y4 = y_max / 2
    y3 = y_max / 4
    y2 = y_max / 8
    ax.set_rlabel_position(0)

    yticks = [0, y2, y3, y4, y_max]
    ax.set_yticks(yticks)
    cat = ['50th', '75th', '95th', '99th', '']
    ax.set_yticklabels(cat[::-1], fontsize=12)

    intervals = [0.1, 1, 5, 10]
    colors = ['#2ED812', '#AEFF00', '#F4EF34', '#FC8604', '#FF4A37']
    bar_width = 2 * np.pi / N

    # 99percentile 
    start_y = yticks[0]
    end_y = yticks[1]
    for j, column in enumerate(stats.columns):
        value_at_99_percentile = stats.loc['99%', column]
        if value_at_99_percentile < intervals[0]:
            ax.bar(angles[j], end_y, width=bar_width, bottom=start_y, facecolor=colors[0], alpha=1,
                   edgecolor='k')
        elif intervals[0] <= value_at_99_percentile < intervals[1]:
            ax.bar(angles[j], end_y, width=bar_width, bottom=start_y, facecolor=colors[1], alpha=1,
                   edgecolor='k')
        elif intervals[1] <= value_at_99_percentile < intervals[2]:
            ax.bar(angles[j], end_y, width=bar_width, bottom=start_y, facecolor=colors[2], alpha=1,
                   edgecolor='k')
        elif intervals[2] <= value_at_99_percentile < intervals[3]:
            ax.bar(angles[j], end_y, width=bar_width, bottom=start_y, facecolor=colors[3], alpha=1,
                   edgecolor='k')
        elif value_at_99_percentile >= intervals[3]:
            ax.bar(angles[j], end_y, width=bar_width, bottom=start_y, facecolor=colors[4], alpha=1,
                   edgecolor='k')

    # 95percentile 
    start_y_95 = yticks[1]
    end_y_95 = yticks[2]
    for j, column in enumerate(stats.columns):
        value_at_95_percentile = stats.loc['95%', column]
        if value_at_95_percentile < intervals[0]:
            ax.bar(angles[j], end_y_95, width=bar_width, bottom=start_y_95, facecolor=colors[0], alpha=1,
                   edgecolor='k')
        elif intervals[0] <= value_at_95_percentile < intervals[1]:
            ax.bar(angles[j], end_y_95, width=bar_width, bottom=start_y_95, facecolor=colors[1], alpha=1,
                   edgecolor='k')
        elif intervals[1] <= value_at_95_percentile < intervals[2]:
            ax.bar(angles[j], end_y_95, width=bar_width, bottom=start_y_95, facecolor=colors[2], alpha=1,
                   edgecolor='k')
        elif intervals[2] <= value_at_95_percentile < intervals[3]:
            ax.bar(angles[j], end_y_95, width=bar_width, bottom=start_y_95, facecolor=colors[3], alpha=1,
                   edgecolor='k')
        elif value_at_95_percentile >= intervals[3]:
            ax.bar(angles[j], end_y_95, width=bar_width, bottom=start_y_95, facecolor=colors[4], alpha=1,
                   edgecolor='k')

            # 75percentile 
    start_y_75 = yticks[2]
    end_y_75 = yticks[3]
    for j, column in enumerate(stats.columns):
        value_at_75_percentile = stats.loc['75%', column]
        if value_at_75_percentile < intervals[0]:
            ax.bar(angles[j], end_y_75, width=bar_width, bottom=start_y_75, facecolor=colors[0], alpha=1,
                   edgecolor='k')
        elif intervals[0] <= value_at_75_percentile < intervals[1]:
            ax.bar(angles[j], end_y_75, width=bar_width, bottom=start_y_75, facecolor=colors[1], alpha=1,
                   edgecolor='k')
        elif intervals[1] <= value_at_75_percentile < intervals[2]:
            ax.bar(angles[j], end_y_75, width=bar_width, bottom=start_y_75, facecolor=colors[2], alpha=1,
                   edgecolor='k')
        elif intervals[2] <= value_at_75_percentile < intervals[3]:
            ax.bar(angles[j], end_y_75, width=bar_width, bottom=start_y_75, facecolor=colors[3], alpha=1,
                   edgecolor='k')
        elif value_at_75_percentile >= intervals[3]:
            ax.bar(angles[j], end_y_75, width=bar_width, bottom=start_y_75, facecolor=colors[4], alpha=1,
                   edgecolor='k')

            # 50percentile 
    start_y_50 = yticks[3]
    end_y_50 = yticks[4]
    for j, column in enumerate(stats.columns):
        value_at_50_percentile = stats.loc['50%', column]
        if value_at_50_percentile < intervals[0]:
            ax.bar(angles[j], end_y_50, width=bar_width, bottom=start_y_50, facecolor=colors[0], alpha=1,
                   edgecolor='k')
        elif intervals[0] <= value_at_50_percentile < intervals[1]:
            ax.bar(angles[j], end_y_50, width=bar_width, bottom=start_y_50, facecolor=colors[1], alpha=1,
                   edgecolor='k')
        elif intervals[1] <= value_at_50_percentile < intervals[2]:
            ax.bar(angles[j], end_y_50, width=bar_width, bottom=start_y_50, facecolor=colors[2], alpha=1,
                   edgecolor='k')
        elif intervals[2] <= value_at_50_percentile < intervals[3]:
            ax.bar(angles[j], end_y_50, width=bar_width, bottom=start_y_50, facecolor=colors[3], alpha=1,
                   edgecolor='k')
        elif value_at_50_percentile >= intervals[3]:
            ax.bar(angles[j], end_y_50, width=bar_width, bottom=start_y_50, facecolor=colors[4], alpha=1,
                   edgecolor='k')

    title = "Reference quotient (RQ) percentiles in final media"
    fig.text(0.5, 0.98, title, fontsize=12, weight="bold", ha="center", va="center")
    legend_elements = [
        mpatches.Patch(color=colors[0], label='<= 0.1'),
        mpatches.Patch(color=colors[1], label='0.1 < and <= 1'),
        mpatches.Patch(color=colors[2], label='1 < and <= 5'),
        mpatches.Patch(color=colors[3], label='5 < and <= 10'),
        mpatches.Patch(color=colors[4], label='> 10'),
    ]

    legend = ax.legend(handles=legend_elements, title='RQ ranges', bbox_to_anchor=(0.05, 1.), loc='upper right',
                       fontsize=12, title_fontsize=12, frameon=False)
    legend.get_title().set_fontweight('bold')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    return fig
