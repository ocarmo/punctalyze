import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import mannwhitneyu, ttest_ind
from statannotations.Annotator import Annotator
from loguru import logger

logger.info('import ok')

# plotting config
plt.rcParams.update({'font.size': 14})
sns.set_palette('Paired')

# configuration
input_folder = 'results/summary_calculations/'
output_folder = 'results/plotting/'

os.makedirs(output_folder, exist_ok=True)


def load_summary_data(input_folder):
    return {
        'puncta_features': pd.read_csv(f'{input_folder}puncta_features.csv'),
        'puncta_features_reps': pd.read_csv(f'{input_folder}puncta_features_reps.csv'),
        'puncta_features_normalized': pd.read_csv(f'{input_folder}puncta_features_normalized.csv'),
        'puncta_features_normalized_reps': pd.read_csv(f'{input_folder}puncta_features_normalized_reps.csv'),
        'percell': pd.read_csv(f'{input_folder}percell_puncta_features.csv'),
        'percell_reps': pd.read_csv(f'{input_folder}percell_puncta_features_reps.csv'),
        'percell_norm': pd.read_csv(f'{input_folder}percell_puncta_features_normalized.csv'),
        'percell_norm_reps': pd.read_csv(f'{input_folder}percell_puncta_features_normalized_reps.csv')
    }


def get_significant_pairs(data, pairs, x, y, hue=None, test='Mann-Whitney', alpha=0.05):
    sig_pairs = []

    for pair in pairs:
        group1, group2 = pair

        if hue:
            # pair = ((cond1, hue1), (cond2, hue2))
            (x1, h1), (x2, h2) = group1, group2
            d1 = data[(data[x] == x1) & (data[hue] == h1)][y]
            d2 = data[(data[x] == x2) & (data[hue] == h2)][y]
        else:
            # pair = (cond1, cond2)
            d1 = data[data[x] == group1][y]
            d2 = data[data[x] == group2][y]

        # run test
        if test == 'Mann-Whitney':
            stat, p = mannwhitneyu(d1, d2, alternative='two-sided')
        else:
            stat, p = ttest_ind(d1, d2, equal_var=False)

        if p < alpha:
            sig_pairs.append(pair)

    return sig_pairs


# --- Plotting Functions ---
def plot_stats(data_raw, data_agg, features, title, save_name, x='condition', hue='tag', pairs=None, order=None, replicate_col='rep', cell_id_col='image_name'):
    group_keys = [x] if hue is None else [x, hue]

    # --- Compute N (replicates) ---
    if replicate_col in data_agg.columns:
        N_per_group = data_agg.groupby(group_keys)[replicate_col].nunique()
    else:
        N_per_group = data_agg.groupby(group_keys).size()

    # --- Compute n (cells) ---
    if cell_id_col in data_raw.columns:
        n_per_group = data_raw.groupby(group_keys)[cell_id_col].nunique()
    else:
        n_per_group = data_raw.groupby(group_keys).size()

    # --- Format text ---
    summary_lines = []
    for cond in N_per_group.index:
        N_val = N_per_group.get(cond, np.nan)
        n_val = n_per_group.get(cond, np.nan)
        summary_lines.append(f"{cond}: N={N_val}, n={n_val}")

    N_min, N_max = N_per_group.min(), N_per_group.max()
    n_min, n_max = n_per_group.min(), n_per_group.max()

    summary_text = f"N = {N_min}–{N_max} replicates\nn = {n_min}–{n_max} cells"

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(30, 30))
    axes = axes.flatten()

    if hue is None:
        for i, feature in enumerate(features):
            ax = axes[i]
            sns.stripplot(data=data_raw, x=x, y=feature, dodge=True, edgecolor='white',
                        linewidth=1, size=8, alpha=0.5, order=order, ax=ax, zorder=0)
            sns.boxplot(data=data_agg, x=x, y=feature, order=order, fill=False, ax=ax, zorder=1)
            sns.stripplot(data=data_agg, x=x, y=feature, dodge=True,
                        linewidth=1, size=8, order=order, color='k', ax=ax, zorder=2)
            sns.despine()
            fig.autofmt_xdate(rotation=45)

            if pairs:
                annotator = Annotator(ax, pairs, data=data_agg, x=x, y=feature, order=order)
                annotator.configure(test='t-test_ind', verbose=0)
                annotator.apply_test()
                annotator.annotate()

            # make horizontal axis line at mean of mCherry-LIP5 construct
            data_agg_lip5 = data_agg[(data_agg['condition'] == 'eGFP-co-mCherry') & (data_agg['mcherry_bait'] == 'LIP5')]
            mean_value = data_agg_lip5[feature].mean()
            ax.axhline(y=mean_value, color='gray', linestyle='--', linewidth=2)

        # turn off unused subplots
        for j in range(len(features), len(axes)):
            axes[j].set_visible(False)

        # find visible axes
        visible_axes = [ax for ax in axes if ax.get_visible()]
       
        # enable only bottom row
        for ax in visible_axes:
            ax.tick_params(axis='x', labelbottom=True)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        fig.text(1.0, 0.9, summary_text)   # add N/n annotation near legend
        fig.suptitle(title, fontsize=18, y=0.99)
        plt.xticks(rotation=45, ha='right') 
        fig.tight_layout()
        plt.show()
        fig.savefig(os.path.join(output_folder, save_name), bbox_inches='tight', pad_inches=0.1, dpi=600)
        plt.close(fig)

    else:   
        for i, feature in enumerate(features):
            ax = axes[i]
            sns.stripplot(data=data_agg, x=x, y=feature, dodge=True, edgecolor='k',
                        linewidth=1, size=8, hue=hue, order=order, ax=ax, zorder=2)
            # store legend details
            handles, labels = ax.get_legend_handles_labels()

            sns.stripplot(data=data_raw, x=x, y=feature, dodge=True, edgecolor='white',
                        linewidth=1, size=8, alpha=0.1, hue=hue, order=order, ax=ax, zorder=1)
            sns.boxplot(data=data_agg, x=x, y=feature, palette=['.9'], hue=hue,
                        order=order, ax=ax, zorder=0)
            
            if ax.legend_ is not None:
                ax.legend_.remove()
            sns.despine()

            if pairs:
                annotator = Annotator(ax, pairs, data=data_agg, x=x, y=feature, hue=hue, order=order)
                annotator.configure(test='Mann-Whitney', verbose=0)
                annotator.apply_test()
                annotator.annotate()

            # make horizontal axis line at mean of mCherry-LIP5 construct
            data_agg_lip5 = data_agg[(data_agg['condition'] == 'eGFP-co-mCherry') & (data_agg['mcherry_bait'] == 'LIP5')]
            mean_value = data_agg_lip5[feature].mean()
            ax.axhline(y=mean_value, color='gray', linestyle='--', linewidth=2)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        for ax in axes[len(features):]:
            ax.axis('off')
        
        fig.text(1.0, 0.9, summary_text)   # add N/n annotation near legend
        fig.suptitle(title, fontsize=18, y=0.99)
        fig.tight_layout()
        fig.legend(handles, labels, bbox_to_anchor=(1.1, 1), title=hue)
        fig.savefig(os.path.join(output_folder, save_name), bbox_inches='tight', pad_inches=0.1, dpi=600)
        plt.close(fig)


if __name__ == '__main__':
    logger.info('Loading data...')
    dfs = load_summary_data(input_folder)

    puncta_features = ['puncta_area', 'puncta_eccentricity', 'puncta_aspect_ratio',
                'puncta_circularity', 'puncta_cv', 'puncta_skew',
                'coi1_partition_coeff', 'cell_std',
                'cell_cv', 'cell_skew']

    percell_features = ['cell_size', 'mean_puncta_area', 'puncta_area_proportion', 'puncta_count',
            'puncta_mean_minor_axis', 'puncta_mean_major_axis', 'puncta_mean_aspect_ratio','avg_eccentricity',
            'coi2_puncta_area_proportion',
            'puncta_cv_mean', 'puncta_skew_mean', 'coi1_partition_coeff', 'coi2_partition_coeff', 
            # 'cell_std',
            'cell_cv', 'cell_skew', 'cell_coi1_intensity_mean']

    # could use combinations function to generate pairs dynamically, but here we define them explicitly
    conditions = dfs['puncta_features']['condition'].unique().tolist()
    bait = dfs['puncta_features']['mcherry_bait'].unique().tolist()
    # make tuples of (condition, bait) for pairs of baits to compare
    paired_list = [((cond, 'LIP5'), (cond, 'LIP5(Y-S)16')) for cond in conditions]  # only compare to WT
    # paired_conditions = combinations(conditions, 2)
    # paired_list = list(paired_conditions)
    # paired_list = [pair for pair in paired_list if 'mCherry-LIP5' in pair]  # only compare to WT
    order = sorted(conditions)
    palette = sns.color_palette('tab10', n_colors=len(conditions))

    # prepare plotting configuration as [(title, features, raw_df, reps_df), (etc...)]
    plotting_configs = [
        ('per puncta, raw', puncta_features, dfs['puncta_features'], dfs['puncta_features_reps'], 'perpuncta_raw.svg'),
        ('per puncta, normalized', puncta_features, dfs['puncta_features_normalized'], dfs['puncta_features_normalized_reps'], 'perpuncta_normalized.svg'),
        ('per cell, raw', percell_features, dfs['percell'], dfs['percell_reps'], 'percell_raw.svg'),
        ('per cell, normalized', percell_features, dfs['percell_norm'], dfs['percell_norm_reps'], 'percell_normalized.svg'),
    ]

