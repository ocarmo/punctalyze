import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
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


# --- Plotting Functions ---
def plot_stats(data_raw, data_agg, features, title, save_name, x='condition', hue='tag', pairs=None, order=None, replicate_col='rep', cell_id_col='image_name'):
    # --- Compute N (replicates) ---
    if replicate_col in data_agg.columns:
        N_per_group = data_agg.groupby([x])[replicate_col].nunique()
    else:
        # fallback: assume each row in aggregated data is a replicate
        N_per_group = data_agg.groupby([x]).size()

    # --- Compute n (cells) ---
    if cell_id_col in data_raw.columns:
        n_per_group = data_raw.groupby([x]).size()

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
            data_agg_lip5 = data_agg[data_agg['condition'] == 'mCherry-LIP5']
            mean_value = data_agg_lip5[feature].mean()
            ax.axhline(y=mean_value, color='gray', linestyle='--', linewidth=2)

        # turn off unused subplots
        for j in range(len(features), len(axes)):
            axes[j].set_visible(False)

        # find visible axes
        visible_axes = [ax for ax in axes if ax.get_visible()]
        # # bottom row = last 3 visible axes (for 3-column layout)
        # bottom_axes = visible_axes[-3:]
        # # hide labels everywhere first
        # for ax in visible_axes:
        #     ax.tick_params(axis='x', labelbottom=False)
       
        # enable only bottom row
        for ax in visible_axes:
            ax.tick_params(axis='x', labelbottom=True)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        fig.text(1.0, 0.9, summary_text)   # add N/n annotation near legend
        fig.suptitle(title, fontsize=18, y=0.99)
        plt.xticks(rotation=45, ha='right') 
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, save_name), bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)

    else:   
        for i, feature in enumerate(features):
            ax = axes[i]
            sns.stripplot(data=data_raw, x=x, y=feature, dodge=True, edgecolor='white',
                        linewidth=1, size=8, alpha=0.4, hue=hue, order=order, ax=ax)
            sns.stripplot(data=data_agg, x=x, y=feature, dodge=True, edgecolor='k',
                        linewidth=1, size=8, hue=hue, order=order, ax=ax)
            sns.boxplot(data=data_agg, x=x, y=feature, palette=['.9'], hue=hue,
                        order=order, ax=ax)
            
            if ax.legend_ is not None:
                ax.legend_.remove()
            sns.despine()

            if pairs:
                annotator = Annotator(ax, pairs, data=data_agg, x=x, y=feature, hue=hue, order=order)
                annotator.configure(test='Mann-Whitney', verbose=0)
                annotator.apply_test()
                annotator.annotate()

        for ax in axes[len(features):]:
            ax.axis('off')
        
        fig.text(1.0, 0.9, summary_text)   # add N/n annotation near legend
        fig.suptitle(title, fontsize=18, y=0.99)
        handles, labels = ax.get_legend_handles_labels()
        fig.tight_layout()
        fig.legend(handles, labels, bbox_to_anchor=(1.1, 1), title=hue)
        fig.savefig(os.path.join(output_folder, save_name), bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)


def plot_partition_coefficients(data_raw, data_agg, save_name, x='tag', hue='condition', order=None):
    palette = ['#A6CEE3', '#1F78B4', '#F5CB5C']

    raw = pd.melt(data_raw, id_vars=['image_name', 'tag', 'condition'],
                  value_vars=['coi1_partition_coeff', 'coi2_partition_coeff'],
                  var_name='channel', value_name='partition_coeff')

    agg = pd.melt(data_agg, id_vars=['rep', 'tag', 'condition'],
                  value_vars=['coi1_partition_coeff', 'coi2_partition_coeff'],
                  var_name='channel', value_name='partition_coeff')

    g = sns.FacetGrid(agg, col='channel', height=4.5, aspect=0.8)
    g.map_dataframe(sns.boxplot, x=x, y='partition_coeff', palette=['.9'], hue=hue, hue_order=order, zorder=0)
    g.map_dataframe(sns.stripplot, x=x, y='partition_coeff', dodge=True, edgecolor='k',
                    linewidth=1, hue=hue, palette=palette, hue_order=order, zorder=2, size=8)

    for ax_i, category in enumerate(g.col_names):
        ax = g.axes.flat[ax_i]
        subset = raw[raw['channel'] == category]
        sns.stripplot(data=subset, x=x, y='partition_coeff', dodge=True,
                      edgecolor='white', linewidth=1, alpha=0.4, hue=hue,
                      palette=palette, hue_order=order, zorder=1, size=8, ax=ax)
        ax.get_legend().remove()
        ax.set_xticklabels(['COI1', 'COI2'])
        ax.set_xlabel('')

    g.set_titles(col_template='{col_name}')
    g.tight_layout()
    g.fig.savefig(os.path.join(output_folder, save_name), bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(g.fig)


if __name__ == '__main__':
    logger.info('Loading data...')
    dfs = load_summary_data(input_folder)

    puncta_features = ['puncta_area', 'puncta_eccentricity', 'puncta_aspect_ratio',
                'puncta_circularity', 'puncta_cv', 'puncta_skew',
                'coi1_partition_coeff', 'cell_std',
                'cell_cv', 'cell_skew']

    percell_features = ['cell_size', 'mean_puncta_area', 'puncta_area_proportion', 'puncta_count',
            'puncta_mean_minor_axis', 'puncta_mean_major_axis', 'puncta_mean_aspect_ratio','avg_eccentricity',
            'puncta_cv_mean', 'puncta_skew_mean', 'coi1_partition_coeff', 'cell_std',
            'cell_cv', 'cell_skew', 'cell_coi1_intensity_mean']

    # # drop rows with 'eGFP' in condition for all dataframes
    # for key in dfs:
    #     dfs[key] = dfs[key][~dfs[key]['condition'].str.contains('eGFP')]

    # # keep rows with 'eGFP' or == 'mCherry-LIP5' in condition for all dataframes
    # for key in dfs:
    #     dfs[key] = dfs[key][dfs[key]['condition'].str.contains('eGFP') | (dfs[key]['condition'] == 'mCherry-LIP5')]

    # could use combinations function to generate pairs dynamically, but here we define them explicitly
    conditions = dfs['puncta_features']['condition'].unique().tolist()
    paired_conditions = combinations(conditions, 2)
    paired_list = list(paired_conditions)
    paired_list = [pair for pair in paired_list if 'mCherry-LIP5' in pair]  # only compare to WT
    order = sorted(conditions)
    # palette = ['#A6CEE3', '#1F78B4', '#F5CB5C']
    palette = sns.color_palette('tab10', n_colors=len(conditions))

    # prepare plotting configuration as [(title, features, raw_df, reps_df), (etc...)]
    plotting_configs = [
        ('per puncta, raw', puncta_features, dfs['puncta_features'], dfs['puncta_features_reps'], 'perpuncta_raw.png'),
        ('per puncta, normalized', puncta_features, dfs['puncta_features_normalized'], dfs['puncta_features_normalized_reps'], 'perpuncta_normalized.png'),
        ('per cell, raw', percell_features, dfs['percell'], dfs['percell_reps'], 'percell_raw.png'),
        ('per cell, normalized', percell_features, dfs['percell_norm'], dfs['percell_norm_reps'], 'percell_normalized.png'),
    ]

    logger.info('Generating paired plots with stats...')
    for title, features, raw_df, reps_df, filename in plotting_configs:
        title
        plot_stats(raw_df, reps_df, features, f'Calculated Parameters - {title}', filename,
                   x='condition', hue=None, pairs=paired_list, order=order)

