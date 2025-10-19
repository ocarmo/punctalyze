import os
import pandas as pd
import seaborn as sns
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
def plot_stats(data_raw, data_agg, features, title, save_name, x='condition', hue='tag', pairs=None, order=None):
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 15))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
        sns.stripplot(data=data_raw, x=x, y=feature, dodge=True, edgecolor='white',
                      linewidth=1, size=8, alpha=0.4, hue=hue, order=order, ax=ax)
        sns.stripplot(data=data_agg, x=x, y=feature, dodge=True, edgecolor='k',
                      linewidth=1, size=8, hue=hue, order=order, ax=ax)
        sns.boxplot(data=data_agg, x=x, y=feature, palette=['.9'], hue=hue,
                    order=order, ax=ax)

        ax.legend_.remove()
        sns.despine()

        if pairs:
            annotator = Annotator(ax, pairs, data=data_agg, x=x, y=feature, hue=hue, order=order)
            annotator.configure(test='Mann-Whitney', verbose=0)
            annotator.apply_test()
            annotator.annotate()

    for ax in axes[len(features):]:
        ax.axis('off')

    fig.suptitle(title, fontsize=18, y=0.99)
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, bbox_to_anchor=(1.1, 1), title=hue)
    fig.savefig(os.path.join(output_folder, save_name), bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig)

# TODO how to make more dynamic so we can plot without hue?
def plot_no_stats(data_raw, data_agg, features, title, save_name, x='tag', hue='condition', order=None, palette=None):
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15, 15))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
        sns.stripplot(data=data_agg, x=x, y=feature, dodge=True, edgecolor='k',
                      linewidth=1, size=8, hue=hue, palette=palette, hue_order=order, zorder=2, ax=ax)
        sns.stripplot(data=data_raw, x=x, y=feature, dodge=True, edgecolor='white',
                      linewidth=1, size=8, alpha=0.4, hue=hue, palette=palette, hue_order=order, zorder=1, ax=ax)
        sns.boxplot(data=data_agg, x=x, y=feature, palette=['.9'],
                    hue=hue, hue_order=order, zorder=0, ax=ax)

        ax.legend_.remove()
        sns.despine()

    for ax in axes[len(features):]:
        ax.axis('off')

    fig.suptitle(title, fontsize=18, y=0.99)
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, bbox_to_anchor=(1.1, 1), title=hue)
    fig.savefig(os.path.join(output_folder, save_name), bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig)

# TODO find all instances of g3bp/rhm1 and make dynamic
def plot_partition_coefficients(data_raw, data_agg, save_name, x='tag', hue='condition', order=None):
    palette = ['#A6CEE3', '#1F78B4', '#F5CB5C']

    raw = pd.melt(data_raw, id_vars=['image_name', 'tag', 'condition'],
                  value_vars=['g3bp_partition_coeff', 'rhm1_partition_coeff'],
                  var_name='channel', value_name='partition_coeff')

    agg = pd.melt(data_agg, id_vars=['rep', 'tag', 'condition'],
                  value_vars=['g3bp_partition_coeff', 'rhm1_partition_coeff'],
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
        ax.set_xticklabels(['FLAG-RHM1', 'GFP-RHM1'])
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
                'coi2_partition_coeff', 'coi1_partition_coeff',
                'cell_cv', 'cell_skew']

    percell_features = ['cell_size', 'mean_puncta_area', 'puncta_area_proportion', 'puncta_count',
            'puncta_mean_minor_axis', 'puncta_mean_major_axis', 'avg_eccentricity',
            'puncta_cv_mean', 'puncta_skew_mean', 'coi2_partition_coeff', 'coi1_partition_coeff',
            'cell_cv', 'cell_skew', 'cell_coi1_intensity_mean']

    # could use combinations function to generate pairs dynamically, but here we define them explicitly
    # paired_conditions = combinations(conditions, 2)
    conditions = ['PBS', 'NaAsO2', 'HS']

    # TODO generate paired conditions dynamically
    # TODO generate conditions list dynamically
    # TODO generate color platte dynamically
    # TODO make order alphabetical as default
    paired_conditions = [(('PBS', 'GFP'), ('PBS', 'FLAG')),
                         (('NaAsO2', 'GFP'), ('NaAsO2', 'FLAG')),
                         (('HS', 'GFP'), ('HS', 'FLAG'))]
    order = ['PBS', 'NaAsO2', 'HS']
    palette = ['#A6CEE3', '#1F78B4', '#F5CB5C']

    # prepare plotting configuration as [(title, features, raw_df, reps_df), (etc...)]
    plotting_configs = [
        ('per puncta, raw', puncta_features, dfs['puncta_features'], dfs['puncta_features_reps'], 'tag-paired_perpuncta_raw.png'),
        ('per puncta, normalized', puncta_features, dfs['puncta_features_normalized'], dfs['puncta_features_normalized_reps'], 'tag-paired_perpuncta_normalized.png'),
        ('per cell, raw', percell_features, dfs['percell'], dfs['percell_reps'], 'tag-paired_percell_raw.png'),
        ('per cell, normalized', percell_features, dfs['percell_norm'], dfs['percell_norm_reps'], 'tag-paired_percell_normalized.png'),
    ]

    logger.info('Generating paired tag plots with stats...')
    for title, features, raw_df, reps_df, filename in plotting_configs:
        plot_stats(raw_df, reps_df, features, f'Calculated Parameters - {title}', filename,
                   x='condition', hue='tag', pairs=paired_conditions, order=order)

    logger.info('Generating paired condition plots (no stats)...')
    for title, features, raw_df, reps_df, filename in plotting_configs:
        plot_no_stats(raw_df, reps_df, features, f'Calculated Parameters - {title}',
                      filename.replace('tag-paired', 'condition-paired'),
                      x='tag', hue='condition', order=order, palette=palette)

    logger.info('Generating partition coefficient plots...')
    plot_partition_coefficients(dfs['percell'], dfs['percell_reps'], 'condition-paired_percell_raw_partition-only.png', order=order)
