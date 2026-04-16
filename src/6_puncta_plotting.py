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
def plot_stats(data_raw, data_agg, features, title, save_name, x='condition', hue='tag', pairs=None, order=None, replicate_col='rep', cell_id_col=None):
    # --- Compute N (replicates) ---
    if replicate_col in data_agg.columns:
        N_per_group = data_agg.groupby([x, hue])[replicate_col].nunique()
    else:
        # fallback: assume each row in aggregated data is a replicate
        N_per_group = data_agg.groupby([x, hue]).size()

    # --- Compute n (cells) ---
    if cell_id_col and cell_id_col in data_raw.columns:
        n_per_group = data_raw.groupby([x, hue])[cell_id_col].nunique()
    else:
        # fallback: assume each row is a cell
        n_per_group = data_raw.groupby([x, hue]).size()

    # --- Format text ---
    summary_lines = []
    for cond in N_per_group.index:
        N_val = N_per_group.get(cond, np.nan)
        n_val = n_per_group.get(cond, np.nan)
        summary_lines.append(f"{cond}: N={N_val}, n={n_val}")

    N_min, N_max = N_per_group.min(), N_per_group.max()
    n_min, n_max = n_per_group.min(), n_per_group.max()

    summary_text = f"N = {N_min}–{N_max} replicates\nn = {n_min}–{n_max} cells"

    # plot
    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(20, 30))
    axes = axes.flatten()

    if hue is None:
        for i, feature in enumerate(features):
            ax = axes[i]
            sns.stripplot(data=data_raw, x=x, y=feature, dodge=True, edgecolor='white',
                        linewidth=1, size=8, alpha=0.1, order=order, ax=ax, zorder=0)
            sns.violinplot(data=data_agg, x=x, y=feature, order=order, color='gray', ax=ax, zorder=1)
            sns.stripplot(data=data_agg, x=x, y=feature, dodge=True, edgecolor='k',
                        linewidth=1, size=8, order=order, ax=ax, zorder=2)
            sns.despine()

            if pairs:
                annotator = Annotator(ax, pairs, data=data_agg, x=x, y=feature, order=order)
                annotator.configure(test='t-test_ind', verbose=0)
                annotator.apply_test()
                annotator.annotate()
        
        for ax in axes[len(features):]:
            ax.axis('off')

        fig.suptitle(title, fontsize=18, y=0.99)
        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, save_name), bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)


    else:   
        for i, feature in enumerate(features):
            ax = axes[i]
            sns.stripplot(data=data_agg, x=x, y=feature, dodge=True, edgecolor='k',
                        linewidth=1, size=8, hue=hue, order=order, ax=ax, zorder=2)
            
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            
            sns.stripplot(data=data_raw, x=x, y=feature, dodge=True, edgecolor='white',
                        linewidth=1, size=8, alpha=0.05, hue=hue, order=order, ax=ax, zorder=1)
            sns.violinplot(data=data_agg, x=x, y=feature, order=order, palette='gray', hue=hue, ax=ax, zorder=0)

            if ax.legend_:
                ax.legend_.remove()
            sns.despine()

            if pairs:
                annotator = Annotator(ax, pairs, data=data_agg, x=x, y=feature, hue=hue, order=order)
                annotator.configure(test='t-test_ind', verbose=0)
                annotator.apply_test()
                annotator.annotate()

            label_map = {
                'y47': 'Pab1',
                'y48': 'Gbp2',
                'y49': 'Hsp104',
                'y50': 'Pbp1',
                'y51': 'Yef3',
                'y52': 'Gapdh'
            }
            ax.set_xticklabels([label_map.get(t.get_text(), t.get_text()) 
                        for t in ax.get_xticklabels()])

        for ax in axes[len(features):]:
            ax.axis('off')

        fig.suptitle(title, fontsize=18, y=0.99)
        fig.tight_layout()
        fig.legend(legend_handles, legend_labels, bbox_to_anchor=(1.1, 1), title='construct')      
        fig.text(1.0, 0.9, summary_text)   # add N/n annotation near legend
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
                'cell_std', 'cell_cv', 'cell_skew']

    percell_features = ['cell_size', 'mean_puncta_area', 'puncta_area_proportion', 'puncta_count',
            'puncta_mean_minor_axis', 'puncta_mean_major_axis', 
            'cell_std', 'cell_cv', 'cell_skew']
    
    # for each dataframe, remove any rows with 'recovered' in the image name
    for key in dfs:
        key
        dfs[key] = dfs[key][~dfs[key]['condition'].str.contains('recover')]

    # could use combinations function to generate pairs dynamically, but here we define them explicitly
    conditions = dfs['puncta_features']['strain'].unique().tolist()
    paired_conditions = combinations(conditions, 2)
    paired_list = list(paired_conditions)
    paired_list = pairs=[
    (('y47', 'brA'), ('y47', 'brG')),
    (('y48', 'brA'), ('y48', 'brG')),
    (('y49', 'brA'), ('y49', 'brG')),
    (('y50', 'brA'), ('y50', 'brG')),
    (('y51', 'brA'), ('y51', 'brG')),
    (('y52', 'brA'), ('y52', 'brG'))
    ]
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

    # plot only per cell data first
    title, features, raw_df, reps_df, filename = plotting_configs[-2]
    plot_stats(raw_df, reps_df, features, f'Calculated Parameters - {title}', filename, x='strain', hue='construct', pairs=paired_list, order=order)

    # pair plots for visualizing feature correlations, per strain
    for strain in raw_df['strain'].unique():
        test_df = raw_df[['strain', 'construct', 'cell_size', 'mean_puncta_area', 'puncta_area_proportion', 'puncta_count',
                'puncta_mean_minor_axis', 'puncta_mean_major_axis', 'cell_coi1_intensity_mean', 'puncta_intensity_mean',
                'cell_std', 'cell_cv', 'cell_skew']]
        sns.pairplot(test_df[test_df['strain'] == strain], hue='construct', kind='kde', corner=True, palette='husl')
        # save
        plt.savefig(f'{output_folder}percell_raw_pairplot_{strain}.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

    # TODO make plotting more dynamic to handle stats/no-stats cases
    logger.info('Generating paired plots with stats...')
    for title, features, raw_df, reps_df, filename in plotting_configs:
        plot_stats(raw_df, reps_df, features, f'Calculated Parameters - {title}', filename,
                   x='strain', hue='construct', pairs=paired_list, order=order)

    # TODO fix partition coefficient plots
    logger.info('Generating partition coefficient plots...')
    plot_partition_coefficients(dfs['percell'], dfs['percell_reps'], 'condition-paired_percell_raw_partition-only.png', order=order)
