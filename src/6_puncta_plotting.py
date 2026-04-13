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
    fig, axes = plt.subplots(nrows=int(round((len(features)/3), 1))+1, ncols=3, figsize=(20, 30))
    axes = axes.flatten()

    if hue is None:
        for i, feature in enumerate(features):
            ax = axes[i]
            sns.stripplot(data=data_raw, x=x, y=feature, dodge=True, edgecolor='white',
                        linewidth=1, size=8, alpha=0.1, order=order, ax=ax, zorder=1)
            sns.violinplot(data=data_agg, x=x, y=feature, order=order, color='gray', ax=ax, zorder=0)
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
            sns.stripplot(data=data_raw, x=x, y=feature, dodge=True, edgecolor='white',
                        linewidth=1, size=8, alpha=0.4, hue=hue, hue_order=hue_order, order=order, ax=ax)
            # store legend handles and labels for later plotting
            handles, labels = ax.get_legend_handles_labels()

            sns.violinplot(data=data_raw, x=x, y=feature, palette=['.9'],   hue=hue, hue_order=hue_order, order=order, ax=ax)
            sns.stripplot(data=data_agg, x=x, y=feature, dodge=True, edgecolor='k',
                        linewidth=1, size=8, hue=hue, hue_order=hue_order, order=order, ax=ax)

            ax.legend_.remove()
            sns.despine()

            if pairs:
                annotator = Annotator(ax, pairs, data=data_agg, x=x, y=feature, hue=hue, hue_order=hue_order, order=order)
                annotator.configure(test='Mann-Whitney', verbose=0)
                annotator.apply_test()
                annotator.annotate()

            # label_map = {
            #     'y47': 'Pab1',
            #     'y48': 'Gbp2',
            #     'y49': 'Hsp104',
            #     'y50': 'Pbp1',
            #     'y51': 'Yef3',
            #     'y52': 'Gapdh'
            # }
            # ax.set_xticklabels([label_map.get(t.get_text(), t.get_text()) 
            #             for t in ax.get_xticklabels()])

        for ax in axes[len(features):]:
            ax.axis('off')

        fig.suptitle(title, fontsize=18, y=0.99)
        fig.tight_layout()
        fig.legend(handles, labels, bbox_to_anchor=(1.1, 1), title=hue)
        fig.text(1.0, 0.9, summary_text)   # add N/n annotation near legend
        fig.savefig(os.path.join(output_folder, save_name), bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    logger.info('Loading data...')
    dfs = load_summary_data(input_folder)

    puncta_features = ['puncta_area', 'puncta_eccentricity', 'puncta_aspect_ratio',
                'puncta_circularity', 'puncta_cv', 'puncta_skew']

    percell_features = ['cell_size', 'mean_puncta_area', 'puncta_area_proportion', 'puncta_count',
            'puncta_mean_aspect_ratio','avg_eccentricity',
            'puncta_cv_mean', 'coi2_partition_coeff', 'coi1_partition_coeff', 'cell_std',
            'cell_cv', 'cell_coi1_intensity_mean']

    # could use combinations function to generate pairs dynamically, but here we define them explicitly
    conditions = dfs['puncta_features']['condition'].unique().tolist()
    paired_conditions = combinations(conditions, 2)
    paired_list = list(paired_conditions)
    paired_list = [pair for pair in paired_list if 'WT' in pair]  # only compare to WT
    order = sorted(conditions)
    # palette = ['#A6CEE3', '#1F78B4', '#F5CB5C']
    palette = sns.color_palette('tab10', n_colors=len(conditions))

    # per dataframe, remove rows if '23-19' or '23-22' in image name (these had very poor binding)
    for key in dfs.keys():
        dfs[key] = dfs[key][~dfs[key]['probes'].str.contains('23-19|23-22')]

    # prepare plotting configuration as [(title, features, raw_df, reps_df), (etc...)]
    plotting_configs = [
        ('per puncta, raw', puncta_features, dfs['puncta_features'], dfs['puncta_features_reps'], 'perpuncta_raw.png'),
        ('per puncta, normalized', puncta_features, dfs['puncta_features_normalized'], dfs['puncta_features_normalized_reps'], 'perpuncta_normalized.png'),
        ('per cell, raw', percell_features, dfs['percell'], dfs['percell_reps'], 'percell_raw.png'),
        ('per cell, normalized', percell_features, dfs['percell_norm'], dfs['percell_norm_reps'], 'percell_normalized.png'),
    ]

    # TODO make plotting more dynamic to handle stats/no-stats cases
    logger.info('Generating paired plots with stats...')
    for title, features, raw_df, reps_df, filename in plotting_configs:
        title
        for probe in raw_df['probes'].unique():
            raw_subset = raw_df[raw_df['probes'] == probe]
            reps_subset = reps_df[reps_df['probes'] == probe]
            plot_stats(raw_subset, reps_subset, features, f'{title} - {probe}', f'{filename.split(".")[0]}_{probe}.png',
                       x='condition', hue='strain', hue_order=['wt-brG', 'dbr1-brG'], pairs=None, style='probes', order=order)
        # all probes
        plot_stats(raw_df, reps_df, features, f'Calculated Parameters - {title}', filename,
                   x='condition', hue='strain', pairs=None, order=order)

