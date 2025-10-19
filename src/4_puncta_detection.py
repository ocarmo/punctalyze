"""
Detect and analyze features of puncta per cell
"""

import os
import importlib.util
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import skimage.io
from skimage import measure, segmentation, morphology
from skimage.morphology import remove_small_objects
from scipy import stats
from scipy.stats import skewtest
from loguru import logger
import functools
# special import, path to script
napari_utils_path = 'src/3_napari.py' # adjust as needed

# load the module dynamically due to annoying file name
spec = importlib.util.spec_from_file_location("napari", napari_utils_path)
napari_utils = importlib.util.module_from_spec(spec)
sys.modules["napari_utils"] = napari_utils
spec.loader.exec_module(napari_utils)
remove_saturated_cells = napari_utils.remove_saturated_cells

logger.info('import ok')

# plotting setup
plt.rcParams.update({'font.size': 14})
sns.set_palette('Paired')

# --- configuration ---
STD_THRESHOLD = 3.8
SAT_FRAC_CUTOFF = 0.01  # for consistency with remove_saturated_cells
COI_1 = 1  # channel of interest for saturation check (e.g., 1 for channel 2)
COI_2 = 0  # secondary channel of interest for comparisons
COI_1_name = 'coi1'  # name of the first channel of interest, for plotting
COI_2_name  = 'coi2'  # name of the second channel of interest, flor plotting
MIN_PUNCTA_SIZE = 16  # minimum size of puncta
SCALE_PX = 0.0779907  # size of one pixel in units specified by the next constant
SCALE_UNIT = 'um'  # units for the scale bar
image_folder = 'results/initial_cleanup/'
mask_folder = 'results/napari_masking/'
output_folder = 'results/summary_calculations/'
proofs_folder = 'results/proofs/'

for folder in [output_folder, proofs_folder]:
    if not os.path.exists(folder):
        os.mkdir(folder)


def feature_extractor(mask, properties=None):
    if properties is None:
        properties = [
            'area', 'eccentricity', 'label',
            'major_axis_length', 'minor_axis_length',
            'perimeter', 'coords'
        ]
    props = measure.regionprops_table(mask, properties=properties)
    return pd.DataFrame(props)


def load_images(image_folder):
    images = {}
    for fn in os.listdir(image_folder):
        if fn.endswith('.npy'):
            name = fn.removesuffix('.npy')
            images[name] = np.load(f'{image_folder}/{fn}')
    return images


def load_masks(mask_folder):
    masks = {}
    for fn in os.listdir(mask_folder):
        if fn.endswith('_mask.npy'):
            name = fn.removesuffix('_mask.npy')
            masks[name] = np.load(f'{mask_folder}/{fn}', allow_pickle=True)
    return masks


def generate_cytoplasm_masks(masks):
    logger.info('removing nuclei from cell masks...')
    cyto_masks = {}
    for name, img in masks.items():
        cell_mask, nuc_mask = img[0], img[1]
        cell_bin = (cell_mask > 0).astype(int) # make binary masks
        nuc_bin = (nuc_mask > 0).astype(int)

        single_cyto = []
        labels = np.unique(cell_mask)
        if labels.size > 1:
            for lbl in labels[labels != 0]:
                cyto = np.where(cell_mask == lbl, cell_bin, 0)
                cyto_minus_nuc = cyto & ~nuc_bin
                if np.any(cyto_minus_nuc):
                    single_cyto.append(np.where(cyto_minus_nuc, lbl, 0))
                else:
                    single_cyto.append(np.zeros_like(cell_mask, dtype=int))
        else:
            single_cyto.append(np.zeros_like(cell_mask, dtype=int))

        cyto_masks[name] = sum(single_cyto)
    logger.info('cytoplasm masks created.')
    return cyto_masks


def filter_saturated_images(images, cytoplasm_masks, masks):
    logger.info('filtering saturated cells...')
    filtered = {}
    for name, img in images.items():
        # Build a stack: [stain, coi, cytoplasm mask]
        stack = np.stack([
            img[COI_2], img[COI_1], cytoplasm_masks[name]
        ])
        # apply  imported saturation check function
        cells = remove_saturated_cells(
            image_stack=stack,
            mask_stack=masks[name],
            COI=COI_1
        )
        filtered[name] = np.stack([stack[COI_2], stack[COI_1], cells])
    logger.info('saturated cells filtered.')
    return filtered


def collect_features(image_dict, STD_THRESHOLD=STD_THRESHOLD):
    logger.info('collecting cell & puncta features...')
    results = []
    for name, img in image_dict.items():
        coi2, coi1, mask = img
        unique_cells = np.unique(mask)[1:]
        contours = measure.find_contours((mask > 0).astype(int), 0.8)
        contour = [c for c in contours if len(c) >= 100]

        for lbl in unique_cells:
            cell_mask = mask == lbl
            coi1_vals = coi1[cell_mask]
            mean_coi1 = coi1_vals.mean()
            std_coi1 = coi1_vals.std()

            threshold = std_coi1 * STD_THRESHOLD
            binary = (coi1 > threshold) & cell_mask
            puncta_labels = morphology.label(binary)
            puncta_labels = remove_small_objects(puncta_labels, min_size=MIN_PUNCTA_SIZE)

            df_p = feature_extractor(puncta_labels).add_prefix('puncta_')
            if df_p.empty:
                df_p.loc[0] = 0

            stats_list = []
            for i, row in df_p.iterrows():
                p_mask = puncta_labels == row['puncta_label']
                puncta_vals = coi1[p_mask]
                cv = puncta_vals.std() / puncta_vals.mean()
                skew_stat = skewtest(puncta_vals).statistic
                mean_p = puncta_vals.mean()
                mean_coi2 = coi2[p_mask].mean()
                stats_list.append((cv, skew_stat, mean_p, mean_coi2))

            df_stats = pd.DataFrame(stats_list,
                                    columns=['puncta_cv', 'puncta_skew',
                                             'puncta_intensity_mean',
                                             'puncta_intensity_mean_in_coi2'])
            df = pd.concat([df_p.reset_index(drop=True), df_stats], axis=1)
            df['image_name'], df['cell_number'] = name, lbl
            df['cell_size'] = cell_mask.sum()
            df['cell_cv'] = std_coi1 / mean_coi1  # coefficient of variation
            df['cell_skew'] = skewtest(coi1_vals).statistic
            df['cell_coi1_intensity_mean'] = mean_coi1
            df['cell_coi2_intensity_mean'] = (coi2[cell_mask]).mean()
            df['cell_coords'] = [contour] * len(df)

            results.append(df)

    logger.info('feature extraction done.')
    return pd.concat(results, ignore_index=True)


def extra_puncta_features(df):
    df = df.copy()  # avoid modifying in place
    df['puncta_aspect_ratio'] = df['puncta_minor_axis_length'] / df['puncta_major_axis_length']
    df['puncta_circularity'] = 12.566 * df['puncta_area'] / (df['puncta_perimeter'] ** 2)
    df['coi2_partition_coeff'] = df['puncta_intensity_mean_in_coi2'] / df['cell_coi2_intensity_mean']
    df['coi1_partition_coeff'] = df['puncta_intensity_mean'] / df['cell_coi1_intensity_mean']

    return df


def aggregate_features_by_group(df, group_cols, agg_cols, agg_func='mean'):
    """
    Aggregate multiple columns by group and merge results into a single DataFrame.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        group_cols (list): Columns to group by.
        agg_cols (list): Columns to aggregate.
        agg_func (str or callable): Aggregation function, default is 'mean'.

    Returns:
        pd.DataFrame: Aggregated dataframe with group_cols and agg_cols.
    """
    grouped_dfs = []
    for col in agg_cols:
        agg_df = df.groupby(group_cols)[col].agg(agg_func).reset_index()
        grouped_dfs.append(agg_df)

    merged_df = functools.reduce(
        lambda left, right: left.merge(right, on=group_cols),
        grouped_dfs
    )
    return merged_df.reset_index(drop=True)


# --- Proof Plotting ---
def generate_proofs(df, image_dict, coi1=COI_1_name, coi2=COI_2_name):
    logger.info('Generating proof plots...')
    for name, img in image_dict.items():
        contour = df.loc[df['image_name']==name, 'cell_coords']
        coord_list = df.loc[df['image_name']==name, 'puncta_coords']

        if contour.empty:
            continue

        coi2, coi1, mask = img
        cell_img = coi1 * (mask > 0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,6))
        ax1.imshow(coi1, cmap='gray_r')
        ax1.imshow(coi2, cmap='Blues', alpha=0.6)

        ax2.imshow(cell_img, cmap='gray_r')
        for line in contour.iloc[0]:
            ax2.plot(line[:,1], line[:,0], c='k', lw=0.5)

        if len(coord_list) > 1:
            for puncta in coord_list:
                if isinstance(puncta, np.ndarray):
                    ax2.plot(puncta[:,1], puncta[:,0], lw=0.5)

        scalebar = ScaleBar(SCALE_PX, SCALE_UNIT, location='lower right',
                            pad=0.3, sep=2, box_alpha=0, color='gray',
                            length_fraction=0.3)
        ax1.add_artist(scalebar)
        ax1.text(50, 2000, COI_1_name, color='gray')
        ax1.text(50, 1800, COI_2_name, color='steelblue')
        fig.suptitle(name, y=0.88)
        fig.tight_layout()
        fig.savefig(f'{proofs_folder}{name}_proof.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    logger.info('proofs saved.')


if __name__ == '__main__':
    logger.info('loading images and masks...')
    images = load_images(image_folder)
    masks = load_masks(mask_folder)

    cyto_masks = generate_cytoplasm_masks(masks)
    filtered = filter_saturated_images(images, cyto_masks, masks)
    features = collect_features(filtered)
    features = extra_puncta_features(features)

    # --- data wrangling and saving ---
    logger.info('starting data wrangling and saving...')
    features['tag'] = features['image_name'].str.split('-').str[0].str.split('_').str[-1]
    features['condition'] = features['image_name'].str.split('_').str[2].str.split('-').str[0]
    features['rep'] = features['image_name'].str.split('_').str[-1].str.split('-').str[0]

    cols = features.columns.tolist()
    cols = [item for item in cols if '_coords' not in item]
    cols = ['puncta_area', 'puncta_eccentricity', 'puncta_aspect_ratio',
            'puncta_circularity', 'puncta_cv', 'puncta_skew',
            'coi2_partition_coeff', 'coi1_partition_coeff',
            'cell_cv', 'cell_skew']
    
    # # remove outliers based on z-score
    # features = features[(np.abs(stats.zscore(features[cols[:-1]])) < 3).all(axis=1)]

    # save the main features dataframe
    features.to_csv(f'{output_folder}puncta_features.csv', index=False)

    # save averages per biological replicate
    rep_df = aggregate_features_by_group(features, ['condition', 'tag', 'rep'], cols)
    rep_df.to_csv(f'{output_folder}puncta_features_reps.csv', index=False)

    # save features normalized to cell intensity of channel of interest
    df_norm = features.copy()
    for col in cols:
        df_norm[col] /= df_norm['cell_coi1_intensity_mean']
    df_norm.to_csv(f'{output_folder}puncta_features_normalized.csv', index=False)

    # save normalized averages per biological replicate
    rep_norm_df = aggregate_features_by_group(df_norm, ['condition', 'tag', 'rep'], cols)
    rep_norm_df.to_csv(f'{output_folder}puncta_features_normalized_reps.csv', index=False)

    logger.info('data wrangling and saving complete.')

    # --- generate proofs ---
    generate_proofs(features, filtered, coi1=COI_1, coi2=COI_2)

    logger.info('pipeline complete.')
