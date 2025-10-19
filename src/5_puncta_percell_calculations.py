
import os
import numpy as np
import pandas as pd
from scipy import stats
import functools
import importlib.util
import sys
from loguru import logger

# special import, path to script
puncta_ana_path = 'src/4_puncta_detection.py'

# load the module dynamically due to annoying file name
spec = importlib.util.spec_from_file_location('puncta_detection', puncta_ana_path)
puncta_detection_utils = importlib.util.module_from_spec(spec)
sys.modules['puncta_detection_utils'] = puncta_detection_utils
spec.loader.exec_module(puncta_detection_utils)
aggregate_features_by_group = puncta_detection_utils.aggregate_features_by_group

logger.info('import ok')

# configuration
input_folder = 'results/summary_calculations/'
output_folder = 'results/summary_calculations/'


def calculate_cell_features(df):
    """Calculate summarized features per cell from puncta features."""
    
    group_cols = ['image_name', 'cell_number']

    # Use pandas groupby + agg with named aggregations
    agg_df = df.groupby(group_cols).agg({
        'puncta_minor_axis_length': 'mean',
        'puncta_major_axis_length': 'mean',
        'puncta_area': ['mean', 'sum', 'count'],
        'cell_size': 'mean',
        'puncta_eccentricity': 'mean',
        'puncta_cv': 'mean',
        'puncta_skew': 'mean',
        'coi2_partition_coeff': 'mean',
        'coi1_partition_coeff': 'mean',
        'cell_cv': 'mean',
        'cell_skew': 'mean',
        'cell_coi1_intensity_mean': 'mean',
        'puncta_intensity_mean': 'mean'
    })

    # Flatten MultiIndex columns from aggregation
    agg_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
        for col in agg_df.columns.values]
    agg_df = agg_df.reset_index()

    # Calculate puncta area proportion (%)
    agg_df['puncta_area_proportion'] = (agg_df['puncta_area_sum'] / agg_df['cell_size_mean']) * 100

    # Rename columns for clarity
    agg_df = agg_df.rename(columns={
        'puncta_minor_axis_length_mean': 'puncta_mean_minor_axis',
        'puncta_major_axis_length_mean': 'puncta_mean_major_axis',
        'puncta_area_mean': 'mean_puncta_area',
        'puncta_area_count': 'puncta_count',
        'puncta_eccentricity_mean': 'avg_eccentricity',
        'puncta_cv_mean': 'puncta_cv_mean',
        'puncta_skew_mean': 'puncta_skew_mean',
        'coi2_partition_coeff_mean': 'coi2_partition_coeff',
        'coi1_partition_coeff_mean': 'coi1_partition_coeff',
        'cell_cv_mean': 'cell_cv',
        'cell_skew_mean': 'cell_skew',
        'cell_coi1_intensity_mean_mean': 'cell_coi1_intensity_mean',
        'puncta_intensity_mean_mean': 'puncta_intensity_mean',
        'cell_size_mean': 'cell_size'
    })

    return agg_df


def save_dataframes(df, features, group_cols=['condition', 'tag', 'rep']):
    # Save raw summary per cell
    df.to_csv(f'{output_folder}percell_puncta_features.csv', index=False)

    # Average by biological replicate using the aggregate_features_by_group function
    rep_df = aggregate_features_by_group(df, group_cols, features)
    rep_df.to_csv(f'{output_folder}percell_puncta_features_reps.csv', index=False)

    # Normalize to cell_coi1_intensity_mean
    df_norm = df.copy()
    for col in features:
        df_norm[col] = df_norm[col] / df_norm['cell_coi1_intensity_mean']

    df_norm.to_csv(f'{output_folder}percell_puncta_features_normalized.csv', index=False)

    # Average normalized data by biological replicate
    rep_norm_df = aggregate_features_by_group(df_norm, group_cols, features)
    rep_norm_df.to_csv(f'{output_folder}percell_puncta_features_normalized_reps.csv', index=False)


if __name__ == '__main__':
    # Load feature information
    feature_information = pd.read_csv(f'{input_folder}puncta_features.csv')

    # Calculate summarized features per cell
    summary = calculate_cell_features(feature_information)

    # Add metadata columns
    summary['tag'] = summary['image_name'].str.split('-').str[0].str.split('_').str[-1]
    summary['condition'] = summary['image_name'].str.split('_').str[2].str.split('-').str[0]
    summary['rep'] = summary['image_name'].str.split('_').str[-1].str.split('-').str[0]

    # Define features of interest (excluding metadata columns)
    cols = summary.columns.tolist()
    cols = [item for item in cols if '_coords' not in item]
    cols = ['cell_size', 'mean_puncta_area', 'puncta_area_proportion', 'puncta_count',
        'puncta_mean_minor_axis', 'puncta_mean_major_axis', 'avg_eccentricity',
        'puncta_cv_mean', 'puncta_skew_mean', 'coi2_partition_coeff', 'coi1_partition_coeff',
        'cell_cv', 'cell_skew', 'cell_coi1_intensity_mean', 'puncta_intensity_mean']

    # remove outliers based on z-score
    summary = summary[(np.abs(stats.zscore(summary[cols[:-1]])) < 3).all(axis=1)]

    # Save dataframes (raw, averaged, normalized, normalized averaged)
    save_dataframes(summary, cols)

    logger.info('saved puncta feature averaged-per-cell dataframes')