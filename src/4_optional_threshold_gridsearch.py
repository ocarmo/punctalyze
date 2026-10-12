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
from scipy.ndimage import gaussian_filter
from loguru import logger
from skimage.filters import threshold_local
from skimage.morphology import white_tophat, disk
import functools
# special import, path to script
napari_utils_path = 'punctalyze/src/3_napari.py' # adjust as needed

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
STD_THRESHOLD = 3  # number of standard deviations above mean for puncta thresholding
SIGMA = 1  # for Gaussian blur
MAD_MULTIPLIER = 3  # multiplier for the MAD-based threshold
THRESHOLD_TYPE = 'gaussian'  # 'gaussian' or 'mad'
SAT_FRAC_CUTOFF = 0.01  # for consistency with remove_saturated_cells
COI_1 = 1  # channel of interest for saturation check (e.g., 1 for channel 2)
COI_2 = 0  # secondary channel of interest for comparisons
COI_1_name = 'mCherry-LIP5'  # name of the first channel of interest, for plotting
COI_2_name  = 'eGFP'  # name of the second channel of interest, for plotting
MIN_PUNCTA_SIZE = 3**2  # minimum size of puncta
SCALE_PX = (294.67/2720) # size of one pixel in units specified by the next constant
SCALE_UNIT = 'um'  # units for the scale bar
image_folder = 'results/initial_cleanup/'
mask_folder = 'results/napari_masking/'
output_folder = 'results/summary_calculations/'
proofs_folder = 'results/proofs/'

# Current method
TEST_SIGMAS = [1, 1.5, 2]
TEST_STD_THRESHOLDS = [2, 2.5, 3]

# Robust MAD method
MAD_SIGMAS = [1, 1.5, 2]
MAD_MULTIPLIERS = [3, 4, 5]

# Percentile method
PERCENTILE_SIGMAS = [1, 1.5, 2]
PERCENTILES = [97, 98, 99]

# Local adaptive threshold
LOCAL_BLOCK_SIZES = [31, 51, 101]

# Difference-of-Gaussians
DOG_PARAMS = [
    (1, 3),
    (1, 5),
    (1.5, 5),
    (2, 5),
    (2, 8),
    (2, 10)
]

# White top-hat
TOPHAT_RADII = [3, 5, 8, 12]

for folder in [output_folder, proofs_folder]:
    if not os.path.exists(folder):
        os.mkdir(folder)


def filter_saturated_images(images, masks):
    # logger.info('filtering saturated cells...')
    filtered = {}
    for name, img in images.items():
        # wrangle masks to match the expected input for the saturation check function
        if len(masks[name].shape) == 2:  # if only one mask layer, add a dummy one
            masks_arr = masks[name]
            masks_arr = np.expand_dims(masks_arr, axis=0) 
        else:
            masks_arr = masks[name]             

        # apply imported saturation check function
        cells = remove_saturated_cells(
            image_stack=img,
            mask_stack=masks_arr,
            COI=COI_1
        )

        # build stack for downstream processing as [coi2, coi1, cell_masks]
        filtered[name] = np.stack([img[COI_2], img[COI_1], cells])
    # logger.info('saturated cells filtered')
    return filtered


def generate_parameter_test_proofs(image_dict):

    for name, img in image_dict.items():

        coi2, coi1, mask = img
        cell_mask = mask > 0

        # ----------------------------------------------------------
        # FIXED DISPLAY RANGE
        # Same raw image display for every panel
        # ----------------------------------------------------------

        vmin, vmax = np.percentile(
            coi1[cell_mask],
            [1, 99]
        )

        # ----------------------------------------------------------
        # PRE-CALCULATE FILTERED IMAGES
        # ----------------------------------------------------------

        gaussian_images = {
            sigma: gaussian_filter(coi1, sigma=sigma)
            for sigma in set(
                TEST_SIGMAS +
                MAD_SIGMAS +
                PERCENTILE_SIGMAS
            )
        }

        # Local adaptive thresholds
        local_thresholds = {
            block_size: threshold_local(
                coi1,
                block_size=block_size,
                method='gaussian'
            )
            for block_size in LOCAL_BLOCK_SIZES
        }

        # Difference of Gaussians
        dog_images = {
            (sigma_small, sigma_large):
                gaussian_images.get(
                    sigma_small,
                    gaussian_filter(coi1, sigma=sigma_small)
                )
                -
                gaussian_filter(coi1, sigma=sigma_large)
            for sigma_small, sigma_large in DOG_PARAMS
        }

        # White top-hat
        tophat_images = {
            radius: white_tophat(
                coi1,
                footprint=disk(radius)
            )
            for radius in TOPHAT_RADII
        }

        # ----------------------------------------------------------
        # BUILD LIST OF METHODS / PARAMETER COMBINATIONS
        # ----------------------------------------------------------

        methods = []

        # ==========================================================
        # 1. GAUSSIAN + SD
        # ==========================================================

        for sigma in TEST_SIGMAS:
            for std_threshold in TEST_STD_THRESHOLDS:

                methods.append({
                    'method': 'Gaussian + SD',
                    'label': (
                        f'σ={sigma}, SD={std_threshold}'
                    ),
                    'type': 'gaussian_sd',
                    'sigma': sigma,
                    'threshold': std_threshold
                })

        # ==========================================================
        # 2. GAUSSIAN + MAD
        # ==========================================================

        for sigma in MAD_SIGMAS:
            for mad_multiplier in MAD_MULTIPLIERS:

                methods.append({
                    'method': 'Robust MAD',
                    'label': (
                        f'σ={sigma}, MAD={mad_multiplier}'
                    ),
                    'type': 'mad',
                    'sigma': sigma,
                    'threshold': mad_multiplier
                })

        # ==========================================================
        # 3. PERCENTILE
        # ==========================================================

        for sigma in PERCENTILE_SIGMAS:
            for percentile in PERCENTILES:

                methods.append({
                    'method': 'Percentile',
                    'label': (
                        f'σ={sigma}, P={percentile}'
                    ),
                    'type': 'percentile',
                    'sigma': sigma,
                    'threshold': percentile
                })

        # ==========================================================
        # 4. LOCAL ADAPTIVE THRESHOLD
        # ==========================================================

        for block_size in LOCAL_BLOCK_SIZES:

            methods.append({
                'method': 'Local',
                'label': f'block={block_size}',
                'type': 'local',
                'block_size': block_size
            })

        # ==========================================================
        # 5. DIFFERENCE OF GAUSSIANS
        # ==========================================================

        for sigma_small, sigma_large in DOG_PARAMS:

            methods.append({
                'method': 'DoG',
                'label': (
                    f'σ={sigma_small}/{sigma_large}'
                ),
                'type': 'dog',
                'sigma_small': sigma_small,
                'sigma_large': sigma_large
            })

        # ==========================================================
        # 6. WHITE TOP-HAT
        # ==========================================================

        for radius in TOPHAT_RADII:

            methods.append({
                'method': 'Top-hat',
                'label': f'radius={radius}',
                'type': 'tophat',
                'radius': radius
            })

        # ----------------------------------------------------------
        # GRID SIZE
        # ----------------------------------------------------------

        ncols = 9
        nrows = int(np.ceil(len(methods) / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(27, nrows * 4),
            squeeze=False
        )

        # ----------------------------------------------------------
        # PROCESS EACH PARAMETER SET
        # ----------------------------------------------------------

        for panel_number, params in enumerate(methods):

            row = panel_number // ncols
            col = panel_number % ncols

            ax = axes[row, col]

            puncta_mask = np.zeros_like(
                cell_mask,
                dtype=bool
            )

            puncta_areas = []
            puncta_intensities = []

            # ------------------------------------------------------
            # DETECT PUNCTA CELL-BY-CELL
            # ------------------------------------------------------

            for lbl in np.unique(mask):

                if lbl == 0:
                    continue

                single_cell = mask == lbl

                coi1_vals = coi1[single_cell]

                mean_coi1 = coi1_vals.mean()
                std_coi1 = coi1_vals.std()

                # ==================================================
                # GAUSSIAN + SD
                # ==================================================

                if params['type'] == 'gaussian_sd':

                    blurred = gaussian_images[
                        params['sigma']
                    ]

                    threshold = (
                        mean_coi1
                        + params['threshold'] * std_coi1
                    )

                    binary = (
                        blurred > threshold
                    )

                # ==================================================
                # ROBUST MAD
                # ==================================================

                elif params['type'] == 'mad':

                    blurred = gaussian_images[
                        params['sigma']
                    ]

                    median = np.median(coi1_vals)

                    mad = np.median(
                        np.abs(coi1_vals - median)
                    )

                    robust_std = 1.4826 * mad

                    threshold = (
                        median
                        + params['threshold'] * robust_std
                    )

                    binary = (
                        blurred > threshold
                    )

                # ==================================================
                # PERCENTILE
                # ==================================================

                elif params['type'] == 'percentile':

                    blurred = gaussian_images[
                        params['sigma']
                    ]

                    threshold = np.percentile(
                        blurred[single_cell],
                        params['threshold']
                    )

                    binary = (
                        blurred > threshold
                    )

                # ==================================================
                # LOCAL ADAPTIVE
                # ==================================================

                elif params['type'] == 'local':

                    threshold_image = local_thresholds[
                        params['block_size']
                    ]

                    binary = (
                        coi1 > threshold_image
                    )

                # ==================================================
                # DIFFERENCE OF GAUSSIANS
                # ==================================================

                elif params['type'] == 'dog':

                    dog = dog_images[
                        (
                            params['sigma_small'],
                            params['sigma_large']
                        )
                    ]

                    # Robust threshold on DoG response
                    dog_values = dog[single_cell]

                    median_dog = np.median(dog_values)

                    mad_dog = np.median(
                        np.abs(
                            dog_values - median_dog
                        )
                    )

                    robust_std_dog = 1.4826 * mad_dog

                    threshold = (
                        median_dog
                        + 4 * robust_std_dog
                    )

                    binary = dog > threshold

                # ==================================================
                # WHITE TOP-HAT
                # ==================================================

                elif params['type'] == 'tophat':

                    tophat = tophat_images[
                        params['radius']
                    ]

                    threshold = np.percentile(
                        tophat[single_cell],
                        99
                    )

                    binary = (
                        tophat > threshold
                    )

                # --------------------------------------------------
                # Restrict to current cell
                # --------------------------------------------------

                binary = binary & single_cell

                # --------------------------------------------------
                # Remove small objects
                # --------------------------------------------------

                puncta_labels = morphology.label(
                    binary
                )

                puncta_labels = remove_small_objects(
                    puncta_labels,
                    min_size=MIN_PUNCTA_SIZE
                )

                puncta_mask |= (
                    puncta_labels > 0
                )

                # --------------------------------------------------
                # Quantitative measurements
                # --------------------------------------------------

                props = measure.regionprops(
                    puncta_labels,
                    intensity_image=coi1
                )

                for prop in props:

                    puncta_areas.append(
                        prop.area
                    )

                    puncta_intensities.append(
                        prop.mean_intensity
                    )

            # ------------------------------------------------------
            # QUANTITATIVE METRICS
            # ------------------------------------------------------

            n_puncta = len(puncta_areas)

            puncta_area_fraction = (
                puncta_mask.sum()
                / cell_mask.sum()
            )

            if n_puncta > 0:

                median_puncta_area = np.median(
                    puncta_areas
                )

                median_puncta_intensity = np.median(
                    puncta_intensities
                )

                puncta_intensity_enrichment = (
                    median_puncta_intensity
                    / np.mean(coi1[cell_mask])
                )

            else:

                median_puncta_area = np.nan
                puncta_intensity_enrichment = np.nan

            # ------------------------------------------------------
            # RAW IMAGE
            # ------------------------------------------------------

            ax.imshow(
                coi1,
                cmap='gray_r',
                vmin=vmin,
                vmax=vmax
            )

            # ------------------------------------------------------
            # PUNCTA OVERLAY
            # ------------------------------------------------------

            overlay = np.zeros(
                (*puncta_mask.shape, 4)
            )

            overlay[puncta_mask] = [
                1, 0, 0, 0.65
            ]

            ax.imshow(overlay)

            # ------------------------------------------------------
            # CELL OUTLINES
            # ------------------------------------------------------

            contours = measure.find_contours(
                cell_mask.astype(int),
                0.8
            )

            for contour in contours:

                if len(contour) >= 100:

                    ax.plot(
                        contour[:, 1],
                        contour[:, 0],
                        c='cyan',
                        lw=0.5
                    )

            # ------------------------------------------------------
            # PANEL LABEL
            # ------------------------------------------------------

            ax.set_title(
                f"{params['method']}\n"
                f"{params['label']}\n"
                f"n={n_puncta}   "
                f"area={puncta_area_fraction:.2%}\n"
                f"median={median_puncta_area:.1f} px²   "
                f"enrich={puncta_intensity_enrichment:.2f}×",
                fontsize=9
            )

            ax.set_axis_off()

        # ----------------------------------------------------------
        # TURN OFF UNUSED PANELS
        # ----------------------------------------------------------

        for panel_number in range(
            len(methods),
            nrows * ncols
        ):

            row = panel_number // ncols
            col = panel_number % ncols

            axes[row, col].set_axis_off()

        # ----------------------------------------------------------
        # FIGURE TITLE
        # ----------------------------------------------------------

        fig.suptitle(
            f'{name} — puncta segmentation optimization',
            fontsize=18,
            y=0.995
        )

        plt.tight_layout()

        # ----------------------------------------------------------
        # SAVE
        # ----------------------------------------------------------

        output_path = (
            f'{proofs_folder}'
            f'{name}_proof_optimization.png'
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches='tight'
        )

        plt.show()
        plt.close(fig)


if __name__ == '__main__':

    image_files = [f for f in os.listdir(image_folder) if f.endswith('.npy')]

    test_images = {}
    all_features = []

    rng = np.random.default_rng(42)

    image_files_short = rng.choice(
        image_files,
        size=min(10, len(image_files)),
        replace=False
    )

    for fname in image_files_short:

        name = fname.removesuffix('.npy')

        mask_path = os.path.join(mask_folder, f"{name}_mask.npy")
        if not os.path.exists(mask_path):
            continue

        logger.info(f'processing {name}')

        image = np.load(os.path.join(image_folder, fname))
        mask = np.load(mask_path, allow_pickle=True)

        cell_mask = mask[0]

        filtered_img = filter_saturated_images(
            {name: image}, 
            {name: cell_mask}
        )[name]

        # Keep for parameter-sweep visualization
        test_images[name] = filtered_img

        del image, mask, filtered_img


    # Parameter sweep after processing
    generate_parameter_test_proofs(
        test_images
    )

    logger.info('pipeline complete')