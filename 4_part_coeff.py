"""Detect puncta, measure features, visualize data
"""

import os
import numpy as np
import seaborn as sns
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
import skimage.io
import functools
import cv2
from skimage import measure, segmentation, morphology
from scipy.stats import skewtest, skew
from skimage import morphology
from skimage.morphology import remove_small_objects
from statannotations.Annotator import Annotator
from loguru import logger
from matplotlib_scalebar.scalebar import ScaleBar
plt.rcParams.update({'font.size': 14})

input_folder = 'python_results/initial_cleanup/'
mask_folder = 'python_results/napari_masking/'
output_folder = 'python_results/summary_calculations/'
plotting_folder = 'python_results/plotting/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

if not os.path.exists(plotting_folder):
    os.mkdir(plotting_folder)

def feature_extractor(mask, properties=False):

    if not properties:
        properties = ['area', 'eccentricity', 'label', 'major_axis_length', 'minor_axis_length', 'perimeter', 'coords']

    return pd.DataFrame(skimage.measure.regionprops_table(mask, properties=properties))

# ----------------Initialise file list----------------
file_list = [filename for filename in os.listdir(
    input_folder) if 'npy' in filename]

images = {filename.replace('.npy', ''): np.load(
    f'{input_folder}{filename}') for filename in file_list}

masks = {masks.replace('_mask.npy', ''): np.load(
    f'{mask_folder}{masks}', allow_pickle=True) for masks in os.listdir(f'{mask_folder}') if '_mask.npy' in masks}

# Assumes images[key] shape is (C, H, W) and masks[key] shape is (1, H, W) or (H, W)
image_mask_dict = {
    key: np.stack([
        images[key][0],  # Channel 0
        images[key][1],  # Channel 1
        masks[key][0] if masks[key].ndim == 3 else masks[key]  # Handle 2D or 3D mask
    ])
    for key in masks
}
# ----------------collect feature information----------------
# remove saturated nucs in case some were added during manual validation
not_saturated = {}
# structure element for eroding nuclear mask
structure_element = np.ones((16, 16)).astype(int)
for name, image in image_mask_dict.items():
    labels_filtered = []
    unique_val, counts = np.unique(image[-1, :, :], return_counts=True)

    # loop to remove saturated nucs (>1% px values > 60000)
    for label in unique_val[1:]:
        pixel_count = np.count_nonzero(image[-1, :, :] == label)
        nuc_mask = np.where(image[-1, :, :] == label, label, 0)
        # erode nuclear masks to avoid bright puncta at nuclear periphery
        nuc_eroded = morphology.erosion(nuc_mask, structure_element)
        ch0_nuc = np.where(nuc_eroded == label, image[1, :, :], 0)
        ch0_nuc_saturated_count = np.count_nonzero(nuc_eroded == 65535)
        if ((ch0_nuc_saturated_count/pixel_count) < 0.05):
            labels_filtered.append(nuc_eroded)

    # add all eroded, non-saturated, masks together
    nucs_filtered = np.sum(labels_filtered, axis=0)

    # stack the filtered masks
    nucs_filtered_stack = np.stack(
        (image[0, :, :], image[1, :, :],image[2, :, :], nucs_filtered))
    not_saturated[name] = nucs_filtered_stack


# now collect nucleolus masks and nuc features info
logger.info('collecting feature info')
feature_information_list = []
for name, image in not_saturated.items():
    # logger.info(f'Processing {name}')
    labels_filtered = []
    unique_val, counts = np.unique(image[-1, :, :], return_counts=True)
    # find nuc outlines for later plotting
    nuc_binary_mask = np.where(image[-1, :, :] !=0, 1, 0)
    contours = measure.find_contours(nuc_binary_mask, 0.8)
    contour = [x for x in contours if len(x) >= 100]
    # loop to extract params from cells
    for num in unique_val[1:]:
        # last channel (-1) is always the mask - in this case, the nuclear mask
        nuc = np.where(image[-1, :, :] == num, image[-1, :, :], 0)
        # channel 2 (1) = peptide intensity 
        pepchan = np.where(image[-1, :, :] == num, image[0, :, :], 0)
        pepchan_mean = np.mean(pepchan[pepchan != 0])
        # channel 3 (2) = making the mask for nucleolus 
        nucleoluschan = np.where(image[-1, :, :] == num, image[1, :, :], 0)
        nucleoluschan_std = np.std(nucleoluschan[nucleoluschan != 0])
        nucleoluschan_mean = np.mean(nucleoluschan[nucleoluschan != 0])
        binary = (nucleoluschan > ((nucleoluschan_std*2))).astype(int)
        nucleolar_masks = measure.label(binary)
        nucleolar_masks = remove_small_objects(nucleolar_masks, 9)
        # measure properties of nucleolar and fbl masks
        nucleolus_properties = feature_extractor(nucleolar_masks).add_prefix('nucleolar_')
       
        # make list for cov and skew, add as columns to properties for npm1 channel
        peptide_nucleol_cv_list = []
        peptide_nucleol_skew_list = []
        peptide_nucleol_intensity_list = []
        for peptide_nucleol_num in np.unique(nucleolar_masks)[1:]:
            # use nucleolus masks for per nucleol measurements (peptide channel)
            peptide_nucleol = np.where(nucleolar_masks == peptide_nucleol_num, image[1,:,:], 0)
            peptide_nucleol = peptide_nucleol[peptide_nucleol!=0]
            # collect coefficient of variance
            peptide_nucleol_cv = np.std(peptide_nucleol) / np.mean(peptide_nucleol)
            peptide_nucleol_cv_list.append(peptide_nucleol_cv)
            # collect skew of intensity distribution
            peptide_nucleol_skew_list.append(skew(peptide_nucleol))
            # collect mean intensity value
            peptide_nucleol_intensity_list.append(np.mean(peptide_nucleol))
        # store measurements
        nucleolus_properties['peptide_nucleol_cv'] = peptide_nucleol_cv_list
        nucleolus_properties['peptide_nucleol_skew'] = peptide_nucleol_skew_list
        nucleolus_properties['peptide_nucleol_intensity'] = peptide_nucleol_intensity_list
        
        # if no nucleols, fill with 0
        if len(nucleolus_properties) < 1:
            nucleolus_properties.loc[len(nucleolus_properties)] = 0

        # make df and add nuc and image info
        properties = pd.concat([nucleolus_properties])
        properties['image_name'] = name
        properties['nuc_number'] = num
        # properties['compartment'] = roi
        properties['nuc_size'] = np.size(nuc[nuc!=0])
        properties['nuc_intensity_mean'] = pepchan_mean

        # add nuc outlines to coords
        properties['nuc_coords'] = [contour]*len(properties)

        feature_information_list.append(properties)

        
        # make list for cov and skew for fbl, add as columns to properties
        peptide_fbl_cv_list = []
        peptide_fbl_skew_list = []
        peptide_fbl_intensity_list = []
        for peptide_fbl_num in np.unique(fbl_masks)[1:]:
            # use fblus masks for per fbl measurements (peptide channel)
            peptide_fbl = np.where(fbl_masks == peptide_fbl_num, image[1,:,:], 0)
            peptide_fbl = peptide_fbl[peptide_fbl!=0]
            # collect coefficient of variance
            fbl_cv = np.std(peptide_fbl) / np.mean(peptide_fbl)
            peptide_fbl_cv_list.append(fbl_cv)
            # collect skew of intensity distribution
            peptide_fbl_skew_list.append(skew(peptide_fbl))
            # collect mean intensity value
            peptide_fbl_intensity_list.append(np.mean(peptide_fbl))
        # store measurements
        fbl_properties['peptide_fbl_cv'] = peptide_fbl_cv_list
        fbl_properties['peptide_fbl_skew'] = peptide_fbl_skew_list
        fbl_properties['peptide_fbl_intensity'] = peptide_fbl_intensity_list
        
        # if no fbls, fill with 0
        if len(fbl_properties) < 1:
            fbl_properties.loc[len(fbl_properties)] = 0

        # make df and add nuc and image info
        properties = pd.concat([fbl_properties])
        properties['image_name'] = name
        properties['nuc_number'] = num
        properties['nuc_size'] = np.size(nuc[nuc!=0])
        properties['nuc_intensity_mean'] = pepchan_mean

        # add nuc outlines to coords
        properties['nuc_coords'] = [contour]*len(properties)

        feature_information_list.append(properties)
        
feature_information = pd.concat(feature_information_list)
logger.info('completed feature collection')

# adding columns based on image_name
feature_information['peptide'] = feature_information['image_name'].str.split('_').str[1].str.split('-').str[-1]
feature_information['rep'] = feature_information['image_name'].str.split('_').str[2].str.split('-').str[-2]

# combine conditions when I jst named them wrong 
feature_information['peptide'] = ['BMAP27' if row == 'BMAP' else row for row in feature_information['peptide']]
feature_information['peptide'] = ['CROTCY3' if row == 'CrotCY3' else row for row in feature_information['peptide']]

# add aspect ratio (like asking 12x5 or 12/5) and circularity
feature_information['peptide_nucleol_aspect_ratio'] = feature_information['nucleolar_minor_axis_length'] / feature_information['nucleolar_major_axis_length']
feature_information['peptide_nucleol_circularity'] = (12.566*feature_information['nucleolar_area'])/(feature_information['nucleolar_perimeter']**2)

# add partitioning coefficient
feature_information['npm1_partition_coeff'] = feature_information['peptide_nucleol_intensity'] / feature_information['nuc_intensity_mean']

#add part coeff for the fbl channel against nuc intensity
feature_information['fbl_partition_coeff_against_nuc'] = feature_information['peptide_fbl_intensity'] / feature_information['nuc_intensity_mean']

#add part coeff for the fbl channel against npm1
feature_information['fbl_partition_coeff_against_npm1'] = feature_information['peptide_fbl_intensity'] / feature_information['peptide_nucleol_intensity']

# save data for plotting coords
feature_information.to_csv(f'{output_folder}fbl_puncta_detection_feature_info.csv')

# make additional df for avgs per replicate
features_of_interest = ['nucleolar_area', 'nucleolar_eccentricity', 'peptide_nucleol_cv','peptide_nucleol_skew', 'peptide_nucleol_intensity', 'nuc_size', 'fbl_area', 'fbl_eccentricity','peptide_fbl_cv', 'peptide_fbl_skew', 'peptide_fbl_intensity', 'peptide_nucleol_aspect_ratio','peptide_nucleol_circularity', 'fbl_partition_coeff_against_npm1','fbl_partition_coeff_against_nuc','npm1_partition_coeff']

nucleol_summary_reps = []
for col in features_of_interest:
    reps_table = feature_information.groupby(['peptide', 'rep']).mean(numeric_only=True)[f'{col}']
    nucleol_summary_reps.append(reps_table)
nucleol_summary_reps_df = functools.reduce(lambda left, right: pd.merge(left, right, on=['peptide', 'rep'], how='outer'), nucleol_summary_reps).reset_index()

# --------------visualize calculated parameters - raw --------------

x = 'peptide'
order = ['BMAP27', 'LT8A', 'LTC1', 'Mollusc', 'PR30', 'HTN3', 'PR39', 'CECRO','CROTCY3', 'negative', 'GP30']

plots_per_fig = 6
num_features = len(features_of_interest)
num_figures = math.ceil(num_features / plots_per_fig)

for fig_num in range(num_figures):
    # Create a new figure
    plt.figure(figsize=(20, 8))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(f'Calculated Parameters - per Nucleol (Fig {fig_num + 1})', fontsize=18, y=0.99)

    # Get the current slice of features
    start_idx = fig_num * plots_per_fig
    end_idx = min(start_idx + plots_per_fig, num_features)
    current_features = features_of_interest[start_idx:end_idx]

    for i, parameter in enumerate(current_features):
        ax = plt.subplot(2, 3, i + 1)
        sns.stripplot(data=feature_information, x=x, y=parameter, dodge=True, edgecolor='white', linewidth=1, size=8, alpha=0.4, order=order, ax=ax)
        sns.boxplot(data=feature_information, x=x, y=parameter, palette=['.9'], order=order, ax=ax)
        ax.set_title(parameter, fontsize=12)
        ax.set_xlabel('')
        plt.xticks(rotation=45)
        sns.despine()

    plt.tight_layout()

    output_path = f'{output_folder}/puncta-features_pernucleol_raw_fig{fig_num + 1}.png'
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()


#---Plotting specific plots --------------

x = 'peptide'
order = ['BMAP27', 'LT8A', 'LTC1', 'Mollusc', 'PR30', 'PR39', 'HTN3', 'CROTCY3']

# ✅ Only include the features you want to plot
features_of_inteNrest = ['npm1_partition_coeff', 'fbl_partition_coeff_against_nuc']

plots_per_fig = 2
num_features = len(features_of_interest)
num_figures = math.ceil(num_features / plots_per_fig)

for fig_num in range(num_figures):
    plt.figure(figsize=(20, 8))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(f'Partitioning Coeff - per Nucleol', fontsize=18, y=0.99)

    start_idx = fig_num * plots_per_fig
    end_idx = min(start_idx + plots_per_fig, num_features)
    current_features = features_of_interest[start_idx:end_idx]

    for i, parameter in enumerate(current_features):
        ax = plt.subplot(2, 3, i + 1)
        sns.stripplot(data=feature_information, x=x, y=parameter, dodge=True, edgecolor='white', linewidth=1, size=8, alpha=0.4, order=order, ax=ax)
        sns.boxplot(data=feature_information, x=x, y=parameter, palette=['.9'], order=order, ax=ax)
        ax.set_title(parameter, fontsize=12)
        ax.set_xlabel('')
        plt.xticks(rotation=45)
        sns.despine()

    plt.tight_layout()

    output_path = f'{output_folder}/partition_coeffs.png'
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()



# ## Below will measure and plot per cell instead of per nucleolus
# # --------------Grab major and minor_axis_length for punctas--------------
# minor_axis = feature_information.groupby(
#     ['image_name', 'nuc_number'])['nucleolar_minor_axis_length'].mean()
# major_axis = feature_information.groupby(
#     ['image_name', 'nuc_number'])['nucleolar_major_axis_length'].mean()

# # --------------Calculate average size of punctas per nuc--------------
# puncta_avg_area = feature_information.groupby(
#     ['image_name', 'nuc_number'])['nucleolar_area'].mean().reset_index()

# # --------------Calculate proportion of area in punctas--------------
# nuc_size = feature_information.groupby(
#     ['image_name', 'nuc_number'])['nuc_size'].mean()
# puncta_area = feature_information.groupby(
#     ['image_name', 'nuc_number'])['nucleolar_area'].sum()
# puncta_proportion = ((puncta_area / nuc_size) *
#                    100).reset_index().rename(columns={0: 'proportion_puncta_area'})

# # --------------Calculate number of 'punctas' per nuc--------------
# puncta_count = feature_information.groupby(
#     ['image_name', 'nuc_number'])['nucleolar_area'].count()

# # --------------Calculate average size of punctas per nuc--------------
# avg_eccentricity = feature_information.groupby(
#     ['image_name', 'nuc_number'])['nucleol_eccentricity'].mean().reset_index()

# # --------------Grab nuc nucleol cov --------------
# nucleol_cv_mean = feature_information.groupby(
#     ['image_name', 'nuc_number'])['nucleol_cv'].mean()

# # --------------Grab nuc nucleol skew --------------
# nucleol_skew_mean = feature_information.groupby(
#     ['image_name', 'nuc_number'])['nucleol_skew'].mean()

# # --------------Grab nuc nucleol partition coeff --------------
# partition_coeff = feature_information.groupby(
#     ['image_name', 'nuc_number'])['partition_coeff'].mean()

# # --------------Grab nuc intensity mean --------------
# nuc_intensity_mean = feature_information.groupby(
#     ['image_name', 'nuc_number'])['nuc_intensity_mean'].mean()

# # --------------Summarise, save to csv--------------
# summary = functools.reduce(lambda left, right: pd.merge(left, right, on=['image_name', 'nuc_number'], how='outer'), [nuc_size.reset_index(), puncta_avg_area, puncta_proportion, puncta_count.reset_index(), minor_axis, major_axis, avg_eccentricity, nucleol_cv_mean, nucleol_skew_mean, partition_coeff, nuc_intensity_mean])
# summary.columns = ['image_name', 'nuc_number',  'nuc_size', 'mean_puncta_area', 'puncta_area_proportion', 'puncta_count', 'puncta_mean_minor_axis', 'puncta_mean_major_axis', 'avg_eccentricity', 'nucleol_cv_mean', 'nucleol_skew_mean', 'partition_coeff', 'nuc_intensity_mean']

# # --------------tidy up dataframe--------------
# # add columns for sorting
# # add peptide name
# summary['peptide'] = summary['image_name'].str.split('_').str[1].str.split('-').str[-1]

# # save
# summary.to_csv(f'{output_folder}puncta_detection_summary.csv')

# # make df where all puncta features are normalized to mean nuc intensity
# normalized_summary = summary.copy()
# for column in normalized_summary.columns[3:-3]:
#     column
#     normalized_summary[column] = normalized_summary[column] / normalized_summary['nuc_intensity_mean']

# # --------------visualize calculated parameters - raw --------------
# features_of_interest = ['mean_puncta_area',
#        'puncta_area_proportion', 'puncta_count', 'avg_eccentricity', 'nucleol_cv_mean', 'nucleol_skew_mean', 'partition_coeff', 'nuc_intensity_mean']
# plt.figure(figsize=(20, 15))
# plt.subplots_adjust(hspace=0.5)
# plt.suptitle('calculated parameters - per nuc', fontsize=18, y=0.99)
# # loop through the length of tickers and keep track of index
# for n, parameter in enumerate(features_of_interest):
#     # add a new subplot iteratively
#     ax = plt.subplot(3, 4, n + 1)

#     sns.stripplot(data=summary, x=x, y=parameter, dodge='True',
#                     edgecolor='k', linewidth=1, size=8, order=order, ax=ax)
#     sns.boxplot(data=summary, x=x, y=parameter,
#                 palette=['.9'], order=order, ax=ax)
    
#     # # statannot stats
#     # annotator = Annotator(ax, pairs, data=summary, x=x, y=parameter, order=order)
#     # annotator.configure(test='t-test_ind', verbose=2)
#     # annotator.apply_test()
#     # annotator.annotate()

#     # formatting
#     sns.despine()
#     ax.set_xlabel('')

# plt.tight_layout()
# plt.savefig(f'{output_folder}puncta-features_pernuc_raw.png', bbox_inches='tight',pad_inches = 0.1, dpi = 300)

# # --------------visualize calculated parameters - normalized --------------
# plt.figure(figsize=(15, 15))
# plt.subplots_adjust(hspace=0.5)
# plt.suptitle('calculated parameters - per nuc, normalized to cytoplasm intensity', fontsize=18, y=0.99)
# # loop through the length of tickers and keep track of index
# for n, parameter in enumerate(summary.columns.tolist()[3:-3]):
#     # add a new subplot iteratively
#     ax = plt.subplot(3, 4, n + 1)

#     # filter df and plot ticker on the new subplot axis
#     sns.stripplot(data=normalized_summary, x=x, y=parameter, dodge='True',
#                     edgecolor='k', linewidth=1, size=8, order=order, ax=ax)
#     sns.boxplot(data=normalized_summary, x=x, y=parameter,
#                 palette=['.9'], order=order, ax=ax)
    
#     # statannot stats
#     annotator = Annotator(ax, pairs, data=normalized_summary, x=x, y=parameter, order=order)
#     annotator.configure(test='t-test_ind', verbose=2)
#     annotator.apply_test()
#     annotator.annotate()

#     # formatting
#     sns.despine()
#     ax.set_xlabel('')

# plt.tight_layout()
# plt.savefig(f'{output_folder}puncta-features_pernuc_normalized.png', bbox_inches='tight',pad_inches = 0.1, dpi = 300)

# -------------- plotting proofs --------------
# plot proofs
for name, image in image_mask_dict.items():
    name
    unique_val, counts = np.unique(image[2, :, :], return_counts=True)

    # extract coords
    nuc = np.where(image[2, :, :] != 0, image[0, :, :], 0)
    image_df = feature_information[(feature_information['image_name'] == name)]
    if len(image_df) > 0:
        nuc_contour = image_df['nuc_coords'].iloc[0]
        coord_list = np.array(image_df.nucleol_coords)

        # plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(image[1])
        ax2.imshow(image[0])
        ax3.imshow(nuc)
        for nuc_line in nuc_contour:
            ax3.plot(nuc_line[:, 1], nuc_line[:, 0], linewidth=0.5, c='w')
        if len(coord_list) > 1:
            for puncta in coord_list:
                if isinstance(puncta, np.ndarray):
                    ax3.plot(puncta[:, 1], puncta[:, 0], linewidth=0.5)
        for ax in fig.get_axes():
            ax.label_outer()

        # # create scale bar TODO need to update scale value
        # scalebar = ScaleBar(0.0779907, 'um', location = 'lower right', pad = 0.3, sep = 2, box_alpha = 0, color='w', length_fraction=0.3)
        # ax3.add_artist(scalebar)

        # title and save
        fig.suptitle(name, y=0.67, size=14)
        fig.tight_layout()

        fig.savefig(f'{plotting_folder}{name}_proof.png', bbox_inches='tight',pad_inches = 0.1, dpi = 300)
        plt.close()