"""Detect puncta, measure features, visualize data
"""

import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
import skimage.io
import functools
import cv2
from skimage import measure, segmentation, morphology
from skimage.filters import threshold_otsu
from scipy import ndimage, stats
from scipy.stats import mannwhitneyu, kruskal, shapiro
from statannotations.Annotator import Annotator
from loguru import logger
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import datasets
plt.rcParams.update({'font.size': 14})

logger.info('Import ok')

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
        properties = ['area', 'eccentricity', 'label', 'major_axis_length', 'minor_axis_length', 'perimeter']

    return pd.DataFrame(skimage.measure.regionprops_table(mask, properties=properties))

# ----------------Initialise file list----------------
file_list = [filename for filename in os.listdir(
    input_folder) if 'npy' in filename]

images = {filename.replace('.npy', ''): np.load(
    f'{input_folder}{filename}') for filename in file_list}

masks = {masks.replace('_mask.npy', ''): np.load(
    f'{mask_folder}{masks}', allow_pickle=True) for masks in os.listdir(f'{mask_folder}') if '_mask.npy' in masks}

# make dictionary from images and masks array
image_mask_dict = {
    key: np.stack([images[key][0, :, :], images[key][1, :, :], masks[key][0, :, :]])
    for key in masks
}

# ----------------collect feature information----------------
# remove saturated cells in case some were added during manual validation
logger.info('removing saturated cells')
not_saturated = {}
for name, image in image_mask_dict.items():
    labels_filtered = []
    # image order: [0] before, after, h342, pol
    # image order: [1] cell_mask, nuc_mask
    # find cells with few pixels and remove
    unique_val, counts = np.unique(image[2, :, :], return_counts=True)

    # loop to remove saturated cells (>1% px values > 60000)
    for label in unique_val[1:]:
        label
        pixel_count = np.count_nonzero(image[2, :, :] == label)
        cell = np.where(image[2, :, :] == label, image[0, :, :], 0)
        saturated_count = np.count_nonzero(cell == 65535)

        if (saturated_count/pixel_count) < 0.01:
            labels_filtered.append(label)

    cells_filtered = np.where(
        np.isin(image[2, :, :], labels_filtered), image[2, :, :], 0)

    # stack the filtered masks
    cells_filtered_stack = np.stack(
        (image[0, :, :], image[1, :, :], cells_filtered))
    not_saturated[name] = cells_filtered_stack

# now collect puncta and cell features info
logger.info('collecting feature info')
feature_information = []
for name, image in not_saturated.items():
    # logger.info(f'Processing {name}')
    labels_filtered = []
    unique_val, counts = np.unique(image[2, :, :], return_counts=True)
    # loop to extract params from cells
    for num in unique_val[1:]:
        num
        #this is pulling single cell/ single masks 
        #below is indicating which channel so its saying the 3rd channel is where the masks are and then get all data from 2nd channel 
        cell = np.where(image[2, :, :] == num, image[0, :, :], 0)
        thresh = threshold_otsu(cell)
        cell_std = np.std(cell[cell != 0])
        cell_mean = np.mean(cell[cell != 0])
        binary = (cell > (cell_std*4)).astype(int)
        puncta_masks = measure.label(binary)
        cell_properties = feature_extractor(puncta_masks)
        properties = pd.concat([cell_properties])
        properties['image_name'] = name
        properties['cell_number'] = num
        properties['cell_size'] = np.size(cell[cell!=0])
        properties['cell_intensity_mean'] = cell_mean
        properties = properties[properties['area'] > 9]
        properties = properties[properties['cell_intensity_mean'] < 50000]
        feature_information.append(properties)
feature_information = pd.concat(feature_information)
logger.info('Completed feature collection')

# --------------Grab major and minor_axis_length for punctas--------------
minor_axis = feature_information.groupby(
    ['image_name', 'cell_number']).mean()['minor_axis_length']
major_axis = feature_information.groupby(
    ['image_name', 'cell_number']).mean()['major_axis_length']

# --------------Calculate average size of punctas per cell--------------
avg_size = feature_information.groupby(
    ['image_name', 'cell_number']).mean()['area'].reset_index()

# --------------Calculate average size of punctas per cell--------------
avg_eccentricity = feature_information.groupby(
    ['image_name', 'cell_number']).mean()['eccentricity'].reset_index()

# --------------Calculate proportion of area in punctas--------------
cell_size = feature_information.groupby(
    ['image_name', 'cell_number']).mean()['cell_size']
puncta_area = feature_information.groupby(
    ['image_name', 'cell_number']).sum()['area']
puncta_proportion = ((puncta_area / cell_size) *
                   100).reset_index().rename(columns={'0': 'proportion_puncta_area'})

# --------------Calculate number of 'punctas' per cell--------------
puncta_count = feature_information.groupby(
    ['image_name', 'cell_number']).count()['area']

# --------------Calculate puncta density (count/area in µm) --------------
puncta_density = (puncta_count / (cell_size/1000000)
                ).reset_index().rename(columns={'area': 'proportion_puncta_area'})

# --------------Grab cell intensity mean --------------
cell_intensity_mean = feature_information.groupby(
    ['image_name', 'cell_number']).mean()['cell_intensity_mean']


# --------------Summarize, save to csv --------------
summary = functools.reduce(lambda left, right: pd.merge(left, right, on=['image_name', 'cell_number'], how='outer'), [avg_size, cell_size.reset_index(
), puncta_area.reset_index(), puncta_proportion, puncta_count.reset_index(), puncta_density, minor_axis, major_axis, cell_intensity_mean, avg_eccentricity])
summary.columns = ['image_name', 'cell_number', 'mean_puncta_area', 'cell_size', 'total_puncta_area',
                   'puncta_area_proportion', 'puncta_count', 'puncta_density', 'puncta_mean_minor_axis', 'puncta_mean_major_axis', 'cell_intensity_mean', 'avg_eccentricity']

# add columns for sorting
summary['peptide'] = summary['image_name'].str.split('_').str[0]
summary['rep'] = summary['image_name'].str.split('_').str[-1].str.split('-').str[0]
summary['condition'] = summary['image_name'].str.extract(r'(?:CROTCY3|LDLDIL)_(.*?)_rep')
        
summary.to_csv(f'{output_folder}puncta_detection_summary.csv')

### Use this when you actually have replicates to grab averages for stats
# summary_condition = []
# for col in summary.columns[2:-3]:
#     reps_table = summary.groupby(['peptide','condition']).mean(numeric_only=True)[f'{col}']
#     summary_condition.append(reps_table)
# summary_condition_df = functools.reduce(lambda left, right: pd.merge(left, right, on=['peptide','condition'], how='outer'), summary_condition).reset_index()

# --------------visualize calculated parameters--------------
features_of_interest = ['mean_puncta_area', 'puncta_area_proportion', 'puncta_count',  'puncta_mean_minor_axis', 'puncta_mean_major_axis', 'avg_eccentricity', 'cell_intensity_mean']
### I want to look at conditions for each peptide, comparing each condition against other conditions for that single peptide 

#--------------------Plotting individual plots --------------------
# Loop through each unique peptide
for peptide in summary['peptide'].unique():
    
    # Filter data for the current peptide
    summary_peptide = summary[summary['peptide'] == peptide]
    # summary_reps_peptide = summary_condition_df[summary_condition_df['peptide'] == peptide] #would do this for replicates 
    order = ['norm','norm_spike','LPD','LPD_spike','starv','starv_spike']
    x = 'condition'
   

    for parameter in ['mean_puncta_area', 'puncta_area_proportion', 'puncta_count',  'puncta_mean_minor_axis', 'puncta_mean_major_axis', 'avg_eccentricity', 'cell_intensity_mean']:
        # plot data
        fig, ax = plt.subplots()
        ax = sns.stripplot(data=summary_peptide, x=x, y=parameter, order=order, dodge='True',
                        edgecolor='white', linewidth=1, size=8, alpha=0.4)
        # sns.stripplot(data=summary_peptide, x=x, y=parameter, order=order, dodge='True', edgecolor='k', linewidth=1, size=8, ax=ax)
        sns.boxplot(data=summary_peptide, x=x, y=parameter,
                    order=order, palette=['.7', '.8'], ax=ax)
        # # statannot stats
        # pairs = [(condition, 'norm') for condition in order if condition != 'norm']
        # annotator = Annotator(ax, pairs, data=summary__reps_peptide, x=x, y=parameter, order=order)
        # annotator.configure(test='Mann-Whitney', verbose=2)
        # annotator.apply_test()
        # annotator.annotate()

        #add a title that also includes peptide iterated through 
        ax.set_title(f'{parameter} for {peptide}')
        # formatting
        sns.despine()
        plt.xticks(rotation=45)
        plt.xlabel('condition')
        plt.ylabel(parameter)
        plt.tight_layout()

        plt.savefig(f'{plotting_folder}2503B_{peptide}_{parameter}.png', format='png', dpi=300)
        plt.show()

#--------------------Plotting a summary plot  --------------------
pairs = [(condition, 'norm') for condition in order if condition != 'norm']
order = ['norm','norm_spike','LPD','LPD_spike','starv','starv_spike']
x = 'condition'

for peptide in summary['peptide'].unique():
    # Filter data for the current peptide
    summary_peptide = summary[summary['peptide'] == peptide]
    summary_reps_peptide = summary[summary['peptide'] == peptide]

    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(hspace=0.4)
    plt.suptitle(f'calculated parameters - {peptide}', fontsize=18, y=0.99)
    for n, parameter in enumerate(features_of_interest):
        # add a new subplot iteratively
        ax = plt.subplot(4, 3, n + 1)
        # filter df and plot ticker on the new subplot axis
        sns.stripplot(data=summary_peptide, x=x, y=parameter, dodge='True',
                        edgecolor='white', linewidth=1, size=8, alpha=0.4, order=order, ax=ax)
        # # store legends info
        # handles, labels = ax.get_legend_handles_labels()
        # # continue plotting - below is for replicates 
        # sns.stripplot(data=summary_peptide, x=x, y=parameter, dodge='True', edgecolor='k', linewidth=1, size=8, order=order, ax=ax)
        sns.boxplot(data=summary_peptide, x=x, y=parameter,
                    palette=['.9'], order=order, ax=ax)
        # # remove all legends
        # ax.legend().remove()
        # # statannot stats
        # annotator = Annotator(ax, pairs, data=summary_peptide, x=x, y=parameter, order=order)
        # annotator.configure(test='Mann-Whitney', verbose=2)
        # annotator.apply_test()
        # annotator.annotate()
        # formatting
        sns.despine()
        plt.xlabel('')
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # plt.legend(handles, labels, bbox_to_anchor=(1.1, 1), title='RHM1 tag')
    plt.savefig(f'{output_folder}2503B_summary-{peptide}.png', bbox_inches='tight',pad_inches = 0.1, dpi = 300)


# ------------------------------Checking normality of the data ---------------------------------
#### ----------------------------IMPORTANT TO FIGURE OUT STAT: ------------------------------
#-----------if p value < 0.05 data is not normal and you should use Mann whitney or kruskal -----
# ----------------mann whitney = 2 conditions, kruskal = more than 2 conditions 
#--------If p >= 0.05, your data may be normally distributed, so an ANOVA might be appropriate---

# List of unique peptides
stats_peptide = summary['peptide'].unique()

# List of unique conditions
stat_conditions = summary['condition'].unique()

# Apply Shapiro-Wilk test for each peptide and condition
for peptide in stats_peptide:
    for condition in stat_conditions:
        # Filter the summary for the current peptide and condition
        condition_summary = summary[(summary['peptide'] == peptide) & (summary['condition'] == condition)]['cell_intensity_mean']  # Replace 'values' with your actual column name
        
        # Apply the Shapiro-Wilk test
        stat, p_value = stats.shapiro(condition_summary)
        
        # Output the results
        print(f"Shapiro-Wilk test for peptide: {peptide}, condition: {condition}:")
        print(f"Test statistic: {stat}")
        print(f"P-value: {p_value}")
        
        # Interpret the result
        if p_value < 0.05:
            print(f"  {peptide} - {condition}: The data is likely NOT normally distributed.\n")
        else:
            print(f"  {peptide} - {condition}: The data appears to be normally distributed.\n")

#----------- Collecting the stat information into a dataframe -----------------------------
# List of unique peptides

stat_unique_peptides = summary['peptide'].unique()

# List of conditions you want to compare (excluding 'norm' for Mann-Whitney)
conditions = ['norm', 'norm_spike', 'LPD', 'LPD_spike', 'starv', 'starv_spike']

# Prepare an empty dictionary to store results
results = {}

# Loop through each peptide
for peptide in stat_unique_peptides:
    # Filter data for the current peptide
    summary_peptide = summary[summary['peptide'] == peptide]
    
    # Loop through all conditions, comparing each to 'norm'
    for condition in conditions:
        # Extract the data for the two conditions (e.g., 'norm' vs 'LPD')
        condition_data = summary_peptide[summary_peptide['condition'] == condition]
        norm_data = summary_peptide[summary_peptide['condition'] == 'norm']
        
        # Perform a Mann-Whitney U test (if comparing two conditions)
        stat, p_value = mannwhitneyu(condition_data['cell_intensity_mean'], norm_data['cell_intensity_mean'])
        
        # Save the results
        results[f'{peptide} {condition} vs norm'] = {'stat': stat, 'p_value': p_value}

# Convert results to DataFrame for easy viewing
results_df = pd.DataFrame(results).T
print(results_df)

results_df.to_csv(f'{output_folder}statistics_summary.csv')



# # -------------- plotting proofs --------------
# # plot proofs
# for name, image in image_mask_dict.items():
#     name
#     unique_val, counts = np.unique(image[2, :, :], return_counts=True)

#     # extract coords
#     cell = np.where(image[2, :, :] != 0, image[1, :, :], 0)
#     image_df = feature_information[(feature_information['image_name'] == name)]
#     if len(image_df) > 0:
#         cell_contour = image_df['cell_coords'].iloc[0]
#         coord_list = np.array(image_df.granule_coords)

#         # plot
#         fig, (ax1, ax2) = plt.subplots(1, 2)
#         ax1.imshow(image[1,:,:], cmap=plt.cm.gray_r)
#         ax1.imshow(image[0,:,:], cmap=plt.cm.Blues, alpha=0.60)
#         ax2.imshow(cell, cmap=plt.cm.gray_r)
#         for cell_line in cell_contour:
#             ax2.plot(cell_line[:, 1], cell_line[:, 0], linewidth=0.5, c='k')
#         if len(coord_list) > 1:
#             for puncta in coord_list:
#                 if isinstance(puncta, np.ndarray):
#                     ax2.plot(puncta[:, 1], puncta[:, 0], linewidth=0.5)
#         for ax in fig.get_axes():
#             ax.label_outer()

#         # Create scale bar
#         scalebar = ScaleBar(0.0779907, 'um', location = 'lower right', pad = 0.3, sep = 2, box_alpha = 0, color='w', length_fraction=0.3)
#         ax1.add_artist(scalebar)

#         # title and save
#         fig.suptitle(name, y=0.78)
#         fig.tight_layout()
#         fig.savefig(f'{plotting_folder}{name}_proof.png', bbox_inches='tight',pad_inches = 0.1, dpi = 300)
#         plt.close()