"""Applies cellpose algorithms to determine cellular and nuclear masks
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from cellpose import models
from cellpose import plot
from loguru import logger
from cellpose.io import logger_setup
logger_setup();

logger.info('Import ok')

input_folder = 'results/initial_cleanup/'
output_folder = 'results/cellpose_masking/'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)


def apply_cellpose(images, image_type='sam', channels = None, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0, use_gpu=True):
    """Apply standard cellpose model to list of images.

    Args:
        images (ndarray): numpy array of 16 bit images
        image_type (str, optional): Cellpose SAM model. Defaults to 'sam' as indicated by the May 2025 update.
        channels (int, optional): define CHANNELS to run segementation on (grayscale=0, R=1, G=2, B=3) where channels = [cytoplasm, nucleus]. Defaults to None with the new update.
        diameter (int, optional): Expected diameter of cell or nucleus. Defaults to None with the new update, but you can edit this.
        flow_threshold (float, optional): maximum allowed error of the flows for each mask. Defaults to 0.4.
        cellprob_threshold (float, optional): The network predicts 3 outputs: flows in X, flows in Y, and cell “probability”. The predictions the network makes of the probability are the inputs to a sigmoid centered at zero (1 / (1 + e^-x)), so they vary from around -6 to +6. Decrease this threshold if cellpose is not returning as many ROIs as you expect. Defaults to 0.0.
        resample (bool, optional): Resampling can create smoother ROIs but take more time. Defaults to False.

    Returns:
        ndarray: array of masks, flows, styles, and diameters
    """
    if channels is None:
        channels = [0, 0]
    model = models.CellposeModel(model_type=image_type, gpu=use_gpu)
    masks, flows, styles = model.eval(
        images, diameter=diameter, channels=channels, flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)
    return masks, flows, styles


def visualise_cell_pose(images, masks, flows, channels = None):
    """Display cellpose results for each image

    Args:
        images (ndarray): single channel (one array)
        masks (ndarray): one array
        flows (_type_): _description_
        channels (_type_, optional): _description_. Defaults to None.
    """
    if channels is None:
        channels = [0, 0]
    for image_number, image in enumerate(images):
        maski = masks[image_number]
        flowi = flows[image_number][0]
        
        fig = plt.figure(figsize=(12, 5))
        plot.show_segmentation(fig, image, maski, flowi, channels=channels)
        plt.tight_layout()
        plt.show()


# ----------------Initialise file list----------------
file_list = [filename for filename in os.listdir(
    input_folder) if 'npy' in filename]

# reading in both channels for each image
imgs = [np.load(f'{input_folder}{filename}')
        for filename in file_list]

# need a library?
images = {filename.replace('.npy', ''): np.load(
    f'{input_folder}{filename}') for filename in file_list}

# ----------------Prepare images for cell segmentation----------------
# Set the channel (ch) that you want to use for cell segmentation as image[ch,:,:]
# Remember that if its channel 1 on microscope software then its channel 0 here (because python is base 0)
mask_channel = [image[1, :, :] for name, image in images.items()]

# set the nuclear channel, usually this is the last channel - change the image[ch,:,:]
nuclear_channel = [image[2, :, :] for name, image in images.items()]

# # Adapt this if you need to brighten the image for segmentation purposes (will not manipulate data)
# # Assume 16-bit images; scale brightness for segmentation only
# # You can comment this out if your images dont need this. but I find that images generally do better with enhancing brightness 
# mask_channel = [
#     np.clip(channel * 5, 0, 65535).astype(np.uint16)  # Keep data in 16-bit
#     for channel in mask_channel
# ]

# ----------------Optimize cell segmentation settings----------------
# this will only run the first four images so you can make sure the masking is sound 
# optimize segmentation of cells
masks, flows, styles = apply_cellpose(
        mask_channel[:4], image_type='sam', diameter=None, flow_threshold=0.0, cellprob_threshold=0)
visualise_cell_pose(mask_channel[:4], masks, flows)

# to check the cell diameter, run ```plt.imshow(mask_channel[0])``` in the interactive window 

# ----------------Segment and save masks----------------
# copy optimized settings from previous 'apply_cellpose' to segment and save cell masks
masks, flows, styles = apply_cellpose(
        mask_channel, image_type='sam', diameter=None, flow_threshold=0.0, cellprob_threshold=0)

# save cell masks before moving on to nuclei (for memory)
np.save(f'{output_folder}cellpose_cellmasks.npy', masks)
logger.info('cell masks saved')

# ----------------Optimize nuclear segmentation settings----------------
# optimize segmentation of nuclei
nuc_masks, flows, styles = apply_cellpose(
        nuclear_channel[:4], image_type='sam', diameter=None, flow_threshold=0.4, cellprob_threshold=0)
visualise_cell_pose(nuclear_channel[:4], nuc_masks, flows)

# ----------------Segment and save masks----------------
# segment nuclei
nuc_masks, flows, styles = apply_cellpose(
        nuclear_channel, image_type='sam', diameter=None, flow_threshold=0.4, cellprob_threshold=0)

# save nuclei masks
np.save(f'{output_folder}cellpose_nucmasks.npy', nuc_masks)
logger.info('nuclei masks saved')